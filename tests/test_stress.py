"""Stress test of a flash-jev server (GPU server), reproducible end to end: long evidence, the capacity limit, many
clients at once, and the same answers under load.

    python tests/test_stress.py --model-dir ~/jev/models/StartLux-Decision-4B-Q8_0-GGUF \\
        --jevbench ~/jev/src/jevbench-7ce310c7 --start int8 [--quick] [--out results/stress/flash-jev-int8.json]
    python tests/test_stress.py --model-dir ... --jevbench ... --url http://127.0.0.1:8092/v1/systemone   # running

--start int8|bf16|w4a8 starts the server with scripts/serve.sh (MODEL, FLASHJEV_FLAGS as there) and stops it at the end.
Reproducible inputs: the long evidence is a seeded synthetic application log (bench/run_long.py) sized with the model's
tokenizer in StartLux's prompt format; JevBench passes run in fixed shuffled orders (random.Random(pass)); every request
carries its own request number (bench/run_http.uniquify), so none reuses another's cached evidence - except in check 5,
which sends the same requests twice on purpose.

Checks (exit status 1 when one fails; every number goes to --out):
  1. long evidence (--lengths, 8k and 50k tokens): answered, and the server counted every token sent (no truncation);
     latency.  A length beyond the engine's largest token bucket is skipped (check 2 covers the limit)
  2. capacity (the largest token bucket, from /health): a request just inside it is answered, one beyond it is refused
     with HTTP 422, and the server goes on answering
  3. 8k-token requests from 8, 16 and 32 clients at once (--levels): no failed request; requests/s
  4. JevBench's 231 public items one at a time (the score), then from 8, 16 and 32 clients, --passes passes each: no
     failed request, and every pass within --score-tolerance of the sequential score (the request number in front of
     the evidence can move a near-tie); requests/s
  5. answers under load: 64 JevBench requests from 32 clients at once (batched together, prefix-cache misses), then
     the same 64 one at a time (mostly prefix-cache hits): the same probabilities to the last bit with int8 and w4a8,
     whose forward passes are batch-invariant.  bf16 runs cuBLAS, whose kernels depend on the batch size: there the
     largest difference is reported, not checked
  6. /health and /stats answer at the end

--quick: 8k evidence only, levels 8 and 32, one pass, 10 s per 8k level (about two minutes).  The procedure for every
engine of the comparison (llama.cpp, SGLang, vLLM, StartLux-Decision, decider, JevK5): scripts/stress.sh.
"""
import argparse
import http.client
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from run_http import Client, GPUSampler, make_body, summarize, uniquify  # noqa: E402
from run_jevbench import load, sequential, throughput, warmup  # noqa: E402
from run_long import QUESTION, concurrency, sized_state  # noqa: E402
import suites  # noqa: E402


def get_json(url, path):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    @property
    def failed(self):
        return [c["check"] for c in self.checks if not c["ok"]]


def long_evidence(a, url, tok, cap, rep, out):
    states = {}
    for target, n in zip(a.lengths, a.n_long):
        state, n_tok = sized_state(tok, target)
        states[target] = state
        if n_tok > cap - 64:
            rep.check(f"long evidence {target}", True, f"skipped: {n_tok} tokens is beyond the largest bucket ({cap})")
            continue
        c = Client(url, timeout=600)
        c.call(make_body(state, QUESTION))                          # warmup
        lat, counted, answered = [], set(), True
        for _ in range(n):
            ms, resp = c.call(make_body(state, QUESTION))
            answered &= "failed" in resp.get("answers", {})
            lat.append(ms)
            counted.add(resp.get("usage", {}).get("input_tokens"))
        whole = all(isinstance(t, int) and n_tok <= t <= n_tok + 64 for t in counted)  # + the request number
        out[str(target)] = {"tokens_sent": n_tok, "tokens_counted": sorted(counted, key=str), **summarize(lat)}
        rep.check(f"long evidence {target}", answered and whole,
                  f"{n_tok} tokens sent, server counted {sorted(counted, key=str)}, p50 {statistics.median(lat):.1f} ms")
    return states


def capacity(url, tok, cap, rep, out):
    c = Client(url, timeout=600)
    inside, n_in = sized_state(tok, cap - 512)
    try:
        ms, resp = c.call(make_body(inside, QUESTION))
        ok_in, det_in = "failed" in resp.get("answers", {}), f"{n_in} tokens answered in {ms:.0f} ms"
    except Exception as e:  # noqa: BLE001
        ok_in, det_in = False, f"{n_in} tokens: {repr(e)[:160]}"
    beyond, n_out = sized_state(tok, cap + 1024)
    try:
        c.call(make_body(beyond, QUESTION))
        ok_out, det_out = False, f"{n_out} tokens answered (expected HTTP 422)"
    except RuntimeError as e:
        ok_out, det_out = "HTTP 422" in str(e), f"{n_out} tokens refused: {str(e)[:60]}"
    st, qs = suites.fixed()["yesno"]
    try:
        Client(url, timeout=60).call(make_body(st, qs))
        ok_after = True
    except Exception:  # noqa: BLE001
        ok_after = False
    out.update(max_tokens=cap, inside_tokens=n_in, beyond_tokens=n_out)
    rep.check("capacity", ok_in and ok_out and ok_after,
              f"largest bucket {cap}: {det_in}; {det_out}; answering afterwards: {ok_after}")


def long_concurrency(a, url, state, rep, out):
    for clients in a.levels:
        gpu = GPUSampler()
        gpu.start()
        r = concurrency(url, state, clients, a.conc_seconds)
        r["gpu"] = gpu.stop()
        out[str(clients)] = r
        rep.check(f"8k evidence, {clients} clients", r["errors"] == 0 and r["requests"] > 0,
                  f"{r['requests_per_s']} req/s, p50 {r.get('p50_ms')} ms, {r['errors']} failed")


def jevbench(a, url, rep, out):
    items, score = load(a.jevbench)
    warmup(url, a.model, items)
    seq = sequential(url, a.model, items, score)
    base = sum(v["correct"] for v in seq.values())
    out["sequential"] = {"correct": base, "n": len(items),
                         "mean_ms": round(statistics.mean(v["ms"] for v in seq.values()), 2)}
    print(f"JevBench one at a time: {base}/{len(items)}", flush=True)
    out["levels"] = {}
    for clients in a.levels:
        r = throughput(url, a.model, items, score, clients, a.passes, GPUSampler())
        out["levels"][str(clients)] = r
        close = all(abs(x - base) <= a.score_tolerance for x in r["correct_per_pass"])
        rep.check(f"JevBench, {clients} clients", r["errors"] == 0 and close,
                  f"{r['requests_per_s']} req/s, p50 {r['latency'].get('p50_ms')} ms, correct per pass "
                  f"{r['correct_per_pass']} (one at a time {base}), {r['errors']} failed")
    return items


def under_load(a, url, items, precision, rep, out):
    """The same 64 requests from 32 clients at once, then one at a time: the answers must not move (int8, w4a8)."""
    pick = random.Random(12345).sample(range(len(items)), min(64, len(items)))
    bodies = [json.dumps({"model": a.model, "state": uniquify(items[i][1]["state"], 900000 + j),
                          "questions": {"decision": items[i][1]["question"]}}).encode() for j, i in enumerate(pick)]
    together, lock, nxt, errors = [None] * len(bodies), threading.Lock(), [0], []

    def worker():
        c = Client(url, timeout=600)
        while True:
            with lock:
                j = nxt[0]
                nxt[0] += 1
            if j >= len(bodies):
                return
            try:
                together[j] = c.call(bodies[j])[1]["answers"]["decision"]
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e)[:200])
                c = Client(url, timeout=600)
    ts = [threading.Thread(target=worker) for _ in range(32)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        out.update(requests=len(bodies), errors=errors[:3])
        rep.check("answers under load", False, f"{len(errors)} of {len(bodies)} requests failed: {errors[0]}")
        return
    c = Client(url, timeout=600)
    alone = [c.call(b)[1]["answers"]["decision"] for b in bodies]

    def probs(ans):
        if "probabilities" in ans:
            return {str(k): float(v) for k, v in ans["probabilities"].items()}
        return {k: float(v) for k, v in ans.items() if isinstance(v, (int, float))}
    same = sum(x == y for x, y in zip(together, alone))
    worst = max(max(abs(probs(x)[k] - probs(y).get(k, 0.0)) for k in probs(x)) for x, y in zip(together, alone))
    out.update(requests=len(bodies), identical=same, max_abs_dp=worst)
    detail = f"{same}/{len(bodies)} answers identical, largest probability difference {worst:.2e}"
    if precision in ("int8", "w4a8"):
        rep.check("answers under load", same == len(bodies), detail)
    else:
        rep.check("answers under load (bf16: reported, not checked)", True, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the model's tokenizer (sizes the long evidence)")
    ap.add_argument("--jevbench", required=True, help="the JevBench checkout (datasets/public and its scorer)")
    ap.add_argument("--url", default="http://127.0.0.1:8092/v1/systemone")
    ap.add_argument("--start", choices=("int8", "bf16", "w4a8"),
                    help="start the server with scripts/serve.sh (and stop it)")
    ap.add_argument("--model", default="StartLux-Decision", help="model field of the requests")
    ap.add_argument("--lengths", default="8000,50000")
    ap.add_argument("--n-long", default="5,3", help="timed requests per length")
    ap.add_argument("--levels", default="8,16,32")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--conc-seconds", type=float, default=20)
    ap.add_argument("--score-tolerance", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.quick:
        a.lengths, a.n_long, a.levels, a.passes, a.conc_seconds = "8000", "3", "8,32", 1, 10
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.n_long = [int(x) for x in a.n_long.split(",")]
    a.levels = [int(x) for x in a.levels.split(",")]
    if a.start:
        r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "serve.sh"), "start", a.start], capture_output=True,
                           text=True)
        print(r.stdout.strip()[-400:], flush=True)
        if r.returncode:
            sys.exit(f"server did not start: {r.stdout[-600:]}{r.stderr[-600:]}")
    rep = Report()
    res = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(), "url": a.url,
           "gpu_info": GPUSampler().info(), "args": {k: v for k, v in vars(a).items()}}
    try:
        st, health = get_json(a.url, "/health")
        res["health"] = health
        cap, precision = int(health.get("max_tokens", 0)), health.get("precision", "?")
        if not cap:
            sys.exit("/health does not report max_tokens: not a flash-jev server, or an older one")
        print(f"server: {health}", flush=True)
        from flash_jev.prompt import Tokenizer
        tok = Tokenizer(os.path.expanduser(a.model_dir))
        res["long"], res["capacity"], res["long_concurrency"], res["jevbench"], res["under_load"] = {}, {}, {}, {}, {}
        states = long_evidence(a, a.url, tok, cap, rep, res["long"])
        capacity(a.url, tok, cap, rep, res["capacity"])
        long_concurrency(a, a.url, states.get(8000) or sized_state(tok, 8000)[0], rep, res["long_concurrency"])
        items = jevbench(a, a.url, rep, res["jevbench"])
        under_load(a, a.url, items, precision, rep, res["under_load"])
        s1, _ = get_json(a.url, "/health")
        s2, stats = get_json(a.url, "/stats")
        res["stats"] = stats
        rep.check("healthy at the end", s1 == 200 and s2 == 200, f"/health {s1}, /stats {s2}")
    except Exception as e:  # noqa: BLE001
        rep.check("run", False, repr(e)[:300])
    finally:
        if a.start:
            subprocess.run(["bash", os.path.join(ROOT, "scripts", "serve.sh"), "stop"], capture_output=True)
    res["checks"] = rep.checks
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"{len(rep.checks) - len(rep.failed)}/{len(rep.checks)} checks passed"
          + (f"; failed: {', '.join(rep.failed)}" if rep.failed else ""))
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
