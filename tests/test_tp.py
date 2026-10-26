"""Tensor parallelism (startlux_prism/tp.py) against one GPU, bit for bit, on the int8 path.

    python tests/test_tp.py --gguf X.gguf --model-dir DIR --calib C.pt --tp 2 --gpus 2,3 --ref-gpu 1 \\
        --jevbench JEVBENCH/datasets/public [--mmproj F] [--tokens 64]
    python tests/test_tp.py --server --gguf X.gguf --model-dir DIR --calib C.pt --tp 2 --gpus 2,3 --ref-gpu 1 \\
        --jevbench JEVBENCH [--think-mode auto] [--clients 1]

The same work runs on an engine over --tp GPUs and on one GPU (--ref-gpu), each in a process of its own, and every
result is compared bit for bit:

1. read-outs: the HTTP suites' requests and every 10th JevBench item, each alone, again (prefix-cache hits), and as one
   batch; requests with long evidence as a miss, a hit, another question about the same evidence, and inside mixed
   batches; a wide choice list (knock-out rounds through the Decider) -> letter logits and answers;
2. decode: thinking prompts of JevBench items prefilled with their state kept, then greedy steps all at once, the same
   sequences joining a running batch at different steps, and seeded sampled steps -> every step's full-vocabulary
   logits (SHA-1 of their bytes) and tokens;
3. images (--mmproj): bench/vision_suite.py's requests through the Decider -> answers and letter logits.

--server: a server with --tp and one on --ref-gpu, then every JevBench item (--think-mode, thoughts returned with
their text, details on) through both: answers, probabilities, thoughts (text, tokens, how they ended) and usage must be
identical; with --clients > 1 the items go from that many clients at once (batches of any composition).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))


# ------------------------------------------------------------------------------------------------ the engine work
def work(a):
    """One side of the comparison (this process: --tp engine, or one GPU) -> {name: list of tensors / values}."""
    if a.tp > 1:
        from startlux_prism import tp as TP
        TP.start(a.tp, a.gpus)
    else:
        from startlux_prism.backends import device_env
        env = device_env(a.gpus)
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            if key in env:
                os.environ[key] = env[key]
            else:
                os.environ.pop(key, None)
    import numpy as np
    import torch
    from startlux_prism import images as I
    from startlux_prism import jevfmt as J
    from startlux_prism.decide import Decider
    from startlux_prism.engine.decode import Decoder, Row, Sampling
    from startlux_prism.engine.model import Engine, Options, group
    from startlux_prism.prompt import rows_for
    from startlux_prism.think import RELEASE_PREFIX, Framing
    from run_http import uniquify
    import suites

    n_seq = 8
    rows_each = a.max_prompt + a.tokens + 8
    opt = Options(precision="int8", calib=a.calib, max_tokens=16384, prefix_cache=8, prefix_cache_tokens=16384,
                  think_seqs=n_seq, think_kv_rows=n_seq * rows_each, mmproj=a.mmproj or "", tp=a.tp)
    t0 = time.time()
    eng = Engine(a.gguf, a.model_dir, opt)
    print(f"[tp {a.tp}] engine loaded in {time.time() - t0:.0f} s", flush=True)
    tok, out = eng.tok, {}

    def cpu(x):
        return [cpu(v) for v in x] if isinstance(x, (list, tuple)) else x.detach().float().cpu()

    def rows(state, questions):
        recs = [J.from_systemone(state, q) for q in questions.values()]
        return rows_for(tok, recs, list(questions))

    # 1. read-outs: alone, again (cache hits), one batch
    reqs = [v for k, v in suites.fixed().items() if k != "wide33"]
    reqs += [(s, q) for _, _, s, q, _ in suites.jevbench(a.jevbench)[::10]]
    groups = [[r.ids for r in rows(s, q)] for s, q in reqs]
    batch, n = [], 0
    for g in groups:
        if n + sum(len(r) for r in g) <= 7500:
            batch.append(g)
            n += sum(len(r) for r in g)
    out["single"] = [cpu(eng.run([g])) for g in groups]
    out["again"] = [cpu(eng.run([g])) for g in groups]
    out["batch"] = cpu(eng.run(batch))
    # long evidence: miss, hit, another question about it, mixed batches
    long = [x for x in suites.jevbench(a.jevbench) if len(json.dumps(x[2])) > 600][:4]
    other = {"other": {"type": "noul", "instructions": "Is the evidence complete enough to act on?"}}
    cache = []
    for i, (_, _, state, questions, _) in enumerate(long):
        st = uniquify(state, 1000 + i)
        r = rows(st, questions)
        cache += [cpu(eng.letter_logits(r)), cpu(eng.letter_logits(r)), cpu(eng.letter_logits(rows(st, other)))]
    fx = suites.fixed()
    ra, rb, ry = (rows(uniquify(fx[k][0], 2000 + j), fx[k][1]) for j, k in enumerate(("startlux3", "longpage2",
                                                                                         "yesno")))
    cache.append(cpu(eng.run([group(rb), group(ry), group(ra)])))
    cache.append(cpu(eng.run([group(rows(uniquify(fx["longpage2"][0], 2001), fx["longpage2"][1])), group(ra),
                              group(ry)])))
    out["cache"] = cache
    out["cache_hits"] = [eng.cache.stats["hits"]]
    # wide choice lists: knock-out rounds through the Decider
    wide_state, wide_q = fx["wide33"]
    answers, _ = Decider(tok, eng).decide(wide_state, wide_q)
    out["wide"] = [json.dumps(answers, sort_keys=True)]
    print(f"[tp {a.tp}] read-outs done", flush=True)

    # 2. decode
    dec = Decoder(eng, a.gguf, max_seqs=n_seq, sampling=Sampling(greedy=True))
    dec.warm_up()
    framing = Framing(a.model_dir)
    prompts = []
    for name, _, state, questions, _ in suites.jevbench(a.jevbench):
        q = next(iter(questions.values()))
        rec = J.from_systemone(state, q, allow_none=q.get("type", "choice") == "choice")
        ids = tok.encode(framing.prompt(rec) + RELEASE_PREFIX)
        if len(ids) <= a.max_prompt:
            prompts.append(ids)
    prompts = prompts[::max(1, len(prompts) // 6)][:6]

    def prefill(seqs):
        keep = [(s["ids"][:-1], s["slot"], s["kv"]) for s in seqs]
        b = eng.launch(eng.prepare([], 0, keep=keep))
        eng.finish(b)
        pos = b.plan.tables["pos"]
        for s, seg in zip(seqs, b.plan.segs[-len(seqs):]):
            s.update(pos=int(pos[seg.start + len(seg.tokens) - 1]) + 1, kv_len=len(seg.tokens), step=0, row=-1)

    def step(seqs, slot, sampling=None):
        rr = []
        for i, s in enumerate(seqs):
            src = s["row"] if s["row"] >= 0 else -1 - s["ids"][-1]
            rr.append(Row(s["slot"], s["kv"], s["kv_len"], s["pos"], s["seed"], s["step"], src, sampling))
            s.update(kv_len=s["kv_len"] + 1, pos=s["pos"] + 1, step=s["step"] + 1, row=i)
        toks = dec.finish(dec.launch(rr, slot))
        lg = dec.logits(len(rr))
        for i, s in enumerate(seqs):
            s["gen"].append(int(toks[i]))
            s["hash"].append(hashlib.sha1(lg[i].cpu().numpy().tobytes()).hexdigest())

    def seqs(seed0=1234):
        return [dict(ids=ids, slot=eng.think_slot0 + i, kv=eng.think_kv0 + i * rows_each, seed=seed0 + i, gen=[],
                     hash=[]) for i, ids in enumerate(prompts)]
    S = seqs()
    prefill(S)
    for k in range(a.tokens):
        step(S, k % 2)
    out["decode_tokens"] = [torch.tensor([s["gen"] for s in S])]
    out["decode_logits"] = [json.dumps([s["hash"] for s in S])]
    S = seqs()                                       # joining a running batch: sequence i at step 8 i
    prefill(S)
    active = []
    for t in range(a.tokens // 2 + 8 * (len(S) - 1)):
        active += [s for i, s in enumerate(S) if t == 8 * i]
        live = [s for s in active if len(s["gen"]) < a.tokens // 2]
        if live:
            step(live, t % 2)
    out["join_tokens"] = [torch.tensor([s["gen"] for s in S])]
    out["join_logits"] = [json.dumps([s["hash"] for s in S])]
    S = seqs(seed0=77)                               # sampled (T 0.6, top-k 20, top-p 0.95), seeded
    prefill(S)
    for k in range(a.tokens):
        step(S, k % 2, Sampling(0.6, 0.95, 20))
    out["sample_tokens"] = [torch.tensor([s["gen"] for s in S])]
    out["sample_logits"] = [json.dumps([s["hash"] for s in S])]
    print(f"[tp {a.tp}] decode done", flush=True)

    # 3. images
    if a.mmproj:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vision_suite", os.path.join(ROOT, "bench", "vision_suite.py"))
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        img = []
        for name, state, qs, values, _ in vs.requests():
            ans, _ = Decider(tok, eng, images=I.Settings()).decide(state, qs, images=values)
            img.append(json.dumps(ans, sort_keys=True))
        out["images"] = img
        print(f"[tp {a.tp}] images done", flush=True)
    torch.save(out, a.out)
    print(f"[tp {a.tp}] -> {a.out}", flush=True)
    if a.tp > 1:
        TP.group().close()
    del np


def compare(pa, pb):
    import torch
    x, y = torch.load(pa), torch.load(pb)
    bad = 0
    for k in x:
        xa, xb = _flat(x[k]), _flat(y.get(k, []))
        same = sum((torch.equal(p, q) if isinstance(p, torch.Tensor) else p == q) for p, q in zip(xa, xb))
        ok = same == len(xa) == len(xb)
        bad += not ok
        print(f"  {k:14s}: {same}/{len(xa)} identical  {'ok' if ok else 'DIFFERENT'}", flush=True)
    return bad


def _flat(x):
    return [t for v in x for t in _flat(v)] if isinstance(x, (list, tuple)) else [x]


# ------------------------------------------------------------------------------------------------ servers
def server_compare(a):
    """--server: JevBench through a --tp server and a one-GPU server; answers and thoughts identical."""
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    from run_http import Client
    from run_jevbench import load
    import threading
    items, _ = load(a.jevbench)
    common = ["--gguf", a.gguf, "--model-dir", a.model_dir, "--precision", "int8", "--calib", a.calib]
    if a.mmproj:
        common += ["--mmproj", a.mmproj]
    procs = []
    logs = []
    for label, gpus, tp, port in (("one GPU", a.ref_gpu, 1, a.port), (f"tp {a.tp}", a.gpus, a.tp, a.port + 1)):
        from startlux_prism.backends import device_env
        env = device_env(gpus)
        log = open(os.path.join(a.log_dir, f"server-tp{tp}.log"), "w")
        logs.append(log)
        cmd = [sys.executable, "-m", "startlux_prism.server", "--port", str(port)] + common
        if tp > 1:
            cmd += ["--tp", str(tp), "--gpus", gpus]
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    try:
        for port in (a.port, a.port + 1):
            _wait_health(port, procs)
        extra = {"think_mode": a.think_mode, "return_thought": True, "return_details": True}
        bodies = [json.dumps({"state": raw["state"], "questions": {"decision": raw["question"]}, **extra}).encode()
                  for _, raw in items]
        res = []
        for port in (a.port, a.port + 1):
            url = f"http://127.0.0.1:{port}/v1/systemone"
            got = [None] * len(bodies)
            nxt, lock = [0], threading.Lock()

            def worker():
                c = Client(url, timeout=1200)
                while True:
                    with lock:
                        i = nxt[0]
                        nxt[0] += 1
                    if i >= len(bodies):
                        return
                    got[i] = c.call(bodies[i])[1]
            t0 = time.time()
            ts = [threading.Thread(target=worker) for _ in range(a.clients)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            print(f"  port {port}: {len(bodies)} items from {a.clients} client(s) in {time.time() - t0:.1f} s",
                  flush=True)
            res.append(got)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    bad = thoughts = 0
    for (tier, raw), r1, r2 in zip(items, res[0], res[1]):
        a1, a2 = _strip(r1), _strip(r2)
        th = (r1["answers"]["decision"].get("details") or {}).get("thought") or {}
        thoughts += bool(th.get("used"))
        if a1 != a2:
            bad += 1
            print(f"  DIFFERENT: {raw['id']}", flush=True)
    print(f"server, think_mode {a.think_mode}, {a.clients} client(s): {len(items) - bad}/{len(items)} answers "
          f"identical (probabilities, thoughts: text, tokens, end; usage), {thoughts} items thought", flush=True)
    return bad


def _strip(resp):
    """An answer without its timings."""
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in ("ms", "total_ms", "timing_ms")}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    return json.dumps({"answers": clean(resp["answers"]), "usage": resp.get("usage")}, sort_keys=True)


def _wait_health(port, procs, timeout=1800):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f"a server exited with code {p.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:                                    # noqa: BLE001
            time.sleep(3)
    raise RuntimeError(f"server on port {port} not up after {timeout} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--jevbench", required=True, help="JevBench's datasets/public (engine) or its checkout (--server)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpus", default="2,3", help="the --tp engine's GPUs")
    ap.add_argument("--ref-gpu", default="1", help="the one-GPU engine's GPU")
    ap.add_argument("--tokens", type=int, default=64, help="decode steps per sequence")
    ap.add_argument("--max-prompt", type=int, default=3000)
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--think-mode", default="auto")
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return work(a)
    if a.server:
        sys.exit(1 if server_compare(a) else 0)
    outs = []
    procs = []
    for tp, gpus in ((1, a.ref_gpu), (a.tp, a.gpus)):
        out = os.path.join(a.out_dir, f"test_tp-{os.getpid()}-tp{tp}.pt")
        outs.append(out)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--tp", str(tp), "--gpus", gpus, "--out", out]
        for k in ("gguf", "model_dir", "calib", "mmproj", "jevbench", "tokens", "max_prompt"):
            v = getattr(a, k)
            if v not in ("", None):
                cmd += [f"--{k.replace('_', '-')}", str(v)]
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"a worker failed: exit codes {codes}")
    print(f"tp {a.tp} (GPUs {a.gpus}) against one GPU ({a.ref_gpu}), bit for bit:")
    bad = compare(*outs)
    print("all ok" if not bad else f"{bad} DIFFERENT")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
