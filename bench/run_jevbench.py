"""JevBench public (231 items) against any /v1/systemone server: accuracy with JevBench's own scorer, per-item latency,
and the label probabilities of every item (for decision agreement between engines).

    python bench/run_jevbench.py --url http://127.0.0.1:8090/v1/systemone --label startlux-prism-bf16 \\
        --jevbench ~/jev/src/jevbench-7ce310c7 --out results/jevbench/startlux-prism-bf16.json
    python bench/run_jevbench.py --compare results/jevbench/a.json results/jevbench/b.json
    python bench/run_jevbench.py --url ... --label ... --conc 8,16,32 [--passes 3] --out results/jevbench_conc/x.json

Scoring follows StartLux-Decision's eval (and s-gen-mlx's): JevBench commit 7ce310c7, score_task on the item's label
set, answers mapped to labels the same way.  Items run one at a time, in file order.  Before the timed pass, a warmup
pass sends every item (or the first --warmup) with a request number in front of its evidence (run_http.uniquify): every
prompt shape has been seen (engines that autotune or capture per shape do it there), but no timed item is in a prefix
cache.

--conc: throughput instead.  After the same warmup pass, per concurrency level c clients send --passes passes over the
231 items (each pass in its own fixed shuffled order, the same for every engine) as fast as answers come back: every
request carries its own request number (run_http.uniquify), so no request replays another's prefix, and each pass is
scored like the sequential run (the request number in front of the evidence can move a near-tie, so a pass may differ
from the sequential score by an item).  Requests per second is all requests over the level's wall time.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_http import Client, GPUSampler, summarize, uniquify  # noqa: E402

TIERS = ("easy", "original", "hard")


def score_keys(crit):
    try:
        return sorted(crit, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(crit)


def options(q):
    kind, crit = q.get("type"), q.get("criteria")
    if kind == "choice":
        return [str(k) for k in crit]
    if kind == "score":
        return [str(i) for i in range(len(crit))] if isinstance(crit, list) else [str(k) for k in crit]
    return ["no", "yes"]


def to_labels(q, ans):
    labels = options(q)
    if q["type"] == "noul":
        p = float(ans["noul"])
        return {"no": 1.0 - p, "yes": p}
    probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
    crit = q.get("criteria")
    if q["type"] == "score":
        if isinstance(crit, dict):
            probs = {str(k): probs.get(str(i), 0.0) for i, k in enumerate(score_keys(crit))}
        elif not all(label in probs for label in labels):
            probs = {str(i): probs.get(str(c), 0.0) for i, c in enumerate(crit)}
    return {label: probs.get(label, 0.0) for label in labels}


def load(jevbench):
    """The public items [(tier, item)], in file order, and score(item, answer) -> (correct, label probabilities) with
    JevBench's own scorer."""
    sys.path.insert(0, jevbench)
    from jevbench.scoring import score_task
    from jevbench.tasks import Task

    items = []
    for tier in TIERS:
        with open(os.path.join(jevbench, "datasets", "public", f"{tier}.jsonl")) as f:
            items += [(tier, json.loads(line)) for line in f]

    def score(raw, answer):
        labels = to_labels(raw["question"], answer)
        return bool(score_task(labels, Task.from_dict(raw))["correct"]), labels
    return items, score


EXTRA = {}            # --extra: request fields added to every request (e.g. StartLux-Decision v1.1's think_mode)


def body(model, raw, unique=True):
    """The request for one item (with a request number in front of its evidence when unique)."""
    return json.dumps({"model": model, "state": uniquify(raw["state"]) if unique else raw["state"],
                       "questions": {"decision": raw["question"]}, **EXTRA}).encode()


def thought(resp):
    """What a v1.1 AutoThink answer says about its thought (return_details), else {}."""
    d = resp["answers"]["decision"].get("details") or {}
    th = d.get("thought") or {}
    out = {"output_tokens": resp.get("usage", {}).get("output_tokens", 0)}
    if th:
        out.update(thought=th.get("used"), thought_tokens=th.get("tokens"), ended=th.get("ended"),
                   thought_ms=th.get("ms"), top=(d.get("first_pass") or {}).get("top"))
    return out


def warmup(url, model, items, n=-1):
    c = Client(url, timeout=600)
    for _, raw in (items if n < 0 else items[:n]):
        c.call(body(model, raw))


def sequential(url, model, items, score):
    """One item at a time, in file order, as they are (no request number) -> {id: tier, ms, correct, probs, tokens}."""
    c = Client(url, timeout=600)
    out = {}
    for tier, raw in items:
        ms, resp = c.call(body(model, raw, unique=False))
        ok, labels = score(raw, resp["answers"]["decision"])
        out[raw["id"]] = {"tier": tier, "ms": round(ms, 3), "correct": ok, "probs": labels,
                          "tokens": resp.get("usage", {}).get("input_tokens"), **thought(resp)}
    return out


def throughput(url, model, items, score, clients, passes, gpu=None):
    """`passes` passes over the items (pass p in random.Random(p)'s order, the same for every engine), every request
    with its own request number, from `clients` clients at once -> requests/s over the wall time, latency, failed
    requests and each pass's score."""
    reqs = []
    for p in range(passes):
        order = list(range(len(items)))
        random.Random(p).shuffle(order)
        reqs += [(p, items[i][1], body(model, items[i][1])) for i in order]
    lock, nxt, done = threading.Lock(), [0], []

    def worker():
        cl = Client(url, timeout=600)
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(reqs):
                return
            try:
                ms, resp = cl.call(reqs[i][2])
                done.append((i, ms, resp))
            except Exception as e:  # noqa: BLE001
                done.append((i, None, repr(e)[:200]))
                cl = Client(url, timeout=600)
    gpu and gpu.start()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    g = gpu.stop() if gpu else {}
    lat, errors, right = [], [], [0] * passes
    for i, ms, resp in done:
        p, raw, _ = reqs[i]
        if ms is None:
            errors.append(resp)
            continue
        lat.append(ms)
        try:
            right[p] += score(raw, resp["answers"]["decision"])[0]
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e)[:200])
    return {"clients": clients, "requests": len(reqs), "seconds": round(wall, 2),
            "requests_per_s": round(len(lat) / wall, 2), "errors": len(errors), "error_examples": errors[:3],
            "correct_per_pass": right, "latency": summarize(lat) if lat else {}, "gpu": g}


def run(a):
    items, score = load(a.jevbench)
    warmup(a.url, a.model, items, a.warmup)
    if a.conc:
        return concurrent(a, items, score)
    out = {"label": a.label, "items": sequential(a.url, a.model, items, score), "tiers": {}}
    for tier in TIERS:
        rows = [v for v in out["items"].values() if v["tier"] == tier]
        out["tiers"][tier] = {"correct": sum(v["correct"] for v in rows), "n": len(rows),
                              "mean_ms": round(statistics.mean(v["ms"] for v in rows), 2),
                              "p50_ms": round(statistics.median(v["ms"] for v in rows), 2)}
    allv = list(out["items"].values())
    out["total"] = {"correct": sum(v["correct"] for v in allv), "n": len(allv),
                    "mean_ms": round(statistics.mean(v["ms"] for v in allv), 2),
                    "p50_ms": round(statistics.median(v["ms"] for v in allv), 2)}
    thinking = [v for v in allv if v.get("thought")]
    if any("thought" in v for v in allv):
        ends = {}
        for v in thinking:
            ends[v["ended"]] = ends.get(v["ended"], 0) + 1
        toks = sorted(v["thought_tokens"] for v in thinking)
        out["thoughts"] = {"n": len(thinking), "share": round(len(thinking) / len(allv), 4), "ended": ends,
                           "tokens_mean": round(statistics.mean(toks), 1) if toks else 0,
                           "tokens_p50": toks[len(toks) // 2] if toks else 0, "tokens_max": toks[-1] if toks else 0,
                           "correct_thinking": sum(v["correct"] for v in thinking),
                           "mean_ms_thinking": round(statistics.mean(v["ms"] for v in thinking), 1) if thinking else 0,
                           "mean_ms_not_thinking": round(statistics.mean(v["ms"] for v in allv if not v.get("thought")),
                                                         2) if len(thinking) < len(allv) else 0}
    t = out["total"]
    print(f"{a.label}: {t['correct']}/{t['n']}  mean {t['mean_ms']} ms  p50 {t['p50_ms']} ms  | " +
          ", ".join(f"{k} {v['correct']}/{v['n']} {v['mean_ms']} ms" for k, v in out["tiers"].items()) +
          (f"  | thoughts {out['thoughts']}" if "thoughts" in out else ""))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


def concurrent(a, items, score):
    gpu = GPUSampler()
    out = {"label": a.label, "url": a.url, "passes": a.passes, "items": len(items), "gpu_info": gpu.info(), "levels": {}}
    for clients in [int(x) for x in a.conc.split(",")]:
        r = out["levels"][str(clients)] = throughput(a.url, a.model, items, score, clients, a.passes, gpu)
        print(f"{a.label}: c={clients:3d}  {r['requests_per_s']:8.2f} req/s  p50 {r['latency'].get('p50_ms', 0):8.1f}  "
              f"p95 {r['latency'].get('p95_ms', 0):8.1f} ms  correct per pass {r['correct_per_pass']} of {len(items)}  "
              f"errors {r['errors']}  SM {r['gpu'].get('sm_util_mean', '-')}%", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    ref = runs[0]
    top = lambda d: max(d, key=d.get)                                    # noqa: E731
    for r in runs:
        same = sum(top(r["items"][k]["probs"]) == top(v["probs"]) for k, v in ref["items"].items())
        dp = [max(abs(r["items"][k]["probs"][lab] - p) for lab, p in v["probs"].items()) for k, v in ref["items"].items()]
        t = r["total"]
        print(f"{r['label']:24s} {t['correct']}/{t['n']}  mean {t['mean_ms']:8.2f} ms  same decision as {ref['label']}: "
              f"{same}/{len(ref['items'])}  max |dp| {max(dp):.4f}  mean {sum(dp) / len(dp):.4f}  "
              f"above 0.05: {sum(d > 0.05 for d in dp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--label")
    ap.add_argument("--model", default="StartLux-Decision", help="model field of the requests")
    ap.add_argument("--jevbench", default=os.path.expanduser("~/jev/src/jevbench-7ce310c7"))
    ap.add_argument("--warmup", type=int, default=-1, help="warmup requests (default: one pass over all items)")
    ap.add_argument("--conc", default="", help="concurrency levels for a throughput run, e.g. 8,16,32")
    ap.add_argument("--passes", type=int, default=3, help="passes over the items per concurrency level (--conc)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--extra", default="", help='JSON object of fields added to every request, e.g. '
                    '\'{"think_mode": "never"}\' or \'{"return_details": true}\' (StartLux-Decision v1.1 controls)')
    a = ap.parse_args()
    EXTRA.update(json.loads(a.extra) if a.extra else {})
    if a.compare:
        compare(a.compare)
    else:
        run(a)


if __name__ == "__main__":
    main()
