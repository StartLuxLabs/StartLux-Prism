"""Prefix-cache benchmark for any /v1/systemone server: sessions of requests about the same evidence, as a decision
service sees them when a client asks a further question about evidence it has already sent (or sends one request's
questions as separate calls).

Every session has its own evidence (bench/run_http.uniquify), so nothing is reused across sessions.  Its first request
computes the evidence; each later request asks other questions about the same evidence, and an engine with a prefix
cache can start from it.  Reported per scenario: latency of first requests and of follow-ups, and the ratio.

    python bench/run_prefix.py --url http://127.0.0.1:8092/v1/systemone --label startlux-prism-int8 \\
        --jevbench ~/jev/src/jevbench-7ce310c7/datasets/public --sessions 40 --out results/prefix/startlux-prism-int8.json

Scenarios:
  ticket       StartLux's latency.py ticket (startlux3), its three questions as three requests in turn
  longpage     the long order page (longpage2), its two questions as two requests
  jevbench     the public JevBench items with the longest evidence: the item's own question, then two general questions
               about the same evidence
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import suites as R  # noqa: E402
from run_http import Client, GPUSampler, pct, uniquify  # noqa: E402

FOLLOW_UPS = {
    "complete": {"type": "noul", "instructions": "Is the evidence complete enough to act on without asking for more?"},
    "risk": {"type": "score", "instructions": "How risky would it be to act on this evidence right away?",
             "criteria": ["no risk", "low", "moderate", "high"]},
}


def scenarios(jevbench_dir, n_long):
    """name -> list of (state, [questions of request 1, questions of request 2, ...]) templates."""
    fx = R.fixed()
    st, qs = fx["startlux3"]
    out = {"ticket": [(st, [{k: v} for k, v in qs.items()])]}
    st, qs = fx["longpage2"]
    out["longpage"] = [(st, [{k: v} for k, v in qs.items()])]
    if jevbench_dir:
        items = R.jevbench(jevbench_dir)
        size = lambda s: len(s if isinstance(s, str) else json.dumps(s))       # noqa: E731
        items.sort(key=lambda x: -size(x[2]))
        out["jevbench"] = [(s, [q, {"complete": FOLLOW_UPS["complete"]}, {"risk": FOLLOW_UPS["risk"]}])
                           for _, _, s, q, _ in items[:n_long]]
    return out


MODEL = "StartLux-Decision"


def body(state, questions):
    return json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()


def stats_url(url):
    return url.rsplit("/v1/", 1)[0] + "/stats"


def server_stats(url):
    try:
        with urllib.request.urlopen(stats_url(url), timeout=5) as r:
            return json.loads(r.read()).get("prefix_cache")
    except Exception:  # noqa: BLE001 - only StartLux-Prism has /stats
        return None


def summ(xs):
    return {"n": len(xs), "mean_ms": round(statistics.mean(xs), 3), "p50_ms": round(pct(xs, 0.5), 3),
            "p95_ms": round(pct(xs, 0.95), 3)} if xs else {}


def run(url, templates, sessions, warmup, gap_ms):
    """Sessions round-robin over the templates, one request at a time on one connection.  -> first-request and
    follow-up latencies (and tokens)."""
    c = Client(url)
    first, follow, tokens = [], [], []
    for s in range(warmup + sessions):
        state, reqs = templates[s % len(templates)]
        state = uniquify(state)
        for j, qs in enumerate(reqs):
            ms, out = c.call(body(state, qs))
            missing = [k for k in qs if k not in out.get("answers", {})]
            if missing:
                raise RuntimeError(f"answers missing {missing}")
            if s >= warmup:
                (first if j == 0 else follow).append(ms)
                tokens.append(out.get("usage", {}).get("input_tokens"))
            if gap_ms:
                time.sleep(gap_ms / 1000)
    return first, follow, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--jevbench", help="JevBench public datasets directory (adds the jevbench scenario)")
    ap.add_argument("--long-items", type=int, default=20, help="longest JevBench items used as session evidence")
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=0, help="pause after each request (think time)")
    ap.add_argument("--scenarios", default="ticket,longpage,jevbench")
    ap.add_argument("--model", default="StartLux-Decision")
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    gpu = None if a.no_gpu else GPUSampler()
    sc = scenarios(a.jevbench, a.long_items)
    result = {"label": a.label, "url": a.url, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
              "gpu_info": gpu.info() if gpu else None, "meta": json.loads(a.meta), "sessions": a.sessions,
              "scenarios": {}, "cache_before": server_stats(a.url)}
    print(f"{a.label}: GPU processes at start {(result['gpu_info'] or {}).get('processes')}", flush=True)
    for name in [s for s in a.scenarios.split(",") if s in sc]:
        gpu and gpu.start()
        first, follow, tokens = run(a.url, sc[name], a.sessions, a.warmup, a.gap_ms)
        tele = gpu.stop() if gpu else {}
        r = {"first": summ(first), "follow_up": summ(follow), "gpu": tele,
             "requests_per_session": len(sc[name][0][1]), "input_tokens_mean": round(statistics.mean(tokens), 1)}
        r["follow_up_over_first_p50"] = round(r["follow_up"]["p50_ms"] / r["first"]["p50_ms"], 3)
        result["scenarios"][name] = r
        print(f"  {name:9s} first p50 {r['first']['p50_ms']:7.2f} mean {r['first']['mean_ms']:7.2f}  |  follow-up p50 "
              f"{r['follow_up']['p50_ms']:7.2f} mean {r['follow_up']['mean_ms']:7.2f} ms  (x{r['follow_up_over_first_p50']:.2f})"
              f"  {r['requests_per_session']} requests/session, {r['input_tokens_mean']:.0f} tokens/request",
              flush=True)
    result["cache_after"] = server_stats(a.url)
    if result["cache_after"]:
        print(f"  server prefix cache: {result['cache_after']}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
