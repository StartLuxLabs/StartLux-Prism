"""Long-context benchmark for any /v1/systemone server: one question about a long piece of evidence (an application
log), sized in StartLux's prompt format to about --lengths tokens (8k and 50k by default).  Latency at one request at
a time, and throughput with --conc clients for the lengths listed in --conc-lengths.

    python bench/run_long.py --url http://127.0.0.1:8092/v1/systemone --label startlux-prism-int8 \\
        --model-dir ~/jev/models/StartLux-Decision-4B-Q8_0-GGUF [--lengths 8000,50000] [--out results/long/x.json]

Throughput at several concurrency levels, without the latency runs (--n 0: a short warmup instead):

    python bench/run_long.py ... --lengths 8000 --n 0 --conc 8,16,32 --conc-lengths 8000 --conc-seconds 30

Every request is unique (bench/run_http.uniquify: a request number at the start of the evidence), so no engine can
reuse a previous request's prefix.  The token count each engine reports (usage.input_tokens) is recorded: an engine
that truncates long evidence shows it there.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from run_http import Client, GPUSampler, make_body, pct, summarize  # noqa: E402

QUESTION = {"failed": {"type": "noul", "instructions": "Does the log show a payment that failed and was never "
                                                       "retried successfully?"}}
SERVICES = ("payment-svc", "order-svc", "auth-svc", "ship-svc", "search-svc", "cart-svc")
LEVELS = ("INFO", "INFO", "INFO", "WARN", "ERROR")


def log_lines(n, seed=7):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        svc, lvl = rnd.choice(SERVICES), rnd.choice(LEVELS)
        oid = 40000 + rnd.randrange(60000)
        msg = {"payment-svc": f"order {oid} charge {rnd.randrange(5, 3000)}.{rnd.randrange(100):02d} USD "
                              f"card ****{rnd.randrange(10000):04d} {'declined: insufficient funds' if lvl == 'ERROR' else 'ok'}",
               "order-svc": f"order {oid} state {rnd.choice(('created', 'paid', 'packed', 'cancelled'))}",
               "auth-svc": f"user u{rnd.randrange(9999)} login {'failed' if lvl != 'INFO' else 'ok'} from 10.{rnd.randrange(255)}.{rnd.randrange(255)}.{rnd.randrange(255)}",
               "ship-svc": f"parcel P{rnd.randrange(10**7):07d} for order {oid} {rnd.choice(('label', 'picked up', 'in transit', 'delivered'))}",
               "search-svc": f"query '{rnd.choice(('desk lamp', 'usb cable', 'office chair', 'monitor arm'))}' {rnd.randrange(40)} hits in {rnd.randrange(3, 400)} ms",
               "cart-svc": f"cart c{rnd.randrange(99999)} {rnd.choice(('add', 'remove', 'checkout'))} sku-{rnd.randrange(9999)}"}[svc]
        out.append(f"2026-09-{1 + i * 30 // max(n, 1):02d} {rnd.randrange(24):02d}:{rnd.randrange(60):02d}:"
                   f"{rnd.randrange(60):02d} {lvl:5s} {svc}[{rnd.randrange(8)}] {msg}")
    return out


def sized_state(tok, target):
    """{'log': ...} whose prompt (StartLux's format, with the question) is about `target` tokens."""
    from startlux_prism import jevfmt as J
    from startlux_prism.prompt import render_text

    def tokens(n):
        st = {"service": "shop backend", "log": "\n".join(log_lines(n))}
        return len(tok.encode(render_text(J.from_systemone(st, QUESTION["failed"]))[0])), st
    lo, hi = 1, 64
    while tokens(hi)[0] < target:
        hi *= 2
    while lo < hi:                          # smallest line count reaching the target
        mid = (lo + hi) // 2
        if tokens(mid)[0] < target:
            lo = mid + 1
        else:
            hi = mid
    n_tok, st = tokens(lo)
    return st, n_tok


def latency(url, state, n, warmup):
    c = Client(url, timeout=600)
    lat, toks = [], set()
    for i in range(warmup + n):
        ms, out = c.call(make_body(state, QUESTION))
        if "failed" not in out.get("answers", {}):
            raise RuntimeError(f"no answer: {str(out)[:200]}")
        if i >= warmup:
            lat.append(ms)
            toks.add(out.get("usage", {}).get("input_tokens"))
    r = summarize(lat)
    r["input_tokens_reported"] = sorted(toks, key=str)
    return r


def concurrency(url, state, clients, seconds):
    stop, lock, done, errors, lat = time.perf_counter() + seconds, threading.Lock(), [0], [0], []

    def worker():
        c = Client(url, timeout=600)
        while time.perf_counter() < stop:
            try:
                ms, _ = c.call(make_body(state, QUESTION))
                with lock:
                    done[0] += 1
                    lat.append(ms)
            except Exception:  # noqa: BLE001
                with lock:
                    errors[0] += 1
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    el = time.perf_counter() - t0
    return {"clients": clients, "seconds": round(el, 1), "requests": done[0], "errors": errors[0],
            "requests_per_s": round(done[0] / el, 2),
            **({"p50_ms": round(statistics.median(lat), 1), "p95_ms": round(pct(lat, 0.95), 1)} if lat else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-dir", required=True, help="tokenizer for sizing the evidence (StartLux's format)")
    ap.add_argument("--lengths", default="8000,50000")
    ap.add_argument("--n", default="20,8", help="timed requests per length")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--conc", default="4", help="clients, or several levels (8,16,32)")
    ap.add_argument("--conc-lengths", default="8000")
    ap.add_argument("--conc-seconds", type=float, default=40)
    ap.add_argument("--meta", default="{}")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    from startlux_prism.prompt import Tokenizer
    tok = Tokenizer(os.path.expanduser(a.model_dir))
    gpu = None if a.no_gpu else GPUSampler()
    res = {"label": a.label, "url": a.url, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
           "gpu_info": gpu.info() if gpu else None, "meta": json.loads(a.meta), "lengths": {}}
    print(f"{a.label}: GPU processes at start {(res['gpu_info'] or {}).get('processes')}", flush=True)
    for target, n in zip([int(x) for x in a.lengths.split(",")], [int(x) for x in a.n.split(",")]):
        state, n_tok = sized_state(tok, target)
        r = {"prompt_tokens_startlux_format": n_tok}
        try:
            if n > 0:
                gpu and gpu.start()
                r["latency"] = latency(a.url, state, n, a.warmup)
                r["latency"]["gpu"] = gpu.stop() if gpu else {}
            else:                                   # throughput only: warm up this length first
                c = Client(a.url, timeout=600)
                for _ in range(max(a.warmup, 1)):
                    c.call(make_body(state, QUESTION))
            if str(target) in a.conc_lengths.split(","):
                levels = [int(x) for x in str(a.conc).split(",")]
                for clients in levels:
                    gpu and gpu.start()
                    cc = concurrency(a.url, state, clients, a.conc_seconds)
                    cc["gpu"] = gpu.stop() if gpu else {}
                    if len(levels) == 1:
                        r["concurrency"] = cc
                    else:
                        r.setdefault("concurrency_levels", {})[str(clients)] = cc
                    print(f"  {target:6d} target: c={clients:3d}  {cc['requests_per_s']:7.2f} req/s  "
                          f"p50 {cc.get('p50_ms', 0):9.1f}  p95 {cc.get('p95_ms', 0):9.1f} ms  errors {cc['errors']}  "
                          f"SM {cc['gpu'].get('sm_util_mean', '-')}%", flush=True)
        except Exception as e:  # noqa: BLE001
            if gpu and gpu.t is not None and gpu.t.is_alive():
                gpu.stop()
            r["error"] = repr(e)[:300]
        res["lengths"][str(target)] = r
        lt = r.get("latency", {})
        cc = r.get("concurrency", {})
        print(f"  {target:6d} target, {n_tok} tokens: "
              + (f"p50 {lt['p50_ms']:9.1f} ms  mean {lt['mean_ms']:9.1f}  min {lt['min_ms']:9.1f}  reported tokens "
                 f"{lt['input_tokens_reported']}" if lt else (f"ERROR {r.get('error')}" if "error" in r else "done"))
              + (f"  |  c={cc['clients']}: {cc['requests_per_s']} req/s, errors {cc['errors']}" if cc else ""),
              flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
