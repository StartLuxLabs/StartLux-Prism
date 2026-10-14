"""Where a request's time goes outside the GPU on the StartLux-Prism server, at one request at a time: how much a native
front end (HTTP, JSON, prompt rendering, planning in C++ instead of Python) could save at most.

    python bench/host_overhead.py --url http://127.0.0.1:8092/v1/systemone [--n 300] [--json out.json]

Per suite (unique requests, as bench/run_http.py sends them), medians of:
  client      round trip measured by the client (Python http.client, as in every benchmark)
  http        the same client's round trip to GET /health: socket, HTTP parsing on both sides, no work
  server      the handler, from the parsed body to the encoded answer (latency_ms in the response)
    split       request -> questions (validation, wide-choice planning)
    rows        prompt rendering (StartLux's jevfmt) + tokenization (Rust tokenizers)
    forward     submit -> logits: queueing, planning and table staging (host_ms), graph launch + GPU (gpu_ms)
    answers     softmax with temperatures, answer objects
  json        response encoding and what the handler does after its last mark
GPU time is gpu_ms (graph launch to completion seen on the host); the rest of `client` is host work, part of which
(client, kernel networking) any server pays.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import statistics
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
import suites as R  # noqa: E402
from run_http import Client, make_body  # noqa: E402


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--suites", default="yesno,review1,readme3,startlux1,startlux3,longpage2")
    ap.add_argument("--json")
    a = ap.parse_args()
    c = Client(a.url)
    out = {}
    u = urllib.parse.urlparse(a.url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    hs = []
    for _ in range(a.warmup + a.n):                  # the same keep-alive client, a request with no work behind it
        t = time.perf_counter()
        conn.request("GET", "/health")
        conn.getresponse().read()
        hs.append((time.perf_counter() - t) * 1000)
    out["http_health_ms"] = med(hs[a.warmup:])
    print(f"GET /health round trip: {out['http_health_ms']:.3f} ms", flush=True)
    fixed = R.fixed()
    print(f"{'suite':10s} {'client':>7s} {'server':>7s} {'split':>6s} {'rows':>6s} {'fwd':>7s} {'host':>6s} "
          f"{'gpu':>7s} {'answ':>6s} {'json+':>6s} | {'non-GPU':>7s} {'%':>5s}", flush=True)
    for name in a.suites.split(","):
        state, questions = fixed[name]
        rec = {k: [] for k in ("client", "server", "split", "rows", "forward", "answers", "host", "gpu", "submit")}
        for i in range(a.warmup + a.n):
            ms, res = c.call(make_body(state, questions))
            if i < a.warmup:
                continue
            tm = res.get("timings", {})
            st = tm.get("stages_ms", {})
            rec["client"].append(ms)
            rec["server"].append(res["latency_ms"])
            for k in ("split", "rows", "forward", "answers"):
                rec[k].append(st.get(k, 0.0))
            rec["host"].append(tm.get("host_ms", 0.0))
            rec["gpu"].append(tm.get("gpu_ms", 0.0))
            rec["submit"].append(tm.get("submit_to_result_ms", 0.0))
        m = {k: med(v) for k, v in rec.items()}
        m["json_rest"] = round(m["server"] - (m["split"] + m["rows"] + m["forward"] + m["answers"]), 3)
        m["non_gpu"] = round(m["client"] - m["gpu"], 3)
        m["non_gpu_pct"] = round(100 * m["non_gpu"] / m["client"], 1)
        out[name] = m
        print(f"{name:10s} {m['client']:7.3f} {m['server']:7.3f} {m['split']:6.3f} {m['rows']:6.3f} {m['forward']:7.3f} "
              f"{m['host']:6.3f} {m['gpu']:7.3f} {m['answers']:6.3f} {m['json_rest']:6.3f} | {m['non_gpu']:7.3f} "
              f"{m['non_gpu_pct']:5.1f}", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
