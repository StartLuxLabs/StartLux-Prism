"""Where a request's HTTP latency goes: client round trip vs the server's own latency_ms (handler start to response
write) vs the engine's host_ms (plan + stage) and gpu_ms (launch to logits on the host), one request at a time.

    python bench/latency_split.py --url http://127.0.0.1:8092/v1/systemone [--request yesno] [--n 100]
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from run_http import Client, make_body  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--request", default="yesno")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    state, questions = suites.fixed()[a.request]
    c = Client(a.url)
    rows, stages = [], {}
    for i in range(a.n + 20):
        ms, resp = c.call(make_body(state, questions))
        if i >= 20:
            t = resp.get("timings") or {}
            rows.append((ms, resp.get("latency_ms", 0), t.get("host_ms", 0), t.get("gpu_ms", 0),
                         t.get("submit_to_result_ms", 0)))
            for k, v in (t.get("stages_ms") or {}).items():
                stages.setdefault(k, []).append(v)
    med = [statistics.median(x) for x in zip(*rows)]
    print(f"{a.request}: client {med[0]:.3f} ms | server handler {med[1]:.3f} | engine host {med[2]:.3f} + "
          f"launch..logits {med[3]:.3f} | outside handler {med[0] - med[1]:.3f} | handler minus engine "
          f"{med[1] - med[2] - med[3]:.3f} | submit..result {med[4]:.3f} (vs engine {med[2] + med[3]:.3f})")
    print("  handler stages (ms, median): " + ", ".join(f"{k} {statistics.median(v):.3f}" for k, v in stages.items()))


if __name__ == "__main__":
    main()
