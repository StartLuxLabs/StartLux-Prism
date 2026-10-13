"""Host-side cost of one request, piece by piece: JSON, prompt rendering, tokenization, batch planning, upload, GPU,
readout, answers.  Explains the gap between engine.run() and the HTTP latency.

    python bench/profile_host.py --gguf X.gguf --model-dir DIR [--request startlux3] [--precision int8]
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.decide import Decider, softmax_t  # noqa: E402
from flash_jev.engine.model import Engine, Options  # noqa: E402
from flash_jev.prompt import render_text, rows_for  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--calib", default=os.path.expanduser("~/jev/calib/startlux-4b-q8_0.pt"))
    ap.add_argument("--request", default="startlux3")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=4096, calib=a.calib))
    state, questions = suites.fixed()[a.request]
    body = json.dumps({"state": state, "questions": questions}).encode()
    tok = eng.tok
    times = {k: [] for k in ("json", "render", "tokenize", "plan", "upload", "gpu+sync", "answers", "decide_total")}
    for i in range(a.n + 5):
        t = [time.perf_counter()]
        req = json.loads(body)
        t.append(time.perf_counter())
        recs = [J.from_systemone(req["state"], q) for q in req["questions"].values()]
        texts = [render_text(r)[0] for r in recs]
        t.append(time.perf_counter())
        ids = tok.encode_batch(texts)
        t.append(time.perf_counter())
        plan, T, R = eng.plan([ids])
        t.append(time.perf_counter())
        eng._upload(plan)
        t.append(time.perf_counter())
        eng._graph("body", T).replay()
        eng._graph("head", R).replay()
        logits = eng.buf["logits"][:plan.counts["read"]].cpu()
        t.append(time.perf_counter())
        probs = [softmax_t(z[:len(r["options"])].tolist(), tok.temperature[r["type"]]) for z, r in zip(logits, recs)]
        json.dumps({"p": probs})
        t.append(time.perf_counter())
        d0 = time.perf_counter()
        Decider(tok, eng).decide(req["state"], req["questions"])
        t.append(time.perf_counter())
        if i >= 5:
            for k, x, y in zip(times, t[:-1], t[1:]):
                times[k].append((y - x) * 1000)
            times["decide_total"][-1] = (t[-1] - d0) * 1000
    for k, v in times.items():
        print(f"{k:14s} {statistics.median(v):8.3f} ms")


if __name__ == "__main__":
    main()
