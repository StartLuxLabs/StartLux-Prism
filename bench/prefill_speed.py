"""Prefill speed of one precision on the engine itself (no HTTP, no prefix cache): ms per forward pass and prompt tokens
per second, for one prompt of each length and for passes full of short prompts (the many-client case).

    python bench/prefill_speed.py --gguf MODEL.gguf --model-dir DIR --precision int8|bf16 [--calib FILE]
        [--lengths 256,1024,4096,8192,16384,32768] [--batches 32x256,64x128,8x1024] [--reps 10] [--tp N --gpus a,b]

Every prompt is fresh random tokens (no two passes share a prefix); each row is read at its last token, as a question
is.  The time is the median of --reps passes after two warm-up passes, from launch to logits on the host.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.engine.model import Engine, Options  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    ap.add_argument("--calib", default="")
    ap.add_argument("--lengths", default="256,1024,4096,8192,16384,32768")
    ap.add_argument("--batches", default="32x256,64x128,8x1024")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallelism over --gpus (int8; startlux_prism/tp.py)")
    ap.add_argument("--gpus", default="", help="with --tp: the GPUs (default: the visible ones)")
    a = ap.parse_args()
    if a.tp > 1:
        from startlux_prism import tp as TP
        TP.start(a.tp, a.gpus or None)
    lengths = [int(x) for x in a.lengths.split(",")]
    batches = [tuple(int(v) for v in b.split("x")) for b in a.batches.split(",")]
    top = max(lengths + [n * L for n, L in batches])
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, max_tokens=top, prefix_cache=0,
                                              tp=a.tp))
    eng.warm_up()
    rng = np.random.default_rng(0)

    def timed(make):
        out = []
        for r in range(a.reps + 2):
            groups = make()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.run(groups)
            if r >= 2:
                out.append((time.perf_counter() - t0) * 1000)
        return statistics.median(out)

    tag = a.precision + (f", tp {a.tp}" if a.tp > 1 else "")
    print(f"{tag}: one prompt per pass", flush=True)
    for L in lengths:
        ms = timed(lambda: [[rng.integers(1000, 100000, L).tolist()]])
        print(f"  {L:6d} tokens: {ms:8.2f} ms  {L / ms * 1000:9.0f} tokens/s", flush=True)
    print(f"{tag}: passes of short prompts (one request each)", flush=True)
    for n, L in batches:
        ms = timed(lambda: [[rng.integers(1000, 100000, L).tolist()] for _ in range(n)])
        print(f"  {n:3d} x {L:5d} tokens: {ms:8.2f} ms  {n * L / ms * 1000:9.0f} tokens/s  {n / ms * 1000:7.1f} "
              f"prompts/s", flush=True)


if __name__ == "__main__":
    main()
