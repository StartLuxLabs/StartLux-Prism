"""Decode speed (engine/decode.py): tokens per second for 1 to 64 sequences at once, at a few context lengths.

    python bench/decode_speed.py --gguf MODEL.gguf --model-dir DIR [--precision int8|w4a8|bf16 --calib FILE]
        [--batches 1,2,4,8,16,32,64] [--contexts 512,4096] [--steps 64] [--profile] [--tp N --gpus a,b]

Every sequence is a prompt of the given length (prefilled, state kept), then --steps decode steps are queued as the
server queues them (each step launched before the previous one's tokens are read).  Reports ms per step and tokens/s;
--profile adds the kernel time per family for one step (torch profiler).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.engine.decode import Decoder, Row, Sampling  # noqa: E402
from startlux_prism.engine.model import Engine, Options  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="int8", choices=("int8", "w4a8", "bf16"))
    ap.add_argument("--calib", default="")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--contexts", default="512,4096")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--attn", default="split", help="decode attention kernels to time: split, tree or both (split,tree)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallelism over --gpus (int8; startlux_prism/tp.py)")
    ap.add_argument("--gpus", default="", help="with --tp: the GPUs (default: the visible ones)")
    a = ap.parse_args()
    if a.tp > 1:
        from startlux_prism import tp as TP
        TP.start(a.tp, a.gpus or None)
    batches = [int(x) for x in a.batches.split(",")]
    contexts = [int(x) for x in a.contexts.split(",")]
    n = max(batches)
    rows_each = max(contexts) + a.steps + 8
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, max_tokens=16384, prefix_cache=0,
                                              think_seqs=n, think_kv_rows=n * rows_each, tp=a.tp))
    for attn in a.attn.split(","):
        dec = Decoder(eng, a.gguf, max_seqs=n, sampling=Sampling(), attn=attn)
        dec.warm_up()
        print(f"{a.precision}{f', tp {a.tp}' if a.tp > 1 else ''}, {attn} attention: decode steps, {a.steps} per "
              f"measurement", flush=True)
        run(a, eng, dec, batches, contexts, rows_each, n)
        if eng.tp is not None:
            eng.tp.forget(dec)
        del dec
        torch.cuda.empty_cache()


def run(a, eng, dec, batches, contexts, rows_each, n):
    rng = np.random.default_rng(0)
    for ctx in contexts:
        # n prompts of ctx random tokens (ids below the special tokens), prefilled in batches that fit
        prompts = [rng.integers(1000, 100000, ctx).tolist() for _ in range(n)]
        seqs = []
        for i in range(0, n, max(1, eng.max_tokens // ctx)):
            part = list(range(i, min(n, i + max(1, eng.max_tokens // ctx))))
            keep = [(prompts[j][:-1], eng.think_slot0 + j, eng.think_kv0 + j * rows_each) for j in part]
            b = eng.launch(eng.prepare([], 0, keep=keep))
            eng.finish(b)
            for j in part:
                seqs.append(dict(slot=eng.think_slot0 + j, kv=eng.think_kv0 + j * rows_each, first=prompts[j][-1]))
        for B in batches:
            def rows(k):
                return [Row(s["slot"], s["kv"], ctx - 1 + k, ctx - 1 + k, 7 + i, k, i if k else -1 - s["first"])
                        for i, s in enumerate(seqs[:B])]
            for w in range(2):                       # warm-up
                dec.finish(dec.launch(rows(w), w % 2))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prev = None
            for k in range(a.steps):
                st = dec.launch(rows(2 + k), k % 2)
                if prev is not None:
                    dec.finish(prev)
                prev = st
            dec.finish(prev)
            dt = (time.perf_counter() - t0) / a.steps
            print(f"  context {ctx:6d}  batch {B:3d}: {dt * 1000:7.2f} ms per step, {B / dt:8.1f} tokens/s", flush=True)
            if a.profile and B in (1, max(batches)):
                profile(dec, rows(2 + a.steps))


def profile(dec, rows):
    """Kernel time of one step, run eagerly (the same kernels as its graph)."""
    from torch.profiler import ProfilerActivity, profile as tprof
    graphs, dec.graphs = dec.graphs, False
    dec.finish(dec.launch(rows, 0))
    with tprof(activities=[ProfilerActivity.CUDA]) as p:
        dec.finish(dec.launch(rows, 0))
        torch.cuda.synchronize()
    dec.graphs = graphs
    fam = {}
    for e in p.key_averages():
        name = e.key.lower()
        if e.device_time_total <= 0 or name.startswith(("memcpy", "memset")):
            continue
        k = ("output head" if "head_q8" in name or "head_quant" in name
             else "projections" if any(x in name for x in ("gemm", "w8a8", "w4a8", "wgmma", "gemv", "nvjet", "cutlass", "sm90"))
             else "attention" if "attn" in name or "attention" in name or "qkv_norm" in name
             else "linear attention" if "conv_step" in name or "gdn_step" in name
             else "norms, swiglu, quant" if any(x in name for x in ("norm", "swiglu", "rows_q8", "fold", "embed"))
             else "top-k, sampling" if "topk" in name or "sample" in name or "sort" in name or "radix" in name
             else "other")
        fam[k] = fam.get(k, 0.0) + e.device_time_total / 1000
    tot = sum(fam.values())
    print("    kernel time per step: " + ", ".join(f"{k} {v:.2f} ms ({v / tot:.0%})" for k, v in
                                                  sorted(fam.items(), key=lambda x: -x[1])), flush=True)


if __name__ == "__main__":
    main()
