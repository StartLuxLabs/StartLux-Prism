"""Small-M projections with the weights resident in L2 (one copy, repeated) vs streamed from DRAM (rotating copies):
how much of a short request's GEMM time is the DRAM stream, and how much the kernels themselves.

    python bench/l2_resident_micro.py [--m 64]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism.kernels import cuda_ext as CX  # noqa: E402
from gemm_micro import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    a = ap.parse_args()
    dev, M = "cuda", a.m
    CX.load()
    for name, N, K, epi in (("gdn_in", 12352, 2560, 0), ("gate_up", 18432, 2560, 0), ("down", 2560, 9216, 1),
                            ("out", 2560, 4096, 1)):
        x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        sa = torch.rand(M, device=dev) * 1e-2
        sw = torch.rand(N, device=dev) * 1e-3
        n = torch.tensor([M], dtype=torch.int32, device=dev)
        out = torch.zeros(M, N, device=dev, dtype=torch.float32 if epi else torch.bfloat16)
        slots = torch.zeros(4, M, N, dtype=torch.int32, device=dev)
        ws = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for _ in range(4)]
        for cfg, st, sp in ((2, 3, 1), (2, 3, 4), (5, 4, 1), (12, 3, 1), (0, 3, 1), (8, 3, 1), (1, 3, 1)):
            if K % (CX.CFGS[cfg][2] * sp) or (sp > 1 and not epi):
                continue
            mode = dict(mode="defer", slots=slots) if sp > 1 else {}

            def run(w):
                CX.gemm(x, sa, w, sw, out, n, epi=epi, cfg=cfg, split=sp, stages=st, **mode)
            hot = timed(lambda: run(ws[0]), 20) * 1000
            def rot():
                for w in ws:
                    run(w)
            cold = timed(rot, 5) / len(ws) * 1000
            mb = N * K / 1e6
            print(f"{name:8s} M={M} cfg {cfg:2d} st {st} split {sp}: L2-resident {hot:6.1f} us ({mb / hot:4.2f} TB/s)"
                  f"  from DRAM {cold:6.1f} us ({mb / cold:4.2f} TB/s)", flush=True)


if __name__ == "__main__":
    main()
