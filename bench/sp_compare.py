"""Plain vs software-pipelined main loop of the CUDA W8A8 GEMM, config by config (tile schedule, best stages and
split), weights streamed from DRAM.

    python bench/sp_compare.py [--m 128,384,1024,4096]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism.kernels import cuda_ext as CX  # noqa: E402
from gemm_micro import SHAPES, timed  # noqa: E402

PAIRS = [(0, 8), (3, 9), (6, 10), (7, 11), (2, 12), (0, 13)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="128,384,1024,4096")
    ap.add_argument("--shapes", default="gdn_in,mlp_gate_up,mlp_down,gdn_out")
    a = ap.parse_args()
    dev = "cuda"
    CX.load()
    for name, (N, K) in SHAPES.items():
        key = name.split(" ")[0]
        if key not in a.shapes.split(","):
            continue
        epi = 1 if key in ("gdn_out", "attn_o", "mlp_down") else 0
        ws = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for _ in range(4)]
        sw = torch.rand(N, device=dev) * 1e-3
        for M in [int(x) for x in a.m.split(",")]:
            x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
            sa = torch.rand(M, device=dev) * 1e-2
            n = torch.tensor([M], dtype=torch.int32, device=dev)
            out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16 if epi == 0 else torch.float32)

            def best(cfg):
                r = None
                for st in (3, 4):
                    for sp in ((1, 2, 4) if epi == 1 else (1,)):
                        if not CX.fits(cfg, st) or K % (CX.CFGS[cfg][2] * sp):
                            continue
                        def f():
                            for w in ws:
                                CX.gemm(x, sa, w, sw, out, n, epi=epi, cfg=cfg, split=sp, stages=st)
                        t = timed(f, 10) / len(ws) * 1000
                        r = min(r or (t, st, sp), (t, st, sp))
                return r
            cells = []
            for plain, piped in PAIRS:
                p, q = best(plain), best(piped)
                cells.append(f"{plain}->{piped}: {p[0]:6.1f} -> {q[0]:6.1f} ({p[0] / q[0]:4.2f}x)")
            print(f"{key:12s} M={M:5d}  " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    main()
