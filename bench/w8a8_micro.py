"""The W8A8 Triton GEMM (kernels/w8a8.py) on the model's projection shapes: exactness against fp32 math on the same
int8 inputs, and time against cuBLAS bf16 at the same M, N, K.

    python bench/w8a8_micro.py [--m 16,64,128,256,512,1024,2048,4096] [--configs]

--configs times every tile configuration in kernels/w8a8.py's search list for each shape and prints the best, for
tuning CONFIGS.
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev.kernels import w8a8 as KW  # noqa: E402
from gemm_micro import SHAPES, timed  # noqa: E402

def search_space(epi, large=False):
    """Tile configurations tried by --configs (bounded so a search compiles in minutes); --large: the tiles for
    hundreds to thousands of tokens."""
    out = []
    dims = ((64, 128), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4), (1, 2)) if large else \
        ((16, 32, 64, 128), (32, 64, 128), (64, 128, 256), (4, 8), (3, 4), (1, 2, 4))
    for bm, bn, bk, w, st, sk in itertools.product(*dims):
        if (bm * bk + bk * bn) * st > 96 * 1024 or (w == 4 and bk >= 128) or (sk > 1 and epi != 1):
            continue
        out.append((bm, bn, bk, 8, w, st, sk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="16,64,128,256,512,1024,2048,4096")
    ap.add_argument("--configs", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shapes", default="", help="comma-separated shape names (default: all)")
    ap.add_argument("--cold", type=int, default=0,
                    help="rotate through this many weight copies (> 96 MB in total) so weights stream from DRAM, as "
                         "in a forward pass, instead of sitting in L2")
    ap.add_argument("--epi", type=int, default=0, help="epilogue: 0 store bf16, 1 add into an fp32 residual")
    ap.add_argument("--large", action="store_true", help="search the large-tile space")
    a = ap.parse_args()
    dev = "cuda"
    for name, (N, K) in SHAPES.items():
        if a.shapes and name.split(" ")[0] not in a.shapes.split(","):
            continue
        w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        ws = torch.rand(N, 1, device=dev) * 1e-3
        copies = [w] + [w.clone() for _ in range(max(0, a.cold - 1))]
        wb = (w.float() * ws).to(torch.bfloat16)
        for M in [int(x) for x in a.m.split(",")]:
            x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
            xs = torch.rand(M, 1, device=dev) * 1e-2
            n = torch.tensor([M], dtype=torch.int32, device=dev)
            out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            KW.gemm(x, xs, w, ws, out, n, K)
            ref = (x.double() @ w.double().t()) * xs.double() * ws.double().t()
            err = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            res = torch.zeros(M, N, device=dev, dtype=torch.float32)
            dst = res if a.epi == 1 else out
            rot = [0]

            def call(cfg=None):
                wc = copies[rot[0] % len(copies)]
                rot[0] += 1
                KW.gemm(x, xs, wc, ws, dst, n, K, epi=a.epi, cfg=cfg)

            def int8_all(cfg=None):
                for _ in range(len(copies)):
                    call(cfg)
            t_int8 = timed(int8_all, a.iters) / len(copies)
            xb = (x.float() * xs).to(torch.bfloat16)
            yb = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            t_bf16 = timed(lambda: torch.mm(xb, wb.t(), out=yb), a.iters)
            gbps = N * K / t_int8 / 1e6
            line = (f"{name:20s} M={M:5d}  int8 {t_int8 * 1000:8.1f} us {2 * M * N * K / t_int8 / 1e9:6.1f} TOPS {gbps:6.0f} GB/s  "
                    f"bf16 {t_bf16 * 1000:8.1f} us {2 * M * N * K / t_bf16 / 1e9:6.1f} TFLOPS  "
                    f"speedup {t_bf16 / t_int8:4.2f}x  max rel err {err:.1e}")
            if a.configs:
                best = None
                for cfg in search_space(a.epi, a.large):
                    try:
                        t = timed(lambda: int8_all(cfg), 10) / len(copies)
                    except Exception:            # noqa: BLE001  out of shared memory / registers
                        continue
                    if best is None or t < best[0]:
                        best = (t, cfg)
                line += f"  best {best[0] * 1000:7.1f} us {best[1]}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
