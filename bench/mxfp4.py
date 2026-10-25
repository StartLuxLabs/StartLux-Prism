"""Benchmark the MXFP4 GEMM: Triton, and CUTLASS when STARTLUX_PRISM_MXFP4_CUTLASS=1.

    python bench/mxfp4.py                       # gate|up and down at M=1, 64, 1024
    python bench/mxfp4.py --m 1,256 --n 24576 --k 4096

Times only the GEMM (packing and scale repacking excluded), after a warm-up.  The weight and
activation nibbles are random and the scales are a fixed exponent, so the numbers are throughput,
not accuracy.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.kernels import mxfp4 as K  # noqa: E402


def time_gemm(fn, n=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def run(M, N, Kd, lib):
    torch.manual_seed(0)
    w = K.MX4(*[t.cuda() for t in K.pack(torch.randn(N, Kd) * 0.02)[:2]], N, Kd)
    wc = K.cutlass_prepare(w)
    q = torch.randint(0, 255, (M, Kd // 2), dtype=torch.uint8, device="cuda")
    s = torch.full((M, Kd // 32), 0x7a, dtype=torch.uint8, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    def triton():
        K.gemm(q, s, w, out, M, epi=0)

    def cutlass():
        K.gemm(q, s, wc, out, M, epi=0)

    dt_t = time_gemm(triton)
    row = f"{M:>5} {N:>6} {Kd:>5}  {dt_t * 1e3:8.3f}  {2 * M * N * Kd / dt_t / 1e12:6.0f}"
    if lib is not None:
        dt_c = time_gemm(cutlass)
        row += f"  {dt_c * 1e3:8.3f}  {2 * M * N * Kd / dt_c / 1e12:6.0f}  {dt_t / dt_c:5.2f}x"
    print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,64,1024", help="comma-separated row counts")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    lib = K._cutlass()
    shapes = ([(int(m), a.n, a.k) for m in a.m.split(",")] if a.n and a.k
              else [(int(m), n, k) for m in a.m.split(",") for n, k in ((24576, 4096), (4096, 12288))])
    print(f"GPU: {torch.cuda.get_device_name()}  CUTLASS: {'yes' if lib else 'no (Triton only)'}")
    header = "    M      N     K  Triton ms  TF/s"
    if lib:
        header += "  CUTLASS ms  TF/s   ratio"
    print(header)
    for M, N, Kd in shapes:
        run(M, N, Kd, lib)


if __name__ == "__main__":
    main()
