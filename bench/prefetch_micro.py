"""Does prefetching a layer's projection weights into L2 pay?  The prefetch kernel's own time for the weights of one
linear-attention layer's out | gate|up | down projections (81 MB), and those three GEMMs at a short request's size
(M = 64) with their weights in DRAM vs prefetched.

    python bench/prefetch_micro.py [--m 64]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.kernels import cuda_ext as CX  # noqa: E402
from startlux_prism.kernels import w8a8 as KW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--delay-us", type=float, default=0.0, help="busy wait between prefetch and GEMMs")
    a = ap.parse_args()
    dev = "cuda"
    M = a.m
    CX.load()
    shapes = {"out": (2560, 4096, 1), "gate_up": (18432, 2560, 0), "down": (2560, 9216, 1)}
    W = {k: torch.randint(-127, 128, (n, kk), device=dev, dtype=torch.int8) for k, (n, kk, _) in shapes.items()}
    S = {k: torch.rand(n, device=dev) * 1e-3 for k, (n, _, _) in shapes.items()}
    X = {k: torch.randint(-127, 128, (M, kk), device=dev, dtype=torch.int8) for k, (_, kk, _) in shapes.items()}
    sa = torch.rand(M, device=dev) * 1e-2
    out = {k: torch.zeros(M, n, device=dev, dtype=torch.float32 if e else torch.bfloat16) for k, (n, _, e) in shapes.items()}
    slots = torch.zeros(4, M, 2560, dtype=torch.int32, device=dev)
    n_rows = torch.tensor([M], dtype=torch.int32, device=dev)
    flush = torch.empty(512 << 20, dtype=torch.int8, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    def gemms():
        for k, (n, kk, e) in shapes.items():
            KW.gemm(X[k], sa, W[k], S[k], out[k], n_rows, kk, epi=e, slots=slots)

    for ctas in (16, 32, 64, 128):
        res = []
        for rep in range(6):
            flush.fill_(rep)                              # evict everything from L2
            torch.cuda.synchronize()
            ev[0].record()
            CX.l2_prefetch(list(W.values()), ctas)
            if a.delay_us:
                torch.cuda._sleep(int(a.delay_us * 2500))           # ~cycles at ~2.5 GHz: the GDN kernels' time
            ev[1].record()
            gemms()
            ev[2].record()
            torch.cuda.synchronize()
            flush.fill_(rep + 1)
            torch.cuda.synchronize()
            ev[3].record()
            gemms()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            if rep:
                res.append((ev[0].elapsed_time(ev[1]) * 1e3, ev[1].elapsed_time(ev[2]) * 1e3,
                            ev[3].elapsed_time(end) * 1e3))
        pf, warm, cold = (sorted(x)[len(x) // 2] for x in zip(*res))
        print(f"M={M} ctas={ctas:4d} delay {a.delay_us:4.0f}: prefetch (+delay) {pf:7.1f} us | three GEMMs cold "
              f"{cold:6.1f} us, after prefetch {warm:6.1f} us", flush=True)


if __name__ == "__main__":
    main()
