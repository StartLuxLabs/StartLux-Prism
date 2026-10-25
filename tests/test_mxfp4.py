"""MXFP4 GEMM: packing, activation quantization, parity against the dequantized product, batch invariance, throughput.

    python tests/test_mxfp4.py [--quick]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.kernels import mxfp4 as K  # noqa: E402


def log(*a):
    print(*a, flush=True)


def test_pack():
    torch.manual_seed(0)
    for n, k in ((64, 128), (128, 4096)):
        w = torch.randn(n, k) * 0.02
        for mode in ("max", "mse"):
            m = K.pack(w, mode)
            assert m.q.dtype == torch.uint8 and m.q.shape == (n, k // 2)
            assert m.s.shape == (n, k // 32)
            dq = K.dequant_packed(m.q, m.s)
            rel = (dq - w).norm() / w.norm()
            log(f"pack {mode:3s} {n}x{k}: rel {rel:.5f}")
            assert rel < 0.2
    # saturated values survive
    w = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 3 + [0.0, 0.5]])
    m = K.pack(w)
    dq = K.dequant_packed(m.q, m.s)
    log("exact grid max err:", (dq - w).abs().max().item())
    assert (dq - w).abs().max().item() < 1e-6


def test_act():
    torch.manual_seed(1)
    for n, k in ((7, 256), (33, 512), (128, 4096)):
        x = torch.randn(n, k, device="cuda") * 0.5
        q = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
        s = torch.empty(n, k // 32, dtype=torch.uint8, device="cuda")
        K.quantize_act(x, q, s)
        dq = K.dequant_packed(q.cpu(), s.cpu())
        rel = (dq - x.cpu().float()).norm() / x.cpu().float().norm()
        log(f"act {n}x{k}: rel {rel:.5f}")
        assert rel < 0.2


def ref_gemm(a_q, a_s, w, M, N):
    a = K.dequant_packed(a_q.cpu(), a_s.cpu())[:M, :w.k]
    b = K.dequant_packed(w.q, w.s)
    return a @ b.T


def test_gemm(quick=False):
    torch.manual_seed(2)
    shapes = [(32, 512, 256), (257, 1024, 512)] if quick else [(32, 512, 256), (257, 1024, 512), (1024, 4096, 4096)]
    for M, N, Kd in shapes:
        w = torch.randn(N, Kd) * 0.02
        wm_cpu = K.pack(w)
        x = torch.randn(M, Kd, device="cuda") * 1.0
        q = torch.empty(M, Kd // 2, dtype=torch.uint8, device="cuda")
        s = torch.empty(M, Kd // 32, dtype=torch.uint8, device="cuda")
        K.quantize_act(x, q, s)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        wm = K.MX4(wm_cpu.q.cuda(), wm_cpu.s.cuda(), N, Kd)
        K.gemm(q, s, wm, out, M, epi=0)
        ref = ref_gemm(q, s, wm_cpu, M, N)
        got = out.float().cpu()
        rel = (got - ref).norm() / ref.norm()
        log(f"gemm {M}x{N}x{Kd}: rel {rel:.5f}")
        assert rel < 2e-3, rel

        # epi=1 adds into the fp32 residual
        h = torch.randn(M, N, device="cuda", dtype=torch.float32)
        h0 = h.clone()
        K.gemm(q, s, wm, h, M, epi=1)
        rel2 = (h.cpu() - (h0.cpu() + ref)).norm() / (h0.cpu() + ref).norm()
        log(f"gemm epi=1 {M}x{N}x{Kd}: rel {rel2:.5f}")
        assert rel2 < 2e-3, rel2


def test_batch_invariance():
    torch.manual_seed(3)
    M, N, Kd = 64, 1024, 512
    w = K.MX4(*[t.cuda() for t in K.pack(torch.randn(N, Kd) * 0.02)[:2]], N, Kd)
    x = torch.randn(M, Kd, device="cuda")
    q = torch.empty(M, Kd // 2, dtype=torch.uint8, device="cuda")
    s = torch.empty(M, Kd // 32, dtype=torch.uint8, device="cuda")
    K.quantize_act(x, q, s)
    out64 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    K.gemm(q, s, w, out64, M, epi=0)
    out32 = torch.empty(32, N, dtype=torch.bfloat16, device="cuda")
    K.gemm(q[:32], s[:32], w, out32, 32, epi=0)
    same = torch.equal(out64[:32], out32)
    log("batch invariance (64 vs 32 rows):", "OK" if same else "DIFF")
    assert same


def test_cutlass():
    if not K.cutlass_enabled():
        log("cutlass: not enabled or .so missing, skip")
        return
    torch.manual_seed(4)
    M, N, Kd = 64, 4096, 4096
    wm = K.pack(torch.randn(N, Kd) * 0.02)
    w = K.MX4(wm.q.cuda(), wm.s.cuda(), N, Kd)
    wc = K.cutlass_prepare(w)
    x = torch.randn(M, Kd, device="cuda")
    q = torch.empty(M, Kd // 2, dtype=torch.uint8, device="cuda")
    s = torch.empty(M, Kd // 32, dtype=torch.uint8, device="cuda")
    K.quantize_act(x, q, s)
    lib = K._CUTLASS
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    K.gemm(q, s, wc, out, M, epi=0)
    K._CUTLASS = False
    try:
        ref = torch.empty_like(out)
        K.gemm(q, s, w, ref, M, epi=0)
    finally:
        K._CUTLASS = lib
    same = torch.equal(out, ref)
    log("cutlass vs triton:", "OK" if same else "DIFF")
    assert same


def bench(quick=False):
    shapes = [(256, 12288, 4096), (1024, 12288, 4096), (1024, 4096, 12288)] if not quick else [(256, 4096, 4096)]
    for M, N, Kd in shapes:
        w = K.MX4(*[t.cuda() for t in K.pack(torch.randn(N, Kd) * 0.02)[:2]], N, Kd)
        q = torch.randint(0, 255, (M, Kd // 2), dtype=torch.uint8, device="cuda")
        s = torch.full((M, Kd // 32), 0x7a, dtype=torch.uint8, device="cuda")
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            K.gemm(q, s, w, out, M, epi=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            K.gemm(q, s, w, out, M, epi=0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        log(f"bench {M}x{N}x{Kd}: {dt*1e3:.2f} ms  {2*M*N*Kd/dt/1e12:.0f} TF/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    test_pack()
    test_act()
    test_gemm(a.quick)
    test_batch_invariance()
    test_cutlass()
    bench(a.quick)
    log("all OK")
