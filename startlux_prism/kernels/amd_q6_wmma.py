"""Experimental exact Q6 WMMA using two signed INT8 products per super-block."""
import triton
import triton.language as tl
from ..backends import math as F


@triton.jit
def _q6_scaled(a_ptr, as_ptr, q_ptr, sc_ptr, d_ptr, out_ptr, n_ptr, M, N, K, LDO,
               EPI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    live = tl.load(n_ptr)
    if tl.program_id(0) * BM >= live:
        return
    k = tl.arange(0, 256)
    y = tl.zeros((BM, BN), tl.float32)
    for b in range(K // 256):
        a = tl.load(a_ptr + r[:, None] * K + b * 256 + k[None, :], r[:, None] < live, 0)
        q = tl.load(q_ptr + n[:, None] * K + b * 256 + k[None, :], n[:, None] < N, 0).to(tl.int32)
        sc = tl.load(sc_ptr + (b * N + n[:, None]) * 16 + k[None, :] // 16,
                     n[:, None] < N, 0).to(tl.int32)
        scaled = q * sc
        lo = (scaled & 127).to(tl.int8)
        hi = (scaled >> 7).to(tl.int8)
        acc = tl.dot(a, tl.trans(lo), out_dtype=tl.int32) + 128 * tl.dot(a, tl.trans(hi), out_dtype=tl.int32)
        t = F.mul_rn(tl.load(d_ptr + b * N + n, n < N, 0)[None, :], acc.to(tl.float32))
        y = F.fma(tl.load(as_ptr + b * M + r, r < live, 0)[:, None], t, y)
    p = out_ptr + r[:, None] * LDO + n[None, :]
    mask = (r[:, None] < live) & (n[None, :] < N)
    if EPI == 0:
        tl.store(p, y.to(tl.bfloat16), mask)
    else:
        tl.store(p, F.add_rn(tl.load(p, mask, 0), y), mask)


def gemm(xq, a, xs, w, out, n_rows, epi=0, cfg=(16, 32, 4, 1)):
    assert w.kind == 'q6k'
    bm, bn, warps, stages = cfg
    _q6_scaled[(triton.cdiv(xq.shape[0], bm), triton.cdiv(w.n, bn))](
        xq, a, w.q, w.sc, w.d, out, n_rows, xq.shape[0], w.n, w.k, out.stride(0),
        epi, bm, bn, num_warps=warps, num_stages=stages)
