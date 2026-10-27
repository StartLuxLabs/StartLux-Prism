"""Precision research: compressed k-quant weights with FP16 activations/WMMA.

This is a different numerical profile from W4A8, not a bit-equivalent speedup.
Weights are dequantized only inside a program; no floating point model copy.
"""
import os

import triton
import triton.language as tl
from ..backends import math as F


@triton.jit
def _weight(qp, sp, mp, dp, dmp, n, k, N: tl.constexpr, K: tl.constexpr, Q4: tl.constexpr):
    b = k // 256
    if Q4:
        p = tl.load(qp + n * (K // 2) + (k // 32) * 16 + k % 16, n < N, 0).to(tl.int32)
        q = (p >> tl.where(k % 32 < 16, 0, 4)) & 15
        sc = tl.load(sp + (b * N + n) * 8 + (k % 256) // 32, n < N, 0).to(tl.float32)
        mn = tl.load(mp + (b * N + n) * 8 + (k % 256) // 32, n < N, 0).to(tl.float32)
        w = F.mul_rn(F.mul_rn(tl.load(dp + b * N + n, n < N, 0), sc), q.to(tl.float32))
        return F.add_rn(w, -F.mul_rn(tl.load(dmp + b * N + n, n < N, 0), mn))
    else:
        q = tl.load(qp + n * K + k, n < N, 0).to(tl.float32)
        sc = tl.load(sp + (b * N + n) * 16 + (k % 256) // 16, n < N, 0).to(tl.float32)
        return F.mul_rn(F.mul_rn(tl.load(dp + b * N + n, n < N, 0), sc), q)


@triton.jit
def _gemm(xp, qp, sp, mp, dp, dmp, yp, livep,
          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LD: tl.constexpr,
          Q4: tl.constexpr, EPI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    live = tl.load(livep)
    if tl.program_id(0) * BM >= live:
        return
    k = tl.arange(0, BK)
    y = tl.zeros((BM, BN), tl.float32)
    for b in range(K // BK):
        kk = b * BK + k
        x = tl.load(xp + r[:, None] * K + kk[None, :], r[:, None] < live, 0)
        w = _weight(qp, sp, mp, dp, dmp, n[:, None], kk[None, :], N, K, Q4).to(tl.float16)
        y = tl.dot(x, tl.trans(w), y)
    ptr = yp + r[:, None] * LD + n[None, :]
    mask = (r[:, None] < live) & (n[None, :] < N)
    if EPI:
        y = F.add_rn(tl.load(ptr, mask, 0), y)
    tl.store(ptr, y, mask)


def gemm(x, w, out, live, epi=0, cfg=None):
    if cfg is None:
        mode = os.environ.get("STARTLUX_PRISM_AMD_W4A16", "auto")
        if mode not in ("auto", "reference"):
            raise ValueError("STARTLUX_PRISM_AMD_W4A16 must be auto or reference")
        if mode == "auto":
            from .amd_w4a16_dispatch import pick
            chosen = pick(x, w, out, epi)
            if chosen is not None:
                from .amd_w4a16_meta import gemm as optimized
                return optimized(x, w, out, live, epi, tuple(chosen))
        cfg = (16, 32, 4, 1, 64)
    bm, bn, nw, ns, bk = (*cfg, 64) if len(cfg) == 4 else cfg
    assert bk in (64, 128, 256)
    _gemm[(triton.cdiv(x.shape[0], bm), triton.cdiv(w.n, bn))](
        x, w.q, w.sc, w.mn, w.d, w.dm, out, live, x.shape[0], w.n, w.k,
        out.stride(0), w.kind == 'q4k', epi, bm, bn, bk, num_warps=nw, num_stages=ns)


@triton.jit
def _embed(ids, qp, sp, mp, dp, dmp, hp, livep,
           N: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, Q4: tl.constexpr):
    r = tl.program_id(0)
    if r >= tl.load(livep):
        return
    n = tl.load(ids + r)
    if n < 0:
        return
    k = tl.arange(0, KP)
    # K is a multiple of 256. Clamp padded inputs to valid addresses before masking the store.
    w = _weight(qp, sp, mp, dp, dmp, n, tl.minimum(k, K - 1), N, K, Q4)
    tl.store(hp + r * K + k, w, k < K)


def embed(ids, w, h, live, rows):
    _embed[(rows,)](ids, w.q, w.sc, w.mn, w.d, w.dm, h, live,
                   w.n, w.k, triton.next_power_of_2(w.k), w.kind == 'q4k', num_warps=4)
