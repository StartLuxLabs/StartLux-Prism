"""W4A16 research: load each block/group's metadata once before broadcasting.

The FP32 dequantization operations and FP16 weights match amd_w4a16; all
candidates must also match its full FP32 GEMM output before performance tests.
"""
import triton
import triton.language as tl
from ..backends import math as F


@triton.jit
def _gemm_meta(xp, qp, sp, mp, dp, dmp, yp, livep,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LD: tl.constexpr,
               Q4: tl.constexpr, EPI: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    live = tl.load(livep)
    if tl.program_id(0) * BM >= live:
        return
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for part in tl.range(0, K // BK, loop_unroll_factor=1, num_stages=1):
        block = part // (256 // BK)
        d = tl.load(dp + block * N + col, col < N, 0)
        if Q4:
            group = (part % (256 // BK)) * (BK // 32) + tl.arange(0, BK // 32)
            ptr = (block * N + col[:, None]) * 8 + group[None, :]
            sc = tl.load(sp + ptr, col[:, None] < N, 0).to(tl.float32)
            mn = tl.load(mp + ptr, col[:, None] < N, 0).to(tl.float32)
            sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 32, 32)), (BN, BK))
            mn = tl.reshape(tl.broadcast_to(mn[:, :, None], (BN, BK // 32, 32)), (BN, BK))
            dm = tl.load(dmp + block * N + col, col < N, 0)
            packed = tl.load(qp + col[:, None] * (K // 2) + part * (BK // 2)
                             + ((k // 32) * 16 + k % 16)[None, :], col[:, None] < N, 0).to(tl.int32)
            q = (packed >> tl.where(k[None, :] % 32 < 16, 0, 4)) & 15
            weight = F.mul_rn(F.mul_rn(d[:, None], sc), q.to(tl.float32))
            weight = F.add_rn(weight, -F.mul_rn(dm[:, None], mn))
        else:
            group = (part % (256 // BK)) * (BK // 16) + tl.arange(0, BK // 16)
            sc = tl.load(sp + (block * N + col[:, None]) * 16 + group[None, :],
                         col[:, None] < N, 0).to(tl.float32)
            sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN, BK // 16, 16)), (BN, BK))
            q = tl.load(qp + col[:, None] * K + part * BK + k[None, :], col[:, None] < N, 0).to(tl.float32)
            weight = F.mul_rn(F.mul_rn(d[:, None], sc), q)
        x = tl.load(xp + row[:, None] * K + part * BK + k[None, :], row[:, None] < live, 0)
        acc = tl.dot(x, tl.trans(weight.to(tl.float16)), acc)
    ptr = yp + row[:, None] * LD + col[None, :]
    mask = (row[:, None] < live) & (col[None, :] < N)
    if EPI:
        acc = F.add_rn(tl.load(ptr, mask, 0), acc)
    tl.store(ptr, acc, mask)


def gemm(x, w, out, live, epi=0, cfg=(32, 64, 4, 1, 64)):
    bm, bn, nw, ns, bk = cfg
    assert bk in (32, 64, 128, 256) and ns == 1
    return _gemm_meta[(triton.cdiv(x.shape[0], bm), triton.cdiv(w.n, bn))](
        x, w.q, w.sc, w.mn, w.d, w.dm, out, live, x.shape[0], w.n, w.k,
        out.stride(0), w.kind == 'q4k', epi, bm, bn, bk, num_warps=nw, num_stages=ns)
