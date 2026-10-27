"""Small-row Q4_K/Q6_K GEMV with exact integer reductions and ordered FP32 epilogues."""

import triton
import triton.language as tl
from ..backends import math as F


@triton.jit
def _gemv(
    a_ptr,
    as_ptr,
    q_ptr,
    sc_ptr,
    mn_ptr,
    d_ptr,
    dm_ptr,
    out_ptr,
    n_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    LDO: tl.constexpr,
    Q4: tl.constexpr,
    EPI: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    k = tl.arange(0, 256)
    y = tl.zeros((BN,), tl.float32)
    for b in range(K // 256):
        a = tl.load(a_ptr + r * K + b * 256 + k).to(tl.int32)
        if Q4:
            idx = b * 128 + (k // 32) * 16 + k % 16
            p = tl.load(q_ptr + n[:, None] * (K // 2) + idx[None, :], n[:, None] < N, 0)
            q = ((p >> tl.where(k[None, :] % 32 < 16, 0, 4)) & 15).to(tl.int32)
            sc = tl.load(sc_ptr + (b * N + n[:, None]) * 8 + k[None, :] // 32, n[:, None] < N, 0).to(tl.int32)
        else:
            q = tl.load(q_ptr + n[:, None] * K + b * 256 + k[None, :], n[:, None] < N, 0).to(tl.int32)
            sc = tl.load(sc_ptr + (b * N + n[:, None]) * 16 + k[None, :] // 16, n[:, None] < N, 0).to(tl.int32)
        acc = tl.sum(q * sc * a[None, :], 1)
        t = F.mul_rn(tl.load(d_ptr + b * N + n, n < N, 0), acc.to(tl.float32))
        if Q4:
            mn = tl.load(mn_ptr + (b * N + n[:, None]) * 8 + k[None, :] // 32, n[:, None] < N, 0).to(tl.int32)
            mins = tl.sum(mn * a[None, :], 1)
            t = F.fma(-tl.load(dm_ptr + b * N + n, n < N, 0), mins.to(tl.float32), t)
        y = F.fma(tl.load(as_ptr + b * M + r), t, y)
    p = out_ptr + r * LDO + n
    if EPI == 0:
        tl.store(p, y.to(tl.bfloat16), n < N)
    else:
        tl.store(p, F.add_rn(tl.load(p, n < N, 0), y), n < N)


def gemv(xq, a, xs, w, out, n_rows, epi=0):
    _gemv[(xq.shape[0], triton.cdiv(w.n, 8))](
        xq,
        a,
        w.q,
        w.sc,
        w.mn,
        w.d,
        w.dm,
        out,
        n_rows,
        xq.shape[0],
        w.n,
        w.k,
        out.stride(0),
        Q4=w.kind == "q4k",
        EPI=epi,
        BN=8,
        num_warps=4,
    )
