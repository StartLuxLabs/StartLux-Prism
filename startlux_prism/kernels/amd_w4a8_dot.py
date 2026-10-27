"""gfx1201 small-row W4A8 with packed signed INT8 dot4 and shared weight tiles.

Integer dot products are reduced within each quantization group before applying
its integer scale. Q4 min correction uses the producer's exact activation sums.
FP32 super-block updates and output rounding match w4a8.gemm_ref.
"""
import torch
import triton
import triton.language as tl
from ..backends import math as F


@triton.jit
def _dot4(a, b):
    # IU8 uses neg_lo to select signed byte interpretation, not arithmetic negation.
    # Q4 bytes are 0..15; Q6 and activation bytes require signed interpretation.
    return tl.inline_asm_elementwise(
        "v_dot4_i32_iu8 $0, $1, $2, 0 neg_lo:[1,1,0];",
        constraints="=v,v,v", args=[a, b], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _gemv_dot(a_ptr, as_ptr, xs_ptr, q_ptr, sc_ptr, mn_ptr, d_ptr, dm_ptr, out_ptr, n_ptr,
              M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, LDO: tl.constexpr,
              Q4: tl.constexpr, EPI: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    live = tl.load(n_ptr)
    if tl.program_id(0) * BM >= live:
        return
    k = tl.arange(0, 64)
    y = tl.zeros((BM, BN), tl.float32)
    for b in range(K // 256):
        a = tl.load(a_ptr + r[:, None] * (K // 4) + b * 64 + k[None, :], r[:, None] < live, 0)
        if Q4:
            idx = b * 32 + (k // 8) * 4 + k % 4
            p = tl.load(q_ptr + n[:, None] * (K // 8) + idx[None, :], n[:, None] < N, 0)
            q = (p >> tl.where(k[None, :] % 8 < 4, 0, 4)) & 0x0f0f0f0f
            products = _dot4(a[:, None, :], q[None, :, :])
            grouped = tl.sum(tl.reshape(products, (BM, BN, 8, 8)), 3)
            g = tl.arange(0, 8)
            scale = tl.load(sc_ptr + (b * N + n[:, None]) * 8 + g[None, :], n[:, None] < N, 0).to(tl.int32)
            acc = tl.sum(grouped * scale[None, :, :], 2)
        else:
            q = tl.load(q_ptr + n[:, None] * (K // 4) + b * 64 + k[None, :], n[:, None] < N, 0)
            products = _dot4(a[:, None, :], q[None, :, :])
            grouped = tl.sum(tl.reshape(products, (BM, BN, 16, 4)), 3)
            g = tl.arange(0, 16)
            scale = tl.load(sc_ptr + (b * N + n[:, None]) * 16 + g[None, :], n[:, None] < N, 0).to(tl.int32)
            acc = tl.sum(grouped * scale[None, :, :], 2)
        t = F.mul_rn(tl.load(d_ptr + b * N + n, n < N, 0)[None, :], acc.to(tl.float32))
        if Q4:
            g = tl.arange(0, 8)
            ptr = xs_ptr + ((b * M + r[:, None]) * 16) + g[None, :]
            sums = tl.load(ptr, r[:, None] < live, 0).to(tl.int32) + 64 * tl.load(ptr + 8, r[:, None] < live, 0).to(tl.int32)
            mn = tl.load(mn_ptr + (b * N + n[:, None]) * 8 + g[None, :], n[:, None] < N, 0).to(tl.int32)
            mins = tl.sum(sums[:, None, :] * mn[None, :, :], 2)
            t = F.fma(-tl.load(dm_ptr + b * N + n, n < N, 0)[None, :], mins.to(tl.float32), t)
        y = F.fma(tl.load(as_ptr + b * M + r, r < live, 0)[:, None], t, y)
    p = out_ptr + r[:, None] * LDO + n[None, :]
    mask = (r[:, None] < live) & (n[None, :] < N)
    if EPI == 0:
        tl.store(p, y.to(tl.bfloat16), mask)
    else:
        tl.store(p, F.add_rn(tl.load(p, mask, 0), y), mask)


def gemv(xq, a, xs, w, out, n_rows, epi=0, cfg=(1, 8, 4)):
    bm, bn, warps = cfg
    _gemv_dot[(triton.cdiv(xq.shape[0], bm), triton.cdiv(w.n, bn))](
        xq.view(torch.int32), a, xs, w.q.view(torch.int32), w.sc, w.mn, w.d, w.dm, out, n_rows,
        xq.shape[0], w.n, w.k, out.stride(0), w.kind == 'q4k', epi, bm, bn, num_warps=warps)
