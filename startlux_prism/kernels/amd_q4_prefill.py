"""Experimental prefill: bounded inner-loop unrolling and exact compact Q4 minimum correction."""
import triton
import triton.language as tl
from ..backends import math as libdevice
from .w4a8 import _tile, _store


@triton.jit
def _q4_loop(a_ptr, as_ptr, xs_ptr, q_ptr, sc_ptr, mn_ptr, d_ptr, dm_ptr, out_ptr, n_ptr, M, N, K, ldo,
                BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, EPI: tl.constexpr, UNROLL: tl.constexpr):
    live = tl.load(n_ptr)
    pm, pn = _tile(tl.program_id(0), M, N, live, BM, BN, GROUP_M)
    if pm * BM >= live:
        return
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    mm, mn = rm < live, rn < N
    k32, k16 = tl.arange(0, 32), tl.arange(0, 16)
    NB = K // 256
    y = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(NB):
        acc = tl.zeros([BM, BN], dtype=tl.int32)
        for j in tl.range(0, 8, loop_unroll_factor=UNROLL):
            g = b * 8 + j
            a = tl.load(a_ptr + rm[:, None] * K + g * 32 + k32[None, :], mask=mm[:, None], other=0)
            p = tl.load(q_ptr + rn[:, None] * (K // 2) + g * 16 + k16[None, :], mask=mn[:, None], other=0)
            w = tl.reshape(tl.permute(tl.join((p & 15).to(tl.int8), (p >> 4).to(tl.int8)), (0, 2, 1)), (BN, 32))
            acc += tl.dot(a, tl.trans(w), out_dtype=tl.int32) * tl.load(sc_ptr + (b * N + rn) * 8 + j, mask=mn,
                                                                        other=0).to(tl.int32)[None, :]
        t = libdevice.mul_rn(tl.load(d_ptr + b * N + rn, mask=mn, other=0.0)[None, :], acc.to(tl.float32))
        # S = low + 64*high. All FP16 operands and integer FP32 partials are exact.
        g16 = tl.arange(0, 16)
        sums = tl.load(xs_ptr + (b * M + rm[:, None]) * 16 + g16[None, :], mm[:, None], 0).to(tl.float16)
        mins_w = tl.load(mn_ptr + (b * N + rn[:, None]) * 8 + g16[None, :] % 8, mn[:, None], 0).to(tl.float32)
        mins_w = (mins_w * tl.where(g16[None, :] < 8, 1., 64.)).to(tl.float16)
        mins = tl.dot(sums, tl.trans(mins_w)).to(tl.int32)
        t = libdevice.fma(-tl.load(dm_ptr + b * N + rn, mask=mn, other=0.0)[None, :], mins.to(tl.float32), t)
        y = libdevice.fma(tl.load(as_ptr + b * M + rm, mask=mm, other=0.0)[:, None], t, y)
    _store(y, out_ptr, rm, rn, mm, mn, ldo, EPI)


def gemm(xq, a, xs, w, out, n_rows, epi=0, cfg=(64,64,4,1,1)):
    assert w.kind == 'q4k'
    bm,bn,warps,stages,unroll=cfg
    _q4_loop[(triton.cdiv(xq.shape[0],bm)*triton.cdiv(w.n,bn),)](
        xq,a,xs,w.q,w.sc,w.mn,w.d,w.dm,out,n_rows,xq.shape[0],w.n,w.k,out.stride(0),
        BM=bm,BN=bn,GROUP_M=8,EPI=epi,UNROLL=unroll,num_warps=warps,num_stages=stages)
