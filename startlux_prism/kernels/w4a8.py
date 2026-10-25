"""W4A8 projections on a Q4_K_M file's own weights: llama.cpp k-quant values as they are (no requantization), int8
activations with one scale per 256 inputs, int8 tensor cores.

Weights, repacked at load (repack):
  Q4_K  w = d * sc * q - dmin * m: q 4-bit (0..15), per 32 inputs a 6-bit scale sc and min m, per 256 fp16 d and dmin.
        q packed two per byte, per 32 inputs 16 bytes, byte t = q[t] | q[t + 16] << 4: one 32-bit word holds a thread's
        k = 4i..4i+3 (low nibbles) and 4i+16..4i+19 (high nibbles) of an m16n8k32 B fragment, two masks unpack it.
  Q6_K  w = d * sc * q: q 6-bit (-32..31), per 16 inputs an int8 scale sc, per 256 fp16 d; kept as int8 q.
The scales (sc, m, d, dmin) are stored super-block-major, so the GEMM's tiles copy them as whole lines.
Activations (kernels/quant.py, GROUP=256): xq int8 with one fp32 scale a per 256 inputs, and for Q4_K's mins the
sum S of xq over each 32 (|S| <= 32 * 127), split into int8 bytes S & 63 and S >> 6 (quantize_ref); a and the split
sums super-block-major too.

Arithmetic (one definition for every kernel, tile and batch: the same bits wherever an output element is computed):
per 256-input super-block b, in increasing b,
  Q4_K  acc = sum over its 8 sub-blocks g of sc[n, g] * (xq . q)_g          int32, exact
        mins = sum over g of m[n, g] * sum(xq)_g                            int32, exact
        t = d[n, b] * float(acc); t = fma(-dmin[n, b], float(mins), t)      fp32, each rounded once
        y = fma(a[m, b], t, y)
  Q6_K  acc = sum over its 16 sub-blocks h of sc[n, h] * (xq . q)_h         int32, exact
        y = fma(a[m, b], d[n, b] * float(acc), y)
then bf16(y), or out + y for the fp32 residual (epi 1).  Sub-block scales are integers, so inside a super-block
everything is integer arithmetic; one floating-point step per 256 inputs.  (Activations per 32 inputs, as llama.cpp's
MMQ, are about twice as close to the unquantized product but need a floating-point step per 32 - tests/quant_sim.py.)
"""
from __future__ import annotations

import functools
import json
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from ..backends import math as libdevice

from .. import gguf_weights as GW
from ..gguf import GGML_Q4_K, GGML_Q6_K

QK_K = 256


class W4(NamedTuple):
    """A repacked k-quant matrix [n, k]: the values row-major, the scales super-block-major (a super-block's scales of
    consecutive rows side by side: the GEMM's tiles read them as whole lines)."""
    kind: str               # "q4k" or "q6k"
    q: torch.Tensor         # q4k: uint8 [n, k / 2] packed; q6k: int8 [n, k]
    sc: torch.Tensor        # q4k: uint8 [k / 256, n, 8]; q6k: int8 [k / 256, n, 16]
    mn: torch.Tensor        # q4k: uint8 [k / 256, n, 8] (mins); q6k: None
    d: torch.Tensor         # fp32 [k / 256, n]
    dm: torch.Tensor        # q4k: fp32 [k / 256, n] (dmin); q6k: None
    n: int
    k: int

    @property
    def shape(self):
        return (self.n, self.k)

    def rows(self, r0, r1):
        """Rows r0:r1 (a slice of a concatenated projection, e.g. the last layer's queries): the values a view, the
        scales copied (the GEMM reads them contiguous)."""
        cut = lambda t: None if t is None else t[:, r0:r1].contiguous()     # noqa: E731
        return W4(self.kind, self.q[r0:r1], cut(self.sc), cut(self.mn), cut(self.d), cut(self.dm), r1 - r0, self.k)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.sc, self.mn, self.d, self.dm) if t is not None)


def repack(kq: GW.KQuant) -> W4:
    """A KQuant matrix (on the GPU) -> W4."""
    raw = kq.raw
    n, nb = raw.shape[0], raw.shape[1]
    if kq.type == GGML_Q4_K:
        d, dm = GW._f16(raw[..., 0:2]).view(n, nb), GW._f16(raw[..., 2:4]).view(n, nb)
        sc, mn = GW._scale_min_k4(raw[..., 4:16].reshape(-1, 12).to(torch.int32))
        q = raw[..., 16:144].reshape(n, nb, 4, 32).to(torch.int32)
        q = torch.stack([q & 0xF, q >> 4], 3).reshape(n, nb * 8, 32)          # [n, sub-block, 32] in k order
        packed = (q[..., :16] | (q[..., 16:] << 4)).to(torch.uint8).reshape(n, kq.k // 2)
        sbm = lambda t: t.to(torch.uint8).reshape(n, nb, 8).transpose(0, 1).contiguous()   # noqa: E731
        return W4("q4k", packed.contiguous(), sbm(sc), sbm(mn), d.t().contiguous(), dm.t().contiguous(), n, kq.k)
    if kq.type == GGML_Q6_K:
        b = raw.reshape(-1, 210)
        ql, qh = b[:, 0:128].to(torch.int32), b[:, 128:192].to(torch.int32)
        parts = []
        for h in range(2):
            lo, hi, top = ql[:, 64 * h:64 * h + 32], ql[:, 64 * h + 32:64 * h + 64], qh[:, 32 * h:32 * h + 32]
            parts += [(lo & 0xF) | ((top & 3) << 4), (hi & 0xF) | (((top >> 2) & 3) << 4),
                      (lo >> 4) | (((top >> 4) & 3) << 4), (hi >> 4) | (((top >> 6) & 3) << 4)]
        q = (torch.cat(parts, 1) - 32).to(torch.int8).reshape(n, kq.k)
        sc = b[:, 192:208].contiguous().view(torch.int8).reshape(n, nb, 16).transpose(0, 1)
        d = GW._f16(b[:, 208:210]).view(n, nb)
        return W4("q6k", q.contiguous(), sc.contiguous(), None, d.t().contiguous(), None, n, kq.k)
    raise ValueError(f"k-quant type {kq.type}: Q4_K and Q6_K are supported")


def concat(ws):
    """Row-concatenation of W4 matrices of one kind (a fused projection, e.g. qkv|z|b|a)."""
    kinds = {w.kind for w in ws}
    if len(kinds) != 1:
        raise ValueError(f"cannot fuse k-quant kinds {kinds}")
    cat = lambda f: None if getattr(ws[0], f) is None else torch.cat([getattr(w, f) for w in ws],      # noqa: E731
                                                                      0 if f == "q" else 1)
    return W4(ws[0].kind, cat("q"), cat("sc"), cat("mn"), cat("d"), cat("dm"), sum(w.n for w in ws), ws[0].k)


def segments(kqs, device):
    """The KQuant matrices of a fused projection (e.g. qkv|z|b|a) -> [(W4, first output column)]: neighbours of one
    kind fused into one matrix (a Q4_K_M file mixes Q4_K and Q6_K inside some, e.g. a Q6_K v under Q4_K q and k)."""
    out, off = [], 0
    for kq in kqs:
        if not isinstance(kq, GW.KQuant):
            raise ValueError(f"the W4A8 path needs k-quant projections, not {type(kq).__name__}")
        w = repack(kq.to(device))
        if out and out[-1][0].kind == w.kind:
            out[-1] = (concat([out[-1][0], w]), out[-1][1])
        else:
            out.append((w, off))
        off += w.n
    return out


def seg_rows(segs, r0, r1):
    """Output columns r0:r1 of a fused projection, as segments with first columns counted from r0."""
    out = []
    for w, off in segs:
        a, b = max(r0, off), min(r1, off + w.n)
        if a < b:
            out.append((w.rows(a - off, b - off), a - r0))
    return out


def seg_width(segs):
    return max(off + w.n for w, off in segs)


def gemm_segs(xq, a, xs, segs, out, n_rows, epi=0, pdl=0):
    """A fused projection: one GEMM per segment, each into its columns of out."""
    for w, off in segs:
        gemm(xq, a, xs, w, out[:, off:off + w.n], n_rows, epi, pdl=pdl)


# ---------------------------------------------------------------------------------------------------------- reference
def quantize_ref(x, group=QK_K):
    """fp32 [m, k] -> (xq int8 [m, k], a fp32 [k / group, m], xs int8 [k / 256, m, 16]) as kernels/quant.py writes
    them: a the scale of each 256 inputs, xs per 256 inputs the sums S of the eight groups of 32 as eight S & 63 then
    eight S >> 6 (both super-block-major, as the GEMM reads them)."""
    m, k = x.shape
    g = x.reshape(m, k // group, group)
    amax = torch.clamp(g.abs().amax(-1), min=1e-10)
    a = amax / torch.full_like(amax, 127.0)          # a true division (torch turns "/ 127.0" into "* (1 / 127)")
    xq = torch.round(g / a[..., None]).to(torch.int8).reshape(m, k)
    s = xq.reshape(m, k // 32, 32).to(torch.int32).sum(-1, dtype=torch.int32).reshape(m, k // QK_K, 8)
    xs = torch.cat([s & 63, s >> 6], -1).to(torch.int8)
    return xq, a.t().contiguous(), xs.transpose(0, 1).contiguous()


def group_sums(xs):
    """xs int8 [k / 256, m, 16] -> the sums of each 32 activations, int32 [m, k / 32]."""
    p = xs.transpose(0, 1).reshape(xs.shape[1], -1, 2, 8).to(torch.int32)
    return (p[:, :, 0] + 64 * p[:, :, 1]).reshape(xs.shape[1], -1)


def fma32(a, b, c):
    """fma(a, b, c) of fp32 tensors with one rounding: a * b is exact in float64, the sum is rounded to odd in float64
    (an exact TwoSum error decides), and a value rounded to odd with 53 bits rounds to 24 as the exact one would."""
    p, c = a.double() * b.double(), c.double()
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    even = (s.view(torch.int64) & 1) == 0
    inf = torch.full_like(s, float("inf"))
    away = torch.nextafter(s, torch.where(err > 0, inf, -inf))         # toward the exact sum
    return torch.where((err != 0) & even, away, s).float()


def gemm_ref(xq, a, xs, w: W4):
    """The arithmetic above, in fp32 with the same roundings (integer parts exact in float64): fp32 [m, n]."""
    m, k = xq.shape
    x64 = xq.to(torch.float64)
    sums = group_sums(xs)
    y = torch.zeros(m, w.n, dtype=torch.float32, device=xq.device)
    for b in range(k // QK_K):
        if w.kind == "q4k":
            q = w.q[:, b * 128:(b + 1) * 128].reshape(w.n, 8, 16).to(torch.int32)
            q = torch.cat([q & 0xF, q >> 4], 2).to(torch.float64)                 # [n, 8, 32]
            xb = x64[:, b * 256:(b + 1) * 256].reshape(m, 8, 32)
            p = torch.einsum("mgk,ngk->mng", xb, q)                               # exact integers
            acc = (p * w.sc[b].to(torch.float64)[None]).sum(-1)
            mins = sums[:, b * 8:(b + 1) * 8].to(torch.float64) @ w.mn[b].to(torch.float64).T
            t = fma32(-w.dm[b][None].expand(m, -1), mins.to(torch.float32), w.d[b][None] * acc.to(torch.float32))
        else:
            q = w.q[:, b * 256:(b + 1) * 256].reshape(w.n, 16, 16).to(torch.float64)
            xb = x64[:, b * 256:(b + 1) * 256].reshape(m, 16, 16)
            p = torch.einsum("mhk,nhk->mnh", xb, q)
            acc = (p * w.sc[b].to(torch.float64)[None]).sum(-1)
            t = w.d[b][None] * acc.to(torch.float32)
        y = fma32(a[b][:, None].expand(m, w.n), t, y)
    return y


# ------------------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _tile(pid, M, N, live, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    num_m, num_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    width = GROUP_M * num_n
    gid = pid // width
    first = gid * GROUP_M
    size = tl.minimum(num_m - first, GROUP_M)
    pm = first + (pid % width) % size
    pn = (pid % width) // size
    return pm, pn


@triton.jit
def _store(y, out_ptr, rm, rn, mm, mn, ldo, EPI: tl.constexpr):
    ptr = out_ptr + rm[:, None] * ldo + rn[None, :]
    msk = mm[:, None] & mn[None, :]
    if EPI == 0:
        tl.store(ptr, y.to(tl.bfloat16), mask=msk)
    else:
        tl.store(ptr, libdevice.add_rn(tl.load(ptr, mask=msk, other=0.0), y), mask=msk)


@triton.jit
def _q4k_kernel(a_ptr, as_ptr, xs_ptr, q_ptr, sc_ptr, mn_ptr, d_ptr, dm_ptr, out_ptr, n_ptr, M, N, K, ldo,
                BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, EPI: tl.constexpr):
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
        mins = tl.zeros([BM, BN], dtype=tl.int32)
        for j in tl.static_range(8):
            g = b * 8 + j
            a = tl.load(a_ptr + rm[:, None] * K + g * 32 + k32[None, :], mask=mm[:, None], other=0)
            p = tl.load(q_ptr + rn[:, None] * (K // 2) + g * 16 + k16[None, :], mask=mn[:, None], other=0)
            w = tl.reshape(tl.permute(tl.join((p & 15).to(tl.int8), (p >> 4).to(tl.int8)), (0, 2, 1)), (BN, 32))
            acc += tl.dot(a, tl.trans(w), out_dtype=tl.int32) * tl.load(sc_ptr + (b * N + rn) * 8 + j, mask=mn,
                                                                        other=0).to(tl.int32)[None, :]
            xsum = tl.load(xs_ptr + (b * M + rm) * 16 + j, mask=mm, other=0).to(tl.int32) + 64 * tl.load(
                xs_ptr + (b * M + rm) * 16 + 8 + j, mask=mm, other=0).to(tl.int32)
            mins += xsum[:, None] * tl.load(mn_ptr + (b * N + rn) * 8 + j, mask=mn, other=0).to(tl.int32)[None, :]
        t = libdevice.mul_rn(tl.load(d_ptr + b * N + rn, mask=mn, other=0.0)[None, :], acc.to(tl.float32))
        t = libdevice.fma(-tl.load(dm_ptr + b * N + rn, mask=mn, other=0.0)[None, :], mins.to(tl.float32), t)
        y = libdevice.fma(tl.load(as_ptr + b * M + rm, mask=mm, other=0.0)[:, None], t, y)
    _store(y, out_ptr, rm, rn, mm, mn, ldo, EPI)


@triton.jit
def _q6k_kernel(a_ptr, as_ptr, q_ptr, sc_ptr, d_ptr, out_ptr, n_ptr, M, N, K, ldo,
                BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, EPI: tl.constexpr):
    live = tl.load(n_ptr)
    pm, pn = _tile(tl.program_id(0), M, N, live, BM, BN, GROUP_M)
    if pm * BM >= live:
        return
    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)
    rn = pn * BN + tl.arange(0, BN)
    mm, mn = rm < live, rn < N
    k32 = tl.arange(0, 32)
    first = k32 < 16
    NB = K // 256
    y = tl.zeros([BM, BN], dtype=tl.float32)
    for b in range(NB):
        acc = tl.zeros([BM, BN], dtype=tl.int32)
        for j in tl.static_range(8):                    # 32 inputs: two 16-input sub-blocks, each its own scale
            g = b * 8 + j
            a = tl.load(a_ptr + rm[:, None] * K + g * 32 + k32[None, :], mask=mm[:, None], other=0)
            w = tl.trans(tl.load(q_ptr + rn[:, None] * K + g * 32 + k32[None, :], mask=mn[:, None], other=0))
            lo = tl.dot(tl.where(first[None, :], a, 0), w, out_dtype=tl.int32)
            hi = tl.dot(tl.where(first[None, :], 0, a), w, out_dtype=tl.int32)
            acc += lo * tl.load(sc_ptr + (b * N + rn) * 16 + 2 * j, mask=mn, other=0).to(tl.int32)[None, :]
            acc += hi * tl.load(sc_ptr + (b * N + rn) * 16 + 2 * j + 1, mask=mn, other=0).to(tl.int32)[None, :]
        t = libdevice.mul_rn(tl.load(d_ptr + b * N + rn, mask=mn, other=0.0)[None, :], acc.to(tl.float32))
        y = libdevice.fma(tl.load(as_ptr + b * M + rm, mask=mm, other=0.0)[:, None], t, y)
    _store(y, out_ptr, rm, rn, mm, mn, ldo, EPI)


CUDA_CFGS = {  # csrc/w4a8_gemm.cu: cfg id -> (BM, BN, warps); 13-16: K groups (few rows: warps on other super-blocks)
    0: (16, 64, 2), 1: (32, 64, 2), 2: (64, 64, 4), 3: (64, 128, 8), 4: (128, 64, 8), 5: (128, 128, 16), 6: (16, 32, 1),
    7: (32, 128, 4), 8: (128, 128, 8), 9: (128, 128, 8), 10: (64, 128, 4), 11: (128, 64, 4), 12: (64, 64, 2),
    13: (16, 32, 8), 14: (16, 32, 4), 15: (32, 32, 8), 16: (32, 32, 4),
    17: (8, 4, 4), 18: (8, 8, 4), 19: (8, 16, 4),   # 17-19: GEMV (up to 8 rows; a warp's own 1, 2 or 4 columns)
    20: (128, 128, 12), 21: (64, 128, 8)}           # 20-21: Hopper's warpgroup MMA (csrc/w4a8_wgmma.cu)
WGMMA_CFGS = (20, 21)


def wgmma_ok(M, N, K, w=None):
    """csrc/w4a8_wgmma.cu runs this product: a Hopper GPU with the sm_90a build, and the operands its TMA copies take
    (K a multiple of 256, M and N of 4, 16-byte aligned values)."""
    from . import cuda_ext as CX
    if not CX.has_wgmma() or K % 256 or M % 4 or N % 4:
        return False
    return w is None or all(t is None or t.data_ptr() % 16 == 0 for t in (w.q, w.sc, w.mn, w.d, w.dm))


@functools.lru_cache(None)
def _table():
    """{(kind, N, K): [(M, (cfg, stages))] by M} measured by bench/w4a8_table.py for the engine's projection shapes on
    this GPU's model: kernels/w4a8_table_<gpu tag>.json (rtx5090); STARTLUX_PRISM_W4A8_TABLE names another file, or "off"."""
    from . import cuda_ext
    path = os.environ.get("STARTLUX_PRISM_W4A8_TABLE") or cuda_ext.table_path("w4a8_table")
    if path == "off" or not os.path.exists(path):
        return {}
    out = {}
    for key, by_m in json.load(open(path))["table"].items():
        kind, n, k = key.split(",")
        out[(kind, int(n), int(k))] = sorted((int(m), tuple(c)) for m, c in by_m.items())
    return out


def pick_cuda(M, N, kind, K=None):
    """(cfg, stages) of csrc/w4a8_gemm.cu: the table's choice for the smallest measured token count >= M (its largest
    beyond them), or pick_rule for a shape it does not have."""
    entries = _table().get((kind, N, K))
    if entries:
        return next((c for m, c in entries if m >= M), entries[-1][1])
    if M >= 128 and K is not None and wgmma_ok(M, N, K):   # Hopper: its warpgroup MMA, 1.4-2.1x the rule's from
        return 20, 3                                         # 128 rows (the 27B's shapes on the H200)
    return pick_rule(M, N, kind)


def pick_rule(M, N, kind):
    """(cfg, stages) of csrc/w4a8_gemm.cu for a shape the table lacks, from bench/w4a8_micro.py --search on the RTX 5090
    (4B shapes): narrow tiles and many CTAs while the weight stream dominates, then 64x32 warp tiles (cfgs 8 and 11)
    for Q4_K and 64x64 tiles (cfg 2) for Q6_K, whose registers cap it."""
    if kind == "q4k":
        if M <= 1:
            return 6, 2
        if M <= 64:
            return (1, 4) if N > 4096 else (0, 4)
        if M <= 128:
            return (7, 2) if N > 4096 else (0, 4)
        if M <= 512:
            return (11, 2) if N > 4096 else (2, 2)
        return (11, 2) if M <= 1024 and N > 4096 else (8, 2)
    if M <= 1:
        return 6, 4
    if M <= 64:
        return (2, 4) if M > 32 and N > 4096 else (0, 4)
    if M <= 256:
        return 7, 2
    return 2, 2


def gemm(xq, a, xs, w: W4, out, n_rows, epi=0, cfg=None, pdl=0):
    """xq int8 [M, K], a fp32 [K / 256, M], xs int8 [K / 256, M, 16] (Q4_K's mins; quantize_ref), w W4 [N, K] ->
    out [M, N]: bf16 (epi 0) or added into fp32 (epi 1).  The CUDA kernel (csrc/w4a8_gemm.cu); cfg (id, stages) or
    w4a8.pick_cuda; pdl: launched to overlap the kernel before (programmatic dependent launch, 1-3: csrc)."""
    if torch.version.hip is not None and xq.dtype == torch.float16:
        if cfg is not None:
            raise ValueError("CUDA cfg IDs cannot be used by ROCm W4A16 kernels")
        from .amd_w4a16 import gemm as fp16_gemm
        return fp16_gemm(xq, w, out, n_rows, epi)
    if torch.version.hip is not None:
        if cfg is not None:
            raise ValueError("CUDA cfg IDs cannot be used by ROCm W4A8 kernels")
        implementation = os.getenv("STARTLUX_PRISM_AMD_GEMM", "auto")
        if implementation in ("auto", "dispatch", "decode", "prefill"):
            from .amd_w4a8_dispatch import run
            if run(xq, a, xs, w, out, n_rows, epi, implementation):
                return
            implementation = "gemv"
        if implementation == "gemv":
            if xq.shape[0] <= 8:
                from .amd_w4a8 import gemv
                return gemv(xq, a, xs, w, out, n_rows, epi=epi)
            return gemm_triton(xq, a, xs, w, out, n_rows, epi=epi)
        if implementation != "reference":
            raise ValueError(f"Unknown ROCm W4A8 implementation: {implementation}")
        return gemm_triton(xq, a, xs, w, out, n_rows, epi=epi)
    from . import cuda_ext as CX
    c, stages = cfg or pick_cuda(xq.shape[0], w.n, w.kind, w.k)
    if c in WGMMA_CFGS and cfg is None and not wgmma_ok(xq.shape[0], w.n, w.k, w):
        c, stages = pick_rule(xq.shape[0], w.n, w.kind)     # operands its TMA copies cannot take (alignment)
    if c in WGMMA_CFGS:
        CX.load().w4a8_wgmma(xq, a, xs, w.q, w.sc, w.mn, w.d, w.dm, out, n_rows, 0 if w.kind == "q4k" else 1, epi,
                             CUDA_CFGS[c][0], stages)
        return
    CX.load().w4a8_gemm(xq, a, xs, w.q, w.sc, w.mn, w.d, w.dm, out, n_rows, 0 if w.kind == "q4k" else 1, epi, c,
                        stages, int(pdl))


def pick(M, N, kind="q4k"):
    """(BM, BN, warps, stages) of the Triton kernels (Q6_K's int8 tiles take twice Q4_K's shared memory)."""
    if torch.version.hip is not None:
        if M <= 32:
            return 16, 32, 4, 1
        return (64, 128, 8, 1) if M >= 512 else (64, 64, 4, 1)
    if M <= 32:
        return 16, 128, 4, 3
    if M <= 128:
        return 64, 64, 4, 3
    return (128, 128, 8, 3) if kind == "q4k" else (128, 64, 8, 2)


def gemm_triton(xq, a, xs, w: W4, out, n_rows, epi=0, cfg=None):
    """The same product in Triton (the CUDA kernel's bits; about 10-200x slower: a reference)."""
    M, K = xq.shape
    N = w.n
    BM, BN, warps, stages = cfg or pick(M, N, w.kind)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    if w.kind == "q4k":
        _q4k_kernel[grid](xq, a, xs, w.q, w.sc, w.mn, w.d, w.dm, out, n_rows, M, N, K, out.stride(0),
                          BM=BM, BN=BN, GROUP_M=8, EPI=epi, num_warps=warps, num_stages=stages)
    else:
        _q6k_kernel[grid](xq, a, w.q, w.sc, w.d, out, n_rows, M, N, K, out.stride(0),
                          BM=BM, BN=BN, GROUP_M=8, EPI=epi, num_warps=warps, num_stages=stages)
