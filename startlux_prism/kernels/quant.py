"""Producers of int8 projection inputs: each writes int8 rows with one fp32 scale per row (symmetric, per token), with
the SmoothQuant factors of the consuming projection applied first (x / s; the weights carry W * s).  One program per
token sees the whole row, so the scale needs no cross-program reduction and quantization adds no pass.

  norm_q8       h (fp32 residual) -> rmsnorm(h) * w / s                     (inputs of qkv|z|b|a, attention qkv, gate|up)
  gated_norm_q8 o [T, HV, D], z -> (rmsnorm_D(o) * w_D * silu(z)) / s       (input of the linear-attention out_proj)
  swiglu_q8     gu [T, 2 FF] -> silu(gate) * up / s                         (input of down_proj)
  rows_q8       x [T, K] bf16 -> x / s                                       (attention output, input of o_proj)

norm_q8 (and fold) also finish a residual add that a split-K output projection deferred (w8a8.Deferred): the splits'
int32 partial sums are added exactly, scaled and added into h with the same roundings as an unsplit projection's
epilogue, so the residual has the same bits whichever way the projection ran.

Tensor parallelism (startlux_prism/tp.py): the inputs of the row-parallel projections (gated norm, attention output,
SwiGLU) hold this rank's heads or columns only, while the per-token scale is the maximum over the whole row.  On a
short pass the ranks exchange their row maxima inside the producer's one launch (mode 3, _ll_amax: per row, a 64-bit
word that carries the pass number above the float's bits, no flag and no barrier); on a long one the producer runs
twice around an NCCL max all-reduce: mode 1 also stores this rank's row maximum, mode 2 quantizes with the maximum
over every rank's (`amax`, at least this rank's own: the maximum of the two is the same number).  Every mode runs the
single-GPU kernel's code (mode 0) whole - the row's values, its maximum, the int8 stores; the other modes only add
scalar stores or loads - because Triton chooses a kernel's register layouts from its loads and stores, and the layout
sets the order of a sum over a row (the gated norm's over a head's D values): a mode 1 that stored only the maximum
summed in another order and was off by an ulp on some rows.  The gated norm also keeps the whole model's head block
(HP), so a shard's heads reduce as the whole model's do.  A maximum is exact in any order, so each rank's int8 values
and scales are the single-GPU kernel's, bit for bit.

Every producer signals at its start that the next kernel may launch (programmatic dependent launch): the CUDA GEMM
that consumes its output then streams its weights while this kernel runs, and waits for it before reading the
activations (csrc/w8a8_gemm.cu).  The signal does nothing for a kernel launched the ordinary way.

Fused sums (tensor parallel, the shortest passes: tp.py): instead of an all-reduce between two kernels, norm_q8 / fold
read every rank's int32 slots of an output projection themselves, over NVLink from the ranks' symmetric-memory
buffers, after a handshake (_tp_sync): the grid's first program sets this rank's flag for the step on every other rank
(the pass number; a release store, system scope), and every program waits for every other rank's (strong loads, then a
fence: an acquire), so what each rank wrote before this kernel is visible.  Integer sums are exact in any order, so the
results are those of the NCCL path and of one GPU (Sync).
"""
import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from ..backends import math as libdevice


@functools.lru_cache(None)
def _pdl():
    """The producers signal programmatic dependent launch (griddepcontrol.launch_dependents) on sm_90 and later only:
    the instruction does not exist before Hopper, and there no GEMM launches early (no dispatch table, so no CUDA GEMM
    with PDL; csrc/w8a8_gemm.cu)."""
    return torch.version.hip is None and torch.cuda.get_device_capability()[0] >= 9


class Sync(NamedTuple):
    """One fused all-reduce step of a tensor-parallel group (startlux_prism/tp.py; module docstring): `epoch` int64 [1], the
    pass's number (the same on every rank); `flags`, every rank's flag array (int64 [channels, ranks]) as mapped
    here, in rank order; `chan`, this step's flags (its index in the pass); `others`, the other ranks' data buffers
    (as mapped here); this process's rank and the group's size."""
    epoch: torch.Tensor
    flags: tuple
    chan: int
    others: tuple
    rank: int
    world: int


def _sync_args(sync, like):
    """Kernel arguments of a Sync (or of none: the single-GPU and NCCL paths): epoch, 4 flag arrays, channel, 3 other
    ranks' buffers; RANK, NR."""
    if sync is None:
        return (like,) * 5 + (0,) + (like,) * 3, 0, 1
    f = tuple(sync.flags) + (sync.flags[0],) * (4 - len(sync.flags))
    o = tuple(sync.others) + (like,) * (3 - len(sync.others))
    return (sync.epoch, *f, sync.chan, *o), sync.rank, sync.world


@triton.jit
def _bump_kernel(ep_ptr):
    tl.store(ep_ptr, tl.load(ep_ptr) + 1)


def bump(epoch):
    """A new pass number (int64 [1]) at the start of a tensor-parallel pass: every rank runs the same passes, so they
    count alike."""
    _bump_kernel[(1,)](epoch, num_warps=1)


@triton.jit
def _tp_sync(ep_ptr, f0, f1, f2, f3, chan, RANK: tl.constexpr, NR: tl.constexpr):
    """Signal this rank's arrival at step `chan` on every other rank, then wait for every other rank's (module
    docstring).  The grid's first program signals (a release store by one thread: what the rank wrote before this
    kernel is visible to whoever sees the flag); every program waits with plain strong loads and one fence after
    them (an acquire).  The first program is dispatched first and waits for nothing before signaling, so no program
    can wait on a signal that cannot come."""
    one = tl.zeros([1], dtype=tl.int64)
    e = tl.load(ep_ptr + one)
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        for q in tl.static_range(NR):
            if q != RANK:
                fq = f0 if q == 0 else (f1 if q == 1 else (f2 if q == 2 else f3))
                _st_release_sys(fq + chan * NR + RANK + one, e)
    mine = f0 if RANK == 0 else (f1 if RANK == 1 else (f2 if RANK == 2 else f3))
    es = tl.max(e, 0)
    for q in tl.static_range(NR):
        if q != RANK:
            while _ld_sys(mine + chan * NR + q + one) < es:
                pass
    _fence_sys(one)


@triton.jit
def _st_release_sys(ptr, val):
    """A release store (system scope) of a 64-bit word by the program's first thread; ptr, val: [1] tensors."""
    tl.inline_asm_elementwise("{ .reg .pred p; .reg .u32 i; mov.u32 i, %tid.x; setp.eq.u32 p, i, 0; "
                              "@p st.release.sys.global.b64 [$1], $2; mov.u32 $0, 0; }", "=r,l,l", [ptr, val],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _fence_sys(like):
    """fence.acq_rel.sys (like: any [1] tensor) - after a strong read that saw another rank's release store, an
    acquire."""
    tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r,l", [like], dtype=tl.int32, is_pure=False,
                              pack=1)


@triton.jit
def _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID: tl.constexpr, NP: tl.constexpr,
                o1, o2, o3, NR: tl.constexpr):
    """NR > 1: the other ranks' slots too (o1 .. o_{NR-1}, read over NVLink, uncached)."""
    tot = tl.load(parts_ptr + r * HID + c, mask=m, other=0)
    for p in tl.static_range(1, NP):
        tot += tl.load(parts_ptr + p * part_stride + r * HID + c, mask=m, other=0)
    for q in tl.static_range(1, NR):
        oq = o1 if q == 1 else (o2 if q == 2 else o3)
        for p in tl.static_range(NP):
            tot += tl.load(oq + p * part_stride + r * HID + c, mask=m, other=0, cache_modifier=".cv")
    v = libdevice.mul_rn(libdevice.mul_rn(tot.to(tl.float32), tl.load(psa_ptr + r)),
                         tl.load(psw_ptr + c, mask=m, other=0.0))
    return libdevice.add_rn(h, v)


@triton.jit
def _norm_q8_kernel(h_ptr, w_ptr, q_ptr, s_ptr, rows_ptr, n_ptr, eps, parts_ptr, part_stride, psa_ptr, psw_ptr,
                    ep_ptr, f0, f1, f2, f3, chan, o1, o2, o3,
                    HID: tl.constexpr, HP: tl.constexpr, GATHER: tl.constexpr, NP: tl.constexpr,
                    RANK: tl.constexpr, NR: tl.constexpr, PDL: tl.constexpr):
    """NR > 1: the deferred add's slots are every rank's (a fused all-reduce, module docstring)."""
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    i = tl.program_id(0).to(tl.int64)
    if i >= tl.load(n_ptr):
        return
    if GATHER:
        r = tl.load(rows_ptr + i).to(tl.int64)
    else:
        r = i
    c = tl.arange(0, HP)
    m = c < HID
    if NR > 1:
        _tp_sync(ep_ptr, f0, f1, f2, f3, chan, RANK, NR)
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    if NP > 0:      # a residual add deferred by a split-K projection, finished here and kept
        h = _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID, NP, o1, o2, o3, NR)
        tl.store(h_ptr + r * HID + c, h, mask=m)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    amax = tl.max(tl.abs(y), 0)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + i * HID + c, libdevice.rint(y / scale).to(tl.int8), mask=m)
    tl.store(s_ptr + i, scale)


@triton.jit
def _st_sys(ptr, val):
    """A strong (relaxed, system scope) 64-bit store by the program's first thread; ptr, val: [1] tensors (which
    every thread holds)."""
    tl.inline_asm_elementwise("{ .reg .pred p; .reg .u32 i; mov.u32 i, %tid.x; setp.eq.u32 p, i, 0; "
                              "@p st.relaxed.sys.global.b64 [$1], $2; mov.u32 $0, 0; }", "=r,l,l", [ptr, val],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ld_sys(ptr):
    """A strong (relaxed, system scope) 64-bit load of a [1] pointer tensor -> scalar."""
    return tl.max(tl.inline_asm_elementwise("ld.relaxed.sys.global.b64 $0, [$1];", "=l,l", [ptr], dtype=tl.int64,
                                            is_pure=False, pack=1), 0)


@triton.jit
def _ll_amax(amax, t, ep_ptr, w0, w1, w2, w3, off, RANK: tl.constexpr, NR: tl.constexpr):
    """Mode 3: a row's maximum over every rank in the producer's one launch (tensor parallel, short passes).  This
    rank's maximum and the pass number travel as one 64-bit word (number above the float's bits: written and read
    whole, so no flag is needed apart from the data) into word (off + t) * NR + RANK of every other rank's buffer
    (w0..w3: every rank's, in rank order); then each other rank's word for this row is read from this rank's buffer
    until it carries this pass's number.  Per row, no barrier: a buffer word is written again only by the same step
    of a later pass, which no rank reaches before every rank has finished this step's kernel."""
    e = tl.load(ep_ptr) & 0xFFFFFFFF
    one = tl.zeros([1], dtype=tl.int64)
    word = (e << 32) | (amax.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF)
    for q in tl.static_range(NR):
        if q != RANK:
            wq = w0 if q == 0 else (w1 if q == 1 else (w2 if q == 2 else w3))
            _st_sys(wq + (off + t) * NR + RANK + one, word + one)
    mine = w0 if RANK == 0 else (w1 if RANK == 1 else (w2 if RANK == 2 else w3))
    for q in tl.static_range(NR):
        if q != RANK:
            v = _ld_sys(mine + (off + t) * NR + q + one)
            while ((v >> 32) & 0xFFFFFFFF) != e:
                v = _ld_sys(mine + (off + t) * NR + q + one)
            amax = tl.maximum(amax, (v & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True))
    return amax


@triton.jit
def _gated_norm_q8_kernel(o_ptr, z_ptr, P, Z_OFF, w_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, eps, amax_ptr,
                          ep_ptr, w0, w1, w2, w3, off,
                          HV: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr,
                          RANK: tl.constexpr, NR: tl.constexpr, PDL: tl.constexpr):
    """HP: HV rounded up to a power of two (the 27B's 48 value heads); the rows past HV are masked to zero.  Tensor
    parallel: HV is this rank's heads, HP the whole model's (the same block, so the same order of the sum over D);
    MODE 1 also stores the row's maximum in amax, MODE 2 quantizes with the maximum over the ranks, MODE 3 exchanges
    the maxima itself (module docstring)."""
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    hh = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = (hh < HV)[:, None]
    idx = hh[:, None] * D + d[None, :]
    o = tl.load(o_ptr + t * HV * D + idx, mask=hm, other=0.0)
    z = tl.load(z_ptr + t * P + Z_OFF + idx, mask=hm, other=0.0).to(tl.float32)
    y = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None] * tl.load(w_ptr + d)[None, :] * (z / (1.0 + tl.exp(-z)))
    y = y * tl.load(inv_s_ptr + idx, mask=hm, other=0.0)
    amax = tl.max(tl.max(tl.abs(y), 1), 0)
    if MODE == 1:
        tl.store(amax_ptr + t, amax)
    if MODE == 2:
        amax = tl.maximum(tl.load(amax_ptr + t), amax)
    if MODE == 3:
        amax = _ll_amax(amax, t, ep_ptr, w0, w1, w2, w3, off, RANK, NR)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * HV * D + idx, libdevice.rint(y / scale).to(tl.int8), mask=hm)
    tl.store(s_ptr + t, scale)


@triton.jit
def _swiglu_piece(row, inv_s_ptr, FF: tl.constexpr, OFF: tl.constexpr, P: tl.constexpr):
    """silu(gate) * up / smoothing of columns OFF .. OFF + P - 1 of a gate|up row (masked to FF)."""
    c = OFF + tl.arange(0, P)
    m = c < FF
    g = tl.load(row + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(row + FF + c, mask=m, other=0.0).to(tl.float32)
    return g / (1.0 + tl.exp(-g)) * u * tl.load(inv_s_ptr + c, mask=m, other=0.0), c, m


@triton.jit
def _swiglu_q8_kernel(gu_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, amax_ptr, ep_ptr, w0, w1, w2, w3, off,
                      FF: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, MODE: tl.constexpr, RANK: tl.constexpr,
                      NR: tl.constexpr, PDL: tl.constexpr):
    """One row, in registers as two power-of-2 pieces (P1 + P2 >= FF: the 27B's 17,408 = 16,384 + 1,024; one power
    of 2 for the whole row left 47% of its lanes masked).  The row's maximum is exact in any order: the same bits.
    MODE 1 / 2 / 3: tensor parallel (module docstring)."""
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    row = gu_ptr + t * 2 * FF
    y1, c1, m1 = _swiglu_piece(row, inv_s_ptr, FF, 0, P1)
    y2, c2, m2 = _swiglu_piece(row, inv_s_ptr, FF, P1, P2)
    amax = tl.maximum(tl.max(tl.abs(y1), 0), tl.max(tl.abs(y2), 0))
    if MODE == 1:
        tl.store(amax_ptr + t, amax)
    if MODE == 2:
        amax = tl.maximum(tl.load(amax_ptr + t), amax)
    if MODE == 3:
        amax = _ll_amax(amax, t, ep_ptr, w0, w1, w2, w3, off, RANK, NR)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * FF + c1, libdevice.rint(y1 / scale).to(tl.int8), mask=m1)
    tl.store(q_ptr + t * FF + c2, libdevice.rint(y2 / scale).to(tl.int8), mask=m2)
    tl.store(s_ptr + t, scale)


@triton.jit
def _rows_q8_kernel(x_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, amax_ptr, ep_ptr, w0, w1, w2, w3, off,
                    K: tl.constexpr, KP: tl.constexpr, MODE: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr,
                    PDL: tl.constexpr):
    """MODE 1 / 2 / 3: tensor parallel (module docstring)."""
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    c = tl.arange(0, KP)
    m = c < K
    y = tl.load(x_ptr + t * K + c, mask=m, other=0.0).to(tl.float32) * tl.load(inv_s_ptr + c, mask=m, other=0.0)
    amax = tl.max(tl.abs(y), 0)
    if MODE == 1:
        tl.store(amax_ptr + t, amax)
    if MODE == 2:
        amax = tl.maximum(tl.load(amax_ptr + t), amax)
    if MODE == 3:
        amax = _ll_amax(amax, t, ep_ptr, w0, w1, w2, w3, off, RANK, NR)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * K + c, libdevice.rint(y / scale).to(tl.int8), mask=m)
    tl.store(s_ptr + t, scale)


# ---- W4A8 (kernels/w4a8.py): one scale per 256 inputs, no smoothing (the weights are the file's own values), and
# the sum of each 32 int8 values (for Q4_K's mins).  The same producers otherwise.
@triton.jit
def _store_g(y, q_ptr, s_ptr, xs_ptr, row, M, K: tl.constexpr, KP: tl.constexpr):
    """y [KP] fp32 (zero past K) -> row `row` of q int8 [M, K], s fp32 [K / 256, M] and xs int8 [K / 256, M, 16]: per
    256 inputs the scale and the sums S of the eight groups of 32 (|S| <= 32 * 127), split as eight S & 63 then eight
    S >> 6 (super-block-major, as the GEMM reads them)."""
    if q_ptr.dtype.element_ty == tl.float16:
        c = tl.arange(0, KP)
        tl.store(q_ptr + row * K + c, y, c < K)
    else:
        yg = tl.reshape(y, (KP // 256, 256))
        scale = libdevice.div_rn(tl.maximum(tl.max(tl.abs(yg), 1), 1e-10), 127.0)     # IEEE divisions, as the reference
        qv = libdevice.rint(libdevice.div_rn(yg, scale[:, None])).to(tl.int8)
        gi = tl.arange(0, KP // 256)
        c = gi[:, None] * 256 + tl.arange(0, 256)[None, :]
        tl.store(q_ptr + row * K + c, qv, mask=c < K)
        tl.store(s_ptr + gi * M + row, scale, mask=gi < K // 256)
        si = tl.arange(0, KP // 32)
        sums = tl.sum(tl.reshape(qv.to(tl.int32), (KP // 32, 32)), 1)
        o = ((si // 8) * M + row) * 16 + si % 8
        tl.store(xs_ptr + o, (sums & 63).to(tl.int8), mask=si < K // 32)
        tl.store(xs_ptr + o + 8, (sums >> 6).to(tl.int8), mask=si < K // 32)


@triton.jit
def _norm_g_kernel(h_ptr, w_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, eps, HID: tl.constexpr, HP: tl.constexpr,
                   PDL: tl.constexpr):
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    c = tl.arange(0, HP)
    m = c < HID
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    _store_g(y, q_ptr, s_ptr, xs_ptr, r, M, HID, HP)


@triton.jit
def _gated_norm_g_kernel(o_ptr, z_ptr, P, Z_OFF, w_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, eps,
                         HV: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, PDL: tl.constexpr):
    """HP: HV rounded up to a power of two (the 27B's 48 value heads); the rows past HV are zero, past K for
    _store_g."""
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    hh = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = (hh < HV)[:, None]
    idx = hh[:, None] * D + d[None, :]
    o = tl.load(o_ptr + t * HV * D + idx, mask=hm, other=0.0)
    z = tl.load(z_ptr + t * P + Z_OFF + idx, mask=hm, other=0.0).to(tl.float32)
    y = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None] * tl.load(w_ptr + d)[None, :] * (z / (1.0 + tl.exp(-z)))
    _store_g(tl.reshape(y, (HP * D,)), q_ptr, s_ptr, xs_ptr, t, M, HV * D, HP * D)


@triton.jit
def _swiglu_g_kernel(gu_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, FF: tl.constexpr, FP: tl.constexpr, PDL: tl.constexpr):
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    c = tl.arange(0, FP)
    m = c < FF
    g = tl.load(gu_ptr + t * 2 * FF + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + t * 2 * FF + FF + c, mask=m, other=0.0).to(tl.float32)
    _store_g(g / (1.0 + tl.exp(-g)) * u, q_ptr, s_ptr, xs_ptr, t, M, FF, FP)


@triton.jit
def _rows_g_kernel(x_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, K: tl.constexpr, KP: tl.constexpr, PDL: tl.constexpr):
    if PDL:                                          # sm_90 and later (_pdl)
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    c = tl.arange(0, KP)
    y = tl.load(x_ptr + t * K + c, mask=c < K, other=0.0).to(tl.float32)
    _store_g(y, q_ptr, s_ptr, xs_ptr, t, M, K, KP)


def norm_g(h, w, q, s, xs, n, T, eps):
    HID = h.shape[1]
    _norm_g_kernel[(T,)](h, w, q, s, xs, n, s.shape[1], eps, HID=HID, HP=_p2(HID), PDL=_pdl(), num_warps=8)


def gated_norm_g(o, proj, z_off, w, q, s, xs, n, T, cfg):
    _gated_norm_g_kernel[(T,)](o, proj, proj.stride(0), z_off, w, q, s, xs, n, s.shape[1], cfg.eps, HV=cfg.v_heads,
                               HP=_p2(cfg.v_heads), D=cfg.gdn_dim, PDL=_pdl(), num_warps=8)


def swiglu_g(gu, q, s, xs, n, T):
    FF = q.shape[1]
    _swiglu_g_kernel[(T,)](gu, q, s, xs, n, s.shape[1], FF=FF, FP=_p2(FF), PDL=_pdl(), num_warps=16)


def rows_g(x, q, s, xs, n, T):
    K = q.shape[1]
    _rows_g_kernel[(T,)](x, q, s, xs, n, s.shape[1], K=K, KP=_p2(K), PDL=_pdl(), num_warps=8)


def _p2(n):
    return 1 << (n - 1).bit_length()


def norm_q8(h, w_over_s, q, s, n, T, eps, rows=None, parts=None, pending=None):
    """parts int32 [S, rows, HID], pending (w8a8.Deferred or None): a residual add to finish in h first (its own
    slots when it names them, else `parts`; with a Sync, every rank's).  The deferred projection's activation scales
    may be `s` itself (each row reads its scale before writing the new one)."""
    HID = h.shape[1]
    np_ = pending.n if pending else 0
    if np_ and pending.parts is not None:
        parts = pending.parts
    assert np_ == 0 or (parts is not None and rows is None and parts.dtype == torch.int32)
    sa, RANK, NR = _sync_args(pending.sync if np_ else None, h)
    _norm_q8_kernel[(T,)](h, w_over_s, q, s, rows if rows is not None else n, n, eps,
                          parts if np_ else h, parts.stride(0) if np_ else 0, pending.a_s if np_ else s,
                          pending.w_s if np_ else s, *sa, HID=HID, HP=_p2(HID), GATHER=rows is not None, NP=np_,
                          RANK=RANK, NR=NR, PDL=_pdl(), num_warps=8)


@triton.jit
def _fold_kernel(h_ptr, parts_ptr, part_stride, psa_ptr, psw_ptr, n_ptr, ep_ptr, f0, f1, f2, f3, chan, o1, o2, o3,
                 HID: tl.constexpr, HP: tl.constexpr, NP: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    c = tl.arange(0, HP)
    m = c < HID
    if NR > 1:
        _tp_sync(ep_ptr, f0, f1, f2, f3, chan, RANK, NR)
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    tl.store(h_ptr + r * HID + c, _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID, NP, o1, o2,
                                              o3, NR), mask=m)


def fold(h, parts, pending, n, T):
    """Finish a deferred residual add (w8a8.Deferred) in h when no norm follows."""
    if pending:
        HID = h.shape[1]
        if pending.parts is not None:
            parts = pending.parts
        sa, RANK, NR = _sync_args(pending.sync, h)
        _fold_kernel[(T,)](h, parts, parts.stride(0), pending.a_s, pending.w_s, n, *sa, HID=HID, HP=_p2(HID),
                           NP=pending.n, RANK=RANK, NR=NR, num_warps=8)


def gated_norm_q8(o, proj, z_off, w, inv_s, q, s, n, T, cfg, mode=0, amax=None, sync=None):
    """mode 0: the whole row here; 1 / 2 (tensor parallel): store this rank's row maxima in amax / quantize with the
    maximum over the ranks - all-reduced into amax, or (sync) read from every rank's (module docstring).
    cfg.v_heads_full: the whole model's value heads (a tensor-parallel shard's config), whose power-of-2 block every
    mode uses."""
    amax = amax if amax is not None else s
    sa, RANK, NR = _sync_args(sync, amax)
    _gated_norm_q8_kernel[(T,)](o, proj, proj.stride(0), z_off, w, inv_s, q, s, n, cfg.eps, amax, *sa[:6],
                                HV=cfg.v_heads, HP=_p2(getattr(cfg, "v_heads_full", cfg.v_heads)), D=cfg.gdn_dim,
                                MODE=mode, RANK=RANK, NR=NR, PDL=_pdl(), num_warps=8)


def swiglu_q8(gu, inv_s, q, s, n, T, mode=0, amax=None, sync=None):
    FF = q.shape[1]
    P1 = 1 << (FF.bit_length() - 1)                 # the largest power of 2 <= FF, then the rest (16 at least)
    amax = amax if amax is not None else s
    sa, RANK, NR = _sync_args(sync, amax)
    _swiglu_q8_kernel[(T,)](gu, inv_s, q, s, n, amax, *sa[:6], FF=FF, P1=P1, P2=max(_p2(FF - P1), 16), MODE=mode,
                            RANK=RANK, NR=NR, PDL=_pdl(), num_warps=4 if T <= 2048 else 16)


def rows_q8(x, inv_s, q, s, n, T, mode=0, amax=None, sync=None):
    K = q.shape[1]
    amax = amax if amax is not None else s
    sa, RANK, NR = _sync_args(sync, amax)
    _rows_q8_kernel[(T,)](x, inv_s, q, s, n, amax, *sa[:6], K=K, KP=_p2(K), MODE=mode, RANK=RANK, NR=NR, PDL=_pdl(),
                          num_warps=8)
