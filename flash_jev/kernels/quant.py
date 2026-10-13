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

Every producer signals at its start that the next kernel may launch (programmatic dependent launch): the CUDA GEMM
that consumes its output then streams its weights while this kernel runs, and waits for it before reading the
activations (csrc/w8a8_gemm.cu).  The signal does nothing for a kernel launched the ordinary way.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID: tl.constexpr, NP: tl.constexpr):
    tot = tl.load(parts_ptr + r * HID + c, mask=m, other=0)
    for p in tl.static_range(1, NP):
        tot += tl.load(parts_ptr + p * part_stride + r * HID + c, mask=m, other=0)
    v = libdevice.mul_rn(libdevice.mul_rn(tot.to(tl.float32), tl.load(psa_ptr + r)),
                         tl.load(psw_ptr + c, mask=m, other=0.0))
    return libdevice.add_rn(h, v)


@triton.jit
def _norm_q8_kernel(h_ptr, w_ptr, q_ptr, s_ptr, rows_ptr, n_ptr, eps, parts_ptr, part_stride, psa_ptr, psw_ptr,
                    HID: tl.constexpr, HP: tl.constexpr, GATHER: tl.constexpr, NP: tl.constexpr):
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
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    if NP > 0:      # a residual add deferred by a split-K projection, finished here and kept
        h = _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID, NP)
        tl.store(h_ptr + r * HID + c, h, mask=m)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    amax = tl.max(tl.abs(y), 0)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + i * HID + c, tl.extra.cuda.libdevice.rint(y / scale).to(tl.int8), mask=m)
    tl.store(s_ptr + i, scale)


@triton.jit
def _gated_norm_q8_kernel(o_ptr, z_ptr, P, Z_OFF, w_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, eps,
                          HV: tl.constexpr, HP: tl.constexpr, D: tl.constexpr):
    """HP: HV rounded up to a power of two (the 27B's 48 value heads); the rows past HV are masked to zero."""
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
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * HV * D + idx, tl.extra.cuda.libdevice.rint(y / scale).to(tl.int8), mask=hm)
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
def _swiglu_q8_kernel(gu_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, FF: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr):
    """One row, in registers as two power-of-2 pieces (P1 + P2 >= FF: the 27B's 17,408 = 16,384 + 1,024; one power
    of 2 for the whole row left 47% of its lanes masked).  The row's maximum is exact in any order: the same bits."""
    tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    row = gu_ptr + t * 2 * FF
    y1, c1, m1 = _swiglu_piece(row, inv_s_ptr, FF, 0, P1)
    y2, c2, m2 = _swiglu_piece(row, inv_s_ptr, FF, P1, P2)
    amax = tl.maximum(tl.max(tl.abs(y1), 0), tl.max(tl.abs(y2), 0))
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * FF + c1, tl.extra.cuda.libdevice.rint(y1 / scale).to(tl.int8), mask=m1)
    tl.store(q_ptr + t * FF + c2, tl.extra.cuda.libdevice.rint(y2 / scale).to(tl.int8), mask=m2)
    tl.store(s_ptr + t, scale)


@triton.jit
def _rows_q8_kernel(x_ptr, inv_s_ptr, q_ptr, s_ptr, n_ptr, K: tl.constexpr, KP: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    c = tl.arange(0, KP)
    m = c < K
    y = tl.load(x_ptr + t * K + c, mask=m, other=0.0).to(tl.float32) * tl.load(inv_s_ptr + c, mask=m, other=0.0)
    amax = tl.max(tl.abs(y), 0)
    scale = tl.maximum(amax, 1e-10) / 127.0
    tl.store(q_ptr + t * K + c, tl.extra.cuda.libdevice.rint(y / scale).to(tl.int8), mask=m)
    tl.store(s_ptr + t, scale)


# ---- W4A8 (kernels/w4a8.py): one scale per 256 inputs, no smoothing (the weights are the file's own values), and
# the sum of each 32 int8 values (for Q4_K's mins).  The same producers otherwise.
@triton.jit
def _store_g(y, q_ptr, s_ptr, xs_ptr, row, M, K: tl.constexpr, KP: tl.constexpr):
    """y [KP] fp32 (zero past K) -> row `row` of q int8 [M, K], s fp32 [K / 256, M] and xs int8 [K / 256, M, 16]: per
    256 inputs the scale and the sums S of the eight groups of 32 (|S| <= 32 * 127), split as eight S & 63 then eight
    S >> 6 (super-block-major, as the GEMM reads them)."""
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
def _norm_g_kernel(h_ptr, w_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, eps, HID: tl.constexpr, HP: tl.constexpr):
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
                         HV: tl.constexpr, HP: tl.constexpr, D: tl.constexpr):
    """HP: HV rounded up to a power of two (the 27B's 48 value heads); the rows past HV are zero, past K for
    _store_g."""
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
def _swiglu_g_kernel(gu_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, FF: tl.constexpr, FP: tl.constexpr):
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
def _rows_g_kernel(x_ptr, q_ptr, s_ptr, xs_ptr, n_ptr, M, K: tl.constexpr, KP: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_ptr):
        return
    c = tl.arange(0, KP)
    y = tl.load(x_ptr + t * K + c, mask=c < K, other=0.0).to(tl.float32)
    _store_g(y, q_ptr, s_ptr, xs_ptr, t, M, K, KP)


def norm_g(h, w, q, s, xs, n, T, eps):
    HID = h.shape[1]
    _norm_g_kernel[(T,)](h, w, q, s, xs, n, s.shape[1], eps, HID=HID, HP=_p2(HID), num_warps=8)


def gated_norm_g(o, proj, z_off, w, q, s, xs, n, T, cfg):
    _gated_norm_g_kernel[(T,)](o, proj, proj.stride(0), z_off, w, q, s, xs, n, s.shape[1], cfg.eps, HV=cfg.v_heads,
                               HP=_p2(cfg.v_heads), D=cfg.gdn_dim, num_warps=8)


def swiglu_g(gu, q, s, xs, n, T):
    FF = q.shape[1]
    _swiglu_g_kernel[(T,)](gu, q, s, xs, n, s.shape[1], FF=FF, FP=_p2(FF), num_warps=16)


def rows_g(x, q, s, xs, n, T):
    K = q.shape[1]
    _rows_g_kernel[(T,)](x, q, s, xs, n, s.shape[1], K=K, KP=_p2(K), num_warps=8)


def _p2(n):
    return 1 << (n - 1).bit_length()


def norm_q8(h, w_over_s, q, s, n, T, eps, rows=None, parts=None, pending=None):
    """parts int32 [S, rows, HID], pending (w8a8.Deferred or None): a residual add to finish in h first.  The
    deferred projection's activation scales may be `s` itself (each row reads its scale before writing the new one)."""
    HID = h.shape[1]
    np_ = pending.n if pending else 0
    assert np_ == 0 or (parts is not None and rows is None and parts.dtype == torch.int32)
    _norm_q8_kernel[(T,)](h, w_over_s, q, s, rows if rows is not None else n, n, eps,
                          parts if np_ else h, parts.stride(0) if np_ else 0, pending.a_s if np_ else s,
                          pending.w_s if np_ else s, HID=HID, HP=_p2(HID), GATHER=rows is not None, NP=np_,
                          num_warps=8)


@triton.jit
def _fold_kernel(h_ptr, parts_ptr, part_stride, psa_ptr, psw_ptr, n_ptr, HID: tl.constexpr, HP: tl.constexpr,
                 NP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    c = tl.arange(0, HP)
    m = c < HID
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    tl.store(h_ptr + r * HID + c, _fold_parts(h, parts_ptr, part_stride, psa_ptr, psw_ptr, r, c, m, HID, NP), mask=m)


def fold(h, parts, pending, n, T):
    """Finish a deferred residual add (w8a8.Deferred) in h when no norm follows."""
    if pending:
        HID = h.shape[1]
        _fold_kernel[(T,)](h, parts, parts.stride(0), pending.a_s, pending.w_s, n, HID=HID, HP=_p2(HID),
                           NP=pending.n, num_warps=8)


def gated_norm_q8(o, proj, z_off, w, inv_s, q, s, n, T, cfg):
    _gated_norm_q8_kernel[(T,)](o, proj, proj.stride(0), z_off, w, inv_s, q, s, n, cfg.eps,
                                HV=cfg.v_heads, HP=_p2(cfg.v_heads), D=cfg.gdn_dim, num_warps=8)


def swiglu_q8(gu, inv_s, q, s, n, T):
    FF = q.shape[1]
    P1 = 1 << (FF.bit_length() - 1)                 # the largest power of 2 <= FF, then the rest (16 at least)
    _swiglu_q8_kernel[(T,)](gu, inv_s, q, s, n, FF=FF, P1=P1, P2=max(_p2(FF - P1), 16),
                            num_warps=4 if T <= 2048 else 16)


def rows_q8(x, inv_s, q, s, n, T):
    K = q.shape[1]
    _rows_q8_kernel[(T,)](x, inv_s, q, s, n, K=K, KP=_p2(K), num_warps=8)
