"""MXFP4 projections: e2m1 weights and activations, one e8m0 scale per 32 inputs, fp32 accumulate.

Weights: packed uint8 [n, k/2] (two e2m1 values per byte, low nibble first; k even) and e8m0 scales uint8 [n, k/32]
(value 2**(s - 127)).  Activations use the same layout for n rows.

The GEMM is Triton's tl.dot_scaled, which lowers to
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X on sm_120.  An optional CUTLASS launcher
(csrc/mxfp4_gemm.cu, built by scripts/build_mxfp4_cutlass.sh) replaces it on SM120/SM121 when enabled.  The Triton K
loop order is fixed by BK for every M, so an output element is the same whatever the batch it is computed in (batch
invariance); CUTLASS's order matches it on the same inputs.
"""
from __future__ import annotations

import ctypes
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

QK = 32          # MXFP4 block: one e8m0 scale per 32 inputs


class MX4(NamedTuple):
    q: torch.Tensor          # uint8 [n, k/2] packed along k
    s: torch.Tensor          # uint8 [n, k/32] e8m0 exponents
    n: int
    k: int
    cs: torch.Tensor | None = None   # CUTLASS-interleaved e8m0 scales (set by cutlass_prepare)

    @property
    def shape(self):
        return (self.n, self.k)

    def rows(self, r0, r1):
        return MX4(self.q[r0:r1], self.s[r0:r1], r1 - r0, self.k)


def _e2m1_code(x):
    """|x| / scale in [0, 6] -> the 3-bit magnitude code of e2m1 (nearest; strict thresholds round midpoints down)."""
    import numpy as np

    a = np.abs(x)
    c = np.zeros(a.shape, np.uint8)
    c[a > 0.25] = 1
    c[a > 0.75] = 2
    c[a > 1.25] = 3
    c[a > 1.75] = 4
    c[a > 2.5] = 5
    c[a > 3.5] = 6
    c[a > 5.0] = 7
    c[(x < 0) & (c != 0)] |= 8
    return c


def pack(w: torch.Tensor, mode: str = "max") -> MX4:
    """A [n, k] fp32/bf16 matrix -> MX4.  mode "max": scale = 2**ceil(log2(amax/6)); "mse": the power of two among
    the two neighbours of amax/6 that minimises the block's squared error.  CPU, row by row."""
    import numpy as np

    w = w.detach().to(torch.float32).cpu().numpy()
    n, k = w.shape
    assert k % QK == 0, (n, k)
    blocks = w.reshape(n, k // QK, QK)
    amax = np.abs(blocks).max(axis=2)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0))
    e = np.clip(e, -127, 127)
    if mode == "mse":
        lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        e0 = e
        e1 = np.clip(e - 1, -127, 127)
        s0 = np.exp2(e0).astype(np.float32)
        s1 = np.exp2(e1).astype(np.float32)
        v0 = lut[_e2m1_code(blocks / s0[..., None])] * s0[..., None]
        v1 = lut[_e2m1_code(blocks / s1[..., None])] * s1[..., None]
        err0 = ((v0 - blocks) ** 2).sum(axis=2)
        err1 = ((v1 - blocks) ** 2).sum(axis=2)
        e = np.where(err1 < err0, e1, e0)
    sc = np.exp2(e).astype(np.float32)
    code = _e2m1_code(blocks / sc[..., None])
    code = code.reshape(n, k)
    q = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    s = (e.astype(np.int16) + 127).astype(np.uint8).reshape(n, k // QK)
    return MX4(torch.from_numpy(q).contiguous(), torch.from_numpy(s).contiguous(), n, k)


def dequant_packed(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Reference dequantization of packed MX4 (fp32) - for tests."""
    import numpy as np

    qn = q.cpu().numpy()
    sn = s.cpu().numpy().astype(np.int16) - 127
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    n, k2 = qn.shape
    out = np.empty((n, k2 * 2), np.float32)
    out[:, 0::2] = lut[qn & 15]
    out[:, 1::2] = lut[qn >> 4]
    out = out.reshape(n, k2 * 2 // QK, QK) * np.exp2(sn).astype(np.float32)[..., None]
    return torch.from_numpy(out.reshape(n, k2 * 2))


# --------------------------------------------------------------------- activation quantization

@triton.jit
def _code(v, sc):
    a = tl.abs(v).to(tl.float32) / sc
    c = tl.zeros(a.shape, dtype=tl.int32)
    c = tl.where(a > 0.25, 1, c)
    c = tl.where(a > 0.75, 2, c)
    c = tl.where(a > 1.25, 3, c)
    c = tl.where(a > 1.75, 4, c)
    c = tl.where(a > 2.5, 5, c)
    c = tl.where(a > 3.5, 6, c)
    c = tl.where(a > 5.0, 7, c)
    return tl.where(v < 0, c | 8, c)

@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, n, K: tl.constexpr, BK: tl.constexpr):
    """One program per row: K inputs in blocks of BK (a multiple of 32), 2 values per output byte."""
    row = tl.program_id(0)
    if row >= n:
        return
    NB: tl.constexpr = BK // 32              # scale blocks per iteration
    H: tl.constexpr = BK // 2                # bytes per iteration
    base = row.to(tl.int64) * K
    qbase = row.to(tl.int64) * (K // 2)
    sbase = row.to(tl.int64) * (K // 32)
    for k0 in range(0, K, BK):
        idx = tl.arange(0, H)
        v0 = tl.load(x_ptr + base + k0 + 2 * idx)             # k even
        v1 = tl.load(x_ptr + base + k0 + 2 * idx + 1)         # k odd
        a0 = tl.abs(v0).to(tl.float32).reshape(NB, H // NB)
        a1 = tl.abs(v1).to(tl.float32).reshape(NB, H // NB)
        amax = tl.maximum(tl.max(a0, 1), tl.max(a1, 1))
        e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 6.0))
        e = tl.minimum(tl.maximum(e, -127.0), 127.0)
        sc = tl.exp2(e)                                        # [NB]
        sc0 = tl.reshape(tl.broadcast_to(sc[:, None], (NB, H // NB)), (H,))
        c0 = _code(v0, sc0)
        c1 = _code(v1, sc0)
        tl.store(q_ptr + qbase + k0 // 2 + idx, (c0 | (c1 << 4)).to(tl.uint8))
        tl.store(s_ptr + sbase + k0 // 32 + tl.arange(0, NB), (e.to(tl.int32) + 127).to(tl.uint8))


@triton.jit
def _quant_scaled_kernel(x_ptr, inv_ptr, q_ptr, s_ptr, n, K: tl.constexpr, BK: tl.constexpr):
    """quantize_act with a per-input-column pre-scale (x * inv), for an already smoothed input."""
    row = tl.program_id(0)
    if row >= n:
        return
    NB: tl.constexpr = BK // 32
    H: tl.constexpr = BK // 2
    base = row.to(tl.int64) * K
    qbase = row.to(tl.int64) * (K // 2)
    sbase = row.to(tl.int64) * (K // 32)
    for k0 in range(0, K, BK):
        idx = tl.arange(0, H)
        v0 = tl.load(x_ptr + base + k0 + 2 * idx).to(tl.float32) * tl.load(inv_ptr + k0 + 2 * idx)
        v1 = tl.load(x_ptr + base + k0 + 2 * idx + 1).to(tl.float32) * tl.load(inv_ptr + k0 + 2 * idx + 1)
        a0 = tl.abs(v0).reshape(NB, H // NB)
        a1 = tl.abs(v1).reshape(NB, H // NB)
        amax = tl.maximum(tl.max(a0, 1), tl.max(a1, 1))
        e = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 6.0)), -127.0), 127.0)
        sc = tl.exp2(e)
        sc0 = tl.reshape(tl.broadcast_to(sc[:, None], (NB, H // NB)), (H,))
        c0 = _code(v0, sc0)
        c1 = _code(v1, sc0)
        tl.store(q_ptr + qbase + k0 // 2 + idx, (c0 | (c1 << 4)).to(tl.uint8))
        tl.store(s_ptr + sbase + k0 // 32 + tl.arange(0, NB), (e.to(tl.int32) + 127).to(tl.uint8))


def quantize_act_scaled(x: torch.Tensor, inv: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> None:
    """x [n, k] * inv [k] -> q uint8 [n, k/2], s uint8 [n, k/32]."""
    n, k = x.shape
    BK = 256 if k % 256 == 0 else 32
    _quant_scaled_kernel[(n,)](x, inv, q, s, n, K=k, BK=BK, num_warps=4)


def quantize_act(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> None:
    """x [n, k] bf16/fp32 -> q uint8 [n, k/2], s uint8 [n, k/32].  q, s are the caller's buffers."""
    n, k = x.shape
    BK = 256 if k % 256 == 0 else 32
    _quant_kernel[(n,)](x, q, s, n, K=k, BK=BK, num_warps=4)


# --------------------------------------------------------------------- GEMM
# --------------------------------------------------------------------- CUTLASS (SM120/121, opt-in)
_CUTLASS = None
_CUTLASS_SFA = {}
CUTLASS_MIN_ROWS = 64          # below this Triton is faster; CUTLASS wins from about 64 rows up


def _cutlass():
    """The CUTLASS SM120 MXFP4 launcher, or None (STARTLUX_PRISM_MXFP4_CUTLASS=1 and an sm_120/121 GPU)."""
    global _CUTLASS
    if _CUTLASS is not None:
        return _CUTLASS or None
    if os.environ.get("STARTLUX_PRISM_MXFP4_CUTLASS", "") not in ("1", "on", "true"):
        _CUTLASS = False
        return None
    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        _CUTLASS = False
        return None
    path = os.environ.get("STARTLUX_PRISM_MXFP4_LIB") or os.path.join(
        os.path.dirname(__file__), "libmxfp4_cutlass.so")
    if not os.path.exists(path):
        _CUTLASS = False
        return None
    lib = ctypes.CDLL(path)
    lib.mxfp4_sfb_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mxfp4_sfb_bytes.restype = ctypes.c_long
    lib.mxfp4_sfa_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mxfp4_sfa_bytes.restype = ctypes.c_long
    lib.mxfp4_prepare_sfa.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p]
    lib.mxfp4_prepare_sfb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p]
    lib.mxfp4_gemm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.mxfp4_gemm.restype = ctypes.c_int
    _CUTLASS = lib
    return lib


def cutlass_enabled():
    return _cutlass() is not None


def cutlass_prepare(w):
    """MX4 with the scales in CUTLASS's interleaved layout (the data stays as it is; both are K-contiguous)."""
    lib = _cutlass()
    if lib is None or w.cs is not None:
        return w
    sb = lib.mxfp4_sfb_bytes(w.n, w.k)
    cs = torch.zeros(sb, dtype=torch.uint8, device=w.q.device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    lib.mxfp4_prepare_sfb(w.s.data_ptr(), cs.data_ptr(), w.n, w.k, stream)
    return MX4(w.q, w.s, w.n, w.k, cs)



def pick(M, N, K, epi=0):
    """(BM, BN, BK, GROUP_M, warps, stages) for a projection shape; BK fixed per (N, K) keeps the K order, and so the
    output bits, the same for every M.  A small heuristic; shape tables can replace it."""
    if N >= 8192:
        if M <= 64:
            return (64, 64, 128, 8, 8, 3)
        if M <= 256:
            return (64, 128, 128, 8, 8, 4)
        return (128, 128, 128, 8, 8, 3)
    if M <= 64:
        return (64, 64, 128, 8, 8, 3)
    if M <= 256:
        return (64, 128, 128, 8, 8, 3)
    return (128, 128, 128, 8, 8, 3)


@triton.jit
def _gemm_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, h_ptr, M, N: tl.constexpr, OUTS, K: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, EPI: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    gid = pid // per_group
    first_m = gid * GROUP_M
    gm = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % gm)
    pid_n = (pid % per_group) // gm
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK // 2)
    rs = tl.arange(0, BK // 32)
    mm = rm < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * (K // 2) + (k0 // 2 + rk)[None, :], mask=mm[:, None], other=0)
        asc = tl.load(as_ptr + rm[:, None] * (K // 32) + (k0 // 32 + rs)[None, :], mask=mm[:, None], other=0)
        b = tl.load(b_ptr + rn[None, :] * (K // 2) + (k0 // 2 + rk)[:, None])
        bsc = tl.load(bs_ptr + rn[:, None] * (K // 32) + (k0 // 32 + rs)[None, :])
        acc = tl.dot_scaled(a, asc, "e2m1", b, bsc, "e2m1", acc=acc, out_dtype=tl.float32)
    mask = mm[:, None] & (rn[None, :] < N)
    if EPI:
        prev = tl.load(h_ptr + rm[:, None] * OUTS + rn[None, :], mask=mask, other=0.0)
        tl.store(h_ptr + rm[:, None] * OUTS + rn[None, :], prev + acc, mask=mask)
    else:
        tl.store(c_ptr + rm[:, None] * OUTS + rn[None, :], acc.to(tl.bfloat16), mask=mask)


def gemm(a_q, a_s, w: MX4, out, n_rows, epi=0, out_stride=None):
    """a_q uint8 [n_rows, k/2], a_s uint8 [n_rows, k/32]; w MX4 [N, k].  epi=0 writes bf16 into `out`; epi=1 adds
    into the fp32 residual `out` (which then holds [n_rows, N] fp32)."""
    M, K = n_rows, w.k
    N = w.n
    lib = _cutlass()
    if (lib is not None and w.cs is not None and M >= CUTLASS_MIN_ROWS and out.stride(0) == N
            and a_q.is_contiguous() and a_s.is_contiguous()):
        key = (M, K, a_q.device.index)
        sfa = _CUTLASS_SFA.get(key)
        nbytes = lib.mxfp4_sfa_bytes(M, K)
        if sfa is None or sfa.numel() < nbytes:
            sfa = torch.zeros(nbytes, dtype=torch.uint8, device=a_q.device)
            _CUTLASS_SFA[key] = sfa
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        lib.mxfp4_prepare_sfa(a_s.data_ptr(), sfa.data_ptr(), M, K, stream)
        rc = lib.mxfp4_gemm(a_q.data_ptr(), sfa.data_ptr(), w.q.data_ptr(), w.cs.data_ptr(), out.data_ptr(),
                            M, N, K, epi, stream)
        if rc:
            raise RuntimeError(f"CUTLASS MXFP4 GEMM failed (status {rc}) at M={M}, N={N}, K={K}, epi={epi}")
        return
    BM, BN, BK, GROUP_M, warps, stages = pick(M, N, K, epi)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    outs = out.stride(0) if out_stride is None else out_stride
    _gemm_kernel[grid](a_q, a_s, w.q, w.s, out, out, M, N=N, OUTS=outs, K=K, BM=BM, BN=BN, BK=BK, GROUP_M=GROUP_M,
                       EPI=epi, num_warps=warps, num_stages=stages)
