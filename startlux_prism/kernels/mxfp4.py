"""Experimental gfx950 native MXFP4 projections, separate from GGUF k-quants.

E2M1 payloads pack adjacent K values low nibble first. E8M0 scales cover 32
values along K. The initial recipe chooses the smallest power-of-two scale
that contains the block maximum in [-6, 6], then rounds ties to even.
"""
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@dataclass
class Packed:
    q: torch.Tensor                 # uint8 [rows, K / 2]
    s: torch.Tensor                 # uint8 [rows, K / 32], biased exponent
    shuffled: torch.Tensor | None = None  # uint8 [ceil(rows/32), K], MFMA16 scale layout

    @property
    def shape(self):
        return self.q.shape[0], self.q.shape[1] * 2

    def rows(self, start, end):
        r0, r1, _ = slice(start, end).indices(self.shape[0])
        sf = self.shuffled[r0 // 32:triton.cdiv(r1, 32)] if self.shuffled is not None and r0 % 32 == 0 else None
        return Packed(self.q[start:end], self.s[start:end], sf)

    def __getitem__(self, rows):
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise ValueError('MXFP4 weights only support contiguous row slices')
        return self.rows(rows.start, rows.stop)

    def nbytes(self):
        """Active payload plus one scale copy; resident storage also includes the auxiliary layout."""
        return self.q.numel() + self.s.numel()


def empty(rows, k, device):
    if k % 32:
        raise ValueError('MXFP4 K must be a multiple of 32')
    return Packed(torch.empty((rows, k // 2), dtype=torch.uint8, device=device),
                  torch.empty((rows, k // 32), dtype=torch.uint8, device=device),
                  torch.full((triton.cdiv(rows, 32), k), 127, dtype=torch.uint8, device=device) if k % 256 == 0 else None)


@triton.jit
def _scale_index(r, b, K: tl.constexpr):
    return ((r // 32) * K + (b // 8) * 256 + (b % 4) * 64
            + (r % 16) * 4 + ((b % 8) // 4) * 2 + ((r % 32) // 16))


@triton.jit
def _shuffle_scale(S, SF, ROWS: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r, b = i // (K // 32), i % (K // 32)
    tl.store(SF + _scale_index(r, b, K), tl.load(S + i, r < ROWS, 127), r < ROWS)


def prepare(packed):
    n, k = packed.shape
    if k % 256 == 0 and packed.shuffled is None:
        packed.shuffled = torch.full((triton.cdiv(n, 32), k), 127, dtype=torch.uint8, device=packed.q.device)
        _shuffle_scale[(triton.cdiv(n * (k // 32), 256),)](packed.s, packed.shuffled, n, k, 256)
    return packed


@triton.jit
def _encode(x):
    a = tl.abs(x)
    c = ((a > .25).to(tl.int32) + (a >= .75).to(tl.int32)
         + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
         + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
         + (a > 5.).to(tl.int32))
    return (c | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _magnitude(q):
    c = (q & 7).to(tl.int32)
    return tl.where(c < 2, c * .5, tl.exp2((c // 2 - 1).to(tl.float32)) * (1. + .5 * (c % 2)))


@triton.jit
def _hadamard32(x, GROUPS: tl.constexpr):
    c = tl.arange(0, 32)
    for step in tl.static_range(5):
        stride = 1 << step
        partner = tl.gather(x, tl.broadcast_to((c ^ stride)[None, :], (GROUPS, 32)), 1)
        x = tl.where((c[None, :] & stride) == 0, x + partner, partner - x)
    return x * 0.1767766952966369


@triton.jit
def _quant(X, Q, S, SF, LIVE, ROWS: tl.constexpr, K: tl.constexpr,
           SX: tl.constexpr, GROUPS: tl.constexpr, MSE: tl.constexpr, ROTATE: tl.constexpr, SHUFFLE: tl.constexpr):
    g = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    r, b = g // (K // 32), g % (K // 32)
    j = tl.arange(0, 16)
    active = (r < ROWS) & (r < tl.load(LIVE))
    off = r[:, None] * SX + b[:, None] * 32 + 2 * j[None, :]
    x0 = tl.load(X + off, active[:, None], 0).to(tl.float32)
    x1 = tl.load(X + off + 1, active[:, None], 0).to(tl.float32)
    if ROTATE:
        rotated = _hadamard32(tl.interleave(x0, x1), GROUPS)
        even = tl.broadcast_to((2 * j)[None, :], (GROUPS, 16))
        x0, x1 = tl.gather(rotated, even, 1), tl.gather(rotated, even + 1, 1)
    nonfinite = tl.sum(((tl.abs(x0) == float('inf')) | (x0 != x0)
                       | (tl.abs(x1) == float('inf')) | (x1 != x1)).to(tl.int32), 1) > 0
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    e = tl.ceil(tl.log2(tl.maximum(amax / 6., 2.**-126)))
    e = tl.where(amax == 0, 0., tl.minimum(tl.maximum(e, -126.), 127.))
    inv = tl.exp2(-e)
    v0, v1 = x0 * inv[:, None], x1 * inv[:, None]
    q0, q1 = _encode(v0), _encode(v1)
    if MSE:
        lo0, lo1 = _encode(v0 * 2.), _encode(v1 * 2.)
        hi0, hi1 = tl.abs(v0) - _magnitude(q0), tl.abs(v1) - _magnitude(q1)
        low0, low1 = tl.abs(v0) - .5 * _magnitude(lo0), tl.abs(v1) - .5 * _magnitude(lo1)
        err_hi = tl.sum(hi0 * hi0 + hi1 * hi1, 1)
        err_lo = tl.sum(low0 * low0 + low1 * low1, 1)
        lower = (err_lo < err_hi) & (e > -126.)
        e = tl.where(lower, e - 1., e)
        q0, q1 = tl.where(lower[:, None], lo0, q0), tl.where(lower[:, None], lo1, q1)
    q = q0 | (q1 << 4)
    q = tl.where(nonfinite[:, None], 0, q)
    s = tl.where(nonfinite, 255, e.to(tl.int32) + 127).to(tl.uint8)
    tl.store(Q + g[:, None] * 16 + j[None, :], q, r[:, None] < ROWS)
    tl.store(S + g, s, r < ROWS)
    if SHUFFLE:
        tl.store(SF + _scale_index(r, b, K), s, r < ROWS)


def quantize_into(x, packed, live, scaling='ceil', rotate=False):
    if x.ndim != 2 or x.stride(1) != 1 or tuple(x.shape) != packed.shape:
        raise ValueError('MXFP4 input must have matching [rows, K] shape and contiguous K')
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('MXFP4 quantization requires floating-point input')
    if scaling not in ('ceil', 'mse'):
        raise ValueError('MXFP4 scaling must be ceil or mse')
    _quant[(triton.cdiv(x.numel() // 32, 16),)](
        x, packed.q, packed.s, packed.shuffled if packed.shuffled is not None else packed.s,
        live, x.shape[0], x.shape[1], x.stride(0), 16, scaling == 'mse', rotate, packed.shuffled is not None)
    return packed


def quantize(x, scaling='ceil', rotate=False):
    out = empty(*x.shape, x.device)
    live = torch.tensor(x.shape[0], dtype=torch.int32, device=x.device)
    return quantize_into(x, out, live, scaling, rotate)


@triton.jit
def _load_shuffled_scale(SF, pid, block, ROWS: tl.constexpr, K: tl.constexpr,
                         TILE: tl.constexpr, BK: tl.constexpr):
    groups = pid * (TILE // 32) + tl.arange(0, TILE // 32)
    offset = tl.arange(0, BK)
    raw = tl.load(SF + groups[:, None] * K + block * BK + offset[None, :],
                  (groups[:, None] * 32 < ROWS) & (block * BK + offset[None, :] < K), 127)
    # Undo the physical lane packing; dot_scaled sees logical per-row scales.
    raw = raw.reshape(TILE // 32, BK // 256, 4, 16, 2, 2)
    return raw.permute(0, 5, 3, 1, 4, 2).reshape(TILE, BK // 32)


@triton.jit
def _gemm(A, AS, W, WS, Y, LIVE, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          LD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SHUFFLED: tl.constexpr,
          GROUP_N: tl.constexpr):
    pid = tl.program_id(0)
    mt, nt = tl.cdiv(M, BM), tl.cdiv(N, BN)
    if GROUP_N:
        start_n = (pid // (mt * GROUP_N)) * GROUP_N
        width = tl.minimum(GROUP_N, nt - start_n)
        local = pid % (mt * GROUP_N)
        pm, pn = local // width, start_n + local % width
    else:
        pm, pn = pid % mt, pid // mt
    r = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    live = tl.load(LIVE)
    if pm * BM >= live:
        return
    k = tl.arange(0, BK // 2)
    sk = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for b in range(tl.cdiv(K, BK)):
        kk, ss = b * (BK // 2) + k, b * (BK // 32) + sk
        a = tl.load(A + r[:, None] * (K // 2) + kk[None, :],
                    (r[:, None] < live) & (kk[None, :] < K // 2), 0)
        w = tl.load(W + n[:, None] * (K // 2) + kk[None, :],
                    (n[:, None] < N) & (kk[None, :] < K // 2), 0)
        if SHUFFLED:
            sa = _load_shuffled_scale(AS, pm, b, M, K, BM, BK)
            sw = _load_shuffled_scale(WS, pn, b, N, K, BN, BK)
        else:
            sa = tl.load(AS + r[:, None] * (K // 32) + ss[None, :],
                         (r[:, None] < live) & (ss[None, :] < K // 32), 127)
            sw = tl.load(WS + n[:, None] * (K // 32) + ss[None, :],
                         (n[:, None] < N) & (ss[None, :] < K // 32), 127)
        acc = tl.dot_scaled(a, sa, 'e2m1', tl.trans(w), sw, 'e2m1', acc)
    tl.store(Y + r[:, None] * LD + n[None, :], acc,
             (r[:, None] < live) & (n[None, :] < N))


@functools.lru_cache(maxsize=1)
def _table():
    path = Path(__file__).with_name('mxfp4_table_gfx950.json')
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    expected = dict(torch=str(torch.__version__), hip=torch.version.hip, triton=triton.__version__,
                    kernel_sha256=kernel_identity(),
                    gpu=torch.cuda.get_device_name())
    if not data.get('qualified') or any(data.get(k) != v for k, v in expected.items()):
        return {}
    return {tuple(row['shape']): tuple(row['config']) for row in data['entries']}


def kernel_identity():
    return hashlib.sha256(''.join(fn.src for fn in (_gemm, _load_shuffled_scale, _scale_index)).encode()).hexdigest()


def gemm(a, w, out, live, cfg=None):
    m, k = a.shape
    n, wk = w.shape
    if wk != k or out.shape != (m, n) or out.stride(1) != 1:
        raise ValueError('MXFP4 product/output dimensions or output strides do not match')
    if not torch.version.hip or not torch.cuda.get_device_properties(out.device).gcnArchName.startswith('gfx950'):
        raise ValueError('Native MXFP4 requires ROCm gfx950')
    if cfg is None:
        mode = os.environ.get('STARTLUX_PRISM_MXFP4_GEMM', 'tuned')
        if mode not in ('tuned', 'reference'):
            raise ValueError('STARTLUX_PRISM_MXFP4_GEMM must be tuned or reference')
        cfg = _table().get((m, n, k)) if mode == 'tuned' else None
        cfg = cfg or (32, 128, 256, 4, 2)
    bm, bn, bk, warps, stages = cfg[:5]
    instruction = cfg[5] if len(cfg) >= 6 else 16
    group_n = cfg[6] if len(cfg) >= 7 else 0
    layout = os.environ.get('STARTLUX_PRISM_MXFP4_SCALE_LAYOUT', 'tiled')
    if layout not in ('tiled', 'rowmajor'):
        raise ValueError('STARTLUX_PRISM_MXFP4_SCALE_LAYOUT must be tiled or rowmajor')
    shuffled = (layout == 'tiled' and a.shuffled is not None and w.shuffled is not None
                and bm >= 32 and bn >= 32 and bk % 256 == 0)
    return _gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a.q, a.shuffled if shuffled else a.s, w.q, w.shuffled if shuffled else w.s, out, live,
        m, n, k, out.stride(0), bm, bn, bk, shuffled, group_n,
        num_warps=warps, num_stages=stages, matrix_instr_nonkdim=instruction)
