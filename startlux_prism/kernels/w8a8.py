"""W8A8 projections: int8 x int8 tensor-core GEMMs with per-group scales along K, fused epilogues.

Weights come from Q8_0 (int8 values, one fp16 scale per 32 inputs).  With GK = 32 they are used exactly; with a
coarser GK they are requantized once at load (a group's scale is the largest of its Q8_0 block scales, values rounded
again).  Activations are quantized per (row, GK group) by the kernel that produces them, so a group's scale needs no
cross-program reduction and quantization costs no extra pass.  Within a group the products accumulate in int32
exactly; across groups in fp32 after scaling by s_a[row, g] * s_w[col, g].

Epilogues: 0 store bf16; 1 add into the fp32 residual stream (output projections); 2 SwiGLU: each BN-wide tile of the
gate|up weight holds BN/2 gate rows then the matching BN/2 up rows, the tile computes silu(gate) * up and writes it
int8 with one scale per row (group = BN/2 columns) - the down projection's input, already quantized; 3 store the raw
int32 sums (gemm_i32: a tensor-parallel shard's part of an output projection, added over the ranks before the scaling).

Batch invariance: with one scale group (GK == K, the engine's case) an output is (float(acc) * s_a) * s_w, each product
rounded, and the residual add rounds once more - written with explicit round-to-nearest intrinsics, exactly as the
CUDA kernel's epilogue and the deferred split-K fold (quant.norm_q8) compute it.  Integer sums are exact, so every
kernel, tile shape and split gives an element the same bits, and a request's numbers do not depend on its batch.
"""
import functools
import hashlib
import json
import math
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from ..backends import math as libdevice


@triton.jit
def _w8a8_kernel(a_ptr, as_ptr, w_ptr, ws_ptr, out_ptr, outs_ptr, n_rows_ptr, M, N, K, stride_om,
                 GK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                 EPI: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    width = GROUP_M * num_n
    first_m = (pid // width) * GROUP_M
    gm = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % gm
    pid_n = (pid % width) // gm
    live = tl.load(n_rows_ptr)
    if pid_m * BM >= live:
        return
    rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    nm = rn < N
    KS: tl.constexpr = GK // SPLIT_K                 # split-K slices one scale group (only with GK == K)
    a_p = a_ptr + rm[:, None].to(tl.int64) * K + pid_k * KS + rk[None, :]
    w_p = w_ptr + rn[None, :].to(tl.int64) * K + pid_k * KS + rk[:, None]
    if EPI == 3:        # the raw int32 sums over all of K (gemm_i32: a tensor-parallel shard's partial products)
        acc3 = tl.zeros([BM, BN], dtype=tl.int32)
        for kk in range(0, K, BK):
            a = tl.load(a_p, mask=(rm < M)[:, None], other=0)
            w = tl.load(w_p, mask=nm[None, :], other=0)
            acc3 = tl.dot(a, w, acc3)
            a_p += BK
            w_p += BK
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc3, mask=(rm < live)[:, None] & nm[None, :])
        return
    G = K // GK
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in range(G):
        acci = tl.zeros([BM, BN], dtype=tl.int32)
        for kk in range(0, KS, BK):
            a = tl.load(a_p, mask=(rm < M)[:, None], other=0)
            w = tl.load(w_p, mask=nm[None, :], other=0)
            acci = tl.dot(a, w, acci)
            a_p += BK
            w_p += BK
        sa = tl.load(as_ptr + rm * G + g, mask=rm < M, other=0.0)
        sw = tl.load(ws_ptr + rn * G + g, mask=nm, other=0.0)
        acc += libdevice.mul_rn(libdevice.mul_rn(acci.to(tl.float32), sa[:, None]), sw[None, :])
    rmask = rm < live
    if EPI == 0:
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16),
                 mask=rmask[:, None] & nm[None, :])
    elif EPI == 1:
        p = out_ptr + rm[:, None] * stride_om + rn[None, :]
        m2 = rmask[:, None] & nm[None, :]
        if SPLIT_K == 1:
            tl.store(p, libdevice.add_rn(tl.load(p, mask=m2, other=0.0), acc), mask=m2)
        else:
            tl.atomic_add(p, acc, mask=m2, sem="relaxed")
    else:
        HB: tl.constexpr = BN // 2
        t = tl.permute(tl.reshape(acc, [BM, 2, HB]), (0, 2, 1))
        gate, up = tl.split(t)
        act = gate / (1.0 + tl.exp(-gate)) * up
        amax = tl.max(tl.abs(act), 1)
        scale = tl.maximum(amax, 1e-10) / 127.0
        q = libdevice.rint(act / scale[:, None]).to(tl.int8)
        cols = pid_n * HB + tl.arange(0, HB)
        tl.store(out_ptr + rm[:, None] * stride_om + cols[None, :], q, mask=rmask[:, None])
        tl.store(outs_ptr + rm * num_n + pid_n, scale, mask=rmask)


@triton.jit
def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, n_rows_ptr, K: tl.constexpr, GK: tl.constexpr):
    """x [M, K] (any float) -> int8 [M, K] and fp32 scales [M, K / GK], symmetric per group."""
    r = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    if r >= tl.load(n_rows_ptr):
        return
    c = g * GK + tl.arange(0, GK)
    x = tl.load(x_ptr + r * K + c).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-10) / 127.0
    tl.store(q_ptr + r * K + c, libdevice.rint(x / scale).to(tl.int8))
    tl.store(s_ptr + r * (K // GK) + g, scale)


# Split-K with atomic adds makes the fp32 sums depend on arrival order: the same request could then get (very slightly)
# different probabilities from one run to the next, and a near-tie can flip.  Off unless STARTLUX_PRISM_NONDETERMINISTIC=1: the
# CUDA kernel splits K serially instead (the splits of a tile add in split order through semaphores) and stream-K
# reduces int32 partial sums; this Triton kernel does not split K.
NONDETERMINISTIC = os.environ.get("STARTLUX_PRISM_NONDETERMINISTIC") == "1"


@functools.lru_cache(None)
def _mi355x_table():
    """Only offline measured shapes on the exact qualified MI355X runtime."""
    from ..backends import architecture
    if torch.version.hip is None or architecture() != "gfx950":
        return {}
    path = os.path.join(os.path.dirname(__file__), "w8a8_table_gfx950.json")
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        data = json.load(file)
    runtime = dict(torch=str(torch.__version__), triton=triton.__version__, hip=torch.version.hip)
    identity = hashlib.sha256(_w8a8_kernel.src.encode()).hexdigest()
    if (data.get("schema") != 1 or data.get("architecture") != "gfx950" or data.get("runtime") != runtime
            or data.get("gpu") != torch.cuda.get_device_name()
            or data.get("kernel_identity") != identity):
        return {}
    return data["entries"]


def pick(M, N, K, epi=0):
    """(BM, BN, BK, GROUP_M, warps, stages, split_k), from cold-L2 searches on the RTX 5090 (bench/w8a8_micro.py
    --configs --cold): weights stream from DRAM as in a forward pass.  Eight warps throughout (four with 128-wide K
    tiles spill).  Up to a few hundred tokens a projection is bound by the weight stream, which needs programs on all
    170 SMs; the 2560-wide output projections, whose epilogue adds into the fp32 residual stream, split K with atomic
    adds for that when NONDETERMINISTIC."""
    if torch.version.hip is not None and os.environ.get("STARTLUX_PRISM_MI355X_GEMM") != "baseline":
        choice = _mi355x_table().get(f"{M},{N},{K},{epi}")
        if choice is not None:
            return tuple(choice)
    if N >= 8192:                                                    # qkv|z|b|a, q|k|v, gate|up
        if M <= 64:
            return (64, 64, 128, 8, 8, 3, 1)
        if M <= 256:
            return (64, 128, 64, 8, 8, 4, 1)
        return (128, 128, 64, 8, 8, 3, 1) if M <= 512 else (64, 128, 64, 8, 8, 3, 1)
    if epi == 1 and M <= 256:                                        # out, o, down projections
        bk = 64 if K <= 4096 else 128
        if M <= 64:
            return (64, 64, bk, 8, 8, 3, 4 if NONDETERMINISTIC else 1)
        return (64, 128, 64, 8, 8, 3, (4 if M <= 128 else 2) if NONDETERMINISTIC else 1)
    if K <= 4096:
        return (128, 64, 64, 8, 8, 3, 1)
    return (128, 128, 64, 8, 8, 3, 2 if (NONDETERMINISTIC and epi == 1 and M <= 512) else 1)


@functools.lru_cache(None)
def _table():
    """(N, K) -> {M: choice}: the fastest of this module's Triton kernel and the CUDA ones (csrc/w8a8_gemm.cu: one CTA
    per tile, or stream-K; on Hopper also csrc/w8a8_wgmma.cu) per projection shape and token count, measured with
    weights streamed from DRAM (bench/cuda_gemm_micro.py --table) on this GPU's model: kernels/gemm_table_<gpu
    tag>.json (rtx5090, h200; without one, the Triton kernel).  STARTLUX_PRISM_GEMM=triton turns it off, STARTLUX_PRISM_GEMM_TABLE names
    another table file."""
    from . import cuda_ext
    if torch.version.hip is not None or os.environ.get("STARTLUX_PRISM_GEMM") == "triton":
        return {}
    path = os.environ.get("STARTLUX_PRISM_GEMM_TABLE") or cuda_ext.table_path("gemm_table")
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))["table"]
    # "NxK": the projection's kernel; "NxKi": int32 sums out (gemm_i32, the tensor-parallel output projections)
    return {tuple(int(x) for x in k.rstrip("i").split("x")) + (("i",) if k.endswith("i") else ()):
            {int(m): c for m, c in v.items()} for k, v in raw.items()}


def _choice(N, K, M, i32=False):
    t = _table().get((N, K, "i") if i32 else (N, K))
    if not t:
        return None
    m = min(t, key=lambda x: abs(math.log(x) - math.log(max(M, 1))))     # nearest measured token count
    return t[m]


class Deferred(NamedTuple):
    """A residual add left to the next kernel that reads the residual (quant.norm_q8 or quant.fold): `n` int32 slots of
    partial sums, and the projection's activation and weight scales to apply to their total.  parts: the slots (int32
    [S, rows, N]) when they are not the reader's own `parts` buffer (tensor parallel: the all-reduced sums, or this
    rank's own with `sync`, a quant.Sync: the reader adds every rank's slots itself)."""
    n: int
    a_s: torch.Tensor
    w_s: torch.Tensor
    parts: torch.Tensor = None
    sync: object = None


def gemm(a, a_s, w, w_s, out, n_rows, gk, epi=0, outs=None, cfg=None, slots=None, pdl=False):
    """a int8 [M, K], a_s fp32 [M, K/gk], w int8 [N, K], w_s fp32 [N, K/gk] -> out (see module docstring).
    Returns a Deferred when an epi 1 product split K into `slots` (int32 [S, rows, N], see cuda_ext.gemm), else None:
    the caller hands it to the next kernel that reads the residual.  pdl: a CUDA kernel is launched to overlap the
    kernel before it (which must signal it, as quant's producers do); the Triton kernel ignores it."""
    M, K = a.shape
    N = w.shape[0]
    if cfg is None and gk == K and epi in (0, 1):
        c = _choice(N, K, M)
        if c and c[0] == "sk":
            from . import cuda_ext
            bm, bn = cuda_ext.CFGS[c[1]][:2]
            if -(-M // bm) * -(-N // bn) > cuda_ext.MAX_TILES:   # more tiles than stream-K's counters (a count beyond
                c = ["cuda", c[1], c[2], 1]                       # the table's, which takes its largest): the same tiles
                                                                  # one per CTA, the same bits
        if c and c[0] == "cuda":
            from . import cuda_ext
            split = c[3] if epi == 1 else 1
            if split > 1 and not NONDETERMINISTIC and slots is not None and slots.shape[1] >= M:
                cuda_ext.gemm(a, a_s, w, w_s, out, n_rows, epi=epi, cfg=c[1], stages=c[2], split=split, mode="defer",
                              slots=slots, pdl=pdl)
                return Deferred(split, a_s, w_s)
            cuda_ext.gemm(a, a_s, w, w_s, out, n_rows, epi=epi, cfg=c[1], stages=c[2], split=split,
                          mode="atomic" if NONDETERMINISTIC or split == 1 else "workspace", pdl=pdl)
            return None
        if c and c[0] == "sk":
            from . import cuda_ext
            cuda_ext.gemm_sk(a, a_s, w, w_s, out, n_rows, epi=epi, cfg=c[1], stages=c[2], ctas_per_sm=c[3],
                             sk_waves=c[4] if len(c) > 4 else cuda_ext.PURE_SK, pdl=pdl)
            return None
        if c and c[0] == "wg":
            from . import cuda_ext
            split = c[3] if len(c) > 3 and epi == 1 else 1
            if split > 1 and slots is not None and slots.shape[1] >= M:
                cuda_ext.gemm_wg(a, a_s, w, w_s, out, n_rows, epi=epi, cfg=c[1], stages=c[2], split=split, slots=slots)
                return Deferred(split, a_s, w_s)
            cuda_ext.gemm_wg(a, a_s, w, w_s, out, n_rows, epi=epi, cfg=c[1], stages=c[2])
            return None
    c = cfg or pick(M, N, K, epi)
    BM, BN, BK, GM, warps, stages = c[:6]
    split = c[6] if len(c) > 6 else 1
    if split > 1 and (epi != 1 or gk != K or (K // split) % BK or not NONDETERMINISTIC):
        split = 1
    BK = min(BK, gk // split)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), split)
    _w8a8_kernel[grid](a, a_s, w, w_s, out, outs if outs is not None else w_s, n_rows, M, N, K, out.stride(0),
                       GK=gk, BM=BM, BN=BN, BK=BK, GROUP_M=GM, EPI=epi, SPLIT_K=split, num_warps=warps,
                       num_stages=stages)
    return None


def gemm_i32(a, a_s, w, w_s, slots, n_rows, max_split=4, pdl=False):
    """The int32 products a w^T (a int8 [M, K], w int8 [N, K]) without the scales (a_s, w_s: the projection's, passed
    only as gemm's kernels take them): an output projection's partial sums on one tensor-parallel rank (K is its share
    of the inputs), to be added over the ranks - exact - and then scaled and added into the residual as the single-GPU
    epilogue does (Deferred, quant.norm_q8).  The kernel comes from the dispatch table's "NxKi" entries (int32-capable
    kernels: the wgmma TMA pipeline, the mma.sync tile kernel); a split over K stores split z's sums in slots[z]
    (int32 [S, rows, N]).  Returns the number of slots written (at most max_split; without an entry, the Triton
    kernel's one)."""
    M, K = a.shape
    N = w.shape[0]
    assert slots.dtype == torch.int32 and slots.shape[1] >= M and slots.shape[2] == N
    c = _choice(N, K, M, i32=True)
    if c and c[0] in ("wg", "cuda") and (c[3] if len(c) > 3 else 1) <= max_split:
        from . import cuda_ext
        split = c[3] if len(c) > 3 else 1
        if c[0] == "wg":
            cuda_ext.gemm_wg(a, a_s, w, w_s, slots[0][:M], n_rows, epi=1, cfg=c[1], stages=c[2], split=split,
                             slots=slots, i32=True)
        else:
            cuda_ext.gemm(a, a_s, w, w_s, slots[0][:M], n_rows, epi=1, cfg=c[1], stages=c[2], split=split,
                          mode="defer", slots=slots, pdl=pdl, i32=True)
        return split
    BM, BN, BK, GM, warps, stages = pick(M, N, K, 0)[:6]
    out = slots[0]
    _w8a8_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN), 1)](a, a_s, w, w_s, out, w_s, n_rows, M, N, K,
                                                               out.stride(0), GK=K, BM=BM, BN=BN, BK=BK, GROUP_M=GM,
                                                               EPI=3, SPLIT_K=1, num_warps=warps, num_stages=stages)
    return 1


def quant_rows(x, gk, n_rows=None):
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.int8, device=x.device)
    s = torch.empty(M, K // gk, dtype=torch.float32, device=x.device)
    if n_rows is None:
        n_rows = torch.tensor([M], dtype=torch.int32, device=x.device)
    _quant_rows_kernel[(M, K // gk)](x, q, s, n_rows, K=K, GK=gk, num_warps=4)
    return q, s


def weight_from_q8(q8, gk, smooth=None):
    """Q8_0 matrix -> (int8 [N, K], fp32 scales [N, K/gk]).  gk == 32 keeps the Q8_0 values; smooth (fp32 [K])
    multiplies input channel k by smooth[k] first (SmoothQuant) and forces requantization."""
    if gk == 32 and smooth is None:
        return q8.qs.contiguous(), q8.d.float().contiguous()
    w = q8.dequant(torch.float32)
    if smooth is not None:
        w = w * smooth[None, :]
    N, K = w.shape
    g = w.view(N, K // gk, gk)
    s = g.abs().amax(-1).clamp_min(1e-10) / 127.0
    q = torch.round(g / s[..., None]).clamp(-127, 127).to(torch.int8).view(N, K)
    return q.contiguous(), s.contiguous()


def interleave_gate_up(gate, up, bn):
    """Rows of gate [FF, K] and up [FF, K] -> [2 FF, K] with tiles of bn rows: bn/2 gate rows, then the same up rows."""
    FF, K = gate.shape
    h = bn // 2
    assert FF % h == 0
    return torch.stack([gate.view(FF // h, h, K), up.view(FF // h, h, K)], 1).reshape(2 * FF, K).contiguous()
