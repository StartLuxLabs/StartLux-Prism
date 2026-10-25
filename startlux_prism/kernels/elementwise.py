"""Row-wise kernels around the projections: embedding gather (Q8_0 rows dequantized), residual add + RMSNorm,
SwiGLU.  The residual stream stays float32; normalized rows go to the projections as bf16."""
import torch
import triton
import triton.language as tl


@triton.jit
def _embed_kernel(ids_ptr, qs_ptr, d_ptr, h_ptr, n_tok_ptr, HID: tl.constexpr, HP: tl.constexpr, QK: tl.constexpr):
    """Rows of the residual stream from the token embeddings.  An id below 0 (an image token, engine/batch.py) keeps
    its row: the vision tower's features were written there before the pass."""
    t = tl.program_id(0).to(tl.int64)
    if t >= tl.load(n_tok_ptr):
        return
    tok = tl.load(ids_ptr + t).to(tl.int64)
    if tok < 0:
        return
    c = tl.arange(0, HP)
    m = c < HID
    q = tl.load(qs_ptr + tok * HID + c, mask=m, other=0).to(tl.float32)
    s = tl.load(d_ptr + tok * (HID // QK) + c // QK, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + t * HID + c, q * s, mask=m)


@triton.jit
def _add_norm_kernel(h_ptr, delta_ptr, w_ptr, out_ptr, rows_ptr, n_ptr, eps,
                     HID: tl.constexpr, HP: tl.constexpr, ADD: tl.constexpr, GATHER: tl.constexpr):
    """h[r] += delta[i] (ADD); out[i] = rmsnorm(h[r]) * w as bf16; r = rows[i] (GATHER) or i."""
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
    if ADD:
        h = h + tl.load(delta_ptr + i * HID + c, mask=m, other=0.0).to(tl.float32)
        tl.store(h_ptr + r * HID + c, h, mask=m)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    tl.store(out_ptr + i * HID + c, y.to(tl.bfloat16), mask=m)


@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, n_ptr, FF: tl.constexpr, BC: tl.constexpr):
    """gu [T, 2*FF] (gate | up) -> silu(gate) * up as bf16 [T, FF]."""
    t = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    if t >= tl.load(n_ptr):
        return
    c = cb * BC + tl.arange(0, BC)
    m = c < FF
    g = tl.load(gu_ptr + t * 2 * FF + c, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + t * 2 * FF + FF + c, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * FF + c, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=m)


@triton.jit
def _gather_kernel(src_ptr, rows_ptr, out_ptr, n_ptr, W: tl.constexpr, WP: tl.constexpr):
    """out[i] = src[rows[i]] (row width W)."""
    i = tl.program_id(0).to(tl.int64)
    if i >= tl.load(n_ptr):
        return
    r = tl.load(rows_ptr + i).to(tl.int64)
    c = tl.arange(0, WP)
    m = c < W
    tl.store(out_ptr + i * W + c, tl.load(src_ptr + r * W + c, mask=m), mask=m)


@triton.jit
def _readout_kernel(h_ptr, norm_ptr, let_ptr, out_ptr, n_ptr, eps, HID: tl.constexpr, NL: tl.constexpr,
                    NLP: tl.constexpr, BK: tl.constexpr):
    """One read row: final RMSNorm, then the option letters' logits (rows of the tied embedding), in a fixed order
    of operations - the same bits for a row whatever the number of rows read with it."""
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    k = tl.arange(0, BK)
    ss = tl.zeros([BK], dtype=tl.float32)
    for k0 in tl.static_range(0, HID, BK):
        x = tl.load(h_ptr + r * HID + k0 + k).to(tl.float32)
        ss += x * x
    inv = tl.rsqrt(tl.sum(ss, 0) / HID + eps)
    j = tl.arange(0, NLP)
    jm = j < NL
    acc = tl.zeros([NLP, BK], dtype=tl.float32)
    for k0 in tl.static_range(0, HID, BK):
        y = tl.load(h_ptr + r * HID + k0 + k).to(tl.float32) * inv * tl.load(norm_ptr + k0 + k).to(tl.float32)
        acc += tl.load(let_ptr + j[:, None] * HID + k0 + k[None, :], mask=jm[:, None], other=0.0) * y[None, :]
    tl.store(out_ptr + r * NL + j, tl.sum(acc, 1), mask=jm)


def _p2(n):
    return 1 << (n - 1).bit_length()


def readout(h, norm, letters, out, n, R, eps):
    """h [R, HID] fp32 residual rows -> out [R, NL] fp32 letter logits (letters [NL, HID] fp32)."""
    HID, NL = h.shape[1], letters.shape[0]
    BK = 256
    assert HID % BK == 0 and h.is_contiguous() and letters.is_contiguous() and letters.dtype == torch.float32
    _readout_kernel[(R,)](h, norm, letters, out, n, eps, HID=HID, NL=NL, NLP=_p2(NL), BK=BK, num_warps=4)


def embed(ids, embed_q8, h, n_tok, T):
    if hasattr(embed_q8, "kind"):
        from .amd_w4a16 import embed as compressed_embed
        return compressed_embed(ids, embed_q8, h, n_tok, T)
    HID = h.shape[1]
    _embed_kernel[(T,)](ids, embed_q8.qs, embed_q8.d, h, n_tok, HID=HID, HP=_p2(HID), QK=32, num_warps=4)


def add_norm(h, delta, w, out, n, T, eps, rows=None, add=True):
    HID = h.shape[1]
    _add_norm_kernel[(T,)](h, delta if delta is not None else h, w, out, rows if rows is not None else n, n, eps,
                           HID=HID, HP=_p2(HID), ADD=add, GATHER=rows is not None, num_warps=8)


def swiglu(gu, out, n, T):
    FF = out.shape[1]
    BC = 2048
    _swiglu_kernel[(T, triton.cdiv(FF, BC))](gu, out, n, FF=FF, BC=BC, num_warps=4)


def gather_rows(src, rows, out, n, R):
    W = src.shape[1]
    _gather_kernel[(R,)](src, rows, out, n, W=W, WP=_p2(W), num_warps=4)
