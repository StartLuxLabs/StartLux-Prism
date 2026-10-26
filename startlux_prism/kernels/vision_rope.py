"""The vision tower's 2-D RoPE in one kernel (vision.py): queries and keys out of the fused qkv projection, rotated in
fp32 and rounded to bf16, with transformers' arithmetic (apply_rotary_pos_emb_vision):

    y = (x * cos) + (rotate_half(x) * sin),   rotate_half(x) = cat(-x[half:], x[:half])

each product and the sum rounded separately (no fused multiply-add), so the result has the bits of the eager PyTorch
ops it replaces - fourteen elementwise kernels per block, which at 1,024 image tokens cost 5-6 ms of the tower's 19 ms
on an H200, and half its kernel launches.

    qkv [n, 3 * H * D] bf16 (query heads, key heads, value heads);  cos, sin [n, D] fp32
    -> q, k [n, H, D] bf16 (contiguous: the layout the eager code hands to scaled_dot_product_attention)
"""
import triton
import triton.language as tl
from ..backends import math as libdevice


@triton.jit
def _rope_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, n, H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
    """One program per (token, query or key): every head of it, [H, D] (D padded to DP)."""
    t = tl.program_id(0).to(tl.int64)
    kind = tl.program_id(1)                       # 0 query, 1 key
    h = tl.arange(0, H)
    d = tl.arange(0, DP)
    m = (d < D)[None, :]
    half: tl.constexpr = D // 2
    src = qkv_ptr + t * (3 * H * D) + kind * (H * D) + h[:, None] * D
    x = tl.load(src + d[None, :], mask=m, other=0.0).to(tl.float32)
    partner = tl.where(d < half, d + half, d - half)
    xp = tl.load(src + partner[None, :], mask=m, other=0.0).to(tl.float32)
    xp = tl.where((d < half)[None, :], -xp, xp)
    c = tl.load(cos_ptr + t * D + d, mask=d < D, other=0.0)[None, :]
    s = tl.load(sin_ptr + t * D + d, mask=d < D, other=0.0)[None, :]
    y = libdevice.add_rn(libdevice.mul_rn(x, c), libdevice.mul_rn(xp, s))
    dst = tl.where(kind == 0, q_ptr, k_ptr) + t * (H * D) + h[:, None] * D + d[None, :]
    tl.store(dst, y.to(tl.bfloat16), mask=m)


def rope_qk(qkv, cos, sin, q, k, heads):
    """qkv [n, 3 * heads * D] bf16, cos / sin [n, D] fp32 -> q, k [n, heads, D] bf16 (written in place)."""
    n, D = cos.shape
    assert qkv.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and heads & (heads - 1) == 0
    _rope_kernel[(n, 2)](qkv, cos, sin, q, k, n, H=heads, D=D, DP=triton.next_power_of_2(D), num_warps=4)
