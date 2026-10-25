"""Model-specific reference for loading and using the smoothed MXFP4 side file.

This is not imported by the engine.  It is the engine-side part of the scheme, extracted so the
quantization and kernel can be wired in without guessing where the scales go:

  - load a side file written by `tools/quantize_mxfp4.py`;
  - fold `1/s` into the preceding norm: `post_norm / s_gu`, `attn_norm / s_qkv` (attention) or
    `attn_norm / s_inp` (GDN);
  - keep `1/s` for the bf16 intermediate of `o`/`out`;
  - quantize the normalized activation and run the MXFP4 `gu`/`down` pair.

The production engine also keeps int8 copies for the thought decoder, uses CUDA graphs, and
routes attention/GDN through the same scales; those paths stay in the engine.  This module is the
minimal reference for the scale handling and one MLP block.
"""
from __future__ import annotations

import torch

from startlux_prism.kernels import mxfp4


def load_side(path, device="cuda"):
    """Load a side file; return (blob, weight(i, group), scale(i, group, n))."""
    blob = torch.load(path, map_location="cpu", weights_only=False)

    def weight(i, group):
        q, s = blob[f"{i}.{group}"]
        return mxfp4.MX4(q.to(device), s.to(device), q.shape[0], q.shape[1] * 2)

    def scale(i, group, n):
        t = blob.get(f"{i}.s_{group}")
        return t.to(device) if t is not None else torch.ones(n, device=device)

    return blob, weight, scale


def fold_norm(norm, scale):
    """A norm whose input projection uses `W * s`: divide the norm by s."""
    return norm / scale


def output_inv(scale):
    """`o`/`out` use `W * s` and divide their bf16 intermediate by s."""
    return 1.0 / scale


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def mlp(x, post_norm_fp4, gu, down, eps=1e-6):
    """One reference MLP block: x + down(silu(gu(rmsnorm(x, post_norm_fp4)))).

    x [M, H], post_norm_fp4 [H] (post_norm / s_gu), gu MX4 [2F, H], down MX4 [H, F].
    """
    h = rms_norm(x, post_norm_fp4, eps)
    M, K = h.shape
    aq = torch.empty(M, K // 2, dtype=torch.uint8, device=h.device)
    as_ = torch.empty(M, K // 32, dtype=torch.uint8, device=h.device)
    mxfp4.quantize_act(h, aq, as_)

    g = torch.empty(M, gu.n, dtype=torch.bfloat16, device=h.device)
    mxfp4.gemm(aq, as_, gu, g, M, epi=0)

    f = gu.n // 2
    act = torch.nn.functional.silu(g[:, :f]) * g[:, f:]
    fq = torch.empty(M, f // 2, dtype=torch.uint8, device=h.device)
    fs = torch.empty(M, f // 32, dtype=torch.uint8, device=h.device)
    mxfp4.quantize_act(act, fq, fs)

    out = x.float().clone()
    mxfp4.gemm(fq, fs, down, out, M, epi=1)
    return out.to(x.dtype)
