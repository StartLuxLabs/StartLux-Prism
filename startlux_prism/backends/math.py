"""Explicit float32 arithmetic: preserve rounding at each W4A8 boundary on HIP."""

import torch
import triton
import triton.language as tl

__all__ = ["add_rn", "mul_rn", "div_rn", "rint", "fma"]

if torch.version.hip is None:
    from triton.language.extra.cuda.libdevice import add_rn, mul_rn, div_rn, rint, fma
else:
    from triton.language.extra.hip.libdevice import rint

    @triton.jit
    def add_rn(x, y):
        return tl.inline_asm_elementwise(
            "v_add_f32 $0, $1, $2", constraints="=v,v,v", args=[x, y], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def mul_rn(x, y):
        return tl.inline_asm_elementwise(
            "v_mul_f32 $0, $1, $2", constraints="=v,v,v", args=[x, y], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def div_rn(x, y):
        return tl.div_rn(x, y)

    @triton.jit
    def fma(x, y, z):
        return tl.fma(x, y, z)
