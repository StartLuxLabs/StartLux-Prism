"""HIP arithmetic and quantized projections against independent CPU reference arithmetic."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
import triton.language as tl
from startlux_prism.backends import math as M
from startlux_prism import gguf_weights as GW
from startlux_prism.gguf import GGML_Q4_K, GGML_Q6_K
from startlux_prism.kernels import w4a8 as W


@triton.jit
def arithmetic(x, y, z, out, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    a = tl.load(x + i, i < N, 0)
    b = tl.load(y + i, i < N, 1)
    c = tl.load(z + i, i < N, 0)
    tl.store(out + i, M.add_rn(a, b), i < N)
    tl.store(out + N + i, M.mul_rn(a, b), i < N)
    tl.store(out + 2 * N + i, M.div_rn(a, b), i < N)
    tl.store(out + 3 * N + i, M.fma(a, b, c), i < N)
    tl.store(out + 4 * N + i, M.rint(a), i < N)


def synthetic(kind, n, k):
    g = torch.Generator().manual_seed(71 + kind + n + k)
    raw = torch.randint(0, 256, (n, k // 256, 144 if kind == GGML_Q4_K else 210), dtype=torch.uint8, generator=g)
    scales = (torch.rand(n, k // 256, generator=g) * 0.01 + 0.001).half()
    if kind == GGML_Q4_K:
        raw[..., :2] = scales.view(torch.uint8).reshape(n, k // 256, 2)
        raw[..., 2:4] = (scales * 0.3).half().view(torch.uint8).reshape(n, k // 256, 2)
    else:
        raw[..., 208:210] = scales.view(torch.uint8).reshape(n, k // 256, 2)
    return GW.KQuant(raw, kind, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--math-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(8)
    if not a.cpu:
        assert torch.version.hip and torch.cuda.device_count() == 1
        g = torch.Generator().manual_seed(1971)
        x = torch.randn(1024, generator=g) * 100
        x[:8] = torch.tensor([0.0, -0.0, 0.5, 1.5, 2.5, -1.5, 127.0, -127.0])
        y = torch.randn(1024, generator=g).abs() + 0.001
        z = torch.randn(1024, generator=g)
        out = torch.empty(5, len(x), device="cuda")
        arithmetic[(triton.cdiv(len(x), 128),)](x.cuda(), y.cuda(), z.cuda(), out, len(x), 128)
        refs = [
            (x.double() + y.double()).float(),
            (x.double() * y.double()).float(),
            (x.double() / y.double()).float(),
            W.fma32(x, y, z),
            x.round(),
        ]
        for name, got, ref in zip(["add_rn", "mul_rn", "div_rn", "fma", "rint"], out.cpu(), refs):
            assert torch.equal(got, ref), (name, (got - ref).abs().max().item(), int((got != ref).sum()))
            print(f"PASS {name}: {len(x)} values bit-identical to CPU reference", flush=True)
        if a.math_only:
            return
    for typ in [GGML_Q4_K, GGML_Q6_K]:
        for k in [256, 512, 5120]:
            kq = synthetic(typ, 33, k)
            wc = W.repack(kq)
            if a.cpu:
                import numpy as np
                from gguf.quants import dequantize
                from gguf.constants import GGMLQuantizationType

                oracle = dequantize(kq.raw.numpy().reshape(33, -1), GGMLQuantizationType(typ))
                assert np.array_equal(kq.dequant().numpy(), oracle)
                print(f"PASS CPU independent gguf-py dequant {wc.kind} K={k}", flush=True)
                continue
            wg = W.repack(kq.to("cuda"))
            for m, live in [(1, 1), (7, 5), (33, 31), (65, 65)]:
                x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + k))
                x[:, ::29] *= 20
                xq, scales, xs = W.quantize_ref(x)
                ref = W.gemm_ref(xq, scales, xs, wc)
                n = torch.tensor([live], dtype=torch.int32, device="cuda")
                xg, sg, xsg = xq.cuda(), scales.cuda(), xs.cuda()
                for epi in [0, 1]:
                    base = torch.randn(m, 33) if epi else torch.zeros(m, 33, dtype=torch.bfloat16)
                    got = base.cuda()
                    W.gemm(xg, sg, xsg, wg, got, n, epi=epi)
                    want = base.clone()
                    want[:live] = base[:live] + ref[:live] if epi else ref[:live].bfloat16()
                    actual = got.cpu()
                    assert torch.equal(actual, want), (
                        wc.kind,
                        k,
                        m,
                        epi,
                        (actual.float() - want.float()).abs().max().item(),
                    )
                print(f"PASS GPU {wc.kind} K={k} M={m} live={live}: both epilogues exact", flush=True)
    print("PASS AMD backend qualification", flush=True)


if __name__ == "__main__":
    main()
