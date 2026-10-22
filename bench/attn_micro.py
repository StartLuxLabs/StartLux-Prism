"""The tree attention kernel alone on one long segment (the shape of a long single-question prompt): time and TFLOPS for
query/key tile sizes, warps and pipeline stages, with the result checked against PyTorch's attention.

    python bench/attn_micro.py [--t 1185,3000] [--configs 32x32x8x2,64x32x8x2,...] [--modes bf16,q8]

Modes: bf16 scores (the default path) and q8 (int8 keys with one scale per token and kv head, queries quantized per
row in the kernel, q k^T on int8 tensor cores).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism.kernels import attn as KA  # noqa: E402
from gemm_micro import timed  # noqa: E402

H, HKV, HD = 16, 4, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="1185,3000")
    ap.add_argument("--configs", default="32x32x8x2,32x64x8x2,64x32x8x2,64x32x8x1,64x64x8x1,32x32x4x2,64x32x16x2")
    ap.add_argument("--modes", default="bf16,q8")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    for T in [int(x) for x in a.t.split(",")]:
        q = torch.randn(T, H, HD, device=dev, dtype=torch.bfloat16)
        qkv = torch.randn(T, H * 2 * HD, device=dev, dtype=torch.bfloat16)        # gates at h*2*HD + HD
        kv = torch.randn(1, 2, HKV, T, HD, device=dev, dtype=torch.bfloat16)
        out = torch.empty(T, H * HD, device=dev, dtype=torch.bfloat16)
        # reference: causal attention per head (GQA), times sigmoid(gate)
        kk = kv[0, 0].repeat_interleave(H // HKV, 0)                                 # [H, T, HD]
        vv = kv[0, 1].repeat_interleave(H // HKV, 0)
        ref = F.scaled_dot_product_attention(q.transpose(0, 1).float(), kk.float(), vv.float(), is_causal=True)
        gate = qkv.view(T, H, 2, HD)[:, :, 1].float()
        ref = (ref.transpose(0, 1) * torch.sigmoid(gate)).reshape(T, H * HD)
        flops = 4 * H * HD * T * (T + 1) / 2
        ranges = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int32, device=dev)
        kf = kv[0, 0].float()                                                       # int8 keys, scale per row
        ksc = (kf.abs().amax(-1).clamp_min(1e-10) / 127)[None]                      # [1, HKV, T]
        k8 = torch.round(kf / ksc[0][..., None]).clamp(-127, 127).to(torch.int8)[None].contiguous()
        for c, mode in [(c, m) for c in a.configs.split(",") for m in a.modes.split(",")]:
            bm, bn, warps, stages = (int(x) for x in c.split("x"))
            blocks = torch.tensor([(t0, min(bm, T - t0), 0, t0) for t0 in range(0, T, bm)], dtype=torch.int32,
                                  device=dev)
            nb = torch.tensor([blocks.shape[0]], dtype=torch.int32, device=dev)

            def f():
                KA._tree_attention_kernel[(blocks.shape[0], H)](
                    q, qkv, qkv.stride(0), kv, 0, kv.stride(1), kv.stride(2), out, blocks, nb, ranges,
                    k8, 0, k8.stride(1), ksc, 0, ksc.stride(1),
                    H=H, HKV=HKV, HD=HD, BM=bm, BN=bn, SCALE=HD ** -0.5, Q8=mode == "q8", num_warps=warps,
                    num_stages=stages)
            try:
                f()
                torch.cuda.synchronize()
                err = (out.float() - ref).abs().max().item()
                us = timed(f, 10) * 1000
            except Exception as e:                                      # noqa: BLE001  out of resources etc.
                print(f"T={T:5d} {c:12s} {mode:4s}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            print(f"T={T:5d} {c:12s} {mode:4s}: {us:8.1f} us  {flops / us / 1e6:6.1f} TFLOPS  max err {err:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
