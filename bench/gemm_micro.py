"""Projection GEMMs of StartLux-Decision-4B on this GPU, by precision: what each library path achieves at the token
counts decision requests have.  Decides the engine's GEMM design.

    python bench/gemm_micro.py [--m 16,64,128,256,512,1024,2048,4096] [--iters 50] [--json out.json]

Paths: bf16 (cuBLAS, fp32 accumulate), fp16 with fp16 accumulate (cuBLAS, half precision reduction allowed), int8
(torch._int_mm: cuBLASLt int8 x int8 -> int32), fp8 e4m3 rowwise scales (torch._scaled_mm).  Each GEMM is timed with
CUDA graphs of `--iters` back-to-back calls, so launch overhead is excluded; TFLOPS = 2*M*N*K / time.  The weight
stream alone (N*K bytes per call at the precision) bounds small M: GB/s is reported too.
"""
import argparse
import json

import torch

def model_shapes(hidden=2560, ffn=9216, k_heads=16, v_heads=32, gdn_dim=128, n_head=16, n_kv=4, head_dim=256):
    """name: (N, K) for y[M, N] = x[M, K] W[N, K]^T, from a qwen35 model's sizes (default StartLux-Decision-4B; the 9B:
    hidden 4096, ffn 12288; the 27B: hidden 5120, ffn 17408, 48 value heads, 24 attention heads)."""
    value = v_heads * gdn_dim
    return {
        "gdn_in (qkv|z|b|a)": (2 * k_heads * gdn_dim + 2 * value + 2 * v_heads, hidden),
        "gdn_out": (hidden, value),
        "attn_qkv": (2 * n_head * head_dim + 2 * n_kv * head_dim, hidden),
        "attn_o": (hidden, n_head * head_dim),
        "mlp_gate_up": (2 * ffn, hidden),
        "mlp_down": (hidden, ffn),
        "attn_q_last": (2 * n_head * head_dim, hidden),     # last layer, read rows only: queries | gates
        "attn_kv_last": (2 * n_kv * head_dim, hidden),      # last layer, every token: keys | values
    }


def gguf_dims(path):
    """model_shapes() keyword arguments from a qwen35 GGUF file's metadata."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from flash_jev.gguf import GGUFFile
    m = GGUFFile(path).meta
    g = lambda k: int(m[f"qwen35.{k}"])                     # noqa: E731
    return dict(hidden=g("embedding_length"), ffn=g("feed_forward_length"), k_heads=g("ssm.group_count"),
                v_heads=g("ssm.time_step_rank"), gdn_dim=g("ssm.state_size"), n_head=g("attention.head_count"),
                n_kv=g("attention.head_count_kv"), head_dim=g("attention.key_length"))


SHAPES = model_shapes()                 # StartLux-Decision-4B


def timed(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / iters)
    return best  # ms per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="16,64,128,256,512,1024,2048,4096")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    dev = "cuda"
    print(torch.cuda.get_device_name(), "torch", torch.__version__, "cuda", torch.version.cuda)
    out = []
    for name, (N, K) in SHAPES.items():
        wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wh = wb.half()
        wi = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        w8 = (torch.randn(N, K, device=dev) * 0.02).to(torch.float8_e4m3fn)
        sw = torch.rand(1, N, device=dev, dtype=torch.float32)
        for M in [int(x) for x in a.m.split(",")]:
            xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xh = xb.half()
            xi = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
            x8 = xb.to(torch.float8_e4m3fn)
            sx = torch.rand(M, 1, device=dev, dtype=torch.float32)
            yb = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            row = {"shape": name, "M": M, "N": N, "K": K}
            paths = {
                "bf16": (lambda: torch.mm(xb, wb.t(), out=yb), 2),
                "fp16acc16": (lambda: torch.mm(xh, wh.t()), 2),
                "int8": (lambda: torch._int_mm(xi, wi.t()), 1),
                "fp8": (lambda: torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16), 1),
            }
            for p, (fn, wbytes) in paths.items():
                try:
                    if p == "int8" and M < 32:            # cuBLASLt int8 wants M > 16
                        xi2 = torch.randint(-127, 128, (32, K), device=dev, dtype=torch.int8)
                        fn = (lambda xi2=xi2: torch._int_mm(xi2, wi.t()))
                    ms = timed(fn, a.iters)
                    row[p] = {"us": round(ms * 1000, 2), "tflops": round(2 * M * N * K / ms / 1e9, 1),
                              "weight_gbps": round(N * K * wbytes / ms / 1e6, 1)}
                except Exception as e:  # noqa: BLE001  a path the build lacks is reported, not fatal
                    row[p] = {"error": str(e).splitlines()[0][:120]}
            out.append(row)
            cells = []
            for p in ("bf16", "fp16acc16", "int8", "fp8"):
                r = row[p]
                cells.append(f"{p} {r['us']:8.1f}us {r['tflops']:6.1f}T" if "us" in r else f"{p} ERR")
            print(f"{name:20s} M={M:5d}  " + "  ".join(cells), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
