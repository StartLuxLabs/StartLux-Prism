"""W4A8 GEMMs (kernels/w4a8.py) on a Q4_K_M file's projection shapes against the W8A8 path (kernels/w8a8.py with its
dispatch table) at the same token counts: microseconds per GEMM (CUDA graphs of back-to-back calls), with weights
streamed from DRAM (enough weight copies to exceed L2).

    python bench/w4a8_micro.py MODEL-Q4_K_M.gguf [--m 1,16,64,128,256,512,1024,2048,4096] [--search]

--search times every CUDA tile config and stage count (csrc/w4a8_gemm.cu) and reports the fastest; otherwise
w4a8.pick_cuda's choice.  --json writes the rows (with --search, every config's time).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism import gguf_weights as GW  # noqa: E402
from startlux_prism.gguf import GGUFFile  # noqa: E402
from startlux_prism.kernels import w4a8 as W  # noqa: E402
from startlux_prism.kernels import w8a8 as KW  # noqa: E402

TENSORS = ("blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.4.ffn_down.weight", "blk.0.ssm_out.weight",
           "blk.0.attn_qkv.weight", "blk.4.attn_qkv.weight")


def timed(fn, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--m", default="1,16,64,128,256,512,1024,2048,4096")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    f = GGUFFile(a.gguf)
    rows = []
    for name in TENSORS:
        if name not in f.tensors:
            continue
        kq = GW._kq(f, name).to("cuda")
        ws = [W.repack(kq) for _ in range(a.copies)]
        N, K = ws[0].n, ws[0].k
        w8 = [torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda") for _ in range(a.copies)]
        sw = torch.rand(N, 1, device="cuda") * 1e-3
        for M in [int(x) for x in a.m.split(",")]:
            x = torch.randn(M, K, device="cuda")
            xq, xa, xs = W.quantize_ref(x)
            n = torch.tensor([M], dtype=torch.int32, device="cuda")
            out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            row = f"{ws[0].kind} {name:22s} [{N:5d}, {K:5d}] M={M:5d}"
            cands = [(c, st) for c in W.CUDA_CFGS for st in (2, 3, 4)] if a.search else [None]
            best, times = None, {}
            for cfg in cands:
                def run4(cfg=cfg):
                    for w in ws:
                        W.gemm(xq, xa, xs, w, out, n, cfg=cfg)
                try:
                    t4 = timed(run4) / len(ws)
                except RuntimeError:          # stages beyond the shared memory
                    continue
                if cfg is not None:
                    times[f"{cfg[0]},{cfg[1]}"] = round(t4, 2)
                if best is None or t4 < best[0]:
                    best = (t4, cfg or W.pick_cuda(M, N, ws[0].kind, K))
            default = W.pick_cuda(M, N, ws[0].kind, K)
            row += f"  w4a8 {best[0]:8.1f} us {list(best[1])}"
            a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
            sa = torch.rand(M, 1, device="cuda") * 1e-2

            def run8():
                for w in w8:
                    KW.gemm(a8, sa, w, sw, out, n, K)
            t8 = timed(run8) / len(w8)
            row += f"  w8a8 {t8:8.1f} us  ratio {best[0] / t8:5.2f}"
            rows.append({"tensor": name, "kind": ws[0].kind, "N": N, "K": K, "M": M, "w4a8_us": round(best[0], 2),
                         "cfg": list(best[1]), "default": list(default), "w8a8_us": round(t8, 2), "times": times})
            print(row, flush=True)
    if a.json:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
