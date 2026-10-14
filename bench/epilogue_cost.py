"""What the CUDA GEMM's epilogue (scale + store / residual add) costs: the projection GEMMs with the dispatch table's
kernels, built normally and built with STARTLUX_PRISM_EPI_SKIP (the same kernels without their output writes), in two processes.

    python bench/epilogue_cost.py [--m 384,1280,4096]
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
import torch
from startlux_prism.kernels import cuda_ext as CX, w8a8 as KW
if sys.argv[2] == "skip":
    from torch.utils.cpp_extension import load
    load(name="startlux_prism_ext_epi_skip", sources=[os.path.join(CX.CSRC, f) for f in ("w8a8_gemm.cu", "l2_prefetch.cu")],
         is_python_module=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120", "-DSTARTLUX_PRISM_EPI_SKIP"],
         extra_cflags=["-O3"], verbose=False)
    CX.load.cache_clear()
    CX.load = lambda: torch.ops.startlux_prism
from gemm_micro import SHAPES, timed
EPI = {"gdn_out": 1, "attn_o": 1, "mlp_down": 1}
out = {}
for M in [int(x) for x in sys.argv[3].split(",")]:
    for name in ("gdn_in", "attn_qkv", "mlp_gate_up", "gdn_out", "mlp_down"):
        N, K = SHAPES[next(k for k in SHAPES if k.split(" ")[0] == name)]
        epi = EPI.get(name, 0)
        x = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
        sa = torch.rand(M, 1, device="cuda") * 1e-2
        ws = [torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8) for _ in range(4)]
        sw = torch.rand(N, 1, device="cuda") * 1e-3
        o = torch.zeros(M, N, device="cuda", dtype=torch.float32 if epi else torch.bfloat16)
        slots = torch.zeros(4, M, N, dtype=torch.int32, device="cuda") if epi else None
        n = torch.tensor([M], dtype=torch.int32, device="cuda")
        it = [0]
        def run():
            it[0] = (it[0] + 1) % 4
            KW.gemm(x, sa, ws[it[0]], sw, o, n, K, epi=epi, slots=slots)
        out[f"{name}@{M}"] = timed(run, 20) * 1000
print(json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="384,1280,4096")
    a = ap.parse_args()
    here = os.path.abspath(__file__)
    res = {}
    for mode in ("normal", "skip"):
        r = subprocess.run([sys.executable, "-c", CHILD, here, mode, a.m], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            sys.exit(1)
        res[mode] = json.loads(r.stdout.strip().splitlines()[-1])
    for k, t in res["normal"].items():
        s = res["skip"][k]
        print(f"{k:18s} with epilogue {t:8.1f} us   without writes {s:8.1f} us   epilogue {t - s:7.1f} us "
              f"({100 * (t - s) / t:4.1f}%)")


if __name__ == "__main__":
    main()
