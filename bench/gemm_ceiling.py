"""How far the large projections are from what the GPU sustains: int8 tensor-core peak (bench/csrc/mma_peak.cu,
register operands only) and the projection GEMMs at a long prompt's row count, each run for a few seconds with the SM
clock and board power sampled (NVML), so every rate is compared at the clock the GPU actually holds under that load.

    python bench/gemm_ceiling.py --mma-peak ./mma_peak [--m 4096] [--seconds 4]

Prints TOPS, mean SM clock and power for: the mma.sync peak; per projection shape, flash-jev's dispatch (the kernel
the engine runs for that row count, with its epilogue) and cuBLASLt int8 (torch._int_mm, int32 out, no epilogue).
Weights rotate over 4 copies (as in a forward pass, where every projection's weights come from DRAM).
"""
import argparse
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev.kernels import w8a8 as KW  # noqa: E402
from gemm_micro import SHAPES  # noqa: E402
from run_http import GPUSampler  # noqa: E402

EPI = {"gdn_out": 1, "attn_o": 1, "mlp_down": 1}


def sustained(fn, seconds, ops_per_call, gpu):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    calls, ms, t_end = 0, 0.0, time.perf_counter() + seconds
    gpu.start()
    while time.perf_counter() < t_end:
        e0.record()
        for _ in range(10):
            fn()
        e1.record()
        e1.synchronize()
        ms += e0.elapsed_time(e1)
        calls += 10
    tele = gpu.stop()
    return ops_per_call * calls / (ms * 1e-3) / 1e12, tele


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mma-peak", default="./mma_peak")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=4)
    ap.add_argument("--shapes", default="gdn_in,gdn_out,attn_qkv,attn_o,mlp_gate_up,mlp_down")
    a = ap.parse_args()
    gpu = GPUSampler(interval=0.02)
    gpu.start()
    out = subprocess.run([a.mma_peak, "sustain", str(a.seconds)], capture_output=True, text=True).stdout.strip()
    tele = gpu.stop()
    print(f"mma.sync int8 peak: {out}; SM clock {tele.get('sm_clock_mhz_mean')} MHz, {tele.get('power_w_mean')} W",
          flush=True)
    dev = "cuda"
    M = a.m
    n_rows = torch.tensor([M], dtype=torch.int32, device=dev)
    for name in a.shapes.split(","):
        key = next(k for k in SHAPES if k.split(" ")[0] == name)
        N, K = SHAPES[key]
        epi = EPI.get(name, 0)
        x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        sa = torch.rand(M, 1, device=dev) * 1e-2
        ws = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for _ in range(4)]
        sw = torch.rand(N, 1, device=dev) * 1e-3
        out = torch.zeros(M, N, device=dev, dtype=torch.float32 if epi else torch.bfloat16)
        slots = torch.zeros(4, M, N, dtype=torch.int32, device=dev) if epi else None
        it = [0]

        def ours():
            it[0] = (it[0] + 1) % 4
            KW.gemm(x, sa, ws[it[0]], sw, out, n_rows, K, epi=epi, slots=slots)

        def lt():
            it[0] = (it[0] + 1) % 4
            torch._int_mm(x, ws[it[0]].t())
        ops = 2.0 * M * N * K
        t_o, g_o = sustained(ours, a.seconds, ops, gpu)
        t_l, g_l = sustained(lt, a.seconds, ops, gpu)
        print(f"{name:12s} M={M} N={N:5d} K={K:4d}: flash-jev {t_o:6.1f} TOPS at {g_o.get('sm_clock_mhz_mean')} MHz "
              f"{g_o.get('power_w_mean')} W | cuBLASLt {t_l:6.1f} TOPS at {g_l.get('sm_clock_mhz_mean')} MHz "
              f"{g_l.get('power_w_mean')} W", flush=True)


if __name__ == "__main__":
    main()
