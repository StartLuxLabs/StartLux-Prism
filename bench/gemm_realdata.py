"""Projection GEMMs on the engine's real weights and activations vs random int8 data (same shapes, same kernels):
does the data change the speed (tensor-core power and clocks depend on operand bit activity)?

    python bench/gemm_realdata.py --gguf X.gguf --model-dir DIR [--m 384]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine.model import Engine, Options  # noqa: E402
from startlux_prism.kernels import w8a8 as KW  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
from gemm_micro import timed  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", default=os.path.expanduser("~/jev/calib/startlux-4b-q8_0.pt"))
    ap.add_argument("--m", type=int, default=384)
    a = ap.parse_args()
    eng = Engine(a.gguf, a.model_dir, Options(precision="int8", max_tokens=4096, calib=a.calib))
    state, questions = suites.fixed()["longpage2"]
    recs = [J.from_systemone(state, q) for q in questions.values()]
    eng.run([[r.ids for r in rows_for(eng.tok, recs, list(questions))]])      # leaves real activations in aq
    M, dev = a.m, eng.device
    n = torch.tensor([M], dtype=torch.int32, device=dev)
    gdn = [i for i in range(eng.cfg.n_layer) if i in eng.gdn_index][:4]
    for name, wk, sk, K in (("gate_up", "gu_q", "gu_s", eng.cfg.hidden), ("gdn_in", "inp_q", "inp_s", eng.cfg.hidden),
                            ("down", "down_q", "down_s", eng.cfg.ffn)):
        real_w = [(eng.layers[i][wk], eng.layers[i][sk]) for i in gdn]
        N = real_w[0][0].shape[0]
        rand_w = [(torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8), real_w[0][1]) for _ in gdn]
        real_a = eng.buf["aq"][:M * K].view(M, K).clone()
        rand_a = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        sa = torch.rand(M, device=dev) * 1e-2
        epi = 1 if name == "down" else 0
        out = torch.zeros(M, N, device=dev, dtype=torch.float32 if epi else torch.bfloat16)
        for label, A, W in (("random A, random W", rand_a, rand_w), ("real A, real W", real_a, real_w),
                            ("real A, random W", real_a, rand_w), ("random A, real W", rand_a, real_w)):
            def f():
                for w, s in W:
                    KW.gemm(A, sa, w, s, out, n, K, epi=epi)
            us = timed(f, 10) / len(W) * 1000
            print(f"{name:8s} M={M} N={N} K={K} {str(KW._choice(N, K, M)):22s} {label:20s} {us:7.1f} us", flush=True)
        print(f"  real A: {real_a.float().abs().mean().item():.1f} mean |x|, {(real_a == 0).float().mean().item():.2f} "
              f"zeros; real W: {real_w[0][0].float().abs().mean().item():.1f} mean |w|")


if __name__ == "__main__":
    main()
