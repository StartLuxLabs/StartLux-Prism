"""The k-quant dequantization (gguf_weights.KQuant: Q4_K, Q5_K, Q6_K, the tensors of Q4_K_M files) against gguf-py's
reference (gguf.quants.dequantize, numpy), bit for bit, on a real file: every k-quant type in it, a few whole tensors
each, on the GPU as the engine runs it; and the Q8_0 copy of a k-quant embedding (KQuant.as_q8) within half a step.

    python tests/test_kquant.py MODEL-Q4_K_M.gguf [--tensors 3]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev import gguf_weights as GW  # noqa: E402
from flash_jev.gguf import GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGUFFile  # noqa: E402

NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensors", type=int, default=3, help="tensors checked per k-quant type")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    f = GGUFFile(a.gguf)
    by_type = {}
    for name, t in f.tensors.items():
        by_type.setdefault(t.type, []).append(name)
    print("tensors by type:", {NAMES.get(k, k): len(v) for k, v in sorted(by_type.items())})
    bad = 0
    for typ in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K):
        names = sorted(by_type.get(typ, []), key=lambda n: -f.tensors[n].nbytes)[:a.tensors]
        for name in names:
            kq = GW._kq(f, name)
            n, k = kq.shape
            ref = dequantize(kq.raw.numpy().reshape(n, -1), GGMLQuantizationType(typ)).reshape(n, k)
            got = kq.to(a.device).dequant(torch.float32).cpu().numpy()
            same = np.array_equal(got, ref)
            bad += not same
            print(f"{NAMES[typ]:5s} {name:32s} [{n}, {k}]  " + ("bit-identical to gguf-py" if same else
                  f"DIFFERS: max |d| {np.abs(got - ref).max():.3e}"))
    emb = GW.load(a.gguf).embed
    if isinstance(emb, GW.KQuant):
        rows = torch.arange(0, emb.shape[0], 97)
        exact = emb.dequant(torch.float32, rows)
        q8 = GW.KQuant(emb.raw[rows], emb.type, emb.k).to(a.device).as_q8()
        approx = q8.dequant(torch.float32).cpu()
        step = (exact.view(len(rows), -1, GW.QK).abs().amax(-1) / 127).repeat_interleave(GW.QK, 1)
        worst = ((approx - exact).abs() / step.clamp_min(1e-12)).max().item()
        ok = worst <= 0.5 + 1e-3
        bad += not ok
        print(f"embedding ({NAMES[emb.type]}) as Q8_0: largest error {worst:.3f} of a step ({'ok' if ok else 'TOO LARGE'})")
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
