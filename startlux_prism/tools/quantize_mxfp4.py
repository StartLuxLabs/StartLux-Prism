"""Pack a StartLux GGUF's projections to MXFP4.

    python -m startlux_prism.tools.quantize_mxfp4 MODEL.gguf OUT.pt [--mode max|mse] [--groups all|gu,down,...]
                                                              [--layers all|0,1,...]

Groups: MLP (`gu` = gate|up, `down`), attention (`qkv`, `o`) and GDN (`inp` = qkv|z|beta|alpha, `out`).
One projection is dequantized, packed and dropped at a time, so the peak host memory is the GGUF plus one
fp32 matrix.  The output is a torch dict `{ "<layer>.<group>": {"q", "s", "n", "k"} }` with the layouts from
`startlux_prism.kernels.mxfp4`: q uint8 [n, k/2], s uint8 [n, k/32].

This is the weight-only scheme.  It does not smooth activations or fold scales into norms; those change the
model's numerics and are a separate, model-specific calibration step.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from startlux_prism import gguf_weights as GW          # noqa: E402
from startlux_prism.kernels import mxfp4 as K          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--mode", default="max", choices=("max", "mse"))
    ap.add_argument("--groups", default="all", help="all, or a comma list: gu,down,qkv,o,inp,out")
    ap.add_argument("--layers", default="all", help="all, or a comma list of layer indices")
    a = ap.parse_args()
    want = None if a.groups == "all" else set(a.groups.split(","))
    want_layers = None if a.layers == "all" else {int(v) for v in a.layers.split(",")}
    w = GW.load(a.gguf)
    blob = {}

    def take(name):
        return want is None or name in want

    def put(i, name, mat):
        p = K.pack(mat, a.mode)
        dq = K.dequant_packed(p.q, p.s)
        rel = (dq - mat).norm() / mat.norm()
        blob[f"{i}.{name}"] = {"q": p.q, "s": p.s, "n": p.n, "k": p.k}
        print(f"layer {i:2d}  {name:5s} rel {rel:.5f}", flush=True)

    for i, l in enumerate(w.layers):
        if want_layers is not None and i not in want_layers:
            continue
        if take("gu") or take("down"):
            gu = torch.cat([l.ffn_gate.dequant(torch.float32), l.ffn_up.dequant(torch.float32)], 0)
            dn = l.ffn_down.dequant(torch.float32)
            if take("gu"):
                put(i, "gu", gu)
            if take("down"):
                put(i, "down", dn)
            del gu, dn

        if w.config.is_attention(i):
            if take("qkv"):
                qkv = torch.cat([l.wq.dequant(torch.float32), l.wk.dequant(torch.float32),
                                 l.wv.dequant(torch.float32)], 0)
                put(i, "qkv", qkv)
                del qkv
            if take("o"):
                o = l.wo.dequant(torch.float32)
                put(i, "o", o)
                del o
        else:
            if take("inp"):
                inp = torch.cat([l.wqkv.dequant(torch.float32), l.wz.dequant(torch.float32),
                                 l.wbeta.dequant(torch.float32), l.walpha.dequant(torch.float32)], 0)
                put(i, "inp", inp)
                del inp
            if take("out"):
                out = l.wout.dequant(torch.float32)
                put(i, "out", out)
                del out

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(blob, a.out)
    print("saved", a.out, f"{os.path.getsize(a.out) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
