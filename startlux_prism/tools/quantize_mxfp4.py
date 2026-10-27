"""Pack a StartLux GGUF's projections to MXFP4.

    python -m startlux_prism.tools.quantize_mxfp4 MODEL.gguf OUT.pt [--mode max|mse]
        [--smooth CALIB.pt --alpha 0.5] [--groups all|gu,down,...] [--layers all|0,1,...]

Groups: MLP (`gu` = gate|up, `down`), attention (`qkv`, `o`) and GDN (`inp` = qkv|z|beta|alpha, `out`).
One projection is dequantized, packed and dropped at a time, so the peak host memory is the GGUF plus one
fp32 matrix.  The output is a torch dict `{ "<layer>.<group>": (q, s) }` (and `s_<group>` when smoothing),
with the layouts from `startlux_prism.kernels.mxfp4`: q uint8 [n, k/2], s uint8 [n, k/32].

Without `--smooth` this is the weight-only scheme.  With `--smooth CALIB.pt` it applies SmoothQuant-style
per-input-channel scales from an int8 calibration file: W is multiplied by s and `s_<group>` is written beside
it.  The consumer divides the corresponding activation by s (fold it into the preceding norm or apply it at
runtime); for `down`, 1/s_down is folded into the `up` rows of `gu`.  The calibration file is model-specific.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from startlux_prism import gguf_weights as GW          # noqa: E402
from startlux_prism.kernels import mxfp4 as K          # noqa: E402


def smooth_scale(act, w, alpha):
    """Per-input-channel SmoothQuant scale, clamped like the int8 path."""
    wa = w.abs().amax(0).clamp_min(1e-5)
    return (act.clamp_min(1e-5) ** alpha / wa ** (1 - alpha)).clamp(1e-3, 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--mode", default="max", choices=("max", "mse"))
    ap.add_argument("--smooth", default="", help="int8 calibration file; enables SmoothQuant-style scaling")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--groups", default="all", help="all, or a comma list: gu,down,qkv,o,inp,out")
    ap.add_argument("--layers", default="all", help="all, or a comma list of layer indices")
    a = ap.parse_args()
    want = None if a.groups == "all" else set(a.groups.split(","))
    want_layers = None if a.layers == "all" else {int(v) for v in a.layers.split(",")}
    w = GW.load(a.gguf)
    calib = torch.load(a.smooth, map_location="cpu", weights_only=False) if a.smooth else None
    blob = {}

    def take(name):
        return want is None or name in want

    def put(i, name, mat, scale=None):
        p = K.pack(mat, a.mode)
        dq = K.dequant_packed(p.q, p.s)
        rel = (dq - mat).norm() / mat.norm()
        blob[f"{i}.{name}"] = (p.q, p.s)
        if scale is not None:
            blob[f"{i}.s_{name}"] = scale
        print(f"layer {i:2d}  {name:5s} rel {rel:.5f}", flush=True)

    for i, l in enumerate(w.layers):
        if want_layers is not None and i not in want_layers:
            continue
        if take("gu") or take("down"):
            gu = torch.cat([l.ffn_gate.dequant(torch.float32), l.ffn_up.dequant(torch.float32)], 0)
            dn = l.ffn_down.dequant(torch.float32)
            s_gu = None
            if calib is not None:
                ffn = dn.shape[1]
                s_gu = smooth_scale(calib[f"{i}:gu"], gu, a.alpha)
                s_dn = smooth_scale(calib[f"{i}:down"], dn, a.alpha)
                gu = gu * s_gu[None, :]
                gu[ffn:] = gu[ffn:] / s_dn[:, None]      # silu(gate) * (up / s_dn)
                dn = dn * s_dn[None, :]
            if take("gu"):
                put(i, "gu", gu, s_gu)
            if take("down"):
                put(i, "down", dn)
            del gu, dn

        if w.config.is_attention(i):
            if take("qkv"):
                qkv = torch.cat([l.wq.dequant(torch.float32), l.wk.dequant(torch.float32),
                                 l.wv.dequant(torch.float32)], 0)
                s = smooth_scale(calib[f"{i}:qkv"], qkv, a.alpha) if calib is not None else None
                if s is not None:
                    qkv = qkv * s[None, :]
                put(i, "qkv", qkv, s)
                del qkv
            if take("o"):
                o = l.wo.dequant(torch.float32)
                s = smooth_scale(calib[f"{i}:o"], o, a.alpha) if calib is not None else None
                if s is not None:
                    o = o * s[None, :]
                put(i, "o", o, s)
                del o
        else:
            if take("inp"):
                inp = torch.cat([l.wqkv.dequant(torch.float32), l.wz.dequant(torch.float32),
                                 l.wbeta.dequant(torch.float32), l.walpha.dequant(torch.float32)], 0)
                s = smooth_scale(calib[f"{i}:inp"], inp, a.alpha) if calib is not None else None
                if s is not None:
                    inp = inp * s[None, :]
                put(i, "inp", inp, s)
                del inp
            if take("out"):
                out = l.wout.dequant(torch.float32)
                s = smooth_scale(calib[f"{i}:out"], out, a.alpha) if calib is not None else None
                if s is not None:
                    out = out * s[None, :]
                put(i, "out", out, s)
                del out

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(blob, a.out)
    print("saved", a.out, f"{os.path.getsize(a.out) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
