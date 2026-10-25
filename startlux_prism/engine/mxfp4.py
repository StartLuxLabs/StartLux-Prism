"""MXFP4 MLP loading for the engine.

This is the model-side half of `kernels/mxfp4.py`: it turns the MLP projections of a StartLux
decoder file into MXFP4 weights and returns the per-layer smoothing scale that belongs on the
preceding norm.  It works for any layer count, hidden size and FFN size; the only model-specific
part is the side-file key convention (`<layer>.gu`, `<layer>.down`, `<layer>.s_gu`) written by
`tools/quantize_mxfp4.py`.

With no side file, the MLP is packed from the GGUF itself (weight-only, scale 1).  With a side
file, `post_norm` must be divided by `s_gu` before the norm is used; `Engine._mxfp4_mlp_weights`
does that.
"""
from __future__ import annotations

import os

import torch

from .. import progress as P
from ..kernels import mxfp4 as K


def side_file(model_dir, explicit="", env="STARTLUX_PRISM_MXFP4_MLP"):
    """The side file to use: explicit > env > model_dir/mxfp4/mlp_smooth.pt > mlp.pt, or "".

    An explicit path (or an environment path) that does not exist is an error: silently falling
    back would change the numerics without saying so.  A missing default means pack from the GGUF.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"MXFP4 side file not found: {explicit}")
        return explicit
    env_path = os.environ.get(env)
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"{env} points at a missing MXFP4 side file: {env_path}")
        return env_path
    root = os.path.join(model_dir, "mxfp4")
    for name in ("mlp_smooth.pt", "mlp.pt"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return ""


def load_mlp(w, cfg, device, model_dir, side="", cutlass=None):
    """{layer: (gu MX4, down MX4, s_gu)} for the model's MLP, optionally from a smoothed side file.

    `gu` is [2 * ffn, hidden] gate|up, `down` is [hidden, ffn].  `s_gu` is [hidden]; the caller
    divides `post_norm` by it.  The side file's q/s data is used as stored; only the CUTLASS scale
    layout is prepared here when the kernel is enabled.
    """
    path = side_file(model_dir, side)
    if path:
        P.step(f"MXFP4 MLP: side file {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out = {}
        ones = torch.ones(cfg.hidden, device=device)
        for i in range(len(w.layers)):
            gq, gs = blob[f"{i}.gu"]
            dq, ds = blob[f"{i}.down"]
            s_gu = blob.get(f"{i}.s_gu", ones.cpu()).to(device)
            out[i] = (K.MX4(gq.to(device), gs.to(device), gq.shape[0], gq.shape[1] * 2),
                      K.MX4(dq.to(device), ds.to(device), dq.shape[0], dq.shape[1] * 2), s_gu)
        del blob
    else:
        P.step("MXFP4 MLP: packing the GGUF's MLP (weight-only; no side file)")
        out = {}
        for i, l in enumerate(w.layers):
            gu = torch.cat([l.ffn_gate.dequant(torch.float32), l.ffn_up.dequant(torch.float32)], 0)
            dn = l.ffn_down.dequant(torch.float32)
            a, b = K.pack(gu), K.pack(dn)
            out[i] = (K.MX4(a.q.to(device), a.s.to(device), a.n, a.k),
                      K.MX4(b.q.to(device), b.s.to(device), b.n, b.k),
                      torch.ones(cfg.hidden, device=device))
            del gu, dn
    if cutlass is None:
        cutlass = K.cutlass_enabled()
    if cutlass:
        P.step("MXFP4 MLP: CUTLASS kernels (interleaved scales)")
        out = {i: (K.cutlass_prepare(gu), K.cutlass_prepare(dn), s) for i, (gu, dn, s) in out.items()}
    return out
