"""The int8 path's activation ranges (SmoothQuant): where they are kept, and computing them once.

int8 (W8A8) needs every projection input's per-channel maximum, measured by a bf16 forward over StartLux-Decision's
calibration requests (engine/model.py: Engine.calibrate).  The first int8 start computes them and keeps them in a file
(--calib, by default under ~/.cache/startlux-prism/calib); later starts read it.  A tensor-parallel group loads its shards
straight from the Q8_0 values, without the bf16 copy that forward needs, so ensure() computes the file first, on one
GPU, in a process of its own (its bf16 weights are freed when it exits).

    python -m startlux_prism.calib MODEL.gguf MODEL_DIR FILE        # what ensure() runs
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

VERSION = 1                         # in the default file name: a changed calibration does not read old files


def auto_precision(gguf):
    """--precision auto: w4a8 for k-quant files (Q4_K_M), int8 for the others (Q8_0)."""
    from .gguf import GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGUFFile
    kinds = {t.type for n, t in GGUFFile(gguf).tensors.items() if n.endswith("ffn_gate.weight")}
    return "w4a8" if kinds & {GGML_Q4_K, GGML_Q5_K, GGML_Q6_K} else "int8"


def default_path(gguf):
    """$STARTLUX_PRISM_CACHE (default ~/.cache/startlux-prism, or ~/.cache/flash-jev where only that one exists: the
    engine's former name)/calib/<file name>-<size>-v<VERSION>.pt."""
    root = os.environ.get("STARTLUX_PRISM_CACHE")
    if not root:
        home = os.path.join(os.path.expanduser("~"), ".cache")
        root = os.path.join(home, "startlux-prism")
        if not os.path.isdir(root) and os.path.isdir(os.path.join(home, "flash-jev")):
            root = os.path.join(home, "flash-jev")
    stem = os.path.basename(gguf).removesuffix(".gguf")
    return os.path.join(root, "calib", f"{stem}-{os.path.getsize(gguf)}-v{VERSION}.pt")


def first_gpu(gpus=None):
    """The first selected GPU, respecting ROCm visibility for calibration workers."""
    from .backends import is_hip
    visible = "HIP_VISIBLE_DEVICES" if is_hip() else "CUDA_VISIBLE_DEVICES"
    for s in (gpus, os.environ.get(visible), os.environ.get("CUDA_VISIBLE_DEVICES")):
        if s and s.split(",")[0].strip():
            return s.split(",")[0].strip()
    return "0"


def ensure(gguf, model_dir, path, gpu):
    """Compute the ranges into `path` on GPU `gpu` unless the file is there."""
    if os.path.exists(path):
        return
    print(f"int8 activation ranges: computing them once on GPU {gpu} (a bf16 forward of the calibration requests), "
          f"kept in {path}", flush=True)
    from .backends import device_env
    env = device_env(gpu)
    r = subprocess.run([sys.executable, "-m", "startlux_prism.calib", gguf, model_dir, path], env=env)
    if r.returncode != 0 or not os.path.exists(path):
        sys.exit(f"computing the int8 activation ranges failed (exit code {r.returncode})")


def main():
    gguf, model_dir, path = sys.argv[1:4]
    from . import progress as P
    from .engine.model import CALIB_TOKENS, Engine, Options
    P.on()
    t0 = time.perf_counter()
    Engine(gguf, model_dir, Options(precision="int8", calib=path, max_tokens=CALIB_TOKENS, prefix_cache=0))
    print(f"int8 activation ranges saved in {path} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
