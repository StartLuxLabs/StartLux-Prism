"""Offline qualified ROCm projection choices; unknown configurations use legacy kernels."""

import functools
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path


def lookup(entries, kind, m, n, k, epi, stride0, stride1):
    """No device-to-host reads: m is the allocated/captured row bucket."""
    if stride1 != 1 or stride0 < n:
        return None
    return entries.get(f"{kind}:{m}:{n}:{k}:{epi}:{stride0}")


def kernel_identity():
    """Only qualified AMD arithmetic/packing participates; unrelated CUDA additions do not."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("amd_w4a8.py", "amd_w4a8_dot.py", "amd_q4_prefill.py", "amd_q6_wmma.py",
                 "amd_w4a8_dispatch.py", "../backends/math.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    source = (root / "w4a8.py").read_text()
    names = {"W4", "repack", "concat", "segments", "seg_rows", "_tile", "_store", "_q4k_kernel", "_q6k_kernel", "pick"}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            # ast.dump's omission of empty fields changed in Python 3.13.
            # Source slices keep the offline identity stable across Python hosts.
            for decorator in node.decorator_list:
                digest.update(ast.get_source_segment(source, decorator).encode())
            digest.update(ast.get_source_segment(source, node).encode())
    return digest.hexdigest()


@functools.lru_cache(None)
def table(family="auto"):
    import torch
    from ..backends import architecture

    if not torch.version.hip or architecture() != "gfx1201":
        return {}
    path = Path(__file__).with_name("w4a8_table_gfx1201_rocm101.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    try:
        identity = dict(architecture="gfx1201", torch=importlib.metadata.version("torch"),
                        triton=importlib.metadata.version("triton"), hip=torch.version.hip)
    except importlib.metadata.PackageNotFoundError:
        return {}  # Some ROCm builds package Triton under a different distribution name.
    if data.get("schema") != 1 or data.get("runtime") != identity:
        return {}
    if data.get("kernel_identity") != kernel_identity():
        return {}
    return data["entries"] if family == "auto" else data.get("family_entries", {}).get(family, {})


def pick(xq, w, out, epi, family="auto"):
    if any(t.data_ptr() % 16 for t in (xq, out, w.q)):
        return None
    if not xq.is_contiguous() or not w.q.is_contiguous():
        return None
    return lookup(table(family), w.kind, xq.shape[0], w.n, w.k, epi, out.stride(0), out.stride(1))


def run(xq, a, xs, w, out, n_rows, epi, mode="auto"):
    """Launch a qualified choice; False leaves the caller on its legacy path.

    The phase switches support offline ablation, never online autotuning.
    Selection uses static metadata only, including during graph capture.
    """
    choice = pick(xq, w, out, epi, mode)
    if choice is None or (mode != "auto" and choice["family"] != mode):
        return False
    implementation, cfg = choice["implementation"], choice.get("cfg")
    if implementation == "gemv":
        from .amd_w4a8 import gemv
        gemv(xq, a, xs, w, out, n_rows, epi)
    elif implementation == "dot":
        from .amd_w4a8_dot import gemv
        gemv(xq, a, xs, w, out, n_rows, epi, cfg=tuple(cfg))
    elif implementation in ("wmma", "tile"):
        from .w4a8 import gemm_triton
        gemm_triton(xq, a, xs, w, out, n_rows, epi, cfg=None if cfg is None else tuple(cfg))
    elif implementation == "q6scaled":
        from .amd_q6_wmma import gemm
        gemm(xq, a, xs, w, out, n_rows, epi, cfg=tuple(cfg))
    elif implementation == "loop":
        from .amd_q4_prefill import gemm
        gemm(xq, a, xs, w, out, n_rows, epi, cfg=tuple(cfg))
    else:
        return False
    return True
