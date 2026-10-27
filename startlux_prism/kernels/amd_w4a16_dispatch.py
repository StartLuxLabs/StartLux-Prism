"""Offline metadata-layout choices for the separate compressed W4A16 profile."""
import ast
import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path


def kernel_identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('amd_w4a16.py', 'amd_w4a16_meta.py', 'amd_w4a16_dispatch.py', '../backends/math.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    source = (root / 'w4a8.py').read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {'W4', 'repack', 'concat', 'segments', 'seg_rows'}:
            for decorator in node.decorator_list:
                digest.update(ast.get_source_segment(source, decorator).encode())
            digest.update(ast.get_source_segment(source, node).encode())
    return digest.hexdigest()


@functools.lru_cache(None)
def table():
    import torch
    from ..backends import architecture
    if not torch.version.hip or architecture() != 'gfx1201':
        return {}
    path = Path(__file__).with_name('w4a16_table_gfx1201_rocm101.json')
    if not path.exists():
        return {}
    try:
        runtime = dict(architecture='gfx1201', torch=importlib.metadata.version('torch'),
                       triton=importlib.metadata.version('triton'), hip=torch.version.hip)
    except importlib.metadata.PackageNotFoundError:
        return {}
    data = json.loads(path.read_text())
    if data.get('schema') != 1 or data.get('runtime') != runtime or data.get('kernel_identity') != kernel_identity():
        return {}
    return data['entries']


def pick(x, w, out, epi):
    import torch
    if x.dtype != torch.float16 or out.dtype != (torch.float32 if epi else torch.float16):
        return None
    if not x.is_contiguous() or not w.q.is_contiguous() or out.stride(1) != 1 or out.stride(0) < w.n:
        return None
    if any(t.data_ptr() % 16 for t in (x, w.q, out)):
        return None
    key = '|'.join(map(str, (w.kind, x.shape[0], w.n, w.k, epi, out.stride(0))))
    return table().get(key)
