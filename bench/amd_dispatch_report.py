"""Offline dispatch diagnosis; never called by a timed kernel launch or graph replay."""
import importlib.metadata
import json
from pathlib import Path


def table_status():
    import torch
    from startlux_prism.backends import architecture
    from startlux_prism.kernels import amd_w4a8_dispatch as D
    path = Path(D.__file__).with_name('w4a8_table_gfx1201_rocm101.json')
    if not path.exists():
        return dict(reason='table_missing', entries=0)
    data = json.loads(path.read_text())
    runtime = dict(architecture=architecture(), torch=importlib.metadata.version('torch'),
                   triton=importlib.metadata.version('triton'), hip=torch.version.hip)
    if data.get('schema') != 1 or data.get('runtime') != runtime:
        return dict(reason='runtime_mismatch', entries=0, runtime=runtime)
    identity = D.kernel_identity()
    if data.get('kernel_identity') != identity:
        return dict(reason='kernel_mismatch', entries=0, kernel_identity=identity)
    return dict(reason='qualified', entries=len(D.table()), kernel_identity=identity, runtime=runtime)


def explain(xq, w, out, epi):
    from startlux_prism.kernels import amd_w4a8_dispatch as D
    result = table_status()
    key = f'{w.kind}:{xq.shape[0]}:{w.n}:{w.k}:{epi}:{out.stride(0)}'
    result['key'] = key
    if result['reason'] != 'qualified':
        return result
    if (any(t.data_ptr() % 16 for t in (xq, w.q, out)) or not xq.is_contiguous()
            or not w.q.is_contiguous() or out.stride(1) != 1 or out.stride(0) < w.n):
        result['reason'] = 'layout_rejected'
    else:
        result['choice'] = D.pick(xq, w, out, epi)
        if result['choice'] is None:
            result['reason'] = 'shape_missing'
    return result
