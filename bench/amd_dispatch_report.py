"""Offline dispatch diagnosis; never called by a timed kernel launch or graph replay."""
import importlib.metadata
import json
from pathlib import Path


def dispatcher(precision):
    if precision == 'w4a16':
        from startlux_prism.kernels import amd_w4a16_dispatch as D
    else:
        from startlux_prism.kernels import amd_w4a8_dispatch as D
    return D


def table_status(precision='w4a8'):
    import torch
    from startlux_prism.backends import architecture
    D = dispatcher(precision)
    path = Path(D.__file__).with_name(f'{precision}_table_gfx1201_rocm101.json')
    if not path.exists():
        return dict(reason='table_missing', entries=0)
    data = json.loads(path.read_text())
    try:
        runtime = dict(architecture=architecture(), torch=importlib.metadata.version('torch'),
                       triton=importlib.metadata.version('triton'), hip=torch.version.hip)
    except importlib.metadata.PackageNotFoundError as exc:
        return dict(reason='runtime_mismatch', entries=0, detail=str(exc))
    if data.get('schema') != 1 or data.get('runtime') != runtime:
        return dict(reason='runtime_mismatch', entries=0, runtime=runtime)
    identity = D.kernel_identity()
    if data.get('kernel_identity') != identity:
        return dict(reason='kernel_mismatch', entries=0, kernel_identity=identity)
    return dict(reason='qualified', entries=len(D.table()), kernel_identity=identity, runtime=runtime)


def explain(xq, w, out, epi, precision='w4a8'):
    D = dispatcher(precision)
    result = table_status(precision)
    separator = '|' if precision == 'w4a16' else ':'
    key = separator.join(map(str, (w.kind, xq.shape[0], w.n, w.k, epi, out.stride(0))))
    result['key'] = key
    if result['reason'] != 'qualified':
        return result
    if precision == 'w4a16':
        import torch
        if xq.dtype != torch.float16 or out.dtype != (torch.float32 if epi else torch.float16):
            result.update(reason='layout_rejected', detail='dtype')
            return result
    if (any(t.data_ptr() % 16 for t in (xq, w.q, out)) or not xq.is_contiguous()
            or not w.q.is_contiguous() or out.stride(1) != 1 or out.stride(0) < w.n):
        result['reason'] = 'layout_rejected'
    else:
        result['choice'] = D.pick(xq, w, out, epi)
        if result['choice'] is None:
            result['reason'] = 'shape_missing'
    return result


if __name__ == '__main__':
    import argparse
    import os
    import sys
    sys.path.insert(0, os.environ.get('PRISM_BENCH_REPO', str(Path(__file__).resolve().parents[1])))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision', choices=('w4a8', 'w4a16'), default='w4a8')
    p.add_argument('--census', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = table_status(a.precision)
    entries = dispatcher(a.precision).table()
    keys = sorted({r['key'] for path in a.census for r in json.loads(path.read_text())})
    report['shapes'] = [dict(key=k, reason='qualified' if
        (k.replace(':', '|') if a.precision == 'w4a16' else k) in entries else 'shape_missing',
        choice=entries.get(k.replace(':', '|') if a.precision == 'w4a16' else k)) for k in keys]
    rendered = json.dumps(report, indent=2)+'\n'
    if a.output: a.output.write_text(rendered)
    print(rendered)
