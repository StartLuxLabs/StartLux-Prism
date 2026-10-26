"""Select native FP4 tiles offline and independently qualify their real shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from startlux_prism.kernels import mxfp4 as F


def dequant(packed):
    q = packed.q
    codes = torch.stack((q & 15, q >> 4), -1).flatten(1)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=q.device, dtype=torch.float64)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
    return (values.reshape(q.shape[0], -1, 32) * torch.exp2(packed.s.double() - 127)[..., None]).flatten(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    best = {}
    for path in a.inputs:
        data = json.loads(Path(path).read_text())
        for row in data.get('projections', data.get('cases', [])):
            if 'complete_ms' not in row:
                continue
            key = tuple(row['shape']) if isinstance(row['shape'], list) else (data['bucket'], row['n'], row['k'])
            if key not in best or row['complete_ms'] < best[key]['selection_ms']:
                best[key] = dict(shape=key, live=row.get('live', data.get('live')), config=row['config'],
                                 selection_ms=row['complete_ms'], selection_file=Path(path).name)
    result = dict(qualified=False, torch=str(torch.__version__), hip=torch.version.hip,
                  triton=triton.__version__, gpu=torch.cuda.get_device_name(),
                  kernel_sha256=F.kernel_identity(), entries=[])
    torch.manual_seed(719)
    for shape, row in sorted(best.items()):
        m, n, k = shape
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        w = F.quantize(torch.randn((n, k), device='cuda', dtype=torch.bfloat16))
        aq = F.quantize(x)
        xd, wd = dequant(aq), dequant(w)
        ref = xd @ wd.T
        bound = 1e-5 + 2e-6 * (xd.abs() @ wd.abs().T)
        live = torch.tensor(m, device='cuda', dtype=torch.int32)
        storage = torch.full((m, n + 9), 42., device='cuda')
        out = storage[:, 3:n + 3]
        F.gemm(aq, w, out, live, row['config'])
        ratio = ((out.double() - ref).abs() / bound).max().item()
        assert ratio <= 1, (shape, row['config'], ratio)
        assert torch.all(storage[:, :3] == 42) and torch.all(storage[:, n + 3:] == 42)
        assert torch.isfinite(out).all()
        first = out.clone()
        for count in sorted({0, 1, row['live'], m}):
            storage.fill_(42)
            live.fill_(count)
            F.gemm(aq, w, out, live, row['config'])
            assert torch.equal(out[:count], first[:count]), (shape, count, 'live-mask arithmetic')
            assert torch.all(out[count:] == 42), (shape, count, 'tail store')
        # Validate capture and repeated replay at the dtype used by the model.
        y = torch.full((m, n), 42., device='cuda', dtype=torch.bfloat16)
        F.gemm(aq, w, y, live, row['config'])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            F.quantize_into(x, aq, live)
            F.gemm(aq, w, y, live, row['config'])
        graph.replay()
        before = y.clone()
        graph.replay()
        assert torch.equal(y, before), (shape, 'graph replay nondeterminism')
        row.update(fp64_reference_pass=True, max_error_bound_fraction=ratio,
                   masks_pass=True, bf16_graph_replay_pass=True)
        result['entries'].append(row)
        print('PASS', shape, row['config'], 'bound fraction', round(ratio, 6), flush=True)
        del graph, x, w, aq, xd, wd, ref, bound, live, storage, out, first, y, before
        torch.cuda.empty_cache()
    result['qualified'] = True
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print('QUALIFIED', len(result['entries']), 'native FP4 dispatch entries', flush=True)


if __name__ == '__main__':
    main()
