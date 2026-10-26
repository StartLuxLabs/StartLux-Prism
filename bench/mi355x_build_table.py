"""Freeze exact, offline measured MI355X tiles; require a minimum gain over fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import triton
from startlux_prism.kernels import w8a8 as W


def build(baseline, candidates, min_gain=1.05):
    entries, measurements = {}, {}
    for base in baseline['projections']:
        key = (base['m'],base['n'],base['k'],base['epi'])
        rows = [r for candidate in candidates for r in candidate['projections']
                if (r['m'],r['n'],r['k'],r['epi']) == key and r.get('exact')]
        if not rows:
            continue
        best = min(rows,key=lambda r:r['us'])
        if base['us']/best['us'] >= min_gain:
            text = ','.join(map(str,key))
            entries[text] = best['cfg']
            measurements[text] = dict(baseline_us=base['us'], selected_us=best['us'], gain=base['us']/best['us'])
    return entries, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidates',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--min-gain',type=float,default=1.05)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    candidates = [json.loads(Path(p).read_text()) for p in args.candidates]
    assert all(c['gpu']==baseline['gpu'] and c['torch']==baseline['torch'] and c['hip']==baseline['hip']
               and c['triton']==baseline['triton'] and c['kernel_identity']==baseline['kernel_identity']
               for c in candidates)
    assert all(r.get('exact') for r in baseline['projections'])
    entries, measurements = build(baseline,candidates,args.min_gain)
    data = dict(schema=1, architecture='gfx950', gpu=baseline['gpu'],
                runtime=dict(torch=baseline['torch'],hip=baseline['hip'],triton=baseline['triton']),
                kernel_identity=hashlib.sha256(W._w8a8_kernel.src.encode()).hexdigest(),
                min_gain=args.min_gain, entries=entries, measurements=measurements)
    assert data['runtime']==dict(torch=str(torch.__version__),hip=torch.version.hip,triton=triton.__version__)
    assert data['kernel_identity']==baseline['kernel_identity']
    Path(args.out).write_text(json.dumps(data,indent=2)+'\n')
    print('Qualified',len(entries),'tiles',flush=True)


if __name__=='__main__':
    main()
