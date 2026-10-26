"""State-reset, paired single-GPU decode benchmark with full-output evidence."""
import argparse
import collections
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

repo = Path(os.environ.get('PRISM_BENCH_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(repo))
from startlux_prism import progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.engine.decode import Decoder, Row, Sampling, DEC_BUCKETS
from startlux_prism.kernels import w4a8 as W, amd_w4a8_dispatch as D
from startlux_prism.backends import architecture

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf', required=True)
p.add_argument('--model-dir', required=True)
p.add_argument('--output', required=True)
p.add_argument('--precision', default='w4a8', choices=('w4a8', 'w4a16'))
p.add_argument('--batches', default='1,2,4,8,9,16')
p.add_argument('--modes', default='auto,reference')
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--context', type=int, default=512)
p.add_argument('--max-tokens', type=int, default=8192, help='prefill capacity; long prompts are packed in separate launches')
p.add_argument('--eager', action='store_true')
p.add_argument('--baseline', type=Path)
p.add_argument('--capture', action='store_true')
p.add_argument('--capture-decode-only', action='store_true')
p.add_argument('--profile', action='store_true')
p.add_argument('--reference-check', action='store_true', help='check generic projection outputs outside timing')
p.add_argument('--table', action='append', default=[], help='mode:path for a same-identity offline table override')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
batches = [int(x) for x in a.batches.split(',')]
modes = a.modes.split(',')
torch.set_num_threads(8); P.on()
assert torch.version.hip and torch.cuda.device_count() == 1
max_b = max(batches); stride = a.context + 256
os.environ['STARTLUX_PRISM_AMD_GEMM'] = 'auto'
os.environ['STARTLUX_PRISM_AMD_W4A16'] = 'auto'
eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision,
    max_tokens=max(a.max_tokens, a.context), graphs=False, prefix_cache=0,
    think_seqs=max_b, think_kv_rows=max_b * stride))
dec = Decoder(eng, a.gguf, max_seqs=max_b, graphs=False,
              sampling=Sampling(greedy=True), max_len=stride)
graph_sets = {mode: ({}, None) for mode in modes}
if a.reference_check: graph_sets.setdefault('reference', ({}, None))
active = modes[0]
overrides = {}
for item in a.table:
    label, path = item.split(':', 1)
    assert label in modes and label not in ('auto','reference'), label
    overrides[label] = json.loads(Path(path).read_text())
if a.precision == 'w4a16':
    from startlux_prism.kernels import amd_w4a16_dispatch as D16
    original_table = D16.table
    def selected_table():
        return overrides[active]['entries'] if active in overrides else original_table()
    D16.table = selected_table
    identity = D16.kernel_identity()
else:
    original_table = D.table
    def selected_table(family='auto'):
        if active not in overrides: return original_table(family)
        data = overrides[active]
        return data['entries'] if family == 'auto' else data.get('family_entries', {}).get(family, {})
    D.table = selected_table
    identity = D.kernel_identity()
for data in overrides.values():
    assert data['kernel_identity'] == identity, 'table override arithmetic identity differs'
    import importlib.metadata
    runtime = dict(architecture=architecture(),torch=importlib.metadata.version('torch'),
        triton=importlib.metadata.version('triton'),hip=torch.version.hip)
    assert data['runtime'] == runtime, 'table override runtime differs'
rng = np.random.default_rng(0)
prompts = [rng.integers(1000, 100000, a.context).tolist() for _ in range(max_b)]
result = dict(protocol='state-reset-paired-v1', precision=a.precision, context=a.context,
    warm_steps=8, timed_steps=64, repeats=a.repeats, graphs=not a.eager,
    logit_steps=[0, 8, 32, 71], measurements={}, checks={}, profiles={},
    max_tokens=eng.max_tokens, load_s=eng.load_s)
result['table_overrides'] = {k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in overrides.items()}
storages = set()
def storage_bytes(value):
    if isinstance(value,torch.Tensor):
        if not value.is_cuda: return 0
        storage=value.untyped_storage();key=storage.data_ptr()
        if key in storages: return 0
        storages.add(key);return storage.nbytes()
    if isinstance(value,dict): return sum(storage_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(storage_bytes(v) for v in value)
    if dataclasses.is_dataclass(value): return sum(storage_bytes(getattr(value,f.name)) for f in dataclasses.fields(value))
    return 0
result['resident_storage_bytes'] = {
    'layer_weights':storage_bytes(eng.layers),
    'embedding_norm_letter_rows':storage_bytes([eng.embed,eng.out_norm,eng.letters]),
    'full_generation_head':storage_bytes([dec.head_w,dec.head_s]),
    'state_and_kv':storage_bytes([eng.state,eng.conv_state,eng.trans,eng.kv]),
    'prefill_buffers':storage_bytes(eng.buf),
    'decode_buffers':storage_bytes([dec.buf,dec.attn_scratch])}
result['full_vocabulary_size'] = dec.vocab
result['allocated_before_capture'] = torch.cuda.memory_allocated()
references = {}

assert Path(W.__file__).resolve().is_relative_to(repo.resolve()), 'benchmark imported the wrong source tree'
result['source_root'] = str(repo.resolve())

def exact(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()

def phase(name, **kw):
    (out/'phase.json').write_text(json.dumps(dict(phase=name, **kw)))

def save():
    result['peak_allocated'] = torch.cuda.max_memory_allocated()
    result['peak_reserved'] = torch.cuda.max_memory_reserved()
    (out/'model-ab.json').write_text(json.dumps(result, indent=2)+'\n')

def activate(mode):
    global active
    graph_sets[active] = (dec._graphs, dec._pool)
    dec._graphs, dec._pool = graph_sets[mode]
    active = mode
    dispatch_mode = 'auto' if mode in overrides else mode
    os.environ['STARTLUX_PRISM_AMD_GEMM'] = dispatch_mode
    if a.precision == 'w4a16': os.environ['STARTLUX_PRISM_AMD_W4A16'] = dispatch_mode
    dec.graphs = not a.eager

def rows(B, k):
    return [Row(eng.think_slot0+i, eng.think_kv0+i*stride,
        a.context-1+k, a.context-1+k, 1234+i, k,
        i if k else -1-prompts[i][-1]) for i in range(B)]

def reset(B):
    chunk = max(1, eng.max_tokens // a.context)
    for first in range(0, B, chunk):
        keep = [(prompts[i][:-1], eng.think_slot0+i, eng.think_kv0+i*stride)
                for i in range(first, min(first+chunk, B))]
        eng.finish(eng.launch(eng.prepare([], 0, keep=keep)))

def measured(B):
    reset(B); tokens = []
    for k in range(8):
        tokens.append(dec.finish(dec.launch(rows(B, k), k % 2)))
    torch.cuda.synchronize(); start = time.perf_counter(); prev = None
    for k in range(8, 72):
        step = dec.launch(rows(B, k), k % 2)
        if prev is not None: tokens.append(dec.finish(prev))
        prev = step
    tokens.append(dec.finish(prev)); seconds = time.perf_counter()-start
    values = np.stack(tokens)
    return dict(ms_per_step=seconds*1000/64, tokens_s=B*64/seconds,
                token_sha256=hashlib.sha256(values.tobytes()).hexdigest()), values

def full_check(B):
    reset(B); tokens=[]; logits=[]
    for k in range(72):
        tokens.append(dec.finish(dec.launch(rows(B, k), k % 2)))
        if k in result['logit_steps']: logits.append(dec.logits(B).cpu().numpy().copy())
    tokens, logits = np.stack(tokens), np.stack(logits)
    assert np.isfinite(logits).all(), (B, 'non-finite full-vocabulary logits')
    assert (tokens >= 0).all() and (tokens < dec.vocab).all(), (B, 'sampled token outside vocabulary')
    return tokens, logits

for B in batches:
    result['measurements'][str(B)] = {}
    for mode in modes:
        activate(mode); phase('warmup', mode=mode, batch=B); measured(B)
        result['measurements'][str(B)][mode] = dict(samples=[])
    for rep in range(a.repeats):
        for mode in modes if rep % 2 == 0 else list(reversed(modes)):
            activate(mode); phase('timed-decode', mode=mode, batch=B, repeat=rep)
            timing, tokens = measured(B)
            key = (B, mode)
            if key in references: assert exact(tokens, references[key])
            else: references[key] = tokens
            samples = result['measurements'][str(B)][mode]['samples']; samples.append(timing)
            result['measurements'][str(B)][mode].update(
                median_ms=statistics.median(x['ms_per_step'] for x in samples),
                median_tokens_s=statistics.median(x['tokens_s'] for x in samples))
            save()
    original = None
    for mode in modes:
        activate(mode); phase('full-logits-not-timed', mode=mode, batch=B)
        tokens, logits = full_check(B)
        assert exact(tokens, references[(B, mode)])
        np.savez(out/f'outputs-{mode}-B{B}.npz', tokens=tokens, logits=logits)
        if original is None: original = (tokens, logits)
        assert all(exact(x,y) for x,y in zip((tokens,logits),original))
        if a.baseline:
            with np.load(a.baseline/f'outputs-auto-B{B}.npz') as old:
                assert exact(tokens, old['tokens']), (mode,B,'baseline tokens')
                assert exact(logits, old['logits']), (mode,B,'baseline logits')
        result['checks'][f'{mode}-B{B}'] = dict(tokens_exact=True, full_logits_exact=True,
            logits_shape=list(logits.shape), compared_to_baseline=bool(a.baseline))
    if a.reference_check and 'reference' not in modes:
        activate('reference'); phase('reference-check-not-timed', batch=B)
        tokens, logits = full_check(B)
        assert all(exact(x,y) for x,y in zip((tokens,logits),original)), (B,'generic reference')
        np.savez(out/f'outputs-reference-B{B}.npz',tokens=tokens,logits=logits)
        result['checks'][f'reference-B{B}'] = dict(tokens_exact=True,full_logits_exact=True,
            logits_shape=list(logits.shape),timed=False)
    print('PERF',B,{k:round(v['median_ms'],4) for k,v in result['measurements'][str(B)].items()},flush=True)
    save()

activate(modes[0]); phase('baseline-return', batch=8 if 8 in batches else batches[0])
B = 8 if 8 in batches else batches[0]
timing,tokens = measured(B); assert exact(tokens,references[(B,modes[0])])
result['return_measurement'] = timing

if a.capture or a.capture_decode_only:
    census=collections.Counter(); fixtures={}; weights={}; locations={}
    for layer,lw in enumerate(eng.layers):
        for role in ('inp','out','qkv','o','gu','down'):
            for w,offset in lw.get(role,[]):
                locations[w.q.data_ptr()] = dict(layer=layer,role=role,offset=offset)
    original=W.gemm; current_phase='decode'
    def observe(x,scale,xs,w,dest,live,epi=0,cfg=None,pdl=0):
        if a.capture_decode_only and current_phase!='decode':
            return original(x,scale,xs,w,dest,live,epi=epi,cfg=cfg,pdl=pdl)
        key=f'{w.kind}:{x.shape[0]}:{w.n}:{w.k}:{epi}:{dest.stride(0)}'
        location=locations.get(w.q.data_ptr(),{})
        census[(current_phase,key,json.dumps(location,sort_keys=True),int(live.item()))]+=1
        if key not in fixtures or (fixtures[key]['live']==0 and int(live.item())>0):
            wkey=json.dumps(dict(kind=w.kind,N=w.n,K=w.k,**location),sort_keys=True)
            if wkey not in weights:
                weights[wkey]={n:(v.cpu() if isinstance(v,torch.Tensor) else v) for n,v in w._asdict().items()}
            fixtures[key]=dict(weight=wkey,xq=x.cpu(),scale=scale.cpu() if scale is not None else None,xs=xs.cpu() if xs is not None else None,
                old_output=dest[:x.shape[0],:w.n].cpu(),live=int(live.item()),epi=epi,stride0=dest.stride(0),phase=current_phase,**location)
        return original(x,scale,xs,w,dest,live,epi=epi,cfg=cfg,pdl=pdl)
    phase('capture-not-timed');W.gemm=observe
    try:
        for B in batches:
            current_phase='prefill';reset(B)
            dec.graphs=False;current_phase='decode';dec.finish(dec.launch(rows(B,0),0))
    finally:W.gemm=original;dec.graphs=not a.eager
    torch.save(dict(weights=weights,fixtures=fixtures),out/'fixtures.pt')
    (out/'shape-census.json').write_text(json.dumps([dict(phase=k[0],key=k[1],location=json.loads(k[2]),live_rows=k[3],calls=v) for k,v in census.items()],indent=2)+'\n')
    result['fixture_keys']=len(fixtures)

if a.profile:
    from torch.profiler import profile,ProfilerActivity
    for mode in modes:
        activate(mode);B=8 if 8 in batches else batches[0];reset(B)
        for k in range(8):dec.finish(dec.launch(rows(B,k),k%2))
        torch.cuda.synchronize();phase('profile-not-timed',mode=mode,batch=B)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for k in range(8,11):dec.finish(dec.launch(rows(B,k),k%2))
            torch.cuda.synchronize()
        path=out/f'trace-{mode}-B{B}.json';prof.export_chrome_trace(str(path))
        trace=json.loads(path.read_text())['traceEvents']
        times=collections.Counter();counts=collections.Counter()
        for e in trace:
            if e.get('cat')=='kernel' and e.get('ph')=='X':times[e['name']]+=e['dur']/3000;counts[e['name']]+=1
        result['profiles'][mode]=[dict(name=k,ms_per_step=v,calls_per_step=counts[k]/3) for k,v in times.most_common()]
result['passed']=True;save();print('PASS paired decode',flush=True)
