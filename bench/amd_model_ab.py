"""State-reset, paired single-GPU decode benchmark with full-output evidence."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from startlux_prism import progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.engine.decode import Decoder, Row, Sampling, DEC_BUCKETS
from startlux_prism.kernels import w4a8 as W, amd_w4a8_dispatch as D

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf', required=True)
p.add_argument('--model-dir', required=True)
p.add_argument('--output', required=True)
p.add_argument('--precision', default='w4a8', choices=('w4a8', 'w4a16'))
p.add_argument('--batches', default='1,2,4,8,9,16')
p.add_argument('--modes', default='auto,reference')
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--context', type=int, default=512)
p.add_argument('--eager', action='store_true')
p.add_argument('--baseline', type=Path)
p.add_argument('--capture', action='store_true')
p.add_argument('--profile', action='store_true')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
batches = [int(x) for x in a.batches.split(',')]
modes = a.modes.split(',')
torch.set_num_threads(8); P.on()
assert torch.version.hip and torch.cuda.device_count() == 1
max_b = max(batches); stride = a.context + 256
os.environ['STARTLUX_PRISM_AMD_GEMM'] = 'auto'
eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision,
    max_tokens=max(8192, a.context * max_b), graphs=False, prefix_cache=0,
    think_seqs=max_b, think_kv_rows=max_b * stride))
dec = Decoder(eng, a.gguf, max_seqs=max_b, graphs=False,
              sampling=Sampling(greedy=True), max_len=stride)
graph_sets = {mode: ({}, None) for mode in modes}
active = modes[0]
rng = np.random.default_rng(0)
prompts = [rng.integers(1000, 100000, a.context).tolist() for _ in range(max_b)]
result = dict(protocol='state-reset-paired-v1', precision=a.precision, context=a.context,
    warm_steps=8, timed_steps=64, repeats=a.repeats, graphs=not a.eager,
    logit_steps=[0, 8, 32, 71], measurements={}, checks={}, profiles={})
references = {}

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
    os.environ['STARTLUX_PRISM_AMD_GEMM'] = mode
    dec.graphs = not a.eager

def rows(B, k):
    return [Row(eng.think_slot0+i, eng.think_kv0+i*stride,
        a.context-1+k, a.context-1+k, 1234+i, k,
        i if k else -1-prompts[i][-1]) for i in range(B)]

def reset(B):
    keep = [(prompts[i][:-1], eng.think_slot0+i, eng.think_kv0+i*stride) for i in range(B)]
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
    return np.stack(tokens), np.stack(logits)

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
            if key in references: assert np.array_equal(tokens, references[key])
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
        assert np.array_equal(tokens, references[(B, mode)])
        np.savez(out/f'outputs-{mode}-B{B}.npz', tokens=tokens, logits=logits)
        if original is None: original = (tokens, logits)
        assert all(np.array_equal(x,y) for x,y in zip((tokens,logits),original))
        if a.baseline:
            with np.load(a.baseline/f'outputs-auto-B{B}.npz') as old:
                assert np.array_equal(tokens, old['tokens']), (mode,B,'baseline tokens')
                assert np.array_equal(logits, old['logits']), (mode,B,'baseline logits')
        result['checks'][f'{mode}-B{B}'] = dict(tokens_exact=True, full_logits_exact=True,
            logits_shape=list(logits.shape), compared_to_baseline=bool(a.baseline))
    print('PERF',B,{k:round(v['median_ms'],4) for k,v in result['measurements'][str(B)].items()},flush=True)
    save()

activate(modes[0]); phase('baseline-return', batch=8 if 8 in batches else batches[0])
B = 8 if 8 in batches else batches[0]
timing,tokens = measured(B); assert np.array_equal(tokens,references[(B,modes[0])])
result['return_measurement'] = timing

if a.capture:
    census=collections.Counter(); fixtures={}; weights={}; locations={}
    for layer,lw in enumerate(eng.layers):
        for role in ('inp','out','qkv','o','gu','down'):
            for w,offset in lw.get(role,[]):
                locations[w.q.data_ptr()] = dict(layer=layer,role=role,offset=offset)
    original=W.gemm; current_phase='decode'
    def observe(x,scale,xs,w,dest,live,epi=0,cfg=None,pdl=0):
        key=f'{w.kind}:{x.shape[0]}:{w.n}:{w.k}:{epi}:{dest.stride(0)}'
        location=locations.get(w.q.data_ptr(),{})
        census[(current_phase,key,json.dumps(location,sort_keys=True))]+=1
        if key not in fixtures:
            wkey=f'{w.kind}:{w.n}:{w.k}'
            if wkey not in weights:
                weights[wkey]={n:(v.cpu() if isinstance(v,torch.Tensor) else v) for n,v in w._asdict().items()}
            fixtures[key]=dict(weight=wkey,xq=x.cpu(),scale=scale.cpu(),xs=xs.cpu() if xs is not None else None,
                old_output=dest[:,:w.n].cpu(),live=int(live.item()),epi=epi,stride0=dest.stride(0),phase=current_phase,**location)
        return original(x,scale,xs,w,dest,live,epi=epi,cfg=cfg,pdl=pdl)
    phase('capture-not-timed');W.gemm=observe
    try:
        for B in batches:
            current_phase='prefill';reset(B)
            dec.graphs=False;current_phase='decode';dec.finish(dec.launch(rows(B,0),0))
    finally:W.gemm=original;dec.graphs=not a.eager
    torch.save(dict(weights=weights,fixtures=fixtures),out/'fixtures.pt')
    (out/'shape-census.json').write_text(json.dumps([dict(phase=k[0],key=k[1],location=json.loads(k[2]),calls=v) for k,v in census.items()],indent=2)+'\n')
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
