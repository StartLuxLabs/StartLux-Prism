"""Real-weight candidate qualification with paired cold/hot/partial-row measurements."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys

import torch
import triton
import triton.language as tl

repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from startlux_prism.kernels import w4a8 as W, amd_w4a8 as G, amd_w4a8_dot as DOT
from startlux_prism.kernels import amd_q4_prefill as Q4, amd_q6_wmma as Q6, amd_w4a8_dispatch as D
from startlux_prism.backends import architecture
from amd_table_selection import select_row

@triton.jit
def scrub(X,N:tl.constexpr):
    i=tl.program_id(0)*4096+tl.arange(0,4096)
    x=tl.load(X+i,i<N,0);tl.store(X+i,x^1,i<N)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--phase',choices=('decode','prefill'),default='decode')
p.add_argument('--round',type=int,choices=(1,2),default=1)
p.add_argument('--limit',type=int)
p.add_argument('--exclude-qualified',type=Path,action='append',default=[],help='skip keys already qualified with this kernel identity')
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);torch.manual_seed(7101)
assert torch.version.hip and torch.cuda.device_count()==1
assert architecture()=='gfx1201','This qualification targets the gfx1201 table'
payload=torch.load(a.fixtures,map_location='cpu',weights_only=False)
runtime=dict(architecture=architecture(),torch=importlib.metadata.version('torch'),
             triton=importlib.metadata.version('triton'),hip=torch.version.hip)
result=dict(protocol='real-weight-seven-paired-v1',experiment=out.name,round=a.round,
    runtime=runtime,kernel_identity=D.kernel_identity(),oracle_row_chunk=128,completed=False,rows=[],failures=[],
    fixture_sha256=hashlib.file_digest(a.fixtures.open('rb'),'sha256').hexdigest())
flush=torch.zeros(128*1024*1024,dtype=torch.int32,device='cuda')
weights={}
for name,data in payload['weights'].items():
    weights[name]=W.W4(**{k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in data.items()})

def equal(left,right):
    return left.dtype==right.dtype and torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))

def save():
    (out/'tuning.json').write_text(json.dumps(result,indent=2)+'\n')

active_candidate=None
resource_records={}
class CaptureKernel:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def run(*args,**kwargs):
            compiled=launch(*args,**kwargs)
            if active_candidate is not None and active_candidate not in resource_records:
                asm=compiled.asm['amdgcn']
                digest=hashlib.sha256(compiled.asm['hsaco']).hexdigest()
                folder=out/'isa';folder.mkdir(exist_ok=True)
                (folder/(digest+'.amdgcn')).write_text(asm)
                (folder/(digest+'.hsaco')).write_bytes(compiled.asm['hsaco'])
                fields={}
                for field in ('vgpr_count','sgpr_count','private_segment_fixed_size','vgpr_spill_count','group_segment_fixed_size'):
                    match=re.search(r'\.'+field+r':\s*(\d+)',asm)
                    fields[field]=int(match.group(1)) if match else None
                fields.update(hsaco_sha256=digest,static_wmma_sites=asm.count('v_wmma_'),static_dot4_sites=asm.count('v_dot4_'))
                resource_records[active_candidate]=fields
            return compiled
        return run
for module,name in ((G,'_gemv'),(DOT,'_gemv_dot'),(W,'_q4k_kernel'),(W,'_q6k_kernel'),(Q4,'_q4_loop'),(Q6,'_q6_scaled')):
    setattr(module,name,CaptureKernel(getattr(module,name)))

def phase(name,**kw):
    (out/'phase.json').write_text(json.dumps(dict(phase=name,**kw)))

records=[(k,f) for k,f in payload['fixtures'].items() if (f['xq'].shape[0]<=32)==(a.phase=='decode')]
excluded=set()
for path in a.exclude_qualified:
    prior=json.loads(path.read_text())
    assert prior['completed'] and prior['kernel_identity']==result['kernel_identity'] and prior['runtime']==runtime
    excluded.update(row['key'] for row in prior['rows'])
records=[(k,f) for k,f in records if k not in excluded]
records.sort(key=lambda item:(item[1]['xq'].shape[0],item[0]))
if a.limit:records=records[:a.limit]
for key,f in records:
    phase('qualify',key=key)
    w=weights[f['weight']];M=f['xq'].shape[0];N=w.n;K=w.k;epi=f['epi'];ldo=f['stride0']
    x=f['xq'].cuda();scale=f['scale'].cuda();xs=f['xs'].cuda()
    dtype=torch.bfloat16 if epi==0 else torch.float32
    # A view with the exact production row stride, plus allocation canaries.
    flat=torch.full(((M+1)*ldo+16,),17.,device='cuda',dtype=dtype)
    y=torch.as_strided(flat,(M,N),(ldo,1),storage_offset=0)
    old=f['old_output'][:M].to(device='cuda',dtype=dtype)
    if epi==0:old.fill_(0.125)
    recorded_live=min(M,f['live'])
    assert recorded_live>0,(key,'fixture has no real live activation; recapture before qualification')
    if recorded_live<M:
        x[recorded_live:].copy_(x[:1].expand(M-recorded_live,-1))
        scale[:,recorded_live:].copy_(scale[:,:1].expand(-1,M-recorded_live))
        xs[:,recorded_live:].copy_(xs[:,:1].expand(-1,M-recorded_live,-1))
        old[recorded_live:].fill_(0.125)
    assert torch.isfinite(scale).all() and torch.isfinite(old).all(),key
    live=torch.tensor([M],device='cuda',dtype=torch.int32)
    funcs={};specs={}
    def add(name,implementation,cfg,fn):
        funcs[name]=fn;specs[name]=dict(implementation=implementation,cfg=list(cfg) if cfg else None)
    args=(x,scale,xs,w,y,live)
    if a.phase=='decode':add('gemv','gemv',None,lambda:G.gemv(*args,epi=epi))
    add('wmma','wmma',None,lambda:W.gemm_triton(*args,epi=epi))
    if a.phase=='decode':
        for bm in (1,2,4,8):
            cfg=(bm,8,4);add(f'dot-{bm}-8-4','dot',cfg,lambda cfg=cfg:DOT.gemv(*args,epi=epi,cfg=cfg))
    for cfg in ((16,32,4,1),(16,64,4,1),(16,64,4,2)):
        add('wmma-'+'-'.join(map(str,cfg)),'wmma',cfg,lambda cfg=cfg:W.gemm_triton(*args,epi=epi,cfg=cfg))
    if w.kind=='q4k':
        cfg=(16 if a.phase=='decode' else 64,64,4,1,1)
        add('loop-'+'-'.join(map(str,cfg)),'loop',cfg,lambda cfg=cfg:Q4.gemm(*args,epi=epi,cfg=cfg))
    else:
        cfg=(16 if a.phase=='decode' else 64,64,4,1)
        add('q6scaled-'+'-'.join(map(str,cfg)),'q6scaled',cfg,lambda cfg=cfg:Q6.gemm(*args,epi=epi,cfg=cfg))
    if a.round==2:
        if a.phase=='decode':
            for cfg in ((1,8,2),(1,16,2),(1,16,4),(2,8,2),(4,8,2),(8,8,2),(4,16,4),(8,16,4)):
                add('dot-'+'-'.join(map(str,cfg)),'dot',cfg,lambda cfg=cfg:DOT.gemv(*args,epi=epi,cfg=cfg))
        else:
            for cfg in ((32,32,4,1),(32,64,4,1),(64,32,4,1),(64,64,4,1)):
                add('wmma-'+'-'.join(map(str,cfg)),'wmma',cfg,lambda cfg=cfg:W.gemm_triton(*args,epi=epi,cfg=cfg))
        if w.kind=='q6k':
            for cfg in ((16,32,4,1),(16,32,8,1),(16,64,8,1),(32,32,4,1)):
                add('q6scaled-'+'-'.join(map(str,cfg)),'q6scaled',cfg,lambda cfg=cfg:Q6.gemm(*args,epi=epi,cfg=cfg))
        else:
            for cfg in ((16,32,4,1,1),(16,32,4,1,2),(16,64,4,1,2)):
                add('loop-'+'-'.join(map(str,cfg)),'loop',cfg,lambda cfg=cfg:Q4.gemm(*args,epi=epi,cfg=cfg))
    baseline='gemv' if M<=8 else 'wmma'
    if a.round==2:
        previous=D.pick(x,w,y,epi)
        if previous is not None:
            add('current',previous['implementation'],previous.get('cfg'),lambda:D.run(*args,epi=epi))
            baseline='current'
    if a.round==2 and a.phase=='decode':
        keep={baseline,'wmma-16-64-4-2'}
        keep.update('dot-'+'-'.join(map(str,c)) for c in ((1,8,2),(1,16,2),(1,16,4),(2,8,2),(4,8,2),(8,8,2),(4,16,4),(8,16,4)))
        keep.update(n for n in funcs if n.startswith(('loop-','q6scaled-')))
        funcs={n:f for n,f in funcs.items() if n in keep}
        specs={n:v for n,v in specs.items() if n in keep}
    resource_records={}
    # Cross-check the entire candidate output against the frozen arithmetic path.
    flat.fill_(17);y.copy_(old);funcs[baseline]();torch.cuda.synchronize();reference=flat.clone()
    # Independent mathematical oracle for all columns, processed in bounded row chunks.
    for first in range(0,M,128):
        last=min(M,first+128)
        oracle=W.gemm_ref(x[first:last],scale[:,first:last].contiguous(),xs[:,first:last].contiguous(),w)
        expected=oracle.to(torch.bfloat16) if epi==0 else old[first:last]+oracle
        assert equal(y[first:last],expected),(key,'independent full-output oracle',first)
    checks={};valid={}
    for name,fn in funcs.items():
        active_candidate=name
        try:
            flat.fill_(17);y.copy_(old);fn();torch.cuda.synchronize()
            correct=equal(flat,reference)
            for active_rows in sorted({0,1,max(1,M//2+1),M-1,M}):
                if active_rows<0:continue
                live.fill_(active_rows);flat.fill_(17);y.copy_(old);fn();torch.cuda.synchronize()
                target=torch.full_like(flat,17);view=torch.as_strided(target,(M,N),(ldo,1));view.copy_(old)
                refview=torch.as_strided(reference,(M,N),(ldo,1));view[:active_rows].copy_(refview[:active_rows])
                correct=correct and equal(flat,target)
            live.fill_(M)
            checks[name]=dict(correct=correct,**specs[name])
            if correct:valid[name]=fn
            else:result['failures'].append(dict(key=key,candidate=name,error='output/canary mismatch'))
        except Exception as exc:
            checks[name]=dict(correct=False,error=repr(exc),**specs[name]);live.fill_(M)
            result['failures'].append(dict(key=key,candidate=name,error=repr(exc)))
        save()
    active_candidate=None
    for name,resource in resource_records.items():checks[name]['resources']=resource
    assert baseline in valid,(key,'baseline failed')
    # Identical graph/event timing for every regime; flush/warm launch precedes timing.
    graphs={}
    for name,fn in valid.items():
        live.fill_(M);y.copy_(old);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs[name]=g
    samples={n:{r:[] for r in ('cold','hot','partial')} for n in valid}
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for repeat in range(7):
        order=list(graphs);random.Random(7101+repeat).shuffle(order)
        for name in order:
            phase('measure',key=key,candidate=name,repeat=repeat)
            for regime in ('cold','hot','partial') if repeat%2==0 else ('partial','hot','cold'):
                live.fill_(max(1,M//2+1) if regime=='partial' else M);y.copy_(old)
                if regime=='hot':
                    for _ in range(3):graphs[name].replay()
                else:scrub[(triton.cdiv(flush.numel(),4096),)](flush,flush.numel())
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name][regime].append(start.elapsed_time(end))
    for name,value in checks.items():
        if value['correct']:
            value.update(samples=samples[name],**{r+'_ms':statistics.median(v) for r,v in samples[name].items()})
        else:value.update(cold_ms=0,hot_ms=0,partial_ms=0)
    row=dict(key=key,M=M,N=N,K=K,kind=w.kind,epi=epi,stride0=ldo,family=a.phase,
             baseline=baseline,candidates=checks,weight_bytes=w.nbytes(),oracle_full_output=True,
             recorded_live_rows=recorded_live,real_live_input=True,
             source_location={k:f.get(k) for k in ('phase','scene','layer','role','offset')})
    row['selection']=select_row(row);result['rows'].append(row);save()
    print('TUNED',key,row['selection']['winner'],len(valid),'candidates',flush=True)
    del graphs,funcs,reference,flat
result['completed']=True;save();print('PASS qualified real shapes',len(result['rows']),flush=True)
