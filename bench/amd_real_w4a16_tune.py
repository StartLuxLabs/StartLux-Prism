"""Same-profile FP16 metadata tuning on real 9B inputs, with independent FP64 bounds."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import statistics
import sys
import torch
import triton
import triton.language as tl
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from startlux_prism.kernels import w4a8 as W,amd_w4a16 as H,amd_w4a16_meta as C,amd_w4a16_dispatch as D
from startlux_prism.backends import architecture
from amd_table_selection import select_row

@triton.jit
def scrub(X,N:tl.constexpr):
    i=tl.program_id(0)*4096+tl.arange(0,4096)
    v=tl.load(X+i,i<N,0);tl.store(X+i,v^1,i<N)

def equal(x,y):return x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))

def dequant(w,first,last):
    q=w.q[first:last];n=last-first
    sc=w.sc[:,first:last].permute(1,0,2).float()
    d=w.d[:,first:last].T.float().repeat_interleave(256,dim=1)
    if w.kind=='q4k':
        packed=q.reshape(n,w.k//32,16).int()
        values=torch.cat((packed&15,packed>>4),dim=-1).reshape(n,w.k).float()
        sc=sc.reshape(n,-1).repeat_interleave(32,dim=1)
        mn=w.mn[:,first:last].permute(1,0,2).reshape(n,-1).float().repeat_interleave(32,dim=1)
        dm=w.dm[:,first:last].T.float().repeat_interleave(256,dim=1)
        return ((d*sc)*values-dm*mn).half().double()
    sc=sc.reshape(n,-1).repeat_interleave(16,dim=1)
    return ((d*sc)*q.float()).half().double()

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--phase',choices=('all','decode','prefill'),default='all')
p.add_argument('--skip-existing',action='store_true')
p.add_argument('--kind',choices=('all','q4k','q6k'),default='all')
p.add_argument('--extended-q6',action='store_true',help='also qualify K128/K256 metadata candidates')
p.add_argument('--exclude-qualified',type=Path,action='append',default=[])
p.add_argument('--quarantine-reference-failures',action='store_true',help='retain the original path for keys failing the unchanged independent bound')
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);assert torch.version.hip and torch.cuda.device_count()==1
assert architecture()=='gfx1201','This qualification targets the gfx1201 table'
data=torch.load(a.fixtures,map_location='cpu',weights_only=False)
weights={name:W.W4(**{k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in d.items()}) for name,d in data['weights'].items()}
flush=torch.zeros(128*1024*1024,dtype=torch.int32,device='cuda')
report=dict(protocol='real-weight-seven-paired-v1',profile='w4a16',experiment=out.name,
    runtime=dict(architecture=architecture(),torch=importlib.metadata.version('torch'),triton=importlib.metadata.version('triton'),hip=torch.version.hip),
    kernel_identity=D.kernel_identity(),rows=[],quarantined=[],completed=False,
    independent_bound='1e-5 + 5e-6 * sum(abs(x*w))',
    fixture_sha256=hashlib.file_digest(a.fixtures.open('rb'),'sha256').hexdigest())
def save():(out/'tuning.json').write_text(json.dumps(report,indent=2)+'\n')
excluded=set()
for path in a.exclude_qualified:
    prior=json.loads(path.read_text())
    assert prior['completed'] and prior['kernel_identity']==report['kernel_identity'] and prior['runtime']==report['runtime']
    excluded.update(row['key'] for row in prior['rows'])
    for blocked in prior.get('quarantined',[]):
        excluded.add(blocked['key'])
        report['quarantined'].append(dict(blocked, inherited_from=str(path)))
decode_configs=[(16,32,4,1,64),(16,64,4,1,64),(32,32,4,1,64),(32,64,4,1,64),
         (16,32,4,1,32),(16,64,4,1,32),(32,64,4,1,32),(16,64,8,1,64)]
for key,f in sorted(data['fixtures'].items()):
    if key.replace(':','|') in excluded:continue
    x=f['xq'].cuda();assert x.dtype==torch.float16
    M=x.shape[0];w=weights[f['weight']];N=w.n;ld=f['stride0'];epi=f['epi']
    if a.kind!='all' and w.kind!=a.kind:continue
    if a.phase=='decode' and M>32 or a.phase=='prefill' and M<=32:continue
    if a.skip_existing and key.replace(':','|') in D.table():continue
    configs=decode_configs
    if M>32:
        configs=[(32,32,4,1,64),(32,64,4,1,64),(64,64,4,1,64),(64,128,8,1,64),
            (32,64,4,1,32),(64,64,4,1,32),(64,128,4,1,32),(64,128,8,1,32)]
    if a.extended_q6 and w.kind=='q6k':
        configs=configs+[(16,32,4,1,128),(16,64,4,1,128),(16,64,4,1,256),
            (32,32,4,1,256),(32,64,4,1,256)]
    dtype=torch.float32 if epi else torch.float16
    prior=torch.full(((M+1)*ld+16,),.125,dtype=dtype,device='cuda')
    old=f['old_output'][:M].to(device='cuda',dtype=dtype)
    if epi==0:old.fill_(.125)
    recorded_live=min(M,f['live'])
    assert recorded_live>0,(key,'fixture has no real live activation')
    if recorded_live<M:
        x[recorded_live:].copy_(x[:1].expand(M-recorded_live,-1));old[recorded_live:].fill_(.125)
    assert torch.isfinite(x).all() and torch.isfinite(old).all(),key
    torch.as_strided(prior,(M,N),(ld,1)).copy_(old if epi else torch.full_like(old,.125))
    backing=prior.clone();y=torch.as_strided(backing,(M,N),(ld,1));live=torch.tensor([M],dtype=torch.int32,device='cuda')
    reference_cfg=(16,32,4,1,64)
    def original():return H.gemm(x,w,y,live,epi,cfg=reference_cfg)
    original();expected=backing.clone()
    fp32=torch.zeros((M,N),dtype=torch.float32,device='cuda')
    H.gemm(x,w,fp32,live,0,cfg=reference_cfg)
    reference_failure=None
    for first in range(0,N,512):
        last=min(first+512,N);rw=dequant(w,first,last);xd=x.double()
        mathematical=xd@rw.T;bound=1e-5+5e-6*(xd.abs()@rw.abs().T)
        error=(fp32[:,first:last].double()-mathematical).abs();ratio=error/bound
        if not bool(torch.all(error<=bound)):
            index=int(ratio.argmax());row_index,column=divmod(index,last-first)
            reference_failure=dict(key=key.replace(':','|'),reason='original_reference_independent_bound',
                chunk=[first,last],failures_in_chunk=int((error>bound).sum()),
                worst_index=[row_index,first+column],max_ratio=float(ratio.max()),
                absolute_error=float(error[row_index,column]),bound=float(bound[row_index,column]),
                mathematical=float(mathematical[row_index,column]),original=float(fp32[row_index,first+column]),
                action='retain original dispatch; no candidate promotion',recorded_live_rows=recorded_live)
            break
    if reference_failure is not None:
        report['quarantined'].append(reference_failure);save()
        print('QUARANTINED W4A16',json.dumps(reference_failure),flush=True)
        if not a.quarantine_reference_failures:raise AssertionError((key,'independent FP64 bound'))
        continue
    functions={'reference':original};specs={'reference':dict(implementation='reference',cfg=list(reference_cfg))}
    for cfg in configs:
        name='meta-'+'-'.join(map(str,cfg));functions[name]=lambda cfg=cfg:C.gemm(x,w,y,live,epi,cfg)
        specs[name]=dict(implementation='meta',cfg=list(cfg))
    candidates={};valid={}
    for name,fn in functions.items():
        (out/'phase.json').write_text(json.dumps(dict(phase='qualify-w4a16',key=key,candidate=name)))
        backing.copy_(prior);kernel=fn();ok=equal(backing,expected)
        for active in sorted({0,1,max(1,M//2+1),M-1,M}):
            live.fill_(active);backing.copy_(prior);fn();want=prior.clone()
            view=torch.as_strided(want,(M,N),(ld,1));refview=torch.as_strided(expected,(M,N),(ld,1))
            view[:active].copy_(refview[:active]);ok=ok and equal(backing,want)
        live.fill_(M)
        if name!='reference':
            probe=torch.empty_like(fp32);C.gemm(x,w,probe,live,0,tuple(specs[name]['cfg']))
            ok=ok and equal(probe,fp32)
        candidates[name]=dict(correct=ok,**specs[name])
        if kernel is not None:
            asm=kernel.asm['amdgcn'];digest=hashlib.sha256(kernel.asm['hsaco']).hexdigest()
            folder=out/'isa';folder.mkdir(exist_ok=True)
            (folder/(digest+'.amdgcn')).write_text(asm);(folder/(digest+'.hsaco')).write_bytes(kernel.asm['hsaco'])
            resources={}
            for field in ('vgpr_count','sgpr_count','private_segment_fixed_size','vgpr_spill_count','group_segment_fixed_size'):
                match=re.search(r'\.'+field+r':\s*(\d+)',asm);resources[field]=int(match.group(1)) if match else None
            resources.update(hsaco_sha256=digest,static_wmma_sites=asm.count('v_wmma_'))
            candidates[name]['resources']=resources
        if ok:valid[name]=fn
        else:candidates[name].update(cold_ms=0,hot_ms=0,partial_ms=0)
    assert 'reference' in valid
    graphs={}
    for name,fn in valid.items():
        backing.copy_(prior);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs[name]=g
    samples={name:{r:[] for r in ('cold','hot','partial')} for name in graphs}
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(7):
        order=list(graphs);random.Random(7101+rep).shuffle(order)
        for name in order:
            for regime in ('cold','hot','partial') if rep%2==0 else ('partial','hot','cold'):
                live.fill_(max(1,M//2+1) if regime=='partial' else M);backing.copy_(prior)
                if regime=='hot':
                    for _ in range(3):graphs[name].replay()
                else:scrub[(triton.cdiv(flush.numel(),4096),)](flush,flush.numel())
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name][regime].append(start.elapsed_time(end))
    for name,values in samples.items():
        candidates[name].update(samples=values,**{r+'_ms':statistics.median(v) for r,v in values.items()})
    row=dict(key=key.replace(':','|'),baseline='reference',candidates=candidates,independent_fp64_bound=True,
             complete_fp32_bytes=True,shape=[M,N,w.k],epi=epi,stride0=ld,recorded_live_rows=recorded_live)
    row['selection']=select_row(row);report['rows'].append(row);save()
    print('TUNED W4A16',key,row['selection']['winner'],flush=True)
report['completed']=True;save();print('PASS W4A16 same-profile tuning',len(report['rows']),flush=True)
