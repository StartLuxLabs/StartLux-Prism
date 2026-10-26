"""Capture real projection inputs, then replay distinct layer weights in model order.

Capture is outside all timing. Replay measures projection kernels only and does
not replace the complete-model benchmark. Table overrides are process-local.
"""
import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch

repo=Path(os.environ.get('PRISM_BENCH_REPO',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(repo),str(repo/'bench')]
from startlux_prism.kernels import w4a8 as W,amd_w4a8_dispatch as D
from startlux_prism.backends import architecture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--capture',action='store_true')
p.add_argument('--precision',choices=('w4a8','w4a16'),default='w4a8')
p.add_argument('--readout-only',action='store_true')
p.add_argument('--gguf');p.add_argument('--model-dir')
p.add_argument('--fixtures',type=Path)
p.add_argument('--service-trace',type=Path,help='completed service run whose packed/read buckets must be covered')
p.add_argument('--tables',nargs='*',default=[],help='label:path; use reference:- for generic GEMM')
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);assert torch.version.hip and torch.cuda.device_count()==1

def equal(x,y):
    return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def phase(name,**kw):(out/'phase.json').write_text(json.dumps(dict(phase=name,**kw)))

if a.capture:
    from startlux_prism import jevfmt as J,progress as P
    from startlux_prism.engine.model import Engine,Options,group
    from startlux_prism.engine.decode import Decoder,Row,Sampling
    from startlux_prism.prompt import rows_for
    import suites
    P.on()
    e=Engine(a.gguf,a.model_dir,Options(precision=a.precision,max_tokens=8192,graphs=False,prefix_cache=0,
        think_seqs=8,think_kv_rows=8*768))
    dec=Decoder(e,a.gguf,max_seqs=8,graphs=False,sampling=Sampling(greedy=True),max_len=768)
    weights={};fixtures={};stream=[];census=collections.Counter();locations={}
    seen_T=set();seen_R=set();service_coverage=None
    for layer,lw in enumerate(e.layers):
        for role in ('inp','out','qkv','o','gu','down'):
            for w,offset in lw.get(role,[]):locations[w.q.data_ptr()]=dict(layer=layer,role=role,offset=offset)
    original=W.gemm;current='readout';scene='';capture_stream=False
    def observe(x,scale,xs,w,dest,live,epi=0,**kw):
        key=f'{w.kind}:{x.shape[0]}:{w.n}:{w.k}:{epi}:{dest.stride(0)}'
        location=locations.get(w.q.data_ptr(),{})
        count=int(live.item())
        census[(scene,current,key,count,json.dumps(location,sort_keys=True))]+=1
        item=None
        replace_fixture=key not in fixtures or (fixtures[key]['live']==0 and count>0)
        if replace_fixture or capture_stream:
            wkey=json.dumps(dict(kind=w.kind,N=w.n,K=w.k,**location),sort_keys=True)
            if wkey not in weights:
                weights[wkey]={k:(v.cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in w._asdict().items()}
            item=dict(weight=wkey,xq=x.cpu().clone(),scale=scale.cpu().clone() if scale is not None else None,
                xs=xs.cpu().clone() if xs is not None else None,
                old_output=dest[:x.shape[0],:w.n].cpu().clone(),live=count,epi=epi,stride0=dest.stride(0),
                phase=current,scene=scene,**location)
            if replace_fixture:fixtures[key]=item
        value=original(x,scale,xs,w,dest,live,epi=epi,**kw)
        if capture_stream:
            item['expected']=dest[:x.shape[0],:w.n].cpu().clone();item['key']=key;stream.append(item)
        return value
    def rows(state,qs):
        return rows_for(e.tok,[J.from_systemone(state,q,allow_none=q['type']=='choice') for q in qs.values()],list(qs))
    W.gemm=observe
    try:
        for scene in ('yesno','readme3','startlux3','longpage2','review1'):
            phase('capture-readout',scene=scene);e.letter_logits(rows(*suites.fixed()[scene]))
            seen_T.add(e.last['bucket']);seen_R.add(e.last['read_bucket'])
        q={'availability':{'type':'noul','instructions':'Is this an availability incident?'}}
        for target in (256,2048,8192):
            lo,hi=0,target
            while lo<hi:
                mid=(lo+hi+1)//2
                if len(rows('The service is unavailable.'+' service'*mid,q)[0].ids)<=target:lo=mid
                else:hi=mid-1
            rr=rows('The service is unavailable.'+' service'*lo,q);assert len(rr[0].ids)==target
            scene=f'length-{target}';phase('capture-readout',scene=scene);e.letter_logits(rr)
            seen_T.add(e.last['bucket']);seen_R.add(e.last['read_bucket'])
        if a.service_trace:
            source=json.loads((a.service_trace/'service.json').read_text());assert source['passed']
            events=[json.loads(line) for line in (a.service_trace/'scheduler-events.jsonl').read_text().splitlines()]
            target_T={event['packed_tokens'] for event in events if event['kind']=='prefill'}
            target_R={event['read_rows'] for event in events if event['kind']=='prefill'}
            service_coverage=dict(source=str(a.service_trace),target_T=sorted(target_T),target_R=sorted(target_R),cases=[])
            source_rows=[json.loads(line) for line in (a.service_trace/'requests.jsonl').read_text().splitlines()]
            for source_scene in ('concurrency-16','readme3','startlux3'):
                records=sorted((r for r in source_rows if r['scene']==source_scene and r['repeat']>=0),
                    key=lambda r:r['repeat']+5*r['index'])[:16]
                groups=[group(rows(r['request']['state'],r['request']['questions'])) for r in records]
                for count in range(1,len(groups)+1):
                    _,T,R=e.plan(groups[:count])
                    if not (T in target_T-seen_T or R in target_R-seen_R):continue
                    scene=f'service-{source_scene}-{count}';phase('capture-service-groups',scene=scene,T=T,R=R)
                    e.run(groups[:count]);seen_T.add(e.last['bucket']);seen_R.add(e.last['read_bucket'])
                    service_coverage['cases'].append(dict(scene=scene,T=T,R=R,request_hashes=[r['request_sha256'] for r in records[:count]]))
            # Some scheduler buckets include a mixture of thinking prefills and
            # readouts. Exercise those exact body buckets with text length cases;
            # these are new real activations, not a claim to replay the HTTP state.
            for target in sorted(target_T-seen_T):
                lo,hi=0,target
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if len(rows('The service is unavailable.'+' service'*mid,q)[0].ids)<=target:lo=mid
                    else:hi=mid-1
                rr=rows('The service is unavailable.'+' service'*lo,q)
                _,T,R=e.plan([group(rr)]);assert T==target,(target,T)
                scene=f'service-length-{target}';phase('capture-service-length',scene=scene,T=T,R=R)
                e.letter_logits(rr);seen_T.add(e.last['bucket']);seen_R.add(e.last['read_bucket'])
                service_coverage['cases'].append(dict(scene=scene,T=T,R=R,kind='exact bucket, fresh text activation'))
            assert target_T<=seen_T and target_R<=seen_R,(target_T-seen_T,target_R-seen_R)
            service_coverage.update(covered_T=sorted(seen_T),covered_R=sorted(seen_R),passed=True)
            (out/'service-coverage.json').write_text(json.dumps(service_coverage,indent=2)+'\n')
        if not a.readout_only:
            scene='decode-B8';current='prefill'
            import numpy as np
            rng=np.random.default_rng(0);prompts=[rng.integers(1000,100000,512).tolist() for _ in range(8)]
            e.finish(e.launch(e.prepare([],0,keep=[(ids[:-1],e.think_slot0+i,e.think_kv0+i*768) for i,ids in enumerate(prompts)])))
            current='decode';capture_stream=True;phase('capture-ordered-decode',batch=8)
            rr=[Row(e.think_slot0+i,e.think_kv0+i*768,511,511,1234+i,0,-1-ids[-1]) for i,ids in enumerate(prompts)]
            dec.finish(dec.launch(rr,0))
    finally:W.gemm=original
    assert len(stream)==(0 if a.readout_only else 144),len(stream)
    payload=dict(weights=weights,fixtures=fixtures,stream=stream)
    phase('save-real-fixtures',keys=len(fixtures),ordered_calls=len(stream));torch.save(payload,out/'fixtures.pt')
    (out/'shape-census.json').write_text(json.dumps([dict(scene=k[0],phase=k[1],key=k[2],live_rows=k[3],location=json.loads(k[4]),calls=v) for k,v in census.items()],indent=2)+'\n')
    (out/'capture.json').write_text(json.dumps(dict(passed=True,keys=len(fixtures),weights=len(weights),ordered_calls=len(stream),
        peak_allocated=torch.cuda.max_memory_allocated(),precision=a.precision,
        kind='real readout fixtures' if a.readout_only else 'real readout fixtures and all-layer B8 projection stream'),indent=2)+'\n')
    print('PASS real readout and ordered stream capture',len(fixtures),len(weights),flush=True)
else:
    assert a.fixtures and a.tables
    payload=torch.load(a.fixtures,map_location='cpu',weights_only=False)
    weights={k:W.W4(**{n:(v.cuda() if isinstance(v,torch.Tensor) else v) for n,v in d.items()}) for k,d in payload['weights'].items()}
    calls=[]
    for f in payload['stream']:
        x=f['xq'].cuda();sc=f['scale'].cuda();xs=f['xs'].cuda();M=x.shape[0];w=weights[f['weight']]
        old=f['old_output'].cuda();flat=torch.full(((M+1)*f['stride0']+16,),17.,dtype=old.dtype,device='cuda')
        y=torch.as_strided(flat,(M,w.n),(f['stride0'],1));live=torch.tensor([f['live']],dtype=torch.int32,device='cuda')
        expected=f['expected'].cuda();calls.append((x,sc,xs,w,y,live,f['epi'],old,flat,expected))
    flush=torch.zeros(128*1024*1024,dtype=torch.int32,device='cuda')
    tables={label:None if path=='-' else json.loads(Path(path).read_text()) for label,path in (item.split(':',1) for item in a.tables)}
    runtime=dict(architecture=architecture(),torch=importlib.metadata.version('torch'),
        triton=importlib.metadata.version('triton'),hip=torch.version.hip)
    for data in tables.values():
        if data is not None:
            assert data['kernel_identity']==D.kernel_identity(), 'table arithmetic identity differs'
            assert data['runtime']==runtime, 'table runtime differs'
    actual=D.table;active=None
    def selected(family='auto'):
        data=tables[active]
        if data is None:return {}
        return data['entries'] if family=='auto' else data.get('family_entries',{}).get(family,{})
    D.table=selected
    result=dict(protocol='layer-order-real-input-v1',projection_calls=len(calls),ordered_layers=True,
        fixture_sha256=hashlib.file_digest(a.fixtures.open('rb'),'sha256').hexdigest(),modes={},passed=False,
        runtime=runtime,table_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in tables.items()})
    graphs={}
    def reset():
        for x,sc,xs,w,y,live,epi,old,flat,expected in calls:flat.fill_(17);y.copy_(old)
    def launch():
        for x,sc,xs,w,y,live,epi,old,flat,expected in calls:W.gemm(x,sc,xs,w,y,live,epi)
    try:
        for active in tables:
            data=tables[active]
            if data is not None:assert data['kernel_identity']==D.kernel_identity()
            os.environ['STARTLUX_PRISM_AMD_GEMM']='reference' if data is None else 'auto'
            phase('stream-qualify',mode=active);reset();launch();torch.cuda.synchronize()
            for *_,y,live,epi,old,flat,expected in calls:
                assert equal(y,expected),(active,'full output')
                expected_flat=torch.full_like(flat,17);torch.as_strided(expected_flat,y.shape,y.stride()).copy_(expected)
                assert equal(flat,expected_flat),(active,'canary')
            reset();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):launch()
            graphs[active]=g;result['modes'][active]=dict(samples_ms=[],all_outputs_exact=True,canaries_exact=True)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for rep in range(7):
            for active in list(tables) if rep%2==0 else list(reversed(tables)):
                phase('stream-measure',mode=active,repeat=rep);reset();flush.add_(1)
                start.record();graphs[active].replay();end.record();end.synchronize()
                result['modes'][active]['samples_ms'].append(start.elapsed_time(end))
        for name,v in result['modes'].items():v['median_ms']=statistics.median(v['samples_ms'])
        result.update(passed=True,peak_allocated=torch.cuda.max_memory_allocated())
        (out/'stream.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS streamed projections',json.dumps(result),flush=True)
    finally:D.table=actual
