"""Same resident compressed model; switch only qualified projection choices outside capture."""
import argparse,collections,hashlib,json,os,statistics,sys,time
from pathlib import Path
import torch
repo=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
assert torch.version.hip and torch.cuda.device_count()==1,'Select exactly one ROCm GPU before starting'

sys.path.insert(0,str(repo/'bench'))
import suites
from startlux_prism import jevfmt as J,progress as P
from startlux_prism.engine.model import Engine,Options,group
from startlux_prism.prompt import rows_for
from startlux_prism.kernels import w4a8 as W,amd_w4a8_dispatch as D
P.on();torch.set_num_threads(8)
model=Path(args.model_dir)
os.environ['STARTLUX_PRISM_AMD_GEMM']='gemv'
e=Engine(args.gguf,str(model),
 Options(precision='w4a8',max_tokens=8192,graphs=False,prefix_cache=4,prefix_cache_tokens=4096))
cache=e.cache;e.cache=None
modes=['gemv','dispatch','decode','prefill','auto']
result={'qualified_entries':len(D.table()),'scenes':{},'hits':{}}
assert D.table(),'runtime identity did not load any qualified choices'
scenes={k:suites.fixed()[k] for k in ['yesno','readme3','startlux3','review1','longpage2']}
q={'availability':{'type':'noul','instructions':'Is this an availability incident?'}}
def rows(st,qs):return rows_for(e.tok,[J.from_systemone(st,x,allow_none=x['type']=='choice') for x in qs.values()],list(qs))
for target in [256,2048,8192]:
 lo,hi=0,target
 while lo<hi:
  mid=(lo+hi+1)//2
  if len(rows('The service is unavailable.'+' service'*mid,q)[0].ids)<=target:lo=mid
  else:hi=mid-1
 st='The service is unavailable.'+' service'*lo
 assert len(rows(st,q)[0].ids)==target
 scenes['length-'+str(target)]=(st,q)
def equal(a,b,label):
 assert len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b)),label
checks={}
for name,(st,qs) in scenes.items():
 rr=rows(st,qs);os.environ['STARTLUX_PRISM_AMD_GEMM']='gemv';ref=e.letter_logits(rr)
 checks[name]=(rr,ref)
 samples={m:[] for m in modes}
 for mode in modes:
  os.environ['STARTLUX_PRISM_AMD_GEMM']=mode;equal(e.letter_logits(rr),ref,(name,mode,'warm'))
 for rep in range(3):
  for mode in modes if rep%2 else list(reversed(modes)):
   os.environ['STARTLUX_PRISM_AMD_GEMM']=mode
   t=time.perf_counter();got=e.letter_logits(rr);dt=(time.perf_counter()-t)*1000
   equal(got,ref,(name,mode));samples[mode].append(dt)
 result['scenes'][name]=dict(samples_ms=samples,median_ms={m:statistics.median(v) for m,v in samples.items()},
   logits=[x.tolist() for x in ref],tokens=e.last['tokens'],bucket=e.last['bucket'])
 (out/'ablation.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PASS model exact A/B',name,result['scenes'][name]['median_ms'],flush=True)
os.environ['STARTLUX_PRISM_AMD_GEMM']='auto'
# Each question alone, reversed, and among unrelated questions; both body and read buckets change.
rr,ref=checks['readme3']
for i,row in enumerate(rr):equal(e.letter_logits([row]),[ref[i]],('alone',i))
equal(e.letter_logits(list(reversed(rr))),list(reversed(ref)),'question reorder')
other,oref=checks['yesno'];equal(e.run([group(other),group(rr)]),oref+ref,'mixed request isolation')
# Cache off/miss/hit/evict, against old recomputation; long evidence guarantees admission.
st='Cache performance evidence. '+'The service is down, staff cannot log in, and payments are failing. '*78
ra=rows(st,{'a':{'type':'noul','instructions':'Should support escalate the incident?'}})
rb=rows(st,{'b':{'type':'noul','instructions':'Is this a service availability incident?'}})
os.environ['STARTLUX_PRISM_AMD_GEMM']='gemv';baseline=e.letter_logits(rb)
os.environ['STARTLUX_PRISM_AMD_GEMM']='auto';equal(e.letter_logits(rb),baseline,'cache off')
e.cache=cache;cache.clear();equal(e.letter_logits(rb),baseline,'cache miss');cache.clear();e.letter_logits(ra)
hits=e.cache_stats()['hits']
equal(e.letter_logits(rb),baseline,'cache hit');assert e.cache_stats()['hits']>hits
for i in range(12):e.letter_logits(rows('Ticket '+str(i)+'. '+st,q))
equal(e.letter_logits(rb),baseline,'cache eviction')
e.cache=None;e.opt.graphs=True
for name in ['yesno','readme3','startlux3','length-256','length-2048']:
 rr,ref=checks[name];equal(e.letter_logits(rr),ref,('graph capture',name));equal(e.letter_logits(rr),ref,('graph replay',name))
e.opt.graphs=False
# Empty eager precompilation must preserve the live tables and recurrent/KV state.
state_before={name:getattr(e,name).cpu().clone() for name in ['state','conv_state','kv','counts']}
e.warm_up_eager()
for name,old in state_before.items():assert torch.equal(getattr(e,name).cpu(),old),('eager warmup changed state',name)
del state_before
equal(e.letter_logits(checks['readme3'][0]),checks['readme3'][1],'readout after eager warmup')
result['empty_eager_warmup_preserves_state']=True
# Static metadata census after timing: no device reads in the production dispatcher.
counter=collections.Counter();orig=W.gemm
def wrapped(x,a,xs,w,y,n,epi=0,**kw):
 choice=D.pick(x,w,y,epi);counter[(w.kind,x.shape[0],w.n,w.k,epi,y.stride(0),choice['implementation'] if choice else 'legacy')]+=1
 return orig(x,a,xs,w,y,n,epi=epi,**kw)
W.gemm=wrapped
for name in ['readme3','startlux3','length-8192']:e.letter_logits(checks[name][0])
W.gemm=orig
result['hits']=[dict(shape=list(k),calls=v) for k,v in counter.items()]
result['passed']=True;result['memory']={'peak_allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.memory_reserved()}
(out/'ablation.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS all exact model, batch, cache and graph checks',flush=True)
