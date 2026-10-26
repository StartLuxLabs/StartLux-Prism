"""Same-resident, paired readout/prefill table ablation with exact full letter logits."""
import argparse
import collections
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
from startlux_prism import jevfmt as J,progress as P
from startlux_prism.engine.model import Engine,Options
from startlux_prism.engine.decode import Decoder
from startlux_prism.prompt import rows_for
from startlux_prism.kernels import w4a8 as W
from startlux_prism.backends import architecture
import suites

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--precision',choices=('w4a8','w4a16'),default='w4a8')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--table',action='append',default=[],help='label:path for same-identity table')
p.add_argument('--modes',default='baseline,auto')
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
assert torch.version.hip and torch.cuda.device_count()==1
torch.set_num_threads(8);P.on()
if a.precision=='w4a16':from startlux_prism.kernels import amd_w4a16_dispatch as D
else:from startlux_prism.kernels import amd_w4a8_dispatch as D
modes=a.modes.split(',');tables={label:json.loads(Path(path).read_text()) for label,path in (s.split(':',1) for s in a.table)}
runtime=dict(architecture=architecture(),torch=importlib.metadata.version('torch'),
 triton=importlib.metadata.version('triton'),hip=torch.version.hip)
for data in tables.values():
 assert data['kernel_identity']==D.kernel_identity(), 'table arithmetic identity differs'
 assert data['runtime']==runtime, 'table runtime differs'
active='auto';original_table=D.table
if a.precision=='w4a16':
 def table():return tables[active]['entries'] if active in tables else original_table()
else:
 def table(family='auto'):
  if active not in tables:return original_table(family)
  data=tables[active]
  return data['entries'] if family=='auto' else data.get('family_entries',{}).get(family,{})
D.table=table
eng=Engine(a.gguf,a.model_dir,Options(precision=a.precision,max_tokens=8192,graphs=False,prefix_cache=0,
    think_seqs=8,think_kv_rows=8*8448))
decoder=Decoder(eng,a.gguf,max_seqs=8,graphs=False,max_len=8448)  # retain the complete generation head

def rows(state,questions):
 return rows_for(eng.tok,[J.from_systemone(state,q,allow_none=q['type']=='choice') for q in questions.values()],list(questions))
def equal(x,y):
 return len(x)==len(y) and all(torch.equal(u.contiguous().view(torch.uint8),v.contiguous().view(torch.uint8)) for u,v in zip(x,y))
def activate(mode):
 global active
 active=mode
 dispatch='auto' if mode in tables or mode=='auto' else 'reference'
 os.environ['STARTLUX_PRISM_AMD_GEMM']=dispatch;os.environ['STARTLUX_PRISM_AMD_W4A16']=dispatch

def phase(name,**kw):(out/'phase.json').write_text(json.dumps(dict(phase=name,**kw)))
scenes={name:rows(*suites.fixed()[name]) for name in ('yesno','readme3','startlux3','longpage2','review1')}
q={'availability':{'type':'noul','instructions':'Is this an availability incident?'}}
for target in (256,2048,8192):
 lo,hi=0,target
 while lo<hi:
  mid=(lo+hi+1)//2
  if len(rows('The service is unavailable.'+' service'*mid,q)[0].ids)<=target:lo=mid
  else:hi=mid-1
 rr=rows('The service is unavailable.'+' service'*lo,q);assert len(rr[0].ids)==target
 scenes[f'length-{target}']=rr
result=dict(protocol='readout-paired-five-v1',precision=a.precision,cache='off',graphs=False,scenes={},passed=False,
 runtime=runtime,full_vocabulary_size=decoder.vocab,
 table_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in tables.items()})
for name,rr in scenes.items():
 ref=None;samples={mode:[] for mode in modes}
 for mode in modes:
  activate(mode);phase('readout-warmup',scene=name,mode=mode);value=eng.letter_logits(rr)
  assert all(torch.isfinite(v).all() for v in value),(name,mode,'non-finite readout logits')
  if ref is None:ref=value
  assert equal(value,ref),(name,mode,'warmup')
 for rep in range(5):
  for mode in modes if rep%2==0 else list(reversed(modes)):
   activate(mode);phase('readout-measure',scene=name,mode=mode,repeat=rep)
   start=time.perf_counter();value=eng.letter_logits(rr);elapsed=(time.perf_counter()-start)*1000
   assert equal(value,ref),(name,mode,rep);samples[mode].append(elapsed)
 result['scenes'][name]=dict(samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items()},
  full_logits_exact=True,logits=[v.tolist() for v in ref],tokens=eng.last['tokens'],bucket=eng.last['bucket'])
 (out/'readout.json').write_text(json.dumps(result,indent=2)+'\n')
 print('READOUT',name,result['scenes'][name]['median_ms'],flush=True)
# Report a new trace only after all timed trials, for representative real scenes.
from torch.profiler import profile,ProfilerActivity
result['profiles']={}
for mode in modes:
 activate(mode)
 for name in ('readme3','length-2048','length-8192'):
  phase('readout-profile',scene=name,mode=mode);eng.letter_logits(scenes[name]);torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
   eng.letter_logits(scenes[name]);torch.cuda.synchronize()
  path=out/f'trace-{mode}-{name}.json';prof.export_chrome_trace(str(path))
  events=json.loads(path.read_text())['traceEvents'];times=collections.Counter();counts=collections.Counter()
  for event in events:
   if event.get('cat')=='kernel' and event.get('ph')=='X':times[event['name']]+=event['dur']/1000;counts[event['name']]+=1
  result['profiles'][mode+'-'+name]=[dict(name=k,ms=v,calls=counts[k]) for k,v in times.most_common()]
result.update(passed=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
(out/'readout.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS paired real readout',flush=True)
