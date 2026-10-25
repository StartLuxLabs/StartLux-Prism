"""Fixed-workload GPU decode, reset state for each measurement; EOS ignored."""
import argparse,hashlib,json,statistics,sys,time,os
from pathlib import Path
import numpy as np
import torch

repo=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
p.add_argument('--precision',choices=('w4a8','w4a16'),default='w4a8')
p.add_argument('--reference-decode',help='Optional baseline decode.json; require identical token hashes')
p.add_argument('--paired',action='store_true',help='Alternate legacy and qualified kernels on one resident model/head')
args=p.parse_args()
if args.paired and args.precision!='w4a8':p.error('--paired currently compares W4A8 kernels only')
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
assert torch.version.hip and torch.cuda.device_count()==1,'Select exactly one ROCm GPU before starting'

sys.path[:0]=[str(repo)]
from startlux_prism import progress as P
from startlux_prism.engine.model import Engine,Options
from startlux_prism.engine.decode import Decoder,Row,Sampling

P.on();torch.set_num_threads(8)
model=Path(args.model_dir);gguf=args.gguf
if args.paired:
 from startlux_prism.kernels.amd_w4a8_dispatch import table
 assert table(),'No qualified table matches this runtime/source'
 os.environ['STARTLUX_PRISM_AMD_GEMM']='auto'
e=Engine(gguf,str(model),Options(precision=args.precision,max_tokens=4096,graphs=False,prefix_cache=0,
 think_seqs=8,think_kv_rows=8192))
dec=Decoder(e,gguf,max_seqs=8,sampling=Sampling(greedy=True),graphs=True);dec.warm_up()
rng=np.random.default_rng(0);prompts=[rng.integers(1000,100000,512).tolist() for _ in range(8)]
result=dict(precision=args.precision,context=512,warm_steps=8,timed_steps=64,rows_each=1024,greedy=True,ignore_eos=True,modes={})
reference={}
def phase(name,**kw):(out/'phase.json').write_text(json.dumps(dict(phase=name,**kw)))
def run(B,graphs,indices=None,check_repeat=True):
 indices=list(range(B)) if indices is None else indices
 dec.graphs=graphs
 keep=[(prompts[indices[i]][:-1],e.think_slot0+i,e.think_kv0+i*1024) for i in range(B)]
 b=e.launch(e.prepare([],0,keep=keep));e.finish(b)
 def rows(k):
  return [Row(e.think_slot0+i,e.think_kv0+i*1024,511+k,511+k,1234+indices[i],k,i if k else -1-prompts[indices[i]][-1]) for i in range(B)]
 tokens=[]
 for k in range(8):tokens.append(dec.finish(dec.launch(rows(k),k%2)))
 torch.cuda.synchronize();t0=time.perf_counter();prev=None
 for k in range(8,72):
  step=dec.launch(rows(k),k%2)
  if prev is not None:tokens.append(dec.finish(prev))
  prev=step
 tokens.append(dec.finish(prev));dt=time.perf_counter()-t0
 a=np.stack(tokens);h=hashlib.sha256(a.tobytes()).hexdigest()
 if check_repeat:
  if B in reference:assert np.array_equal(a,reference[B]),'decode changed across repeats/graphs'
  else:reference[B]=a.copy()
 else:return a
 return dict(seconds=dt,ms_per_step=dt*1000/64,tokens_per_s=B*64/dt,token_sha256=h)

if args.paired:
 # Preserve independent captured kernels for each implementation while sharing
 # the same model, decoder/head, inputs and state buffers on the selected GPU.
 active_mode='auto';graph_sets={'auto':(dec._graphs,dec._pool),'gemv':({},None)}
 def activate(mode):
  global active_mode
  if mode!=active_mode:
   graph_sets[active_mode]=(dec._graphs,dec._pool)
   dec._graphs,dec._pool=graph_sets[mode]
   active_mode=mode
  os.environ['STARTLUX_PRISM_AMD_GEMM']=mode
 result['legacy_modes']={};result['paired_same_resident_model']=True
 for graphs in [False,True]:
  result['modes'][str(graphs)]={};result['legacy_modes'][str(graphs)]={}
  for B in [1,4,8]:
   for mode in ['gemv','auto']:
    activate(mode);phase('decode-warmup',graphs=graphs,batch=B,implementation=mode);run(B,graphs)
   samples={'gemv':[],'auto':[]}
   for rep in range(5):
    for mode in (['gemv','auto'] if rep%2 else ['auto','gemv']):
     activate(mode);phase('decode-paired',graphs=graphs,batch=B,repeat=rep,implementation=mode)
     samples[mode].append(run(B,graphs))
   result['modes'][str(graphs)][str(B)]=samples['auto']
   result['legacy_modes'][str(graphs)][str(B)]=samples['gemv']
   (out/'decode.json').write_text(json.dumps(result,indent=2)+'\n')
   print('PERF paired decode graphs',graphs,'batch',B,{mode:round(statistics.median(x['tokens_per_s'] for x in rows),3) for mode,rows in samples.items()},flush=True)
 activate('auto')
else:
 for graphs in [False,True]:
  result['modes'][str(graphs)]={}
  for B in [1,4,8]:
   phase('decode-warmup',graphs=graphs,batch=B);run(B,graphs)
   samples=[]
   for rep in range(5):
    phase('decode',graphs=graphs,batch=B,repeat=rep);samples.append(run(B,graphs))
   result['modes'][str(graphs)][str(B)]=samples
   (out/'decode.json').write_text(json.dumps(result,indent=2)+'\n')
   print('PERF decode graphs',graphs,'batch',B,'tok/s',round(statistics.median(x['tokens_per_s'] for x in samples),2),flush=True)
for i in range(8):
 alone=run(1,False,[i],False)
 assert np.array_equal(alone[:,0],reference[8][:,i]),('decode batch isolation',i)
np.savez(out/'all-decode-tokens.npz',**{str(k):v for k,v in reference.items()})
if args.reference_decode:
 base_ref=json.loads(Path(args.reference_decode).read_text())
 for mode,batches in result['modes'].items():
  for batch,samples in batches.items():
   assert {r['token_sha256'] for r in samples}=={r['token_sha256'] for r in base_ref['modes'][mode][batch]},'optimized tokens differ from B0'
if args.paired:
 # Verify that the two graph caches really contain different projection kernels.
 # This diagnostic is after all throughput samples and is not included in them.
 from torch.profiler import profile,ProfilerActivity
 result['graph_implementation_checks']={}
 for mode in ['gemv','auto']:
  activate(mode);dec.graphs=True
  keep=[(prompts[0][:-1],e.think_slot0,e.think_kv0)]
  e.finish(e.launch(e.prepare([],0,keep=keep)))
  rows=[Row(e.think_slot0,e.think_kv0,511,511,1234,0,-1-prompts[0][-1])]
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
   dec.finish(dec.launch(rows,0));torch.cuda.synchronize()
  path=out/('graph-'+mode+'-trace.json');trace.export_chrome_trace(str(path))
  events=json.loads(path.read_text())['traceEvents']
  kernels=[x['name'] for x in events if x.get('cat')=='kernel']
  has_dot=any('_gemv_dot' in x for x in kernels)
  assert has_dot==(mode=='auto'),('graph reused wrong implementation',mode,has_dot)
  result['graph_implementation_checks'][mode]=dict(kernels=len(kernels),has_dot4=has_dot)
 activate('auto')
result.update(exact_tokens_across_repeats_graphs_and_batch=True,memory=dict(allocated=torch.cuda.memory_allocated(),
 reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated()))
(out/'decode.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS decode performance and token equality',flush=True)
