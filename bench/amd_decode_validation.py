"""Single GPU real-model decode, batch isolation, and teacher-forced prefill checks."""
import argparse
import json
from pathlib import Path
import sys
import time
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
sys.path.insert(0,str(repo/'tests'))
sys.path.insert(0,str(repo/'bench'))
import torch
from startlux_prism import jevfmt as J,progress as P
from startlux_prism.engine.model import Engine,Options
from startlux_prism.engine.decode import Decoder,Sampling
from startlux_prism.think import Framing,RELEASE_PREFIX
from test_decode import Seqs,prefill_logits
import suites

p=argparse.ArgumentParser()
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--output',required=True);p.add_argument('--tokens',type=int,default=8)
p.add_argument('--graphs',action='store_true')
p.add_argument('--precision',choices=('w4a8','w4a16'),default='w4a8')
a=p.parse_args();P.on()
eng=Engine(a.gguf,a.model_dir,Options(precision=a.precision,max_tokens=2048,graphs=a.graphs,
    prefix_cache=0,think_seqs=8,think_kv_rows=32768))
dec=Decoder(eng,a.gguf,max_seqs=8,graphs=a.graphs,sampling=Sampling(greedy=True))
framing=Framing(a.model_dir)
ps=[]
for name in ['yesno','review1','readme3','startlux3']:
    state,qs=suites.fixed()[name];q=next(iter(qs.values()))
    rec=J.from_systemone(state,q,allow_none=q['type']=='choice')
    ids=eng.tok.encode(framing.prompt(rec)+RELEASE_PREFIX)
    ps.append((name,ids))
result=dict(precision=a.precision,shared_gpu=bool(int(__import__("os").environ.get("PRISM_SHARED_GPU","0"))),sequences=[],graphs=a.graphs)
start=time.perf_counter()
s=Seqs(eng,dec,ps,4096)
s.prefill(s.s);s.run(s.s,a.tokens)
print('greedy decode',a.tokens,'tokens per sequence',time.perf_counter()-start,flush=True)
for row in s.s:
    ids=row['ids']+row['gen'][:-1]
    positions=list(range(len(row['ids'])-1,len(row['ids'])-1+a.tokens))
    ref,letters=prefill_logits(eng,dec,ids,positions)
    got=torch.stack(row['logits'])
    delta=(got-ref).abs()
    agree=int((got.argmax(1).cpu()==ref.argmax(1).cpu()).sum())
    print('teacher forcing',row['name'],'max',delta.max().item(),'mean',delta.mean().item(),'agree',agree,a.tokens,flush=True)
    assert torch.isfinite(got).all()
    assert agree>=a.tokens-1,'teacher-forced argmax mismatch'
    one=Seqs(eng,dec,[(row['name'],row['ids'])],4096,seeds=[row['seed']])
    one.prefill(one.s);one.run(one.s,a.tokens)
    assert one.s[0]['gen']==row['gen'],'batch isolation token mismatch'
    assert all(torch.equal(u,v) for u,v in zip(one.s[0]['logits'],row['logits'])),'batch isolation logit mismatch'
    result['sequences'].append(dict(name=row['name'],generated=row['gen'],text=eng.tok.tok.decode(row['gen']),
        max_logit_delta=float(delta.max()),mean_logit_delta=float(delta.mean()),top1_agree=agree))
rel=Sampling()
combined=Seqs(eng,dec,ps,4096,seeds=[123,456,789,1011],sampling=rel)
combined.prefill(combined.s);combined.run(combined.s,a.tokens)
for i,row in enumerate(combined.s):
    alone=Seqs(eng,dec,[(row['name'],row['ids'])],4096,seeds=[row['seed']],sampling=rel)
    alone.prefill(alone.s);alone.run(alone.s,a.tokens)
    assert alone.s[0]['gen']==row['gen'],'sampled tokens depend on batch'
reordered=Seqs(eng,dec,list(reversed(ps)),4096,seeds=list(reversed([123,456,789,1011])),sampling=rel)
reordered.prefill(reordered.s);reordered.run(reordered.s,a.tokens)
assert [r['gen'] for r in reversed(reordered.s)]==[r['gen'] for r in combined.s],'sampled tokens depend on batch order'
result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
    free_total=torch.cuda.mem_get_info())
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print('PASS decode teacher forcing, exact batch isolation and seeded sampling',json.dumps(result),flush=True)
