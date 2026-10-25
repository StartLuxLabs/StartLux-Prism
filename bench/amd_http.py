"""One GPU server, upstream fixed suites, unique evidence and recorded HTTP samples."""
import argparse,itertools,json,statistics,subprocess,sys,threading,time,urllib.request,os
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True)
p.add_argument('--port',type=int,default=18092)
p.add_argument('--precision',choices=['w4a8','w4a16'],default='w4a8')
a=p.parse_args();a.gguf=str(Path(a.gguf).resolve());a.model_dir=str(Path(a.model_dir).resolve());out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(repo/'bench'),str(repo)]
import run_http as H
import suites

H._REQUEST_NO=itertools.count(100000)
URL=f'http://127.0.0.1:{a.port}/v1/systemone'
phase_info={};lock=threading.Lock();raw=(out/'http_samples.jsonl').open('w')
def phase(name,**kw):
 global phase_info
 phase_info=dict(phase=name,**kw);(out/'phase.json').write_text(json.dumps(phase_info))

OriginalClient=H.Client
class Client(OriginalClient):
 def call(self,body):
  ms,response=super().call(body)
  req=json.loads(body)
  assert set(req['questions'])<=set(response['answers'])
  sample=dict(t=time.time(),phase=phase_info,ms=ms,questions=len(req['questions']),request=req,response=response)
  with lock:raw.write(json.dumps(sample)+'\n');raw.flush()
  return ms,response
H.Client=Client
import run_prefix as P

model=Path(a.model_dir)
cmd=[sys.executable,'-m','startlux_prism.server','--gguf',a.gguf,
 '--model-dir',str(model),'--precision',a.precision,'--host','127.0.0.1','--port',str(a.port),'--max-tokens','8192',
 '--budget','8192','--graph-max-tokens','2048','--no-graphs','--prefix-cache','4','--prefix-cache-tokens','4096',
 '--think-mode','never','--max-think-budget','256','--think-seqs','1','--think-kv-rows','12288',
 '--context-length','8192','--mmproj','none','--tp','1']
result=dict(command=cmd,latency={},lengths={},sweep=[],prefix={},autothink={})
def save():(out/'http.json').write_text(json.dumps(result,indent=2)+'\n')
def get(path):
 with urllib.request.urlopen(URL.rsplit('/v1/',1)[0]+path,timeout=5) as r:return json.load(r)

log=(out/'server.log').open('w');server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=repo)
try:
 deadline=time.monotonic()+900
 while True:
  if server.poll() is not None:raise RuntimeError('server exited; see server.log')
  try:result['health']=get('/health');break
  except OSError:
   if time.monotonic()>deadline:raise TimeoutError('server readiness')
   time.sleep(.5)
 print('PERF server ready',result['health'],flush=True);save()
 fx=suites.fixed()
 for name in ['yesno','readme3','startlux3','review1','wide33','longpage2']:
  st,qs=fx[name];cl=Client(URL);phase('http-latency-warmup',suite=name)
  for _ in range(5):cl.call(H.make_body(st,qs,True))
  rounds=[]
  for rep in range(3):
   phase('http-latency',suite=name,repeat=rep)
   rounds.append(H.latency(URL,name,st,qs,10 if name=='longpage2' else 34,0,None,True))
  result['latency'][name]=rounds;save()
  print('PERF HTTP latency',name,[x['p50_ms'] for x in rounds],flush=True)
 from startlux_prism import jevfmt as J
 from startlux_prism.prompt import Tokenizer,rows_for
 tok=Tokenizer(str(model));length_q={'availability':{'type':'noul','instructions':'Is this an availability incident?'}}
 for target in [256,2048,8192]:
  # Fixed-width request IDs preserve shape while preventing evidence-cache replay.
  def size(n):
   state=H.uniquify('The service is unavailable.'+' service'*n,123456)
   return len(rows_for(tok,[J.from_systemone(state,length_q['availability'])],['availability'])[0].ids)
  lo,hi=0,target
  while lo<hi:
   mid=(lo+hi+1)//2
   if size(mid)<=target:lo=mid
   else:hi=mid-1
  state='The service is unavailable.'+' service'*lo
  assert size(lo)==target,(target,size(lo))
  phase('http-length-warmup',prompt_tokens=target)
  Client(URL).call(H.make_body(state,length_q,True))
  rounds=[]
  for rep in range(3):
   phase('http-length',prompt_tokens=target,repeat=rep)
   rounds.append(H.latency(URL,str(target),state,length_q,10 if target==8192 else 34,0,None,True))
  result['lengths'][str(target)]=rounds;save()
  print('PERF HTTP length',target,[x['p50_ms'] for x in rounds],flush=True)
 mix=[fx[k] for k in ['yesno','readme3','startlux3']]
 for c in [1,2,4,8,16,32,64]:
  phase('http-concurrency-warmup',concurrency=c);H.sweep(URL,mix,[c],5,None,True)
  for rep in range(3):
   phase('http-concurrency',concurrency=c,repeat=rep)
   row=H.sweep(URL,mix,[c],30,None,True)[0];row['repeat']=rep
   result['sweep'].append(row);save()
 for name,templates in P.scenarios(None,0).items():
  phase('http-prefix',scenario=name)
  before=get('/stats');first,follow,tokens=P.run(URL,templates,20,2,0)
  result['prefix'][name]=dict(first=P.summ(first),follow=P.summ(follow),first_raw=first,follow_raw=follow,
   input_tokens=tokens,before=before,after=get('/stats'));save()
  print('PERF HTTP prefix',name,result['prefix'][name]['first'],result['prefix'][name]['follow'],flush=True)
 st,qs=fx['yesno'];cl=Client(URL)
 H._REQUEST_NO=itertools.count(700000)
 phase('http-autothink-warmup')
 body=json.loads(H.make_body(st,qs,True));body.update(think_mode='always',think_budget=16,return_details=True,return_thought=True)
 cl.call(json.dumps(body).encode())
 for budget in [16,64,256]:
  samples=[]
  for rep in range(10):
   phase('http-autothink',budget=budget,repeat=rep)
   body=json.loads(H.make_body(st,qs,True));body.update(think_mode='always',think_budget=budget,return_details=True,return_thought=True)
   ms,answer=cl.call(json.dumps(body).encode());samples.append(dict(ms=ms,response=answer))
  result['autothink'][str(budget)]=samples;save()
  print('PERF AutoThink',budget,[round(x['ms'],2) for x in samples],flush=True)
 result['final_stats']=get('/stats');result['passed']=True;save()
 print('PASS HTTP performance benchmarks',flush=True)
finally:
 server.terminate()
 try:server.wait(timeout=30)
 except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
 log.close();raw.close()
