"""Single-GPU service measurements with fixed work, actual tokens, and raw request evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

repo=Path(os.environ.get('PRISM_BENCH_REPO',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(repo/'bench'),str(repo)]
import run_http as H
import suites

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18126)
p.add_argument('--precision',choices=('w4a8','w4a16'),default='w4a8')
p.add_argument('--graphs',action='store_true');p.add_argument('--think-seqs',type=int,default=8)
p.add_argument('--budget',type=int,default=8192);p.add_argument('--overlap-tokens',type=int,default=1024)
p.add_argument('--prefix-cache',type=int,default=4)
p.add_argument('--graph-max-tokens',type=int,default=2048)
p.add_argument('--sections',default='fixed,lengths,concurrency,thought,mixed,cache')
p.add_argument('--replay-requests',type=Path,help='reuse the exact measured request bodies from a completed service run')
p.add_argument('--client-levels',default='1,8,16',help='client concurrency levels; each timed round must contain at least this many requests')
a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
sections=set(a.sections.split(','));url=f'http://127.0.0.1:{a.port}'
client_levels=[int(value) for value in a.client_levels.split(',')]
assert client_levels and min(client_levels)>0
H._REQUEST_NO=itertools.count(100000)
cmd=[sys.executable,'-m','startlux_prism.server','--gguf',str(Path(a.gguf).resolve()),
     '--model-dir',str(Path(a.model_dir).resolve()),'--precision',a.precision,'--host','127.0.0.1',
     '--port',str(a.port),'--max-tokens','8192','--context-length','8192','--budget',str(a.budget),
     '--graph-max-tokens',str(a.graph_max_tokens),'--think-seqs',str(a.think_seqs),'--think-kv-rows',str(a.think_seqs*8448),
     '--max-think-budget','256','--prefix-cache',str(a.prefix_cache),'--prefix-cache-tokens','4096',
     '--overlap-tokens',str(a.overlap_tokens),'--think-mode','never','--mmproj','none','--tp','1',
     '--graphs' if a.graphs else '--no-graphs']
entry=Path(__file__).with_name('amd_service_entry.py')
if not entry.exists() and (out/'command.json').exists():
    original_script=Path(json.loads((out/'command.json').read_text())['original'][0])
    entry=original_script.with_name('amd_service_entry.py')
snapshot=out/'service-entry.py'
snapshot.write_text(entry.read_text())
entry=snapshot
cmd=[sys.executable,str(entry),*cmd[3:]]
os.environ['PRISM_SERVICE_EVENTS']=str(out/'scheduler-events.jsonl')
result=dict(command=cmd,precision=a.precision,scenes={},passed=False,
    ttft_definition='internal ThinkJob start to first consumed token; endpoint is non-streaming')
raw=(out/'requests.jsonl').open('w');lock=threading.Lock()
workload_labels={}
replay={}
if a.replay_requests:
    for line in a.replay_requests.read_text().splitlines():
        row=json.loads(line)
        if row['repeat']<0:continue
        body=json.dumps(row['request']).encode()
        assert hashlib.sha256(body).hexdigest()==row['request_sha256'],'request body serialization changed'
        replay.setdefault(row['scene'],[]).append((row['repeat']+5*row['index'],body))
        workload_labels[row['request_sha256']]=row['workload_class']
    replay={name:[body for _,body in sorted(values)] for name,values in replay.items()}
    result['replay_requests_sha256']=hashlib.sha256(a.replay_requests.read_bytes()).hexdigest()
def save():(out/'service.json').write_text(json.dumps(result,indent=2)+'\n')
def phase(name,**kw):(out/'phase.json').write_text(json.dumps(dict(phase=name,**kw)))
def get(path):
    with urllib.request.urlopen(url+path,timeout=10) as response:return json.load(response)
def cache_summary(values):
    summary=C.summ(values)
    if len(values)<200:summary.pop('p95_ms',None)
    summary['formal_p95_available']=len(values)>=200
    return summary
def request(scene,body,rep,index):
    client=H.Client(url+'/v1/systemone',timeout=600)
    try:ms,response=client.call(body)
    finally:
        if client.conn:client.conn.close()
    assert set(json.loads(body)['questions']) <= set(response['answers'])
    digest=hashlib.sha256(body).hexdigest()
    record=dict(scene=scene,repeat=rep,index=index,ms=ms,response=response,
        request_sha256=digest,request=json.loads(body),workload_class=workload_labels.get(digest,scene))
    with lock:raw.write(json.dumps(record)+'\n');raw.flush()
    return record
def scene(name,bodies,clients=1):
    if a.replay_requests:
        assert name in replay and len(replay[name])==len(bodies),(name,'replay request count differs')
        bodies=replay[name]
    assert min(len(bodies[rep::5]) for rep in range(5))>=clients,(name,'too few requests per round to exercise client concurrency')
    phase('service-warmup',scene=name,clients=clients)
    for i in range(min(3,len(bodies))):
        warm=json.loads(bodies[i]);state=warm['state']
        if isinstance(state,str):warm['state']=re.sub(r'^\[request \d{6}\]',f'[request {900000+i}]',state)
        elif isinstance(state,dict):warm['state']={**state,'request_id':str(900000+i)}
        request(name+'-warmup',json.dumps(warm).encode(),-1,i)
    phase('service-measured',scene=name,clients=clients)
    all_rows=[];rounds=[]
    for rep in range(5):
        selected=bodies[rep::5];start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            rows=list(pool.map(lambda item:request(name,item[1],rep,item[0]),enumerate(selected)))
        elapsed=time.perf_counter()-start;all_rows.extend(rows)
        rounds.append(dict(seconds=elapsed,requests=len(rows),client_concurrency_cap=clients,requests_s=len(rows)/elapsed,
            output_tokens=sum(r['response'].get('usage',{}).get('output_tokens',0) for r in rows)))
    ms=sorted(r['ms'] for r in all_rows)
    item=dict(requests=len(ms),client_concurrency_cap=clients,minimum_requests_per_round=min(r['requests'] for r in rounds),rounds=rounds,p50_ms=statistics.median(ms),
        median_requests_s=statistics.median(r['requests_s'] for r in rounds),
        total_output_tokens=sum(r['output_tokens'] for r in rounds),
        total_seconds=sum(r['seconds'] for r in rounds))
    item['actual_output_tokens_s']=item['total_output_tokens']/item['total_seconds']
    if len(ms)>=200:item['p95_ms']=ms[int(.95*(len(ms)-1))]
    item['classes']={}
    for kind in sorted({r['workload_class'] for r in all_rows}):
        values=sorted(r['ms'] for r in all_rows if r['workload_class']==kind)
        item['classes'][kind]=dict(requests=len(values),p50_ms=statistics.median(values))
        if len(values)>=200:item['classes'][kind]['p95_ms']=values[int(.95*(len(values)-1))]
    result['scenes'][name]=item;save();print('SERVICE',name,item,flush=True)

with (out/'server.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
    try:
        phase('service-start');start=time.monotonic()
        while True:
            if proc.poll() is not None:raise RuntimeError('owned server exited')
            try:result['health']=get('/health');break
            except OSError:
                if time.monotonic()-start>900:raise TimeoutError('server readiness')
                time.sleep(.5)
        result['ready_s']=time.monotonic()-start;save()
        fx=suites.fixed()
        if 'fixed' in sections:
            for name in ('yesno','readme3','startlux3','longpage2'):
                state,questions=fx[name]
                count=200 if name!='longpage2' else 30
                scene(name,[H.make_body(state,questions,True) for _ in range(count)])
        if 'lengths' in sections:
            from startlux_prism import jevfmt as J
            from startlux_prism.prompt import Tokenizer,rows_for
            tok=Tokenizer(a.model_dir);question={'availability':{'type':'noul','instructions':'Is this an availability incident?'}}
            for target in (256,2048,8192):
                def size(n):
                    state=H.uniquify('The service is unavailable.'+' service'*n,123456)
                    return len(rows_for(tok,[J.from_systemone(state,question['availability'])],['availability'])[0].ids)
                lo,hi=0,target
                while lo<hi:
                    mid=(lo+hi+1)//2
                    if size(mid)<=target:lo=mid
                    else:hi=mid-1
                assert size(lo)==target
                scene(f'length-{target}',[H.make_body('The service is unavailable.'+' service'*lo,question,True) for _ in range(30)])
        if 'concurrency' in sections:
            mix=[fx[k] for k in ('yesno','readme3','startlux3')]
            for clients in client_levels:
                bodies=[H.make_body(*mix[i%len(mix)],True) for i in range(240)]
                scene(f'concurrency-{clients}',bodies,clients)
        if 'thought' in sections:
            state,questions=fx['yesno']
            for clients in client_levels:
                bodies=[]
                for i in range(max(40,5*clients)):
                    body=json.loads(H.make_body(state,questions,True))
                    body.update(think_mode='always',think_budget=64,return_details=True,return_thought=True)
                    bodies.append(json.dumps(body).encode())
                scene(f'thought-{clients}',bodies,clients)
        if 'mixed' in sections:
            bodies=[]
            for i in range(240):
                kind='long' if i%24==0 else 'thought' if i%8==0 else 'short'
                state,questions=fx['longpage2' if kind=='long' else 'yesno']
                body=json.loads(H.make_body(state,questions,True))
                if kind=='thought':body.update(think_mode='always',think_budget=64,return_details=True)
                encoded=json.dumps(body).encode();bodies.append(encoded)
                workload_labels[hashlib.sha256(encoded).hexdigest()]=kind
            scene('mixed-short-long-thought',bodies,8)
        if 'cache' in sections:
            import run_prefix as C
            if a.replay_requests:
                # Keep cache-session IDs after the frozen measured requests,
                # even when an earlier section was generated with more bodies.
                ids=[]
                for bodies in replay.values():
                    for body in bodies:
                        state=json.loads(body)['state']
                        if isinstance(state,dict) and 'request_id' in state:ids.append(int(state['request_id']))
                        elif isinstance(state,str):
                            match=re.match(r'^\[request (\d{6})\]',state)
                            if match:ids.append(int(match.group(1)))
                assert ids
                H._REQUEST_NO=itertools.count(max(ids)+1)
            result['cache']={}
            for name,templates in C.scenarios(None,0).items():
                before=get('/stats');phase('cache',scenario=name)
                first,follow,tokens=C.run(url+'/v1/systemone',templates,20,2,0)
                result['cache'][name]=dict(first=cache_summary(first),follow=cache_summary(follow),
                    first_raw=first,follow_raw=follow,input_tokens=tokens,before=before,after=get('/stats'));save()
        result['final_stats']=get('/stats');result['passed']=True;save()
    finally:
        proc.terminate()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raw.close();result['server_stopped']=True;save()
print('PASS service matrix',flush=True)
