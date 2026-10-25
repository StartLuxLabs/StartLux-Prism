"""Start one owned GPU server, qualify HTTP behavior, optionally run sustained load, then stop it."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
repo=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(repo/'bench'),str(repo)]
import suites
import vision_suite

p=argparse.ArgumentParser();p.add_argument('--gguf',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=18092)
p.add_argument('--precision',choices=['w4a8','w4a16'],default='w4a8')
p.add_argument('--seconds',type=int,default=0);p.add_argument('--requests',type=int,default=0)
p.add_argument('--graphs',action='store_true');p.add_argument('--readout-only',action='store_true')
p.add_argument('--mmproj',default='none');p.add_argument('--max-tokens',type=int,default=2048)
a=p.parse_args()
a.gguf=str(Path(a.gguf).resolve());a.model_dir=str(Path(a.model_dir).resolve())
if a.mmproj!='none':a.mmproj=str(Path(a.mmproj).resolve())
out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
url=f'http://127.0.0.1:{a.port}'
def http(body=None,path='/v1/systemone'):
    request=urllib.request.Request(url+path,data=None if body is None else json.dumps(body).encode(),
                                   headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=600) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
cmd=[sys.executable,'-m','startlux_prism.server','--gguf',a.gguf,'--model-dir',a.model_dir,'--port',str(a.port),
     '--host','127.0.0.1','--precision',a.precision,'--max-tokens',str(a.max_tokens),'--prefix-cache','4','--prefix-cache-tokens','4096',
     '--think-seqs','1','--max-think-budget','256','--think-kv-rows','4096','--context-length',str(a.max_tokens),
     '--mmproj',a.mmproj,'--graph-max-tokens','512']
cmd+=['--graphs'] if a.graphs else ['--no-graphs']
assert not a.readout_only,'latest upstream qualification retains the full generation head'
# The upstream endpoint intentionally has no allocator telemetry. Instrument only
# GET /health in this owned diagnostic subprocess; request handling is unchanged.
bootstrap=out.with_suffix('.server-entry.py')
bootstrap.write_text('import sys;sys.path.insert(0,'+repr(str(repo))+')\n'+'''import json,runpy,torch
import startlux_prism.httpserve as H
original=H.serve
def serve(host,port,handle):
    def measured(method,path,body):
        code,payload=handle(method,path,body)
        if method=='GET' and path=='/health':
            data=json.loads(payload)
            data['diagnostic_allocator_telemetry']=True
            data['memory']={'allocated':torch.cuda.memory_allocated(),
                'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),
                'free_total':torch.cuda.mem_get_info()}
            payload=json.dumps(data).encode()
        return code,payload
    return original(host,port,measured)
H.serve=serve
runpy.run_module('startlux_prism.server',run_name='__main__')
''')
cmd=[sys.executable,str(bootstrap),*cmd[3:]]
report=dict(shared_gpu=bool(int(__import__("os").environ.get("PRISM_SHARED_GPU","0"))),command=cmd,checks=[],requests=0,errors=0,qualification_pass=False)
start=time.monotonic()
with out.with_suffix('.server.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,text=True)
    try:
        health=None
        while time.monotonic()-start<1800:
            if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}; see server log')
            try:
                status,health=http(path='/health')
                if status==200:break
            except (OSError,TimeoutError):pass
            time.sleep(1)
        assert health is not None,'server readiness timeout'
        report['ready_s']=time.monotonic()-start
        report['initial_health']=health
        print('Ready',report['ready_s'],json.dumps(health),flush=True)
        def check(label,body=None,expected=200,path='/v1/systemone'):
            t=time.monotonic();code,data=http(body,path)
            report['checks'].append(dict(name=label,status=code,elapsed_s=time.monotonic()-t,response=data))
            assert code==expected,(label,code,data)
            print('PASS HTTP',label,code,flush=True)
            return data
        st,qs=suites.fixed()['yesno']
        base=dict(state=st,questions=qs,think_mode='never')
        alone=check('never',base)
        check('models',path='/v1/models');check('stats',path='/stats')
        check('bad gate',dict(base,gate=2),422)
        check('bad budget',dict(base,think_budget=0),422)
        check('oversized prompt',dict(base,state='unavailable '*10000),422)
        check('missing questions',{'state':'test'},422)
        for name in ['readme3','startlux3','wide33','review1']:
            s,q=suites.fixed()[name]
            data=check(name,dict(state=s,questions=q,think_mode='never'))
            assert all(ans['type'] in ['choice','noul','score'] for ans in data['answers'].values())
        if a.readout_only:
            check('thinking refused in readout-only',dict(base,think_mode='always'),422)
        else:
            for mode,gate in [('auto',0),('auto',1),('always',.6)]:
                data=check(f'{mode} gate={gate}',dict(base,think_mode=mode,gate=gate,think_budget=16,
                    return_details=True,return_thought=True))
                if gate==1 or mode=='always':assert data['usage']['output_tokens']>0,data
            data=check('default budget clamped to 256',dict(base,think_mode='always',
                return_details=True,return_thought=True))
            assert 0<data['usage']['output_tokens']<=256,data
            data=check('requested 4096 budget clamped to server 256',dict(base,think_mode='always',
                think_budget=4096,return_details=True,return_thought=True))
            assert 0<data['usage']['output_tokens']<=256,data
        image_bodies=[]
        if a.mmproj!='none':
            for count in [1,2,8]:
                body=dict(state='Inspect the colour swatches.',questions={'red':{'type':'noul',
                    'instructions':'Is the first colour swatch red?'}},think_mode='never',
                    images=[vision_suite.tile(c) for c in ['red','blue','green','black','yellow','orange','blue','red'][:count]])
                check(f'{count} images',body);image_bodies.append(body)
            if not a.readout_only:
                check('image with thought',dict(image_bodies[0],think_mode='always',think_budget=16,
                    return_details=True,return_thought=True))
        else:
            check('disabled images refused',dict(base,images=[vision_suite.tile('red')]),422)
        def same(_):
            code,data=http(base)
            assert code==200 and data['answers']==alone['answers'],('concurrent output mismatch',code,data)
            return data
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(same,range(32)))
        print('PASS HTTP 32 requests from 8 clients: exact answers',flush=True)
        barrier=threading.Barrier(16)
        def overloaded(_):
            barrier.wait()
            body=dict(base) if a.readout_only else dict(base,think_mode='always',think_budget=16)
            return http(body)
        with ThreadPoolExecutor(max_workers=16) as pool:overload=list(pool.map(overloaded,range(16)))
        assert all(code==200 for code,_ in overload),overload
        report['concurrent_thought_requests']=len(overload)
        check('recovery after concurrent thoughts',base)
        print('PASS 16 concurrent requests and drain',flush=True)
        # Exercise eviction before measuring retained memory, then keep churning a
        # working set larger than the four-entry cache during the sustained run.
        cache_bodies=[dict(base,state=f'Incident {i}: '+
            'The service is down, staff cannot log in, and payments are failing. '*24)
            for i in range(12)]
        for body in cache_bodies:
            assert http(body)[0]==200
        for body in image_bodies:
            assert http(body)[0]==200
        before_stats=http(path='/stats')[1]
        assert before_stats['prefix_cache']['evictions']>0,'cache churn did not evict'
        warm=http(path='/health')[1]['memory']
        t0=time.monotonic();last=t0;latencies=[]
        report.update(input_tokens=0,output_tokens=0,thought_requests=0,image_requests=0)
        samples=out.with_suffix('.samples.jsonl').open('w')
        report['control_checks']=0;report['memory_samples']=[]
        def sustained(i):
            body=dict(base);category='text'
            if i%3==0:
                body=dict(cache_bodies[(i//3)%len(cache_bodies)]);category='cache'
            if i%17==0 and not a.readout_only:
                body.update(think_mode='always',think_budget=16);category='thought'
            if image_bodies and i%53==0:
                body=dict(image_bodies[(i//53)%len(image_bodies)]);category='image'
                if i%106==0 and not a.readout_only:
                    body.update(think_mode='always',think_budget=16);category='image-thought'
            t=time.monotonic();code,data=http(body)
            return i,body,category,time.monotonic()-t,code,data
        with ThreadPoolExecutor(max_workers=4) as pool:
            while time.monotonic()-t0<a.seconds or report['requests']<a.requests:
                for i,body,category,elapsed,code,data in pool.map(sustained,range(report['requests'],report['requests']+4)):
                    latencies.append(elapsed);report['requests']+=1
                    if code!=200:
                        report['errors']+=1;raise RuntimeError(('unexpected HTTP error during load',code,data))
                    if category=='text':
                        assert data['answers']==alone['answers'],'state leaked into text control'
                        report['control_checks']+=1
                    usage=data.get('usage',{})
                    report['input_tokens']+=usage.get('input_tokens',0)
                    report['output_tokens']+=usage.get('output_tokens',0)
                    report['thought_requests']+=usage.get('output_tokens',0)>0
                    report['image_requests']+=bool(body.get('images'))
                    samples.write(json.dumps(dict(index=i,category=category,elapsed_s=elapsed,usage=usage,
                        images=len(body.get('images',[])),timings=data.get('timings',{}),timing_ms=data.get('timing_ms',{})))+'\n')
                if time.monotonic()-last>=30:
                    report['elapsed_s']=time.monotonic()-t0
                    report['latest_health']=http(path='/health')[1]
                    report['memory_samples'].append(dict(t_s=report['elapsed_s'],memory=report['latest_health']['memory']))
                    out.write_text(json.dumps(report,indent=2)+'\n')
                    samples.flush()
                    print('stability',round(report['elapsed_s'],1),'seconds',report['requests'],'requests',flush=True)
                    last=time.monotonic()
        final=http(path='/health')[1];report['final_health']=final
        samples.close()
        report.update(stability_s=time.monotonic()-t0,final_stats=http(path='/stats')[1])
        report['sustained_cache_evictions']=report['final_stats']['prefix_cache']['evictions']-before_stats['prefix_cache']['evictions']
        if a.seconds>=1800:
            assert report['sustained_cache_evictions']>0 and report['control_checks']>0
            if not a.readout_only:assert report['thought_requests']>0
            if image_bodies:assert report['image_requests']>0
        report['requests_per_second']=report['requests']/max(report['stability_s'],1e-9)
        report['stable_allocated_growth']=final['memory']['allocated']-warm['allocated']
        assert report['stable_allocated_growth']<=256*2**20,report['stable_allocated_growth']
        if latencies:
            ordered=sorted(latencies)
            report['latency_s']={str(q):ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))] for q in [.5,.95,.99]}
        report['qualification_pass']=True
        print('PASS HTTP qualification',report['requests'],'sustained requests',report['stability_s'],'seconds',flush=True)
    finally:
        proc.terminate()
        try:proc.wait(timeout=30)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        report['server_stopped']=True
        out.write_text(json.dumps(report,indent=2)+'\n')
