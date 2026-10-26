"""Owned benchmark server instrumentation; does not change request or GPU arithmetic.

Internal thought TTFT is measured from ThinkJob.t0 to first consumed token. It
is not a streaming HTTP TTFT: /v1/systemone returns one completed response.
"""
import collections
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch

repo=Path(os.environ.get('PRISM_BENCH_REPO',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(repo))
from startlux_prism import httpserve as H,think as T

path=Path(os.environ['PRISM_SERVICE_EVENTS'])
log=path.open('w',buffering=64*1024)
histogram=collections.Counter()
retire=T.ThinkScheduler._retire

def measured_retire(self,item,overlapped):
    kind,work,step=item
    first=[j for j in work if not getattr(j,'_bench_first_seen',False) and not j.done] if kind=='decode' else []
    result=retire(self,item,overlapped)
    now=time.perf_counter()
    if kind=='decode':
        histogram[len(work)]+=1
        record=dict(kind='decode',t=now,active_rows=len(work),bucket=step.B,overlapped=overlapped,
            launch_to_retire_ms=(now-step.t_launch)*1000,
            internal_thought_first_token_ms=[(now-j.t0)*1000 for j in first if j.t0>0])
        for j in first:j._bench_first_seen=True
    else:
        record=dict(kind='prefill',t=now,request_groups=len(work),packed_tokens=step.T,
            read_rows=step.R,overlapped=overlapped,prepare_to_retire_ms=(now-step.t0)*1000)
    log.write(json.dumps(record)+'\n')
    return result
T.ThinkScheduler._retire=measured_retire
serve=H.serve

def diagnostic_serve(host,port,handle):
    def measured(method,path,body):
        code,payload=handle(method,path,body)
        if method=='GET' and path in ('/health','/stats'):
            data=json.loads(payload)
            data['diagnostic_allocator_telemetry']=True
            data['memory']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            data['decode_rows_histogram']=dict(histogram)
            payload=json.dumps(data).encode();log.flush()
        return code,payload
    return serve(host,port,measured)
H.serve=diagnostic_serve
runpy.run_module('startlux_prism.server',run_name='__main__')
