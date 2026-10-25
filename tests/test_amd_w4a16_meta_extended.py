"""Gate FP16 projection optimization against independent dequantization and the frozen profile."""
import json,os,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_amd_backend import synthetic
from startlux_prism.gguf import GGML_Q4_K,GGML_Q6_K
from startlux_prism.kernels import w4a8 as W,amd_w4a16 as H,amd_w4a16_meta as C
assert torch.version.hip and torch.cuda.device_count()==1
torch.set_num_threads(8)
configs=[(16,32,4,1,64),(32,32,4,1,64),(32,64,4,1,64),(64,64,4,1,64),
         (32,64,4,1,32),(64,64,4,1,32),(64,128,4,1,32),(64,128,8,1,32)]
def equal(x,y):return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
count=0
for typ in [GGML_Q4_K,GGML_Q6_K]:
 for k in ([256,5120] if typ==GGML_Q4_K else [256,5120,17408]):
  raw=synthetic(typ,33,k);refw=raw.dequant(torch.float32).half().double();w=W.repack(raw.to('cuda'))
  x=torch.randn(65,k,generator=torch.Generator().manual_seed(k)).half();x[0]=0;x[1,::17]*=20
  reference=x.double()@refw.T;bound=1e-5+5e-6*(x.double().abs()@refw.abs().T)
  x=x.cuda();live=torch.tensor([65],device='cuda',dtype=torch.int32)
  baseline=torch.ones((65,40),device='cuda');H.gemm(x,w,baseline[:,3:36],live,1)
  assert torch.all((baseline[:,3:36].cpu().double()-1-reference).abs()<=bound)
  for cfg in (configs if typ==GGML_Q4_K else configs+[(16,64,4,1,256)]):
   for dtype,epi in [(torch.float32,0),(torch.float32,1),(torch.float16,0),(torch.bfloat16,0)]:
    for n in [0,1,64,65]:
     live.fill_(n);want=torch.randn((65,40),generator=torch.Generator().manual_seed(42+n+epi)).to(device='cuda',dtype=dtype);got=want.clone()
     H.gemm(x,w,want[:,3:36],live,epi);C.gemm(x,w,got[:,3:36],live,epi,cfg)
     assert equal(got,want),(typ,k,cfg,dtype,n,float((got.float()-want.float()).abs().max()))
     count+=1
  print('PASS hoisted metadata exact profile',typ,k,flush=True)
live.fill_(65);y=torch.ones((65,40),device='cuda');fn=lambda:C.gemm(x,w,y[:,3:36],live,1,(16,64,4,1,256))
fn();full=y.clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):y.fill_(1);fn()
torch.cuda.current_stream().wait_stream(stream)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):fn()
for n in [65,1,0,64]:
 live.fill_(n);y.fill_(1);graph.replay();want=torch.ones_like(y);want[:n]=full[:n]
 assert equal(y,want),('graph live rows',n)
Path(os.environ['ACCEL_OUT'],'checks.json').write_text(json.dumps(dict(full_tensor_checks=count,graph_masks=True,independent_fp64_bound=True,exact_frozen_profile=True))+'\n')
print('PASS',count,'exact profile checks and graph masks',flush=True)
