"""Research profile: independent dequantization, bounded FP32 accumulation and masking."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from startlux_prism.gguf import GGML_Q4_K,GGML_Q6_K
from startlux_prism.kernels import w4a8 as W,amd_w4a16 as H
from test_amd_backend import synthetic

torch.set_num_threads(8)
assert torch.cuda.device_count()==1 and torch.version.hip
cases=0
for typ in [GGML_Q4_K,GGML_Q6_K]:
 for k in [256,512,5120]:
  raw=synthetic(typ,33,k);refw=raw.dequant(torch.float32);w=W.repack(raw.to('cuda'))
  ids=torch.tensor([0,32,-1],device='cuda',dtype=torch.int32);live=torch.tensor([3],device='cuda',dtype=torch.int32)
  h=torch.full((3,k),42.,device='cuda');H.embed(ids,w,h,live,3)
  assert torch.equal(h[:2].cpu(),refw[[0,32]])
  assert torch.all(h[2]==42)
  x=torch.randn(35,k,generator=torch.Generator().manual_seed(k)).half();x[0]=0;x[1,::17]*=20
  ref=x.double()@refw.half().double().T
  bound=1e-5+5e-6*(x.double().abs()@refw.half().double().abs().T)
  xg=x.cuda();baseline=None
  for cfg in [(bm,bn,4,1,bk) for bm,bn in [(16,32),(32,32),(16,64)] for bk in [64,128]]:
   y=torch.ones((35,40),device='cuda');live.fill_(35)
   H.gemm(xg,w,y[:,3:36],live,1,cfg)
   got=y[:,3:36].cpu().double()-1
   assert torch.all((got-ref).abs()<=bound),(typ,k,cfg,float(((got-ref).abs()/bound).max()))
   if baseline is None:baseline=y.clone()
   else:assert torch.equal(y,baseline),'batch/tile invariance'
   for n in [0,1,34]:
    y.fill_(42);live.fill_(n);H.gemm(xg,w,y[:,3:36],live,0,cfg)
    assert torch.all(y[n:]==42) and torch.all(y[:,:3]==42) and torch.all(y[:,36:]==42)
   cases+=1
  print('PASS W4A16 independent reference',typ,k,flush=True)
print('PASS',cases,'W4A16 configurations')
