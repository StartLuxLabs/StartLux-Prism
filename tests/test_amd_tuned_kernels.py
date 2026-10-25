"""All shipped configurations against full independent CPU arithmetic, with strided stores."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from startlux_prism.gguf import GGML_Q4_K,GGML_Q6_K
from startlux_prism.kernels import w4a8 as W,amd_w4a8_dot as D,amd_q4_prefill as Q4,amd_q6_wmma as Q6
from test_amd_backend import synthetic

def equal(a,b):return a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def launch(impl,cfg,x,a,xs,w,y,n,epi):
 if impl=='dot':return D.gemv(x,a,xs,w,y,n,epi,cfg)
 if impl=='loop':return Q4.gemm(x,a,xs,w,y,n,epi,cfg)
 if impl=='q6scaled':return Q6.gemm(x,a,xs,w,y,n,epi,cfg)
 if impl=='tile':return W.gemm_triton(x,a,xs,w,y,n,epi,cfg)
 raise AssertionError(impl)

def main():
 torch.set_num_threads(8);assert torch.version.hip and torch.cuda.device_count()==1
 data=json.loads((Path(W.__file__).parent/'w4a8_table_gfx1201_rocm101.json').read_text())
 configs=sorted({(key.split(':')[0],v['implementation'],tuple(v['cfg'])) for key,v in data['entries'].items()})
 checked=0
 for kind,typ in [('q4k',GGML_Q4_K),('q6k',GGML_Q6_K)]:
  selected=[(impl,cfg) for q,impl,cfg in configs if q==kind]
  for k in [256,5120,17408]:
   raw=synthetic(typ,33,k);cpu=W.repack(raw);w=W.repack(raw.to('cuda'))
   for m in [8,65]:
    x=torch.randn(m,k,generator=torch.Generator().manual_seed(k+m));x[0]=0;x[1,::2]=127;x[1,1::2]=-127
    q,a,xs=W.quantize_ref(x);ref=W.gemm_ref(q,a,xs,cpu)
    qg,ag,xg=q.cuda(),a.cuda(),xs.cuda();n=torch.tensor([m],device='cuda',dtype=torch.int32)
    for impl,cfg in selected:
     for live in [0,1,m-1,m]:
      n.fill_(live)
      for epi in [0,1]:
       original=torch.ones((m,40),dtype=torch.float32 if epi else torch.bfloat16);got=original.cuda();want=original.clone()
       launch(impl,cfg,qg,ag,xg,w,got[:,3:36],n,epi)
       want[:live,3:36]=1+ref[:live] if epi else ref[:live].bfloat16()
       assert equal(got.cpu(),want),(kind,k,m,live,epi,impl,cfg)
       checked+=1
   print('PASS all shipped configurations',kind,k,len(selected),flush=True)
 print('PASS full CPU oracle checks',checked,'configurations',len(configs),flush=True)

if __name__=='__main__':main()
