"""Exact W4A8 metadata reuse and optional bounded, load-time weight reordering.

Scaled integer weights use two signed INT8 dot products. Per-256-input FP32
updates preserve the frozen arithmetic. Canonical weights always remain available
for unqualified shapes; graph capture performs no allocation or repacking.
"""
import time
import torch
import triton
import triton.language as tl
from ..backends import math as F

@triton.jit
def _gemm_meta(A,AS,XS,Q,SC,MN,D,DM,Y,LIVE,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,LD:tl.constexpr,
                                                                Q4:tl.constexpr,EPI:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,UNROLL:tl.constexpr,LAYOUT:tl.constexpr=0):
    r=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN)
    live=tl.load(LIVE)
    if tl.program_id(0)*BM>=live:return
    k=tl.arange(0,BK)
    y=tl.zeros((BM,BN),tl.float32)
    for b in range(K//256):
        acc=tl.zeros((BM,BN),tl.int32)
        for part in tl.range(0,256//BK,loop_unroll_factor=UNROLL,num_stages=1):
            a=tl.load(A+r[:,None]*K+b*256+part*BK+k[None,:],r[:,None]<live,0)
            if Q4:
                pk=tl.arange(0,BK//2)
                if LAYOUT==1:offset=(b*N+n[:,None])*128
                else:offset=n[:,None]*(K//2)+b*128
                packed=tl.load(Q+offset+part*(BK//2)+pk[None,:],n[:,None]<N,0)
                pp=tl.reshape(packed,(BN,BK//32,16))
                q=tl.reshape(tl.permute(tl.join((pp&15).to(tl.int32),(pp>>4).to(tl.int32)),(0,1,3,2)),(BN,BK))
                g=part*(BK//32)+tl.arange(0,BK//32)
                sc=tl.load(SC+(b*N+n[:,None])*8+g[None,:],n[:,None]<N,0).to(tl.int32)
                sc=tl.reshape(tl.broadcast_to(sc[:,:,None],(BN,BK//32,32)),(BN,BK))
            else:
                if LAYOUT==1:offset=(b*N+n[:,None])*256
                else:offset=n[:,None]*K+b*256
                q=tl.load(Q+offset+part*BK+k[None,:],n[:,None]<N,0).to(tl.int32)
                g=part*(BK//16)+tl.arange(0,BK//16)
                sc=tl.load(SC+(b*N+n[:,None])*16+g[None,:],n[:,None]<N,0).to(tl.int32)
                sc=tl.reshape(tl.broadcast_to(sc[:,:,None],(BN,BK//16,16)),(BN,BK))
            scaled=q*sc;lo=(scaled&127).to(tl.int8);hi=(scaled>>7).to(tl.int8)
            acc+=tl.dot(a,tl.trans(lo),out_dtype=tl.int32)+128*tl.dot(a,tl.trans(hi),out_dtype=tl.int32)
        t=F.mul_rn(tl.load(D+b*N+n,n<N,0)[None,:],acc.to(tl.float32))
        if Q4:
            g16=tl.arange(0,16);g8=tl.arange(0,8)
            sums=tl.load(XS+(b*M+r[:,None])*16+g16[None,:],r[:,None]<live,0).to(tl.float16)
            mg=tl.load(MN+(b*N+n[:,None])*8+g8[None,:],n[:,None]<N,0).to(tl.float32)
            mm=tl.reshape(tl.permute(tl.join(mg,mg*64.),(0,2,1)),(BN,16)).to(tl.float16)
            mins=tl.dot(sums,tl.trans(mm)).to(tl.int32)
            t=F.fma(-tl.load(DM+b*N+n,n<N,0)[None,:],mins.to(tl.float32),t)
        y=F.fma(tl.load(AS+b*M+r,r<live,0)[:,None],t,y)
    ptr=Y+r[:,None]*LD+n[None,:];mask=(r[:,None]<live)&(n[None,:]<N)
    if EPI:y=F.add_rn(tl.load(ptr,mask,0),y)
    tl.store(ptr,y,mask)


def prepare_weight(w):
    """Cache one immutable weight in super-block-major order, without expansion."""
    block = 128 if w.kind == 'q4k' else 256
    packed = w.q.reshape(w.n, w.k // 256, block).permute(1, 0, 2).contiguous()
    # The tensor owns this cache: no global data-pointer registry or stale reuse.
    w.q._prism_w4a8_block_major = packed
    return packed


def prepare(layers, budget=1 << 30):
    """Prepare only qualified shapes, before any graph capture, within 1 GiB."""
    from .amd_w4a8_dispatch import table
    entries = table()
    eligible = set()
    for key, choice in entries.items():
        cfg = choice.get('cfg', ())
        if choice['implementation'] == 'scaled_meta' and len(cfg) == 7 and cfg[6] == 1:
            kind, _, n, k, *_ = key.split(':')
            eligible.add((kind, int(n), int(k)))
    candidates, seen = [], set()
    for layer in layers:
        for value in layer.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                w, _ = item
                if not hasattr(w, 'kind') or id(w.q) in seen:
                    continue
                seen.add(id(w.q))
                if (w.kind, w.n, w.k) in eligible:
                    candidates.append(w)
    # Cold-cache benefit per additional byte favors Q4 down before gate/up.
    candidates.sort(key=lambda w: (0 if (w.kind, w.n, w.k) == ('q4k', 4096, 12288) else 1))
    start = time.perf_counter()
    used, count = 0, 0
    budget = max(0, min(budget, 1 << 30))
    for w in candidates:
        size = w.q.numel() * w.q.element_size()
        if used + size > budget:
            continue
        prepare_weight(w)
        used += size
        count += 1
    if count:
        torch.cuda.synchronize()
    return dict(bytes=used, weights=count, budget=budget, seconds=time.perf_counter()-start)


def gemm(x, a, xs, w, y, live, epi=0, cfg=(16, 64, 4, 1, 256, 1)):
    bm, bn, nw, ns, bk, unroll = cfg[:6]
    packed = getattr(w.q, '_prism_w4a8_block_major', None) if len(cfg) == 7 and cfg[6] == 1 else None
    layout = int(packed is not None)
    return _gemm_meta[(triton.cdiv(x.shape[0], bm), triton.cdiv(w.n, bn))](
        x, a, xs, w.q if packed is None else packed, w.sc, w.mn, w.d, w.dm, y, live,
        x.shape[0], w.n, w.k, y.stride(0), w.kind == 'q4k', epi, bm, bn, bk, unroll, layout,
        num_warps=nw, num_stages=ns)
