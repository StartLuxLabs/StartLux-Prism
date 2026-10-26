"""The e4m3 KV cache (--kv-fp8) on the GPU: qkv_norm_rope writing keys and values as e4m3 bytes with one fp32 scale
per token and kv head (against its own bf16 writes: within e4m3's rounding, the queries unchanged), then tree attention
(the body's blocks and the head's one-query blocks, causal and bidirectional) and the decode attention kernel reading
them, each against fp32 softmax attention over the same dequantized keys and values.

    python tests/test_kv_fp8.py
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.engine import batch as B  # noqa: E402
from startlux_prism.kernels import attn as KA  # noqa: E402
from startlux_prism.kernels import decode as KD  # noqa: E402
from startlux_prism.backends import supports_kv_fp8  # noqa: E402

KV0 = 64                                         # first KV arena row of the batch (the engine's KV_SCRATCH0)


def reference(q, gate, k, v, segs, bidir):
    """fp32 attention per segment.  q, gate [T, H, HD]; k, v [T, HKV, HD] (packed order) -> [T, H, HD]."""
    H, HKV, HD = q.shape[1], k.shape[1], q.shape[2]
    out = torch.empty_like(q)
    for s in segs:
        a, L = s.start, len(s.tokens)
        qs = q[a:a + L].transpose(0, 1)
        ks = k[a:a + L].transpose(0, 1).repeat_interleave(H // HKV, 0)
        vs = v[a:a + L].transpose(0, 1).repeat_interleave(H // HKV, 0)
        sc = qs @ ks.transpose(1, 2) * HD ** -0.5
        if not bidir:
            sc = sc.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=q.device).tril(), float("-inf"))
        out[a:a + L] = (sc.softmax(-1) @ vs).transpose(0, 1)
    return out * torch.sigmoid(gate)


def main():
    if not supports_kv_fp8():
        print("skip: e4m3 keys and values need gfx950 or sm_89 or newer")
        return
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = types.SimpleNamespace(n_head=24, n_kv_head=4, head_dim=256, rope_dim=64, eps=1e-6)   # the 27B's heads
    H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    lengths = (1, 5, 33, 130, 257, 640)
    rng = np.random.default_rng(0)
    rows = [list(rng.integers(1000, 2000, n)) for n in lengths]
    plan = B.plan_batch([[r] for r in rows], 0, KV0, whole=True, bm=KA.launch("BM"))
    T, t = plan.n_tok, plan.tables
    dt = lambda x: torch.as_tensor(x, dtype=torch.int32, device=dev)        # noqa: E731

    # 1. the writes: the same projections through the bf16 and the e4m3 arenas
    P = H * 2 * HD + 2 * HKV * HD                    # per head query|gate, then keys, values
    qkv = torch.randn(T, P, device=dev).to(torch.bfloat16)
    q_norm, k_norm = 1 + 0.1 * torch.randn(HD, device=dev), 1 + 0.1 * torch.randn(HD, device=dev)
    inv_freq = (1.0 / (1e6 ** (torch.arange(0, cfg.rope_dim, 2, dtype=torch.float64) / cfg.rope_dim))).float().to(dev)
    pos, kvrow, n_tok = dt(t["pos"]), dt(t["kvrow"]), dt([T])
    kv_bf = torch.zeros(1, 2, HKV, KV0 + T, HD, dtype=torch.bfloat16, device=dev)
    kv8 = torch.zeros(1, 2, HKV, KV0 + T, HD, dtype=torch.uint8, device=dev)
    kvs = torch.zeros(1, 2, HKV, KV0 + T, device=dev)
    q_bf = torch.empty(T, H, HD, dtype=torch.bfloat16, device=dev)
    q_8 = torch.empty_like(q_bf)
    KA.qkv_norm_rope(qkv, q_norm, k_norm, pos, kvrow, inv_freq, q_bf, kv_bf, 0, n_tok, T, cfg)
    KA.qkv_norm_rope(qkv, q_norm, k_norm, pos, kvrow, inv_freq, q_8, kv8, 0, n_tok, T, cfg, kv8s=kvs, kv_mode=16)
    torch.cuda.synchronize()
    assert torch.equal(q_bf, q_8), "the queries must not depend on the KV arena's format"
    rl = kvrow.long()
    deq = kv8.view(torch.float8_e4m3fn).float() * kvs[..., None]                 # [1, 2, HKV, rows, HD]
    got, ref = deq[0][:, :, rl], kv_bf.float()[0][:, :, rl]                       # [2, HKV, T, HD]
    amax = ref.abs().amax(-1, keepdim=True)
    bound = ref.abs() / 16 + amax / 512 + 1e-6       # e4m3: 3 mantissa bits; each row's largest |value| maps to 448
    worst = ((got - ref).abs() / bound).max().item()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"writes: e4m3 keys and values against bf16: rel {rel:.4f}, worst {worst:.2f} of the rounding bound")
    assert worst <= 1.0, worst

    # 2. tree attention on the e4m3 arena against fp32 attention over the same dequantized keys and values
    k = got[0].transpose(0, 1).to(torch.bfloat16).float()                          # [T, HKV, HD], as the kernel reads
    v = got[1].transpose(0, 1).to(torch.bfloat16).float()
    gate = qkv[:, :H * 2 * HD].view(T, H, 2, HD)[:, :, 1]
    blocks, ranges, read_blocks = dt(t["blocks"]), dt(t["ranges"]), dt(t["read_blocks"])
    n_blocks, n_read = dt([len(t["blocks"])]), dt([len(t["read_blocks"])])
    read = torch.as_tensor(plan.read, dtype=torch.long, device=dev)
    refs = {m: reference(q_8.float(), gate.float(), k, v, plan.segs, m) for m in (True, False)}
    for bidir in (True, False):
        out = torch.zeros(T, H * HD, dtype=torch.bfloat16, device=dev)
        KA.tree_attention(q_8, qkv, kv8, 0, out, blocks, n_blocks, len(t["blocks"]), ranges, cfg, kv8s=kvs,
                          kv_mode=16, bidir=bidir)
        outr = torch.zeros(len(plan.read), H * HD, dtype=torch.bfloat16, device=dev)
        KA.tree_attention(q_8[read].contiguous(), qkv[read].contiguous(), kv8, 0, outr, read_blocks, n_read,
                          len(plan.read), ranges, cfg, read=True, kv8s=kvs, kv_mode=16, bidir=bidir)
        out_bf = torch.zeros_like(out)
        KA.tree_attention(q_bf, qkv, kv_bf, 0, out_bf, blocks, n_blocks, len(t["blocks"]), ranges, cfg, bidir=bidir)
        torch.cuda.synchronize()
        r = refs[bidir]
        e = (out.view(T, H, HD).float() - r).abs().max().item()
        er = (outr.view(-1, H, HD).float() - r[read]).abs().max().item()
        d = (out.float() - out_bf.float()).abs().max().item()
        print(f"tree attention, {'bidirectional' if bidir else 'causal':13s}: body max |err| {e:.2e}, read max |err| "
              f"{er:.2e} (outputs up to {r.abs().max().item():.2f}); against the bf16 arena {d:.2e}", flush=True)
        assert e < 2e-2 and er < 2e-2, (bidir, e, er)

    # 3. decode attention: each segment's last token reading all of its keys from the e4m3 arena (kv_len: the keys
    # before the token's own, which the kernel reads as well)
    nb = len(plan.segs)
    starts = [int(t["kvrow"][s.start]) for s in plan.segs]
    assert all(list(t["kvrow"][s.start:s.start + len(s.tokens)]) == list(range(st, st + len(s.tokens)))
               for s, st in zip(plan.segs, starts))
    kv_start, kv_len = dt(starts), dt([len(s.tokens) - 1 for s in plan.segs])
    scratch = KD.attn_scratch(nb, max(lengths), cfg, dev)
    for name, arena, kw in (("e4m3", kv8, dict(kv8s=kvs, kv_mode=16)), ("bf16", kv_bf, {})):
        qd = (q_8 if arena is kv8 else q_bf)[read].contiguous()
        out = torch.zeros(nb, H * HD, dtype=torch.bfloat16, device=dev)
        KD.attention(qd, qkv[read].contiguous(), arena, 0, kv_start, kv_len, scratch, out, dt([nb]), nb, cfg, **kw)
        torch.cuda.synchronize()
        if name == "e4m3":
            e = (out.view(nb, H, HD).float() - refs[False][read]).abs().max().item()
            print(f"decode attention, e4m3 arena: max |err| {e:.2e}")
            assert e < 2e-2, e
            out8 = out
        else:
            d = (out8.float() - out.float()).abs().max().item()
            print(f"decode attention: e4m3 against the bf16 arena {d:.2e}")
    print("ok: e4m3 KV cache")


if __name__ == "__main__":
    main()
