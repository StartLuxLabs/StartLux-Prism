"""Bidirectional tree attention (StartLux-Decision v1.1 Instant models) against PyTorch on random inputs (GPU).

    python tests/test_bidir.py

Whole rows (batch.plan_batch whole=True) of several lengths in one packed batch.  Every query of a segment must read
all of its segment's keys (bidir) or the keys up to its own (causal), with GQA and the sigmoid output gate: the body's
blocks, the head's one-query blocks (read=True) and the int8-score variant (k8), each against fp32 softmax attention
over each segment alone.  The masks must matter (bidirectional and causal outputs differ except at each segment's last
token) and the kernel's error must be the same size in both modes.
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.engine import batch as B  # noqa: E402
from startlux_prism.kernels import attn as KA  # noqa: E402

KV0 = 64                                         # first KV arena row of the batch (the engine's KV_SCRATCH0)


def reference(q, gate, k, v, segs, bidir):
    """fp32 attention per segment.  q, gate [T, H, HD]; k, v [T, HKV, HD] (packed order) -> [T, H, HD]."""
    H, HKV, HD = q.shape[1], k.shape[1], q.shape[2]
    out = torch.empty_like(q)
    for s in segs:
        a, L = s.start, len(s.tokens)
        qs = q[a:a + L].transpose(0, 1)                                       # [H, L, HD]
        ks = k[a:a + L].transpose(0, 1).repeat_interleave(H // HKV, 0)       # [H, L, HD]
        vs = v[a:a + L].transpose(0, 1).repeat_interleave(H // HKV, 0)
        sc = qs @ ks.transpose(1, 2) * HD ** -0.5
        if not bidir:
            sc = sc.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=q.device).tril(), float("-inf"))
        out[a:a + L] = (sc.softmax(-1) @ vs).transpose(0, 1)
    return out * torch.sigmoid(gate)


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = types.SimpleNamespace(n_head=24, n_kv_head=4, head_dim=256)       # the 27B's attention heads
    H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    lengths = (1, 5, 33, 130, 257, 640)
    rng = np.random.default_rng(0)
    rows = [list(rng.integers(1000, 2000, n)) for n in lengths]
    plan = B.plan_batch([[r] for r in rows], 0, KV0, whole=True, bm=KA.launch("BM"))
    T, t = plan.n_tok, plan.tables
    assert [len(s.tokens) for s in plan.segs] == list(lengths)

    q = torch.randn(T, H, HD, device=dev).to(torch.bfloat16)
    gate = torch.randn(T, H, HD, device=dev).to(torch.bfloat16)
    k = torch.randn(T, HKV, HD, device=dev).to(torch.bfloat16)
    v = torch.randn(T, HKV, HD, device=dev).to(torch.bfloat16)
    P = H * 2 * HD + 2 * HKV * HD                    # the projection's row: per head query|gate, then keys, values
    qkv = torch.zeros(T, P, dtype=torch.bfloat16, device=dev)
    qkv[:, :H * 2 * HD].view(T, H, 2, HD)[:, :, 1] = gate
    kv = torch.zeros(1, 2, HKV, KV0 + T, HD, dtype=torch.bfloat16, device=dev)
    kvrow = torch.as_tensor(t["kvrow"], dtype=torch.long, device=dev)
    kv[0, 0][:, kvrow] = k.transpose(0, 1)
    kv[0, 1][:, kvrow] = v.transpose(0, 1)
    sc = k.float().abs().amax(-1).clamp_min(1e-10) / 127.0                  # int8 keys: one scale per token, head
    k8 = torch.zeros(1, HKV, KV0 + T, HD, dtype=torch.int8, device=dev)
    k8s = torch.zeros(1, HKV, KV0 + T, device=dev)
    k8[0][:, kvrow] = torch.round(k.float() / sc[..., None]).to(torch.int8).transpose(0, 1)
    k8s[0][:, kvrow] = sc.transpose(0, 1)
    dt = lambda x: torch.as_tensor(x, dtype=torch.int32, device=dev)        # noqa: E731
    blocks, ranges, read_blocks = dt(t["blocks"]), dt(t["ranges"]), dt(t["read_blocks"])
    n_blocks, n_read = dt([len(t["blocks"])]), dt([len(t["read_blocks"])])
    read = torch.as_tensor(plan.read, dtype=torch.long, device=dev)

    refs = {m: reference(q.float(), gate.float(), k.float(), v.float(), plan.segs, m) for m in (True, False)}
    k8f = (k8[0][:, kvrow].float() * k8s[0][:, kvrow][..., None]).transpose(0, 1)    # the int8 keys, dequantized
    qsc = q.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127.0
    q8f = torch.round(q.float() / qsc) * qsc
    refs8 = {m: reference(q8f, gate.float(), k8f, v.float(), plan.segs, m) for m in (True, False)}
    errs = {}
    for bidir in (True, False):
        for name, k8arg, ref in (("bf16", None, refs[bidir]), ("int8 scores", (k8, k8s), refs8[bidir])):
            out = torch.zeros(T, H * HD, dtype=torch.bfloat16, device=dev)
            KA.tree_attention(q, qkv, kv, 0, out, blocks, n_blocks, len(t["blocks"]), ranges, cfg, k8=k8arg,
                              bidir=bidir)
            got = out.view(T, H, HD).float()
            outr = torch.zeros(len(plan.read), H * HD, dtype=torch.bfloat16, device=dev)
            KA.tree_attention(q[read].contiguous(), qkv[read].contiguous(), kv, 0, outr, read_blocks, n_read,
                              len(plan.read), ranges, cfg, k8=k8arg, read=True, bidir=bidir)
            gotr = outr.view(-1, H, HD).float()
            torch.cuda.synchronize()
            e, er = (got - ref).abs().max().item(), (gotr - ref[read]).abs().max().item()
            errs[(bidir, name)] = max(e, er)
            print(f"{'bidirectional' if bidir else 'causal':13s} {name:11s}  body max |err| {e:.2e}  "
                  f"read max |err| {er:.2e}  (outputs up to {ref.abs().max().item():.2f})", flush=True)
            assert e < 2e-2 and er < 2e-2, (bidir, name, e, er)
    last = read
    diff = (refs[True] - refs[False]).abs()
    not_last = torch.ones(T, dtype=torch.bool, device=dev)
    not_last[last] = False
    print(f"bidirectional vs causal reference: max {diff[not_last].max().item():.2f} away from the last tokens, "
          f"{diff[last].max().item():.1e} at them")
    assert diff[not_last].max() > 0.1 and diff[last].max() < 1e-5
    for name in ("bf16", "int8 scores"):
        assert errs[(True, name)] < 4 * errs[(False, name)] + 1e-3, (name, errs)
    print("ok: bidirectional tree attention")


if __name__ == "__main__":
    main()
