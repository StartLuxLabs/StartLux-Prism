"""The CUDA W8A8 GEMM (csrc/w8a8_gemm.cu) against exact integer math: every tile config, both epilogues, split-K,
stream-K, and odd shapes (N not a multiple of the tile, live rows < allocated rows, more CTAs than k-iterations).
Stream-K is also run repeatedly (its tile counters must come back to zero) and replayed in a CUDA graph with the live
row count changing between replays.

Batch invariance: every config, split mode (but atomic adds) and stream-K must give the same bits as the Triton kernel
(kernels/w8a8.py) - bf16 stores and fp32 residual adds, deferred split-K folded by quant.fold.  On Hopper with an sm_90a
build the warpgroup MMA kernels (csrc/w8a8_wgmma.cu) too: every tile config and stage count, and the TMA pipeline's
deferred split-K.

    python tests/test_cuda_gemm.py [--quick]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev.kernels import cuda_ext as CX  # noqa: E402
from flash_jev.kernels import quant as KQ  # noqa: E402
from flash_jev.kernels import w8a8 as KW  # noqa: E402

SHAPES = [(64, 12352, 2560, 64), (100, 2560, 9216, 77), (384, 18432, 2560, 361), (1, 8192, 2560, 1),
          (256, 2560, 4096, 256), (8, 128, 256, 3), (2048, 18432, 2560, 1999),
          # StartLux-Decision-9B's projections (hidden 4096, feed-forward 12288)
          (64, 12352, 4096, 64), (100, 4096, 12288, 77), (384, 24576, 4096, 361), (256, 4096, 4096, 256),
          # the 27B's (hidden 5120, feed-forward 17408, 48 value heads, 24 attention heads)
          (64, 16480, 5120, 64), (100, 5120, 17408, 77), (384, 34816, 5120, 361), (256, 5120, 6144, 256)]


def rel(x, ref):
    return ((x.double() - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()


def main():
    quick = "--quick" in sys.argv
    torch.manual_seed(0)
    dev = "cuda"
    CX.load()
    bad = 0
    for M, N, K, live in SHAPES:
        a = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        sa = torch.rand(M, device=dev) * 1e-2
        sw = torch.rand(N, device=dev) * 1e-3
        n = torch.tensor([live], dtype=torch.int32, device=dev)
        ref = ((a.double() @ w.double().t()) * sa.double()[:, None] * sw.double()[None, :])[:live]
        # the Triton kernel's bits: what every CUDA path must reproduce
        tri0 = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        KW.gemm(a, sa.view(M, 1), w, sw.view(N, 1), tri0, n, K, epi=0, cfg=KW.pick(M, N, K, 0))
        tri1 = torch.ones(M, N, device=dev, dtype=torch.float32)
        KW.gemm(a, sa.view(M, 1), w, sw.view(N, 1), tri1, n, K, epi=1, cfg=KW.pick(M, N, K, 1))
        same = lambda x, y: torch.equal(x[:live], y[:live])          # noqa: E731
        for cfg, (bm, bn, bk, _) in CX.CFGS.items():
            if K % bk:
                continue
            for stages in (3, 4):
                if not CX.fits(cfg, stages):
                    continue
                # one tile per CTA (and split-K for the residual add)
                out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
                CX.gemm(a, sa, w, sw, out, n, epi=0, cfg=cfg, stages=stages)
                err = rel(out[:live], ref)
                ok = err < 8e-3 and (live == M or out[live:].abs().max().item() == 0)
                diff = [] if same(out, tri0) else ["bf16"]
                e1 = 0.0
                first = None
                for split, mode in ((1, "cluster"), (2, "defer"), (4, "defer"), (2, "cluster"), (4, "cluster"),
                                    (2, "workspace"), (2, "atomic")):
                    if K % (bk * split) or (split > 1 and mode == "cluster" and bm * bn * 4 > (bm + bn) * bk * stages):
                        continue
                    if mode == "workspace" and M * N * split * 4 > CX.WS_BYTES:
                        continue
                    for rep in range(2):                    # deterministic split-K: the same bits every time
                        res = torch.ones(M, N, device=dev, dtype=torch.float32)
                        if mode == "defer" and split > 1:     # int32 partial sums in slots, folded after
                            slots = torch.full((split, M, N), -(1 << 30), dtype=torch.int32, device=dev)
                            CX.gemm(a, sa, w, sw, res, n, epi=1, cfg=cfg, split=split, stages=stages, mode=mode,
                                    slots=slots)
                            KQ.fold(res, slots, KW.Deferred(split, sa, sw), n, M)
                        else:
                            CX.gemm(a, sa, w, sw, res, n, epi=1, cfg=cfg, split=split, stages=stages, mode=mode)
                        e1 = max(e1, rel(res[:live] - 1, ref))
                        if mode != "atomic" and not same(res, tri1) and f"{mode}{split}" not in diff:
                            diff.append(f"{mode}{split}")
                        if split > 1 and mode != "atomic":
                            ok = ok and (first is None or first[0] != (split, mode) or torch.equal(first[1], res))
                            first = ((split, mode), res)
                ok = ok and CX._workspace(a.device)[1].abs().max().item() == 0
                ok = ok and e1 < 1e-4
                # stream-K: bf16 store and residual add, one and two CTAs per SM, twice in a row
                esk = 0.0
                for per_sm, skw in ([(1, 0)] if quick else [(1, 0), (2, 0), (1, 1), (1, CX.PURE_SK)]):
                    for _ in range(2):
                        o = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
                        CX.gemm_sk(a, sa, w, sw, o, n, epi=0, cfg=cfg, stages=stages, ctas_per_sm=per_sm,
                                   sk_waves=skw)
                        esk = max(esk, rel(o[:live], ref))
                        ok = ok and (live == M or o[live:].abs().max().item() == 0)
                        r = torch.ones(M, N, device=dev, dtype=torch.float32)
                        CX.gemm_sk(a, sa, w, sw, r, n, epi=1, cfg=cfg, stages=stages, ctas_per_sm=per_sm,
                                   sk_waves=skw)
                        esk = max(esk, rel(r[:live] - 1, ref) * 80)       # fp32 add: 80x tighter than the bf16 bar
                        if not (same(o, tri0) and same(r, tri1)) and f"sk{per_sm}.{skw}" not in diff:
                            diff.append(f"sk{per_sm}.{skw}")
                ok = ok and esk < 8e-3 and CX._workspace(a.device)[1].abs().max().item() == 0 and not diff
                bad += not ok
                print(f"M={M:4d} N={N:5d} K={K:5d} live={live:4d} cfg={cfg} stages={stages}: bf16 {err:.1e}, "
                      f"fp32 add {e1:.1e}, stream-K {esk:.1e}, "
                      f"{'bits = Triton' if not diff else 'bits differ from Triton: ' + ','.join(diff)}  "
                      f"{'ok' if ok else 'FAIL'}", flush=True)
        if not CX.has_wgmma():
            continue
        for cfg, stages in ((c, s) for c in CX.WG_CFGS for s in (3, 4, 5, 6) if CX.wg_fits(c, s)):
            if K % 128:
                continue
            # Hopper's warpgroup MMA kernel: bf16 store and residual add, twice each (no state between calls)
            ok, diff, err, e1 = True, [], 0.0, 0.0
            for _ in range(2):
                o = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
                CX.gemm_wg(a, sa, w, sw, o, n, epi=0, cfg=cfg, stages=stages)
                err = max(err, rel(o[:live], ref))
                ok = ok and (live == M or o[live:].abs().max().item() == 0)
                r = torch.ones(M, N, device=dev, dtype=torch.float32)
                CX.gemm_wg(a, sa, w, sw, r, n, epi=1, cfg=cfg, stages=stages)
                e1 = max(e1, rel(r[:live] - 1, ref))
                ok = ok and (live == M or (r[live:] == 1).all().item())
                diff += [d for d, x, y in (("bf16", o, tri0), ("add", r, tri1)) if not same(x, y) and d not in diff]
                for split in (2, 4) if CX.WG_CFGS[cfg][2] else ():      # deferred split-K (TMA pipeline)
                    if K % (128 * split):
                        continue
                    res = torch.ones(M, N, device=dev, dtype=torch.float32)
                    slots = torch.full((split, M, N), -(1 << 30), dtype=torch.int32, device=dev)
                    CX.gemm_wg(a, sa, w, sw, res, n, epi=1, cfg=cfg, stages=stages, split=split, slots=slots)
                    KQ.fold(res, slots, KW.Deferred(split, sa, sw), n, M)
                    ok = ok and (live == M or (res[live:] == 1).all().item())
                    if not same(res, tri1) and f"defer{split}" not in diff:
                        diff.append(f"defer{split}")
            ok = ok and err < 8e-3 and e1 < 1e-4 and not diff
            bad += not ok
            print(f"M={M:4d} N={N:5d} K={K:5d} live={live:4d} wgmma cfg={cfg} stages={stages}: bf16 {err:.1e}, "
                  f"fp32 add {e1:.1e}, {'bits = Triton' if not diff else 'bits differ from Triton: ' + ','.join(diff)}"
                  f"  {'ok' if ok else 'FAIL'}", flush=True)
    # graph replay with a different live row count each time (the stream-K split is computed on the device)
    M, N, K = 384, 12352, 2560
    a = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    w = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    sa, sw = torch.rand(M, device=dev) * 1e-2, torch.rand(N, device=dev) * 1e-3
    n = torch.tensor([M], dtype=torch.int32, device=dev)
    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    full = (a.double() @ w.double().t()) * sa.double()[:, None] * sw.double()[None, :]
    CX.gemm_sk(a, sa, w, sw, out, n, cfg=0, stages=4)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        CX.gemm_sk(a, sa, w, sw, out, n, cfg=0, stages=4)
    for live in (1, 97, 128, 129, 300, 384):
        n.fill_(live)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        err = rel(out[:live], full[:live])
        ok = err < 8e-3 and (live == M or out[live:].abs().max().item() == 0)
        bad += not ok
        print(f"graph replay, live={live:3d}: stream-K rel err {err:.1e}  {'ok' if ok else 'FAIL'}", flush=True)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
