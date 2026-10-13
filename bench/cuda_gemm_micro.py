"""CUDA W8A8 GEMM (csrc) vs the Triton one vs cuBLASLt int8 (torch._int_mm, int32 out, no epilogue) on the model's
projection shapes, with weights streamed from DRAM (rotating copies, as in a forward pass).  The CUDA kernel is
searched over tile configs x stages x (split-K | stream-K with 1 or 2 CTAs per SM); the best is reported as
(schedule, cfg, stages, split or CTAs per SM).

    python bench/cuda_gemm_micro.py [--m 64,128,256,384,1024,2048] [--copies 4] [--json out.json] [--live-frac 0.8]
    python bench/cuda_gemm_micro.py --m 8,32,64,128,192,256,384,512,768,1024,2048,4096,8192 \
        --table flash_jev/kernels/gemm_table_rtx5090.json

--table writes the dispatch table kernels/w8a8.py reads: per (N x K, M) the fastest of the Triton kernel
(["triton"]), the tile-per-CTA CUDA kernel (["cuda", cfg, stages, split]) and stream-K (["sk", cfg, stages, per_sm, sk_waves]).

--live-frac f: the GEMM runs on M rows of which only f*M are live (as in a padded token bucket).
--gguf MODEL.gguf: that model's projection shapes (default StartLux-Decision-4B's); the 9B's go into the same table
(keys N x K differ):

    python bench/cuda_gemm_micro.py --gguf ~/jev/models/StartLux-Decision-9B-Q8_0-GGUF/StartLux-Decision-9B-Q8_0.gguf \
        --buckets --max-m 8192 --merge-into flash_jev/kernels/gemm_table_rtx5090.json

(--max-m 8192: the buckets the 4B's entries were measured at; a larger bucket takes the 8192 entry, kernels/w8a8.py).
"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev.kernels import cuda_ext as CX  # noqa: E402
from flash_jev.kernels import w8a8 as KW  # noqa: E402
from gemm_micro import gguf_dims, model_shapes, timed  # noqa: E402

EPI = {"gdn_out": 1, "attn_o": 1, "mlp_down": 1}


def fold_us(rows, n, slots):
    """Time the residual's next reader (norm_q8) spends adding deferred split-K slots: the kernel with `slots`
    partial sums to fold minus the same kernel without."""
    key = (rows, n, slots)
    if key not in _FOLD:
        from flash_jev.kernels import quant as KQ
        h = torch.randn(rows, n, device="cuda")
        parts = torch.zeros(slots, rows, n, dtype=torch.int32, device="cuda")
        w = torch.ones(n, device="cuda")
        q = torch.empty(rows, n, dtype=torch.int8, device="cuda")
        sc = torch.empty(rows, device="cuda")
        cnt = torch.tensor([rows], dtype=torch.int32, device="cuda")
        from flash_jev.kernels.w8a8 import Deferred
        pend = Deferred(slots, torch.rand(rows, device="cuda"), torch.rand(n, device="cuda"))
        with_fold = timed(lambda: KQ.norm_q8(h, w, q, sc, cnt, rows, 1e-6, parts=parts, pending=pend), 10)
        without = timed(lambda: KQ.norm_q8(h, w, q, sc, cnt, rows, 1e-6), 10)
        _FOLD[key] = max(0.0, with_fold - without) * 1000
    return _FOLD[key]


_FOLD = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="64,128,256,384,1024,2048")
    ap.add_argument("--copies", type=int, default=4, help="weight copies rotated through (at least enough for 3x "
                    "--l2-mb, so that they stream from DRAM as in a forward pass)")
    ap.add_argument("--l2-mb", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json")
    ap.add_argument("--live-frac", type=float, default=1.0)
    ap.add_argument("--shapes", default="", help="comma-separated shape names (default: all)")
    ap.add_argument("--table", help="write the dispatch table (JSON) here")
    ap.add_argument("--merge-into", help="update this dispatch table (JSON) with the measured shapes only, keeping the "
                    "others")
    ap.add_argument("--reuse", help="rows JSON of an earlier run: take its CUDA timings, time only the Triton kernel")
    ap.add_argument("--cfgs", default="", help="comma-separated CUDA tile configs to search (default: all)")
    ap.add_argument("--no-sk", action="store_true", help="skip the stream-K schedules")
    ap.add_argument("--mma-max-m", type=int, default=0, help="time the mma.sync kernels (tile, stream-K) only up to "
                    "this many rows (0: always; Hopper: the wgmma kernel wins above a few rows)")
    ap.add_argument("--gguf", help="the model whose projection shapes to time (default StartLux-Decision-4B)")
    ap.add_argument("--tp", type=int, default=1, help="a tensor-parallel shard's shapes (flash_jev/tp.py): the "
                    "column-parallel projections' rows and the output projections' inputs divided by this; the output "
                    "projections with int32 sums out (w8a8.gemm_i32, table keys NxKi)")
    ap.add_argument("--buckets", action="store_true", help="M = the engine's token and read buckets, each timed with "
                    "the live row count in the middle of the bucket's range (the rows a padded bucket really has)")
    ap.add_argument("--max-m", type=int, default=0, help="with --buckets: only buckets up to this many rows")
    a = ap.parse_args()
    live_of = {}
    if a.buckets:
        from flash_jev.engine.model import READ_BUCKETS, TOKEN_BUCKETS
        for bs in (TOKEN_BUCKETS, READ_BUCKETS):
            prev = 0
            for m in bs:
                live_of[m] = max(live_of.get(m, 0), (prev + m) // 2 + 1)
                prev = m
        a.m = ",".join(str(m) for m in sorted(live_of) if not a.max_m or m <= a.max_m)
    prev = {(r["shape"], r["M"]): r for r in json.load(open(a.reuse))} if a.reuse else {}
    a.cfgs = [int(c) for c in a.cfgs.split(",") if c]
    dev = "cuda"
    CX.load()
    rows, table = [], {}
    dims = gguf_dims(a.gguf) if a.gguf else {}
    if a.tp > 1:                                                  # a shard: its heads and FFN columns
        dims = dims or dict(hidden=2560, ffn=9216, k_heads=16, v_heads=32, gdn_dim=128, n_head=16, n_kv=4,
                            head_dim=256)
        dims.update({k: dims[k] // a.tp for k in ("ffn", "k_heads", "v_heads", "n_head", "n_kv")})
    shapes = model_shapes(**dims)
    seen = set()
    for name, (N, K) in shapes.items():
        key = name.split(" ")[0]
        if (N, K) in seen:                                        # 9B: attn_o has gdn_out's shape
            continue
        seen.add((N, K))
        if a.shapes and key not in a.shapes.split(","):
            continue
        epi = EPI.get(key, 0)
        copies = max(a.copies, -(-3 * (a.l2_mb << 20) // (N * K)))    # 4 copies of a 10 MB matrix would sit in L2
        ws = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for _ in range(copies)]
        sw = torch.rand(N, device=dev) * 1e-3
        if a.tp > 1 and epi == 1:                                 # an output projection's int32 sums (gemm_i32)
            rows += time_i32(a, key, N, K, ws, sw, live_of, table)
            continue
        for M in [int(x) for x in a.m.split(",")]:
            x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
            sa = torch.rand(M, device=dev) * 1e-2
            n = torch.tensor([live_of.get(M) or max(1, int(M * a.live_frac))], dtype=torch.int32, device=dev)
            out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16 if epi == 0 else torch.float32)
            slots = torch.zeros(4, M, N, dtype=torch.int32, device=dev) if epi == 1 else None      # deferred split-K partial sums

            def rot(fn):
                def f():
                    for w in ws:
                        fn(w)
                return lambda: timed(f, a.iters) / len(ws) * 1000

            tri_cfg = KW.pick(M, N, K, epi)                       # the Triton kernel itself, not the dispatch table
            t_tri = rot(lambda w: KW.gemm(x, sa[:, None], w, sw[:, None], out, n, K, epi=epi, cfg=tri_cfg))()
            old = prev.get((key, M))
            if old:                                               # CUDA and cuBLASLt timings of an earlier run
                best_tile, best_sk = (old["tile_us"], *old["tile_cfg"]), (old["sk_us"], *old["sk_cfg"])
                t_lt = old["cublaslt_int_mm_us"]
            else:
                xl = x if M > 16 else torch.cat([x, x.new_zeros(17 - M, K)])     # cuBLASLt int8 needs M > 16
                t_lt = rot(lambda w: torch._int_mm(xl, w.t()))()
                best_tile, best_sk = None, None
                splits = (1, 2, 4) if epi == 1 else (1,)        # residual add: deferred (deterministic) split-K
                mma = not a.mma_max_m or M <= a.mma_max_m
                for cfg, st, sp in itertools.product(CX.CFGS if mma else (), (3, 4), splits):
                    if K % (CX.CFGS[cfg][2] * sp) or not CX.fits(cfg, st) or (a.cfgs and cfg not in a.cfgs):
                        continue
                    bm, bn, bk, _ = CX.CFGS[cfg]

                    try:
                        t = rot(lambda w: CX.gemm(x, sa, w, sw, out, n, epi=epi, cfg=cfg, split=sp, stages=st,
                                                  mode="defer", slots=slots))()
                    except Exception:  # noqa: BLE001
                        continue
                    if sp > 1:
                        # the residual's next reader (norm_q8) then reads the sp slots and writes the residual back
                        t += fold_us(int(n.item()), N, sp)
                    if best_tile is None or t < best_tile[0]:
                        best_tile = (t, "tile", cfg, st, sp)
                for cfg, st, per_sm, skw in itertools.product(CX.CFGS if mma else (), (3, 4), (1, 2),
                                                              (0, 1, CX.PURE_SK)):
                    if K % CX.CFGS[cfg][2] or not CX.fits(cfg, st) or (a.cfgs and cfg not in a.cfgs) or a.no_sk:
                        continue
                    if per_sm == 2 and not CX.fits(cfg, st) or (per_sm == 2 and 2 * (sum(CX.CFGS[cfg][:2]) *
                                                                                       CX.CFGS[cfg][2] * st) > 100 << 10):
                        continue                                  # two CTAs would not fit on an SM together
                    bm, bn = CX.CFGS[cfg][:2]
                    tiles = -(-int(n.item()) // bm) * -(-N // bn)
                    if skw != CX.PURE_SK and tiles <= per_sm * CX.sm_count():
                        continue                                  # one wave or less: the same as pure stream-K
                    try:
                        t = rot(lambda w: CX.gemm_sk(x, sa, w, sw, out, n, epi=epi, cfg=cfg, stages=st,
                                                     ctas_per_sm=per_sm, sk_waves=skw))()
                    except Exception:  # noqa: BLE001
                        continue
                    if best_sk is None or t < best_sk[0]:
                        best_sk = (t, "sk", cfg, st, per_sm, skw)
            best_wg = None                                        # Hopper: the warpgroup MMA kernel
            for cfg, st in ((c, s) for c in CX.WG_CFGS for s in (3, 4, 5, 6) if CX.has_wgmma() and CX.wg_fits(c, s)):
                # residual add on the TMA pipeline, up to 512 rows: deferred split-K as well
                splits = (1, 2, 4) if epi == 1 and CX.WG_CFGS[cfg][2] and M <= 512 else (1,)
                for sp in splits:
                    if K % (128 * sp):
                        continue
                    t = rot(lambda w: CX.gemm_wg(x, sa, w, sw, out, n, epi=epi, cfg=cfg, stages=st, split=sp,
                                                 slots=slots))()
                    if sp > 1:
                        t += fold_us(int(n.item()), N, sp)
                    if best_wg is None or t < best_wg[0]:
                        best_wg = (t, "wg", cfg, st) + ((sp,) if sp > 1 else ())
            best = min(x for x in (best_tile, best_sk, best_wg) if x)
            best_tile = best_tile or (float("inf"), "tile", -1, 0, 0)
            best_sk = best_sk or (float("inf"), "sk", -1, 0, 0, 0)
            table.setdefault(f"{N}x{K}", {})[str(M)] = (["triton"] if t_tri < best[0] else
                                                        ["cuda", *best[2:]] if best[1] == "tile" else
                                                        [best[1], *best[2:]])
            ops = 2 * int(n.item()) * N * K
            r = {"shape": key, "M": M, "live": int(n.item()), "N": N, "K": K, "triton_us": round(t_tri, 1),
                 "cublaslt_int_mm_us": round(t_lt, 1), "cuda_us": round(best[0], 1), "cuda_cfg": best[1:],
                 "tile_us": round(best_tile[0], 1), "tile_cfg": best_tile[1:], "sk_us": round(best_sk[0], 1),
                 "sk_cfg": best_sk[1:], "cuda_tops": round(ops / best[0] / 1e6, 1)}
            if best_wg:
                r.update(wg_us=round(best_wg[0], 1), wg_cfg=best_wg[1:])
            rows.append(r)
            print(f"{key:12s} M={M:5d} live={r['live']:5d}  triton {t_tri:7.1f}  cublasLt {t_lt:7.1f}  "
                  f"tile {best_tile[0]:7.1f} {best_tile[2:]}  stream-K {best_sk[0]:7.1f} {best_sk[2:]}  "
                  + (f"wgmma {best_wg[0]:7.1f} {best_wg[2:]}  " if best_wg else "") +
                  f"-> {best[0]:7.1f} us ({r['cuda_tops']:5.0f} TOPS, {N * K / best[0] / 1e6:4.2f} TB/s weights)",
                  flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    if a.table:
        json.dump({"gpu": torch.cuda.get_device_name(), "source": SOURCE, "table": table}, open(a.table, "w"), indent=1)
    if a.merge_into:
        merge(a.merge_into, table, torch.cuda.get_device_name())


def time_i32(a, key, N, K, ws, sw, live_of, table):
    """A tensor-parallel output projection's int32 sums (w8a8.gemm_i32) at each M: the Triton kernel, the mma.sync tile
    kernel (up to --mma-max-m rows) and the wgmma TMA pipeline, each with 1, 2 or 4 K splits up to 512 rows - a split
    charged with the fold reading tp x its slots instead of tp (the fused all-reduce reads every rank's).  Entries go
    to table["NxKi"] as ["triton"], ["cuda", cfg, stages, split] or ["wg", cfg, stages, split]."""
    import triton
    dev, out_rows = "cuda", []
    for M in [int(x) for x in a.m.split(",")]:
        x = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
        sa = torch.rand(M, device=dev) * 1e-2
        n = torch.tensor([live_of.get(M) or max(1, int(M * a.live_frac))], dtype=torch.int32, device=dev)
        live = int(n.item())
        slots = torch.zeros(4, M, N, dtype=torch.int32, device=dev)

        def rot(fn):
            def f():
                for w in ws:
                    fn(w)
            return timed(f, a.iters) / len(ws) * 1000

        BM, BN, BK, GM, warps, stages = KW.pick(M, N, K, 0)[:6]
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), 1)
        cands = [(rot(lambda w: KW._w8a8_kernel[grid](x, sa, w, sw, slots[0], sw, n, M, N, K, N, GK=K, BM=BM, BN=BN,
                                                       BK=BK, GROUP_M=GM, EPI=3, SPLIT_K=1, num_warps=warps,
                                                       num_stages=stages)), ["triton"])]
        splits = (1, 2, 4) if M <= 512 else (1,)
        mma = not a.mma_max_m or M <= a.mma_max_m
        for cfg, st, sp in itertools.product(CX.CFGS if mma else (), (3, 4), splits):
            if K % (CX.CFGS[cfg][2] * sp) or not CX.fits(cfg, st) or (a.cfgs and cfg not in a.cfgs):
                continue
            t = rot(lambda w: CX.gemm(x, sa, w, sw, slots[0][:M], n, epi=1, cfg=cfg, split=sp, stages=st,
                                      mode="defer", slots=slots, i32=True))
            cands.append((t + _split_cost(live, N, sp, a.tp), ["cuda", cfg, st, sp]))
        for cfg, st in ((c, s) for c in CX.WG_CFGS for s in (3, 4, 5, 6)
                        if CX.has_wgmma() and CX.WG_CFGS[c][2] and CX.wg_fits(c, s)):
            for sp in splits:
                if K % (128 * sp):
                    continue
                t = rot(lambda w: CX.gemm_wg(x, sa, w, sw, slots[0][:M], n, epi=1, cfg=cfg, stages=st, split=sp,
                                             slots=slots, i32=True))
                cands.append((t + _split_cost(live, N, sp, a.tp), ["wg", cfg, st, sp]))
        t, best = min(cands, key=lambda c: c[0])
        table.setdefault(f"{N}x{K}i", {})[str(M)] = best
        out_rows.append({"shape": key + " (int32)", "M": M, "live": live, "N": N, "K": K, "triton_us": round(cands[0][0], 1),
                         "cuda_us": round(t, 1), "cuda_cfg": best})
        print(f"{key:12s} int32 M={M:5d} live={live:5d}  triton {cands[0][0]:7.1f}  -> {t:7.1f} us {best}  "
              f"({N * K / t / 1e6:4.2f} TB/s weights)", flush=True)
    return out_rows


def _split_cost(rows, n, split, tp):
    """The fold's extra time for `split` int32 slots per rank instead of one (tp ranks' slots read)."""
    return 0.0 if split == 1 else max(0.0, fold_us(rows, n, tp * split) - fold_us(rows, n, tp))


SOURCE = ("bench/cuda_gemm_micro.py (weights streamed from DRAM; tile = one CTA per tile, sk = stream-K, wg = Hopper "
          "warpgroup MMA)")


def merge(path, table, gpu):
    """Replace the measured shapes' entries of a dispatch table (a new table on a GPU model without one)."""
    old = json.load(open(path)) if os.path.exists(path) else {"gpu": gpu, "source": SOURCE, "table": {}}
    assert old.get("gpu") in (None, gpu), f"{path} was measured on {old.get('gpu')}"
    old["table"].update(table)
    old["source"] = old.get("source", "") + f"; re-measured: {', '.join(sorted(table))}"
    json.dump(old, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
