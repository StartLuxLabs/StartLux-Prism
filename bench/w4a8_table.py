"""The W4A8 GEMM's tuning table (kernels/w4a8.py pick_cuda): every projection shape the engine runs for the given
Q4_K_M files - the segments it fuses (gate|up, qkv|z|b|a, the attention's q|k|v ...) and the row slices of the last
layer and the head - timed in every CUDA tile config and stage count (csrc/w4a8_gemm.cu) at the engine's token and
read bucket sizes up to 8,192 (larger buckets use the 8,192 entry), weights streamed from DRAM (as
bench/w4a8_micro.py); the fastest per shape and count goes into the table.

    python bench/w4a8_table.py A-Q4_K_M.gguf [B-Q4_K_M.gguf ...] --out FULL.json [--merge FULL.json] [--m 320,640]
        [--table startlux_prism/kernels/w4a8_table_rtx5090.json]

--out writes every config's time as well, --table the choices only (the engine's file).  --merge starts from an
earlier --out file (when it exists) and times only the (shape, count) pairs it lacks.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism import gguf_weights as GW  # noqa: E402
from startlux_prism.engine.model import READ_BUCKETS, TOKEN_BUCKETS  # noqa: E402
from startlux_prism.kernels import w4a8 as W  # noqa: E402
from w4a8_micro import timed  # noqa: E402

M_GRID = sorted(set(READ_BUCKETS) | {m for m in TOKEN_BUCKETS if m <= 8192})


def engine_shapes(path):
    """{(kind, N, K): a W4 of that shape} over the projections the engine builds for this file (engine/model.py
    _w4_weights, _body_w4, _head_w4)."""
    w = GW.load(path)
    cfg = w.config
    qn = cfg.n_head * cfg.head_dim * 2
    shapes = {}

    def add(segs):
        for s, _ in segs:
            shapes.setdefault((s.kind, s.n, s.k), s)

    for i, l in enumerate(w.layers):
        add(W.segments([l.ffn_gate, l.ffn_up], "cuda"))
        add(W.segments([l.ffn_down], "cuda"))
        if cfg.is_attention(i):
            qkv = W.segments([l.wq, l.wk, l.wv], "cuda")
            add(qkv)
            add(W.segments([l.wo], "cuda"))
            if i == cfg.n_layer - 1:                                 # the last layer's keys and values; the head's
                add(W.seg_rows(qkv, qn, W.seg_width(qkv)))           # queries
                add(W.seg_rows(qkv, 0, qn))
        else:
            add(W.segments([l.wqkv, l.wz, l.wbeta, l.walpha], "cuda"))
            add(W.segments([l.wout], "cuda"))
        torch.cuda.empty_cache()
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge")
    ap.add_argument("--table")
    ap.add_argument("--m", default=",".join(map(str, M_GRID)))
    ap.add_argument("--copies", type=int, default=4)
    a = ap.parse_args()
    table, times, done = {}, {}, set()
    if a.merge and os.path.exists(a.merge):
        old = json.load(open(a.merge))
        table, times = old["table"], old["times"]
        done = {(k, m) for k, by_m in table.items() for m in by_m}
    seen = set()
    for path in a.gguf:
        for (kind, N, K), w0 in sorted(engine_shapes(path).items()):
            key = f"{kind},{N},{K}"
            todo = [int(x) for x in a.m.split(",") if (key, x) not in done]
            if key in seen or not todo:
                continue
            seen.add(key)
            ws = [w0] + [W.W4(*[t.clone() if torch.is_tensor(t) else t for t in w0]) for _ in range(a.copies - 1)]
            table.setdefault(key, {})
            times.setdefault(key, {})
            for M in todo:
                xq, xa, xs = W.quantize_ref(torch.randn(M, K, device="cuda"))
                n = torch.tensor([M], dtype=torch.int32, device="cuda")
                out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                res = {}
                for c, (bm, _, _) in W.CUDA_CFGS.items():     # tiles far from the token count skipped
                    if (M <= 64 and bm >= 128) or (M >= 512 and bm <= 16):
                        continue
                    for st in (2, 3, 4):
                        def run(cfg=(c, st)):
                            for w in ws:
                                W.gemm(xq, xa, xs, w, out, n, cfg=cfg)
                        try:
                            res[f"{c},{st}"] = round(timed(run) / len(ws), 2)
                        except RuntimeError:             # stages beyond the shared memory
                            pass
                best = min(res, key=res.get)
                table[key][str(M)] = [int(v) for v in best.split(",")]
                times[key][str(M)] = res
                rule = W.pick_rule(M, N, kind)
                print(f"{key:20s} M={M:5d}  best {best:5s} {res[best]:8.1f} us   rule {rule} "
                      f"{res.get(f'{rule[0]},{rule[1]}', float('nan')):8.1f} us", flush=True)
            del ws
            torch.cuda.empty_cache()
    table = {k: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for k, v in table.items()}
    json.dump({"gpu": torch.cuda.get_device_name(), "note": "bench/w4a8_table.py", "table": table, "times": times},
              open(a.out, "w"), indent=0)
    if a.table:
        json.dump({"gpu": torch.cuda.get_device_name(),
                   "source": "bench/w4a8_table.py (weights streamed from DRAM; every config and stage count timed at the "
                             "engine's bucket sizes up to 8,192)", "table": table}, open(a.table, "w"), indent=1)
    print(f"wrote {a.out}: {len(table)} shapes")


if __name__ == "__main__":
    main()
