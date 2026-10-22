"""A/B two engine configurations in one process: requests alternate between them, so GPU clock and thermal changes hit
both alike, and the median of the paired differences is the effect of the option.  Also checks that both return the
same logits when the option should not change results (--same-bits).

    python bench/ab.py --gguf MODEL.gguf --model-dir DIR --calib FILE --jevbench PUBLIC_DIR \\
        --a l2_prefetch=0 --b l2_prefetch=512 [--reps 100] [--same-bits]

--a / --b: comma-separated Options fields (engine/model.py), e.g. "prefix_cache=0" or "l2_prefetch=512,l2_prefetch_ctas=64".
--table-a / --table-b: GEMM dispatch tables (kernels/w8a8.py): each engine records its graphs with its own table's
kernels, so two tables can be compared in one process.
Requests: the fixed suites (unique copies, as the HTTP benchmark sends them) and three JevBench items (longest, 61st,
121st by evidence length); every request is a cache miss.
"""
import argparse
import dataclasses
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine.model import Engine, Options, group  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
from run_http import uniquify  # noqa: E402
import suites  # noqa: E402


def options(spec, base):
    kw = dict(base)
    fields = {f.name: f.type for f in dataclasses.fields(Options)}
    for item in [x for x in spec.split(",") if x]:
        k, v = item.split("=", 1)
        t = fields[k]
        kw[k] = (v.lower() in ("1", "true", "yes")) if t in (bool, "bool") else type(Options().__getattribute__(k))(v)
    return Options(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--table-a", default="")
    ap.add_argument("--table-b", default="")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--suites", default="startlux3,startlux1,readme3,yesno,longpage2,review1")
    ap.add_argument("--same-bits", action="store_true", help="require identical logits from A and B")
    a = ap.parse_args()
    base = dict(precision="int8", max_tokens=8192, calib=a.calib)
    engines = {}
    from startlux_prism.kernels import w8a8 as KW
    for name, spec, table in (("A", a.a, a.table_a), ("B", a.b, a.table_b)):
        if table:
            os.environ["STARTLUX_PRISM_GEMM_TABLE"] = os.path.abspath(table)
        else:
            os.environ.pop("STARTLUX_PRISM_GEMM_TABLE", None)
        KW._table.cache_clear()
        e = Engine(a.gguf, a.model_dir, options(spec, base))
        e.warm_up()                                  # graphs record the kernels this table chooses
        engines[name] = e
    tok = engines["A"].tok
    items = sorted(suites.jevbench(a.jevbench), key=lambda x: -len(str(x[2])))
    cases = {k: suites.fixed()[k] for k in a.suites.split(",")}
    for rank in (0, 60, 120):
        cases[f"jevbench#{rank + 1}"] = (items[rank][2], items[rank][3])
    print(f"A: {a.a or 'defaults'} {a.table_a}   B: {a.b or 'defaults'} {a.table_b}", flush=True)
    bad = 0
    for name, (st, qs) in cases.items():
        ts = {"A": [], "B": []}
        diff_bits = 0
        for i in range(a.reps):
            u = uniquify(st)
            rows = rows_for(tok, [J.from_systemone(u, q) for q in qs.values()], list(qs))
            out = {}
            for k in (("A", "B") if i % 2 else ("B", "A")):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out[k] = engines[k].run([group(rows)])
                torch.cuda.synchronize()
                ts[k].append((time.perf_counter() - t) * 1000)
            diff_bits += not all(torch.equal(x, y) for x, y in zip(out["A"], out["B"]))
        d = [b - x for x, b in zip(ts["A"][10:], ts["B"][10:])]
        m = {k: statistics.median(v[10:]) for k, v in ts.items()}
        bad += a.same_bits and diff_bits > 0
        print(f"{name:16s} A {m['A']:8.3f}  B {m['B']:8.3f} ms  B-A {statistics.median(d):+.3f} ms "
              f"({100 * statistics.median(d) / m['A']:+.2f}%)"
              f"{'' if not a.same_bits else ('  same bits' if not diff_bits else f'  DIFFERENT bits in {diff_bits}')}",
              flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
