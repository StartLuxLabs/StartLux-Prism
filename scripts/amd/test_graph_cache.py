"""Actual model: exact cache/graph behavior and same-card eager/graph measurements."""

import os
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import torch
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.prompt import rows_for
import suites


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=("w4a8", "w4a16"), default="w4a8")
    p.add_argument("--reps", type=int, default=12)
    a = p.parse_args()
    P.on()
    eng = Engine(
        a.gguf,
        a.model_dir,
        Options(
            precision=a.precision,
            max_tokens=2048,
            graphs=False,
            prefix_cache=4,
            prefix_cache_tokens=4096,
            prefix_cache_min=16,
        ),
    )
    cases = {}
    for name in ["yesno", "readme3", "startlux3", "review1"]:
        state, qs = suites.fixed()[name]
        cases[name] = rows_for(
            eng.tok, [J.from_systemone(state, q, allow_none=q["type"] == "choice") for q in qs.values()], list(qs)
        )

    def equal(x, y, label):
        assert len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)), label

    result = dict(precision=a.precision, shared_gpu=os.getenv("PRISM_SHARED_GPU") == "1", cases={})
    for name, rows in cases.items():
        eng.opt.share = False
        baseline = eng.letter_logits(rows)
        eng.opt.share = True
        miss = eng.letter_logits(rows)
        hit = eng.letter_logits(rows)
        equal(baseline, miss, name + " cache miss")
        equal(miss, hit, name + " cache hit")
        eng.opt.graphs = True
        graphed = eng.letter_logits(rows)
        equal(hit, graphed, name + " graph cache hit")
        eng.opt.share = False
        equal(baseline, eng.letter_logits(rows), name + " graph recomputation")
        times = {}
        for graphs in [False, True]:
            eng.opt.graphs = graphs
            eng.letter_logits(rows)
            samples = []
            for _ in range(a.reps):
                t = time.perf_counter()
                got = eng.letter_logits(rows)
                samples.append(time.perf_counter() - t)
                equal(baseline, got, name + " repeated invariant")
            times["graph" if graphs else "eager"] = samples
        result["cases"][name] = dict(
            times_s=times,
            eager_median_ms=statistics.median(times["eager"]) * 1000,
            graph_median_ms=statistics.median(times["graph"]) * 1000,
        )
        print("PASS cache miss/hit, eager/graph exact", name, json.dumps(result["cases"][name]), flush=True)
    eng.opt.share = True
    eng.opt.graphs = True
    rows = list(cases.values())[1]
    original = eng.letter_logits(rows)
    for i in range(12):
        st = "Cache eviction test " + str(i) + ". " + "Customer reports an unavailable account. " * 20
        q = {"type": "noul", "instructions": "Is the account available?"}
        rs = rows_for(eng.tok, [J.from_systemone(st, q)], ["decision"])
        eng.letter_logits(rs)
    equal(original, eng.letter_logits(rows), "cache eviction churn")
    result.update(
        cache_stats=eng.cache_stats(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        free_total=torch.cuda.mem_get_info(),
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print("PASS graph and cache qualification", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
