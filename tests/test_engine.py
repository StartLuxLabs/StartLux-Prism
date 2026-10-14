"""Engine against the fp32 reference on real requests (GPU, checkpoint required).

    python tests/test_engine.py --gguf MODEL.gguf --model-dir DIR [--jevbench PUBLIC_DIR] [--n 12] [--no-graphs]

For every request: letter logits of each question from the engine (tree batch: shared evidence computed once) and from
the reference (each prompt alone, fp32, exact Q8_0 weights); reports the largest logit and probability differences and
whether the decision agrees.  Also checks the engine without evidence sharing and with graphs against eager.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

import torch  # noqa: E402

from startlux_prism import gguf_weights, jevfmt as J  # noqa: E402
from startlux_prism.decide import softmax_t  # noqa: E402
from startlux_prism.engine.model import Engine, Options  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
from startlux_prism.reference import Reference  # noqa: E402
import suites  # noqa: E402


def requests(jevbench, n):
    out = [(k, s, q) for k, (s, q) in suites.fixed().items() if k != "wide33"]
    if jevbench:
        items = suites.jevbench(jevbench)
        step = max(1, len(items) // n)
        out += [(i, s, q) for i, _, s, q, _ in items[::step][:n]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--jevbench")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "int8", "w4a8"))
    ap.add_argument("--calib", default="")
    ap.add_argument("--repeat", type=int, default=0, help="also run every request this many more times and require "
                    "bit-identical logits (determinism)")
    a = ap.parse_args()
    # no prefix cache (tests/test_prefix_cache.py has its own): with a 9B's bf16 weights and the fp32 reference's in
    # the same process, its 5 GiB would not fit
    eng = Engine(a.gguf, a.model_dir, Options(max_tokens=a.max_tokens, graphs=not a.no_graphs, precision=a.precision,
                                              calib=a.calib, prefix_cache=0))
    print(f"engine loaded in {eng.load_s:.1f} s, system prefix {eng.sys_len} tokens", flush=True)
    ref = Reference(gguf_weights.load(a.gguf), chunked=True)
    ref.set_letters(eng.tok.letters)
    worst_logit, worst_p, agree, total = 0.0, 0.0, 0, 0
    nondet = [0]
    for name, state, questions in requests(a.jevbench, a.n):
        recs = [J.from_systemone(state, q) for q in questions.values()]
        rows = rows_for(eng.tok, recs, list(questions))
        got = eng.letter_logits(rows)
        info = dict(eng.last)
        for _ in range(a.repeat):
            again = eng.letter_logits(rows)
            if any(not torch.equal(x, y) for x, y in zip(got, again)):
                print(f"NONDETERMINISTIC: {name}: logits differ between identical runs", flush=True)
                nondet[0] += 1
        eng.opt.share = False
        flat = eng.letter_logits(rows)
        eng.opt.share = True
        want = ref.letter_logits(rows)
        for r, g, f, w in zip(rows, got, flat, want):
            n = r.n_options
            dl = float((g[:n] - w[:n]).abs().max())
            ds = float((g[:n] - f[:n]).abs().max())
            T = eng.tok.temperature[r.kind]
            pg, pw = softmax_t(g[:n].tolist(), T), softmax_t(w[:n].tolist(), T)
            dp = max(abs(x - y) for x, y in zip(pg, pw))
            same = max(range(n), key=pg.__getitem__) == max(range(n), key=pw.__getitem__)
            worst_logit, worst_p = max(worst_logit, dl), max(worst_p, dp)
            agree += same
            total += 1
            tie = "" if same else (f" (reference top two {sorted(pw)[-1]:.4f} / {sorted(pw)[-2]:.4f}, engine "
                                   f"{sorted(pg)[-1]:.4f} / {sorted(pg)[-2]:.4f})")
            print(f"{str(name)[:22]:22s} {r.key[:10]:10s} len {len(r.ids):5d}  |dlogit| {dl:.4f}  |dp| {dp:.5f}  "
                  f"share-vs-flat {ds:.4f}  {'same' if same else 'DIFF'}{tie}  {info}", flush=True)
    print(f"\nmax |dlogit| {worst_logit:.4f}, max |dp| {worst_p:.5f}, same decision {agree}/{total}" +
          (f", deterministic over {a.repeat + 1} runs: {'yes' if not nondet[0] else 'NO'}" if a.repeat else ""))


if __name__ == "__main__":
    main()
