"""The MXFP4 engine path is batch-invariant, as the int8 path is: every request alone and all of them in one batch give
the same letter logits, bit for bit (GPU, a Q8_0 checkpoint).  Alone, a short request's output projections take split K
and leave their residual add to the next kernel that reads the residual; in a batch of thousands of tokens they do
not, so that deferred add must land exactly once either way.

    python tests/test_mxfp4_batch.py --gguf MODEL-Q8_0.gguf --model-dir DIR --calib CALIB.pt [--mxfp4-side SIDE.pt]
        [--jevbench PUBLIC_DIR] [--n 48] [--precision mxfp4|int8]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

import torch  # noqa: E402

from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine.model import Engine, Options, group  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--mxfp4-side", default="")
    ap.add_argument("--jevbench", help="the public items' directory: --n of them join the fixed requests")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=32768, help="the largest batch (a smaller GPU: lower it)")
    ap.add_argument("--precision", default="mxfp4", choices=("mxfp4", "int8"))
    a = ap.parse_args()
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, mxfp4_side=a.mxfp4_side,
                                              max_tokens=a.max_tokens, prefix_cache=0))
    reqs = [(k, s, q) for k, (s, q) in suites.fixed().items() if k != "wide33"]
    if a.jevbench:
        items = suites.jevbench(a.jevbench)
        step = max(1, len(items) // a.n)
        reqs += [(i, s, q) for i, _, s, q, _ in items[::step][:a.n]]
    rows = [rows_for(eng.tok, [J.from_systemone(s, q) for q in qs.values()], list(qs)) for _, s, qs in reqs]
    alone = [eng.letter_logits(r) for r in rows]

    batches, cur, n = [], [], 0                  # in order, as many requests as fit each batch
    for i, r in enumerate(rows):
        t = sum(len(x.ids) for x in r)
        if cur and n + t > eng.max_tokens:
            batches.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += t
    batches.append(cur)

    diff, worst, total = 0, 0.0, 0
    for b in batches:
        out = eng.run([group(rows[i]) for i in b])
        tokens = eng.last["tokens"]
        k = 0
        for i in b:
            for x, y in zip(alone[i], out[k:k + len(rows[i])]):
                total += 1
                if not torch.equal(x, y):
                    diff += 1
                    worst = max(worst, float((x - y).abs().max()))
                    print(f"DIFF {str(reqs[i][0])[:30]:30s} |dlogit| {float((x - y).abs().max()):.4f}", flush=True)
            k += len(rows[i])
        print(f"batch of {len(b)} requests, {tokens:,} tokens: {total} rows so far, {diff} differ from alone",
              flush=True)
    if diff:
        print(f"FAILED: {diff} of {total} rows differ between alone and batched ({a.precision}), max |dlogit| {worst:.4f}")
        sys.exit(1)
    print(f"ok: {a.precision}: {total} rows, alone and batched the same bits")


if __name__ == "__main__":
    main()
