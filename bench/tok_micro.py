"""Tokenizer cost on the request path: encode vs encode_batch (rayon) for a request's prompts, and the split
(head cached + tails) path the server uses.

    python bench/tok_micro.py MODEL_DIR [--request startlux3]
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.prompt import Tokenizer, render_text, rows_for, split_prompt  # noqa: E402
from run_http import uniquify  # noqa: E402
import suites  # noqa: E402


def t(fn, n=200):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--request", default="startlux3")
    a = ap.parse_args()
    tok = Tokenizer(a.model_dir)
    state, questions = suites.fixed()[a.request]
    recs = [J.from_systemone(state, q) for q in questions.values()]
    texts = [render_text(r)[0] for r in recs]
    heads_tails = [split_prompt(x) for x in texts]
    print(f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}  {len(texts)} prompts, "
          f"{sum(len(tok.encode(x)) for x in texts)} tokens")
    print(f"  encode each           {t(lambda: [tok.encode(x) for x in texts]):7.3f} ms")
    print(f"  encode_batch          {t(lambda: tok.encode_batch(texts)):7.3f} ms")
    print(f"  tails: encode each    {t(lambda: [tok.encode(tl) for _, tl in heads_tails]):7.3f} ms")
    print(f"  tails: encode_batch   {t(lambda: tok.encode_batch([tl for _, tl in heads_tails])):7.3f} ms")
    print(f"  head (one)            {t(lambda: tok.encode(heads_tails[0][0])):7.3f} ms")
    print(f"  render (from_systemone + render_text) {t(lambda: [render_text(J.from_systemone(state, q))[0] for q in questions.values()]):7.3f} ms")
    print(f"  rows_for, cached head {t(lambda: rows_for(tok, recs, list(questions))):7.3f} ms")
    u = [uniquify(state) for _ in range(250)]
    it = iter(u)
    print(f"  rows_for, new head    {t(lambda: rows_for(tok, [J.from_systemone(next(it), q) for q in questions.values()], list(questions)), n=240):7.3f} ms")


if __name__ == "__main__":
    main()
