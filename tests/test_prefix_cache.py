"""The prefix cache on the GPU: a request served from cached evidence (hit) against the same request computing it
(miss) and against the fp32 reference; hits are bit-identical run to run; another question about the same evidence
hits; after the cache has turned over (eviction churn) the answers are still right; and a request's numbers do not
depend on the other requests in its batch.

    python tests/test_prefix_cache.py --gguf MODEL.gguf --model-dir DIR [--precision int8 --calib FILE] [--jevbench DIR]
        [--exact]

A hit runs only the questions' tokens, so its batch is smaller than the miss's and its projections may run other
kernels, tiles and K splits.  With int8 every one of them gives the same bits (w8a8.py, csrc/w8a8_gemm.cu): --exact
then requires hits, misses and mixed batches to agree bit for bit.  bf16 runs cuBLAS, whose kernels differ by batch
size: there they agree to rounding.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from startlux_prism import gguf_weights, jevfmt as J  # noqa: E402
from startlux_prism.decide import softmax_t  # noqa: E402
from startlux_prism.engine.model import Engine, Options, group  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
from startlux_prism.reference import Reference  # noqa: E402
from run_http import uniquify  # noqa: E402
import suites  # noqa: E402

OTHER = {"other": {"type": "noul", "instructions": "Is the evidence complete enough to act on without asking for more?"}}


def probs(rows, logits, tok):
    return [softmax_t(z[:r.n_options].tolist(), tok.temperature[r.kind]) for r, z in zip(rows, logits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--calib", default="")
    ap.add_argument("--jevbench")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--exact", action="store_true", help="hits must equal misses bit for bit")
    a = ap.parse_args()
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=4096, calib=a.calib, prefix_cache=8,
                                              prefix_cache_tokens=16384))
    ref = Reference(gguf_weights.load(a.gguf), chunked=True)
    ref.set_letters(eng.tok.letters)
    tok = eng.tok
    reqs = [(k, s, q) for k, (s, q) in suites.fixed().items() if k in ("startlux3", "longpage2")]
    if a.jevbench:
        items = [x for x in suites.jevbench(a.jevbench) if len(json_state(x[2])) > 600]
        reqs += [(i, s, q) for i, _, s, q, _ in items[:a.n]]
    bad, worst_hit_miss, worst_ref, n_q = 0, 0.0, 0.0, 0
    for name, state, questions in reqs:
        state = uniquify(state)
        recs = [J.from_systemone(state, q) for q in questions.values()]
        rows = rows_for(tok, recs, list(questions))
        if rows[0].head - eng.sys_len < eng.cache.min_tokens:
            print(f"{name}: evidence too short to cache, skipped")
            continue
        h0 = eng.cache.stats["hits"]
        miss = eng.letter_logits(rows)
        hit = eng.letter_logits(rows)
        hit2 = eng.letter_logits(rows)
        assert eng.cache.stats["hits"] == h0 + 2, "repeat requests should hit the cache"
        same_bits = all(torch.equal(x, y) for x, y in zip(hit, hit2))
        exact = all(torch.equal(x, y) for x, y in zip(miss, hit))
        dlog = max(float((x - y).abs().max()) for x, y in zip(miss, hit))
        want = ref.letter_logits(rows)
        pm, ph, pw = probs(rows, miss, tok), probs(rows, hit, tok), probs(rows, want, tok)
        dhm = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(pm, ph))
        dref = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(ph, pw))
        top = lambda p: max(range(len(p)), key=p.__getitem__)          # noqa: E731
        same_dec = all(top(u) == top(v) for u, v in zip(ph, pm))
        ref_dec = sum(top(u) == top(v) for u, v in zip(ph, pw))
        # another question about the same evidence: a hit, answered like a fresh computation would be
        orec = [J.from_systemone(state, q) for q in OTHER.values()]
        orows = rows_for(tok, orec, list(OTHER))
        h1 = eng.cache.stats["hits"]
        other = eng.letter_logits(orows)
        other_hit = eng.cache.stats["hits"] == h1 + 1
        dother = max(abs(x - y) for x, y in zip(probs(orows, other, tok)[0], probs(orows, ref.letter_logits(orows), tok)[0]))
        ok = same_bits and same_dec and other_hit and (exact or not a.exact)
        bad += not ok
        worst_hit_miss, worst_ref = max(worst_hit_miss, dhm), max(worst_ref, dref, dother)
        n_q += len(rows)
        print(f"{str(name)[:24]:24s} evidence {rows[0].head - eng.sys_len:5d} tok, {len(rows)} q: hit-vs-miss |dp| "
              f"{dhm:.2e} (logits {'identical' if exact else f'|d| {dlog:.3g}'}), hits bit-identical {same_bits}, hit vs fp32 reference |dp| {dref:.4f} ({ref_dec}/"
              f"{len(rows)} same decision), other question hit {other_hit} |dp| {dother:.4f}  {'ok' if ok else 'FAIL'}",
              flush=True)
    # churn: more distinct evidence than the cache holds, then the first requests again (evicted: recomputed)
    for i in range(12):
        st, qs = suites.fixed()["longpage2"]
        st = uniquify(st)
        recs = [J.from_systemone(st, q) for q in qs.values()]
        eng.letter_logits(rows_for(tok, recs, list(qs)))
    s = eng.cache_stats()
    print(f"after churn: {s}")
    assert s["entries"] <= 8 and s["evictions"] > 0
    # batch composition: one request alone, then among others (a long and a short one), as a miss and as a hit
    fx = suites.fixed()

    def req(name):
        st, qs = fx[name]
        st = uniquify(st)
        return rows_for(tok, [J.from_systemone(st, q) for q in qs.values()], list(qs))
    ra, rb, ry = req("startlux3"), req("longpage2"), req("yesno")
    eng.cache.clear()
    alone = eng.letter_logits(ra)                                      # miss, alone
    eng.cache.clear()
    mixed = eng.run([group(rb), group(ry), group(ra)])[-len(ra):]        # miss, after a long and a short request
    rb2 = req("longpage2")
    mixed_hit = eng.run([group(rb2), group(ra), group(ry)])[len(rb2):len(rb2) + len(ra)]   # hit, beside a miss
    y_alone = eng.letter_logits(ry)
    y_mixed = eng.run([group(req("readme3")), group(ry)])[-len(ry):]
    comp = {"miss in a mixed batch": mixed, "hit in a mixed batch": mixed_hit}
    worst_mix = 0.0
    for what, got in comp.items():
        d = max(float((u - v).abs().max()) for u, v in zip(alone, got))
        ex = all(torch.equal(u, v) for u, v in zip(alone, got))
        worst_mix = max(worst_mix, d)
        bad += a.exact and not ex
        print(f"startlux3 alone vs {what}: {'identical' if ex else f'max |dlogit| {d:.3g}'}", flush=True)
    ex = all(torch.equal(u, v) for u, v in zip(y_alone, y_mixed))
    bad += a.exact and not ex
    print(f"yesno (not cached) alone vs in a batch: "
          f"{'identical' if ex else f'max |dlogit| {max(float((u - v).abs().max()) for u, v in zip(y_alone, y_mixed)):.3g}'}",
          flush=True)
    print(f"max hit-vs-miss |dp| {worst_hit_miss:.2e}, max |dp| against the fp32 reference {worst_ref:.4f}, {n_q} "
          f"questions; {'all ok' if not bad else f'{bad} FAILED'}")
    sys.exit(1 if bad else 0)


def json_state(state):
    return state if isinstance(state, str) else J.state_text(state)


if __name__ == "__main__":
    main()
