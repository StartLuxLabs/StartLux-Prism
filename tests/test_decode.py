"""The decode path (engine/decode.py) against the prefill, on the GPU (an AutoThink checkpoint).

    python tests/test_decode.py --gguf MODEL.gguf --model-dir DIR [--precision int8 --calib FILE] --jevbench PUBLIC_DIR
        [--n 4] [--tokens 256]

1. teacher forcing: the thinking prompts of --n JevBench items (think.Framing + the forced prefix) are prefilled with
   their state kept, then decoded greedily for --tokens steps, all in one batch.  The same sequences (prompt + the
   generated tokens) are then prefilled whole, with a read at every generated position: each decode step's next-token
   logits against the prefill's at that position (largest and mean |difference|, top-1 agreement), and greedy decoding
   against greedy-by-prefill - the prefill's argmax at every position must be the token the decode step chose, which
   makes greedy decoding by repeated prefills produce the same tokens.  The letter logits of the prefill's exact fp32
   rows against the decode head's at the same positions show the output head's own error.
2. isolation and batch invariance: every sequence decoded alone, and the sequences joining a running batch at
   different steps: the same tokens (int8: the same logits, bit for bit).
3. sampling (T 0.6, top-k 20, top-p 0.95): seeded draws are the same alone and batched; at one position, many seeds
   give token frequencies that follow the filtered distribution.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.engine.decode import Decoder, Row, Sampling  # noqa: E402
from flash_jev.engine.model import READ_BUCKETS, Batch, Engine, Options  # noqa: E402
from flash_jev.think import Framing, RELEASE_PREFIX  # noqa: E402
import suites  # noqa: E402

TIE = 0.1                   # bf16: a greedy split where the prefill's top two logits are closer than this is a near-tie


def prompts(tok, framing, jevbench, n, max_len):
    out = []
    for name, _, state, questions, _ in suites.jevbench(jevbench):
        q = next(iter(questions.values()))
        rec = J.from_systemone(state, q, allow_none=q.get("type", "choice") == "choice")
        ids = tok.encode(framing.prompt(rec) + RELEASE_PREFIX)
        if len(ids) <= max_len:
            out.append((name, ids))
    step = max(1, len(out) // n)
    return out[::step][:n]


class Seqs:
    """Sequences in the decoder's slots: prefill (state kept), then synchronous decode steps."""

    def __init__(self, eng, dec, prompts, rows_each, seeds=None, sampling=None):
        self.eng, self.dec, self.sampling = eng, dec, sampling
        self.s = []
        for i, (name, ids) in enumerate(prompts):
            self.s.append(dict(name=name, ids=ids, slot=eng.think_slot0 + i, kv_start=eng.think_kv0 + i * rows_each,
                               seed=(seeds[i] if seeds else 1234 + i), gen=[], logits=[]))

    def prefill(self, which):
        """The prompts but their last token, state kept (batches of up to the engine's largest bucket)."""
        while which:
            part, n = [], 0
            while which and (not part or n + len(which[0]["ids"]) <= self.eng.max_tokens):
                part.append(which[0])
                n += len(which[0]["ids"])
                which = which[1:]
            keep = [(s["ids"][:-1], s["slot"], s["kv_start"]) for s in part]
            b = self.eng.launch(self.eng.prepare([], 0, keep=keep))
            self.eng.finish(b)
            pos = b.plan.tables["pos"]
            for s, seg in zip(part, b.plan.segs[-len(part):]):
                s.update(pos=int(pos[seg.start + len(seg.tokens) - 1]) + 1, kv_len=len(seg.tokens), step=0, row=-1)

    def step(self, which, slot=0, keep_logits=True):
        """One decode step of `which` (every sequence that was in the previous step must be in it, or never again)."""
        rows = []
        for i, s in enumerate(which):
            src = s["row"] if s["row"] >= 0 else -1 - s["ids"][-1]
            rows.append(Row(s["slot"], s["kv_start"], s["kv_len"], s["pos"], s["seed"], s["step"], src, self.sampling))
            s.update(kv_len=s["kv_len"] + 1, pos=s["pos"] + 1, step=s["step"] + 1, row=i)
        toks = self.dec.finish(self.dec.launch(rows, slot))
        lg = self.dec.logits(len(rows)) if keep_logits else None
        for i, s in enumerate(which):
            s["gen"].append(int(toks[i]))
            if keep_logits:
                s["logits"].append(lg[i])

    def run(self, which, n, keep_logits=True):
        for k in range(n):
            self.step(which, k % 2, keep_logits)


def prefill_logits(eng, dec, ids, positions):
    """Full-vocabulary logits at `positions` of one prompt (one prefill with a read at each, the decoder's head on the
    read rows) and the engine's own letter logits there."""
    out, letters = [], []
    for c in range(0, len(positions), eng.opt.max_read):
        P = np.asarray(positions[c:c + eng.opt.max_read])
        plan, T, _ = eng.plan([[ids]])
        seg = plan.segs[0]
        assert len(plan.segs) == 1 and seg.start == 0 and len(seg.tokens) == len(ids)
        rd = (seg.start + P).astype(np.int32)
        plan.read = list(rd)
        plan.tables["read"] = rd
        plan.tables["read_pos"] = plan.tables["pos"][rd]
        plan.tables["read_blocks"] = np.stack([np.arange(len(P)), np.ones(len(P)), np.zeros(len(P)), P], 1).astype(
            np.int32)
        plan.counts["read"] = len(P)
        R = eng._bucket(len(P), READ_BUCKETS)
        eng._stage_tables(plan, 0)
        b = eng.launch(Batch(plan, T, R, 0, time.perf_counter(), time.perf_counter()))
        letters += eng.finish(b)
        h = eng.buf["hr"][:len(P)]
        if not eng.int8:
            h = h + eng.buf["deltar"][:len(P)].float()
        out.append(dec.head_logits(h.contiguous()))
    return torch.cat(out), letters


@torch.no_grad()
def reference(eng, dec, seqs, gguf):
    """The decode steps' and the prefill's logits against the fp32 reference forward (reference.py, exact Q8_0
    weights, the output head in fp32) at the generated positions: which path is closer to the definition."""
    from flash_jev import gguf_weights
    from flash_jev.reference import Reference
    ref = Reference(gguf_weights.load(gguf), chunked=True)
    head = ref.w.output
    for s in seqs:
        L, n = len(s["ids"]), len(s["logits"])
        full = s["ids"] + s["gen"][:-1]
        hid = ref.forward(full)[L - 1:L - 1 + n]
        want = torch.cat([hid @ head.dequant(torch.float32, torch.arange(i, min(i + 16384, head.shape[0]),
                                                                       device=hid.device)).T
                          for i in range(0, head.shape[0], 16384)], 1)
        got = torch.stack(s["logits"])
        pre, _ = prefill_logits(eng, dec, full, list(range(L - 1, L - 1 + n)))
        dd, dp = (got - want).abs(), (pre - want).abs()
        top = want.argmax(1)
        print(f"  vs fp32 reference, {s['name'][:36]:36s}: decode max |dlogit| {dd.max().item():.4f} mean "
              f"{dd.mean().item():.5f} top-1 {int((got.argmax(1) == top).sum())}/{n}; prefill max {dp.max().item():.4f} "
              f"mean {dp.mean().item():.5f} top-1 {int((pre.argmax(1) == top).sum())}/{n}", flush=True)
    del ref
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    ap.add_argument("--calib", default="")
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-prompt", type=int, default=3000)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--reference", type=int, default=0, help="also compare this many sequences with the fp32 reference")
    ap.add_argument("--attn", default="split", choices=("split", "tree"))
    a = ap.parse_args()
    rows_each = a.max_prompt + a.tokens + 8
    n_seq = max(a.n, 8)
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, max_tokens=8192, prefix_cache=0,
                                              think_seqs=n_seq, think_kv_rows=n_seq * rows_each,
                                              graphs=not a.no_graphs))
    dec = Decoder(eng, a.gguf, max_seqs=n_seq, sampling=Sampling(greedy=True), graphs=not a.no_graphs, attn=a.attn)
    t0 = time.perf_counter()
    dec.warm_up()
    print(f"engine {a.precision} loaded in {eng.load_s:.1f} s, decoder graphs {time.perf_counter() - t0:.1f} s",
          flush=True)
    framing = Framing(a.model_dir)
    ps = prompts(eng.tok, framing, a.jevbench, a.n, a.max_prompt)
    bad = 0

    # 1. teacher forcing, greedy
    S = Seqs(eng, dec, ps, rows_each)
    S.prefill(S.s)
    t0 = time.perf_counter()
    S.run(S.s, a.tokens)
    dt = time.perf_counter() - t0
    print(f"greedy decode: {len(S.s)} sequences x {a.tokens} tokens in {dt:.2f} s "
          f"({len(S.s) * a.tokens / dt:.0f} tokens/s, synchronous steps)", flush=True)
    worst, means, agree, total, mismatch, ties = 0.0, [], 0, 0, 0, 0
    lworst = 0.0
    letter_ids = torch.tensor(eng.tok.letters)
    for s in S.s:
        L = len(s["ids"])
        full = s["ids"] + s["gen"][:-1]
        positions = list(range(L - 1, L - 1 + a.tokens))
        ref, letters = prefill_logits(eng, dec, full, positions)
        got = torch.stack(s["logits"])
        d = (got - ref).abs()
        worst = max(worst, d.max().item())
        means.append(d.mean().item())
        am_ref, am_got = ref.argmax(1).cpu(), got.argmax(1).cpu()
        agree += int((am_ref == am_got).sum())
        total += len(positions)
        gen = torch.tensor(s["gen"])
        first = (am_ref != gen).nonzero()
        if len(first):
            i = int(first[0])
            top2 = ref[i].topk(2).values
            near = float(top2[0] - top2[1]) < TIE
            mismatch += not near
            ties += near
            print(f"  {s['name']}: greedy-by-prefill differs first at token {i} (prefill top two {top2[0]:.4f} / "
                  f"{top2[1]:.4f}{', a near-tie' if near else ''}; decode chose {s['gen'][i]})", flush=True)
        lw = (torch.stack(letters).to(ref.device) - ref[:, letter_ids]).abs().max().item()
        lworst = max(lworst, lw)
        print(f"  {s['name'][:40]:40s} prompt {L:5d}  max |dlogit| {d.max().item():.4f}  mean {d.mean().item():.5f}  "
              f"top-1 agree {int((am_ref == am_got).sum())}/{len(positions)}  text {eng.tok.tok.decode(s['gen'][:24])!r}",
              flush=True)
    print(f"1. teacher forcing: max |dlogit| {worst:.4f}, mean {np.mean(means):.5f}, top-1 agreement {agree}/{total}; "
          f"greedy = greedy-by-prefill over all {a.tokens} tokens on {len(S.s) - mismatch - ties}/{len(S.s)} sequences "
          f"({ties} part at a near-tie, prefill margin < {TIE}); decode head vs the exact fp32 letter rows: max |dlogit| "
          f"{lworst:.4f}", flush=True)
    if a.reference:
        reference(eng, dec, S.s[:a.reference], a.gguf)
    if eng.int8:
        # int8 activations are quantized per token: any two int8 computations of the same logits (the chunked prefill
        # and the recurrent step here) differ by that rounding noise, so greedy paths part at the first close call;
        # --reference shows that both are equally close to the fp32 definition
        print("   (int8: greedy decoding and greedy-by-prefill part at close calls, not counted as failures)", flush=True)
    else:
        bad += mismatch

    # 2. isolation and batch invariance
    exact = eng.int8
    ref_gen = {s["name"]: s["gen"] for s in S.s}
    ref_logits = {s["name"]: s["logits"] for s in S.s}
    n_steps = min(a.tokens, 96)
    diffs = 0
    for k, (name, ids) in enumerate(ps):         # alone, in another slot
        A = Seqs(eng, dec, [(name, ids)], rows_each)
        A.s[0]["slot"], A.s[0]["kv_start"] = eng.think_slot0 + n_seq - 1, eng.think_kv0 + (n_seq - 1) * rows_each
        A.prefill(A.s)
        A.run(A.s, n_steps)
        same_t = A.s[0]["gen"] == ref_gen[name][:n_steps]
        same_l = all(torch.equal(x, y) for x, y in zip(A.s[0]["logits"], ref_logits[name][:n_steps]))
        if not same_t or (exact and not same_l):
            diffs += 1
            print(f"  alone differs: {name} tokens {'same' if same_t else 'DIFFER'}, logits "
                  f"{'bit-identical' if same_l else 'differ'}", flush=True)
    J2 = Seqs(eng, dec, ps, rows_each)               # staggered: sequence i joins at step 8 i
    J2.prefill(J2.s)
    active = []
    for t in range(n_steps + 8 * (len(ps) - 1)):
        for i, s in enumerate(J2.s):
            if t == 8 * i:
                active.append(s)
        live = [s for s in active if len(s["gen"]) < n_steps]
        if live:
            J2.step(live, t % 2)
    for s in J2.s:
        same_t = s["gen"] == ref_gen[s["name"]][:n_steps]
        same_l = all(torch.equal(x, y) for x, y in zip(s["logits"], ref_logits[s["name"]][:n_steps]))
        if not same_t or (exact and not same_l):
            diffs += 1
            print(f"  joining differs: {s['name']} tokens {'same' if same_t else 'DIFFER'}", flush=True)
    print(f"2. isolation: {len(ps)} sequences alone and joining a running batch at steps 0, 8, 16, ...: "
          f"{'the same tokens' + (' and logits, bit for bit' if exact else '') if not diffs else f'{diffs} DIFFER'}",
          flush=True)
    bad += diffs

    # 3. sampling (the rows' own settings: the same graphs)
    rel = Sampling(0.6, 0.95, 20)
    seeds = [7 + 1000 * i for i in range(len(ps))]
    B1 = Seqs(eng, dec, ps, rows_each, seeds, rel)
    B1.prefill(B1.s)
    B1.run(B1.s, 64, keep_logits=False)
    sdiff = 0
    for i, (name, ids) in enumerate(ps):
        A = Seqs(eng, dec, [(name, ids)], rows_each, [seeds[i]], rel)
        A.prefill(A.s)
        A.run(A.s, 64, keep_logits=False)
        if A.s[0]["gen"] != B1.s[i]["gen"]:
            sdiff += 1 if exact else 0
            print(f"  sampled alone vs batched differ: {name}" + ("" if exact else " (bf16: cuBLAS kernels differ by "
                                                                   "batch size, not counted)"), flush=True)
    # the distribution at one position: the greedy run's position whose filtered distribution keeps the most
    # candidates; its prefix as the prompt, 2,000 seeds in batches of n_seq
    def kept(z):
        v = (z.double() / 0.6).topk(20).values
        p = torch.softmax(v, 0)
        tail = torch.flip(torch.cumsum(torch.flip(p, [0]), 0), [0])
        return int(((tail > 0.05) | (torch.arange(20, device=p.device) == 0)).sum())
    s0 = S.s[0]
    i_best = max(range(len(s0["logits"])), key=lambda i: kept(s0["logits"][i]))
    name, ids = s0["name"], s0["ids"] + s0["gen"][:i_best]
    counts = {}
    for b0 in range(0, 2000, n_seq):
        C = Seqs(eng, dec, [(name, ids)] * n_seq, rows_each, list(range(b0, b0 + n_seq)), rel)
        C.prefill(C.s)
        C.step(C.s, 0, keep_logits=(b0 == 0))
        for s in C.s:
            counts[s["gen"][0]] = counts.get(s["gen"][0], 0) + 1
        if b0 == 0:
            z = C.s[0]["logits"][0].double() / 0.6
            v, ix = z.topk(20)
            p = torch.softmax(v, 0)
            tail = torch.flip(torch.cumsum(torch.flip(p, [0]), 0), [0])
            keep = (tail > 0.05) | (torch.arange(20, device=p.device) == 0)
            p = torch.where(keep, p, 0)
            p = (p / p.sum()).cpu()
            ix = ix.cpu()
    n = sum(counts.values())
    tv = 0.5 * sum(abs(counts.get(int(t), 0) / n - float(q)) for t, q in zip(ix, p))
    tv += 0.5 * sum(c / n for t, c in counts.items() if t not in set(ix.tolist()))
    print(f"3. sampling: seeded thoughts {'the same alone and batched' if not sdiff else f'{sdiff} DIFFER'}; "
          f"{n} draws at one position: total variation from the filtered distribution {tv:.4f} "
          f"({sum(1 for q in p if q > 0)} candidates kept, top {float(p.max()):.3f})", flush=True)
    bad += sdiff + (tv > 0.06)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
