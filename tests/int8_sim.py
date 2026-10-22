"""The engine's W8A8 scheme simulated in fp32 on the reference (no GPU): SmoothQuant (alpha 0.75) calibrated on
StartLux-Decision's calibration requests, per-output-channel int8 weights, per-token int8 activations, with the same
shared smoothing groups as the engine (qkv|z|b|a, q|k|v, gate|up share one input).  Compares decisions and
probabilities with the unquantized reference.

    python tests/int8_sim.py MODEL.gguf MODEL_DIR JEVBENCH_PUBLIC_DIR [N_ITEMS]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from startlux_prism import gguf_weights, jevfmt as J, reference as REF  # noqa: E402
from startlux_prism.calibration import CALIBRATION  # noqa: E402
from startlux_prism.decide import softmax_t  # noqa: E402
from startlux_prism.prompt import Tokenizer, rows_for  # noqa: E402

ALPHA = 0.75


def groups(w):
    """id(Q8 matrix) -> (group key, [matrices of the group])"""
    out = {}
    for i, l in enumerate(w.layers):
        sets = [("gu", (l.ffn_gate, l.ffn_up)), ("down", (l.ffn_down,))]
        if w.config.is_attention(i):
            sets += [("qkv", (l.wq, l.wk, l.wv)), ("o", (l.wo,))]
        else:
            sets += [("inp", (l.wqkv, l.wz, l.wbeta, l.walpha)), ("out", (l.wout,))]
        for name, mats in sets:
            for m in mats:
                out[id(m)] = ((i, name), mats)
    return out


def main():
    gguf, model_dir, bench = sys.argv[1:4]
    n_items = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    torch.set_num_threads(16)
    tok = Tokenizer(model_dir)
    ref = REF.Reference(gguf_weights.load(gguf), device="cpu")
    ref.set_letters(tok.letters)
    G = groups(ref.w)                    # the reference's own copies of the matrices
    exact = REF.linear
    amax = {}

    def recording(x, q8):
        key, _ = G[id(q8)]
        m = x.abs().amax(0)
        amax[key] = torch.maximum(amax[key], m) if key in amax else m
        return exact(x, q8)

    REF.linear = recording
    for state, questions in CALIBRATION:
        recs = [J.from_systemone(state, q) for q in questions.values()]
        for r in rows_for(tok, recs, list(questions)):
            ref.forward(r.ids)
    print(f"calibrated {len(amax)} projection groups", flush=True)
    cache = {}

    def prepared(key, mats):
        if key not in cache:
            Wf = torch.cat([m.dequant(torch.float32) for m in mats])
            s = (amax[key].clamp_min(1e-5) ** ALPHA / Wf.abs().amax(0).clamp_min(1e-5) ** (1 - ALPHA)).clamp(1e-3, 1e3)
            per = {}
            for m in mats:
                Wm = m.dequant(torch.float32) * s
                ws = Wm.abs().amax(1).clamp_min(1e-8) / 127
                per[id(m)] = (torch.round(Wm / ws[:, None]).clamp(-127, 127) * ws[:, None])
            cache[key] = (s, per)
        return cache[key]

    def w8a8(x, q8):
        key, mats = G[id(q8)]
        s, per = prepared(key, mats)
        xs = x / s
        sc = xs.abs().amax(1, keepdim=True).clamp_min(1e-10) / 127
        xq = torch.round(xs / sc) * sc
        return xq @ per[id(q8)].T

    import suites
    items = suites.jevbench(bench)
    step = max(1, len(items) // n_items)
    picks = [("readme3",) + suites.fixed()["readme3"], ("startlux3",) + suites.fixed()["startlux3"]]
    picks += [(i, s, q) for i, _, s, q, _ in items[::step][:n_items]]
    same, total, worst = 0, 0, 0.0
    for name, state, questions in picks:
        recs = [J.from_systemone(state, q) for q in questions.values()]
        rows = rows_for(tok, recs, list(questions))
        REF.linear = exact
        want = ref.letter_logits(rows)
        REF.linear = w8a8
        got = ref.letter_logits(rows)
        for r, a, b in zip(rows, want, got):
            n, T = r.n_options, tok.temperature[r.kind]
            pa, pb = softmax_t(a[:n].tolist(), T), softmax_t(b[:n].tolist(), T)
            d = max(abs(x - y) for x, y in zip(pa, pb))
            ok = max(range(n), key=pa.__getitem__) == max(range(n), key=pb.__getitem__)
            same += ok
            total += 1
            worst = max(worst, d)
            print(f"{str(name)[:24]:24s} {r.key[:9]:9s} len {len(r.ids):5d} |dp| {d:.4f} {'same' if ok else 'DIFF'}",
                  flush=True)
    print(f"W8A8 simulation: same decision {same}/{total}, max |dp| {worst:.4f}")


if __name__ == "__main__":
    main()
