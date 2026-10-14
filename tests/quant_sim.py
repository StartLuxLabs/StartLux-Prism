"""Quantization schemes simulated in fp32 on the reference (fake quantization of every projection's weights and inputs),
compared with the unquantized reference on decisions and probabilities.  Decides whether a lower-precision tensor-core
path is worth building before writing its kernels.

    python tests/quant_sim.py MODEL.gguf MODEL_DIR JEVBENCH_PUBLIC_DIR [--n 24] [--schemes w8a8,nvfp4,nvfp4_sq,mxfp4]
        [--device cpu]

Schemes (all with the engine's shared smoothing groups: qkv|z|b|a, q|k|v and gate|up share one input):
  w8a8      the engine's int8 path: SmoothQuant alpha 0.75, int8 per output channel / per token
  nvfp4     W4A4 NVFP4: e2m1 values, one e4m3 scale per 16 inputs, one fp32 scale per tensor (weights) or per token
            (activations) - the block-scaled FP4 tensor-core format of Blackwell (mma kind::mxf4nvf4, 2x the int8 rate
            on the RTX 5090)
  nvfp4_sq  nvfp4 after SmoothQuant alpha 0.5
  mxfp4     W4A4 MXFP4: e2m1 values, one power-of-two (e8m0) scale per 32 inputs
  w4a8      NVFP4 weights, the engine's int8 per-token activations (SmoothQuant 0.75): 4-bit weights halve the bytes a
            short request streams; the best case for a 4-bit-weight path
  w4a8_mx   MXFP4 weights x MXFP8 activations (e4m3 values, one power-of-two scale per 32 inputs): what Blackwell's
            block-scaled mma (kind::mxf8f6f4) multiplies directly, at the FP8 rate
  kq_a8g32  the file's own weights as they are (a Q4_K_M file: its Q4_K / Q6_K values), int8 activations with one scale
            per 32 inputs and no smoothing: a native W4A8 path with llama.cpp's activation blocks (its MMQ: Q8_1)
  kq_a8g256 the same with one activation scale per 256 inputs (a Q4_K super-block: the sub-block scales then apply in
            integer arithmetic inside it)
  kq_a8tok  the same with one activation scale per token (as the int8 path, without smoothing)
A scheme may add attention quantization: "+qk8" (q and k int8, one scale per token and head: q k^T on int8 tensor
cores, 4x the bf16 rate on the RTX 5090) and "+pv8" (softmax probabilities and v in e4m3), e.g. w8a8+qk8.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from startlux_prism import gguf_weights, jevfmt as J, reference as REF  # noqa: E402
from startlux_prism.calibration import CALIBRATION  # noqa: E402
from startlux_prism.decide import softmax_t  # noqa: E402
from startlux_prism.prompt import Tokenizer, rows_for  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def round_e2m1(x):
    """Nearest e2m1 value (ties to the larger magnitude, as hardware converters differ only there)."""
    grid = E2M1.to(x.device, x.dtype)
    a = x.abs().clamp(max=6.0)
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2)
    return torch.sign(x) * grid[idx]


def round_e4m3(x):
    """Nearest positive float8 e4m3fn value (scales are positive)."""
    return x.clamp(max=448.0).to(torch.float8_e4m3fn).to(x.dtype)


def nvfp4(x):
    """Fake-quantize the last dim in blocks of 16: e2m1 values, e4m3 block scales, an fp32 scale per row."""
    shape = x.shape
    rows = x.reshape(-1, shape[-1])
    g = rows.abs().amax(1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)       # per row: block scales fit e4m3
    b = rows.reshape(rows.shape[0], -1, 16)
    s = round_e4m3(b.abs().amax(-1, keepdim=True) / 6.0 / g[:, :, None]).clamp_min(2 ** -9) * g[:, :, None]
    return (round_e2m1(b / s) * s).reshape(shape)


def mxfp4(x):
    """Fake-quantize the last dim in blocks of 32: e2m1 values, power-of-two block scales."""
    shape = x.shape
    b = x.reshape(-1, shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)                          # OCP MX: 2^(floor(log2 amax) - emax)
    return (round_e2m1(b / s) * s).reshape(shape)


def mxfp8(x):
    """Fake-quantize the last dim in blocks of 32: e4m3 values, power-of-two block scales (OCP MXFP8)."""
    shape = x.shape
    b = x.reshape(-1, shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 8)                          # emax of e4m3 = 8
    return ((b / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype) * s).reshape(shape)


def qk8(x):
    """q or k [L, heads, d]: int8, one scale per token and head."""
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    return torch.round(x / sc).clamp(-127, 127) * sc


def pv8(p, v):
    """softmax probabilities (<= 1) scaled into e4m3's range, v e4m3 with one scale per token and head."""
    pq = (p * 448.0).to(torch.float8_e4m3fn).to(p.dtype) / 448.0
    sv = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return pq, (v / sv).to(torch.float8_e4m3fn).to(v.dtype) * sv


def int8_rows(x):
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    return torch.round(x / sc).clamp(-127, 127) * sc


def int8_groups(g):
    def q(x):
        b = x.reshape(*x.shape[:-1], -1, g)
        sc = b.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        return (torch.round(b / sc).clamp(-127, 127) * sc).reshape(x.shape)
    return q


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


SCHEMES = {  # name -> (smoothing alpha or None, weight quantizer, activation quantizer)
    "w8a8": (0.75, int8_rows, int8_rows),
    "nvfp4": (None, nvfp4, nvfp4),
    "nvfp4_sq": (0.5, nvfp4, nvfp4),
    "mxfp4": (None, mxfp4, mxfp4),
    "w4a8": (0.75, nvfp4, int8_rows),
    "w4a8_mx": (0.75, mxfp4, mxfp8),
    "kq_a8g32": (None, None, int8_groups(32)),           # weights None: the file's own values, unquantized further
    "kq_a8g256": (None, None, int8_groups(256)),
    "kq_a8tok": (None, None, int8_rows),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("model_dir")
    ap.add_argument("jevbench")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--schemes", default="w8a8,nvfp4,nvfp4_sq,mxfp4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer(a.model_dir)
    ref = REF.Reference(gguf_weights.load(a.gguf), device=a.device)
    ref.set_letters(tok.letters)
    G = groups(ref.w)
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
    REF.linear = exact
    print(f"calibrated {len(amax)} projection groups", flush=True)

    def make(scheme):
        alpha, wq, aq = SCHEMES[scheme.split("+")[0]]
        if wq is None:                                 # the file's weights as they are: quantize the inputs only
            return lambda x, q8: exact(aq(x), q8)
        cache = {}

        def prepared(key, mats):
            if key not in cache:
                Wf = torch.cat([m.dequant(torch.float32) for m in mats]).to(a.device)
                if alpha is None:
                    s = torch.ones(Wf.shape[1], device=a.device)
                else:
                    s = (amax[key].clamp_min(1e-5) ** alpha /
                         Wf.abs().amax(0).clamp_min(1e-5) ** (1 - alpha)).clamp(1e-3, 1e3)
                cache[key] = (s, {id(m): wq(m.dequant(torch.float32).to(a.device) * s) for m in mats})
            return cache[key]

        def linear(x, q8):
            key, mats = G[id(q8)]
            s, per = prepared(key, mats)
            return aq(x / s) @ per[id(q8)].T
        return linear

    import suites
    items = suites.jevbench(a.jevbench)
    step = max(1, len(items) // a.n)
    picks = [("readme3",) + suites.fixed()["readme3"], ("startlux3",) + suites.fixed()["startlux3"]]
    picks += [(i, s, q) for i, _, s, q, _ in items[::step][:a.n]]
    schemes = [x for x in a.schemes.split(",") if x]
    stats = {sch: [0, 0, 0.0, 0.0] for sch in schemes}           # same, total, max |dp|, sum |dp|
    for name, state, questions in picks:
        recs = [J.from_systemone(state, q) for q in questions.values()]
        rows = rows_for(tok, recs, list(questions))
        REF.linear = exact
        want = ref.letter_logits(rows)
        line = f"{str(name)[:22]:22s}"
        for sch in schemes:
            REF.linear = make(sch)
            REF.Reference.ATTN_QK = qk8 if "+qk8" in sch else None
            REF.Reference.ATTN_PV = pv8 if "+pv8" in sch else None
            got = ref.letter_logits(rows)
            REF.Reference.ATTN_QK = REF.Reference.ATTN_PV = None
            for r, w_, g_ in zip(rows, want, got):
                n, T = r.n_options, tok.temperature[r.kind]
                pw, pg = softmax_t(w_[:n].tolist(), T), softmax_t(g_[:n].tolist(), T)
                d = max(abs(x - y) for x, y in zip(pw, pg))
                ok = max(range(n), key=pw.__getitem__) == max(range(n), key=pg.__getitem__)
                st = stats[sch]
                st[0] += ok
                st[1] += 1
                st[2] = max(st[2], d)
                st[3] += d
            line += f"  {sch} |dp| {d:.4f}{'' if ok else ' DIFF'}"
        REF.linear = exact
        print(line, flush=True)
    for sch, (ok, tot, mx, sm) in stats.items():
        print(f"{sch:16s}: same decision {ok}/{tot}, max |dp| {mx:.4f}, mean |dp| {sm / tot:.4f}")


if __name__ == "__main__":
    main()
