"""Two engine configurations against the fp32 reference on every public JevBench item: how often each makes the
reference's decision, and how far its probabilities are from the reference's.  For a change that alters numerics
(a lower-precision path), before it becomes a default.

    python tests/accuracy_ab.py --gguf MODEL.gguf --model-dir DIR --calib FILE --jevbench PUBLIC_DIR \\
        --a "" --b attn_qk8=1 [--n 231]

--a / --b: comma-separated Options fields (engine/model.py), as in bench/ab.py.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from flash_jev import gguf_weights, jevfmt as J  # noqa: E402
from flash_jev.decide import softmax_t  # noqa: E402
from flash_jev.engine.model import Engine, group  # noqa: E402
from flash_jev.prompt import rows_for  # noqa: E402
from flash_jev.reference import Reference  # noqa: E402
from ab import options  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--n", type=int, default=231)
    a = ap.parse_args()
    base = dict(precision="int8", max_tokens=8192, calib=a.calib, prefix_cache=0)
    engines = {}
    for name, spec in (("A", a.a), ("B", a.b)):
        e = Engine(a.gguf, a.model_dir, options(spec, base))
        e.warm_up()
        engines[name] = e
    tok = engines["A"].tok
    ref = Reference(gguf_weights.load(a.gguf), chunked=True)
    ref.set_letters(tok.letters)
    items = suites.jevbench(a.jevbench)[:a.n]
    st = {k: {"same": 0, "dp": [], "worst": (0.0, "")} for k in ("A", "B")}
    ab_dp, n = [], 0
    for iid, tier, state, questions, _ in items:
        rows = rows_for(tok, [J.from_systemone(state, q) for q in questions.values()], list(questions))
        want = ref.letter_logits(rows)
        got = {k: e.run([group(rows)]) for k, e in engines.items()}
        for i, r in enumerate(rows):
            T = tok.temperature[r.kind]
            pw = softmax_t(want[i][:r.n_options].tolist(), T)
            top = max(range(r.n_options), key=pw.__getitem__)
            ps = {}
            for k in ("A", "B"):
                p = ps[k] = softmax_t(got[k][i][:r.n_options].tolist(), T)
                d = max(abs(x - y) for x, y in zip(p, pw))
                s = st[k]
                s["same"] += max(range(r.n_options), key=p.__getitem__) == top
                s["dp"].append(d)
                if d > s["worst"][0]:
                    s["worst"] = (d, iid)
            ab_dp.append(max(abs(x - y) for x, y in zip(ps["A"], ps["B"])))
            n += 1
    print(f"A: {a.a or 'defaults'}   B: {a.b or 'defaults'}   {n} questions against the fp32 reference")
    for k in ("A", "B"):
        s = st[k]
        dp = sorted(s["dp"])
        print(f"{k}: same decision {s['same']}/{n}, |dp| mean {statistics.mean(dp):.4f}, p95 {dp[int(0.95 * len(dp))]:.4f}, "
              f"max {s['worst'][0]:.4f} ({s['worst'][1]})")
    print(f"A vs B: |dp| mean {statistics.mean(ab_dp):.4f}, max {max(ab_dp):.4f}")


if __name__ == "__main__":
    main()
