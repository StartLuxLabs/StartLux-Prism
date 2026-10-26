"""Single GPU real-model decode, batch isolation, and teacher-forced prefill checks."""

import os
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import torch
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.engine.decode import Decoder, Sampling
from startlux_prism.think import Framing, RELEASE_PREFIX
from test_decode import Seqs, prefill_logits
import suites


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=("int8", "w4a8", "w4a16"), default="w4a8")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--graphs", action="store_true")
    a = p.parse_args()
    P.on()
    eng = Engine(
        a.gguf,
        a.model_dir,
        Options(precision=a.precision, max_tokens=2048, graphs=a.graphs, prefix_cache=0, think_seqs=2, think_kv_rows=8192),
    )
    dec = Decoder(eng, a.gguf, max_seqs=2, graphs=a.graphs, sampling=Sampling(greedy=True))
    framing = Framing(a.model_dir)
    ps = []
    for name in ["yesno", "review1"]:
        state, qs = suites.fixed()[name]
        q = next(iter(qs.values()))
        rec = J.from_systemone(state, q, allow_none=q["type"] == "choice")
        ids = eng.tok.encode(framing.prompt(rec) + RELEASE_PREFIX)
        ps.append((name, ids))
    result = dict(precision=a.precision, shared_gpu=os.getenv("PRISM_SHARED_GPU") == "1", sequences=[], graphs=a.graphs)
    start = time.perf_counter()
    s = Seqs(eng, dec, ps, 4096)
    s.prefill(s.s)
    s.run(s.s, a.tokens)
    print("greedy decode", a.tokens, "tokens per sequence", time.perf_counter() - start, flush=True)
    for row in s.s:
        ids = row["ids"] + row["gen"][:-1]
        positions = list(range(len(row["ids"]) - 1, len(row["ids"]) - 1 + a.tokens))
        ref, letters = prefill_logits(eng, dec, ids, positions)
        got = torch.stack(row["logits"])
        delta = (got - ref).abs()
        agree = int((got.argmax(1).cpu() == ref.argmax(1).cpu()).sum())
        print(
            "teacher forcing",
            row["name"],
            "max",
            delta.max().item(),
            "mean",
            delta.mean().item(),
            "agree",
            agree,
            a.tokens,
            flush=True,
        )
        assert torch.isfinite(got).all()
        assert agree >= a.tokens - 1, "teacher-forced argmax mismatch"
        one = Seqs(eng, dec, [(row["name"], row["ids"])], 4096, seeds=[row["seed"]])
        one.prefill(one.s)
        one.run(one.s, a.tokens)
        assert one.s[0]["gen"] == row["gen"], "batch isolation token mismatch"
        assert all(torch.equal(u, v) for u, v in zip(one.s[0]["logits"], row["logits"])), (
            "batch isolation logit mismatch"
        )
        result["sequences"].append(
            dict(
                name=row["name"],
                generated=row["gen"],
                text=eng.tok.tok.decode(row["gen"]),
                max_logit_delta=float(delta.max()),
                mean_logit_delta=float(delta.mean()),
                top1_agree=agree,
            )
        )
    rel = Sampling()
    combined = Seqs(eng, dec, ps, 4096, seeds=[123, 456], sampling=rel)
    combined.prefill(combined.s)
    combined.run(combined.s, a.tokens)
    for i, row in enumerate(combined.s):
        alone = Seqs(eng, dec, [(row["name"], row["ids"])], 4096, seeds=[row["seed"]], sampling=rel)
        alone.prefill(alone.s)
        alone.run(alone.s, a.tokens)
        assert alone.s[0]["gen"] == row["gen"], "sampled tokens depend on batch"
    result.update(
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        free_total=torch.cuda.mem_get_info(),
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print("PASS decode teacher forcing, exact batch isolation and seeded sampling", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
