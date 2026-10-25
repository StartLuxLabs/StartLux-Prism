"""Compare every candidate probability to a frozen independent FP32 model oracle."""

import os
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options
from startlux_prism.prompt import rows_for
from startlux_prism.decide import softmax_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-samples", type=int, default=500)
    p.add_argument("--precision", choices=("w4a8", "w4a16"), default="w4a8")
    a = p.parse_args()
    P.on()
    data = [json.loads(x) for x in Path(a.oracle).read_text().splitlines()]
    assert len(data) >= a.min_samples, (len(data), a.min_samples)
    for record in data:
        counts = [len(record[key]) for key in ("questions", "rows", "logits", "n_options")]
        if not counts[0] or len(set(counts)) != 1:
            raise ValueError("oracle question/row/logit/option counts must match")
        if any(not 0 < n <= len(z) for n, z in zip(record["n_options"], record["logits"], strict=True)):
            raise ValueError("oracle option count exceeds available logits")
    e = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=2048, graphs=False, prefix_cache=0))
    errors = []
    cases = []
    agree = 0
    clear = 0
    clear_agree = 0
    t = time.perf_counter()
    for i, d in enumerate(data):
        qs = d["questions"]
        rows = rows_for(
            e.tok, [J.from_systemone(d["state"], q, allow_none=q["type"] == "choice") for q in qs.values()], list(qs)
        )
        assert [r.ids for r in rows] == d["rows"], "oracle/candidate prompt tokens differ"
        got = e.letter_logits(rows)
        for row, z, ref, n in zip(rows, got, d["logits"], d["n_options"], strict=True):
            temp = e.tok.temperature[row.kind]
            p0 = softmax_t(ref[:n], temp)
            p1 = softmax_t(z[:n].tolist(), temp)
            err = [abs(x - y) for x, y in zip(p0, p1)]
            same = int(np.argmax(p0)) == int(np.argmax(p1))
            agree += same
            gap = sorted(p0)[-1] - sorted(p0)[-2] if n >= 2 else 1.0
            if gap > 0.02:
                clear += 1
                clear_agree += same
            errors.extend(err)
            cases.append(
                dict(
                    name=d["name"],
                    kind=row.kind,
                    reference_p=p0,
                    candidate_p=p1,
                    max_error=max(err),
                    same_decision=same,
                    reference_margin=gap,
                )
            )
        if i % 25 == 24:
            print("model reference comparison", i + 1, "/", len(data), flush=True)
    result = dict(
        precision=a.precision,
        q_components=os.getenv("STARTLUX_PRISM_AMD_Q_COMPONENTS", "all") if a.precision=="w4a16" else None,
        shared_gpu=os.getenv("PRISM_SHARED_GPU") == "1",
        samples=len(cases),
        probability_values=len(errors),
        probability_mae=float(np.mean(errors)),
        probability_p99=float(np.percentile(errors, 99)),
        probability_max=float(np.max(errors)),
        decision_agreement=agree / len(cases),
        clear_margin_samples=clear,
        clear_margin_agreement=clear_agree / max(clear, 1),
        elapsed_s=time.perf_counter() - t,
        cases=cases,
    )
    result["gate_pass"] = (
        result["probability_mae"] <= 1e-3
        and result["probability_p99"] <= 1e-2
        and result["clear_margin_agreement"] >= 0.995
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        "PASS" if result["gate_pass"] else "FAIL",
        "frozen probability gate",
        json.dumps({k: v for k, v in result.items() if k != "cases"}),
        flush=True,
    )
    sys.exit(0 if result["gate_pass"] else 1)


if __name__ == "__main__":
    main()
