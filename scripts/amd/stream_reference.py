"""Independent FP32 oracle on one GPU, with only one layer resident at a time.

Uses the original plain-PyTorch model definition. Quantized weights stay on CPU;
per-layer floating weights are cached only while that layer's independent prompts run.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from startlux_prism import reference as R, gguf_weights as G, jevfmt as J
from startlux_prism.prompt import Tokenizer, rows_for
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cpu-anchor", help="an existing CPU oracle JSONL to cross-check the streamed reference")
    p.add_argument(
        "--activation-int8", action="store_true", help="diagnose per-256 INT8 activation error with otherwise FP32 math"
    )
    p.add_argument("--input-records", help="use specified frozen prompts instead of generating new prompts")
    a = p.parse_args()
    assert torch.version.hip and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    w = G.load(a.gguf)
    ref = R.Reference(w, device="cpu")
    tok = Tokenizer(a.model_dir)
    ref.set_letters(tok.letters)
    letters = ref.letter_rows.cuda()
    temps = [
        (
            "Refund requested after a duplicate payment.",
            {
                "type": "choice",
                "instructions": "Choose the intent.",
                "criteria": {
                    "billing": "Payment or refund",
                    "technical": "Software malfunction",
                    "other": "Other request",
                },
            },
        ),
        (
            "The account cannot log in and shows a password error.",
            {
                "type": "choice",
                "instructions": "Choose the intent.",
                "criteria": {
                    "billing": "Payment or refund",
                    "technical": "Software malfunction",
                    "other": "Other request",
                },
            },
        ),
        (
            "All employees cannot access the service before a customer meeting.",
            {"type": "noul", "instructions": "Should support handle this urgently?"},
        ),
        (
            "A customer asks whether the product has a blue theme.",
            {"type": "noul", "instructions": "Is this a report of a service outage?"},
        ),
        (
            "The battery stopped working and support has not replied.",
            {
                "type": "score",
                "instructions": "How satisfied is the customer?",
                "criteria": ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"],
            },
        ),
        (
            "The information provided does not identify any product or issue.",
            {
                "type": "choice",
                "instructions": "Which specific failure occurred?",
                "criteria": {"payment": "Duplicate payment", "login": "Password error", "delivery": "Missing delivery"},
            },
        ),
    ]
    records = []
    anchors = {}
    if a.cpu_anchor:
        for line in Path(a.cpu_anchor).read_text().splitlines():
            r = json.loads(line)
            anchors[r["name"]] = r["logits"]
            r.pop("logits")
            records.append(r)
    if not a.cpu_anchor:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
        import suites

        state, qs = suites.fixed()["yesno"]
        rows = rows_for(tok, [J.from_systemone(state, q) for q in qs.values()], list(qs))
        records.append(
            dict(
                name="yesno",
                state=state,
                questions=qs,
                rows=[r.ids for r in rows],
                kinds=[r.kind for r in rows],
                n_options=[r.n_options for r in rows],
            )
        )
    for i in range(a.count):
        state, q = temps[i % len(temps)]
        state = f"Ticket {i + 1000}. " + state
        row = rows_for(tok, [J.from_systemone(state, q, allow_none=q["type"] == "choice")], ["decision"])[0]
        records.append(
            dict(
                name=f"synthetic-{i:04d}",
                state=state,
                questions={"decision": q},
                rows=[row.ids],
                kinds=[row.kind],
                n_options=[row.n_options],
            )
        )
    if a.input_records:
        records = [
            {k: v for k, v in json.loads(line).items() if k != "logits"}
            for line in Path(a.input_records).read_text().splitlines()
        ]
    if any(any(len(record[key]) != 1 for key in ("questions", "rows", "kinds", "n_options")) for record in records):
        raise ValueError("the streamed oracle expects one prompt per input record")
    cache = {}

    def linear(x, q):
        if id(q) not in cache:
            cache[id(q)] = q.dequant(torch.float32)
        if a.activation_int8:
            b = x.reshape(*x.shape[:-1], -1, 256)
            scale = b.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
            x = (torch.round(b / scale).clamp(-127, 127) * scale).reshape(x.shape)
        return x @ cache[id(q)].T

    R.linear = linear
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), out.open("w") as f:
        for first in range(0, len(records), a.batch):
            batch = records[first : first + a.batch]
            t = time.perf_counter()
            hs = [w.embed.dequant(torch.float32, torch.tensor(r["rows"][0])).cuda() for r in batch]
            angles = [
                R.rope_cos_sin(torch.arange(len(h), device="cuda"), w.config.rope_dim, w.config.rope_theta, "cuda")
                for h in hs
            ]
            for i, l in enumerate(w.layers):
                cache.clear()
                lw = G.Layer(
                    **{
                        x.name: (
                            getattr(l, x.name).to("cuda")
                            if isinstance(getattr(l, x.name), (G.Q8, G.KQuant, torch.Tensor))
                            else getattr(l, x.name)
                        )
                        for x in dataclasses.fields(G.Layer)
                    }
                )
                for j, h in enumerate(hs):
                    x = R.rms_norm(h, lw.attn_norm, w.config.eps)
                    h = h + (ref._attention(x, lw, *angles[j]) if w.config.is_attention(i) else ref._gdn(x, lw))
                    x = R.rms_norm(h, lw.post_norm, w.config.eps)
                    hs[j] = h + linear(
                        torch.nn.functional.silu(linear(x, lw.ffn_gate)) * linear(x, lw.ffn_up), lw.ffn_down
                    )
                if i % 16 == 15:
                    print("oracle batch", first, "layer", i + 1, flush=True)
                del lw
            cache.clear()
            norm = w.out_norm.cuda()
            for record, h in zip(batch, hs):
                z = R.rms_norm(h[-1], norm, w.config.eps) @ letters.T
                record["logits"] = [z.cpu().tolist()]
                if record["name"] in anchors and not a.activation_int8:
                    error = (z.cpu() - torch.tensor(anchors[record["name"]][0])).abs().max().item()
                    print("CPU/streamed oracle anchor delta", record["name"], error, flush=True)
                    assert error < 1e-3, ("independent reference implementations disagree", error)
                f.write(json.dumps(record) + "\n")
            f.flush()
            torch.cuda.synchronize()
            print(
                "PASS oracle batch",
                first,
                len(batch),
                "seconds",
                time.perf_counter() - t,
                "peak",
                torch.cuda.max_memory_allocated(),
                flush=True,
            )
    print("PASS independent streamed reference", len(records), "activation_int8", a.activation_int8, flush=True)


if __name__ == "__main__":
    main()
