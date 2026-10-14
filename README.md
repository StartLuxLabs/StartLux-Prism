# flash-jev

**A CUDA inference engine for JEV decision models.** It serves StartLux-Decision v1 (4B, 9B, 27B) and v1.1 AutoThink
(27B) straight from their GGUF files, behind the TypeSafe `/v1/systemone` API, on the RTX 5090 and the H200, on one GPU
or several.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v11-hero-dark.svg">
  <img src="docs/images/v11-hero.svg" width="840" alt="StartLux-Decision v1.1 27B AutoThink on one H200, flash-jev int8 against the official engine in FP8. Mean ms per request, one at a time: one question of about 256 tokens 19.0 against 53.1; about 8,192 tokens 439.8 against 531.2; the README's three questions 17.4 against 92.9; the same with thoughts 785.1 against 1,296.0. JevBench requests per second: read-outs from 1 client 24.2 against 12.1; from 64 clients 30.7 against 26.3; with thoughts from 64 clients 7.6 against 6.5.">
</picture>

## Why flash-jev

- **Faster than the official engine.** On one H200, against StartLux-Decision v1.1's own engine on vLLM 0.31 in FP8:
  2.8x faster on a short question, 5.3x on a three-question request, 1.17x the throughput from 64 clients. On eight
  H200s: 238 read-outs per second against 138 for its BF16 release.
- **The same answers.** Prompts, read-outs, thoughts and decision rules are the model's own; only the forward pass is
  new. v1.1's evaluation suites score 92.94 on flash-jev int8, 92.97 on the official BF16 server and 92.83 in FP8. An
  answer is bit-identical whatever it is batched with, on any number of GPUs, from the cache or not.
- **All of v1.1.** AutoThink thoughts generated on the GPU (95 tokens/s for one sequence, 3,832 for 64 at once),
  `none_probability` and `allow_none`, up to 8 images per request.
- **One GPU or several.** Data parallelism for throughput (four H200s give 3.92x one), tensor parallelism (`--tp 2` or
  `4`) for latency.
- **The released files as they are.** Q8_0 runs as int8 (W8A8) and Q4_K_M on its own 4-bit weights (W4A8), so the 27B
  fits a 32 GB RTX 5090.

## Quick start

Needs Linux, an RTX 5090 (sm_120) or a Hopper GPU (sm_90a), Python 3.12 with PyTorch for CUDA 13, Triton and a CUDA
toolkit. The CUDA extension builds on first use.

```bash
M=~/models/StartLux-Decision-v1.1-27B-AutoThink-GGUF
G=$M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf
python -m flash_jev.server --gguf $G --model-dir $M --port 8092    # ready when it prints "Ready"
```

The first start measures the int8 activation ranges once and keeps them in `~/.cache/flash-jev/calib`.

```bash
curl -s localhost:8092/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"}}}'
```

```text
{"answers": {"team": {"type": "choice", "choice": "billing", "confidence": 0.993, "probabilities": {...},
                      "none_probability": 0.002},
             "urgent": {"type": "noul", "noul": 0.855}},
 "usage": {"input_tokens": 191, "output_tokens": 29}, ...}
```

`urgent` was unsure on its first read-out (below `gate`, 0.9), so the model thought for 29 tokens before answering.

On several GPUs, with the same flags:

```bash
python -m flash_jev.dp --gpus 0,1,2,3,4,5,6,7 --gguf $G --model-dir $M --port 8092   # data parallel: most throughput
python -m flash_jev.server --tp 2 --gpus 0,1 --gguf $G --model-dir $M --port 8092    # tensor parallel: less latency
python -m flash_jev.dp --gpus 0,1,2,3 --tp 2 --gguf $G --model-dir $M --port 8092    # both
```

## Serving

Endpoints: `POST /v1/systemone` (also `/v1/decisions`), `GET /v1/models`, `GET /health`, `GET /stats`.

| request field | default | |
|---|---|---|
| `state`, `questions` | | the evidence (text or JSON) and typed questions: `choice`, `noul` (yes/no), `score` |
| `images` | | up to 8, base64 or data URIs |
| `think_mode` | `auto` | `auto` thinks when the first read-out's top probability is below `gate`; `never`; `always` |
| `gate`, `mix`, `think_budget` | 0.9, 0.75, 4,096 | the threshold to think, the weight of the read-out after the thought, the longest thought |
| `none_check`, `allow_none` | on (v1.1), off | "None of the above" as an extra option; a null choice when it wins |
| `return_details`, `return_thought` | off | each answer's read-outs and timings; the thought's text |

| server flag | default | |
|---|---|---|
| `--precision` | `auto` | `int8` for Q8_0 files, `w4a8` for Q4_K_M, `bf16` as the reference |
| `--gpus`, `--tp` | | the GPUs; GPUs per model (2 or 4 for the 27B) |
| `--max-tokens` | from free memory | the largest batch (131,072 tokens for the 27B on an H200); longer requests get 422 |
| `--prefix-cache-tokens` | 65,536 | KV rows kept for evidence that comes again |
| `--think-seqs` | 64 | thoughts generated at once |
| `--none-check`, `--allow-none` | | the defaults of the request fields |

Every field and flag, the model folders and the GPUs: [docs/serving.md](docs/serving.md).

## Performance

StartLux-Decision v1.1 27B AutoThink, one H200 per server, the same requests:

| | official, BF16 | official, FP8 | **flash-jev int8** |
|---|---|---|---|
| one short question, ms | 54.2 | 53.1 | **19.0** |
| three questions, ms (with thoughts) | 93.1 (1,775) | 92.9 (1,296) | **17.4 (785)** |
| one question of 8,192 tokens, ms | 728.8 | 531.2 | **439.8** |
| JevBench read-outs, requests/s from 64 clients | 18.3 | 26.3 | **30.7** |
| JevBench with thoughts, requests/s from 64 clients | 5.89 | 6.52 | **7.61** |

Under load both engines are bound by the tensor cores, which run int8 and FP8 at the same rate, so the lead narrows to
17%. On several H200s, read-outs per second: one GPU 30.9, four 121.1, eight 237.8 (the official engine on eight:
137.5). On StartLux-Decision v1, flash-jev has the lowest latency of every engine compared (1.6-2.3x below the next)
and 1.6-2.4x their throughput.

Methods, every engine, table and chart: [docs/benchmarks.md](docs/benchmarks.md).

## Accuracy

The speed does not cost accuracy. On StartLux-Decision v1's own evaluation (12,351 decisions), int8 averages 91.14
against the reference's 91.15 on the 4B and 90.80 against 91.06 on the 9B. About 1% of the decisions change, all near
ties, and calibration does not move. v1.1 27B with thoughts: 92.94, against 92.97 on the official BF16 server. Every
suite: [docs/benchmarks.md](docs/benchmarks.md#accuracy-startlux-decision-v1s-evaluation).

## How it works

- **One forward pass per batch, as CUDA graphs.** A request's questions branch off their shared evidence inside the
  graph, so the evidence is computed once.
- **A prefix cache** keeps evidence across requests: a follow-up question takes 5.4 ms instead of 54.
- **int8 tensor cores with exact int32 sums**, so an answer does not depend on its batch, the number of GPUs or the
  cache.
- **Kernels chosen per GPU** from measured tables. On Hopper the int8 GEMMs run on warpgroup MMA at 1,240-1,420 TOPS.
- **Thoughts decoded on the GPU** in CUDA graphs, between prefill batches, seeded per question.

More: [design and code map](docs/design.md), [optimization log](docs/optimizations.md),
[correctness and tests](docs/testing.md).

## License

flash-jev's own code: to be decided by the owner. Vendored from StartLux-Decision (Apache-2.0,
`LICENSES/StartLux-Decision-Apache-2.0.txt`): `flash_jev/jevfmt.py`, `flash_jev/calibration.py`, and `flash_jev/loops.py`
from v1.1's package. Model weights are not part of this repository; StartLux-Decision's weights are CC BY-NC 4.0
(non-commercial).
