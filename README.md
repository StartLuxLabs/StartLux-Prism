<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.png">
    <img src="docs/images/logo.png" width="420" alt="StartLux Prism">
  </picture>
</p>

**A fast CUDA inference engine for StartLux-Decision models.** It serves StartLux-Decision v1 (4B, 9B, 27B) and v1.1
AutoThink (4B, 9B, 27B) straight from their GGUF files, behind the TypeSafe `/v1/systemone` API, on NVIDIA's RTX 40
and 50 series and its A and H series (tested on the RTX 4090, RTX 5090, A800 and H200), on one GPU or several. Like a
prism that splits light into its spectrum, it turns each question into a calibrated probability for every option.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v11-hero-dark.svg">
  <img src="docs/images/v11-hero.svg" width="840" alt="StartLux-Decision v1.1 27B AutoThink on one H200, StartLux-Prism int8 against the official engine in FP8. Mean ms per request, one at a time: one question of about 256 tokens 19.0 against 53.1; about 8,192 tokens 439.8 against 531.2; the README's three questions 17.4 against 92.9; the same with thoughts 785.1 against 1,296.0. JevBench requests per second: read-outs from 1 client 24.2 against 12.1; from 64 clients 30.7 against 26.3; with thoughts from 64 clients 7.6 against 6.5.">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v1-hero-dark.svg">
  <img src="docs/images/v1-hero.svg" width="840" alt="StartLux-Decision v1 27B on one H200, StartLux-Prism int8 against the official engine in FP8. Mean ms per request, one at a time: one question of about 256 tokens 18.9 against 54.5; about 8,192 tokens 438.5 against 539.1; the README's three questions 17.2 against 96.4. JevBench requests per second: from 1 client 24.4 against 12.0; from 64 clients 30.9 against 25.4.">
</picture>

## Why StartLux-Prism

- **Faster than the official engine.** The 27B on one H200, against StartLux-Decision's own engine on vLLM 0.31 in FP8:
  on v1.1, 2.8x faster on a short question, 5.3x on a three-question request, 1.17x the throughput from 64 clients; on
  v1, 2.9x, 5.6x and 1.22x. On eight H200s: 238 read-outs per second against 138 for the official engine in BF16. More
  results, from many clients and under stress: [docs/benchmarks.md](docs/benchmarks.md).
- **The same answers.** Prompts, read-outs, thoughts and decision rules are the model's own; only the forward pass is
  new, so the scores are the official engine's ([below](#the-official-engines-accuracy-in-int8-and-in-4-bits)). An
  answer is bit-identical whatever it is batched with, on any number of GPUs, from the cache or not.
- **All of v1.1.** AutoThink thoughts generated on the GPU (95 tokens/s for one sequence, 3,832 for 64 at once), from
  the Q8_0 and the Q4_K_M files; `none_probability` and `allow_none`, up to 8 images per request.
- **Many GPU architectures.** The A series (Ampere: A800), the RTX 40 series (Ada: RTX 4090), the H series (Hopper:
  H200) and the RTX 50 series (Blackwell: RTX 5090), with drivers for CUDA 12 or 13. In 4 bits the 27B runs on one
  32 GB RTX 5090: v1.1's Q4_K_M file, thinking, scores there what the official engine scores on an H200.
- **One GPU or several.** Data parallelism for throughput (four H200s give 3.92x one), tensor parallelism (`--tp 2` or
  `4`) for latency.
- **The released files as they are.** Q8_0 runs as int8 (W8A8) and Q4_K_M on its own 4-bit weights (W4A8), read from
  the GGUF at start-up with no conversion step.

### The official engine's accuracy, in int8 and in 4 bits

StartLux-Decision v1.1 27B AutoThink with thoughts: StartLux-Prism int8 from the Q8_0 file, and W4A8 from the Q4_K_M
file on one RTX 5090, against the official engine on the BF16 release.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v11-accuracy-dark.svg">
  <img src="docs/images/v11-accuracy.svg" width="840" alt="StartLux-Decision v1.1 27B AutoThink with thoughts. StartLux-Decision v1's seven evaluation suites, average accuracy: StartLux-Prism int8 92.94, StartLux-Prism W4A8 93.07, official engine BF16 92.97, FP8 92.83. JevBench public, items right of 231: 224, 226, 226, 225.">
</picture>

## Install

Needs Linux, an NVIDIA GPU from Ampere on (sm_80 and later: tuned on the RTX 5090 and Hopper, also run on an RTX 4090
and an A800), Python 3.12 with its headers and venv (on Ubuntu `python3.12-dev python3.12-venv`) and `g++`: the CUDA
kernels are compiled on the first start, for the GPU they run on. With an NVIDIA driver for CUDA 13 (580 or later):

```bash
git clone https://github.com/StartLuxLabs/StartLux-Prism && cd StartLux-Prism
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # PyTorch 2.14.1 for CUDA 13, Triton, nvcc 13.0
source scripts/cuda_env.sh          # nvcc from pip as CUDA_HOME (in every new shell)
```

With an older driver (550 or 570: CUDA 12), in place of the last two lines: PyTorch for CUDA 12.6 and a CUDA 12
toolkit's nvcc for the kernels:

```bash
pip install -r requirements-cu12.txt
export CUDA_HOME=/usr/local/cuda    # a CUDA 12 toolkit (in every new shell)
```

A model is its GGUF file and its release's folder files (`config.json`, `decision_config.json`, the tokenizer and the
chat template) in one directory. The v1 GGUF repositories carry them; for v1.1 they come from the full release:

```bash
M=~/models/StartLux-Decision-v1.1-27B-AutoThink-GGUF
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --local-dir $M
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink --exclude "*.safetensors" --local-dir $M   # tokenizer and config
```

Add the `mmproj-*.gguf` file to read images. The weights are CC BY-NC 4.0 (non-commercial).

## Quick start

```bash
G=$M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf
python -m startlux_prism.server --gguf $G --model-dir $M --port 8092    # ready when it prints "Ready"
```

It prints each loading step as it goes, then `Ready`, its endpoints and a request to try
([what a start prints](docs/serving.md#what-a-start-prints)).

The first start measures the int8 activation ranges once and keeps them in `~/.cache/startlux-prism/calib`.

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
             "urgent": {"type": "noul", "noul": 0.749}},
 "usage": {"input_tokens": 191, "output_tokens": 0}, ...}
```

Both first read-outs reached the gate (0.6), so the model answered without a thought. With `"gate": 0.9` (the setting
of the published numbers) `urgent` (0.749) is below it, and the model thinks for 29 tokens before answering 0.855.
`none_probability` is the chance that no listed option fits. `choice` is null only when "None of the above" is the most
likely option; we suggest also treating 0.2 or more as "no listed option fits" (it catches about two thirds of the
questions whose right answer was removed, for about 2% of ordinary ones).

On several GPUs, with the same flags:

```bash
python -m startlux_prism.dp --gpus 0,1,2,3,4,5,6,7 --gguf $G --model-dir $M --port 8092   # data parallel: most throughput
python -m startlux_prism.server --tp 2 --gpus 0,1 --gguf $G --model-dir $M --port 8092    # tensor parallel: less latency
python -m startlux_prism.dp --gpus 0,1,2,3 --tp 2 --gguf $G --model-dir $M --port 8092    # both
```

To serve other machines (`--host 0.0.0.0`; the server has no authentication of its own), keep the compiled kernels
and the calibration across restarts, or run it under systemd: [docs/serving.md](docs/serving.md#running-as-a-service).

## Serving

Endpoints: `POST /v1/systemone` (also `/v1/decisions`), `GET /v1/models`, `GET /health`, `GET /stats`.

| request field | default | |
|---|---|---|
| `state`, `questions` | | the evidence (text or JSON) and typed questions: `choice`, `noul` (yes/no), `score` |
| `images` | | up to 8, base64 or data URIs |
| `think_mode` | `auto` | `auto` thinks when the first read-out's top probability is below `gate`; `never`; `always` |
| `gate`, `mix`, `think_budget` | 0.6, 0.75, 4,096 | the threshold to think (the published numbers used 0.9), the weight of the read-out after the thought, the longest thought |
| `none_check`, `allow_none` | on, on (v1.1) | "None of the above" as an extra option, reported as `none_probability`; a null choice when it is the most likely option |
| `return_details`, `return_thought` | off | each answer's read-outs and timings; the thought's text |

| server flag | default | |
|---|---|---|
| `--precision` | `auto` | `int8` for Q8_0 files, `w4a8` for Q4_K_M, `bf16` as the reference |
| `--gpus`, `--tp` | | the GPUs; GPUs per model (2 or 4 for the 27B) |
| `--max-tokens` | from free memory | the largest batch (131,072 tokens for the 27B on an H200); longer requests get 422 |
| `--prefix-cache-tokens` | 65,536 | KV rows kept for evidence that comes again |
| `--think-seqs` | as many as fit | thoughts generated at once (64 for the 27B on an H200, 16 for the 9B in int8 on an RTX 5090) |
| `--gate`, `--none-check`, `--allow-none` | 0.6, on, on (v1.1) | the defaults of the request fields |

Every field and flag, the model folders and the GPUs: [docs/serving.md](docs/serving.md).

## Performance

StartLux-Decision v1.1 27B AutoThink, one H200 per server, the same requests:

| | official, BF16 | official, FP8 | **StartLux-Prism int8** |
|---|---|---|---|
| one short question, ms | 54.2 | 53.1 | **19.0** |
| three questions, ms (with thoughts) | 93.1 (1,775) | 92.9 (1,296) | **17.4 (785)** |
| one question of 8,192 tokens, ms | 728.8 | 531.2 | **439.8** |
| JevBench read-outs, requests/s from 64 clients | 18.3 | 26.3 | **30.7** |
| JevBench with thoughts, requests/s from 64 clients | 5.89 | 6.52 | **7.61** |

StartLux-Decision v1 27B, one H200 per server, the same requests (v1 answers in one pass):

| | official, BF16 | official, FP8 | **StartLux-Prism int8** |
|---|---|---|---|
| one short question, ms | 54.7 | 54.5 | **18.9** |
| three questions, ms | 95.4 | 96.4 | **17.2** |
| one question of 8,192 tokens, ms | 727.2 | 539.1 | **438.5** |
| JevBench read-outs, requests/s from 64 clients | 18.1 | 25.4 | **30.9** |

Under load both engines are bound by the tensor cores, which run int8 and FP8 at the same rate, so the lead narrows to
17% on v1.1 and 22% on v1. On several H200s, v1.1's read-outs per second: one GPU 30.9, four 121.1, eight 237.8 (the
official engine on eight: 137.5).

Against every other engine compared on v1 (SGLang, vLLM, llama.cpp, StartLux-Decision's PyTorch server, decider and
JevK5), StartLux-Prism has the lowest latency (1.6-2.3x below the next) and 1.6-2.4x their throughput.

Methods, every engine, table and chart: [docs/benchmarks.md](docs/benchmarks.md).

## Accuracy

The speed does not cost accuracy. On StartLux-Decision v1's own evaluation (12,351 decisions), int8 averages 91.14
against the reference's 91.15 on the 4B and 90.80 against 91.06 on the 9B. About 1% of the decisions change, all near
ties, and calibration does not move. v1.1 27B with thoughts: 92.94 in int8 and 93.07 in W4A8, against 92.97 on the
official BF16 server. Every suite: [docs/benchmarks.md](docs/benchmarks.md#accuracy-startlux-decision-v1s-evaluation).

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

StartLux-Prism's own code: to be decided by the owner. Vendored from StartLux-Decision (Apache-2.0,
`LICENSES/StartLux-Decision-Apache-2.0.txt`): `startlux_prism/jevfmt.py`, `startlux_prism/calibration.py`, and `startlux_prism/loops.py`
from v1.1's package. Model weights are not part of this repository; StartLux-Decision's weights are CC BY-NC 4.0
(non-commercial).
