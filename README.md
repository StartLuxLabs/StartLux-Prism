# flash-jev

**A CUDA inference engine for JEV decision models: hand-written kernels, one forward pass per batch, bit-exact
answers, on the RTX 5090 and the H200.**

JEV ("System One") models answer typed questions about a piece of evidence (a choice among options, a yes/no, a
score) with calibrated probabilities, read from the option letters' logits at the last prompt token. flash-jev serves
[StartLux-Decision v1](https://github.com/StartLuxLabs/StartLux-Decision-v1) (4B, 9B, 27B) and StartLux-Decision v1.1
AutoThink (27B), Qwen3.5 dense hybrids of Gated DeltaNet and gated attention, straight from their GGUF files behind the
TypeSafe `/v1/systemone` API. Prompts, read-outs, thoughts and decision rules are the models' own
(`flash_jev/jevfmt.py`, `flash_jev/think.py`): only the forward pass differs from the reference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v11-hero-dark.svg">
  <img src="docs/images/v11-hero.svg" width="840" alt="StartLux-Decision v1.1 27B AutoThink on one H200, flash-jev int8 against the official engine in FP8. Mean ms per request, one at a time: one question of about 256 tokens 19.0 against 53.1; about 8,192 tokens 439.8 against 531.2; the README's three questions 17.4 against 92.9; the same with thoughts 785.1 against 1,296.0. JevBench requests per second: read-outs from 1 client 24.2 against 12.1; from 64 clients 30.7 against 26.3; with thoughts from 64 clients 7.6 against 6.5.">
</picture>

## Highlights

- **StartLux-Decision v1.1, thoughts and images included.** AutoThink models think as v1.1's own package does, with the
  thoughts generated on the GPU (95 tokens/s for one sequence, 3,832 for 64 at once; every thought seeded and
  batch-invariant). `none_probability`, the request controls and up to 8 images per request work as in v1.1.
- **Faster than the model's official engine.** Against StartLux-Decision v1.1's own engine on vLLM 0.31 in FP8, one
  H200 each: 2.8x faster on a short question, 5.3x on a three-question request, 1.2x at 8,192 tokens, and 1.17x the
  throughput from 64 clients, with or without thoughts. Its evaluation suites score the same (92.94 against 92.83).
- **Lowest latency of every engine compared** (StartLux-Decision v1). JevBench one request at a time: 1.6-2.3x below
  the next engine. StartLux's own 3-question request is 3.8-4.8x faster than on the next engine (SGLang): 10.2 ms on the
  4B, 25.0 ms on the 27B.
- **1.6-2.4x the throughput under load.** JevBench from 32 clients: 85 requests/s on the 4B (next: 39.5), 53 on the 9B
  (22.6), 30.6 on the 27B on an H200 (18.8).
- **Long evidence in one pass.** 50,000 tokens in 1.96 s on the 4B and 3.69 s on the 27B (SGLang: 2.91 and 4.94 s). A
  request longer than the largest batch is refused with HTTP 422, never cut.
- **Follow-up questions from a prefix cache.** A further question about evidence already sent takes 5.4 ms instead of
  54 (SGLang 113, vLLM 43).
- **Deterministic and batch-invariant.** On the int8 and W4A8 paths, a request's probabilities are bit-identical on
  every run, whatever the request is batched with, on any of several GPUs, and whether its evidence hit the cache.
- **Accuracy kept.** On StartLux-Decision-v1's own evaluation (12,351 decisions), int8 averages 91.14 against the
  reference's 91.15 (4B) and 90.80 against 91.06 (9B).
- **4-bit files on their own weights.** Q4_K_M files run as W4A8 on int8 tensor cores, 3.9-6.9x faster than llama.cpp
  on the same file. This way the 27B fits a 32 GB RTX 5090.
- **Several GPUs, the same answers.** Data parallelism for throughput (`flash_jev.dp`, one server per GPU behind a
  router): three H200s give 2.89x the read-out throughput of one. Tensor parallelism for latency (`--tp 2` or `4`): a
  JevBench item with thoughts takes 902 ms on two H200s and 768 on four (1,209 on one), an 8,192-token prompt 301 and
  242 ms (428). Either way, every answer is the single GPU's, bit for bit.
- **Kernels per GPU.** Every projection's kernel, tile and K split comes from a table measured on the GPU model. On
  Hopper, the int8 GEMMs run on warpgroup MMA with TMA and warp specialization (1,240-1,420 TOPS).

## StartLux-Decision v1.1

The server reads a model folder's kind as v1.1's own package does (`flash_jev/modelcfg.py`) and answers as that
release does:

| folder | how it answers | run here |
|---|---|---|
| v1 (`"format": "StartLux-Decision-v1"`) | in one pass, exactly as before v1.1 support | 4B, 9B, 27B |
| v1.1 AutoThink | one pass, then a thought when the first read-out is unsure | 27B: int8 and bf16 (W4A8: one pass) |
| v1.1 Instant (prompt read bidirectionally) | not served yet: refused at start-up | |

- **Thoughts.** `think_mode` `auto` (the default) thinks when the first read-out's top probability is below `gate`
  (0.9). `never` answers in one pass, and `always` thinks first.
  - **Sampling:** as v1.1 samples a thought: temperature 0.6, top-k 20, top-p 0.95, a seed from the prompt, at most
    4,096 tokens, and v1.1's loop detector.
  - **Answer:** a second read-out of the prompt and the thought, mixed 0.25 / 0.75 with the first.
  - **Generation:** on the GPU, as decode steps recorded as CUDA graphs for up to 64 sequences, interleaved with the
    prefill batches.
- **None of the above.** With `none_check` (on by default for v1.1), every choice question gets the extra option "None
  of the above". Its probability is reported as `none_probability`, and the listed options are renormalised.
  `allow_none` also lets the choice be null when that option wins.
- **Images.** With the folder's `mmproj` vision file, a request can carry up to 8 images, placed, resized and
  positioned as v1.1 does. Thoughts read them too.
- **The same answers as v1.1's package.** Prompts match byte for byte, decisions from the same letter logits match to
  the float, and whole requests with thoughts match its pipeline ([Correctness](#correctness)).

Files run here: `StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf` (28.60 GB, SHA-256 `28757be9a8fae525...`) and
`mmproj-StartLux-Decision-v1.1-27B-AutoThink-F16.gguf` (0.93 GB, `b6ff3c746a642ac4...`), from
`startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF`.

## Performance

### StartLux-Decision v1.1 27B AutoThink against its official engine

The official engine is Startlux-decide-inference (v1.1's package, commit `b70404d`) with the flags its launcher picks
for vLLM 0.31.0, on the BF16 release and with `--quantization fp8`. flash-jev int8 runs from the Q8_0 GGUF release. Each
server has one H200 to itself, and all three get the same requests. The release holds no MTP weights, so neither
engine drafts tokens.

| | official, BF16 | official, FP8 | **flash-jev int8** |
|---|---|---|---|
| one question, ~256 prompt tokens (ms, one at a time) | 54.2 | 53.1 | **19.0** |
| one question, ~8,192 tokens (ms) | 728.8 | 531.2 | **439.8** |
| the official README's three questions (ms) | 93.1 | 92.9 | **17.4** |
| the same, with thoughts (ms) | 1,775 | 1,296 | **785** |
| JevBench read-outs, requests/s from 1 / 64 clients | 10.6 / 18.3 | 12.1 / 26.3 | **24.2 / 30.7** |
| JevBench with thoughts, requests/s from 16 / 64 clients | 3.77 / 5.89 | 4.66 / 6.52 | **4.94 / 7.61** |
| a thought's tokens/s (one sequence) | 63 | 94 | 94 |
| evaluation suites with thoughts, average of 12,351 decisions | 92.97 | 92.83 | 92.94 |

- **One request at a time,** flash-jev's read-outs take less than half the time.
- **Under load,** both engines are bound by the tensor cores, which run int8 and FP8 at the same rate on an H200, and
  the lead narrows to 17%.
- **Thoughts** take the same time per token as on the official FP8 server.
- **Accuracy:** over many samples of JevBench's hard items, thoughts written by int8 are slightly less accurate: 221.2
  correct per pass of 231, against 224.9-226.0 for the official engine.

Setup, every measurement and the accuracy experiments are in
[docs/benchmarks.md](docs/benchmarks.md#startlux-decision-v11-27b-autothink-against-its-official-engine).

### StartLux-Decision v1: every engine

Every engine runs alone on the same GPU with the same weights: llama.cpp the Q8_0 files, and the PyTorch engines a
bf16 checkpoint converted from them (`flash_jev/tools/gguf_to_hf.py`). Requests are unique, so no engine can replay a
cached prefix, and all come through the same `/v1/systemone` HTTP client; JevBench is scored with its own scorer.

| Model | File (Hugging Face repository) | Size | SHA-256 (first 16 hex digits) | Run on |
|---|---|---|---|---|
| StartLux-Decision-v1-4B | [`StartLux-Decision-4B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF) | 4.48 GB | `3de47f14ca280781` | RTX 5090 |
|  | [`StartLux-Decision-4B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B-Q4_K_M-GGUF) | 2.71 GB | `36c2aec2c3741f06` | RTX 5090 |
| StartLux-Decision-v1-9B | [`StartLux-Decision-9B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B-Q8_0-GGUF) | 9.53 GB | `9daae5054f92c3d1` | RTX 5090 |
|  | [`StartLux-Decision-9B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B-Q4_K_M-GGUF) | 5.63 GB | `d2757a83badaa8ba` | RTX 5090 |
| StartLux-Decision-v1-27B | [`StartLux-Decision-27B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B-Q8_0-GGUF) | 28.60 GB | `5868a8a2df103c63` | H200 |
|  | [`StartLux-Decision-27B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B-Q4_K_M-GGUF) | 16.55 GB | `7c23c38501989d30` | RTX 5090, H200 |

Every file is byte-identical to StartLux's release of 2026-09-29 (SHA-256 checked on 2026-10-13).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/jevbench-latency-dark.svg">
  <img src="docs/images/jevbench-latency.svg" width="840" alt="StartLux-Decision v1 on JevBench, mean ms per item one request at a time, for every engine on the 4B and 9B (RTX 5090) and the 27B (H200): flash-jev int8 15.0, 21.6 and 41.0; decider with torch.compile and FP8 33.4, 50.3, 66.8; flash-jev bf16 29.6, 52.1, 68.2; SGLang native 42.0, 59.8, 90.1; SGLang via adapter 40.1, 62.1, 92.0; StartLux-Decision server 47.0, 71.2, 96.0; decider default 48.5, 76.1, 99.9; JevK5 49.8, 74.4, 101.1; vLLM 55.3, 84.9, 119.1; llama.cpp 134.7, 162.7, 589.2.">
</picture>

| engine | JevBench, mean ms one at a time (4B / 9B / 27B) | requests/s from 32 clients | score, of 231 |
|---|---|---|---|
| **flash-jev int8** | **15.0 / 21.6 / 41.0** | **85.2 / 53.3 / 30.6** | 205 / 200 / 207 |
| flash-jev bf16 | 29.6 / 52.1 / 68.2 | 37.0 / 19.9 / 16.7 | 205 / 201 / 209 |
| decider 1.9.0, torch.compile + FP8 | 33.4 / 50.3 / 66.8 | 35.0 / 22.6 / 15.4 | (199 / 202 / 206) |
| SGLang 0.5.21, native `/v1/systemone` | 42.0 / 59.8 / 90.1 | 39.5 / 21.7 / 18.8 | 196 / 199 / 211 |
| SGLang 0.5.21, via adapter | 40.1 / 62.1 / 92.0 | 37.4 / 20.5 / 17.7 | 205 / 201 / 209 |
| StartLux-Decision server | 47.0 / 71.2 / 96.0 | 22.9 / 14.2 / 10.4 | 204 / 201 / 209 |
| decider 1.9.0, default | 48.5 / 76.1 / 99.9 | 20.7 / 12.6 / 9.6 | (199 / 202 / 208) |
| JevK5 0.3.3 | 49.8 / 74.4 / 101.1 | 21.8 / 13.6 / 9.9 | (202 / 199 / 211) |
| vLLM 0.31, via adapter | 55.3 / 84.9 / 119.1 | 33.6 / 18.9 / 15.4 | 205 / 201 / 209 |
| llama.cpp b10639 (Q8_0) | 134.7 / 162.7 / 589.2 | 6.5 / 5.7 / 1.4 | 205 / 200 / 209 |

- **Scores:** the model cards give 204 (4B), 200 (9B) and 209 (27B).
- **decider and JevK5** ship no StartLux checkpoint, so they run StartLux's weights as same-shape stand-ins. Their
  speed compares directly, but their scores (in parentheses) come from prompts these weights were not trained with. On
  their own 4B checkpoints, decider scores 194 (FP8) and 193, and JevK5 202.
- **SGLang's native endpoint** also renders its own prompts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/jevbench-throughput-dark.svg">
  <img src="docs/images/jevbench-throughput.svg" width="840" alt="StartLux-Decision v1 on JevBench, requests per second from 32 clients, for every engine on the 4B, 9B (RTX 5090) and 27B (H200): flash-jev int8 85.2, 53.3, 30.6; decider with torch.compile and FP8 35.0, 22.6, 15.4; flash-jev bf16 37.0, 19.9, 16.7; SGLang native 39.5, 21.7, 18.8; SGLang via adapter 37.4, 20.5, 17.7; StartLux-Decision server 22.9, 14.2, 10.4; decider default 20.7, 12.6, 9.6; JevK5 21.8, 13.6, 9.9; vLLM 33.6, 18.9, 15.4; llama.cpp 6.5, 5.7, 1.4.">
</picture>

Single requests and long evidence (p50 ms, one request at a time; 4B / 9B / 27B):

| engine | StartLux's 3-question request | 8,000 tokens of evidence | 50,000 tokens of evidence |
|---|---|---|---|
| **flash-jev int8** | **10.20 / 15.50 / 25.01** | **162.5 / 244 / 432** | **1,960 / 2,582 / 3,693** |
| flash-jev bf16 | 19.06 / 29.84 / 38.60 | - | - |
| decider, torch.compile + FP8 | 75.18 / 111.56 / 158.57 | 328.7 / 458 / 698 | 3,708 / 4,925 / 7,301 |
| SGLang, native | 47.55 / 73.98 / 97.93 | - | - |
| SGLang, via adapter | 49.29 / 81.50 / 95.48 | 342.2 / 603 / 704 | 2,911 / 4,515 / 4,936 |
| StartLux-Decision server | 49.89 / 89.56 / 108.23 | 424.2 / 699 / 870 | fails above 32,768 tokens |
| decider, default | 115.77 / 200.75 / 271.83 | 521.5 / 799 / 1,062 | 3,272 / out of memory / 6,779 |
| JevK5 | 80.98 / 123.53 / 149.94 | 417.7 / 690 / 847 | 3,263 / 5,031 / 5,783 |
| vLLM, via adapter | 74.19 / 111.01 / 175.80 | 440.5 / 702 / 833 | 3,359 / 5,039 / 5,743 |
| llama.cpp | 313.35 / 346.41 / 1,308.49 | 730.3 / 926 / 3,514 | 5,326 / 6,406 / 28,623 |

"-": not run (the long-evidence runs take one configuration per engine). The methods, every table and chart, the
many-client runs, the longest request without chunked prefill and the stress tests are in
**[docs/benchmarks.md](docs/benchmarks.md)**.

### int8 or bf16

Both paths run the same Q8_0 weights, dequantized to bf16 or requantized to int8:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/v11-precision-dark.svg">
  <img src="docs/images/v11-precision.svg" width="840" alt="flash-jev int8 against bf16 on StartLux-Decision v1.1 27B, one H200. Prefill tokens per second: one prompt of 1,024 tokens 18,865 against 11,571; 8,192 tokens 19,219 against 11,424; 32,768 tokens 15,810 against 10,155; passes of 8,192 tokens of short prompts 20,855 against 11,958. Decode ms per step: 1 sequence 10.47 against 14.89; 8 sequences 11.06 against 15.90; 64 sequences 16.70 against 21.74.">
</picture>

- **Speed:** int8 is 1.6-1.75x faster in prefill and 1.3-1.44x in decode.
- **Through the server:** JevBench runs at 30.7 requests/s from 64 clients, against 16.9 for bf16.
- **Accuracy:** one-pass answers score the same. Thoughts written by bf16 are slightly more accurate (226.2 against
  220.5 correct per pass of 231).

### Several GPUs

Two ways to use more GPUs, both giving the single GPU's answers, bit for bit (27B v1.1 on H200s, int8, JevBench):
- **Data parallelism** (`python -m flash_jev.dp`): one whole server per GPU behind a router. More throughput; a
  request takes as long as on one GPU.
- **Tensor parallelism** (`--tp 2` or `4`): one model split over the GPUs. Less time per request, mostly for long
  prompts and thoughts.

Data parallelism, three H200s:

| | one GPU | three GPUs |
|---|---|---|
| read-outs from 64 clients, requests/s | 30.7 | **88.7** (2.89x) |
| with thoughts, requests/s (64 clients on one GPU, 192 on three) | 7.6 | **14.1** |
| one request at a time, mean ms | 41.1 | 42.2 (the router's hop) |

A three-pass run with thoughts cannot end before its longest thought (about 40 s for 4,000 tokens), so that gain shows
in longer runs.

Tensor parallelism, two and four H200s on NVSwitch:

| | one GPU | two GPUs | four GPUs |
|---|---|---|---|
| prefill, one 8,192-token prompt, ms | 427.6 | 301.2 (1.42x) | **241.6** (1.77x) |
| prefill, one 256-token prompt, ms | 17.5 | 17.3 (1.01x) | **15.1** (1.16x) |
| a lone thought's decode step, ms | 10.47 | 7.68 (1.36x) | **6.51** (1.61x) |
| JevBench one at a time, read-outs, mean ms | 41.0 | 34.4 | **29.0** |
| JevBench one at a time, with thoughts, mean ms | 1,209 | 902 | **768** |
| read-outs from 64 clients, requests/s | 30.9 | 43.3 | **54.0** |
| with thoughts from 64 clients, requests/s | 6.28 | 8.26 | **9.81** |

Short prompts gain little: the sums and maxima exchanged between the GPUs take a fifth to a third of a pass on two GPUs
and two fifths on four. For throughput per GPU, data parallelism is ahead (88.7 requests/s on three GPUs against 54.0
on four); tensor parallelism is the one that shortens a request. Details:
[docs/benchmarks.md](docs/benchmarks.md#tensor-parallelism-the-27b-on-2-and-4-h200s).

## Accuracy

**StartLux-Decision v1.** Does the speed cost accuracy? These numbers are from StartLux-Decision-v1's own evaluation,
unchanged ([`eval/suites.py`](https://github.com/StartLuxLabs/StartLux-Decision-v1)):
- **Suites:** the seven accuracy suites of the Intern-Decision bundle (the three public JevBench tiers, Typed
  Decisions, ToolACE, AG News, WildJailBreak; 12,351 decisions), plus its 96-question calibration pilot.
- **Method:** each engine's `/v1/systemone`, one request at a time, on the RTX 5090.
- **Reference:** StartLux-Decision's own server on the same GPU, the same weights and the same client. It reproduces
  StartLux's published averages (4B 91.15, published 91.17; 9B 91.06, published 91.08).

| model | engine | suite average | JevBench, of 231 | same decision as the reference | right → wrong / wrong → right | pilot expected Brier |
|---|---|---|---|---|---|---|
| 4B | StartLux-Decision server (reference) | 91.15 | 204 | | | 0.4991 |
| 4B | **flash-jev int8** | **91.14** | 205 | 98.96% | 70 / 47 | 0.4998 |
| 4B | flash-jev bf16 | 91.26 | 205 | 99.83% | 11 / 9 | 0.4994 |
| 4B | flash-jev W4A8 (Q4_K_M file) | 90.46 | 201 | 98.06% | 127 / 99 | 0.4930 |
| 9B | StartLux-Decision server (reference) | 91.06 | 201 | | | 0.4774 |
| 9B | **flash-jev int8** | **90.80** | 200 | 99.27% | 49 / 40 | 0.4762 |
| 9B | flash-jev bf16 | 91.05 | 201 | 99.90% | 6 / 6 | 0.4774 |
| 9B | flash-jev W4A8 (Q4_K_M file) | 90.84 | 200 | 98.70% | 84 / 68 | 0.4784 |
| 27B | flash-jev W4A8 (Q4_K_M file) | 91.76 | 207 | (published, original weights: 91.82, 208) | | 0.4826 |

- **bf16 matches the reference:** 99.8-99.9% of the decisions are the same; the rest come from bf16 rounding in two
  implementations.
- **int8 (W8A8), the fast path,** scores the same suite average on the 4B and 0.26 lower on the 9B (mostly one of the
  72 items of the Original tier):
  - 1.0% and 0.7% of the decisions change, all near-ties, slightly more often from right to wrong: 23 and 9 net of
    12,351 (0.19% and 0.07%).
  - Calibration does not move (pilot Brier within 0.0012).
  - From 8 clients at once, the answers are bit-identical to one request at a time.
- **W4A8 on the Q4_K_M files** costs what the files' 4-bit weights cost: StartLux publishes 201, 201 and 208 of 231 on
  JevBench for the 4B, 9B and 27B Q4_K_M files, and W4A8 gives 201, 200 and 207.

**StartLux-Decision v1.1 27B AutoThink, with thoughts** (the same suites, against the official engine on the same
H200): flash-jev int8 92.94, the official BF16 server 92.97, FP8 92.83. Thoughts lift the average from 92.00 in one
pass, most of it on JevBench's hard tier (85.59 → 94.59).

Every suite and the agreement per suite: [docs/benchmarks.md](docs/benchmarks.md#accuracy-startlux-decision-v1s-evaluation).

## Quick start

flash-jev needs:
- Linux with an NVIDIA GPU: an RTX 5090 (sm_120) or a Hopper GPU (sm_90a);
- Python 3.12 with PyTorch (CUDA 13), Triton and a CUDA toolkit. The CUDA extension is built on first use, for the GPU
  it runs on.

The benchmark servers have no internet: `scripts/setup_server.sh` installs wheels fetched elsewhere into `~/jev/venv`,
with the pip CUDA 13 toolkit as `CUDA_HOME`, and writes `~/jev/env.sh`.

```bash
source ~/jev/env.sh
MODEL=~/jev/models/StartLux-Decision-4B-Q8_0-GGUF scripts/serve.sh start int8    # port 8092; waits for /health
curl -s localhost:8092/v1/systemone -H 'Content-Type: application/json' -d '{"state": {"ticket": "I was charged twice."},
  "questions": {"urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"}}}'
scripts/serve.sh stop
```

```text
{"answers": {"urgent": {"type": "noul", "noul": 0.5502554493555311}}, "usage": {"input_tokens": 70, "output_tokens": 0},
 "model": "StartLux-Decision-4B (flash-jev)", "latency_ms": 5.257, "timings": {"bucket": 64, "gpu_ms": 4.334, ...}}
```

`MODEL` is the GGUF repository: the `.gguf` file, with its tokenizer and config beside it. Weights load straight from
the GGUF and are requantized on the GPU at start-up. The int8 path's SmoothQuant activation statistics are computed on
the first start and kept in the `--calib` file.

StartLux-Decision v1.1 AutoThink, with thoughts and images, without the script:

```bash
M=~/jev/models/StartLux-Decision-v1.1-27B-AutoThink-GGUF
python -m flash_jev.server --gguf $M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --model-dir $M --precision int8 \
  --calib ~/jev/calib/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.pt --port 8092
curl -s localhost:8092/v1/systemone -H 'Content-Type: application/json' -d '{"state": {"ticket": "I was charged twice
  for order #4411 and the app still shows it as unpaid."}, "questions": {"team": {"type": "choice", "instructions":
  "Which team should handle this ticket?", "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
  "technical": "App and login"}}}, "return_details": true}'
```

The answer's `none_probability` is the chance that no listed option fits. `details` shows the first read-out, the
thought (if any) and the read-out after it. The `mmproj-*.gguf` beside the model file is loaded automatically, so
`"images": [...]` works too.

Several GPUs, one server each behind a router on one port:

```bash
python -m flash_jev.dp --gpus 0,1,2,3 --port 8092 --gguf $M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf \
  --model-dir $M --precision int8 --calib ~/jev/calib/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.pt
```

One model over two GPUs (tensor parallelism), for lower latency:

```bash
python -m flash_jev.server --tp 2 --gpus 0,1 --gguf $M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf \
  --model-dir $M --precision int8 --calib ~/jev/calib/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.pt --port 8092
```

## Models, GPUs and precisions

| `--precision` | weights | activations | used for |
|---|---|---|---|
| `int8` (W8A8) | int8 per output channel, requantized from the GGUF with SmoothQuant at start-up | int8 per token | Q8_0 files (`auto`) |
| `w4a8` | the file's own Q4_K / Q6_K blocks, as stored | int8 per 256 inputs | Q4_K_M files (`auto`) |
| `bf16` | dequantized from the GGUF, cuBLAS | bf16 | the parity reference |

Measured:
- **RTX 5090:** the 4B and 9B (Q8_0: int8 and bf16; Q4_K_M: W4A8) and the 27B's Q4_K_M (W4A8).
- **H200:** the v1 27B (Q8_0: int8 and bf16; Q4_K_M: W4A8 and int8), and the v1.1 27B AutoThink (Q8_0: int8 and bf16;
  int8 on 2 and 4 GPUs).

To add another model of the family, run `scripts/prepare_model.sh` (its folders for every engine), then
`scripts/new_model.sh` (dispatch-table entries for its projection shapes, int8 calibration, correctness tests).

## Serving

Endpoints: `POST /v1/systemone` (also `POST /v1/decisions`), `GET /v1/models`, `GET /health` (the folder's kind and
the defaults) and `GET /stats` (batches, prefix-cache hits and evictions, how thoughts ended).

Request fields (the TypeSafe / StartLux-Decision format; the v1.1 fields as its package defines and validates them):

| field | default | effect |
|---|---|---|
| `state` | | the evidence: a string or any JSON value |
| `questions` | | `{key: {"type": "choice" \| "noul" \| "score", "instructions", "criteria"}}` |
| `images` | none | v1.1 with a vision tower: base64 strings or data URIs, at most `--max-images` (8), at the state's `<image>` marks or before it; each resized to at most `--max-pixels` (1,048,576) and read as one token per 32 x 32 pixels |
| `think_mode` | `auto` on AutoThink folders | `auto`, `never` or `always` |
| `gate`, `mix`, `think_budget` | 0.9, 0.75, 4,096 | the threshold to think, the weight of the read-out after the thought, the longest thought |
| `none_check` | true on v1.1 folders | the extra option "None of the above" and `none_probability` |
| `allow_none` | false | `choice` is null when "None of the above" is the most likely option (also per question) |
| `return_details`, `return_thought` | false | per-answer read-outs and `timing_ms`, and the thought's text |

A malformed body gets 422 on v1.1 folders and 400 on v1 folders, as each release's own server answers.

Server flags:

| flags | default | |
|---|---|---|
| `--precision`, `--calib` | `auto` | the precision (above) and the int8 activation ranges (computed and saved on the first start) |
| `--max-tokens`, `--mem-fraction` | 0, 0.9 | the largest batch; 0 sizes it from free GPU memory (65,536 tokens for the 4B on the RTX 5090, 131,072 for the 27B on the H200). Longer requests get HTTP 422 |
| `--budget` | 8,192 | tokens per batched pass when requests queue; a longer request runs alone |
| `--prefix-cache`, `--prefix-cache-tokens`, `--prefix-cache-min` | 64, 65,536, 64 | cached evidence states (0: off), their KV rows, the shortest evidence cached |
| `--none-check`, `--allow-none` | `auto` (on for v1.1) | the defaults of the request fields |
| `--think-mode`, `--max-think-budget`, `--think-seqs`, `--think-kv-rows` | the folder's, 4,096, 64, auto | thoughts: the default mode, the largest budget, thoughts generated at once (144 MiB of state each on the 27B) and their KV rows (think-seqs x (budget + 1,024)) |
| `--mmproj`, `--max-images`, `--max-pixels` | `auto`, 8, 1,048,576 | the vision tower (`auto`: the `mmproj-*.gguf` beside `--gguf`; `none`: text only); images need Pillow |
| `--graph-max-tokens`, `--no-graphs` | | CUDA graphs for every token bucket (recorded at start-up) or fewer |
| `--attn-qk8` | off | opt-in: q k^T on int8 tensor cores, 15% off at 50,000 tokens at a small probability error |
| `--tp`, `--gpus` | 1, the visible GPUs | tensor parallelism: one model over this many GPUs (int8; 2 or 4 on the 27B) |

**Several GPUs, data parallelism** (`python -m flash_jev.dp --gpus ...`, the server's flags pass through):
- **Workers:** one whole server per GPU, each with its own prefix cache and thoughts.
- **Routing:** the router forwards every request whole to the worker that last answered the same evidence, unless
  that worker has more than `--affinity-slack` (2) requests in flight beyond the least loaded one; otherwise to the
  least loaded worker.
- **Failures:** a worker that fails is skipped until its `/health` answers again.

**One model over several GPUs, tensor parallelism** (`--tp N --gpus a,b,...`, the int8 path):
- **Sizes:** N divides the attention heads, kv heads, linear-attention key heads and FFN width: 2 or 4 on the 27B.
- **Processes:** the server's process is rank 0 on the first GPU and starts one process per further GPU
  (`python -m flash_jev.tp`, not run by hand), which replays its passes. If one of them dies, the server exits.
- **Memory:** each GPU holds 1/N of the projections, of the KV rows and of the recurrent states (72 MiB per thought at
  2 GPUs), so the buffers above are sized per GPU (the smallest GPU's choice; rank 0 also holds the vision tower).
- **Tuning:** `FLASHJEV_TP_FAST_ROWS` (16,384) and `FLASHJEV_TP_PULL_ROWS` (128 on 2 GPUs, 4 on 4) set the longest
  passes whose maxima and sums stay inside the kernels.

## How it works

**One forward pass per batch, recorded as CUDA graphs.** A batch is a packed forest of token segments:
- the system text, computed once at start-up and kept;
- the evidence a request's questions share, computed once per batch;
- each question's own tokens, which continue from the evidence's state.

Each pass runs two graphs: the body over all packed tokens of a token bucket, and the head (the last layer, at the
answer positions only) over a read bucket. All batch structure lives in device tables uploaded before replay, so one
graph per bucket serves every batch.

**Questions branch off the shared evidence, inside the graph.** Linear-attention layers scan in two waves: first the
segments whose parent state exists, then their children. Attention reads the system-text and parent key ranges in
place. Other engines either recompute the evidence for every question or share it on an eager path outside their
graphs.

**Evidence cached across requests.** A later request with the same evidence starts its questions from the evidence's
cached state. Every linear-attention layer's recurrent state and conv tail sits in an arena slot, and the attention
layers' keys and values in reserved KV-arena rows. Entries are looked up by exact token match (and the images'
digests) and evicted least recently used. The pass that computes new evidence writes the entry on the way.

**Batch-invariant numerics.** Which GEMM kernel, tile and K split runs depends on the batch's size, but every one of
them gives an element the same bits: exact int32 sums, then the same explicitly rounded scale and add, with split-K
partials kept in int32. The rest of the pass is per segment and per row. So a request's probabilities do not depend on
what it is batched with, on concurrency, or on the cache.

**Kernels per GPU.** The extension is built for the GPU it runs on (sm_120 or sm_90a). Each projection's kernel, tile
and K split is chosen per shape and token count from a table measured on that GPU model
(`kernels/gemm_table_<gpu>.json`):
- **RTX 5090:** mma.sync kernels (one CTA per tile, split-K, stream-K) and a Triton kernel.
- **H200:** warpgroup-MMA kernels with TMA copies and warp specialization (`csrc/w8a8_wgmma.cu`: 1,240-1,420 TOPS at
  4,000-8,000 tokens, where mma.sync reaches about 530 and cuBLASLt's int8 870-910).

Every choice gives the same bits, so it changes only the speed.

**int8 tensor cores.** On the RTX 5090, INT8 runs at 838 TOPS dense, BF16 with fp32 accumulation at 209.5 TFLOPS, and
int8 weights halve the weight stream that bounds short requests.
- **Weights:** requantized per output channel from the Q8_0 values with SmoothQuant (alpha 0.75, calibrated on
  StartLux-Decision's own calibration requests).
- **Activations:** quantized per token by the kernels that produce them (norms, gated norm, SwiGLU, attention output).
- **Accumulation:** exact in int32 over all of K, scaled once in the epilogue. Output projections add straight into the
  fp32 residual stream.
- **Q4_K_M files** keep their 4-bit weights: the W4A8 GEMM unpacks Q4_K / Q6_K blocks next to the MMA, with int8
  activations per 256 inputs (`csrc/w4a8_gemm.cu`).

**Last layer at the read positions only.** Keys and values are computed for every token; queries, attention output,
MLP, norm and the 26 letter rows only where answers are read.

**Thoughts.** A question that thinks gets its thinking prompt (the folder's chat template with thinking on) prefilled
whole inside an ordinary batch: its recurrent state and conv tails go into a state slot of its own, and its keys and
values into KV rows of its own. Then come the decode steps (`engine/decode.py`, one CUDA graph per batch size, up to 64
sequences):
- the conv and recurrence steps run in place on the slot;
- attention covers the sequence's rows in 512-key splits;
- the output head runs on the GGUF's Q8_0 rows;
- the sampler runs in vLLM's order, drawing from a counter-based generator seeded per question, so a thought is the
  same whatever it is batched with.

The scheduler alternates prefill batches with decode steps, keeping one step in flight while requests arrive.

**Images.** `images.py` places, decodes and resizes images as v1.1 does. The vision tower (`vision.py`: the `mmproj`
GGUF's patch embedding, 27 blocks and 2 x 2 merger in bf16) encodes each image alone, op for op as transformers' Qwen3.5
vision model, with CUDA graphs per image size. Its features go into the residual rows of the image's tokens. Image
tokens carry Qwen3.5's M-RoPE positions (time, row, column) in a table the attention layers read.

**Tensor parallelism.** One process per GPU (`tp.py`). Each GPU holds its share of the heads (attention: its query
heads and the kv heads they read; linear attention: its key heads and the value heads that read them), of the MLP's
columns and, for decoding, of the vocabulary; embedding, norms and the residual stream are on every GPU.
- **Exact sums:** the output projections leave int32 partial sums, added over the GPUs (exact in any order), then
  scaled and added once as on one GPU. The per-token scales of their inputs take the maximum over the GPUs. So every
  answer, cache hit, batch and thought is the single GPU's, bit for bit.
- **Four all-reduces per layer:** the two maxima are exchanged inside the kernels that need them (per row, a 64-bit
  word that carries the pass number). The two sums of the shortest passes (decode steps on 2 GPUs) are read by the
  kernel that adds them into the residual, straight from the other GPUs' memory over NVLink. The other sums go through
  NCCL inside the CUDA graphs.
- **One plan:** the server's process plans every pass and publishes it to the others through shared memory, and every
  GPU replays the same CUDA graphs.

Code:

| files | role |
|---|---|
| `flash_jev/engine/model.py`, `engine/batch.py` | the engine and its graphs; the forest planner |
| `kernels/gdn.py` | conv + prep, the WY chunk kernel, the two-wave state scan, the gated norm |
| `kernels/attn.py` | QK norm + RoPE + KV arena, tree flash attention |
| `kernels/w8a8.py`, `kernels/quant.py`, `csrc/` | the int8 and W4A8 paths |
| `reference.py` | a plain fp32 PyTorch forward, the yardstick for every kernel |
| `decide.py`, `think.py`, `engine/decode.py`, `loops.py` | requests and answers; thoughts |
| `images.py`, `vision.py` | images |
| `server.py`, `dp.py`, `tp.py` | HTTP and scheduler; several GPUs: data and tensor parallelism |

What was kept, measured, tried and dropped, with numbers: [docs/optimizations.md](docs/optimizations.md).

## Correctness

- **Against an fp32 reference forward** (`tests/test_engine.py`, 23 questions):
  - bf16 makes the reference's decision on all 23 on every model (largest probability difference 0.003-0.005);
  - int8 on 22 (4B: one near-tie flips) or 23 (9B, 27B), largest difference 0.034-0.043;
  - W4A8 on 23 (4B) and 22 (9B), 0.031-0.032.
- **Bit for bit:**
  - every GEMM config, split mode and stream-K gives the Triton kernel's bits, the wgmma kernels included
    (`tests/test_cuda_gemm.py`);
  - prefix-cache hits equal recomputing (`tests/test_prefix_cache.py --exact`);
  - 64 answers sent by 32 clients at once equal the same requests one at a time (`tests/test_stress.py`);
  - the answers through the data-parallel router equal one GPU's.

  The bf16 path is deterministic run to run, but cuBLAS picks its kernels by batch size, so it is not batch-invariant.
- **StartLux-Decision v1.1** (`tests/test_v11.py`):
  - 5,787 random prompts render to the same text and tokens as v1.1's package;
  - 4,000 random requests (choice lists up to 300 options, every none_check / allow_none combination) get the same
    answers, every float, from the same letter logits;
  - v1 folders answer exactly as before v1.1 support.
- **Thoughts** (`tests/test_think.py`, `tests/test_decode.py`):
  - v1.1's loop detector is reproduced on 5,000 texts;
  - 2,885 thinking and read prompts, 899 with images, are identical in text and tokens;
  - 300 whole requests with 499 thoughts give the same answers, details and usage as v1.1's pipeline, to the float;
  - decode steps agree with the prefill under teacher forcing;
  - sequences alone, batched or joining a running batch give the same tokens;
  - JevBench from 64 clients gives the same thoughts as one request at a time.
- **Images** (`tests/test_vision.py`, `tests/test_vision_engine.py`, `tests/vision_reference.py`):
  - the tower's features are bit-identical to transformers' on 9 image sizes;
  - 36 prompts with 1 to 8 images are token-identical to v1.1's, with its positions;
  - the 53 questions of 24 image requests get transformers' decisions, all right;
  - text answers stay bit-identical with the tower loaded.
- **Tensor parallelism** (`tests/test_tp.py`, the v1.1 27B on 2 and 4 GPUs against one, bit for bit):
  - the letter logits of 30 requests alone, again from the prefix cache and as one 7,492-token batch;
  - long evidence as a miss, a hit, another question about it and inside mixed batches; a 33-option list;
  - six thoughts' 64 decode steps, greedy, joining a running batch and sampled (each step's 248,320 logits and token);
  - 24 questions about images; on the v1 27B, the same 30 requests;
  - JevBench with thoughts through the server (231 items from 16 clients, 67 of them thinking): the same answers,
    probabilities and thoughts, to the last character;
  - every int32-capable GEMM's per-GPU sums, added and folded, give the one-GPU bits (`tests/test_cuda_gemm.py`).

## Tests and benchmarks

On the GPU server, from the repository (`scripts/sync.sh` copies the working tree there):

```bash
M=~/jev/models/StartLux-Decision-4B-Q8_0-GGUF; JB=~/jev/src/jevbench-7ce310c7
python tests/test_engine.py --gguf $M/StartLux-Decision-4B-Q8_0.gguf --model-dir $M --jevbench $JB/datasets/public
python tests/test_stress.py --model-dir $M --jevbench $JB --start int8 [--quick] --out results/stress/flash-jev-int8.json
python bench/run_jevbench.py --url http://127.0.0.1:8092/v1/systemone --label flash-jev --out results/jevbench/x.json
python bench/charts.py                                                        # docs/images
```

| Test | Checks |
|---|---|
| `tests/test_engine.py` | every precision against the fp32 reference forward (`--repeat`: run-to-run bits) |
| `tests/test_cuda_gemm.py` | the CUDA GEMMs against exact integer math and the Triton kernel's bits |
| `tests/test_w4a8.py`, `tests/test_kquant.py` | the W4A8 GEMM against its reference arithmetic; k-quant dequantization against gguf-py, bit for bit |
| `tests/test_prefix_cache.py --exact` | cache hits bit for bit |
| `tests/test_batch.py`, `tests/test_prompt.py` | the forest planner; prompts byte for byte against the chat template |
| `tests/test_v11.py` | StartLux-Decision v1.1 on the CPU: prompts and answers against its own package (`STARTLUX_DECIDE` = its checkout); v1 folders against the code before v1.1 support |
| `tests/test_think.py`, `tests/test_decode.py` | thoughts against v1.1's package (loop detector, prompts, whole requests with fake engines); decode steps against the prefill (teacher forcing, isolation, sampling) |
| `tests/test_vision.py`, `tests/test_vision_engine.py`, `tests/vision_reference.py` | images: placement, pixels, tokens and positions against v1.1 and transformers; the tower and image batches on the GPU; answers against transformers' model |
| `tests/test_dp.py` | the data-parallel router on the CPU, against fake workers: evidence affinity, routing by load, failover |
| `tests/test_tp.py` | tensor parallelism against one GPU, bit for bit: read-outs, cache hits, batches, wide choices, decode logits and tokens, images (`--tp 2 --gpus 2,3 --ref-gpu 1`); `--server`: JevBench with thoughts through a `--tp` server and a one-GPU server |
| `tests/test_stress.py` | a running server under load: long evidence counted in full, the capacity limit refused cleanly, many clients without a failed request, bit-identical answers under load |
| `scripts/accuracy_v1.sh` | StartLux-Decision-v1's evaluation suites through every flash-jev precision and StartLux's own server |
| `bench/run_jevbench.py`, `bench/prefill_speed.py`, `bench/decode_speed.py`, `bench/vision_latency.py` | JevBench through any `/v1/systemone` server (latency, throughput, agreement); prefill and decode on the engine itself (`--tp N` too); images |

## Documents

| Document | Contents |
|---|---|
| [Benchmarks and stress tests](docs/benchmarks.md) | every model, GPU and engine: methods, tables, charts, many clients, long context, the official v1.1 engine, images, several GPUs (data and tensor parallelism), stress tests |
| [Optimization log](docs/optimizations.md) | where the time goes; what was kept, measured, tried and dropped |

## License

flash-jev's own code: to be decided by the owner. Vendored from StartLux-Decision (Apache-2.0,
`LICENSES/StartLux-Decision-Apache-2.0.txt`): `flash_jev/jevfmt.py`, `flash_jev/calibration.py`, and `flash_jev/loops.py`
from v1.1's package. Model weights are not part of this repository; StartLux-Decision's weights are CC BY-NC 4.0
(non-commercial).
