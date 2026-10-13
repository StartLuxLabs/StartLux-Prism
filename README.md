# flash-jev

**A CUDA inference engine for JEV decision models: hand-written kernels, one forward pass per batch, bit-exact
answers, on the RTX 5090 and the H200.**

JEV ("System One") models answer typed questions about a piece of evidence (a choice among options, a yes/no, a
score) with calibrated probabilities, read from the option letters' logits at the last prompt token; they generate
nothing. flash-jev serves [StartLux-Decision v1](https://github.com/StartLuxLabs/StartLux-Decision-v1) at 4B, 9B and 27B
(Qwen3.5 dense hybrids of Gated DeltaNet and gated attention) straight from their GGUF files on one NVIDIA GPU, behind
the TypeSafe `/v1/systemone` API. Prompt format, readout, temperatures and wide-choice rounds are StartLux-Decision's
own (`flash_jev/jevfmt.py` is vendored unchanged), so only the forward pass differs from the reference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/jevbench-latency-dark.svg">
  <img src="docs/images/jevbench-latency.svg" width="840" alt="JevBench mean ms per item, one request at a time, for every engine on StartLux-Decision-v1-4B and -9B (RTX 5090) and -27B (H200), Q8_0 files: flash-jev int8 15.0, 21.6 and 41.0; decider with torch.compile and FP8 33.4, 50.3, 66.8; flash-jev bf16 29.6, 52.1, 68.2; SGLang native 42.0, 59.8, 90.1; SGLang via adapter 40.1, 62.1, 92.0; StartLux-Decision server 47.0, 71.2, 96.0; decider default 48.5, 76.1, 99.9; JevK5 49.8, 74.4, 101.1; vLLM 55.3, 84.9, 119.1; llama.cpp 134.7, 162.7, 589.2.">
</picture>

## Highlights

- **Lowest latency of every engine compared.** JevBench's 231 public items one request at a time: 1.6-2.3x below the
  next engine's mean latency (decider with torch.compile and FP8); StartLux's own 3-question request 3.8-4.8x faster
  than the next (SGLang): 10.2 ms on the 4B, 25.0 ms on the 27B.
- **1.6-2.4x the throughput under load.** JevBench from 32 clients: 85 requests/s on the 4B (next: 39.5), 53 on the
  9B (22.6), 30.6 on the 27B on an H200 (18.8). No failed request at 8, 16 or 32 clients.
- **Long evidence in one pass.** 50,000 tokens in 1.96 s on the 4B (SGLang 2.91 s) and 3.69 s on the 27B on an H200
  (SGLang with FlashAttention 3, 4.94 s). The largest batch is sized from free GPU memory (65,536 tokens on the RTX
  5090, 131,072 on the H200); a longer request is refused with HTTP 422, never cut.
- **Follow-up questions from a prefix cache.** A further question about evidence already sent starts from the
  evidence's cached state: 5.4 ms instead of 54 ms on the longest JevBench items (SGLang 113 ms, vLLM 43 ms).
- **Deterministic and batch-invariant.** On the int8 and W4A8 paths a request gets bit-identical probabilities on
  every run, whatever it is batched with, however many clients are sending, and whether its evidence hit the cache.
- **Accuracy kept.** On StartLux-Decision-v1's own evaluation (12,351 decisions) flash-jev int8 averages 91.14
  against 91.15 for StartLux's own implementation on the same weights (4B) and 90.80 against 91.06 (9B); bf16 matches
  it on 99.8-99.9% of the decisions ([Accuracy](#accuracy)).
- **4-bit files on their own weights.** Q4_K_M runs as W4A8 on int8 tensor cores: 3.9-6.9x faster than llama.cpp on
  the same file, with the same JevBench score (within 1), and the 27B fits a 32 GB RTX 5090 this way.
- **Kernels per GPU.** The extension is built for the GPU it runs on, and every projection's kernel, tile and K split
  comes from a table measured on that GPU model; on Hopper, warpgroup-MMA int8 GEMMs with TMA and warp specialization
  (1,240-1,420 TOPS).

## Performance

Every engine alone on the same GPU, unique requests (no engine can replay a cached prefix) through the same
`/v1/systemone` HTTP client, JevBench scored with its own scorer; StartLux-Decision-v1-4B and -9B on an RTX 5090, -27B
on an H200, every engine from the same Q8_0 file (below).

**Models, for every benchmark, stress test and accuracy run.** StartLux-Decision v1, StartLux's GGUF releases on Hugging
Face (published 2026-09-29), unchanged: every
file run here is byte-identical to the release (SHA-256 checked on 2026-10-13). The repositories were named
`StartLux-Decision-<size>-<precision>-GGUF` when we downloaded them and now carry a `v1`; the old names redirect.

| Model | File (Hugging Face repository) | Size | SHA-256 (first 16 hex digits) | Run on |
|---|---|---|---|---|
| StartLux-Decision-v1-4B | [`StartLux-Decision-4B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF) | 4.48 GB | `3de47f14ca280781` | RTX 5090 |
|  | [`StartLux-Decision-4B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-4B-Q4_K_M-GGUF) | 2.71 GB | `36c2aec2c3741f06` | RTX 5090 |
| StartLux-Decision-v1-9B | [`StartLux-Decision-9B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B-Q8_0-GGUF) | 9.53 GB | `9daae5054f92c3d1` | RTX 5090 |
|  | [`StartLux-Decision-9B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-9B-Q4_K_M-GGUF) | 5.63 GB | `d2757a83badaa8ba` | RTX 5090 |
| StartLux-Decision-v1-27B | [`StartLux-Decision-27B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B-Q8_0-GGUF) | 28.60 GB | `5868a8a2df103c63` | H200 |
|  | [`StartLux-Decision-27B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-v1-27B-Q4_K_M-GGUF) | 16.55 GB | `7c23c38501989d30` | RTX 5090, H200 |

The other engines run the same weights: llama.cpp the same files, and the PyTorch engines (StartLux-Decision's own
server, SGLang, vLLM, decider, JevK5) a bf16 checkpoint converted from the Q8_0 file by `flash_jev/tools/gguf_to_hf.py`
(StartLux reports that the Q8_0 files make the original weights' decision on 100% of the public JevBench items).
decider and JevK5 also ran on their own checkpoints (Mapika/decider-4b v2, JevK5 v0.3). Full hashes:
[docs/benchmarks.md](docs/benchmarks.md#setup).

**JevBench, every engine** (the 231 public items):

| engine | mean ms per item, one at a time (4B / 9B / 27B) | requests/s from 32 clients (4B / 9B / 27B) | score, of 231 (4B / 9B / 27B) |
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

The model cards give 204 (4B), 200 (9B) and 209 (27B). decider and JevK5 ship no StartLux checkpoint, so they run
StartLux's weights as same-shape stand-ins: their speed compares directly, but their scores (in parentheses) come from
prompts these weights were not trained with; on their own 4B checkpoints decider scores 194 (FP8) and 193, JevK5 202.
SGLang's native endpoint also renders its own prompts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/jevbench-throughput-dark.svg">
  <img src="docs/images/jevbench-throughput.svg" width="840" alt="JevBench requests per second from 32 clients for every engine on StartLux-Decision-v1-4B, -9B (RTX 5090) and -27B (H200), Q8_0 files: flash-jev int8 85.2, 53.3, 30.6; decider with torch.compile and FP8 35.0, 22.6, 15.4; flash-jev bf16 37.0, 19.9, 16.7; SGLang native 39.5, 21.7, 18.8; SGLang via adapter 37.4, 20.5, 17.7; StartLux-Decision server 22.9, 14.2, 10.4; decider default 20.7, 12.6, 9.6; JevK5 21.8, 13.6, 9.9; vLLM 33.6, 18.9, 15.4; llama.cpp 6.5, 5.7, 1.4.">
</picture>

**Single requests and long evidence, every engine** (p50 ms, one request at a time):

| engine | StartLux's 3-question request (4B / 9B / 27B) | 8,000 tokens of evidence (4B / 9B / 27B) | 50,000 tokens of evidence (4B / 9B / 27B) |
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

"-": not run (the long-evidence runs take one configuration per engine; flash-jev's 9B ran them with `--prefix-cache 0`, its largest batch then 65,536 tokens). Methods, every table and chart, many-client
runs, the longest request without chunked prefill and the stress tests: **[docs/benchmarks.md](docs/benchmarks.md)**.

## Accuracy

Does the speed cost accuracy? StartLux-Decision-v1's own evaluation, unchanged (`eval/suites.py` of
[StartLuxLabs/StartLux-Decision-v1](https://github.com/StartLuxLabs/StartLux-Decision-v1)): the seven accuracy suites of
the Intern-Decision bundle (the three public JevBench tiers, Typed Decisions, ToolACE, AG News, WildJailBreak; 12,351
decisions) and its 96-question calibration pilot, through each engine's `/v1/systemone`, one request at a time, on the
RTX 5090 (2026-10-13). The reference is StartLux-Decision's own server on the same GPU, the same weights (bf16,
converted from the same Q8_0 file) and the same client; it reproduces StartLux's published averages (4B 91.15,
published 91.17; 9B 91.06, published 91.08).

| model | engine | suite average | JevBench public, of 231 | same decision as the reference | right → wrong / wrong → right | pilot expected Brier |
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

- **bf16 is the reference's equal**: 99.8-99.9% of the decisions are the same, the rest bf16 rounding between two
  implementations.
- **int8 (W8A8), the fast path**: the same suite average on the 4B, 0.26 lower on the 9B (mostly one of the 72 items of
  the Original tier). 1.0% and 0.7% of the decisions change, near-ties, slightly more often from right to wrong: 23 and
  9 net of 12,351 (0.19% and 0.07%). Calibration does not move (pilot Brier within 0.0012). From 8 clients at once the
  answers are bit-identical to one request at a time.
- **W4A8 on the Q4_K_M files** costs what the files' 4-bit weights cost: StartLux publishes 201, 201 and 208 of 231 on
  JevBench for the 4B, 9B and 27B Q4_K_M files; W4A8 gives 201, 200 and 207. (The 27B's Q8_0 and bf16 do not fit the
  RTX 5090: its int8 path and the reference need the H200.)

Every suite, the agreement per suite and the method: [docs/benchmarks.md](docs/benchmarks.md#accuracy-startlux-decision-v1s-evaluation).

## Quick start

flash-jev runs on Linux with one NVIDIA GPU, an RTX 5090 (sm_120) or a Hopper GPU (sm_90a), and Python 3.12 with
PyTorch (CUDA 13), Triton and a CUDA toolkit; its CUDA extension is built on first use for the GPU it runs on. The
benchmark servers have no internet: `scripts/setup_server.sh` installs wheels fetched elsewhere into `~/jev/venv`, with
the pip CUDA 13 toolkit as `CUDA_HOME`, and writes `~/jev/env.sh`.

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

`MODEL` is the GGUF repository (the `.gguf` file in it, its tokenizer and config beside it). Without the script:
`python -m flash_jev.server --gguf FILE.gguf --model-dir REPOSITORY --precision auto --calib CALIB.pt --port 8092`.
Weights load straight from the GGUF and are requantized on the GPU at start-up (a few seconds); the int8 path's
SmoothQuant activation statistics are computed on the first start and kept in the `--calib` file.

## Models, GPUs and precisions

| `--precision` | weights | activations | used for |
|---|---|---|---|
| `int8` (W8A8) | int8 per output channel, requantized from the GGUF with SmoothQuant at start-up | int8 per token | Q8_0 files (`auto`) |
| `w4a8` | the file's own Q4_K / Q6_K blocks, as stored | int8 per 256 inputs | Q4_K_M files (`auto`) |
| `bf16` | dequantized from the GGUF, cuBLAS | bf16 | the parity reference |

Measured (StartLux-Decision v1, the files listed under [Performance](#performance)): the 4B and 9B (Q8_0: int8 and
bf16; Q4_K_M: W4A8) and the 27B's Q4_K_M (W4A8) on an RTX 5090; the 27B (Q8_0: int8 and bf16; Q4_K_M: W4A8 and int8)
on an H200. Another model of the family:
`scripts/prepare_model.sh` (its folders for every engine), then `scripts/new_model.sh` (dispatch-table entries for its
projection shapes, int8 calibration, correctness tests).

## How it works

**One forward pass per batch, recorded as CUDA graphs.** A batch is a packed forest of token segments: the system
text (computed once at start-up and kept), the evidence a request's questions share (computed once per batch), and
each question's own tokens, which continue from the evidence's state. Two graphs per pass: the body over all packed
tokens of a token bucket, and the head (the last layer at the answer positions only) over a read bucket. All batch
structure lives in device tables uploaded before replay, so one graph per bucket serves every batch.

**Questions branch off the shared evidence, inside the graph.** Linear-attention layers scan in two waves (segments
whose parent state exists, then their children), and attention reads the system-text and parent key ranges in place.
Other engines either recompute the evidence for every question (one padded row per question) or share it on an eager
path outside their graphs.

**Evidence cached across requests.** A later request with the same evidence (another question about a document
already sent, the next call of a session) starts its questions from the evidence's cached state: every
linear-attention layer's recurrent state and conv tail in an arena slot, the attention layers' keys and values in
reserved KV-arena rows, looked up by exact token match and evicted least recently used (never an entry the batch being
planned uses). The pass that computes new evidence writes the entry on the way.

**Batch-invariant numerics.** Which GEMM kernel, tile and K split runs depends on the batch's size; every one of them
gives an element the same bits (exact int32 sums, then the same explicitly rounded scale and add, split-K partials
kept in int32), and the rest of the forward pass is per segment and per row. So a request's probabilities do not depend
on what it is batched with, on concurrency, or on the cache.

**Kernels per GPU.** The extension is built for the GPU it runs on (sm_120 on the RTX 5090, sm_90a on Hopper), and
each projection's kernel, tile and K split is chosen per shape and token count from a table measured on that GPU model
(`kernels/gemm_table_<gpu>.json`): mma.sync kernels (one CTA per tile, split-K, stream-K) and a Triton kernel on the
RTX 5090; on the H200 warpgroup-MMA kernels with TMA copies and warp specialization (`csrc/w8a8_wgmma.cu`: 1,240-1,420
TOPS at 4,000-8,000 tokens, where mma.sync reaches about 530 on that GPU and cuBLASLt's int8 870-910). Attention and
linear-attention tiles are set per GPU as well. Every choice gives the same bits, so it changes the speed only.

**int8 tensor cores.** On the RTX 5090, INT8 runs at 838 TOPS dense while BF16 with fp32 accumulation runs at 209.5
TFLOPS (NVIDIA RTX Blackwell whitepaper), and int8 weights halve the weight stream that bounds short requests. Weights
are requantized per output channel from the Q8_0 values with SmoothQuant (alpha 0.75, calibrated on StartLux-Decision's
own calibration requests), activations per token, accumulated exactly in int32 over all of K and scaled once in the
epilogue. Activations are quantized by the kernels that produce them (norms, gated norm, SwiGLU, attention output), and
output projections add straight into the fp32 residual stream. Q4_K_M files keep their 4-bit weights: the W4A8 GEMM
unpacks Q4_K / Q6_K blocks next to the MMA, with int8 activations per 256 inputs (`csrc/w4a8_gemm.cu`).

**Last layer at the read positions only.** Keys and values for every token; queries, attention output, MLP, norm and
the 26 letter rows only where answers are read.

Code: `flash_jev/engine/model.py` (engine, graphs), `engine/batch.py` (forest planner), `kernels/gdn.py` (conv + prep,
WY chunk kernel, two-wave state scan, gated norm), `kernels/attn.py` (QK norm + RoPE + KV arena, tree flash attention),
`kernels/w8a8.py`, `kernels/quant.py` and `csrc/` (int8 and W4A8 paths), `reference.py` (a plain fp32 PyTorch forward,
the yardstick for every kernel), `decide.py` (requests, wide choices, answers), `server.py` (HTTP, scheduler). What was
kept, measured, tried and dropped, with numbers: [docs/optimizations.md](docs/optimizations.md).

## Correctness

- **Against an fp32 reference forward** (`tests/test_engine.py`, 23 questions): bf16 makes the reference's decision
  on all 23 on every model (largest probability difference 0.003-0.005); int8 on 22 (4B: one near-tie flips) or 23 (9B,
  27B), largest difference 0.034-0.043; W4A8 on 23 (4B) and 22 (9B), 0.031-0.032.
- **Against StartLux-Decision's own server** on the 231 JevBench items (4B): the same decision on 231 (bf16) and 228
  (int8: three near-ties flip).
- **Bit for bit:** every GEMM config, split mode and stream-K gives the Triton kernel's bits, the wgmma kernels
  included (`tests/test_cuda_gemm.py`); prefix-cache hits equal recomputing (`tests/test_prefix_cache.py --exact`);
  64 answers sent by 32 clients at once equal the same requests one at a time (`tests/test_stress.py`). The bf16 path
  is deterministic run to run, but cuBLAS picks its kernels by batch size, so it is not batch-invariant.

## Serving

`POST /v1/systemone` (the TypeSafe / StartLux-Decision request and answer format), `GET /v1/models`, `/health`,
`/stats` (batches, prefix-cache hits and evictions).

The longest batch is sized from the GPU's memory at start-up, as vLLM's `--gpu-memory-utilization` and SGLang's
`--mem-fraction-static` do: after the weights and the prefix cache are resident, the engine takes the largest token
bucket whose buffers (activations, KV rows, recurrent-state scratch, per-batch tables; sized on the meta device) fit in
`--mem-fraction` (0.9) of the GPU less what is already in use and 1.5 GiB for graphs and workspaces: 65,536 tokens for
the 4B on the RTX 5090, 131,072 for the 27B on the H200; `--max-tokens N` sets it instead. Every bucket up to it has its
CUDA graph, recorded at start-up in under a second (`--graph-max-tokens` limits them, `--no-graphs` turns them off).
The scheduler packs requests into batches of up to `--budget` (8,192) tokens; a longer request runs alone in its own
bucket, and a request beyond the largest bucket is refused with HTTP 422.

Prefix cache: `--prefix-cache N` entries (64; 0 turns it off), `--prefix-cache-tokens` KV rows (65,536),
`--prefix-cache-min` the shortest evidence cached (64 tokens). Opt-in: `--attn-qk8` (q k^T on int8 tensor cores, 15%
off at 50,000 tokens, at a small probability error). `FLASHJEV_FLAGS` passes flags through `scripts/serve.sh`.

## Tests and benchmarks

On the GPU server, from the repository (`scripts/sync.sh` copies the working tree there):

```bash
M=~/jev/models/StartLux-Decision-4B-Q8_0-GGUF; JB=~/jev/src/jevbench-7ce310c7
python tests/test_engine.py --gguf $M/StartLux-Decision-4B-Q8_0.gguf --model-dir $M --jevbench $JB/datasets/public
python tests/test_stress.py --model-dir $M --jevbench $JB --start int8 [--quick] --out results/stress/flash-jev-int8.json
setsid bash scripts/stress.sh > ~/jev/logs/stress.out 2>&1 < /dev/null &     # every engine, one after another
python bench/report.py [--long | --stress]                                     # the tables
python bench/charts.py                                                        # docs/images
```

| Test | Checks |
|---|---|
| `tests/test_engine.py` | every precision against the fp32 reference forward (`--repeat`: run-to-run bits) |
| `tests/test_cuda_gemm.py` | the CUDA GEMMs against exact integer math and the Triton kernel's bits |
| `tests/test_w4a8.py`, `tests/test_kquant.py` | the W4A8 GEMM against its reference arithmetic; k-quant dequantization against gguf-py, bit for bit |
| `tests/test_prefix_cache.py --exact` | cache hits bit for bit |
| `tests/test_batch.py`, `tests/test_prompt.py` | the forest planner; prompts byte for byte against the chat template |
| `scripts/accuracy_v1.sh` | StartLux-Decision-v1's evaluation suites through every flash-jev precision and StartLux's own server (`bench/accuracy_tables.py`, `bench/compare_preds.py`) |
| `tests/test_stress.py` | a running server under load: long evidence counted in full, the capacity limit refused cleanly, 8k-token requests and JevBench from 8-32 clients without a failed request, bit-identical answers under load |

## Documents

| Document | Contents |
|---|---|
| [Benchmarks and stress tests](docs/benchmarks.md) | every model, GPU and engine: methods, tables, charts, many clients, long context, stress tests |
| [Optimization log](docs/optimizations.md) | where the time goes; what was kept, measured, tried and dropped |

## License

flash-jev's own code: to be decided by the owner. Vendored: `flash_jev/jevfmt.py` and `flash_jev/calibration.py` from
StartLux-Decision (Apache-2.0, `LICENSES/StartLux-Decision-Apache-2.0.txt`). Model weights are not part of this
repository; StartLux-Decision's weights are CC BY-NC 4.0 (non-commercial).
