# ROCm integration with upstream FP8 KV support

This update merges upstream main `27ad32d47ff36e12329692405dae4108e5a18241`
into the AMD PR, then brings in the official 9B projection tuning and benchmark
tools described in [the frozen qualification report](amd_r9700_9b.md).
The earlier graph-state and default-thought-budget fixes (#17 and #18) are now
part of upstream main.

## Conflict resolution

The conflicts were in tree and decode attention. Upstream added e4m3 KV storage
and scale arguments, while the AMD W4A16 path uses FP16 intermediate buffers.
The merged implementation retains the FP8 storage/read paths and passes the new
scale arguments through the engine and decoder. Attention probabilities are
cast to the actual value tensor dtype before the dot product: BF16 for upstream
CUDA/FP8 values, FP16 for ROCm W4A16. ROCm keeps its smaller attention tile,
including the readout path at head dimension 256.

FP8 remains disabled by default. Both the server and direct engine API explicitly
reject FP8 KV on ROCm because it has not been qualified here. This is a software
support boundary, not a claim about R9700 FP8 hardware capability. CUDA retains
upstream's sm_89-or-newer check and incompatible-option check. No NVIDIA runtime
test was possible on this host.

## Measurement scope

Integration checks use one Radeon AI PRO R9700 (`gfx1201`) with the locked ROCm
10.1 image, Torch `2.14.0+rocm10.1.0`, Triton
`3.8.0+git669b31ac.rocm10.1.0`, and HIP `7.16.26385`. GPU jobs run serially with
separate per-stage compiler caches. The complete vocabulary and generation head
remain resident; no CPU inference offload or second GPU is used.

## Fresh merged-source validation

The eight integration stages completed on code checkpoint
`194c3c16c39957df200c3d3e59ed7d65b8236ef4`. Later changes add documentation and results only.
The production-source SHA256 is `1a5a2474e25eb796ef635938ba1a1d7f438b7ec2a46b295c012d55b67238ba1a`
(sorted `startlux_prism` `.py`, `.cu`, `.h`, `.json` paths followed by their bytes).
[Machine-readable results](../bench/results/r9700_pr19_integration_20261026.json)
include per-round timings, source/runtime identity, comparison flags and evidence
manifest hashes. All main-stage telemetry samples identify only the owned GPU
processes; the initial GPU preflight was empty for every stage.

The 9B graph protocol uses 512 prompt tokens, eight warmup plus 64 measured greedy
steps, five rounds, and a maximum allocation of 16 sequences. W4A16 results:

| Active sequences | ms/step | Aggregate tokens/s | Tokens/s per sequence |
|---:|---:|---:|---:|
| 1 | 16.221 | 61.65 | 61.65 |
| 2 | 16.504 | 121.19 | 60.59 |
| 4 | 17.246 | 231.94 | 57.98 |
| 8 | 18.531 | 431.70 | 53.96 |
| 9 | 19.089 | 471.47 | 52.39 |
| 16 | 21.476 | 745.02 | 46.56 |

Every generated token and full-vocabulary logit snapshot (steps 0/8/32/71) matches
the frozen source byte-for-byte. All six W4A16 graph latency changes are within
0.2%. W4A16 eager B1/B8/B16 and 27B W4A16 graph B1/B8 also pass the frozen-output
comparisons; their latency changes are within 1%. The latter is a 27B model
regression check, not a new 27B service qualification.

Experimental W4A8 graph B1/B8/B16 yields
59.72/416.62/699.67
aggregate tokens/s. It passes the same-profile output comparison at all six batch
sizes, with latency changes within 0.4%. **W4A8 still fails the independent
probability gate** and is not combined with W4A16 accuracy into one claim.

Both 9B profiles exactly reproduce the frozen 501-case main and 20-case held-out
probabilities. W4A16 passes the existing MAE/P99/decision thresholds, with main
MAE `0.00005972788395`, P99 `0.000461230365`, and 100% clear-margin agreement.
The independent reference uses the same quantized GGUF weights. The 18 original
W4A16 operator-bound quarantines remain on their original fallback paths.

Operator tests cover 2,960 shipped-configuration reference checks, 384 packed-dot4
cases, 144 Q6 WMMA cases, and 1,600 W4A16 full-tensor checks with graph live-row
masking. Backend/scheduler tests pass (27 tests and 11 subtests). Both AMD profiles
pass teacher-forced decode checks, mixed/reordered request isolation, and
graph/cache reuse. The real server CLI rejects FP8 before opening a missing model
file, and the direct engine guard is covered by a unit test. The upstream FP8 GPU
test explicitly skips on ROCm; it is not reported as passed CUDA validation.

## Fresh HTTP replay and latency rechecks

All **1,840** measured requests reproduce the frozen semantic answers, thought
content and token usage exactly. The service matrix uses a default thinking mode
of `never` and explicit request modes for thinking scenes. The configuration
otherwise matches the eight-slot service below. All owned servers stopped after
the checks. The full-matrix measurements are retained:

| Scene | P50 ms | P95 ms | Requests/s |
|---|---:|---:|---:|
| readme3 | 61.45 | 62.03 | 16.122 |
| concurrency-8 | 478.77 | 567.36 | 16.604 |
| concurrency-16 | 1693.78 | 3214.26 | 11.199 |
| mixed-short-long-thought | 297.25 | 1633.32 | 14.509 |

Relative to the historical optimized source, concurrency-16 P95 increased 5.46%
and thought-16 P50 increased 6.67%, triggering the existing 5% recheck rule.
Their throughput changes were +21.08% and -0.60%, respectively. These flags are
preserved; the full matrix is not described as passing every latency threshold.

A separate five-round comparison reruns just these two scenes on the frozen
source and then the merged source, with identical bodies/configuration and
fresh per-stage caches. Each source completes 320 measured requests, and all
semantic outputs and token counts match. Changes from the contemporaneous old
source to the merged source are:

| Scene | P50 change | P95 change | Requests/s change |
|---|---:|---:|---:|
| concurrency-16 | +2.45% | +2.84% | -2.53% |
| thought-16 | +0.86% | not reported (80 requests) | -1.76% |

Neither original latency flag repeats in this focused old/new comparison. The focused run does not replace the original matrix or establish a
causal scheduling explanation. High-concurrency prefill/scheduling remains an
optimization target; no service speedup is attributed to this merge.

The earlier 1,800-second sustained run and five-run official-entry measurements
remain historical evidence in the 9B report. This update does not claim a fresh
sustained or NVIDIA hardware qualification.

## Reproduce the 9B configuration

Use the matching ROCm/Torch/Triton environment from [the AMD setup guide](amd.md).
Select one HIP ordinal after checking its physical GPU mapping. The `0` below is
an example, not a portable physical GPU index.

```bash
MODEL=/path/to/StartLux-Decision-v1.1-9B-AutoThink-GGUF
HIP_VISIBLE_DEVICES=0 python -m startlux_prism.server \
  --gguf "$MODEL/StartLux-Decision-v1.1-9B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --precision w4a16 --mmproj none --tp 1 \
  --max-tokens 8192 --context-length 8192 --max-think-budget 256 \
  --think-seqs 8 --think-kv-rows 67584 \
  --prefix-cache 4 --prefix-cache-tokens 4096 \
  --budget 8192 --overlap-tokens 1024 --graphs --graph-max-tokens 2048
```

For the state-reset decode measurement, use a new output directory:

```bash
HIP_VISIBLE_DEVICES=0 python bench/amd_model_ab.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-9B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --precision w4a16 --output /tmp/prism-9b-decode \
  --batches 1,2,4,8,9,16 --modes auto --context 512 --max-tokens 8192 --repeats 5
```

The benchmark writes all generated tokens and four complete-vocabulary logit
snapshots per sequence. `--baseline /path/to/frozen-output-directory` adds the
byte comparisons used here; `--eager` tests the uncaptured decode path. Fixed-batch
throughput is aggregate output across all active sequences, not speed per user.
