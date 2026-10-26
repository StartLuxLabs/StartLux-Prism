# MI355X next-version execution results

Date: 2026-10-26. Branch: `feat/mi355x`.
Plan: [native MXFP4 first](mi355x_next_version_speedup_plan.md).

This is the incremental execution record. Each completed part adds its evidence,
measured outcome and remaining limitations. Projection results and model-level
latency/quality are reported separately.

## Baseline

The tuned INT8 implementation is `4534c11`; the approved work plan is `afeab60`.
Its recorded `readme3` HTTP p50 is 20.54 ms. See the
[INT8 performance report](mi355x_performance.md) for all five suites and raw data.
At the start of this iteration, the target was 12 ms and no FP4 model-level gain
had been established. The completed measurements appear below.
The [frozen manifest](../bench/results/mi355x_fp4_frozen_20261026.json) records
the Q8 file/calibration/tokenizer hashes, seven exact HTTP request bodies and
the original BF16 revision and shard hashes.

## Part 1: native MXFP4 instruction feasibility

**Completed:** a new experimental `kernels/mxfp4.py` packs E2M1 payloads and E8M0
scales and executes `tl.dot_scaled` on the existing MI355X stack. Disassembly
contains `v_mfma_scale_f32_16x16x128_f8f6f4`, with `cbsz:4 blgp:4` selecting FP4
operands. This establishes native matrix execution, rather than a software
dequantization followed by BF16 GEMM. It does not establish model accuracy.

Stack unchanged: PyTorch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, Triton `3.8.0`,
GPU `gfx950`. Tests use only `HIP_VISIBLE_DEVICES=0`; foreign jobs are left alone.

The original BF16 model source is
[`startlux-models/StartLux-Decision-v1.1-27B-AutoThink`](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink/tree/dd62d5cd84a483d75bdd548880bbf08bdd3f67a4),
pinned at `dd62d5cd84a483d75bdd548880bbf08bdd3f67a4`. All 14 shards downloaded
and their complete SHA256 hashes verified against the pinned upstream identities.

Independent validation: **13 tests passed**, covering FP32/BF16/FP16 encoding,
midpoint ties, signed zero, non-finite blocks, three K sizes and three native
GEMM configurations, partial output tiles and changing live counts under graph
replay. Products are checked against a CPU FP64 reference on independently
quantized operands, with an accumulation-error bound.

Reproduce: `HIP_VISIBLE_DEVICES=0 python3 -m pytest -q tests/test_mxfp4.py`.

## Part 2: initial streaming projection measurements

**Completed:** the first six-configuration sweep uses 144 live rows in a
192-row bucket. Each weight rotation exceeds 512 MiB, twice the chip's Infinity
Cache. Timings include dynamic activation quantization, packing and scales.
These are synthetic projection measurements, not whole-model speedups.

| Projection | Best GEMM | Quantization + GEMM | Useful FP4 TFLOPS, complete path |
|---|---:|---:|---:|
| MLP gate/up | 55.68 µs | 58.86 µs | 872 |
| MLP down | 61.54 µs | 70.47 µs | 364 |
| GDN input | 33.84 µs | 38.05 µs | 639 |
| GDN output | 23.60 µs | 28.46 µs | 318 |

The [initial sweep data](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_fp4_initial_20261026.json) records
all six configurations. Timings are the lowest of five warmed graph timings,
used for candidate selection rather than release p50 estimates. Initial kernels use
ordinary row-major payloads/scales. MLP down has a long K and warrants further
layout/tile work. The earlier INT8 105 µs probe used 192 live rows, so it is not
a matching useful-work comparison with this 144-live-row sweep.

Next: matched INT8 comparisons, more tile/layout candidates and original-BF16
model quality/latency checks. The model-level 1.5× intermediate gate is open.

## Part 3: original-weight model integration and first ablation

**Completed:** original safetensors weights now have a text-only experimental
loader that converts GDN head layouts to Prism's order and preserves original
BF16 embedding/readout weights. Native FP4 projections run inside the existing
graph-enabled engine. No Q8/Q4 weights are used to construct FP4 matrices.

The initial comparison uses the same five rendered request suites, including
the HTTP none-option setting. It uses the initial conservative FP4 tile for all
shapes; it does not yet use the best shape-specific tiles from the sweep.

| Precision / projection coverage | `readme3` engine p50 | Fixed-suite decisions agreeing with original BF16 |
|---|---:|---:|
| Original BF16 | 20.23 ms | 10/10 |
| FP4 gate/up only | 21.54 ms | 10/10 |
| FP4 down only | 22.22 ms | 10/10 |
| FP4 MLP | 23.53 ms | 10/10 |
| FP4 all large projections | 24.46 ms | 10/10 |

**Outcome:** the initial model candidate is slower and is not a speedup to ship.
All tested repeated runs are deterministic. The all-projection FP4 candidate
passes exact serial-versus-alone and mixed-request logit checks for `readme3`;
the BF16 library route and hybrid candidates vary with batch geometry and do not
pass that exactness check. The all-FP4 candidate's largest probability change
against BF16 is about 0.1604 on this small suite. Ten decisions cannot establish
task accuracy parity or qualify a default precision change.

Data: [initial model ablation](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_fp4_model_initial_20261026.json). Engine time must not be compared
directly with the historical HTTP p50. Next: select and validate shape-specific
FP4 kernels, repeat matched end-to-end timing, and evaluate decision quality on
a larger labeled suite. Current INT8 remains the default validated profile.

## Part 4: qualified FP4 dispatch and model improvement

**Completed:** selected **49** exact M/N/K dispatch entries across seven token/
read buckets from 304 streaming candidates, plus a 20-candidate MFMA32 probe.
Every selected shape passes an independent GPU FP64 product reference, partial
row/output-stride guards and BF16 graph replay checks. The table is pinned to
GPU, PyTorch/HIP/Triton versions and the native kernel source hash; unknown
shapes/stacks retain the initial implementation. `STARTLUX_PRISM_MXFP4_GEMM=reference`
selects that implementation for comparison.

| HTTP-matched request rendering | Original BF16 engine p50 | Tuned native FP4 engine p50 |
|---|---:|---:|
| `yesno` | 14.54 ms | 11.25 ms |
| `review1` | 16.78 ms | 12.58 ms |
| `readme3` | 20.36 ms | 14.86 ms |
| `startlux3` | 33.25 ms | 26.88 ms |
| `longpage2` | 89.41 ms | 73.62 ms |

The `readme3` FP4 result improves from 24.46 ms in the initial ablation to
14.86 ms, and is about 1.37× faster than the BF16 reference in this run. The
10/10 fixed-suite decisions, deterministic repeats and checked exact batch
isolation are retained. These are engine measurements over 30 warmed calls,
not HTTP release qualification; the 12 ms HTTP target is still open.

AITER source `be0237f1f9fab0d01707c064380a917bbd2bfbb8` was also probed with
the same operands. Its default Triton kernels did not improve the dominant
gate/up projection, and default Gluon kernels were substantially slower on
these shapes. They were not selected. ASM remains a separate candidate.

Next: matched current-INT8/FP4 HTTP runs, labeled quality evaluation, then GDN
and remaining projection/quantization bottlenecks.

## Part 5: labeled public quality screen

**Completed:** the original BF16 and initial `ceil`-scaled all-projection FP4
profiles were evaluated through the same decision adapter/scorer on all 231
public JevBench tasks, source revision `7ce310c7262ed49cc85853339a8a42459298e3f3`.
No thoughts or cross-request prefix-cache hits were used.

| Tier | Tasks | Original BF16 correct | Native FP4 `ceil` correct |
|---|---:|---:|---:|
| Easy | 48 | 48 | 48 |
| Original | 72 | 71 | 71 |
| Hard | 111 | 94 | 92 |
| Total | 231 | 213 (92.21%) | 211 (91.34%) |

**Outcome:** the candidate fails the proposed quality budgets: aggregate loss
is 0.87 percentage points, and hard-tier loss is 1.80 points. It remains
experimental. This public suite is a useful screen with limited sample size;
it does not establish quality across all production decisions.

Next: compare an independently tested per-block reconstruction-error scale
recipe against the original recipe. Both the failed candidate and subsequent
results remain in this record. Do not silently relax the quality budget.

## Part 6: freeze a better FP4 recipe

**Completed:** two further recipes were screened on the same public suite.
Choosing the lower reconstruction-error scale scored 209/231; restricting the
original recipe to MLP projections scored 208/231. Neither was selected.

A paired, normalized 32-value Hadamard transform is applied to both the weight
blocks and activation blocks before FP4 quantization. Before quantization these
orthogonal transforms preserve each block's dot product; they spread large
values within a block. Weight transforms run offline, and activation transforms
are included in the timed quantization kernel.

**Selected experimental recipe:** all large projections, `ceil` E8M0 scales,
paired 32-value rotation. It scores **213/231**, matching BF16's easy/original/
hard counts **48/48, 71/72 and 94/111**. It preserves the checked exact batch
isolation and deterministic repeats. The public screen passes the provisional
empirical loss budget; larger independent quality qualification remains open.
The repeated public screen is now development evidence, not an untouched test.

This recipe runs at about 15.47 ms on `readme3` before subsequent layout/GDN
work. Lower quantization MSE alone was not a reliable decision-quality guide.

## Part 7: offline artifact and scale layout

**Completed:** `python -m startlux_prism.tools.convert_mxfp4` generated an
all-projection rotated FP4 artifact from the original BF16 source: **64 layer
files, 12,936,273,920 bytes**. `mxfp4.json` records source config/index identities,
quantizer identity, rotation/scaling recipe, shapes and per-file SHA256 hashes.
The runtime checks these before loading the cached projections. Original BF16
source files remain necessary for embeddings, readout and other unquantized
weights; artifact bytes alone are not the full resident-model size.

The native GEMM now also accepts an MFMA-oriented scale layout. Activation
quantization writes that layout directly; static weight scales are prepared
once. Canonical scales remain available for independent validation and fallback,
so resident storage includes an auxiliary scale copy. All 49 dispatch shapes
were requalified after this change. On the rotated model, `readme3` measured
15.25 ms and the fixed-suite exact isolation checks still passed.

Reordering workgroups to improve XCD locality produced small, shape-specific
wins. The selected ordering remains explicit in each qualified table entry.

The optional AITER ASM probe matched the native product exactly, but full-path
timings were about 60.4 µs for gate/up and 69.0 µs for down, including runtime
scale-layout conversion. It was not selected for the runtime.

## Part 8: shared GDN optimization with exact INT8 outputs

**Completed:** five paired rounds, 100 measured calls per suite/mode/round,
varying only GDN launch parameters on the current tuned INT8 model. All **5,000
measured requests** retained exact logits. Pooled engine p50:

| Workload | Previous GDN | Selected GDN | Speedup |
|---|---:|---:|---:|
| `yesno` | 15.612 ms | 14.968 ms | 1.043× |
| `review1` | 18.007 ms | 17.245 ms | 1.044× |
| `readme3` | 20.336 ms | 19.735 ms | 1.030× |
| `startlux3` | 34.172 ms | 29.940 ms | 1.141× |
| `longpage2` | 95.796 ms | 80.019 ms | 1.197× |

Selected `gfx950` launch parameters: `INTRA_WARPS=4`, `HREC_BV=64`, `OUT_BV=64`,
`HREC_NS=1`, `OUT_NS=1`. These now replace the conservative Radeon-derived
MI355X defaults; Radeon and NVIDIA profiles remain separate.

A recurrence experiment with two pipeline stages aborted in LLVM on the pinned
stack (virtual-register dominance errors). It was rejected. Recurrence stays
at one stage; later tuning runs exclude the failing stage parameter.

## Part 9: actual request geometry and rejected GEMM experiments

The matching HTTP none-option settings compute **155 live tokens in a 192-row
bucket** for `readme3`, 372/384 for `startlux3`, and 1194/1280 for `longpage2`.
The earlier 144/192, 361/384 and 1185/1280 probes are synthetic/pre-none-option
geometries. Their mask tests cover full buckets, but their useful-operation
rates must not be presented as exact HTTP workload measurements.

The shape-specific FP4 table was requalified after adding the scale layout and
workgroup ordering. A second matched rotation-enabled sweep improves selected
384-row shapes. Very wide M tiles and a shared-weight 128+tail prototype did
not improve the primary projection and were rejected. The tail prototype is
retained as an [archived research script](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/mi355x_mxfp4_tail_probe.py), outside the runtime and current source tree.

The native precision remains experimental. The primary 12 ms HTTP target and
broader independent quality qualification are still open; no theoretical peak
ratio is being presented as a measured whole-model gain.

## Part 10: regression, cache and stability qualification

**Completed:** **57 tests and 11 subtests pass**, including the original focused
MI355X suite, new independent FP4 encoding/arithmetic tests, dispatch guards and
artifact source/checksum rejection tests.

The selected artifact matches direct quantization exactly on all five fixed
suites, including complete letter logits. Eager and graph results, question
reordering, isolated/mixed requests, shared/unshared evidence, and prefix-cache
miss/hit/eviction checks pass exactly within this FP4 profile.

**10,000 further graph-enabled model requests pass** with identical complete
outputs and unchanged allocated device memory: 19,993,782,784 bytes before and
after the stability pass. Peak allocated memory during loading was
55,189,419,520 bytes; the current loader first materializes original BF16
projections before replacing them, so offline packing does not yet minimize
startup peak memory.

The selected HIP ordinal is `0`, which maps on this host to PCI `0000:75:00.0`
(management-tool GPU 3). HIP ordinals and management-tool indices differ.

## Part 11: matched HTTP results and service checks

**Completed:** three alternating server rounds, 100 measured requests per
suite/profile/round after warmup: **4,500 HTTP measurements**. All profiles use
the same frozen request bodies, no thoughts, no cross-request prefix cache,
the same runtime image and one visible MI355X. INT8-next changes only the GDN
launch defaults; its complete answers/probabilities and usage match INT8-stock.
FP4 uses the selected original-weight rotated recipe and offline artifact.

| Workload | Previous tuned INT8 p50 | Next INT8 p50 | Native FP4 p50 | FP4 speedup vs previous INT8 | FP4 p95 |
|---|---:|---:|---:|---:|---:|
| `yesno` | 15.98 ms | 15.39 ms | 10.86 ms | 1.47× | 10.95 ms |
| `review1` | 18.31 ms | 17.63 ms | 12.41 ms | 1.47× | 12.51 ms |
| `readme3`, primary | 20.66 ms | 20.15 ms | **14.56 ms** | **1.42×** | 14.68 ms |
| `startlux3` | 34.60 ms | 30.42 ms | **22.03 ms** | **1.57×** | 22.12 ms |
| `longpage2` | 95.95 ms | 80.51 ms | **56.96 ms** | **1.68×** | 57.29 ms |

Both next-INT8 and native-FP4 servers also pass **32 requests from eight
concurrent clients**, with exact answers/probabilities and usage versus serial
execution within their respective precision profiles. These service checks
are separate from the latency samples.

The 23 ms `startlux3` and 65 ms `longpage2` targets are met. The 10 ms `yesno`,
11 ms `review1` and **12 ms primary `readme3` targets remain open**. The primary
1.5× intermediate goal is also open. FP4 is still opt-in: the public accuracy
screen matches BF16's total and tier scores, but six of 231 public argmax
decisions differ, and broader independent quality qualification is unfinished.

The [compact summary](../bench/results/mi355x_next_summary_20261026.json) remains
in the source tree. The [evidence index](../bench/results/README.md) describes the
workspace archive; raw links below point to the immutable published snapshot:

- [HTTP samples, p50/p95 and responses](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_next_http_20261026.json).
- [Paired exact INT8 GDN samples](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_gdn_int8_ab_20261026.json).
- [Original BF16 and rotated FP4 public predictions](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_fp4_quality_r32_20261026.json).
- [10,000-request graph/cache/isolation validation](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_fp4_validation_20261026.json).
- [Next-INT8 concurrent API check](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_next_concurrency_int8-next_20261026.json)
  and [FP4 concurrent API check](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_next_concurrency_mxfp4-r32_20261026.json).
- [Versioned FP4 artifact manifest](https://github.com/zihaomu/StartLux-Prism/blob/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench/results/mi355x_fp4_artifact_manifest_20261026.json),
  including full source-shard hashes. Runtime checks cover source weights as
  well as the FP4 payload files; this adds startup work, outside warmed latency.

Reproduce the HTTP comparison after building `startlux-prism:mi355x-next`:

```bash
ARTIFACTS=../artifacts/startlux-prism/2026-10-26
M=../models/StartLux-Decision-v1.1-27B-AutoThink
HIP_VISIBLE_DEVICES=0 python3 bench/mi355x_next_http.py \
  --model-dir "$M" --gguf StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf \
  --calib "$ARTIFACTS/results/mi355x-27b-calib.pt" \
  --mxfp4-artifact ../models/StartLux-Decision-v1.1-27B-AutoThink-MXFP4-R32 \
  --manifest bench/results/mi355x_fp4_frozen_20261026.json \
  --rounds 3 --reps 100 --out "$ARTIFACTS/results/mi355x-next-http.json"
```

The remaining speed work should start with a fresh profile of the selected
rotated FP4 stack. Prioritize projection/code-generation improvements and
producer/quantization fusion using the actual HTTP live counts. Retain the
failed tile/tail experiments as evidence; larger tiles alone did not solve
the primary workload. Do not enable FP4 by default until the remaining gates
are satisfied.

## Part 12: PR packaging and workspace cleanup

**Completed:** moved 18 raw captures and five rejected research scripts out of
the source tree, together with local logs, traces, calibration files and caches.
The workspace archive is `../artifacts/startlux-prism/2026-10-26/`; its manifest
records paths, sizes and SHA256 hashes for 351 relocated files. The 23 formerly
tracked files match the published source snapshot byte for byte. Compact
summaries, the frozen workload, runtime tables and supported benchmark tools
remain in the repository. See the [evidence index](../bench/results/README.md).

The HTTP comparison now accepts explicit model, calibration and FP4 artifact
paths and mounts those inputs individually. A simulated two-round run checked
six server launches and 30 measurement groups, including paths with spaces,
alternating precision order, stable responses and invalid-input rejection.
Seventeen artifact-integrity/backend/dispatch regression tests pass on the
matched image without GPU access. Local documentation targets and archive
checksums also pass.

At this packaging stage, runtime, kernel, test and serving-script sources matched
the measured snapshot; performance and quality results were retained from that
qualification. This step did not add a new GPU latency measurement.

## Part 13: MI355X review scope and revalidation

**Completed:** the previous 106-file diff included 61 files overlapping the
unmerged Radeon PR #19, including 36 unchanged files. Removing raw captures had
reduced added lines but retained the Radeon implementation, reports and tools.

Another **35 files** are preserved byte for byte in
`../artifacts/startlux-prism/2026-10-26/review-scope/`, with a checksum manifest
and the published `d84a7c7` snapshot. These include Radeon-only dispatch/tuning
kernels, two `gfx1201` tables, their tests, Radeon reports/tools, five historical
INT8 experiment drivers and the generated performance figure. Radeon support
remains separate in PR #19. The runtime now rejects ROCm architectures other
than `gfx950` and keeps MI355X's existing portable W4A8/W4A16 choices.

The remaining **71 files** comprise 28 runtime files, 13 test/reference files,
13 benchmark/evidence files, seven serving/validation scripts, four documents
and six root setup/documentation files. All **69 retained Triton JIT function
bodies** are byte-identical to the previous snapshot; native FP4/INT8 dispatch
identities and the qualified `gfx950` settings are preserved.

Revalidation on MI355X passes **50 tests and 11 subtests**, including native FP4,
exact INT8 graph replay, dispatch/source guards, artifact integrity, batching,
warm-up, thought budgets and lazy decode capture. Independent Q4_K/Q6_K GPU
arithmetic and both epilogues pass, six CPU dequantization cases match the
optional pinned `gguf==0.19.0` oracle, and **36 W4A16 configurations** pass the
independent product/mask checks.

The actual 4B Q4 model passes eager/graph, sharing, cache miss/hit/eviction and
mixed-batch checks for both W4A8 and W4A16 after removing Radeon dispatch. These
are functional checks on shared hardware; their single-sample timings are not
new release latency results. Logs and model reports are kept in `review-scope/`.
The earlier matched 27B latency/quality results and remaining FP4 gates stand
as recorded above.

## Part 14: current-runtime FP4 rebaseline

**Completed:** 2,500 warmed HTTP calls using runtime `14d6e27`, image
`sha256:284183e504fb7cee0c643da0aec18e13b009b46f5032d6620098eacfa704b93d`.
Five server rounds each measured 100 calls per suite after 20 warmups, with the
same frozen payloads, selected H32/ceil FP4 artifact, no thoughts and no
cross-request prefix cache. All five suites' complete answers/probabilities and
usage match the earlier FP4 capture exactly, including across the new rounds.

| Workload | FP4 p50 | FP4 p95 | p50 target | Status |
|---|---:|---:|---:|---|
| `yesno` | 10.902 ms | 10.936 ms | 10 ms | Open |
| `review1` | 12.496 ms | 12.578 ms | 11 ms | Open |
| `readme3` | **14.693 ms** | **14.744 ms** | **12 ms** | **Open: 2.693 ms / 18.3% reduction needed** |
| `startlux3` | 21.990 ms | 22.126 ms | 23 ms | Met in this run |
| `longpage2` | 57.135 ms | 58.102 ms | 65 ms | Met in this run |

This run uses HIP ordinal **1**, PCI **`0000:05:00.0`**, management index **0**;
the historical comparison used another physical MI355X. This is a fresh FP4
baseline, not evidence of an additional runtime speedup. The selected GPU had
no processes before launch. Later telemetry contains 44 samples with only the
benchmark allocating VRAM, plus an additional zero-byte context; an exclusive
GPU reservation was not established. Round-median spreads remain below 2%.
Future candidate claims still require paired comparisons on the same GPU.

For `readme3`, median GPU-stream elapsed time is **14.217 ms** and the median
per-request client time minus stream elapsed time is **0.487 ms**. The latter
includes CPU, scheduling and transport costs; it is not a pure CPU profile.
Even eliminating that entire exposed remainder would not close the primary
gap. Stream time includes transfers and is not matrix compute time.

The HTTP tool now records every request's timing fields, immutable image IDs,
workload/harness hashes, pooled p50/p95 and per-round medians. Its optional
`--candidate-image` mode compares two images under the same FP4 recipe, alternates
order and rejects changed answers/probabilities or usage. A simulated two-round
A/B check verifies launch order, image pinning, raw timings and aggregation.

Raw data and telemetry: `../artifacts/startlux-prism/mi355x-pipeline-study/s0-20261026/`.
The `pipeline_study_20261026` entry in the
[compact summary](../bench/results/mi355x_next_summary_20261026.json) records
sample counts, geometry, image identity, per-round medians and artifact hashes.

## Part 15: matching FP4 profiling and first stage attribution

**Implemented and exercised on MI355X:** `bench/profile.py` accepts the offline
FP4 artifact/recipe and frozen HTTP bodies, preserves evidence boundaries and
v1.1 none options, separates body/head graph events, exports a graph trace,
and attributes FP4 activation quantization and GEMM by phase and shape.
The old INT8-only detail hook and legacy synthetic rendering remain available
with explicit timing scope. Multi-pass and image requests are rejected in the
new one-pass manifest mode instead of silently profiling different work.

CPU checks compare all five rendered groups against the actual Decider path,
including identical tokens and evidence boundaries. The real `readme3` run
matches HTTP's **155 live tokens / 192 body rows / three reads / four head rows**.
The selected table has all **49 qualified entries**; the H32/ceil recipe and
runtime kernel source identities are unchanged.

Thirty graph replays yield a **14.280 ms** median, with body/head medians of
**14.127 / 0.152 ms**. One separate diagnostic graph trace contains 1,034 kernels
on one stream:

| Kernel family | Summed kernel duration | Share of traced kernel duration |
|---|---:|---:|
| Native FP4 GEMM, 257 calls | 7.719 ms | 53.4% |
| H32 activation quantization and packing, 257 calls | 2.179 ms | 15.1% |
| GDN intra/recurrence/output/preparation/gated norm | 3.246 ms | 22.4% |
| Other kernels | 1.324 ms | 9.2% |

The trace's kernel sum is 14.468 ms and its first-to-last kernel span is
14.628 ms, about 2.4% above the median graph replay without tracing. These
single-trace family totals are diagnostic evidence, not independently repeated
stage benchmarks. Eager event hooks report larger durations because host launch
gaps enter the intervals; those numbers are not the graph critical-path budget.

The next candidates are producer/rotation/packing fusion, shape-specific FP4
projection efficiency and GDN. Tiny-head/readout or host-only changes cannot
explain the complete 2.693 ms gap at their measured sizes. Actual-shape FP4
bandwidth/compute ceilings, full five-suite stage attribution, paired candidate
wins and broader held-out quality remain open in the
[study plan](mi355x_next_version_speedup_plan.md).

Traces and shape-level timings are under
`../artifacts/startlux-prism/mi355x-pipeline-study/s1-20261026/`, indexed by the
compact summary. Reproduction commands and timing definitions are in the
[evidence index](../bench/results/README.md). This update changes measurement
tools and documentation; it introduces no additional inference-kernel speedup.
