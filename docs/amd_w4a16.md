# Experimental compressed W4A16 on R9700

This opt-in profile uses the same 16.55 GB v1.1 27B AutoThink Q4_K_M checkpoint
on one Radeon AI PRO R9700 (`gfx1201`). It passes the local frozen FP32 readout
thresholds and the service checks below. Several speed targets remain unmet;
this is not general AMD or production qualification. The separate W4A8 profile
and its unmet FP32 gate are unchanged. CUDA hardware regression remains open.

## Select the profile

Use the [ROCm environment and single-GPU launch instructions](amd.md), changing
`--precision w4a8` to `--precision w4a16`. Automatic precision selection remains
W4A8 for Q4_K_M. All weights stay on the selected GPU in compressed form; no
full-model floating-point copy or CPU offload is used during inference.

Q4/Q6 projections locally dequantize weights to FP16 and use FP16 activations
with FP32 accumulation. The selected intermediate buffers use FP16, and embedding
gather preserves the original compressed embedding rather than converting it to
Q8. The full generation head still uses Q8 weights and activation quantization.
This changes numerical semantics; it is not a bit-equivalent W4A8 optimization.

`STARTLUX_PRISM_AMD_W4A16=auto` loads the offline table when runtime, source,
shape, dtype, stride and alignment match. Unknown layouts or environments use
the original W4A16 kernel. `reference` forces that original implementation.
Restart or recapture graphs after changing the mode. The measured profile uses
`STARTLUX_PRISM_AMD_Q_COMPONENTS=all` (the default); the other component switches
are research ablations without the complete service qualification below.

## Correctness and table selection

The original metadata screen checked all 300 real shape/layout buckets and
accepted 293 under the cold/hot performance gates. A bounded small-row Q6 search
then retested 15 affected layouts, including 51 graph/live-row checks, and
produced a 299-entry final table. The 45 unchanged small Q4 and 240 larger layouts
reuse their original evidence; they are not described as rerun. One small-N Q6
M=32 layout retains the original implementation.

Every admitted shape met at least 10% cold/hot improvement and, where applicable,
5% partial-row improvement against its current same-precision baseline. The final
ported code passed 688 independent-dequantization/full-output checks, including
padding, nonuniform prior output, FP32/FP16/BF16 storage, K=17408 and graph masks.
Selected code objects use `v_wmma_f32_16x16x16_f16`, with maximum VGPR 148 and zero
scratch/spills. Static ISA does not establish dynamic utilization.

The unchanged 501-case FP32 oracle gave MAE 5.3225201486e-05,
P99 0.000440378584225, and 457/457 clear-margin decisions.
The 20 additional frozen cases gave MAE 1.52802480331e-05,
P99 7.55903740435e-05, and 20/20 decisions. Every probability
matched the original W4A16 profile. These sets are numerical checks, not a graded
model-capability benchmark. Thresholds remain MAE ≤0.001, P99 ≤0.01 and clear-margin
agreement ≥99.5%. The public `scripts/amd/validate_reference.py --precision w4a16`
entry was used for both final comparisons; it retains the frozen prompt checks.

## Decode and readout

The final integration uses 512 prompt tokens, 8 warm steps and 64 fixed timed
steps, ignoring EOS, with five runs per mode/batch and a resident generation head.
Every token matches the research pair and all eight individually run sequences.

| Mode | Batch | Final total tokens/s | Per-sequence tokens/s |
|---|---:|---:|---:|
| Eager | 1 | 20.451 | 20.451 |
| Eager | 4 | 77.934 | 19.484 |
| Eager | 8 | 146.410 | 18.301 |
| Graph | 1 | 20.368 | 20.368 |
| Graph | 4 | 78.193 | 19.548 |
| Graph | 8 | 146.969 | 18.371 |

The same-resident-model five-pair comparison for the small Q6 extension measured
eager B1 18.207→20.350, B4 69.774→77.682, and B8 131.865→146.110 total tokens/s.
Both sides already use the metadata implementation; these are not comparisons
against W4A8 or the original slow W4A16 prefill. Separate graph capture recorded
changed code-object hashes and all tokens remained equal. B1 is below the local
25 tokens/s target.

The first metadata implementation reduced same-precision engine readout from
1513.577 to 151.304 ms for readme3, and 80608.565 to 7495.632 ms for 8192 tokens.
The later small Q6 extension initially produced an 8K median regression of 8.63%.
All slow samples are retained. A six-pair rerun measured 7478.789→7474.041 ms,
with no pair above a 0.11% slowdown; actual large-M code hashes were identical.
The original slowdown's cause is not established. Engine timings do not replace
HTTP measurements.

## Complete HTTP protocol

The final profile ran the same enhanced protocol as W4A8: unique evidence IDs,
resident generation head, explicit 8K readout capacity, three rounds, 102 requests
per common scene and 30 per long scene. The following values are medians of the
three round statistics, in milliseconds. No cross-precision speedup percentage
or numerical equivalence is inferred from this table.

| Scene | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| yesno | 103.749 | 104.074 | 104.084 |
| readme3 | 154.879 | 155.486 | 156.318 |
| startlux3 | 291.884 | 292.693 | 293.525 |
| review1 | 108.679 | 109.457 | 109.582 |
| wide33 | 305.745 | 307.628 | 309.578 |
| longpage2 | 920.178 | 921.466 | 921.466 |
| 256 | 198.156 | 199.906 | 200.419 |
| 2048 | 1560.769 | 1691.459 | 1697.549 |
| 8192 | 7519.204 | 7547.067 | 7547.067 |

All 21 concurrency rounds had zero errors. At concurrency eight, drained
throughput was 6.040 requests/s and median P95 was
1458.369 ms. The original W4A16 full HTTP baseline was not run;
the 30% same-precision HTTP improvement target is therefore not assessed.

| Client concurrency | Requests/s | P95 ms |
|---|---:|---:|
| 1 | 5.270 | 297.257 |
| 2 | 5.370 | 586.959 |
| 4 | 5.930 | 781.661 |
| 8 | 6.040 | 1458.369 |
| 16 | 5.960 | 2809.319 |
| 32 | 6.040 | 5658.768 |
| 64 | 6.000 | 13782.310 |

Actual AutoThink used ten requests per budget; a budget is not a generated length.

| Maximum budget | Actual output-token range | HTTP P50 ms |
|---|---:|---:|
| 16 | 16–16 | 1244.720 |
| 64 | 52–64 | 3619.149 |
| 256 | 48–89 | 3645.023 |

## Sustained service and startup

The standard 2K configuration, with full generation head and projector, completed
1800.087 seconds and 7660 requests with 0 unexpected
errors. It included 515 thought requests, 145
image requests, 4716 exact text-control checks and
2649 sustained cache evictions. Drained allocated growth
was 0 bytes; the observed device-memory peak was
22.593 GiB.

The complete graph/cache/shared/reordered question, seeded decode and visual
checks passed in the ported tree. Small-image coverage is 1/2/8 images, including
mixed batches and eviction; it does not qualify eight maximum-resolution images.
The 8K readout check does not qualify 8K AutoThink. Teacher-forcing top-1 agreement
is not a claim that every logit equals the prefill reference.

The sustained latency analysis retains every request, split into four equal
request-count windows (not equal wall-time windows). The first and last windows
are reported by category to avoid mixing text, cache and thought costs.
One mixed-load run does not establish a repeatable or causal regression.

| Category | First-window P50 / P95 ms | Last-window P50 / P95 ms |
|---|---:|---:|
| cache | 472.818 / 766.331 | 472.652 / 759.675 |
| image | 580.148 / 1219.901 | 579.713 / 1216.769 |
| image-thought | 2194.733 / 4098.263 | 2190.195 / 3291.259 |
| text | 472.462 / 757.822 | 472.532 / 757.410 |
| thought | 1880.513 / 2381.508 | 1878.208 / 2373.383 |

- cold-jit: readiness 60.385 seconds, compiler objects 0→225.
- hot-start: readiness 19.063 seconds, compiler objects 225→225.

The OS file cache was not flushed, so cold JIT is not cold-disk startup. These
startup requests use standard 2K settings and can reuse prefixes. The documented
workspace entry was also exercised with actual readout and thought requests,
then stopped normally. Final qualified stages' runtime/server logs showed no fault matches;
host kernel logs were inaccessible. Dynamic WMMA utilization/model DRAM traffic
and a same-workload RTX 4090 comparison remain unmeasured.

## Unmet targets and reproducibility

- `eager_B1_total_tokens_s`: measured 20.4508; target min 25.
- `8192_HTTP_P50_ms`: measured 7519.2; target max 6900.

Keep the same model, selected GPU and locked Torch/Triton pair when reproducing.
Add `--precision w4a16` to the graph/cache, decode-model, vision, HTTP and sustained
commands in [the AMD guide](amd.md). For fixed decode, use
`bench/amd_decode.py --precision w4a16` without `--paired`; the latter currently
compares W4A8 implementations only. The independent operator check is
`python tests/test_amd_w4a16_meta_extended.py` with one ROCm GPU visible.
`bench/amd_decode_validation.py --precision w4a16` reproduces the four-prompt,
four-seed teacher-forcing and reverse-batch protocol; add `--graphs` for that mode.
`--gguf`, `--model-dir` and `--output` select its inputs and result file.
`scripts/amd/test_readout_isolation.py --precision w4a16` additionally checks
eight individual questions against mixed/reordered batches, sharing and graphs.

The [compact evidence](../bench/results/r9700_w4a16_20261026.json) preserves
round-level measurements, original slow samples, selection decisions, shape
samples, source/runtime identities and every target assessment. The two existing
bugfix dependencies remain separate: #17 and #18 must be applied to this backend
for the complete graph/AutoThink qualification. Other architectures, models and
runtime combinations need their own qualification.
