# Experimental gfx1201 validation

This page preserves the original prototype and trimmed-backend checks. The later
kernel changes and final integration measurements are recorded separately in
[the acceleration report](amd_acceleration.md).

## Scope and identity

- Radeon AI PRO R9700, gfx1201, 32 GB, one visible GPU per process.
- Workspace-provided ROCm 10.1.0 / HIP 7.16.26385; Python 3.13.13.
- Standalone vision reference: Transformers 5.16.1; independent block oracle: gguf 0.19.0.
- Torch `2.14.0+rocm10.1.0`; paired Triton
  `3.8.0+git669b31ac.rocm10.1.0`.
- Main GGUF SHA-256:
  `245f89a8fbd95d0fccae02aba1a1a63448bab41c308920ad6de247e3b3752588`.
- Projector SHA-256:
  `b6ff3c746a642ac44ea14f679f3a6e788a8b7c2cb4474378fa895e86f1b937eb`.

The environment came from an immutable local foundation image; no public image
availability or other PyTorch/ROCm combinations are claimed. NVIDIA execution
has not been tested here. The original checks below ran while other jobs were
active; the acceleration report separately records whether the selected GPU
itself was shared.

## Original prototype evidence

The broader prototype predates the trimmed upstream patch and includes local
launchers and service features that are deliberately not part of this PR.
These results are supporting context, not a claim that those auxiliary features
are in the patch or a substitute for rechecking its exact integration checkout.

| Check | Prototype result |
|---|---|
| FP32 add/multiply/divide/FMA/rint | 1024 values per operation equal to CPU arithmetic |
| W4A8 | Q4/Q6, K=256/512/5120, M=1/7/33/65, N=33, live padding, two epilogues equal to CPU reference |
| Actual 27B cache/graphs | cache off/miss/hit/eviction and eager/graphs agree exactly on four scenes |
| Decode | 2 sequences x 8 tokens; teacher-forcing top-1 8/8 each; alone/batch logits and seeded draws agree |
| Vision tower | 9 resolutions/aspect ratios agree bit-for-bit with the same-weight Transformers tower |
| Actual 27B vision | 23 synthetic scenes, 52/52 answers; 1/2/8 images, cache/graphs/mixed batches agree |
| Capacity | 256 through 8192 prefill tokens finite; 8193 refused; no decoder/head in the 8K check |
| Mixed service load | 1800.09 seconds, 10,560 requests, 0 unexpected errors, 307 thoughts and 103 image requests |
| Memory during mixed load | 23,240,195,584 bytes allocated after warm-up; final growth 0 bytes; 40 snapshots stable |

The mixed service run is primarily repeated short requests with warm caches;
image cache hits are not cold vision latency. It used a bounded-request helper
from the local prototype, which is not part of this backend contribution. No
exclusive-GPU speedup or overall 30% performance improvement is claimed.

## Frozen numerical limit

The independent reference uses plain PyTorch FP32 operations, original GGUF
weights and identical prompt tokens. It streams one layer on the same selected
card, sequentially with the candidate. A CPU anchor agreed within 1.62e-5 logits.
The frozen 501-row oracle SHA-256 is
`1a7da606458b5310c630172208f7c2c05b489fdab13afc2fe3cbef6139ba7aa8`.

| Measure | Prototype result | Local candidate threshold |
|---|---:|---:|
| Probability MAE | 0.00310922 | <= 0.001 |
| Probability P99 | 0.0232160 | <= 0.01 |
| Maximum probability error | 0.0354659 | report |
| All decisions matching | 483/501 | report |
| Reference margin > 0.02 | 452/457 = 98.9059% | >= 99.5% |

All three candidate thresholds fail and remain unchanged. Thirteen flips are
near ties; five exceed margin 0.02. This is a numerical regression set, not 501
independent graded evaluation tasks. An independent INT8-activation-only
diagnostic already shows nontrivial error, but BF16 intermediates, GDN chunking,
embedding/head conversion and implementation differences need further analysis.
There is no claim of full-model bit equality between AMD and NVIDIA.

## Trimmed patch recheck

Rechecked on 2026-10-24, applying #17 and #18 to the trimmed backend. The
integration checkout was `4875c36` (initial backend `6af596d` plus the two fixes),
with production-core SHA-256
`8d8502376f43c11a20eefe5a50d0acd9f1041a97f7a487deb8d158251c2fbd02`.
The core stayed unchanged throughout these checks. Subsequent changes package
CLI entry points safely and record results; they do not change the engine/kernels.

| Trimmed-core check | Result |
|---|---|
| CPU guard/planner and independent gguf-py dequantization | passed |
| Exact GPU arithmetic and Q4/Q6 projections | passed |
| Actual 27B four-scene graph/cache regression | exact, including eviction; 3 repetitions per eager/graph scene |
| Actual 27B decode | 2 x 8 tokens, teacher-forcing 8/8 each; exact batch isolation and seeded sampling |
| Actual 27B vision | 23 scenes, 52/52 answers; image/cache/graph/mixed batches exact |
| Independent vision tower | all 9 resolutions/aspect ratios bit-identical |
| Frozen 501-row probability comparison | unchanged: every case value equals the prototype; original gates still fail |
| Actual server HTTP | never/readme3/wide33, omitted/oversized thought budgets, invalid inputs, recovery, 4 clients x 16 requests passed |

The HTTP recheck uses the upstream server without the prototype's added queue or
readout-only features. It is a short integration check, not a new 30-minute run.
All computations used one selected R9700 while existing jobs continued running.

Five public CLI files are import-safe and their help entry points pass without a
GPU. A malformed oracle is rejected before model initialization. After packaging
those entry points, the public streamed-oracle and comparator were also run on
the fixed yes/no anchor: prompt tokens match the frozen anchor, maximum FP32
logit difference is 1.62e-5, and candidate probabilities match the prototype. Its
original precision gate remains failed, as expected; it is not counted as a
release pass.


## Separate compressed W4A16 profile (2026-10-26)

The historical W4A8 figures above remain unchanged. The explicit W4A16 profile
passes the unchanged frozen 501-case readout thresholds and the additional
20-case set, with all probabilities matching its own frozen profile. Its final
source also reproduces the W4A8 probabilities and fixed decode tokens unchanged.

The W4A16 integration completed independent operator, request-isolation,
graph/cache, teacher-forcing/seeded decode, vision, full HTTP, 1800-second mixed
service and launch/stop checks. The original invalid cross-request diagnostic
call and its corrected grouped rerun are retained in the workspace execution log.
Several speed goals remain unmet; passing the local numerical set does not
establish general model quality or production support. See the
[profile-specific report](amd_w4a16.md) and
[round-level evidence](../bench/results/r9700_w4a16_20261026.json).
