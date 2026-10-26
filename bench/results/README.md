# AMD benchmark evidence

The source tree keeps three small, reproducible records:

- [MI355X INT8 summary](mi355x_int8_20261026.json): measured bandwidth/compute ceilings,
  matched latency results and profiler counters.
- [MI355X next-version summary](mi355x_next_summary_20261026.json): matched INT8/FP4
  HTTP p50/p95, quality screen, validation counts and source identities. Its
  `pipeline_study_20261026` entry adds the current-runtime FP4 rebaseline and a
  diagnostic graph profile without replacing the historical matched comparison.
- [Frozen workload manifest](mi355x_fp4_frozen_20261026.json): exact HTTP requests,
  checkpoint/calibration hashes and pinned original BF16 shard identities.

The [execution report](../../docs/mi355x_next_version_results.md) includes the
protocol, rejected candidates and remaining quality/latency gates.

Raw captures, profiler traces, calibration files, logs and rejected research
scripts belong outside the checkout. This workspace archives them at:

```text
starlux_workspace/
  StartLux-Prism-mi355x/                 # source worktree
  models/                               # original checkpoints and FP4 artifacts
  artifacts/startlux-prism/2026-10-26/
    benchmark-results/                  # original tracked captures
    results/                            # local runs, traces, logs and calibration
    research/bench/                      # rejected experiments
    caches/                             # pytest, Python and rocprofiler scratch
    review-scope/                       # Radeon-only files and historical drivers
    manifest.json                       # original paths, sizes and SHA256 hashes
```

All 23 formerly tracked captures/research scripts were preserved byte for byte.
They remain available remotely at the immutable
[published source snapshot](https://github.com/zihaomu/StartLux-Prism/tree/32c77d63aa3d485f9d145251407fb6a0796a74d8/bench).
Report links to archived evidence use that commit so they also work in a fresh clone.
The local archive is not required to build, serve or run unit tests.

An additional 35 files from the published
[`d84a7c7` snapshot](https://github.com/zihaomu/StartLux-Prism/tree/d84a7c7f5a22daf4f927ffef829464480b17ca9e)
are preserved in `review-scope/` with a separate checksum manifest. These are
Radeon-specific tuning, reports, tests and tools, plus historical INT8 experiment
drivers and the generated figure. Radeon support remains in PR #19. This source
tree qualifies only `gfx950`; it retains the portable W4A8/W4A16 kernels used by
MI355X, the native FP4/INT8 qualification tools and model/API checks.

For new measurements, select an output directory outside the checkout and pass
model, calibration and converted-artifact paths explicitly. For example:

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

Use a fresh output path for every run. The harness pins image IDs, hashes the
workload and harness, saves per-request server timings, and reports p50/p95
with per-round medians. For a future speed-only FP4 candidate, use
`--profiles mxfp4-r32 --image BASE_IMAGE --candidate-image CANDIDATE_IMAGE`;
the two images alternate order and must preserve complete answers and usage.
Use at least `--rounds 5 --reps 100 --warmup 20` for the planned comparison.
Confidence intervals and drift assessment require paired analysis of the saved
rounds; the harness does not automatically establish statistical significance.

Inside the matched ROCm environment, profile the same one-pass text payload:

```bash
python3 bench/profile.py --gguf "$M" --model-dir "$M" --precision mxfp4 \
  --mxfp4-artifact "$FP4_ARTIFACT" \
  --manifest bench/results/mi355x_fp4_frozen_20261026.json --request readme3 \
  --max-tokens 4096 --reps 30 --gemm-detail \
  --graph-trace "$STUDY_OUT/readme3-graph.json" \
  --trace "$STUDY_OUT/readme3-eager.json" --out "$STUDY_OUT/readme3-profile.json"
```

Here `M` is the original BF16 directory, `FP4_ARTIFACT` the matching converted
directory, and `STUDY_OUT` an artifact directory visible inside that environment.
The profiler retains evidence boundaries and v1.1 none-option behavior. Wide
multi-pass requests are explicitly rejected; measure those through HTTP.
Graph events exclude transfers. Eager projection events include host launch
gaps and must not be summed into an HTTP latency estimate.

The fresh raw samples, traces, telemetry and checksum manifest are stored at
`../artifacts/startlux-prism/mi355x-pipeline-study/`, outside the checkout.
