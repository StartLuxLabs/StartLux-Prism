# MI355X: 27B INT8 decision latency

The follow-up implementation roadmap is in the
[next-version speedup plan](mi355x_next_version_speedup_plan.md).
Completed stages are recorded in the [next-version results](mi355x_next_version_results.md).

On one MI355X, the README's three-question request improves from **27.01 to
20.54 ms over HTTP (1.315×)**, with identical answers and usage. Across five fixed
request suites, measured HTTP speedups range from **1.033× to 1.334×**. These are
single-client, warmed inference results with the cross-request prefix cache
disabled. Startup and calibration are excluded.

![Measured latency and projection throughput](https://raw.githubusercontent.com/zihaomu/StartLux-Prism/d84a7c7f5a22daf4f927ffef829464480b17ca9e/docs/images/mi355x-performance.svg)

## Hardware ceilings

AMD specifies 8 TB/s of HBM bandwidth, 5,033.2 **dense** INT8 TOPS and 2,516.6
dense BF16 TFLOPS. The larger structured-sparsity figures do not apply here.
See [AMD's MI355X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).

| Test | Measured | Fraction of theoretical peak |
|---|---:|---:|
| Triton streaming copy, 512 MiB per operand, read + write traffic | 5.77 TB/s | 72% |
| PyTorch device copy, same sizes | 5.45 TB/s | 68% |
| PyTorch INT8 → INT32 GEMM, 8192³ | 2,785 TOPS | 55% |
| PyTorch BF16 GEMM, 8192³ | 1,784 TFLOPS | 71% |

These calibrate this software stack and workload sizes; they are not register-only
instruction peaks. Each copy operand exceeds MI355X's 256 MB Infinity Cache.
The roofline plot uses useful operations and minimum tensor bytes. Repeated
fetches and padded matrix work reduce practical efficiency below that envelope.

## What limited the initial port

On the three-question request, 256 INT8 GEMM dispatches consumed **20.92 ms,
79.3% of the eager GPU kernel time**. The graph replay took 26.46 ms. GDN scan,
recurrent and output kernels were the next substantial contributors.

The portable INT8 fallback inherited tile rules measured on NVIDIA. Those rules
left avoidable weight traffic on MI355X. In one representative projection,
`M=192, N=34816, K=5120`, profiler counters show:

| Metric | Baseline | Selected tile |
|---|---:|---:|
| Matrix tile `(BM, BN, BK, warps, stages)` | `(64,128,64,8,4)` | `(128,128,64,8,4)` |
| Counter-instrumented kernel duration | 124.24 µs | 105.00 µs |
| External L2 read traffic (`FetchSize`) | 567.6 MB | 368.0 MB |
| External read bytes / weight bytes | 3.18× | 2.06× |
| External L2 read rate | 4.57 TB/s | 3.50 TB/s |
| MFMA utilization counter | 10.83% | 17.84% |
| Useful INT8 throughput from those durations | 551 TOPS | 652 TOPS |

The lower read rate accompanies less traffic and a faster kernel. The selected
128-row tile also pads the 192 rows to 256 physical rows, so useful TOPS and
physical matrix-instruction utilization measure different quantities.

`FetchSize` is an L2/Fabric traffic measurement. It does **not** establish pure
HBM traffic: on CDNA4, requests attributed to HBM can be served by the last level
cache. AMD documents this distinction in its
[L2/Fabric metric definitions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.1.0/conceptual/cdna/l2-cache.html).

Thus the evidence points to weight movement, tile reuse and resource allocation
as the immediate latency targets. Peak dense TOPS alone is an incomplete target
for these small row counts.

## Applied change and whole-request results

The dispatcher now selects **67 offline-qualified tiles** by allocated row bucket,
matrix dimensions and epilogue. The table requires the exact GPU name, Torch,
Triton and HIP versions, plus a hash of the kernel source. Missing matches use
the previous tile rule. Selection uses host metadata; it performs no online
search or device-scalar read. The kernel also masks activation-scale loads in
partial row tiles.

Integer products, scale rounding and residual arithmetic are unchanged. All
**364 qualification cases** matched a library INT32 product with ordered FP32
scaling, bit for bit. Three alternating baseline/tuned engine rounds verified
identical complete letter logits. HTTP checks verified identical answers and
token usage. The focused regression suite passes **33 tests and 11 subtests**.

| Request suite | Engine before → after | HTTP before → after | HTTP speedup |
|---|---:|---:|---:|
| `yesno` | 16.02 → 15.34 ms | 16.48 → 15.96 ms | 1.033× |
| `review1` | 24.09 → 18.02 ms | 24.46 → 18.33 ms | 1.334× |
| `readme3` | 26.57 → 20.19 ms | 27.01 → 20.54 ms | 1.315× |
| `startlux3` | 39.87 → 33.92 ms | 40.52 → 34.62 ms | 1.170× |
| `longpage2` | 111.01 → 95.87 ms | 111.10 → 96.03 ms | 1.157× |

The tuned profile puts INT8 GEMMs at 14.72 ms of 20.23 ms eager kernel time.
GDN is the next useful target after further reducing repeated GEMM weight
traffic. Single short questions improve much less than multi-question requests.

## Reproduction and scope

Measured 2026-10-26 on MI355X (`gfx950`), with Python 3.12, PyTorch
`2.14.0+rocm10.1.0`, HIP `7.16.26385` and Triton `3.8.0`, using
`StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf` and its existing SmoothQuant
calibration. Power and clock controls were unchanged. The model and runtime
instructions are in [mi355x.md](mi355x.md).

Engine numbers are the median of three round medians, each with 30 measured
calls after three warmups. Mode order alternates between rounds. HTTP numbers
are three round medians of 20 requests each, through separately warmed baseline
and tuned servers. A single client keeps its connection open, thinking is
disabled, and prefix caching is disabled. The constant system prefix remains
part of Prism's normal engine design.

Microbenchmarks rotate more than 512 MiB of weight copies, beyond twice the
Infinity Cache capacity. HIP-event times exclude host launch overhead and use
the fastest of five graph replays. Source/kernel identity is checked when
building the table. Hardware counters were collected in separate instrumented
runs, one metric per pass; their kernel durations are not HTTP timings.

From a matched ROCm environment with the checkout mounted:

```bash
# Set G, M and CALIB to the same checkpoint, folder and calibration in this environment.
RUNS=../artifacts/startlux-prism/2026-10-26/results/mi355x-perf
STARTLUX_PRISM_MI355X_GEMM=baseline python3 bench/mi355x_roofline.py \
  --check --out "$RUNS/roofline-baseline.json"
python3 bench/profile.py --gguf "$G" --model-dir "$M" --precision int8 --calib "$CALIB" \
  --request readme3 --max-tokens 4096 --gemm-detail \
  --out "$RUNS/tuned-readme3.json" --trace "$RUNS/tuned-readme3-trace.json"

```

`bench/mi355x_build_table.py` freezes measured INT8 choices. The historical
paired-latency, counter, GDN ablation and figure-generation drivers are archived
under `../artifacts/startlux-prism/2026-10-26/review-scope/bench/` and remain in the
[published source snapshot](https://github.com/zihaomu/StartLux-Prism/tree/d84a7c7f5a22daf4f927ffef829464480b17ca9e/bench).
Use that snapshot to replay the historical experiment. The current matched HTTP
comparison is reproduced in the [next-version results](mi355x_next_version_results.md).
Set `STARTLUX_PRISM_MI355X_GEMM=baseline` before server startup to compare with
the previous rule. Existing graphs retain their captured choices.

The compact measurements are tracked in
[`bench/results/mi355x_int8_20261026.json`](../bench/results/mi355x_int8_20261026.json).
Full samples, Chrome traces, counter CSVs and logs are archived locally under
`../artifacts/startlux-prism/2026-10-26/results/mi355x-perf`; see the
[evidence index](../bench/results/README.md). Pass an external output path for
new runs. This work measures single-GPU decision latency; it does
not establish AutoThink decode, multi-client throughput, tensor-parallel or
W4A8/W4A16 speedups.
