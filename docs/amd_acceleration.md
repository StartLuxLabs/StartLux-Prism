# Experimental R9700 acceleration

This report covers the 16.55 GB v1.1 27B AutoThink Q4_K_M checkpoint on one
Radeon AI PRO R9700. It does not establish production numerical qualification.
The original W4A8 FP32 probability gate remains unmet; see
[validation scope](amd_validation.md). The mixed sustained run has passed; enhanced HTTP has also completed. Cold/hot JIT startup measurements have completed.

This report retains the strict W4A8 measurements. The separate compressed
[W4A16 profile](amd_w4a16.md) has its own numerical and service qualification;
its results are not included in the W4A8 speedup percentages below.

## Qualified first combination

The implementation uses an offline table to select packed signed-byte dot4,
Q4 WMMA with compact minimum correction, and Q6 WMMA with exact integer planes.
Selection includes the exact runtime, kernel source, shape, epilogue, stride and
alignment. Unrecognized cases keep the original implementation. There is no
online tuning, host read of the live-row scalar, CPU offload, or full floating
point weight expansion.

The table contains 591 entries. The selected configurations passed 912 complete
independent CPU projection checks; Q6 additionally passed 144 CPU/graph checks.
Whole-model ablation compared eight scenes under five implementation switches.
All logits agreed, including shared evidence, reordered/mixed questions, prefix
miss/hit/eviction, and eager/graphs. All 501 frozen probability cases exactly
reproduced the original experimental backend.

## Fixed-workload decode

Both implementations share the same resident model and complete generation
head. Each run prefills 512 prompt tokens, warms eight steps, then generates
64 timed steps, ignoring EOS. Graph caches are separate; a post-timing trace
confirmed that the old graph and optimized graph contain different kernels.
Every generated token matches the original frozen baseline. All eight sequences
also match individually executed sequences.

| Mode | Batch | Original total tokens/s | Optimized total tokens/s |
|---|---:|---:|---:|
| Eager, ten alternating pairs | 1 | 18.514 | 27.622 |
| Eager, five alternating pairs | 4 | 25.501 | 81.772 |
| Eager, five alternating pairs | 8 | 26.599 | 113.288 |
| Graph, five alternating pairs | 1 | 18.525 | 27.868 |
| Graph, five alternating pairs | 4 | 25.514 | 82.418 |
| Graph, five alternating pairs | 8 | 26.637 | 113.905 |

These are medians of paired rounds, not sums of independently selected best
samples. Batch-eight eager throughput is about 14.16 tokens/s per sequence.
The first five eager batch-one pairs were variable: the optimized median was
24.421 tokens/s, with individual results 23.041–27.519; the old implementation
also had a 12.433 tokens/s sample. All samples were retained. A subsequent ten
pair check, with unchanged source and workload, measured 27.458–27.698 tokens/s
for the optimized eager path. No single cause of the earlier variation is proven.

## Prefill and service limits

The first whole-model engine ablation, excluding the resident generation head,
measured 225.892→174.334 ms for readme3, 425.506→344.495 ms for startlux3,
1380.744→1082.041 ms for longpage2 and 9967.831→8039.819 ms for a full 8192-token
prompt. These are engine timings, not HTTP results. The 19–24% improvements do
not satisfy the local 30% text-latency target.

Profiling the optimized implementation still attributes roughly 89% of short
text kernel time and 80% of 8K kernel time to Q4/Q6 projections. Graph replay
adds little to the measured short readout or decode performance. A bounded
batch-capacity scan did not reach the target of 5.5 requests/s with P95 below
1600 ms at concurrency eight; increasing client concurrency alone is not a fix.

## Hardware evidence and scope

The observed prefill ISA uses gfx1201 INT8 WMMA, with exact FP16 WMMA for Q4
minimum correction. Decode uses signed-byte dot4. The selected first-combination
code objects have zero scratch; maximum VGPR counts are 72 for dot4, 190 for Q4
and 243 for Q6. Static instruction presence does not measure matrix utilization.

A checked 1 GiB streaming read reached 633.75–634.21 GB/s. Copy measured 556.02
GB/s when counting both read and write bytes. Model DRAM traffic and WMMA
utilization were not measured with valid counters. Logical weight bytes divided
by time must not be reported as measured DRAM utilization.

The stack was ROCm 10.1.0 / HIP 7.16.26385, Python 3.13.13, Torch
2.14.0+rocm10.1.0 and Triton 3.8.0+git669b31ac.rocm10.1.0. The selected GPU had
no observed external compute queues during these stages, while other host jobs
continued. There is no same-configuration RTX 4090 result and no extrapolation
to other Radeon or Instinct architectures. CUDA hardware regression is open.

Reproduction commands and timing-field definitions are in
[the AMD guide](amd.md#reproduce-acceleration-measurements).

## Mixed sustained service

The frozen integration completed 1800.409 seconds and 7132 sustained requests
with zero unexpected errors. It included 480 actual thought requests, 135 image
requests and 4389 exact text-control checks. The cache evicted 2467 entries
during the sustained phase. Drained allocated memory grew by zero bytes.

This standard 2K configuration retained the complete generation head and matching
projector. The observed device-memory peak was 23.45 GiB. These are mixed,
partly cached service requests, not cold-image performance measurements. The
owned test container stopped; its runtime/server logs showed no GPU fault,
illegal-access, OOM or traceback match. Host kernel logs were inaccessible, so
no global kernel-log audit is claimed.

## Completed HTTP protocol

The candidate and baseline used the same complete generation head, 8K capacity,
unique evidence IDs and workload protocol. Common scenes used 102 timed requests
each; long-page and 8K scenes used 30, all across three rounds.

| Scene | Original P50 ms | Optimized P50 ms | Optimized P95 ms | P50 improvement |
|---|---:|---:|---:|---:|
| yesno | 147.550 | 116.577 | 116.915 | 20.99% |
| readme3 | 235.125 | 184.265 | 184.601 | 21.63% |
| startlux3 | 441.302 | 356.496 | 356.956 | 19.22% |
| review1 | 154.765 | 121.755 | 122.111 | 21.33% |
| wide33 | 451.218 | 360.920 | 361.433 | 20.01% |
| longpage2 | 1408.913 | 1098.428 | 1098.958 | 22.04% |
| 256 | 296.578 | 239.762 | 240.047 | 19.16% |
| 2048 | 2348.921 | 1853.519 | 1860.923 | 21.09% |
| 8192 | 9968.347 | 8066.182 | 8099.382 | 19.08% |

All 21 concurrency rounds completed without errors. At concurrency eight, median
throughput was 5.07 requests/s and median P95 was 1716.430 ms. The text-latency and
concurrency targets remain unmet. There was no median P50/P95 regression above
5% in the fixed text/length scenes.

For 1918 identical request bodies, numerical answers and usage matched exactly,
including all 30 actual AutoThink requests, generated thought text, token counts
and termination reasons. Different precision profiles are not included in these
strict-equivalence speedup percentages.

Actual AutoThink used ten fixed requests at each budget. Outputs and usage matched
the baseline; a budget of 256 did not mean that 256 tokens were generated.

| Maximum budget | Actual output-token range | Original HTTP P50 ms | Optimized HTTP P50 ms |
|---|---:|---:|---:|
| 16 | 16–16 | 1523.127 | 1085.655 |
| 64 | 47–64 | 3952.132 | 2714.928 |
| 256 | 53–87 | 4219.184 | 2890.459 |

## Startup

The standard 2K service became ready in 44.948 seconds with an empty compiler
cache and 22.357 seconds after restarting with that cache. Warmup produced 81
code objects; the four tested first requests added none. These are two measured
launches, not startup percentiles. The OS file cache was not flushed.

These startup requests omit the benchmark's unique evidence ID and use a 2K
capacity, so their latencies do not replace the enhanced HTTP table. Repeated
requests can benefit from the prefix cache. Both owned test containers stopped;
their runtime/server log fault scans were empty.

Round-level measurements, runtime/source identities and unmet engineering gates
are retained in [the compact evidence](../bench/results/r9700_20261025.json).
