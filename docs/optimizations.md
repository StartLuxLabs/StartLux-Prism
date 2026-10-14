# Optimization log

What was tried on the RTX 5090, what it bought, and what did not work. Latencies are p50 over HTTP with unique
requests (`docs/benchmarks.md`); kernel times from `bench/profile.py` and the GEMM microbenchmarks.

## Where the time goes

One forward pass (int8, CUDA graphs), by kernel family:

| request (computed tokens) | GPU ms | projections (GEMM) | linear attention (conv, chunk, scan, output) | attention | norms, SwiGLU, quant |
|---|---|---|---|---|---|
| yesno (55) | 4.2 | 69% | 18% | 2% | 10% |
| startlux3 (361) | 9.4 | 66% | 20% | 2% | 10% |
| longpage2 (1,185) | 22.7 | 73% | 14% | 5% | 7% |
| JevBench hard-sol-b-long_policy-06 (2,617) | 48.6 | 67% | 15% | 8% | 10% |

JevBench's mean latency is dominated by its 36 prompts of 2,000-4,000 tokens (56% of the total time): projections at
large M first, then attention, whose cost grows with the square of the prompt.

Measured ceilings of this GPU (`bench/csrc/mma_peak.cu`, `bench/csrc/l2_bw.cu`): int8 `mma.sync` 886 TOPS (register
operands), L2 reads 6.2-7.1 TB/s, DRAM reads 1.70 TB/s.

## Kept

* **Pipelined scheduler** (`server.py`): the next batch is planned and its tables staged while the GPU runs the current
  one, then queued behind it; the scheduler waits only after that. With all per-batch tables in one flat buffer (one
  host-to-device copy instead of 26 small torch calls, each a GIL release and re-acquire) and a 0.5 ms GIL switch
  interval, throughput at 64 clients went from 194 to 273 requests/s; GPU utilization under load from 71% to 98-100%.
* **Tokenizer without its thread pool** (`prompt.py`): waking the tokenizers library's rayon pool for a request's 2-4
  short strings cost about 1 ms per request inside the server (startlux3: 12.9 -> 11.6 ms).
* **Thread-safe head cache**: the LRU of tokenized prompt heads raced under concurrent unique requests (KeyError ->
  HTTP 422 at 64 clients).
* **conv_prep over blocks of 16 tokens** (`kernels/gdn.py`): one program per 16 tokens x channel group instead of per
  token: 0.49 -> ~0.15 ms at 361 tokens.
* **Minimal HTTP/1.1 server** (`httpserve.py`): a hand-parsed keep-alive server instead of `http.server`; ~0.1 ms.
* **Direct execution on an idle server** (`server.py`): when nothing is queued or running, the request's own thread
  plans, launches and waits for its forward pass instead of handing it to the scheduler thread and waiting to be
  handed the result back (two thread wake-ups, and the two threads contending for the GIL). Requests that arrive
  meanwhile queue for the scheduler's next batch as before. 14-17% on every latency suite (yesno 6.37 -> 5.33 ms,
  longpage2 33.2 -> 27.6 ms), throughput unchanged.
* **GEMM dispatch table per token bucket** (`bench/cuda_gemm_micro.py --buckets`): each bucket timed with the live row
  count in the middle of its range, choosing among the Triton kernel, the CUDA tile kernel and stream-K; tile
  granularity matters more than the bucket (262 rows in 128-row tiles compute 384).
* **Software-pipelined GEMM main loop** (configs 8-13 of `csrc/w8a8_gemm.cu`): fragments of the next k32 step load
  from shared memory while the current MMAs run, across k-tile boundaries; the barrier moves to a tile's last step.
  1.08-1.30x on the 256x128 / 128x256 tiles at every size and 1.1-1.3x at 128 rows; neutral for 128x128 at large M.
* **Deterministic split-K, deferred into the next norm**: split-K with fp32 atomic adds made the same request's
  probabilities vary between runs (a near-tie decision flipped between test runs). Reducing inside the GEMM
  deterministically costs a synchronization tail: int32 partials through global memory (last split to arrive adds
  them) 11.6 us, a thread block cluster adding over distributed shared memory 14.9 us, splits taking turns
  through semaphores 16.3 us, against 7.4 us atomic (2560x4096 at 64 rows). Instead each split stores its int32
  partial sums in its own slot and the next kernel that reads the residual anyway (`norm_q8`, or `quant.fold` after
  the last layer) adds them, scales the total and adds it into the residual: 7.4 us, plus 1-10 us in the norm
  (measured, and charged to the choice in the table: splitting stops paying at a few thousand rows).
  `STARTLUX_PRISM_NONDETERMINISTIC=1` re-enables atomic split-K.
* **Prefix cache across requests** (`engine/prefix_cache.py`, `engine/batch.py`): a request whose evidence is cached
  runs only its questions' tokens, from the entry's linear-attention state and conv tail (an arena slot) and with the
  entry's keys and values as the attention's parent range. Follow-ups: 9.3 -> 4.7 ms (230 evidence tokens), 19.1 ->
  4.8 ms (935), 48.0 -> 5.5 ms (2,000-3,700). Misses write the entry in the pass that computes the evidence anyway.
  Two refinements took the cost of a miss from up to +0.16 ms to at most +0.13 ms (+1.3%), and to +0.07 ms on the
  one-question ticket: a first-wave segment with one child is scanned as one chain (the state saved on the way, no
  second-wave launch to wait for: +0.16 -> +0.08 ms on startlux1), and the conv-tail saves moved into the conv kernel
  (one launch fewer per linear-attention layer on every pass, cache or not). A batch writing more new entries than its
  tables hold (first version: 5) failed with HTTP 422 under load; the planner now computes the rest uncached, up to
  64 new entries per batch.
* **Residual add with batched loads** (`csrc/w8a8_gemm.cu`, epilogue ADD): the output projections add into the
  fp32 residual stream; the epilogue read, added and wrote one element pair after another, and since a store may
  alias the next load, every read paid the full DRAM latency. Measured with a build that skips the epilogue's writes
  (`bench/epilogue_cost.py`): 29% of the down projection at 4,096 rows (106 of 366 us; its main loop alone runs at
  ~740 TOPS), 8-15% of the 2,560-wide projections at 1,280 rows, 0-8% for the bf16-storing projections. Now each
  16-row slice issues all its residual loads first, then adds and stores (float2): down projection 366 -> 310 us at
  4,096 rows, 130 -> 118 us at 1,280; output projection 65 -> 58 us at 1,280. Same arithmetic, same bits.
* **Dispatch table measured with weights really streamed from DRAM** (`bench/cuda_gemm_micro.py`): it rotated 4
  copies of each weight matrix to keep them out of L2, but 4 copies of the 10-24 MB output projections fit in the
  96 MB L2, so their small-M entries were chosen on L2-resident weights (5.6 us where DRAM streaming takes 9.0 us for
  the 2,560 x 4,096 projection at 1-64 rows; 8.8 against 17.4 us for 2,560 x 9,216). The rotation now holds at least
  3x the L2 size; the two residual-add shapes were re-measured (with the batched-load epilogue). Against the previous
  table, in one process (`bench/ab.py --table-a/--table-b`, 100 interleaved pairs): readme3 -2.2%, longpage2 -1.0%,
  startlux1 -0.9%, startlux3 -0.8%, the longest JevBench item -0.4%, the rest within +-0.4% - and identical bits for
  every request, though the two tables pick other kernels, tiles and splits for many buckets.
* **Programmatic dependent launch for the GEMMs of short passes** (`Options.pdl`, `pdl_mode`; buckets up to 512
  tokens): the kernels that produce GEMM inputs (`norm_q8`, `gated_norm_q8`, `rows_q8`, `swiglu_q8`) signal at their
  start that the next kernel may launch; the CUDA GEMM, launched with programmatic stream serialization, starts while
  they run, streams the first 4 k-tiles of its weight panel toward L2 (CTAs of the first M tile only), then waits
  (`griddepcontrol.wait`) before reading activations, scales or the residual. `bench/ab.py` against ordinary launches
  (60 interleaved pairs, identical bits): yesno -1.0%, review1 -1.0%, startlux3 -1.0%, a 120-token JevBench item
  -0.9%, startlux1 / readme3 -0.3%; long prompts unchanged (not used above 512 tokens). Variants: early launch without
  any prefetch, within noise; every CTA prefetching its whole panel, 6-17% slower (at a few hundred rows several CTAs
  share a panel and prefetch it again, and the prefetches crowd out the main loop's own loads).
* **Question tokens cached, offsets not computed** (`prompt.py`): of the Python request path (rendering, tokenizing,
  planning, answers, JSON), 75-85% is the tokenizer's Rust code; the question tails (question block + assistant turn
  opening) repeat across requests about new evidence, so their tokens are remembered by text (LRU, like the heads),
  and `encode_batch_fast` skips the character offsets (18% faster, same ids). Profiled on the development Mac (CPU
  only): rows_for 297 -> 155 us for startlux3, 658 -> 430 us for longpage2, 70 -> 43 us for yesno; the whole host
  path 395 -> 252 us (startlux3). `tests/test_prompt.py`: 241 prompts still token-identical to the chat template.
* **Batch-invariant numerics** (int8): a request's probabilities are the same bits whatever else is in its batch,
  whether its evidence came from the prefix cache, and at any concurrency. The only batch-dependent choice in the
  forward pass is which GEMM kernel, tile and K split the dispatch table picks for the bucket; integer sums are exact
  in any order, so it suffices that every path scales and adds the int32 total with the same roundings: (float(acc)
  x s_a) x s_w, then the residual add, each rounded separately with explicit `_rn` intrinsics in the CUDA epilogue,
  the Triton kernel and the deferred fold (an FMA's single rounding had made the CUDA tile kernel's residual add
  differ from Triton's), and deferred split-K partials kept as int32 until summed (fp32 partials rounded twice). The
  final norm and letter logits moved from cuBLAS (algorithm chosen by row count) to a fixed-order Triton kernel.
  Attention, the linear-attention scan and the norms were already per-segment and per-row. Checked by
  `tests/test_cuda_gemm.py` (every config, split mode and stream-K: the Triton kernel's bits) and
  `tests/test_prefix_cache.py --exact` (cache hit = miss = the same request among others, bit for bit). Cost: none
  measurable (JevBench 205/231 at 15.8 ms mean, suites within run-to-run noise). Before, hits and misses differed by up to 0.025 in probability (int8 rounding noise from
  different kernels), which can flip a near-tie. The bf16 path (cuBLAS) is not batch-invariant.
* **Attention 4 warps x 3 stages** instead of 8 x 2 (same 32x32 tiles): 1.3x on long segments, 105 TFLOPS at 3,000
  tokens (`bench/attn_micro.py`).
* **Launch checks** in the CUDA GEMM: a tile config whose stages exceed 99 KB of shared memory used to fail silently.

## Measured before deciding: three proposed optimizations (2026-10-09)

Three candidates were proposed with estimates: a faster large-matrix GEMM (about +10% on long prompts), SwiGLU and
its quantization fused into the gate|up GEMM (about +3%), and the remaining Python request handling moved to native
code (about 0.5 ms on short requests). Measured upper bounds (logs: `results/eval3/`):

* **Where the GPU time goes** (`bench/profile.py --gemm-detail`, int8): GEMMs 64.5% of a 3,912-token pass (74 ms),
  74% at 1,185 tokens, 71% at 361, 67% at 55; SwiGLU + quantization 3.9 / 2.6 / 2.7 / 3.1%; attention and the
  linear-attention kernels most of the rest. At 55 tokens the GEMMs stream the 3.57 GB of int8 weights at 1.3 TB/s
  (floor at 1.70 TB/s: 2.1 ms) and about 200 small per-layer kernels take 1.3 ms of 4.1.
* **Large-matrix GEMM** (`bench/gemm_ceiling.py`, `bench/cuda_gemm_micro.py`): sustained, the int8 mma.sync peak is
  988 TOPS (2.27 GHz, 311 W), but every real GEMM at 4,096 rows runs into the 575 W board limit (and lower clocks).
  There, on the wide projections (inputs, qkv, gate|up), StartLux-Prism is at or above cuBLASLt (555-559 against 538-551
  TOPS); on the K-heavy, 2,560-wide output projections cuBLASLt is 17-24% faster - about half of which is our fused
  residual add (an fp32 read and write per output, 42 MB at 4,096 rows) that cuBLASLt's bare int32 product does not
  do. Matching cuBLASLt there: 2.6 ms of a 74 ms pass at 4,096 tokens (3.5%), 0.6 ms at 2,560 (1.4%), nothing at
  1,280 tokens and below (we are faster there; below a few hundred rows the GEMMs are bound by the weight stream).
* **SwiGLU + quantization in the gate|up epilogue**: the kernel it removes is 2.6-3.9% of GPU time. Per-token
  quantization needs the whole 9,216-wide row's maximum, which no GEMM tile sees: a fused epilogue either writes
  bf16 activations + per-row maxima (atomic max) for a lighter quantize pass - half the bytes of the gate|up output
  and of the quantize pass's read, about 2.7% at 4,096 tokens and nothing at short ones - or quantizes per tile
  (per-group scales), which makes the down projection's main loop scale every group in fp32: its split-K partial
  sums would then be fp32 and batch invariance lost, or no split-K and short requests slower.
* **Native request handling** (`bench/host_overhead.py`, `bench/spin_client.py`): of a yesno request's 5.30 ms,
  0.57 ms is not GPU time - but 0.23-0.31 ms of it is the benchmark client waking up from its blocking socket read
  (a client that polls instead sees 0.035 ms outside the server's handler): the same for every engine and outside
  any server. Inside the server, 0.25 ms (yesno) to 0.6 ms (startlux3); prompt rendering + tokenization is the
  largest part and 88% of it is the tokenizer's Rust code already (250 of 283 us for startlux3). A native front end
  would remove about 0.15-0.2 ms: 3-4% on the shortest requests, 1.5-2% on startlux3, under 1% on long prompts.

Ranked by measured gain on the request mix (JevBench: most items short): native request handling (~0.15-0.2 ms on
every request), the two-pass SwiGLU fusion and the K-heavy GEMMs (~1.5-3.5% on prompts over 2,500 tokens only). Larger
than all three: the ~200 small per-layer kernels of a short pass - the linear-attention chunk kernels alone (intra,
two scan waves, output) take 0.67 ms of yesno's 4.1 ms for one 64-token chunk per layer.

## W4A8 for Q4_K_M files (2026-10-11)

* **Design, from a simulation first** (`tests/quant_sim.py`, 66 questions against the fp32 reference of the Q4_K_M
  file itself): the file's Q4_K / Q6_K weights as they are with int8 activations per 32 inputs (llama.cpp's MMQ): mean
  |dp| 0.0031 (4B), 0.0026 (9B); per 256 inputs (one Q4_K super-block): 0.0063, 0.0056; per token, unsmoothed: 0.0186,
  0.0180 (one decision flipped on the 9B); requantizing the weights to the int8 path's per-row W8A8: 0.0126.  Per 256
  was taken: inside a super-block the 6-bit sub-block scales and mins are integers, so the dot products, their scales
  and the mins' activation sums add exactly in int32, with one floating-point step per 256 inputs; per 32 would need
  one per 32.
* **Kernel** (`csrc/w4a8_gemm.cu`): w8a8_gemm.cu's main loop with 128 inputs per stage; Q4_K weights repacked so one
  ldmatrix word per thread holds a k32 B fragment's 8 nibbles (two masks unpack it); Q6_K kept as int8 and run as two
  m16n8k16 MMAs per k32 step (one scale per 16); mins added at the super-block's end with dp2a (16-bit activation
  sums x 8-bit mins).  Bit-identical to the reference arithmetic in all 8 tile configs x 3 stage counts and to the
  Triton version (`tests/test_w4a8.py`), so batch-invariant.  The Triton version came first and is the reference:
  10-200x slower (K=32 dots and the unpacking compile badly).
* **Measured** (`bench/w4a8_micro.py --search`, 4B shapes, against W8A8's dispatch table): 1.0-1.4x W8A8's time up
  to 64 rows (half the weight bytes, but no split-K yet), 0.7-2.1x at 256 rows, 1.2-1.9x for Q4_K and 2.3-2.7x for
  Q6_K at 4,096 rows.  End to end on the 4B: 1.5-1.7x the int8 path's latency; the GEMMs are 22 of 27 ms of GPU time on
  startlux3 (W8A8: 12 of 16.5, profiled).
* **Why slower at large M:** W8A8 issues about 1.4 other instructions per MMA and is tensor-bound; W4A8 about 13 (four
  integer multiply-adds for the sub-block scales, about six for each output's floating-point step per super-block -
  two conversions, three multiplies, a subtraction, an add and four dp2a for the mins - and the unpacking), so the
  issue slots, not the tensor cores, set the pace.  Tried without gain: all MMAs of a step before the scaling,
  double-buffered fragments within a stage.  Cut registers from 170-190 to 128 per thread (mins summed at the
  super-block's end, MMAs per column tile, scale words read once per stage): Q4_K 15-22% faster.
* **Second pass: where the time went** (no profiler counters on this host: `RmProfilingAdminOnly`), from ablations -
  the kernel compiled with one part removed, timed on the 4B's shapes (results wrong, only the time counts):
  at 4,096 rows the per-256 floating-point step took ~21% of Q4_K's time, the sub-block scale multiply-adds 11-16%
  (Q6_K: 28-31%, two per output and k32 step), the unpacking 0-5%; computing on stale tiles (no copies after the
  prologue) saved 13-30%, and leaving out only the scale copies 6-23% - more than the A and W tiles: stored row-major
  ([N, K / 32]), each column's few scale bytes sat a row apart, so a stage's 768 small copies touched as many 32-byte
  sectors as the tiles themselves.  The tile skeleton alone (copies, ldmatrix, MMAs, barriers) ran at 720-800 TOPS,
  above W8A8.  The 4x4-warp 128x128 tile ran one CTA per SM (shared memory) yet at 128 registers (spilling).
* **Second pass: changes** (all bit-identical to the reference arithmetic in every config, `tests/test_w4a8.py`):
  - scales, mins, d and dmin, and the activations' scales and sums, super-block-major (a super-block's values of
    consecutive rows side by side): a stage's copies read whole lines.  Q4_K 2560x9216 at 4,096 rows -16%, at 64 -40%;
  - the per-super-block values copied with the super-block's second stage only (the first copies its scales): -2-4%;
  - 64x32 and 32x64 warp tiles at up to 255 registers (configs 8-12, launch bounds per config);
  - the main loop a super-block per iteration: its first sub-block starts the integer sums (no reset, no branch);
  - Q4_K's mins as two m16n8k16 MMAs per tile and super-block - the activation sums split S = (S & 63) + 64 (S >> 6)
    into int8 bytes by the producers - instead of four dp2a per output;
  - the floating-point step fused: t = d * acc, then fma(-dmin, mins, t) and fma(a, t, y), one rounding each (two
    instructions fewer per output; the Triton and torch references fused alike, the torch one emulating the single
    rounding exactly: the float64 product, the sum rounded to odd, then to fp32): -2-7%;
  - a tuning table over the engine's own projection shapes (fused segments, row slices) and token counts
    (`bench/w4a8_table.py`, `kernels/w4a8_table_rtx5090.json`), the rule (`pick_rule`) for anything else.
* **Second pass: result** (`bench/w4a8_micro.py`, 4B shapes, against the first version): Q4_K 11-20% faster at 4,096
  rows and ~17% at 256, Q6_K 7%; at 4,096 rows Q4_K now takes 1.04-1.6x W8A8's time and Q6_K 2.3-2.6x.  End to end
  (`scripts/compare.sh startlux-prism-w4a8`): JevBench mean 22.1 -> 19.7 ms (p50 8.2 -> 6.6) on the 4B and 36.6 -> 32.1 (11.7 ->
  10.0) on the 9B, the HTTP suites' p50 8-27% lower, throughput 186 -> 198 and 105 -> 117 requests/s.  The tuning
  table alone: 12% below the rule at up to 64 rows, 5% at 64-512, 1% beyond (worst rule choice 1.9x).
* **Second pass: tried, not kept:** programmatic dependent launch as on the W8A8 path (launched during the activation
  producer, the first stages' weight rows prefetched into L2): JevBench within noise, wide33 7% slower; Q6_K fragments
  single-buffered to free registers: ptxas spilled more, not less (64x64x2 warps 80 -> 248 bytes).

## H200 (Hopper): the 27B (2026-10-12)

* **int8 GEMMs on Hopper's warpgroup MMA** (`csrc/w8a8_wgmma.cu`). On the H200 the mma.sync kernels of
  `csrc/w8a8_gemm.cu` ran at 510-540 TOPS from 1,000 to 8,000 rows (the 27B's shapes, weights streamed from DRAM),
  cuBLASLt's int8 kernels (`torch._int_mm`, int32 output, no epilogue) at 770-920, and the int8 engine was barely
  faster than bf16 on cuBLAS (startlux3: 33.5 against 36.1 ms): Hopper's tensor cores take half their int8 rate through
  mma.sync.  wgmma.mma_async issues a 64 x N x 32 product per warpgroup from operands in shared memory; the kernels
  keep w8a8_gemm.cu's 128-byte swizzle (it is the layout wgmma's descriptors read) and its arithmetic, so every output
  has the Triton kernel's bits (`tests/test_cuda_gemm.py`: all tile configs and stage counts, both epilogues, live
  rows < allocated rows, unaligned output views).  Two pipelines:
  - cp.async copies by every thread, one MMA group in flight, a CTA barrier per stage: 750-850 TOPS;
  - TMA + warp specialization: a producer warpgroup (one thread, 40 registers) loads each stage with two TMA copies
    (`cp.async.bulk.tensor`, tensor maps in the same 128-byte swizzle) that complete on the stage's mbarrier; the MMA
    warpgroups (232 registers: 128 int32 accumulators each for 64 x 256) wait on it and release the stage on a second
    mbarrier once their MMAs on it completed - STAGES - 1 stages in flight, no CTA barriers: 1,000-1,140 TOPS at
    4,096-8,192 rows.
* **Where the TMA kernel's time went** (ablations - the kernel compiled with one part removed, results wrong, only
  the time counts; 2,048 and 8,192 rows): without its epilogue 1,440-1,640 TOPS against 1,000-1,220 with it; without
  the weight copies after the prologue 1,100-1,150; without any copies 1,120-1,490.  Not the loads but the epilogue:
  each thread stored its fragments as 4-byte pieces of 8 rows per instruction, one CTA per SM (shared memory), so the
  tensor cores idled through 30% of a bf16-output projection and 15-20% of a residual-add one.  Now the warps stage
  their rows, scaled, in the pipeline's shared memory (rows padded by 16 bytes: conflict-free fragment stores) and
  copy them out in 16-byte pieces; the residual add loads 8 pieces per lane before its first add.  Same arithmetic,
  same bits.  bf16-output projections 25-35% faster (gdn_in at 2,048 rows 316 -> 238 us), residual-add ones 5-15%
  up to 2,048 rows (mlp_down at 8,192 rows 3% slower).
* **Small batches:** 64-wide tiles (64x64, 128x64 on the TMA pipeline: four times the tiles of 256-wide ones, so
  the SMs fill; the 5,120-wide output projections had 40 tiles of 128 for 132 SMs), and for the residual-add
  projections deferred split-K as on the mma.sync path (each K split stores its int32 sums in its own slot; the
  residual's next reader adds the slots, then scales: the unsplit bits).  At 64 rows gdn_out 16.3 -> 12.7 us, mlp_down
  36.4 -> 28.0 (the mma.sync split-K kernel before), gate|up 57.1 -> 53.8.
* **Result** (`kernels/gemm_table_h200.json`, `bench/cuda_gemm_micro.py --mma-max-m 64`: the mma.sync kernels timed
  only up to 64 rows, where they can still win): the TMA kernel for 169 of 175 shape x row-count entries.  1,240-1,420
  TOPS at 4,096-8,192 rows (gdn_out 1,070-1,110), 860-1,270 at 1,024, 660-1,140 at 256: 1.3-2.0x cuBLASLt's int8
  from 256 rows, 2.4x the mma.sync kernels.  At 64 rows the weight stream reaches 2.5-3.4 TB/s of the H200's 4.8
  (cuBLASLt 2.0-3.3).
* **int64 offsets.** The 27B's KV arena with the default 64-entry prefix cache holds 2.4 billion elements (16 attention
  layers x 2 x 4 KV heads x 74,000 rows x 256): the attention kernel added its layer offset (still under 2^31) and the
  head and key/value strides in 32 bits, which wrapped - an illegal address on the H200 (the RTX 5090 ran the 27B with
  `--prefix-cache 8`).  The H200's largest batch, 131,072 tokens, also takes [T, 2 x 17,408] and [T, 16,480] row
  offsets past 2^31 (at 61,681 and 130,313 tokens): every Triton kernel's row index is now int64 (index arithmetic
  only - the same bits); the CUDA kernels already used 64-bit row offsets.
* **The linear-attention chunk kernels' launch parameters per GPU** (`kernels/gdn.py`, `LAUNCH_BY_GPU`): on the H200
  the scan with 64-wide value tiles and 2 stages (RTX 5090: 32, 3) and the output kernel with 128-wide tiles and 1
  stage (64, 2): 31% and 21% faster on the 27B's 32-request batch, 12% and 15% on a 1,185-token request (a tile splits
  independent value columns: the same bits).
* **Attention tiles per GPU** (`kernels/attn.py`, `LAUNCH_BY_GPU`; the planner's query blocks follow,
  `batch.plan_batch(bm=)`): the RTX 5090's 32-query x 32-key tiles, sized for its 99 KB of shared memory, ran the
  27B's attention (24 heads of 256) at 126-141 TFLOPS on the H200 (`bench/attn_micro.py`); 128 x 64 with 8 warps and
  2 stages at 276 (4,000 tokens) and 366 (12,000).  A 50,000-token request 5.64 -> 3.67 s (SGLang with FlashAttention
  3: 4.94 s), 8,000 tokens 482 -> 441 ms.  The key tile sets the order of a row's softmax accumulation, so the H200's
  numbers differ in the last bits from the RTX 5090 tiles' (int8 against the fp32 reference: 23/23 decisions, largest
  probability difference 0.034); they stay the same for a request whatever its batch (prefix-cache hits bit-identical
  to misses).  The head's one-query blocks keep the 32 x 32 tiles everywhere.
* **SwiGLU + quantization as two power-of-2 pieces** (`kernels/quant.py`): one power of 2 for a whole gate|up row
  left 47% of the 27B's lanes masked (17,408 of 32,768); the row now sits in registers as 16,384 + 1,024 columns.  The
  same bits (the row's maximum is exact in any order); on the H200 1.3-1.6x faster from 384 rows (8,192 rows: 321 ->
  216 us, 3.3 TB/s), on the RTX 5090 1.1-1.2x at 384-1,280 rows and unchanged beyond.
* **End to end** (`bench/profile.py`, the 27B in int8 on the H200, CUDA graphs): yesno (55 computed tokens) 11.4 ms,
  startlux3 (361) 23.5 ms, longpage2 (1,185) 69.3 ms, a 32-request batch (6,317) 323.7 ms - against 36.1, 113.4 and
  652 ms on the bf16 path (cuBLAS's nvjet kernels) for the last three: 1.5x, 1.6x, 2.0x.  With the TMA kernel's first epilogue and a provisional table
  they were 33.5, 82.2 and 386 ms.  The batch's GPU time now: projections 71%, linear attention (chunk, scan, output,
  conv) 18%, SwiGLU + quantization 3.6%, norms 2.2%, attention 2.1%.
* **Build per GPU.** The extension is built for the GPU it runs on (`cuda_ext.cuda_arch()`: sm_120 on the RTX 5090,
  sm_90a on Hopper - the architecture-specific target wgmma and setmaxnreg need; `STARTLUX_PRISM_CUDA_ARCH` overrides), and
  the dispatch tables are per GPU model (`gemm_table_<gpu>.json`, `w4a8_table_<gpu>.json`).

## Images (2026-10-13)

* **One Triton kernel for the vision tower's RoPE** (`kernels/vision_rope.py`), with transformers' roundings, so the
  same bits: 19 -> 15 ms for a 1,024-token image.
* **Vision-tower graphs**: a CUDA graph per image size up to 576 tokens, recorded the second time a size appears (16
  sizes kept, one memory pool): 6.4 -> 2.1 ms at 64 tokens. Above 576 tokens a graph gained 5% or less: not kept.
* **Image decoding**: v1.1's base64 regular expression took 50 ms per 3 MB image; a strict decode runs first now, with
  the regular expression only as the fallback (12 ms), and a request's images decode in threads: about 600 -> 144 ms
  of decoding for eight 1024 x 1024 images.
* **The body graph is recorded before the image features are written**: recording runs the pass once, which turned the
  image tokens' rows into their residuals before the replay read them (probabilities about 0.5 too flat).
* **Measured and dropped: cuDNN attention in the tower** (7.0 against 10.4 ms per 1,024-token image over the 27 blocks):
  cuDNN chooses its kernel plan per process, so features and answers changed across restarts. The tower's attention is
  pinned to PyTorch's flash kernel.

## Tensor parallelism: one 27B over 2 or 4 H200s (2026-10-14)

`--tp N` (`startlux_prism/tp.py`), the int8 path, for latency: the bar was the single GPU's bits for every answer, cache hit,
batch and thought (`tests/test_tp.py`), and it holds at 2 and 4 GPUs.  Numbers: [benchmarks](benchmarks.md#tensor-parallelism-the-27b-on-2-and-4-h200s).

* **Shards** (Megatron-style): attention q|gate, k, v by head (a GPU's 12 or 6 query heads read its 2 or 1 kv heads),
  linear attention by key head with the value heads that read it (llama.cpp's tiled order: value head j reads key head
  j % 16, so a GPU holds value heads m x 16 + its key heads, and the kernels' `h % HK` still holds), MLP by column;
  o, out and down by input.  The decode output head by vocabulary (62,080 or 124,160 rows per GPU), gathered before the
  top-k (below).  Embedding, norms, the residual stream and the letter read-out on every GPU.
* **Exact by construction.**  SmoothQuant's per-channel smoothing and the per-row weight scales are computed on the
  whole matrices and then sliced, so a GPU's int8 weights are the single GPU's rows or columns.  An output projection
  stores int32 partial sums (`w8a8.gemm_i32`: the Triton kernel's new epilogue 3, the mma.sync tile kernel and the
  wgmma TMA pipeline with one or more int32 slots - the C++ took one-split int32 slots, nothing else changed); the sums
  of the GPUs are added as integers and the residual's next reader scales and adds them with the one-GPU epilogue's
  roundings.  The int8 input of an output projection needs its row's maximum over the whole row, which the GPUs
  exchange (below; a maximum is exact in any order).  `tests/test_cuda_gemm.py` folds every int32-capable kernel's
  per-GPU sums into the one-GPU bits (2 and 4 GPUs, 31 kernels each).
* **Found on the way: Triton's layouts set the order of a sum.**  The first two-GPU run was off on 1 row in 361 (a
  scale 64 ulps away) and the four-GPU kernel test on 83: (1) the producer's first launch stored only the row maximum
  and returned, so Triton chose its register layout from that one scalar store instead of the int8 row store - and the
  layout fixes the order of the gated norm's sum of squares over a head's 128 values, so the maxima came out an ulp off;
  now every launch runs the single-GPU kernel's code whole (int8 stores included) and adds one scalar store or load,
  and the gated norm keeps the whole model's 64-head block; (2) at 4 GPUs the input projection's rows are 4,120 values
  wide, which Triton does not specialize as divisible by 16 (16,480 is), and that changed the layouts of the kernels
  reading those rows: the buffer's rows are padded to a multiple of 16.  Both found by comparing per-layer residuals
  against one GPU, then the kernel alone on real inputs.
* **All-reduces in the kernels where they beat NCCL** (measured on the engine, pass by pass, both ways:
  `STARTLUX_PRISM_TP_FAST_ROWS`, `STARTLUX_PRISM_TP_PULL_ROWS`).  A layer has four all-reduces: two row maxima (a float per
  token) and two int32 sums (a 5,120-wide row per token).  Through NCCL inside the CUDA graphs they took 4.4 ms of a
  9.5 ms decode step at 4 GPUs (5.1 ms without them, `STARTLUX_PRISM_TP_ABLATE`).
* **Maxima as words (kept, passes of up to 16,384 tokens).**  The producer (gated norm, attention output, SwiGLU)
  writes its row's maximum, with the pass number in the upper half of a 64-bit word, into that row's slot on every
  other GPU, and reads theirs from its own buffer until each carries this pass's number: one launch, no flag, no
  barrier (a word is rewritten only by a later pass, after every GPU has finished this kernel).  Against NCCL's max
  between two launches of the producer, at 4 GPUs: a decode step 7.19 against 9.48 ms (one sequence) and 10.11
  against 12.90 (64 sequences); one prompt of 256 tokens 15.05 against 17.34 ms, of 8,192 tokens 241.3 against
  248.4; at 2 GPUs alike (8.25 against 10.29 ms for one sequence, 301.5 against 312.2 for 8,192 tokens).
* **Sums read by their consumer (kept for the shortest passes: up to 128 rows on 2 GPUs, 4 on 4).**  The output
  projection leaves its int32 slots in symmetric memory (torch's `_symmetric_memory`: every GPU's buffer mapped by
  every other) and the kernel that adds them into the residual (norm_q8 / fold) reads every GPU's slots itself over
  NVLink, after a handshake: its grid's first program sets this GPU's flag for the step on every other GPU (the pass
  number, from a counter every GPU bumps at the start of each pass; a release store, system scope), and every program
  waits for the others' flags (strong loads, then a fence).  Two buffers alternate, so one is rewritten only after
  every GPU has passed the next step's handshake, i.e. finished reading it.  It wins only where little data moves: at
  2 GPUs a decode step takes 7.70 ms against NCCL's 8.25 (one sequence) and 12.90 against 13.75 (64), a 64-token
  prompt 10.72 against 11.70 ms; 128 tokens tie (14.08, 14.27), 256 lose (18.38 against 17.40) and 1,024 lose badly
  (56.9 against 45.2).  At 4 GPUs, where every GPU reads three others' slots, only the smallest decode batches gain
  (one sequence: 6.51 against 7.19 ms); 8 sequences tie, 64 lose (14.63 against 10.11), and a 1,024-token prompt
  took 113.8 ms against 37.2.
* **The first version** exchanged the maxima through the same flags and a second launch of the producer, and read
  the sums in the consumer, on every pass of up to 512 rows: 15.1 ms per 64-sequence decode step and 38.2 ms for a
  256-token prompt at 4 GPUs (one GPU: 16.7 and 17.6 ms) - the 4-GPU prompt slower than one GPU by half.  Every
  program of a consumer signalled every other GPU with an atomic exchange and spun with atomic adds; one writer per
  flag (the first program, a plain release store) and plain loads for the wait made 4-GPU passes with in-kernel sums
  7% faster at 1,024 tokens (122.8 -> 113.8 ms) and changed nothing at 2 GPUs: kept for its smaller cost, but the
  sums' crossover with NCCL stayed where it was.
* **Dropped: a two-shot int32 all-reduce** (each GPU sums its 1/N of the elements from every GPU's slots and writes
  the total to every GPU, in one kernel with a barrier per program pair before and after; torch's symmetric-memory
  all-reduces - one-shot, two-shot, multimem - have no int32).  Alone, in graphs, against NCCL: 10-15% faster at 2
  GPUs from 128 rows (79.5 against 92.1 us at 1,024 rows), slower at 4 GPUs at every size (15.5 against 12.5 us at
  one row, 132 against 115 at 1,024).
* **Vocabulary-parallel decode head.**  The replicated head read the whole 1.27 GB Q8_0 head on every GPU per step;
  each GPU now computes its share of the logits (a logit is the same arithmetic whatever columns run with it), and a
  fused gather (the same flags) copies the other GPUs' shares into every GPU's full row before the top-k, so every GPU
  samples the same token from the same 248,320 logits.
* **Processes.** Rank 0 plans; a pass's tables go through pinned shared memory (`cudaHostRegister` on a /dev/shm
  mapping: every GPU copies them itself, no NCCL broadcast), its header through a shared ring (a follower polls it,
  and naps after 2 s idle).  Image features are encoded on rank 0 first (so nothing fails after the pass is published)
  and broadcast.  Every rank compiles its Triton kernels into a cache of its own: four ranks compiling the same kernel
  into one cache directory raced (a rank read another's half-written entry).  So does every worker of `startlux_prism.dp`
  (`--tp 2` on four GPUs: two rank 0s on one cold cache died the same way at start).  The CUDA extension's source directory
  is resolved to its real path: a follower that imported startlux_prism by another path (its PYTHONPATH) hashed another
  build command and rebuilt the extension, and two ranks rebuilding in turn waited on each other's build lock (a hang
  at start).
* **Dispatch tables for the shards** (`bench/cuda_gemm_micro.py --tp N`): the column-parallel shapes, and the output
  projections with int32 out (keys `NxKi`, a split charged with the fold reading every GPU's extra slots).  Without
  them the shards ran the Triton fallback: the 4-GPU JevBench run with thoughts took 80 s from 16 clients against 65 s
  on one GPU; with them and the all-reduces above, 41 s on 4 GPUs and 49 s on 2 (`tests/test_tp.py --server`).

## Opt-in

* **int8 attention scores** (`--attn-qk8`, `Options.attn_qk8`): keys also kept as int8 (one scale per token and kv
  head, written with the bf16 keys), queries quantized per row in the attention kernel, q k^T on int8 tensor cores
  (int8 runs at 4x the rate of bf16 with fp32 accumulation on this GPU); softmax and p v unchanged, batch-invariant
  (per-token scales). Attention 1.45-1.48x faster on long segments (`bench/attn_micro.py --modes bf16,q8`: 619 -> 426
  us at 3,000 tokens). End to end (`bench/ab.py`): -3.2% on the longest JevBench item, -1.3% longpage2, -0.4 to -1.2%
  elsewhere. Accuracy on all 231 public JevBench questions against the fp32 reference (`tests/accuracy_ab.py`):
  the same decision as the reference on 227 in both, but the probabilities move further from it - mean |dp| 0.0126
  -> 0.0148, p95 0.060 -> 0.064, max 0.117 -> 0.148. Calibrated probabilities being the product, off by default.
  Simulated first (`tests/quant_sim.py`, 30 questions): adding fp8 p v as well (`+pv8`) flipped a decision and raised
  the mean |dp| to 0.017, so it was not built.

## Tried, not kept

* **H200: short segments on small attention tiles** (a second attention launch per layer over the blocks of segments
  under 64 tokens, with the RTX 5090's 32 x 32 tiles, beside the 128 x 64 ones): a question of a few dozen tokens
  then computes no empty rows, but the extra launch cost more - A/B on the 27B (two rounds each, 200 requests per
  suite): startlux3 25.45 against 24.9-25.1 ms, startlux1 23.1 against 22.7, wide33 31.7 against 31.0; JevBench p50
  14.0-14.5 either way.
* **Stream-K and data-parallel + stream-K GEMM schedules** (Osama et al. 2023): correct (int32 partial tiles through a
  workspace, last arrival reduces), and chosen by the table for a few shapes, but no end-to-end gain: the per-SM main
  loop, not wave quantization, limits these GEMMs.
* **GQA-packed attention** (the 4 query heads of a KV head as one 64-row tile, 16-token blocks): 1.52 ms vs 1.39 ms for
  the plain 32-token kernel on longpage2; the 128-row variant does not fit in registers / shared memory.
* **Tokenizer thread pool on** (the default): see above.
* **Software-pipelined loads in the sequential GDN scan** (chunk rows computed instead of looked up, so the next
  chunks' loads can be issued early): 3.73 -> 3.60 ms on a 2,617-token prompt (kept, as it costs nothing); the scan is
  bound by its chain of dependent MMAs per chunk, not by loads. Narrower value tiles (more programs) were slower.
* **Keeping the GPU clock up** at one request at a time (2.27-2.33 GHz against 2.5-2.6 GHz under load): a one-warp
  spin kernel on its own stream between requests did not raise the clock and doubled yesno's p95.
* **L2 prefetch of a layer's weights** while its attention / linear-attention kernels run (gate|up weights, 47 MB,
  `l2_prefetch` kernel on a side stream forked and joined inside the graph; `Options.l2_prefetch`): the estimate was
  ~13 us per layer (the 64-row gate|up GEMM reaches ~2 TB/s from L2 against ~1.2 from DRAM). Measured in one process
  against the same engine without it (`bench/ab.py`, 100 interleaved pairs, identical bits): yesno +0.2%,
  startlux3 +2.6%, startlux1 +3.9%, readme3 +5.7%, a 480-token JevBench item +3.8%, long prompts +0.1% - slower:
  the prefetch overlaps the output projection's own weight stream and takes SMs from the small kernels. Off.
* **4-bit weights, 8-bit activations** (W4A8, simulated, `tests/quant_sim.py`, 30 questions): short requests stream
  weights from DRAM, so half the weight bytes would be the largest single gain for them - but NVFP4 weights (from the
  Q8_0 values) with the engine's int8 activations flipped a decision (29/30) and nearly doubled the mean |dp| (0.0233
  against 0.0126); MXFP4 weights with MXFP8 activations, what the block-scaled mma multiplies directly: 27/30, mean
  0.076. Not built. (Starting from the original bf16 checkpoint instead of Q8_0 might narrow the gap; untested.)
* **FP4 (W4A4)**, simulated in fp32 on the reference over 22 questions (`tests/quant_sim.py`): NVFP4 kept 22/22
  decisions but with mean |dp| 0.045 (max 0.21) against W8A8's 0.016 (0.12); with SmoothQuant 19/22 (max 0.80); MXFP4
  21/22 (max 0.93). Calibrated probabilities are the product, so the 2x FP4 tensor rate is not taken.

## Open

* Images: a deterministic attention kernel for the vision tower (the 3.4 ms per 1,024-token image cuDNN would save);
  base64 decoding holds the GIL (about 12 ms per 3 MB image); the tower runs on the scheduler's stream, so it does not
  overlap with the decoder's passes.
* Tensor parallelism: the int8 path only (bf16 and W4A8 run on one GPU). At 4 GPUs the all-reduces still take about
  two fifths of a prefill pass and a fifth to a third of a decode step, mostly NCCL's int32 sums of the output
  projections. The in-kernel thresholds (`STARTLUX_PRISM_TP_FAST_ROWS`, `STARTLUX_PRISM_TP_PULL_ROWS`) were measured on H200s on
  NVSwitch only.
* Hopper (H200): W4A8 still runs on mma.sync (Q4_K_M's projections at 200-300 TOPS against W8A8's 1,250 on wgmma:
  90% of the 27B's W4A8 GPU time); the GEMMs at up to 64 rows (2.5-3.4 TB/s of 4.8); the TMA kernel's epilogue
  still does not overlap the next tile's main loop (one CTA per SM: a persistent kernel with two MMA warpgroups taking
  turns would hide it); the linear-attention chunk kernels (18% of a large batch); attention at 366 TFLOPS of the
  H200's 989 (bf16; FlashAttention 3 reaches 600+ with wgmma and TMA).
* W4A8 (above): Q6_K, still 2.2-2.6x W8A8's time at 1,000+ rows - two scale multiply-adds per output and k32 step,
  and its 64x64 tiles' int8 weights near L2's bandwidth (larger tiles spill); deterministic split-K for small batches
  (the K reduction then defined in fixed chunks for every kernel, so still batch-invariant).  (The 27B's 48 value
  heads: the three Triton kernels that took the head count as a power of two now pad and mask it - 2026-10-11.)

* The fp32 residual add of the 2,560-wide output projections at large M: after batching its loads, still 10-15% of
  those GEMMs at 2,500-4,000 rows (the residual's DRAM traffic follows the tile's main loop instead of overlapping it;
  an L2 prefetch of the tile before the main loop did not help: it competes with the main loop's own reads).
* SwiGLU + quantization (2.6-3.9% of GPU time): a two-pass fusion into the gate|up epilogue (bf16 activations + per-row
  maxima, then a lighter quantize pass) - about 2-3% on long prompts (measured above).
* Short passes: ~200 small per-layer kernels take about a third of a 55-token pass (the linear-attention chunk
  kernels 0.67 ms of 4.1); the 64-row GEMMs stream weights at 1.3 TB/s of 1.70.
* The host path inside the server: 0.25-0.6 ms per request, most of it the tokenizer (Rust) and the planner (numpy).
* Attention at about 55% of the bf16 tensor-core peak (bf16 with fp32 accumulation runs at half rate on this GPU):
  7.6 ms of a 3,900-token pass.
