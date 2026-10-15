# How StartLux-Prism works

The engine's design and a map of the code. The short version is in the [README](../README.md); what was measured, kept and dropped is in [optimizations.md](optimizations.md).


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

**Kernels per GPU.** The extension is built for the GPU it runs on (sm_120, sm_90a, or sm_80 / 86 / 89 for Ampere and
Ada). Each projection's kernel, tile and K split is chosen per shape and token count from a table measured on that GPU
model (`kernels/gemm_table_<gpu>.json`):
- **RTX 5090:** mma.sync kernels (one CTA per tile, split-K, stream-K) and a Triton kernel.
- **Ampere and Ada (A800, RTX 30 and 40 series):** no table measured yet - the int8 GEMMs run the Triton kernel, W4A8
  picks its tiles by rule - and no PDL or thread-block clusters (Hopper's), the rest of the code as on the others.
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
| `startlux_prism/engine/model.py`, `engine/batch.py` | the engine and its graphs; the forest planner |
| `kernels/gdn.py` | conv + prep, the WY chunk kernel, the two-wave state scan, the gated norm |
| `kernels/attn.py` | QK norm + RoPE + KV arena, tree flash attention |
| `kernels/w8a8.py`, `kernels/quant.py`, `csrc/` | the int8 and W4A8 paths |
| `reference.py` | a plain fp32 PyTorch forward, the yardstick for every kernel |
| `decide.py`, `think.py`, `engine/decode.py`, `loops.py` | requests and answers; thoughts |
| `images.py`, `vision.py` | images |
| `server.py`, `dp.py`, `tp.py` | HTTP and scheduler; several GPUs: data and tensor parallelism |

What was kept, measured, tried and dropped, with numbers: [optimizations.md](optimizations.md).
