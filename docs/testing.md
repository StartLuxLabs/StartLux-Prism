# Correctness and tests

How StartLux-Prism's answers are checked against the models' own packages and an fp32 reference, and how to run the tests and benchmarks. The short version is in the [README](../README.md).

## Correctness

- **Against an fp32 reference forward** (`tests/test_engine.py`, 23 questions):
  - bf16 makes the reference's decision on all 23 on every model (largest probability difference 0.003-0.005);
  - int8 on 22 (4B: one near-tie flips) or 23 (9B, 27B), largest difference 0.034-0.043;
  - W4A8 on 23 (4B) and 22 (9B), 0.031-0.032.
- **Bit for bit:**
  - every GEMM config, split mode and stream-K gives the Triton kernel's bits, the wgmma kernels included
    (`tests/test_cuda_gemm.py`);
  - prefix-cache hits equal recomputing (`tests/test_prefix_cache.py --exact`);
  - 64 answers sent by 32 clients at once equal the same requests one at a time (`tests/test_stress.py`);
  - the answers through the data-parallel router equal one GPU's.

  The bf16 path is deterministic run to run, but cuBLAS picks its kernels by batch size, so it is not batch-invariant.
- **StartLux-Decision v1.1** (`tests/test_v11.py`):
  - 5,787 random prompts render to the same text and tokens as v1.1's package;
  - 4,000 random requests (choice lists up to 300 options, every none_check / allow_none combination) get the same
    answers, every float, from the same letter logits;
  - v1 folders answer exactly as before v1.1 support.
- **Thoughts** (`tests/test_think.py`, `tests/test_decode.py`):
  - v1.1's loop detector is reproduced on 5,000 texts;
  - 2,885 thinking and read prompts, 899 with images, are identical in text and tokens;
  - 300 whole requests with 499 thoughts give the same answers, details and usage as v1.1's pipeline, to the float;
  - decode steps agree with the prefill under teacher forcing (int8, bf16 and W4A8);
  - sequences alone, batched or joining a running batch give the same tokens and logits, bit for bit (int8 and W4A8);
  - JevBench from 64 clients gives the same thoughts as one request at a time.
- **Images** (`tests/test_vision.py`, `tests/test_vision_engine.py`, `tests/vision_reference.py`):
  - the tower's features are bit-identical to transformers' on 9 image sizes;
  - 36 prompts with 1 to 8 images are token-identical to v1.1's, with its positions;
  - the 53 questions of 24 image requests get transformers' decisions, all right;
  - text answers stay bit-identical with the tower loaded.
- **Tensor parallelism** (`tests/test_tp.py`, the v1.1 27B on 2 and 4 GPUs against one, bit for bit):
  - the letter logits of 30 requests alone, again from the prefix cache and as one 7,492-token batch;
  - long evidence as a miss, a hit, another question about it and inside mixed batches; a 33-option list;
  - six thoughts' 64 decode steps, greedy, joining a running batch and sampled (each step's 248,320 logits and token);
  - 24 questions about images; on the v1 27B, the same 30 requests;
  - JevBench with thoughts through the server (231 items from 16 clients, 67 of them thinking): the same answers,
    probabilities and thoughts, to the last character;
  - every int32-capable GEMM's per-GPU sums, added and folded, give the one-GPU bits (`tests/test_cuda_gemm.py`).

## Tests and benchmarks

On the GPU server, from the repository (`scripts/sync.sh` copies the working tree there):

```bash
M=~/jev/models/StartLux-Decision-4B-Q8_0-GGUF; JB=~/jev/src/jevbench-7ce310c7
python tests/test_engine.py --gguf $M/StartLux-Decision-4B-Q8_0.gguf --model-dir $M --jevbench $JB/datasets/public
python tests/test_stress.py --model-dir $M --jevbench $JB --start int8 [--quick] --out results/stress/startlux-prism-int8.json
python bench/run_jevbench.py --url http://127.0.0.1:8092/v1/systemone --label StartLux-Prism --out results/jevbench/x.json
python bench/charts.py                                                        # docs/images
```

| Test | Checks |
|---|---|
| `tests/test_engine.py` | every precision against the fp32 reference forward (`--repeat`: run-to-run bits) |
| `tests/test_cuda_gemm.py` | the CUDA GEMMs against exact integer math and the Triton kernel's bits |
| `tests/test_w4a8.py`, `tests/test_kquant.py` | the W4A8 GEMM against its reference arithmetic; k-quant dequantization against gguf-py, bit for bit |
| `tests/test_kv_fp8.py` | the e4m3 KV cache (`--kv-fp8`, sm_89 or newer): its writes against the bf16 ones; tree attention (causal and bidirectional) and decode attention against fp32 attention over the same dequantized keys and values |
| `tests/test_prefix_cache.py --exact` | cache hits bit for bit |
| `tests/test_batch.py`, `tests/test_prompt.py` | the forest planner; prompts byte for byte against the chat template |
| `tests/test_v11.py` | StartLux-Decision v1.1 on the CPU: prompts and answers against its own package (`STARTLUX_DECIDE` = its checkout); v1 folders against the code before v1.1 support |
| `tests/test_think.py`, `tests/test_decode.py` | thoughts against v1.1's package (loop detector, prompts, whole requests with fake engines); decode steps against the prefill (teacher forcing, isolation, sampling) |
| `tests/test_vision.py`, `tests/test_vision_engine.py`, `tests/vision_reference.py` | images: placement, pixels, tokens and positions against v1.1 and transformers; the tower and image batches on the GPU; answers against transformers' model |
| `tests/test_dp.py` | the data-parallel router on the CPU, against fake workers: evidence affinity, routing by load, failover, `--tp` groups |
| `tests/test_tp.py` | tensor parallelism against one GPU, bit for bit: read-outs, cache hits, batches, wide choices, decode logits and tokens, images (`--tp 2 --gpus 2,3 --ref-gpu 1`); `--server`: JevBench with thoughts through a `--tp` server and a one-GPU server |
| `tests/test_stress.py` | a running server under load: long evidence counted in full, the capacity limit refused cleanly, many clients without a failed request, bit-identical answers under load |
| `scripts/accuracy_v1.sh` | StartLux-Decision-v1's evaluation suites through every StartLux-Prism precision and StartLux's own server |
| `bench/run_jevbench.py`, `bench/prefill_speed.py`, `bench/decode_speed.py`, `bench/vision_latency.py` | JevBench through any `/v1/systemone` server (latency, throughput, agreement); prefill and decode on the engine itself (`--tp N` too); images |
