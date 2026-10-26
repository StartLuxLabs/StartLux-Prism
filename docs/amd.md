# Experimental single-GPU ROCm support

The backend supports Radeon AI PRO R9700 (`gfx1201`, 32 GB) and the official
9B/27B AutoThink Q4_K_M checkpoints. Weights remain compressed and resident on one
GPU. The full generation head is retained. Tensor/data parallelism, other AMD
architectures, Instant/bidirectional prompts and FP8 KV are outside this profile.

## Setup and launch

Install a matched ROCm PyTorch/Triton environment, then
`pip install -r requirements-rocm.txt`. The validated stack is ROCm 10.1,
Torch `2.14.0+rocm10.1.0`, Triton `3.8.0+git669b31ac.rocm10.1.0` and HIP
`7.16.26385`. The NVIDIA `requirements.txt` is not the ROCm installation entry.

Select exactly one HIP ordinal before Python starts. HIP ordinals can differ
from physical GPU indices; `0` below is an example. Use the matching tokenizer
and model configuration in `MODEL`.

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

For the 27B, start with `--max-tokens 2048 --context-length 2048 --think-seqs 1
--think-kv-rows 4096 --no-graphs` and its corresponding GGUF/model directory.
Without explicit capacity flags, ROCm uses these smaller defaults. Eager startup
compiles common buckets before serving; `--graphs` enables graph capture.

## Precision and dispatch

Q4_K_M automatically selects W4A8. Select `--precision w4a16` explicitly for the
profile that passes the local frozen FP32 probability thresholds. W4A16 uses FP16
activations/intermediates and the original compressed embedding; the generation
head still uses Q8. W4A8 does not pass those probability thresholds.

The reference uses the same quantized GGUF, not an unquantized checkpoint. This
is experimental support with local qualification, not a general AMD accuracy
claim. Eighteen W4A16 shape keys whose original path exceeds the independent
operator bound retain their original fallback. No numerical bound is relaxed.

Offline tables check architecture, exact runtime versions, kernel identity,
shape, dtype, stride and alignment. Unmatched cases use the original fallback.
Table entries include the measured 9B small-batch shapes that previously fell
back to row-wise GEMV. There is no runtime autotuning or full-weight expansion.
`STARTLUX_PRISM_AMD_W4A16=reference` and `STARTLUX_PRISM_AMD_GEMM=reference`
select the respective original projection paths. Restart the process after a
dispatch change so that existing graphs cannot retain a different selection.

## Core checks

Run each command in the ROCm environment with one GPU visible:

```bash
python -m pytest -q tests/test_backend_guards.py tests/test_canonical_batch.py \
  tests/test_amd_w4a8_dispatch.py tests/test_amd_w4a16_dispatch.py \
  tests/test_eager_warmup.py tests/test_decode_graph_capture.py tests/test_think_budget.py
python tests/test_amd_backend.py
python tests/test_amd_tuned_kernels.py
python tests/test_amd_w4a8_dot.py
python tests/test_amd_q6_wmma.py
python tests/test_amd_w4a16_meta_extended.py
```

The PR description links the separately preserved benchmark tools, numerical
evidence and full reports, including the RTX 4090 comparison. Those experiment
artifacts are outside the core implementation diff. NVIDIA hardware regression
and 9B visual-input qualification remain open.
