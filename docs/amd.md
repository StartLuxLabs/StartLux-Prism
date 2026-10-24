# Experimental ROCm single-GPU backend

This is an experimental W4A8 path for Radeon AI PRO R9700 (`gfx1201`, 32 GB).
The initial checkpoint is `StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf`
(16,547,400,096 bytes). Other architectures and full-model BF16/W8A8 are not
qualified. ROCm tensor/data parallelism is deliberately rejected.

This draft depends on the separate fixes for
[lazy decode graph capture](https://github.com/StartLuxLabs/StartLux-Prism/pull/17)
and [omitted thought budgets](https://github.com/StartLuxLabs/StartLux-Prism/pull/18).
Apply those fixes before the complete AutoThink/graph checks below. The backend
diff itself does not duplicate either fix.

## Environment and launch

Use an existing ROCm build of PyTorch and its matching Triton build. Install the
application dependencies with `pip install -r requirements-rocm.txt`.
`requirements.txt` selects CUDA packages and is not the ROCm installation entry.
The validation stack is recorded below; this PR does not install the ROCm runtime
or provide a prebuilt container.

The model directory needs its matching tokenizer, release configuration and chat
template, as in the main README. Set one **HIP ordinal** before Python starts:

```bash
MODEL=/path/to/StartLux-Decision-v1.1-27B-AutoThink-GGUF
HIP_VISIBLE_DEVICES=0 python -m startlux_prism.server \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --precision w4a8 --mmproj none \
  --max-tokens 2048 --prefix-cache 4 --prefix-cache-tokens 4096 \
  --think-seqs 1 --max-think-budget 256 --think-kv-rows 4096 \
  --context-length 2048 --no-graphs
```

`0` is an example HIP ordinal, not a portable physical/PCI index. HIP enumeration
can differ from management-tool indices. The process must see exactly one GPU;
inside it the device is `cuda:0`, including on ROCm. There is no machine-specific
card reservation policy in the backend.

The conservative ROCm defaults are 2,048 packed tokens, four prefix slots with
4,096 KV rows, one thinking sequence, a 256-token thought limit and eager mode.
Packed capacity includes system/evidence/question tokens. The buffer preflight
keeps a 2 GiB reserve and explains which capacities to reduce when it cannot fit.
`GET /health` identifies the actual backend/architecture and graph setting.

Opt into graphs with `--graphs`; buckets above `--graph-max-tokens` stay eager.
The initial graph cap is 512 tokens. Graphs were not consistently faster in shared
GPU measurements. `--no-graphs` also disables vision graph recording on ROCm.

For images, pass the matching `mmproj-*-F16.gguf` with `--mmproj`. Small-image
1/2/8-image checks do not imply support for eight maximum-resolution images.
The 8K prefill capacity check does not qualify 8K AutoThink.

## Kernels and numerical scope

The backend avoids CUDA extensions, PTX, PDL, WGMMA and NVIDIA tuning tables.
Q4_K/Q6_K remain compressed; there is no full-model BF16 expansion or activation
calibration. Small row counts use an integer-reduction GEMV; larger rows use the
existing Triton GEMM with gfx1201 tiles. `STARTLUX_PRISM_AMD_GEMM=reference` forces
the basic GEMM for diagnosis; `gemv` is the default. CUDA cfg IDs are rejected.

Explicit FP32 operations preserve W4A8 rounding/order. GDN and attention use
gfx1201 resource settings. Canonical evidence boundaries preserve GDN chunk
roundings when sharing/cache changes; the NVIDIA planner default is unchanged.
Q8 head/embedding conversion uses a smaller temporary chunk on HIP only.

**The independent FP32 readout gate is unmet.** The prototype's frozen 501-row
comparison produced MAE 0.003109, P99 0.023216, and 452/457 (98.906%) matching
decisions where reference margin exceeds 0.02. Local candidate thresholds were
0.001, 0.01 and 99.5%; these are not upstream's established acceptance criteria.
The sample set is six synthetic templates plus a yes/no anchor, not a graded
model-quality benchmark. No threshold was relaxed.

An independent activation-INT8-only diagnostic also exceeds those thresholds,
but remaining differences still need attribution. These results do not prove
that all differences are inherent quantization error, nor that the AMD port adds
no error. CUDA hardware regression and a comparable upstream W4A8 numerical
baseline remain open. Do not describe this as production-qualified AMD support.

## Reproduce checks

Run each GPU command with only the selected GPU visible. The tests assume the
ROCm PyTorch/Triton environment above. Optional independent-reference checks use
`gguf==0.19.0`; the vision reference uses Transformers' Qwen3.5 implementation.

```bash
python -m pytest -q tests/test_backend_guards.py
python tests/test_canonical_batch.py
python tests/test_amd_backend.py --cpu       # independent gguf-py block dequantization
HIP_VISIBLE_DEVICES=0 python tests/test_amd_backend.py
mkdir -p results/rocm
HIP_VISIBLE_DEVICES=0 python scripts/amd/test_graph_cache.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --output results/rocm/graph-cache.json --reps 3
HIP_VISIBLE_DEVICES=0 python scripts/amd/test_decode_model.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --output results/rocm/decode.json --graphs
```

`scripts/amd/test_vision_model.py` additionally takes `--mmproj` and checks actual
27B image/cache/graph/mixed batches and the synthetic visual suite.
`tests/vision_reference.py encoder` compares the vision tower to Transformers.

To reproduce the independent FP32 comparison, first generate an oracle with
`scripts/amd/stream_reference.py --gguf FILE --model-dir DIR --output oracle.jsonl
--count 500`, then run `scripts/amd/validate_reference.py --gguf FILE --model-dir
DIR --oracle oracle.jsonl --output comparison.json` on the same card, sequentially.
Only one FP32 layer is resident at a time in the diagnostic oracle. The candidate
engine still keeps its inference weights/compute on the GPU. The comparison
intentionally exits nonzero when the original candidate thresholds fail.

Set `PRISM_SHARED_GPU=1` when recording tests alongside other workloads. Timings
from such runs are provisional. See [validation details](amd_validation.md).
