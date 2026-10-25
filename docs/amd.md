# Experimental ROCm single-GPU backend

This is an experimental single-GPU backend for Radeon AI PRO R9700 (`gfx1201`, 32 GB).
W4A8 remains the automatic Q4_K_M profile; compressed W4A16 is an explicit option.
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

For a Docker service, configure network access explicitly. Host networking keeps
this loopback URL reachable from the host. With bridge networking, publish the
port and bind the server to a reachable container interface instead.

The conservative ROCm defaults are 2,048 packed tokens, four prefix slots with
4,096 KV rows, one thinking sequence, a 256-token thought limit and eager mode.
Packed capacity includes system/evidence/question tokens. The buffer preflight
keeps a 2 GiB reserve and explains which capacities to reduce when it cannot fit.
`GET /health` identifies the actual backend/architecture and graph setting.

Opt into graphs with `--graphs`; buckets above `--graph-max-tokens` stay eager.
The initial graph cap is 512 tokens. Five-repeat fixed-workload decode measurements
on the selected R9700 showed little additional benefit from graphs.
`--no-graphs` also disables vision graph recording on ROCm.

For images, pass the matching `mmproj-*-F16.gguf` with `--mmproj`. Small-image
1/2/8-image checks do not imply support for eight maximum-resolution images.
The 8K prefill capacity check does not qualify 8K AutoThink.

## Explicit W4A16 profile

Change the launch argument to `--precision w4a16` to select the separately measured
FP16-activation profile. It retains compressed weights and the full Q8 generation
head. It changes projection/intermediate/embedding numerical semantics and passes
the local frozen FP32 readout thresholds. Single-sequence decode remains slower
than the optimized W4A8 path, and several engineering speed goals are unmet.
See [the W4A16 scope, measurements and reproduction commands](amd_w4a16.md).

The W4A8 controls and FP32 limitation below describe W4A8. For W4A16,
`STARTLUX_PRISM_AMD_W4A16=auto` selects the qualified table and `reference` restores
its original FP16 projection implementation. The measured component setting is
`STARTLUX_PRISM_AMD_Q_COMPONENTS=all` (the default). Keep precision profiles
separate when comparing outputs and speedups.

## Kernels and numerical scope

The backend avoids CUDA extensions, PTX, PDL, WGMMA and NVIDIA tuning tables.
Q4_K/Q6_K remain compressed; there is no full-model BF16 expansion or activation
calibration. `STARTLUX_PRISM_AMD_GEMM=auto` is the default. An offline table selects
packed signed-byte dot4 for qualified small-row projections, Q4 WMMA with compact
minimum correction, or Q6 WMMA with two exact integer planes. The table checks the
architecture, exact Torch/Triton/HIP versions, kernel source identity, dimensions,
output layout and pointer alignment. Unknown cases retain the original GEMV/GEMM
path; there is no online autotuning or device-scalar read in dispatch.

`STARTLUX_PRISM_AMD_GEMM=gemv` restores the original small-row GEMV / larger-row
GEMM policy. `reference` forces the basic GEMM. The `dispatch`, `decode` and
`prefill` modes isolate one family for diagnosis. CUDA cfg IDs are rejected.
Changing a mode does not change kernels already captured in a graph: restart the
process or recapture the graphs when comparing modes.

AMD eager startup compiles common empty body/readout buckets and restores live
counts. `gpu_ms` remains the original launch-to-result host duration for API
compatibility; `launch_to_logits_ms` names it explicitly. `stream_elapsed_ms`
measures the HIP event span, including stream gaps; it is not pure kernel time.
Queued requests additionally report their own `queue_to_prepare_ms` and the
number of groups in their batch.

Explicit FP32 operations preserve W4A8 rounding/order. GDN and attention use
gfx1201 resource settings. Canonical evidence boundaries preserve GDN chunk
roundings when sharing/cache changes; the NVIDIA planner default is unchanged.
Q8 head/embedding conversion uses a smaller temporary chunk on HIP only.

**The W4A8 independent FP32 readout gate is unmet.** The prototype's frozen 501-row
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

## Reproduce acceleration measurements

Keep the complete generation head resident and compare modes on the same selected
card. The fixed decode benchmark uses 512 prompt tokens, 8 warm-up steps, 64 timed
steps, five alternating pairs for B=1/4/8, and ignores EOS. It checks every token
against all repetitions, graph modes and individually executed sequences.

```bash
HIP_VISIBLE_DEVICES=0 python bench/amd_decode.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --out results/rocm/decode-paired --paired
HIP_VISIBLE_DEVICES=0 python bench/amd_ablation.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --out results/rocm/ablation
HIP_VISIBLE_DEVICES=0 python bench/amd_http.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --out results/rocm/http
```

The HTTP protocol starts and stops its own localhost server, uses unique evidence,
retains every response and runs three rounds per scenario. It includes full
8192-token prompts, c1–64 sending windows of at least 30 seconds, cache sessions,
and actual AutoThink budgets 16/64/256. Budget limits are not generated-token
counts. Run it sequentially with other GPU benchmarks. It uses an explicit 8K
readout configuration; this does not qualify 8K AutoThink or change the 2K service
default.

`bench/amd_build_table.py` reconstructs the offline table from completed dispatch,
decode and prefill stage directories (`--dispatch`, `--decode`, `--prefill`,
`--output`, `--selection`). The source JSON hashes are embedded in the table.
Promotion requires cold/hot gains of at least 10%, plus the recorded partial-row
check where applicable; model-level validation remains mandatory.

The sustained check starts and stops its own localhost server and exercises real
thoughts, cache eviction and optional images. To reproduce the standard 2K run
with the matching projector and complete generation head:

```bash
HIP_VISIBLE_DEVICES=0 python bench/amd_stability.py \
  --gguf "$MODEL/StartLux-Decision-v1.1-27B-AutoThink-Q4_K_M.gguf" \
  --model-dir "$MODEL" --mmproj "$MODEL/mmproj-StartLux-Decision-v1.1-27B-AutoThink-F16.gguf" \
  --seconds 1800 --output results/rocm/stability.json
```

The check validates control outputs and drained allocator growth. GPU fault
visibility also depends on access to runtime and host kernel logs.
