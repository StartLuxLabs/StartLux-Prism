#!/bin/bash
# Bring up another StartLux-Decision GGUF model on this GPU (the 9B: hidden 4096, feed-forward 12288, untied output
# head; the same linear-attention and attention heads as the 4B):
#   1  the comparison's model folders (scripts/prepare_model.sh, CPU) while the GEMM dispatch table gets entries for its
#      projection shapes (bench/cuda_gemm_micro.py --gguf --buckets --max-m 8192 --merge-into, GPU)
#   2  correctness: the CUDA GEMM against exact integer math and the Triton kernel's bits (tests/test_cuda_gemm.py, the
#      9B's shapes included), the engine against the fp32 reference in int8 and bf16 (tests/test_engine.py; the first
#      int8 engine computes the SmoothQuant activation ranges and keeps them in CALIB), and the prefix cache bit for bit
#      (tests/test_prefix_cache.py --exact)
# For a k-quant file (Q4_K_M), the W4A8 path too: its GEMM table entries for the model's projection shapes
# (bench/w4a8_table.py, merged into flash_jev/kernels/w4a8_table_<gpu>.json; every config's time in
# results/w4a8_table_times_<gpu>.json), its GEMM bit for bit (tests/test_w4a8.py), the engine and the prefix cache in w4a8.
# Then the stress comparison for the model: scripts/stress.sh with the environment prepare_model.sh prints.
#
#   MODEL=~/jev/models/StartLux-Decision-9B-Q8_0-GGUF bash scripts/new_model.sh
#
# The dispatch tables are repository files per GPU model (flash_jev/kernels/gemm_table_<gpu>.json, w4a8_table_<gpu>.json,
# <gpu> = rtx5090, h200, ...; a new GPU starts without them): copy them back after this.
set -uo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
source "$JEV/env.sh"
MODEL=${MODEL:?MODEL: the GGUF repository directory}
GGUF=${GGUF:-$(ls "$MODEL"/*.gguf | grep -v "/mmproj-" | head -1)}  # not the vision tower (mmproj-*.gguf)
NAME=$(basename "$GGUF" .gguf)
CALIB=${CALIB:-$JEV/calib/$NAME.pt}
JB=$JEV/src/jevbench-7ce310c7/datasets/public
GPU_TAG=$(python -c "from flash_jev.kernels.cuda_ext import gpu_tag; print(gpu_tag())")   # rtx5090, h200: the tables
# Hopper: the wgmma kernels win above a few rows, so the mma.sync ones are timed only up to 64
MMA_MAX=$(python -c "from flash_jev.kernels.cuda_ext import has_wgmma; print('--mma-max-m 64' if has_wgmma() else '')")
mkdir -p results "$JEV/logs"

echo "=== $NAME: model folders and dispatch table entries ($(date +%H:%M:%S))"
MODEL="$MODEL" bash scripts/prepare_model.sh > "$JEV/logs/prepare_$NAME.log" 2>&1 &
prep=$!
python bench/cuda_gemm_micro.py --gguf "$GGUF" --buckets --max-m 8192 $MMA_MAX \
  --merge-into "flash_jev/kernels/gemm_table_$GPU_TAG.json" --json "results/gemm_$NAME.json" 2>&1 \
  | grep --line-buffered -v Warning > "$JEV/logs/gemm_$NAME.log"
tail -2 "$JEV/logs/gemm_$NAME.log"
wait "$prep" && tail -2 "$JEV/logs/prepare_$NAME.log" || { echo "prepare_model.sh failed:"; tail -20 "$JEV/logs/prepare_$NAME.log"; }

echo "=== $NAME: correctness ($(date +%H:%M:%S))"
python tests/test_cuda_gemm.py --quick 2>&1 | grep -v Warning > "$JEV/logs/gemm_test_$NAME.log"
echo "test_cuda_gemm: $(tail -1 "$JEV/logs/gemm_test_$NAME.log"); bits = Triton: $(grep -c "bits = Triton" "$JEV/logs/gemm_test_$NAME.log") configs"
for prec in int8 bf16; do
  echo "--- test_engine $prec"
  python tests/test_engine.py --gguf "$GGUF" --model-dir "$MODEL" --jevbench "$JB" --precision "$prec" --calib "$CALIB" \
    2>&1 | grep -v Warning | tail -2
done
echo "--- test_prefix_cache --exact"
python tests/test_prefix_cache.py --gguf "$GGUF" --model-dir "$MODEL" --precision int8 --calib "$CALIB" --jevbench "$JB" \
  --n 4 --exact 2>&1 | grep -v Warning | cut -c1-200 | tail -3
KQ=$(python -c "import sys; from flash_jev.gguf import GGUFFile, GGML_Q4_K, GGML_Q6_K; \
print(int(GGUFFile(sys.argv[1]).tensors['blk.0.ffn_gate.weight'].type in (GGML_Q4_K, GGML_Q6_K)))" "$GGUF")
if [ "$KQ" = 1 ]; then
  echo "=== $NAME: W4A8 GEMM table entries and correctness ($(date +%H:%M:%S))"
  python bench/w4a8_table.py "$GGUF" --merge "results/w4a8_table_times_$GPU_TAG.json" \
    --out "results/w4a8_table_times_$GPU_TAG.json" --table "flash_jev/kernels/w4a8_table_$GPU_TAG.json" 2>&1 \
    | grep -v Warning > "$JEV/logs/w4a8_table_$NAME.log"
  tail -1 "$JEV/logs/w4a8_table_$NAME.log"
  echo "test_w4a8: $(python tests/test_w4a8.py "$GGUF" 2>&1 | grep -v Warning | tail -1)"
  echo "--- test_engine w4a8"
  python tests/test_engine.py --gguf "$GGUF" --model-dir "$MODEL" --jevbench "$JB" --precision w4a8 2>&1 \
    | grep -v Warning | tail -2
  echo "--- test_prefix_cache --exact (w4a8)"
  python tests/test_prefix_cache.py --gguf "$GGUF" --model-dir "$MODEL" --precision w4a8 --jevbench "$JB" --n 4 \
    --exact 2>&1 | grep -v Warning | cut -c1-200 | tail -3
fi
echo "=== $NAME: done ($(date +%H:%M:%S))"
