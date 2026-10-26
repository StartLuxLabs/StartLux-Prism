#!/usr/bin/env bash
# Usage: scripts/serve_mi355x.sh MODEL_DIR GGUF_BASENAME_OR_DOT [server options...]
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 MODEL_DIR GGUF_BASENAME [server options...]" >&2
    exit 2
fi
model_dir=$(realpath -- "$1")
gguf=$2
shift 2
if [[ "$gguf" == "." && -f "$model_dir/model.safetensors.index.json" ]]; then
    source_arg=/models
elif [[ "$gguf" != "$(basename -- "$gguf")" || ! -f "$model_dir/$gguf" ]]; then
    echo "GGUF_BASENAME must name a model file inside MODEL_DIR" >&2
    exit 2
else
    source_arg="/models/$gguf"
fi
image=${STARTLUX_PRISM_ROCM_IMAGE:-startlux-prism:mi355x}
exec docker run --rm --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    -v "$model_dir:/models:ro" \
    -v "${STARTLUX_PRISM_ROCM_CACHE:-startlux-prism-mi355x-cache}:/root/.cache" \
    "$image" --gguf "$source_arg" --model-dir /models "$@"
