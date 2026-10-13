#!/bin/bash
# The full engine comparison on a GPU of its own: wait until no process computes on the GPU for a minute (another
# tenant's workload would skew every number), then run every engine through scripts/compare.sh, each in its strongest
# configuration measured here, and flash-jev once more at the end to show the conditions held.
#
#   setsid bash scripts/compare_when_free.sh > ~/jev/logs/compare-all.out 2>&1 < /dev/null &
#
# Labels get TAG (default -free).
set -uo pipefail
cd "$(dirname "$0")/.."
export TAG=${TAG:--free}
free=0
while [ "$free" -lt 6 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then free=$((free + 1)); else free=0; fi
  sleep 10
done
echo "GPU free since $(date -d '-60 seconds' +%H:%M:%S): $(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader)"
bash scripts/compare.sh flash-jev-int8 flash-jev-bf16 startlux sglang sglang-native vllm llamacpp decider decider-real jevk5 \
  jevk5-real
# decider's opt-in torch.compile and FP8 linears (its model card's fastest settings)
DECIDER_ENV="DECIDER_COMPILE=1 DECIDER_FP8=1" DECIDER_TAG=-compile-fp8 bash scripts/compare.sh decider-real decider
TAG=-free-recheck bash scripts/compare.sh flash-jev-int8
echo "comparison done $(date +%H:%M:%S)"
