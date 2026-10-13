#!/bin/bash
# Re-run comparison entries that produced no result (a server that crashed), up to 3 attempts each, keeping every
# attempt's server log and a Python stack at any segfault (PYTHONFAULTHANDLER).
#
#   setsid bash scripts/rerun_failed.sh "engine[:ENV=value ...]" ... > ~/jev/logs/rerun.out 2>&1 < /dev/null &
#   e.g. scripts/rerun_failed.sh sglang decider "decider-real:DECIDER_ENV=DECIDER_COMPILE=1 DECIDER_FP8=1|DECIDER_TAG=-compile-fp8"
set -uo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
export TAG=${TAG:--free} PYTHONFAULTHANDLER=1
mkdir -p "$JEV/logs/attempts"
for spec in "$@"; do
  engine=${spec%%:*}
  envs=""
  [ "$spec" != "$engine" ] && envs=${spec#*:}
  for attempt in 1 2 3; do
    start=$(date +%s)
    echo "=== $engine attempt $attempt [$envs] ($(date +%H:%M:%S))"
    (IFS='|'; for kv in $envs; do export "$kv"; done; bash scripts/compare.sh "$engine")
    for f in decider sglang adapter-sglang vllm jevk5 startlux_server; do
      [ -f "$JEV/logs/$f.log" ] && [ "$(stat -c %Y "$JEV/logs/$f.log")" -ge "$start" ] && \
        cp "$JEV/logs/$f.log" "$JEV/logs/attempts/${engine}-${attempt}-$f.log"
    done
    # done when both result files were written by this attempt
    fresh=$(find results/jevbench results/http -name "*.json" -newermt "@$start" | wc -l)
    [ "$fresh" -ge 2 ] && { echo "--- $engine: results written"; break; }
    echo "--- $engine attempt $attempt: no complete result"
  done
done
echo "reruns done $(date +%H:%M:%S)"
