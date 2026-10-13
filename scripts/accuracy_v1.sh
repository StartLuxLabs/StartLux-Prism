#!/bin/bash
# StartLux-Decision-v1's accuracy evaluation (eval/suites.py: the seven Intern-Decision suites + the calibration pilot)
# through each engine's /v1/systemone, on the RTX 5090: flash-jev int8 / bf16 / w4a8 against StartLux-Decision's own
# server (the reference implementation) on the same weights.  One request at a time (as the published numbers), plus
# flash-jev int8 from 8 concurrent clients (its answers must not change under load).
#
#   setsid bash scripts/accuracy_v1.sh > ~/jev/logs/accuracy_v1.out 2>&1 < /dev/null &
#
# STARTLUX_V1: a checkout of github.com/StartLuxLabs/StartLux-Decision-v1 (its eval/ and startlux_decision/); BENCH: the
# folder eval/fetch_benchmarks.sh fills (intern-decision @ 2f81580, jevbench @ 7ce310c7); M: the GGUF repositories and
# the bf16 checkpoints converted from them (scripts/prepare_model.sh).  Results: results/accuracy_v1/ (scores-*.txt,
# agreement-*.txt, *.pilot_score/metrics.json, the predictions).
set -uo pipefail
JEV=${JEV:-$HOME/jev}
source "$JEV/env.sh"
export BENCH=${BENCH:-$JEV/bench-v1}
SRC=${STARTLUX_V1:-$JEV/src/StartLux-Decision}
export STARTLUX_V1=$SRC
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$REPO/results/accuracy_v1
M=${M:-$JEV/models}
mkdir -p "$OUT"
cd "$REPO"
log() { echo "[$(date +%T)] $*"; }

predict() {   # name url shards: the suites (shards clients at once), then the pilot (one at a time)
  local name=$1 url=$2 shards=$3 pids=() rc=0
  rm -rf "$OUT/$name"
  for i in $(seq 0 $((shards - 1))); do
    (cd "$SRC" && python eval/suites.py predict --endpoint "$url" --out "$OUT/$name" --shard "$i" --shards "$shards" \
      > "$OUT/$name.predict.$i.log" 2>&1) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  (cd "$SRC" && python eval/suites.py pilot --endpoint "$url" --out "$OUT/$name.pilot.jsonl" > "$OUT/$name.pilot.log" 2>&1) || rc=1
  return $rc
}

flashjev() {  # name prec model_dir shards [flags]
  local name=$1 prec=$2 model=$3 shards=$4 flags=${5:-}
  for attempt in 1 2; do
    log "flash-jev $name: start ($prec, attempt $attempt)"
    MODEL=$model FLASHJEV_FLAGS="$flags" scripts/serve.sh start "$prec" 8092 || { scripts/serve.sh stop; continue; }
    local t0=$SECONDS
    if predict "$name" http://127.0.0.1:8092 "$shards"; then
      log "flash-jev $name: done in $((SECONDS - t0)) s"; scripts/serve.sh stop; return 0
    fi
    log "flash-jev $name: a client failed (server log tail:)"; tail -5 "$JEV/logs/flash-jev-$prec.log"
    scripts/serve.sh stop
  done
  return 1
}

reference() { # name hf_dir
  local name=$1 hf=$2
  for attempt in 1 2; do
    log "reference $name: start (attempt $attempt)"
    HF=$hf scripts/baselines.sh startlux > "$OUT/$name.server.log" 2>&1 || { scripts/baselines.sh stop > /dev/null; continue; }
    local t0=$SECONDS
    if predict "$name" http://127.0.0.1:8091 1; then
      log "reference $name: done in $((SECONDS - t0)) s"; scripts/baselines.sh stop > /dev/null; return 0
    fi
    log "reference $name: a client failed"; tail -5 "$JEV/logs/startlux_server.log"
    scripts/baselines.sh stop > /dev/null
  done
  return 1
}

score() {     # tag dirs...: accuracy table, pilot scores, agreement with the reference (first dir)
  local tag=$1; shift
  (cd "$SRC" && python eval/suites.py score "$@" --json "$OUT/scores-$tag.json") > "$OUT/scores-$tag.txt" 2>&1
  for d in "$@"; do
    n=$(basename "$d")
    (cd "$BENCH/intern-decision" && python -m src.eval.score_known_distribution \
      --dataset benchmarks/known-distribution-pilot-v1 --predictions "$OUT/$n.pilot.jsonl" --output "$OUT/$n.pilot_score") \
      > "$OUT/$n.pilot_score.log" 2>&1
  done
  python "$REPO/bench/compare_preds.py" "$@" > "$OUT/agreement-$tag.txt" 2>&1
  log "scored $tag:"; cat "$OUT/scores-$tag.txt"; cat "$OUT/agreement-$tag.txt"
}

log "start; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
# ---- StartLux-Decision-4B
flashjev flash-jev-int8-4B int8 "$M/StartLux-Decision-4B-Q8_0-GGUF" 1
flashjev flash-jev-int8-4B-c8 int8 "$M/StartLux-Decision-4B-Q8_0-GGUF" 8
flashjev flash-jev-bf16-4B bf16 "$M/StartLux-Decision-4B-Q8_0-GGUF" 1
flashjev flash-jev-w4a8-4B w4a8 "$M/StartLux-Decision-4B-Q4_K_M-GGUF" 1
reference startlux-ref-4B "$M/StartLux-Decision-4B-bf16-from-Q8_0"
score 4B "$OUT/startlux-ref-4B" "$OUT/flash-jev-int8-4B" "$OUT/flash-jev-int8-4B-c8" "$OUT/flash-jev-bf16-4B" \
  "$OUT/flash-jev-w4a8-4B"
# ---- StartLux-Decision-9B
flashjev flash-jev-int8-9B int8 "$M/StartLux-Decision-9B-Q8_0-GGUF" 1
flashjev flash-jev-bf16-9B bf16 "$M/StartLux-Decision-9B-Q8_0-GGUF" 1
flashjev flash-jev-w4a8-9B w4a8 "$M/StartLux-Decision-9B-Q4_K_M-GGUF" 1
reference startlux-ref-9B "$M/StartLux-Decision-9B-bf16-from-Q8_0"
score 9B "$OUT/startlux-ref-9B" "$OUT/flash-jev-int8-9B" "$OUT/flash-jev-bf16-9B" "$OUT/flash-jev-w4a8-9B"
# ---- StartLux-Decision-27B's Q4_K_M (its bf16 reference does not fit this GPU: compared with the published numbers)
flashjev flash-jev-w4a8-27B w4a8 "$M/StartLux-Decision-27B-Q4_K_M-GGUF" 1 "--prefix-cache 8"
score 27B "$OUT/flash-jev-w4a8-27B"
log "ACC DONE; GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
