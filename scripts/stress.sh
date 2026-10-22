#!/bin/bash
# The stress comparison on the GPU server: every engine alone on the GPU, one after another (scripts/compare.sh: the
# same client, unique requests, a GPU no other process computes on), then reruns of any entry whose server crashed
# (Python processes on this machine segfault at random, docs/benchmarks.md).  Steps:
#   0  the standard comparison: JevBench one item at a time (score, latency), the       -> results/jevbench/, http/
#      request suites one at a time, the 1-64 client sweep
#   1  JevBench's 231 public items from 8, 16 and 32 clients, 3 passes per level      -> results/jevbench_conc/
#      (the standard comparison's engine settings)
#   2  one request at a time with 8k and 50k tokens of evidence, and 4 clients at 8k  -> results/long/
#   3  8k-token requests from 8, 16 and 32 clients, 30 s per level                    -> results/long_conc/
#   4  the longest request each engine answers without chunked prefill               -> results/maxctx/
#      (scripts/max_context.sh)
#   5  tests/test_stress.py on StartLux-Prism (STRESS_PREC, int8): the checks behind the   -> results/stress/
#      numbers
# Steps 2 and 3 run with llama.cpp -c 262144 --parallel 4 (65,536 tokens per slot), vLLM --max-model-len 65536, SGLang
# --context-length 65536 and decider DECIDER_MAX_STATE_TOKENS=65536 (every engine counts all 50k tokens: the result
# files record the input tokens each one reports); StartLux-Prism sizes its largest batch from free GPU memory.
#
#   setsid bash scripts/stress.sh > ~/jev/logs/stress.out 2>&1 < /dev/null &
#   python bench/report.py --stress                # the tables (after copying results/ back)
#
# STEPS (default 0,1,2,3,5) picks steps; ENGINES (steps 0, 1) and LONG_ENGINES (2, 3) override the engine lists; the
# decider compile + FP8 entries run unless NO_DECIDER_FP8=1.  Labels: <engine>-cmp2 for steps 0 and 1, <engine>-long
# for 2 and 3, as in docs/benchmarks.md (TAG1 / TAG2 override).  STARTLUX_PRISM_LONG_FLAGS: StartLux-Prism server flags for steps 2
# and 3 (the 9B: --prefix-cache 0, whose memory then goes to a larger batch).
#
# DECIDER_EXTRA: settings added to every decider run (the 9B: DECIDER_GRAPH_TOKEN_BUDGET=16384).
#
# Another model: scripts/prepare_model.sh prints its environment (MODEL, HF, CAUSAL, ...); then for the 9B
#   MODEL=... HF=... (as printed) SGLANG_MAMBA=96 DECIDER_EXTRA=DECIDER_GRAPH_TOKEN_BUDGET=16384 \
#   ENGINES="startlux-prism-int8 startlux-prism-bf16 startlux sglang sglang-native vllm llamacpp decider jevk5" TAG1=-9b TAG2=-9b-long \
#   STARTLUX_PRISM_LONG_FLAGS="--prefix-cache 0" bash scripts/stress.sh
# (decider-real and jevk5-real are their own 4B checkpoints.  Next to 17.9 GB of bf16 weights: SGLang's 160 recurrent-
# state slots do not fit, 96 (19 running requests) do; decider's default CUDA-graph budget of 32,768 tokens runs out of
# memory in its warm-up, 16,384 fits.  StartLux-Prism's largest batch is 49,152 tokens with its prefix cache, 65,536
# without: the long steps run without it.)
set -uo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
STEPS=${STEPS:-0,1,2,3,5}
ENGINES=${ENGINES:-startlux-prism-int8 startlux-prism-bf16 startlux sglang sglang-native vllm llamacpp decider decider-real jevk5 jevk5-real}
LONG_ENGINES=${LONG_ENGINES:-startlux-prism-int8 startlux sglang vllm llamacpp jevk5 decider}
TAG1=${TAG1:--cmp2}; TAG2=${TAG2:--long}
LONG_ENV="LLAMA_CTX=262144 LLAMA_PARALLEL=4 VLLM_MAXLEN=65536 SGLANG_CTX=65536"
FP8="DECIDER_COMPILE=1 DECIDER_FP8=1"
DX=${DECIDER_EXTRA:-}
export PYTHONFAULTHANDLER=1
mkdir -p results/jevbench_conc results/long results/long_conc results/stress

label() {   # engine [decider tag] -> the label scripts/compare.sh gives it (without TAG)
  case "$1" in
    startlux-prism-int8|startlux-prism-bf16|startlux-prism-w4a8) echo "startlux-prism-${1#startlux-prism-}" ;;
    startlux) echo startlux-torch-bf16 ;;
    sglang) echo sglang-0.5.21 ;;
    sglang-gguf) echo sglang-0.5.21-gguf ;;
    sglang-native) echo sglang-0.5.21-native ;;
    vllm) echo vllm-0.31 ;;
    llamacpp) echo llamacpp-q8_0 ;;
    decider|decider-real) echo "${1/decider/decider-1.9.0}${2:-}" ;;
    jevk5|jevk5-real) echo "${1/jevk5/jevk5-0.3.3}" ;;
  esac
}

clean() {   # result file: present, and no request failed (a crashed server shows as failed requests)
  "$JEV/venv/bin/python" - "$1" <<"PY"
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    sys.exit(1)
d = json.load(open(p))
if "levels" in d:                                          # JevBench throughput
    lv = list(d["levels"].values())
else:                                                      # long evidence: latency runs and throughput levels
    lv = []
    for r in d.get("lengths", {}).values():
        if any(k in r.get("error", "") for k in ("Connection", "Remote", "reset", "refused", "BrokenPipe")):
            sys.exit(1)
        lv += list(r.get("concurrency_levels", {}).values()) + ([r["concurrency"]] if "concurrency" in r else [])
sys.exit(0 if all(v.get("errors", 1) == 0 for v in lv) else 1)
PY
}

fresh() {   # since file...: every file written at or after `since` (a result left by an earlier run does not count)
  local since=$1; shift
  for f in "$@"; do [ "$(stat -c %Y "$f" 2>/dev/null || echo 0)" -ge "$since" ] || return 1; done
}

run() {     # bench file engine [VAR=value ...]: run one entry, and again (twice at most) while it has no clean result
  local bench=$1 file=$2 engine=$3 t0; shift 3
  for attempt in 1 2 3; do
    [ "$attempt" -gt 1 ] && echo "--- $file: no clean result; attempt $attempt"
    t0=$(date +%s)
    env BENCH="$bench" "$@" bash scripts/compare.sh "$engine"
    fresh "$t0" "$file" && clean "$file" && return
  done
  echo "--- $file: no clean result after 3 attempts"
}

echo "stress comparison: steps $STEPS, $(basename "${MODEL:-StartLux-Decision-4B}") ($(date +%H:%M:%S))"
if [[ ",$STEPS," == *,0,* ]]; then
  for e in $ENGINES; do
    for attempt in 1 2 3; do
      t0=$(date +%s)
      env TAG="$TAG1" "DECIDER_ENV=$DX" bash scripts/compare.sh "$e"
      fresh "$t0" "results/http/$(label "$e")$TAG1.json" "results/jevbench/$(label "$e")$TAG1.json" && break
      echo "--- $e: no complete result (attempt $attempt)"
    done
  done
  if [ "${NO_DECIDER_FP8:-0}" != 1 ]; then
    for e in $ENGINES; do
      [ "$e" = decider ] || [ "$e" = decider-real ] || continue
      for attempt in 1 2 3; do
        t0=$(date +%s)
        env TAG="$TAG1" "DECIDER_ENV=$FP8 $DX" DECIDER_TAG=-compile-fp8 bash scripts/compare.sh "$e"
        fresh "$t0" "results/http/$(label "$e" -compile-fp8)$TAG1.json" && break
      done
    done
  fi
  echo "step 0 done ($(date +%H:%M:%S))"
fi
if [[ ",$STEPS," == *,1,* ]]; then
  for e in $ENGINES; do
    run jevconc "results/jevbench_conc/$(label "$e")$TAG1.json" "$e" TAG="$TAG1" "DECIDER_ENV=$DX"
  done
  if [ "${NO_DECIDER_FP8:-0}" != 1 ]; then
    for e in $ENGINES; do
      [ "$e" = decider ] || [ "$e" = decider-real ] || continue
      run jevconc "results/jevbench_conc/$(label "$e" -compile-fp8)$TAG1.json" "$e" TAG="$TAG1" \
        "DECIDER_ENV=$FP8 $DX" DECIDER_TAG=-compile-fp8
    done
  fi
  echo "step 1 done ($(date +%H:%M:%S))"
fi
for step in 2 3; do
  [[ ",$STEPS," == *,$step,* ]] || continue
  bench=long; dir=results/long
  [ "$step" = 3 ] && { bench=longconc; dir=results/long_conc; }
  for e in $LONG_ENGINES; do
    if [ "$e" = decider ]; then
      run "$bench" "$dir/$(label decider)$TAG2.json" decider TAG="$TAG2" $LONG_ENV \
        "DECIDER_ENV=DECIDER_MAX_STATE_TOKENS=65536 $DX"
      [ "${NO_DECIDER_FP8:-0}" = 1 ] || run "$bench" "$dir/$(label decider -compile-fp8)$TAG2.json" decider \
        TAG="$TAG2" $LONG_ENV "DECIDER_ENV=DECIDER_MAX_STATE_TOKENS=65536 $FP8 $DX" DECIDER_TAG=-compile-fp8
    elif [ "${e#startlux-prism-}" != "$e" ]; then
      run "$bench" "$dir/$(label "$e")$TAG2.json" "$e" TAG="$TAG2" $LONG_ENV "STARTLUX_PRISM_FLAGS=${STARTLUX_PRISM_LONG_FLAGS:-}"
    else
      run "$bench" "$dir/$(label "$e")$TAG2.json" "$e" TAG="$TAG2" $LONG_ENV
    fi
  done
  echo "step $step done ($(date +%H:%M:%S))"
done
if [[ ",$STEPS," == *,4,* ]]; then
  bash scripts/max_context.sh
fi
if [[ ",$STEPS," == *,5,* ]]; then
  bash scripts/serve.sh stop; bash scripts/baselines.sh stop; sleep 5
  prec=${STRESS_PREC:-int8}
  (source "$JEV/env.sh"
   python tests/test_stress.py --model-dir "${MODEL:-$JEV/models/StartLux-Decision-4B-Q8_0-GGUF}" \
     --jevbench "$JEV/src/jevbench-7ce310c7" --start "$prec" --out "results/stress/startlux-prism-$prec$TAG1.json" 2>&1 \
     | grep -v Warning)
fi
echo "stress comparison done ($(date +%H:%M:%S))"
