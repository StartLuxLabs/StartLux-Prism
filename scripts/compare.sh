#!/bin/bash
# The engine comparison, one engine at a time on the GPU server: start the engine, JevBench (accuracy + per-item
# latency), the HTTP suites (latency at concurrency 1) and the concurrency sweep (throughput), stop it.  Every engine
# gets the same client, the same unique requests (bench/run_http.py) and the same warmup.
#
#   scripts/compare.sh flash-jev-int8 flash-jev-bf16 flash-jev-w4a8 sglang sglang-native sglang-gguf vllm startlux llamacpp decider decider-real jevk5 \
#       jevk5-real
#
# Results: results/jevbench/<label>.json and results/http/<label>.json (TAG=-x appends -x to the labels).
# N (timed requests per suite, 100), SWEEP (concurrency levels, 1,4,16,64), SWEEP_SECONDS (15) override the defaults.
# BENCH=prefix runs bench/run_prefix.py instead (sessions of follow-up requests about the same evidence; SESSIONS, 40):
# results/prefix/<label>.json; BENCH=long runs bench/run_long.py (8k and 50k tokens of evidence): results/long/.
# Long contexts need LLAMA_CTX / LLAMA_PARALLEL, VLLM_MAXLEN, SGLANG_CTX and DECIDER_ENV=DECIDER_MAX_STATE_TOKENS=...
# BENCH=jevconc: JevBench throughput at CONC clients (8,16,32; PASSES passes over the items per level, 3):
# results/jevbench_conc/; BENCH=longconc: 8k-token requests at CONC clients (CONC_SECONDS per level, 30):
# results/long_conc/ (run it with the long-context settings above).
# FLASHJEV_FLAGS passes extra flags to the flash-jev server (e.g. --prefix-cache 0).
set -uo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
N=${N:-100}; SWEEP=${SWEEP:-1,4,16,64}; SWEEP_SECONDS=${SWEEP_SECONDS:-15}; TAG=${TAG:-}
BENCH=${BENCH:-standard}; SESSIONS=${SESSIONS:-40}; JEVBENCH=${JEVBENCH:-$JEV/src/jevbench-7ce310c7/datasets/public}
MODEL=${MODEL:-$JEV/models/StartLux-Decision-4B-Q8_0-GGUF}     # the GGUF repository (also the tokenizer that sizes
WEIGHTS=$(basename "$MODEL" | sed 's/-GGUF$//')                # the long evidence); scripts/baselines.sh: the rest

measure() {   # label url meta-json [model]
  (source "$JEV/env.sh"
   if [ "$BENCH" = long ]; then       # one question about 8k / 50k tokens of evidence
     python bench/run_long.py --url "$2" --label "$1$TAG" --model-dir "$MODEL" \
       --meta "$3" --out "results/long/$1$TAG.json" 2>&1 | grep -v Warning | tail -4
     exit
   fi
   if [ "$BENCH" = jevconc ]; then    # JevBench throughput at several concurrency levels
     python bench/run_jevbench.py --url "$2" --label "$1$TAG" --model "${4:-StartLux-Decision}" --conc "${CONC:-8,16,32}" \
       --passes "${PASSES:-3}" --out "results/jevbench_conc/$1$TAG.json" 2>&1 | grep -v Warning | tail -3
     exit
   fi
   if [ "$BENCH" = longconc ]; then   # 8k-token requests at several concurrency levels
     python bench/run_long.py --url "$2" --label "$1$TAG" --model-dir "$MODEL" \
       --lengths 8000 --n 0 --conc "${CONC:-8,16,32}" --conc-lengths 8000 --conc-seconds "${CONC_SECONDS:-30}" \
       --meta "$3" --out "results/long_conc/$1$TAG.json" 2>&1 | grep -v Warning | tail -4
     exit
   fi
   if [ "$BENCH" = prefix ]; then     # sessions of requests about the same evidence (prefix caches)
     python bench/run_prefix.py --url "$2" --label "$1$TAG" --model "${4:-StartLux-Decision}" --jevbench "$JEVBENCH" \
       --sessions "$SESSIONS" --meta "$3" --out "results/prefix/$1$TAG.json" 2>&1 | tail -5
     exit
   fi
   python bench/run_jevbench.py --url "$2" --label "$1$TAG" --model "${4:-StartLux-Decision}" \
     --out "results/jevbench/$1$TAG.json" 2>&1 | tail -1
   python bench/run_http.py --url "$2" --label "$1$TAG" --n "$N" --sweep "$SWEEP" --sweep-seconds "$SWEEP_SECONDS" \
     --meta "$3" --out "results/http/$1$TAG.json" 2>&1 | tail -12)
}

stop_all() {
  bash scripts/serve.sh stop
  bash scripts/baselines.sh stop
  sleep 5
}

GPU=${CUDA_VISIBLE_DEVICES:-}; GPU=${GPU%%,*}  # the GPU the engines run on (several GPUs, other jobs: set
SMI="nvidia-smi ${GPU:+-i $GPU}"               # CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER=PCI_BUS_ID); else all GPUs

gpu_idle() {  # every engine runs on a GPU no other process computes on: wait up to 10 min, else stop the comparison
  local t=0                # (GPU_NEIGHBORS_OK=1: run anyway; the result files record the other processes)
  if [ "${GPU_NEIGHBORS_OK:-0}" = 1 ]; then
    echo "GPU neighbors allowed: $($SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';') $($SMI --query-gpu=utilization.gpu --format=csv,noheader)"
    return
  fi
  while [ -n "$($SMI --query-compute-apps=pid --format=csv,noheader)" ]; do
    [ "$t" = 0 ] && echo "waiting for the GPU: $($SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
    t=$((t + 10)); [ "$t" -ge 600 ] && { echo "GPU still in use after 10 min; comparison stopped"; exit 1; }
    sleep 10
  done
  echo "GPU idle: $($SMI --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader)"
}

for engine in "$@"; do
  stop_all
  echo "=== $engine ($(date +%H:%M:%S))"
  gpu_idle
  case "$engine" in
    flash-jev-int8|flash-jev-bf16|flash-jev-w4a8)
      prec=${engine#flash-jev-}
      bash scripts/serve.sh start "$prec" 8092 > /dev/null || continue
      measure "flash-jev-$prec" http://127.0.0.1:8092/v1/systemone \
        "{\"engine\": \"flash-jev\", \"precision\": \"$prec\", \"weights\": \"$WEIGHTS GGUF\"}" ;;
    sglang|sglang-native)
      bash scripts/baselines.sh sglang > /dev/null || continue
      if [ "$engine" = sglang ]; then
        measure sglang-0.5.21 http://127.0.0.1:8097/v1/systemone \
          '{"engine": "SGLang 0.5.21 via framework_systemone.py (/v1/score, StartLux prompts)", "flags": "see scripts/baselines.sh"}'
      else
        measure sglang-0.5.21-native http://127.0.0.1:8095/v1/systemone \
          '{"engine": "SGLang 0.5.21 native /v1/systemone (its own prompt rendering, not the StartLux format)"}' startlux-4b
      fi ;;
    sglang-gguf)
      bash scripts/baselines.sh sglang-gguf > /dev/null || continue
      measure sglang-0.5.21-gguf http://127.0.0.1:8097/v1/systemone \
        "{\"engine\": \"SGLang 0.5.21 GGUF path (its GGUF kernels on the file's blocks; bench/adapters/sglang_gguf.py) via framework_systemone.py\", \"weights\": \"$WEIGHTS GGUF\"}" ;;
    vllm)
      VLLM_FLAGS=${VLLM_FLAGS--cc.backend=eager} bash scripts/baselines.sh vllm > /dev/null || continue
      measure vllm-0.31 http://127.0.0.1:8097/v1/systemone \
        "{\"engine\": \"vLLM 0.31.0 via framework_systemone.py (chat completions, logprob_token_ids)\", \"flags\": \"${VLLM_FLAGS--cc.backend=eager} --gdn-prefill-backend triton\"}" ;;
    startlux)
      bash scripts/baselines.sh startlux > /dev/null || continue
      measure startlux-torch-bf16 http://127.0.0.1:8091/v1/systemone \
        '{"engine": "StartLux-Decision server (torch, fla, causal-conv1d, CUDA graphs)"}' ;;
    decider|decider-real)
      bash scripts/baselines.sh "$engine" > /dev/null || continue
      if [ "$engine" = decider ]; then
        what="$WEIGHTS weights in decider (same-shape speed stand-in; decider renders its own prompts, so the JevBench score is not meaningful)"
      else
        what='its own checkpoint Mapika/decider-4b v2'
      fi
      measure "${engine/decider/decider-1.9.0}${DECIDER_TAG-}" http://127.0.0.1:8093/v1/systemone \
        "{\"engine\": \"Mapika decider 1.9.0 (decider.serve)\", \"model\": \"$what\", \"env\": \"${DECIDER_ENV-}\"}" ;;
    jevk5|jevk5-real)
      bash scripts/baselines.sh "$engine" > /dev/null || continue
      if [ "$engine" = jevk5 ]; then
        what="$WEIGHTS weights in JevK5 (same-shape speed stand-in)"
      else
        what='its own checkpoint (JevK5 v0.3 Q8_0 GGUF, dequantized to bf16 by gguf_to_hf)'
      fi
      measure "${engine/jevk5/jevk5-0.3.3}" http://127.0.0.1:8094/v1/systemone \
        "{\"engine\": \"JevK5 0.3.3 (jevk5.server)\", \"model\": \"$what\"}" ;;
    llamacpp)
      bash scripts/baselines.sh llamacpp > /dev/null || continue
      measure llamacpp-q8_0 http://127.0.0.1:8090/v1/systemone \
        '{"engine": "llama.cpp llama-server b10639 + StartLux gguf_server", "weights": "Q8_0 GGUF"}' ;;
    *)
      echo "unknown engine $engine" >&2 ;;
  esac
done
stop_all
