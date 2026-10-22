#!/bin/bash
# The longest single request each engine answers in ONE forward pass over the whole prompt (no chunked prefill), on
# this GPU: steps the evidence length up (bench/run_long.py: one warm-up and one timed request per length) until a
# request fails or the engine does not start.  StartLux-Prism never chunks (default, and without its prefix cache); SGLang
# with --chunked-prefill-size -1 (its default memory split, and one that leaves activations more room); vLLM with
# --no-enable-chunked-prefill and --max-num-batched-tokens = --max-model-len (it profiles one pass of that many tokens
# at start-up and refuses to start when the KV cache cannot hold one such request).  Last, SGLang and vLLM at their
# defaults (chunked prefill) with a 256k-token request, for the context they reach by chunking.
#
#   setsid bash scripts/max_context.sh > ~/jev/logs/maxctx.out 2>&1 < /dev/null &
#
# Results: results/maxctx/<config>-<length>.json; the log shows each engine's memory split.  LENGTHS overrides the steps,
# MODEL the GGUF repository (4B Q8_0), STARTLUX_PRISM_PREC StartLux-Prism's precision (int8), MAXCTX_ENGINES the engines ("prism sglang
# vllm"; a Q4_K_M file: "prism" with STARTLUX_PRISM_PREC=w4a8), CONFIG_TAG a suffix for every config name (another model or GPU).
set -uo pipefail
JEV=${JEV:-$HOME/jev}
cd "$(dirname "$0")/.."
LENGTHS=${LENGTHS:-16000 32000 48000 64000 96000 128000 192000 256000}
MODEL=${MODEL:-$JEV/models/StartLux-Decision-4B-Q8_0-GGUF}
PREC=${STARTLUX_PRISM_PREC:-int8}
ENGINES=" ${MAXCTX_ENGINES:-prism sglang vllm} "
LOG=$JEV/logs
mkdir -p results/maxctx

probe() {   # config url length: one request of about `length` tokens (StartLux's format); status 0 when answered
  (source "$JEV/env.sh"
   python bench/run_long.py --url "$2" --label "$1" --model-dir "$MODEL" --lengths "$3" --n 1 --warmup 1 \
     --conc-lengths 0 --no-gpu --meta "{\"config\": \"$1\"}" --out "results/maxctx/$1-$3.json" 2>&1 \
     | grep -v Warning | tail -1 | cut -c1-260
   python -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if 'latency' in d['lengths'][sys.argv[2]] else 1)" \
     "results/maxctx/$1-$3.json" "$3")
}

steps() {   # config url: every length until the first failure
  for n in $LENGTHS; do
    probe "$1" "$2" "$n" || { echo "--- $1: first failure at $n"; return; }
  done
  echo "--- $1: every length answered"
}

stop_all() { bash scripts/serve.sh stop; bash scripts/baselines.sh stop; sleep 5; }

memory() {  # what each engine says about its memory split at start-up
  grep -h -i "tokens per batch\|available_gpu_mem\|mem_fraction_static=\|KV cache memory\|GPU KV cache size\|Maximum concurrency\|out of memory\|OutOfMemory\|Segfault" "$@" 2>/dev/null | cut -c1-300 | tail -6
}

prism() {    # config [server flags]
  stop_all
  echo "=== $1 ($(date +%H:%M:%S))"
  STARTLUX_PRISM_FLAGS="${2:-}" bash scripts/serve.sh start "$PREC" > /dev/null || { echo "--- $1: did not start"; return; }
  memory "$LOG/startlux-prism-$PREC.log"
  steps "$1" http://127.0.0.1:8092/v1/systemone
}

sglang() {  # config, with SGLANG_* set by the caller
  stop_all
  echo "=== $1 ($(date +%H:%M:%S)) mem ${SGLANG_MEM:-default} mamba ${SGLANG_MAMBA:-160} flags ${SGLANG_FLAGS:-}"
  bash scripts/baselines.sh sglang > /dev/null || { echo "--- $1: did not start"; memory "$LOG/sglang.log"; return; }
  memory "$LOG/sglang.log"
  steps "$1" http://127.0.0.1:8097/v1/systemone
  memory "$LOG/sglang.log"
}

vllm_at() { # config max_model_len length [flags]: start vLLM for that context, one length
  stop_all
  echo "=== $1 max-model-len $2 ($(date +%H:%M:%S)) flags ${4:-}"
  VLLM_MAXLEN=$2 VLLM_FLAGS="-cc.backend=eager ${4:-}" bash scripts/baselines.sh vllm > /dev/null \
    || { echo "--- $1: did not start with max-model-len $2"; memory "$LOG/vllm.log"; return 1; }
  memory "$LOG/vllm.log"
  probe "$1" http://127.0.0.1:8097/v1/systemone "$3" || { echo "--- $1: failed at $3"; return 1; }
}

if [[ $ENGINES == *" prism "* ]]; then
  prism "startlux-prism-$PREC${CONFIG_TAG:-}"
  prism "startlux-prism-$PREC-nocache${CONFIG_TAG:-}" "--prefix-cache 0"
fi
if [[ $ENGINES == *" sglang "* ]]; then
  SGLANG_CTX=262144 SGLANG_FLAGS="--chunked-prefill-size -1" sglang "sglang-nochunk${CONFIG_TAG:-}"
  SGLANG_CTX=262144 SGLANG_MEM=0.6 SGLANG_MAMBA=16 SGLANG_FLAGS="--chunked-prefill-size -1" sglang \
    "sglang-nochunk-mem0.6${CONFIG_TAG:-}"
fi
if [[ $ENGINES == *" vllm "* ]]; then
  for L in 65536:64000 131072:128000 196608:192000 262144:256000; do
    vllm_at "vllm-nochunk${CONFIG_TAG:-}" "${L%%:*}" "${L##*:}" \
      "--no-enable-chunked-prefill --max-num-batched-tokens ${L%%:*}" || break
  done
fi
[[ $ENGINES == *" sglang "* ]] && SGLANG_CTX=262144 LENGTHS=256000 sglang "sglang-chunked${CONFIG_TAG:-}"
[[ $ENGINES == *" vllm "* ]] && vllm_at "vllm-chunked${CONFIG_TAG:-}" 262144 256000
stop_all
echo "max_context done $(date +%H:%M:%S)"
