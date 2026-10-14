#!/bin/bash
# Start a baseline /v1/systemone server on the GPU server (one at a time; each prints its URL).
#
#   scripts/baselines.sh llamacpp      # llama.cpp llama-server + StartLux-Decision's gguf_server (the official GGUF path)
#   scripts/baselines.sh startlux      # StartLux-Decision's own torch server (fla + causal-conv1d + CUDA graphs) on the
#                                      # bf16 checkpoint converted from the same GGUF (startlux_prism/tools/gguf_to_hf.py)
#   scripts/baselines.sh vllm          # vLLM (OpenAI server, bf16 CausalLM view of the same weights) behind
#                                      # bench/adapters/framework_systemone.py (StartLux prompts + letter readout)
#   scripts/baselines.sh sglang        # SGLang (/v1/score) behind the same adapter
#   scripts/baselines.sh sglang-gguf   # SGLang on the GGUF file itself (its GGUF kernels; bench/adapters/sglang_gguf.py)
#   scripts/baselines.sh decider       # Mapika decider 1.9.0 (decider.serve, its own /v1/systemone) on StartLux-Decision-4B's
#                                      # weights: a same-shape speed stand-in (decider renders its own prompts, so answers
#                                      # are not StartLux's); decider-real: on its own checkpoint Mapika/decider-4b v2
#   scripts/baselines.sh jevk5         # JevK5 0.3.3 (jevk5.server) on StartLux-Decision-4B's weights (speed stand-in);
#                                      # jevk5-real: on its own checkpoint (its Q8_0 GGUF converted by gguf_to_hf)
#   scripts/baselines.sh stop
#
# Environment: JEV (default ~/jev), MODEL (GGUF repo dir), HF (converted checkpoint dir), LLAMA (llama.cpp build dir),
# CAUSAL (the checkpoint as Qwen3_5ForCausalLM for the frameworks, keys model.language_model.*), CAUSAL_FLAT (the same
# with keys model.* -- SGLang's loader does not map model.language_model.*, and silently leaves those weights
# uninitialized), VLLM_FLAGS / SGLANG_FLAGS (extra server flags), LLAMA_CTX / LLAMA_PARALLEL / VLLM_MAXLEN /
# SGLANG_CTX (context: 16384 tokens, llama.cpp split over 4 slots), DECIDER_ENV (extra decider settings, e.g.
# "DECIDER_COMPILE=1 DECIDER_FP8=1"), DECIDER_SHAPE / JEVK5_SHAPE (the stand-in folders: decider's and JevK5's
# configs over MODEL's weights; scripts/prepare_model.sh builds every folder for another GGUF file, e.g. the 9B).
# The adapter answers /v1/systemone on port 8097 for both frameworks.
set -uo pipefail
JEV=${JEV:-$HOME/jev}
source "$JEV/env.sh"
MODEL=${MODEL:-$JEV/models/StartLux-Decision-4B-Q8_0-GGUF}
HF=${HF:-$JEV/models/StartLux-Decision-4B-bf16-from-Q8_0}
GGUF=${GGUF:-$(ls "$MODEL"/*.gguf | grep -v "/mmproj-" | head -1)}  # not the vision tower (mmproj-*.gguf)
LLAMA=${LLAMA:-/opt/llama.cpp-b10639}
CAUSAL=${CAUSAL:-$JEV/models/startlux-causallm}
CAUSAL_FLAT=${CAUSAL_FLAT:-$JEV/models/startlux-causallm-flat}
SRC=$JEV/src/StartLux-Decision
HERE=$(cd "$(dirname "$0")/.." && pwd)
LOG=$JEV/logs
mkdir -p "$LOG"
export STARTLUX_PRISM_BENCH=1   # marks every process started here: `stop` kills nothing else (other users' vLLM / SGLang servers
                      # on a shared machine have the same command lines, possibly under the same account)

ours() {      # pid: a process these scripts started
  tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep -qx STARTLUX_PRISM_BENCH=1   # (a gone process: no message)
}

adapter() {   # backend upstream_port: the /v1/systemone adapter in front of a framework server
  (cd "$HERE" && exec setsid python bench/adapters/framework_systemone.py --backend "$1" \
    --upstream "http://127.0.0.1:$2" --model-dir "$CAUSAL" --port 8097 > "$LOG/adapter-$1.log" 2>&1 < /dev/null) &
  echo $! > "$LOG/adapter-$1.pid"
  wait_up 8097 "$LOG/adapter-$1.pid" "$LOG/adapter-$1.log" 60
  echo "$1 via adapter: http://127.0.0.1:8097/v1/systemone"
}

wait_up() {   # port pidfile logfile timeout_s: until /health answers; fail if the process dies or time runs out
  local t=0
  until curl -sf "localhost:$1/health" > /dev/null; do
    kill -0 "$(cat "$2")" 2>/dev/null || { echo "server on port $1 exited:"; tail -20 "$3"; exit 1; }
    t=$((t + 2)); [ "$t" -ge "$4" ] && { echo "server on port $1 not up after $4 s"; tail -20 "$3"; exit 1; }
    sleep 2
  done
}

case "${1:-}" in
  llamacpp)
    # the system build links CUDA 13 runtime libraries that are not installed system-wide: use the pip ones
    LD_LIBRARY_PATH="$CUDA_HOME/lib:$LLAMA:${LD_LIBRARY_PATH:-}" setsid "$LLAMA/llama-server" \
      -m "$GGUF" -ngl 99 -c "${LLAMA_CTX:-16384}" --parallel "${LLAMA_PARALLEL:-4}" --port 8081 \
      > "$LOG/llama-server.log" 2>&1 < /dev/null &
    echo $! > "$LOG/llama-server.pid"
    wait_up 8081 "$LOG/llama-server.pid" "$LOG/llama-server.log" 300
    (cd "$SRC" && exec setsid python -m startlux_decision.gguf_server --model-dir "$MODEL" \
      --llama http://127.0.0.1:8081 --port 8090 > "$LOG/gguf_server.log" 2>&1 < /dev/null) &
    echo $! > "$LOG/gguf_server.pid"
    wait_up 8090 "$LOG/gguf_server.pid" "$LOG/gguf_server.log" 300
    echo "llama.cpp + gguf_server: http://127.0.0.1:8090/v1/systemone"
    ;;
  startlux)
    # transformers >= 5.1x picks the fla / causal-conv1d kernels by itself (hub_kernels fallback), but StartLux's check
    # still looks for the removed `is_fast_path_available` flag and reports them inactive; the override skips only
    # that check (the log's graph self-test and the timings confirm the fast path).
    (cd "$SRC" && STARTLUX_ALLOW_SLOW=1 exec setsid python -m startlux_decision.server --model "$HF" --port 8091 \
      --no-images > "$LOG/startlux_server.log" 2>&1 < /dev/null) &
    echo $! > "$LOG/startlux_server.pid"
    wait_up 8091 "$LOG/startlux_server.pid" "$LOG/startlux_server.log" 900
    curl -s localhost:8091/health; echo
    echo "StartLux torch server: http://127.0.0.1:8091/v1/systemone"
    ;;
  vllm)
    # torch.compile of this hybrid model hung (>25 min in one inductor thread) on vLLM 0.31, so the default here is
    # eager; VLLM_FLAGS="" tries the compiled + CUDA-graph default.  The FlashInfer GDN prefill crashed in graph capture
    # on sm_120, hence the Triton (FLA) GDN prefill.  Memory: vLLM's own default fraction (0.9) on a GPU of its own;
    # VLLM_MEM overrides it (0.55 while the GPU was shared with another tenant's 10 GB process).
    source "$JEV/env-vllm.sh"
    setsid vllm serve "$CAUSAL" --port 8096 --gpu-memory-utilization "${VLLM_MEM:-0.9}" --max-model-len "${VLLM_MAXLEN:-16384}" \
      --served-model-name startlux-4b --gdn-prefill-backend triton ${VLLM_FLAGS---enforce-eager} \
      > "$LOG/vllm.log" 2>&1 < /dev/null &
    echo $! > "$LOG/vllm.pid"
    wait_up 8096 "$LOG/vllm.pid" "$LOG/vllm.log" 3600
    adapter vllm 8096
    ;;
  sglang)
    # Memory: SGLang's own default static fraction on a GPU of its own (SGLANG_MEM sets --mem-fraction-static; it is a
    # fraction of the memory free at startup).  Its default mamba state pool capped it at 6 running requests (5 state
    # slots per request with the radix cache); 160 slots (~7.7 GB) allow 32.
    # The scheduler segfaulted (in pure-Python frames, after tree-cache work) twice with the default Rust radix tree
    # core; SGLANG_TREE_CORE=python selects SGLang's own Python tree core.
    source "$JEV/env-sglang.sh"
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=${SGLANG_TREE_CORE:-python} setsid python -m sglang.launch_server \
      --model-path "$CAUSAL_FLAT" --port 8095 ${SGLANG_MEM:+--mem-fraction-static $SGLANG_MEM} \
      --max-mamba-cache-size "${SGLANG_MAMBA:-160}" --context-length "${SGLANG_CTX:-16384}" --served-model-name startlux-4b ${SGLANG_FLAGS-} \
      > "$LOG/sglang.log" 2>&1 < /dev/null &
    echo $! > "$LOG/sglang.pid"
    wait_up 8095 "$LOG/sglang.pid" "$LOG/sglang.log" 3600
    adapter sglang 8095
    ;;
  sglang-gguf)
    # SGLang's own GGUF path on the GGUF file itself (for Q4_K_M: its MMQ / MMVQ kernels on the k-quant blocks), through
    # bench/adapters/sglang_gguf.py, which supplies what SGLang 0.5.21 lacks for qwen35 files (the tensor-name map,
    # llama.cpp's conversion undone while loading).  The file is linked into a folder beside CAUSAL_FLAT's text-only
    # config and tokenizer (the repository's own config.json is the multimodal one).  Server flags as for `sglang`.
    # Its startup died at random on this host (SIGILL / SIGSEGV inside the interpreter, see docs/benchmarks.md, "Crashes
    # on the RTX 5090 server") in about half the launches, so it gets five tries.
    source "$JEV/env-sglang.sh"
    side=$JEV/models/$(basename "$GGUF" .gguf)-sglang
    mkdir -p "$side"
    for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja; do
      cp "$CAUSAL_FLAT/$f" "$side/"
    done
    ln -sf "$GGUF" "$side/"
    for try in 1 2 3 4 5; do
      (cd "$HERE" && SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=${SGLANG_TREE_CORE:-python} exec setsid python \
        bench/adapters/sglang_gguf.py --model-path "$side/$(basename "$GGUF")" --tokenizer-path "$side" --port 8095 \
        ${SGLANG_MEM:+--mem-fraction-static $SGLANG_MEM} --max-mamba-cache-size "${SGLANG_MAMBA:-160}" \
        --context-length "${SGLANG_CTX:-16384}" --served-model-name startlux-4b ${SGLANG_FLAGS-} \
        > "$LOG/sglang-gguf.log" 2>&1 < /dev/null) &
      echo $! > "$LOG/sglang-gguf.pid"
      (wait_up 8095 "$LOG/sglang-gguf.pid" "$LOG/sglang-gguf.log" 3600) && break
      [ "$try" = 5 ] && exit 1
      echo "SGLang (GGUF) died during startup (try $try); again"
    done
    adapter sglang 8095
    ;;
  decider|decider-real)
    model=${DECIDER_SHAPE:-$JEV/models/decider-shape-startlux}
    [ "$1" = decider-real ] && model=$JEV/models/decider-4b
    source "$JEV/env-ref.sh"
    (cd "$JEV/src/decider-v1.9.0" && exec env DECIDER_MODEL="$model" ${DECIDER_ENV-} \
      PYTHONPATH="$JEV/src/decider-v1.9.0" setsid uvicorn decider.serve:app --host 127.0.0.1 --port 8093 \
      > "$LOG/decider.log" 2>&1 < /dev/null) &
    echo $! > "$LOG/decider.pid"
    wait_up 8093 "$LOG/decider.pid" "$LOG/decider.log" 2400
    echo "decider ($model): http://127.0.0.1:8093/v1/systemone"
    ;;
  jevk5|jevk5-real)
    model=${JEVK5_SHAPE:-$JEV/models/jevk5-shape-startlux}
    [ "$1" = jevk5-real ] && model=$JEV/models/jevk5-4b-bf16-from-Q8_0
    source "$JEV/env-ref.sh"
    (cd "$JEV/src/jevk5-v0.3.3" && PYTHONPATH="$JEV/src/jevk5-v0.3.3" exec setsid python -m jevk5.server \
      --model "$model" --port 8094 > "$LOG/jevk5.log" 2>&1 < /dev/null) &
    echo $! > "$LOG/jevk5.pid"
    wait_up 8094 "$LOG/jevk5.pid" "$LOG/jevk5.log" 2400
    echo "JevK5 ($model): http://127.0.0.1:8094/v1/systemone"
    ;;
  stop)
    # each server's session (setsid: its PID) and children, and the servers themselves whatever started them ([x] keeps
    # pgrep from matching this script's own command line); of these, only the processes started here (ours)
    pats=("[s]tartlux_decision.server" "[s]tartlux_decision.gguf_server" "[l]lama-server -m" "[f]lash_jev.server"
          "[f]ramework_systemone.py" "[v]llm serve /" "[V]LLM::EngineCore" "[s]glang.launch_server" "[s]glang_gguf.py" "[s]glang::"
          "[u]vicorn decider.serve" "[j]evk5.server")
    targets() {
      for p in "$LOG"/*.pid; do [ -s "$p" ] && { pgrep -s "$(cat "$p")"; pgrep -P "$(cat "$p")"; cat "$p"; }; done
      for pat in "${pats[@]}"; do pgrep -u "$USER" -f "$pat"; done
    }
    for q in $(targets | sort -u); do ours "$q" && kill "$q" 2>/dev/null; done
    sleep 5   # then whatever ignored SIGTERM (llama-server once kept its GPU memory)
    for q in $(targets | sort -u); do ours "$q" && kill -9 "$q" 2>/dev/null; done
    rm -f "$LOG"/*.pid
    true
    ;;
  *)
    echo "usage: $0 llamacpp|startlux|vllm|sglang|sglang-gguf|decider|decider-real|jevk5|jevk5-real|stop" >&2; exit 2;;
esac
