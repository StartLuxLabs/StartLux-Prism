#!/bin/bash
# Start / stop the StartLux-Prism /v1/systemone server on the GPU server (detached; logs in $JEV/logs).
#
#   scripts/serve.sh start [int8|bf16|w4a8|auto] [port]   # waits until /health answers (STARTLUX_PRISM_FLAGS: server flags)
#   scripts/serve.sh stop
#
# MODEL: the GGUF repository directory (StartLux-Decision-4B-Q8_0-GGUF; GGUF: the file in it), CALIB: the int8
# activation ranges (computed on the first start and kept there; one file per model).
set -uo pipefail
JEV=${JEV:-$HOME/jev}
source "$JEV/env.sh"
MODEL=${MODEL:-$JEV/models/StartLux-Decision-4B-Q8_0-GGUF}
GGUF=${GGUF:-$(ls "$MODEL"/*.gguf | grep -v "/mmproj-" | head -1)}  # not the vision tower (mmproj-*.gguf)
case "$(basename "$GGUF")" in
  StartLux-Decision-4B-Q8_0.gguf) CALIB=${CALIB:-$JEV/calib/startlux-4b-q8_0.pt} ;;
  *) CALIB=${CALIB:-$JEV/calib/$(basename "$GGUF" .gguf).pt} ;;
esac
LOG=$JEV/logs
mkdir -p "$LOG"
cd "$(dirname "$0")/.."
export STARTLUX_PRISM_BENCH=1   # marks the server started here: stop kills nothing else (as in scripts/baselines.sh)

ours() {      # pid: still the process started here (not a reused PID)
  tr '\0' '\n' 2>/dev/null < "/proc/$1/environ" | grep -qx STARTLUX_PRISM_BENCH=1   # (a gone process: no message)
}

stop() {
  local pid
  if [ -s "$LOG/startlux-prism.pid" ]; then
    pid=$(cat "$LOG/startlux-prism.pid")
    ours "$pid" && kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do ours "$pid" || break; sleep 0.5; done
    ours "$pid" && kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$LOG/startlux-prism.pid"
}

case "${1:-}" in
  start)
    stop
    prec=${2:-int8}; port=${3:-8092}
    setsid python -m startlux_prism.server --gguf "$GGUF" --model-dir "$MODEL" \
      --precision "$prec" --calib "$CALIB" --port "$port" ${STARTLUX_PRISM_FLAGS:-} \
      > "$LOG/startlux-prism-$prec.log" 2>&1 < /dev/null &
    echo $! > "$LOG/startlux-prism.pid"
    for _ in $(seq 1 300); do
      curl -sf -m 2 "localhost:$port/health" > /dev/null && { tail -2 "$LOG/startlux-prism-$prec.log"; exit 0; }
      kill -0 "$(cat "$LOG/startlux-prism.pid")" 2>/dev/null || { echo "server exited:"; tail -20 "$LOG/startlux-prism-$prec.log"; exit 1; }
      sleep 2
    done
    echo "server not up after 600 s"; tail -20 "$LOG/startlux-prism-$prec.log"; exit 1
    ;;
  stop)
    stop
    ;;
  *)
    echo "usage: $0 start [int8|bf16] [port] | stop" >&2; exit 2;;
esac
