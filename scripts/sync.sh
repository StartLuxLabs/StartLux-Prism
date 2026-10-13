#!/bin/bash
# Copy the working tree to the GPU server (code only; models, wheels and results live there).
#   HOST=user@gpu-server scripts/sync.sh
set -euo pipefail
HOST=${HOST:?set HOST=user@gpu-server}
cd "$(dirname "$0")/.."
rsync -a --delete --exclude .git --exclude __pycache__ --exclude 'results/' --exclude '*.so' --exclude build \
  ./ "$HOST:jev/flash-jev/"
