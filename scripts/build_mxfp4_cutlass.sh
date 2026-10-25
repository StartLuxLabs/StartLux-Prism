#!/usr/bin/env bash
# Build the optional CUTLASS MXFP4 GEMM for Blackwell SM120/SM121.  SM100 is compile-checked
# separately; SM103 is planned (see docs/blackwell.md).
# CUTLASS_HOME must point at CUTLASS 4.8.x (tested: 4.8.1); CUDA 12.8 or newer.
# SM121 (RTX PRO Blackwell) needs CUDA_ARCH=sm_121a.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUTLASS_HOME:?set CUTLASS_HOME to the CUTLASS 4.8.x source tree}"
CUDA_ARCH="${CUDA_ARCH:-sm_120a}"
nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC -arch="$CUDA_ARCH" \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I "$CUTLASS_HOME/include" -I "$CUTLASS_HOME/tools/util/include" \
  startlux_prism/csrc/mxfp4_gemm.cu -o startlux_prism/kernels/libmxfp4_cutlass.so
echo "wrote $(pwd)/startlux_prism/kernels/libmxfp4_cutlass.so ($CUDA_ARCH)"
