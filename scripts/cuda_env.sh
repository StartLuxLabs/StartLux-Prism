# Source this (in the environment that ran `pip install -r requirements.txt`): the pip CUDA 13 toolkit becomes
# CUDA_HOME, so that the first start can build StartLux-Prism's CUDA kernels with it.
#
#   source scripts/cuda_env.sh
#
# A machine with a CUDA 13.0 toolkit installed can set CUDA_HOME to it instead (export CUDA_HOME=/usr/local/cuda-13.0).
_cu=$(python -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2>/dev/null)
if [ -z "$_cu" ] || [ ! -x "$_cu/bin/nvcc" ]; then
  echo "cuda_env.sh: no nvcc under site-packages/nvidia/cu13 - run pip install -r requirements.txt first" >&2
  return 1 2>/dev/null || exit 1
fi
# the layout torch's extension builder links against: lib64, and unversioned library names
[ -e "$_cu/lib64" ] || ln -s lib "$_cu/lib64"
for _so in cudart cublas cublasLt nvrtc; do
  _v=$(ls "$_cu/lib/lib$_so.so."* 2>/dev/null | head -1)
  if [ -n "$_v" ] && [ ! -e "$_cu/lib/lib$_so.so" ]; then ln -s "$(basename "$_v")" "$_cu/lib/lib$_so.so"; fi
done
export CUDA_HOME="$_cu" CUDA_PATH="$_cu"
export PATH="$_cu/bin:$PATH" LD_LIBRARY_PATH="$_cu/lib:${LD_LIBRARY_PATH:-}"
unset _cu _so _v
nvcc --version | tail -1
