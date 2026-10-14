#!/bin/bash
# Offline environment for StartLux-Prism on the GPU server (no internet, no root).
#
#   JEV=~/jev scripts/setup_server.sh
#
# Expects $JEV/wheels/{linux-cp312,uv,sdist,srv_req.txt}, fetched elsewhere (scripts/fetch_wheels.md) and copied over.
# Creates $JEV/venv (system python 3.12) with torch, triton, nvcc and the CUDA 13 headers/libraries from pip, and
# $JEV/env.sh, which sets CUDA_HOME to the pip CUDA tree so nvcc, torch extensions and cmake find a full toolkit.
set -euo pipefail
JEV=${JEV:-$HOME/jev}
W=$JEV/wheels
mkdir -p "$JEV/bin"

if [ ! -x "$JEV/bin/uv" ]; then
  python3 - "$W"/uv/uv-*.whl "$JEV/bin" <<'EOF'
import sys, zipfile, os
whl, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.endswith("/scripts/uv"):
            open(os.path.join(out, "uv"), "wb").write(z.read(n))
os.chmod(os.path.join(out, "uv"), 0o755)
EOF
fi
UV="$JEV/bin/uv"
"$UV" --version

[ -d "$JEV/venv" ] || "$UV" venv --python /usr/bin/python3 "$JEV/venv"
"$UV" pip install --python "$JEV/venv/bin/python" --offline --no-index --find-links "$W/linux-cp312" -r "$W/srv_req.txt"

SP=$("$JEV/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
CU=$SP/nvidia/cu13
# a toolkit layout for tools that expect one: lib64, unversioned .so names for linking
[ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
for so in cudart cublas cublasLt nvrtc; do
  v=$(ls "$CU/lib/lib$so.so."* 2>/dev/null | head -1 || true)
  [ -n "$v" ] && [ ! -e "$CU/lib/lib$so.so" ] && ln -s "$(basename "$v")" "$CU/lib/lib$so.so"
done

cat > "$JEV/env.sh" <<EOF
# source this: StartLux-Prism environment
source "$JEV/venv/bin/activate"
export CUDA_HOME="$CU"
export CUDA_PATH="$CU"
export PATH="$CU/bin:\$PATH"
export LD_LIBRARY_PATH="$CU/lib:\${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export TRITON_CACHE_DIR="$JEV/.triton"
export TORCH_EXTENSIONS_DIR="$JEV/.torch_extensions"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
source "$JEV/env.sh"
nvcc --version | tail -2
python - <<'EOF'
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
a = torch.randn(256, 256, device="cuda")
print("matmul ok", float((a @ a).sum()) == float((a @ a).sum()))
EOF
