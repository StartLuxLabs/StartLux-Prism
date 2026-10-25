# Blackwell MXFP4 GEMM

An optional MXFP4 (e2m1 weights and activations, one e8m0 scale per 32 inputs, fp32
accumulate) projection kernel for Blackwell, not for one memory size. The CUTLASS launcher runs
on SM120/SM121 (GeForce and RTX PRO); SM100 is compile-checked and SM103 is planned. It falls back
to the pure-Triton `tl.dot_scaled` kernel when the shared library is absent or the GPU is not
SM120/SM121.

No model change is required: `startlux_prism/kernels/mxfp4.py` exposes packing, activation
quantization, and a GEMM over row-major `[N, K]` weights. The CUTLASS launcher is built from
`startlux_prism/csrc/mxfp4_gemm.cu` and loaded with `ctypes`; it has no PyTorch extension
build step and no dependency on the engine.

## Layout

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `MX4.q` | uint8 | `[N, K/2]` | two e2m1 codes per byte, low nibble first; `K` even |
| `MX4.s` | uint8 | `[N, K/32]` | e8m0 exponent; value = `2**(s - 127)` |
| `MX4.cs` | uint8 | flat | optional CUTLASS-interleaved copy of the weight scales |

`pack(w)` quantizes and packs bf16/fp32 weights (`mode="max"` or `"mse"`).
`quantize_act(x, q, s)` writes the same layout for `n` rows of activations.
`gemm(a_q, a_s, w, out, n_rows, epi=0)` computes `out = A @ B.T` with bf16 output (`epi=0`)
or adds into an fp32 residual (`epi=1`). The Triton K-loop order is fixed for every `M`, so
the output bits for a row do not depend on its batch position.

The CUTLASS path needs the CUTLASS scale layout. `cutlass_prepare(w)` repacks the weight
scales once (about 3.1 MB for a 24576x4096 gate|up, 1.6 MB for a 4096x12288 down) and leaves
the packed weight bytes untouched. Activation scales are repacked per call.

## Weight quantization

`pack(w, mode)` is the weight scheme: the e2m1 grid (0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6), one e8m0
exponent per 32 inputs, chosen by `max` (the smallest power of two that keeps the block's maximum on the
grid) or `mse` (the better of that exponent and one below it).  The packed layout is `q` uint8 [n, k/2]
and `s` uint8 [n, k/32].

A GGUF converter is in `startlux_prism/tools/quantize_mxfp4.py`:

```bash
python -m startlux_prism.tools.quantize_mxfp4 MODEL.gguf MODEL-mxfp4.pt --mode max
python -m startlux_prism.tools.quantize_mxfp4 MODEL.gguf MODEL-mxfp4-smooth.pt \
  --smooth int8_calib.pt --alpha 0.5
```

It packs the MLP, attention and GDN projections one at a time and writes
`{"<layer>.<group>": (q, s), "<layer>.s_<group>": scale}`; `--layers 0,1` limits a test run.  Load one with
`q, s = blob["0.gu"]; mxfp4.MX4(q.cuda(), s.cuda(), q.shape[0], q.shape[1] * 2)`.  The `--smooth` path is the
SmoothQuant-style scheme used for the 9B numbers: `W * s`, with the matching activation division folded into
the preceding norm or applied at runtime; for `down`, `1/s_down` is folded into the `up` rows.  On the 9B model,
layer 0's MLP `gu`/`down` and GDN `inp`/`out` packed `q`/`s` and `s_*` match the research
`mlp_smooth.pt`/`full_smooth.pt` exactly.  The engine uses this file through `--precision mxfp4`;
`startlux_prism/tools/mxfp4_reference.py` remains a compact standalone example of the folding.

## Build

CUTLASS 4.8.x and CUDA 12.8 or newer:

```bash
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass /opt/cutlass
CUTLASS_HOME=/opt/cutlass ./scripts/build_mxfp4_cutlass.sh
```

The default `sm_120a` covers RTX 50 series. RTX PRO Blackwell (SM121) uses
`CUDA_ARCH=sm_121a ./scripts/build_mxfp4_cutlass.sh`; it has not been tested on hardware.
The script writes `startlux_prism/kernels/libmxfp4_cutlass.so`, which is git-ignored.

Enable it in Python:

```python
import os
os.environ["STARTLUX_PRISM_MXFP4_CUTLASS"] = "1"
os.environ["STARTLUX_PRISM_MXFP4_LIB"] = "/path/to/libmxfp4_cutlass.so"
from startlux_prism.kernels import mxfp4

w = mxfp4.cutlass_prepare(mxfp4.MX4(q.cuda(), s.cuda(), N, K))
mxfp4.gemm(a_q, a_s, w, out, M)
```

`STARTLUX_PRISM_MXFP4_LIB` defaults to the `.so` next to `mxfp4.py`. Without the environment
variable, or on a GPU that is not SM120/SM121, `gemm` uses Triton.

## Validation

`python tests/test_mxfp4.py` checks packing, activation quantization, GEMM against the
dequantized product, batch invariance, and throughput. On the RTX 5060 Ti:

- CUTLASS and Triton agree exactly on the same MX4 inputs (`max abs diff 0`) for epi 0 and
  epi 1, at `M=1/64/128`, including a 24576x4096 gate|up and a 4096x12288 down.
- The Triton path is bit-identical for the same row at `M=32` and `M=64`.
- 32 engine requests through the CUTLASS path made 32/32 the same choice with `max |dp| = 0`
  against Triton; the kernel changes latency, not decisions.

## Performance

RTX 5060 Ti, 16 GB, idle beside the server. The 16 GB card is the validation card; the kernel has
no VRAM-size gate. Per-GEMM speedup of CUTLASS over Triton with the same packed inputs:

| projection (N, K) | M=1 | M=64 | M=1024 | M=4096 |
|---|---|---|---|---|
| gate|up (24576, 4096) | 1.41x | 1.44x | 1.60x | 1.63x |
| down (4096, 12288) | 1.03x | 1.09x | 1.53x | 1.60x |

At the end-to-end level, on StartLux-Decision-v1.1-9B-AutoThink with the smoothed calibration above,
32k context, FP8 KV, `think_seqs=4`, and the same binary/config.  The MXFP4 MLP engine path and the
converter are in this PR; the FP8 KV arena is a separate research change.  These numbers show the kernel
on that configuration, not a promise that every context fits on 16 GB without FP8 KV:

| workload | Triton | CUTLASS | |
|---|---|---|---|
| 32 read-out items, mean latency | 104.8 ms | **89.9 ms** | **1.17x** |
| 1,799-token read-out, p50 | 222.1 ms | **192.6 ms** | **1.15x** |
| read-out, 1 client | 9.1 req/s | **10.6 req/s** | 1.16x |
| read-out, 64 clients | 9.0 req/s | **9.6 req/s** | 1.07x |
| 8 thoughts, 4 clients, budget 512 | 27.6 s | **24.3 s** | 1.14x |
| GPU memory at 32k | 13.6 GB | 14.1 GB | +0.5 GB |

Caveats:

- Prefill and read-out benefit most; thought-heavy requests gain less (1.14x) because decode
  and thought generation dominate. Short read-outs gain about 1.17x.
- The Triton baseline in the per-GEMM table is this package's `pick` heuristic, not a tuned
  per-shape table. At large `M` a tuned Triton table narrows the gap; the engine A/B below
  used one. The down projection at small `M` is close to parity (1.03x-1.09x).
- Numbers are from the 9B model on a 16 GB card. The 27B and multi-GPU paths were not measured.
  On 32 GB and larger Blackwell cards the same kernel frees the same MLP memory for longer context
  or more concurrent requests, but those runs are not measured yet; no code path checks VRAM size.

## Architecture support

| architecture | GPUs | status |
|---|---|---|
| SM120 | RTX 5060 Ti, 5070, 5080, 5090 | **supported**, validated on the 5060 Ti |
| SM121 | RTX PRO Blackwell | **supported**, builds with `CUDA_ARCH=sm_121a`; untested |
| SM100 | B200, GB200 | compile-checked with `ArchTag = cutlass::arch::Sm100` and `-arch=sm_100a`; untested |
| SM103 | B300, GB300 | planned; needs CUTLASS's FP4 Ultra 1Sm/2Sm structure |

The launch geometry is the SM120 cooperative Pingpong schedule, 128x128x128, one CTA per
tile. SM100 uses the block-scaled UMMA builder; the MXFP4 layout and scale-bridge code are
the same. SM103's naive builder does not compile: its FP4 Ultra path needs
`MmaTileShape1Sm = 128x256x768`, `MmaTileShape2Sm = 256x256x768`, dynamic 1/2-SM clusters,
`KernelTmaWarpSpecialized{1,2}SmBlockScaledMxNvf4UltraVs16Sm103`, and A/B as
`cute::tuple<ElementA, ElementSFA>` (CUTLASS example 89). That is a separate instantiation,
not an architecture-tag switch.

## Reference engine integration (model-specific)

The real engine path is `Options(precision="mxfp4")` / `--precision mxfp4`: it builds the W8A8
attention/GDN weights from a Q8_0 file, loads the MLP side file, folds `post_norm / s_gu`, and runs the
MXFP4 MLP in prefill, the read-out head, and the thought decoder.  It is still MLP-only; full-FP4
attention/GDN and the FP8 KV arena are separate research paths.

`startlux_prism/tools/mxfp4_reference.py` is the compact reference for the same scale handling; it is not
imported by the engine and runs one MLP block:

```python
from startlux_prism.tools import mxfp4_reference as R
blob, weight, scale = R.load_side("MODEL-mxfp4-smooth.pt")
post_norm_fp4 = R.fold_norm(post_norm, scale(i, "gu", hidden))
gu, down = weight(i, "gu"), weight(i, "down")
out = R.mlp(x, post_norm_fp4, gu, down)
```

For attention, use `R.fold_norm(attn_norm, scale(i, "qkv", hidden))`; for GDN use `s_inp`; for their output
projections use `R.output_inv(scale(i, "o" or "out", dim))` on the bf16 intermediate.  The engine's
`engine/model.py` and `engine/decode.py` do the same thing with CUDA graphs and the W8A8 attention/GDN path;
this module is the minimal scale-handling reference.

## Engine integration

`--precision mxfp4` is opt-in and runs the MLP in MXFP4 with the rest W8A8:

```bash
python -m startlux_prism.server --gguf MODEL.gguf --model-dir MODEL_DIR --precision mxfp4 \
  --calib int8_calib.pt --mxfp4-side MODEL_DIR/mxfp4/mlp_smooth.pt
```

The side file is optional: without one, the engine packs the GGUF's MLP (weight-only, scale 1).  A Q8_0
file and the int8 calibration file are required.  The engine loads the side file once, folds
`post_norm / s_gu`, keeps the W8A8 attention/GDN path, and uses the MXFP4 MLP in prefill, the read-out
head and the thought decoder.  It falls back to Triton when the CUTLASS `.so` is missing or the GPU is not
SM120/SM121.  FP8 KV and full-FP4 attention/GDN are separate experiments, not part of this path.

The C ABI is intentionally small: `mxfp4_sfb_bytes`, `mxfp4_sfa_bytes`,
`mxfp4_prepare_sfb`, `mxfp4_prepare_sfa`, and `mxfp4_gemm`. The shared library keeps one
scratch workspace per process, sized by CUTLASS at the first call.
