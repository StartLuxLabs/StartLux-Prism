"""flash-jev's CUDA C++ kernels (flash_jev/csrc), compiled on first use with the toolkit in CUDA_HOME and registered
as torch ops (graph-capturable: they launch on the current stream)."""
import functools
import os

import torch

# the canonical path: the build command holds it, and processes that imported flash_jev by different paths (a script's
# sys.path entry, a tensor-parallel follower's PYTHONPATH) otherwise rebuilt the extension in turn, each waiting for
# the other's build lock
CSRC = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc"))
CFGS = {  # cfg id -> (BM, BN, BK, warps) of the instantiations in w8a8_gemm.cu (8..13: pipelined main loop)
    0: (128, 128, 64, 8), 1: (64, 128, 64, 4), 2: (64, 64, 128, 4), 3: (128, 64, 64, 4), 4: (64, 256, 64, 8),
    5: (32, 128, 128, 4), 6: (256, 128, 64, 8), 7: (128, 256, 64, 8),
    8: (128, 128, 64, 8), 9: (128, 64, 64, 4), 10: (256, 128, 64, 8), 11: (128, 256, 64, 8), 12: (64, 64, 128, 4),
    13: (128, 128, 128, 8),
}
SMEM_LIMIT = 101376           # bytes of dynamic shared memory per block on sm_120 (99 KB)
WS_BYTES = 96 << 20          # stream-K partial tiles: 2 slots per CTA of BM x BN int32 (2 x 340 x 256 x 128)
MAX_TILES = 1 << 16          # stream-K tile counters


@functools.lru_cache(None)
def gpu_tag(device=None):
    """The GPU model, lower case without spaces: "rtx5090", "h200" - the suffix of the tuning tables measured on it
    (gemm_table_<tag>.json, w4a8_table_<tag>.json)."""
    name = torch.cuda.get_device_name(device).lower().replace("nvidia", "").replace("geforce", "")
    return "".join(name.split())


def table_path(prefix):
    """kernels/<prefix>_<gpu tag>.json: a tuning table measured on this GPU's model (it may not exist)."""
    return os.path.join(os.path.dirname(__file__), f"{prefix}_{gpu_tag()}.json")


@functools.lru_cache(None)
def cuda_arch():
    """The GPU architecture the kernels are built for: FLASHJEV_CUDA_ARCH, else this GPU's - "120" on the RTX 5090, "90a"
    on Hopper (H100 / H200: the architecture-specific build, which the wgmma kernel of csrc/w8a8_wgmma.cu needs)."""
    arch = os.environ.get("FLASHJEV_CUDA_ARCH")
    if arch:
        return arch
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}" + ("a" if major == 9 else "")


@functools.lru_cache(None)
def load():
    from torch.utils.cpp_extension import load as _load

    arch = cuda_arch()
    # torch adds -gencode flags of its own for TORCH_CUDA_ARCH_LIST (default: this GPU): name the same architecture,
    # so that the library holds one build (on Hopper an sm_90 build next to the sm_90a one would lack the wgmma kernel)
    base = arch.rstrip("a")                     # "90a" -> TORCH_CUDA_ARCH_LIST "9.0a", "120" -> "12.0"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{base[:-1]}.{base[-1]}{arch[len(base):]}")
    sources = ("w8a8_gemm.cu", "w8a8_wgmma.cu", "w4a8_gemm.cu", "l2_prefetch.cu")
    _load(name="flash_jev_ext", sources=[os.path.join(CSRC, f) for f in sources],
          is_python_module=False,
          extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo"]
          + (["-DFLASHJEV_SM90A"] if arch == "90a" else []),
          extra_cflags=["-O3"], verbose=False)
    return torch.ops.flash_jev


# cfg id -> (BM, BN, TMA pipeline) of csrc/w8a8_wgmma.cu
WG_CFGS = {0: (128, 256, False), 1: (128, 128, False), 2: (64, 256, False), 3: (64, 128, False),
           4: (128, 256, True), 5: (128, 128, True), 6: (64, 256, True), 7: (64, 128, True), 8: (64, 64, True),
           9: (128, 64, True)}
WG_SMEM_LIMIT = 232448        # bytes of dynamic shared memory per block on sm_90 (227 KB)


@functools.lru_cache(None)
def has_wgmma(device=None):
    """The Hopper W8A8 kernel (csrc/w8a8_wgmma.cu) runs here: an sm_90 GPU and an sm_90a build."""
    return cuda_arch() == "90a" and torch.cuda.get_device_capability(device)[0] == 9


def wg_fits(cfg, stages):
    bm, bn, tma = WG_CFGS[cfg]
    return 3 <= stages <= 6 and stages * ((bm + bn) * 128 + (16 if tma else 0)) + 1024 <= WG_SMEM_LIMIT


def gemm_wg(a, a_s, w, w_s, out, n_rows, epi=0, cfg=0, stages=4, split=1, slots=None, i32=False):
    """The same product (epi 0 bf16 store, 1 fp32 residual add) on Hopper's warpgroup MMA (csrc/w8a8_wgmma.cu).
    split > 1 (epi 1, TMA configs): deferred split-K - split z stores its int32 partial sums in slots[z] (int32 [S,
    rows, N]) for the residual's next reader to add, as gemm's mode "defer".  i32: int32 sums into slots even with
    one split (w8a8.gemm_i32)."""
    stride = 0
    if split > 1 or i32:
        assert epi == 1 and slots is not None and slots.shape[0] >= split and slots.shape[2] == out.shape[1]
        out, stride = slots[0][:out.shape[0]], slots.stride(0)
    load().w8a8_wgmma(a, a_s.reshape(-1), w, w_s.reshape(-1), out, n_rows, epi, cfg, stages, split, stride)


def fits(cfg, stages):
    """Whether a tile config's pipeline stages fit in shared memory."""
    bm, bn, bk, _ = CFGS[cfg]
    return (bm + bn) * bk * stages <= SMEM_LIMIT


@functools.lru_cache(None)
def sm_count(device=None):
    return torch.cuda.get_device_properties(device or torch.cuda.current_device()).multi_processor_count


@functools.lru_cache(None)
def _workspace(device):
    """Stream-K workspace and tile counters (zero; every call leaves the counters it used at zero again).  Allocated
    once, on the first (eager) call, outside any graph capture."""
    return (torch.zeros(WS_BYTES // 4, dtype=torch.int32, device=device),
            torch.zeros(MAX_TILES, dtype=torch.int32, device=device))


SPLIT_MODES = {"atomic": 0, "workspace": 1, "cluster": 2, "defer": 3}


def gemm(a, a_s, w, w_s, out, n_rows, epi=0, cfg=0, split=1, stages=4, mode="cluster", slots=None, pdl=False,
         i32=False):
    """a int8 [M, K], a_s fp32 [M], w int8 [N, K], w_s fp32 [N] -> out (bf16 store, or fp32 add when epi == 1).
    With split > 1 (epi 1 only) the K splits of a tile combine by `mode`: "defer" (split z stores its int32 partial
    sums in slots[z] - int32 [S, rows, N] - and the next reader of the residual adds them, scales the total and adds
    it, see w8a8.Deferred; no synchronization), "cluster" (int32 partial sums added over distributed shared memory in
    a thread block cluster), "workspace" (int32 partial sums through global memory, the last split to arrive adds
    them) - all three give the bits of an unsplit product - or "atomic" (fp32 atomic adds into out: order-dependent).
    i32 (mode "defer"): int32 sums into slots even with one split (w8a8.gemm_i32).  pdl: launched to overlap the
    previous kernel (programmatic dependent launch, see csrc/w8a8_gemm.cu)."""
    ws, counters = _workspace(a.device) if split > 1 and mode == "workspace" else (None, None)
    stride = 0
    if (split > 1 or i32) and mode == "defer":
        assert slots is not None and slots.shape[0] >= split and slots.shape[2] == out.shape[1]
        out, stride = slots[0][:out.shape[0]], slots.stride(0)
    load().w8a8_gemm(a, a_s.reshape(-1), w, w_s.reshape(-1), out, n_rows, epi, cfg, split, stages, SPLIT_MODES[mode],
                     ws, counters, stride, int(pdl))   # pdl: 0 off, 1-3 (csrc/w8a8_gemm.cu)


PURE_SK = 1 << 20           # sk_waves value that leaves no data-parallel waves


def l2_prefetch(tensors, ctas=32):
    """Prefetch the bytes of up to four contiguous tensors into L2 (evict-last), on the current stream."""
    load().l2_prefetch(list(tensors), ctas)


def gemm_sk(a, a_s, w, w_s, out, n_rows, epi=0, cfg=0, stages=4, ctas_per_sm=1, sk_waves=0, pdl=False):
    """The same product on ctas_per_sm x SMs persistent CTAs: whole waves of tiles data-parallel, the ragged last wave
    (plus sk_waves full waves; PURE_SK: everything) stream-K (see csrc/w8a8_gemm.cu)."""
    ws, counters = _workspace(a.device)
    grid = ctas_per_sm * sm_count(a.device.index)
    load().w8a8_gemm_sk(a, a_s.reshape(-1), w, w_s.reshape(-1), out, n_rows, ws, counters, epi, cfg, stages, grid,
                        sk_waves, int(pdl))
