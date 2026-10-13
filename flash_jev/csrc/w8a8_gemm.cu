// W8A8 GEMM for sm_80+ (built for sm_120): int8 x int8 -> int32 on mma.sync.m16n8k32 tensor cores, fused scale epilogue.
//
//   C[m, n] = (sum_k A[m, k] W[n, k]) * sa[m] * sw[n]
//   A [M, K] int8 row-major (activations, one scale per row), W [N, K] int8 row-major (weights, one scale per output
//   channel): both K-contiguous ("TN").  EPI 0 writes bf16 out[m * ldo + n]; EPI 1 adds into fp32 out[m * ldo + n]
//   (the residual stream).
//
// Batch invariance: an output element is the same bits whichever kernel, tile shape or split computes it.  The int32
// sum over K is exact in any order and any split; the epilogue then rounds exactly as the Triton kernel does
// (kernels/w8a8.py): v = (float(acc) * sa) * sw, each product rounded, then out + v rounded (explicit _rn
// intrinsics: no contraction into an FMA, whose single rounding would differ).  So a request's numbers do not depend
// on which other requests share its batch, or on whether its evidence came from the prefix cache.
//
// Classic multistage main loop: cp.async 16-byte copies global -> shared (STAGES deep), XOR-swizzled 16-byte chunks so
// ldmatrix reads of 8 rows hit distinct banks, ldmatrix.x4 fragments, mma.sync int8 with int32 accumulators held in
// registers for the whole K loop, scales applied once at the end.  Rows at or beyond *n_rows (the live rows of a
// padded bucket) are skipped by whole tiles.
//
// Two schedules of that main loop:
//   w8a8_kernel     one CTA per output tile (and K slice with split-K): the grid is tiles x split.
//   w8a8_sk_kernel  data-parallel + stream-K (Osama et al., "Stream-K: Work-centric Parallel Decomposition for Dense
//                   Matrix-Matrix Multiplication on the GPU", 2023): a persistent grid of G CTAs.  Whole waves of tiles
//                   run data-parallel (tile c, c + G, ... of CTA c); the tiles of the last, partial wave (and
//                   sk_waves more full waves) are split stream-K: their tiles x k-tiles iteration space is divided
//                   evenly among the G CTAs, so no SM idles in a ragged last wave.  (At 384 rows the 18432-wide gate|up
//                   projection has 432 tiles of 128x128 on 340 CTA slots: 1.27 waves.  A 128-row batch has 97 tiles
//                   for the 12352-wide input projection: 57% of the SMs.)  A CTA's stream-K range is whole tiles plus
//                   at most one partial tile at each end.  Partial tiles are reduced through a workspace: every
//                   contributor stores its int32 accumulators in its slot and bumps the tile's counter; the last to
//                   arrive adds the other slots (integer sums are exact, so arrival order does not matter), runs the
//                   epilogue and resets the counter for the next call.  No CTA waits for another.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int bytes = pred ? 16 : 0;  // zero-fill out-of-range rows
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int32_t* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// byte offset of 16-byte chunk `c` of row `r` in a tile whose rows are BK bytes (BK = 64 or 128)
template <int BK>
__device__ __forceinline__ int swz(int r, int c) {
  if constexpr (BK == 64) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
  } else {
    return r * 128 + ((c ^ (r & 7)) << 4);
  }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
struct Tile {
  static constexpr int THREADS = WARPS_M * WARPS_N * 32;
  static constexpr int WM = BM / WARPS_M;  // warp tile rows
  static constexpr int WN = BN / WARPS_N;  // warp tile cols
  static constexpr int MT = WM / 16;       // m16 tiles per warp
  static constexpr int NT = WN / 8;        // n8 tiles per warp
  static constexpr int CH = BK / 16;       // 16-byte chunks per row
  static constexpr int ACC = MT * NT * 4;  // int32 accumulators per thread
  static constexpr int SMEM = STAGES * (BM + BN) * BK;
  static_assert(NT % 2 == 0, "ldmatrix.x4 loads two n8 tiles");
};

// acc += A[m0:m0+BM, kt0*BK:kt1*BK] W[n0:n0+BN, same]^T, through the STAGES-deep cp.async pipeline in `smem`.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void mainloop(const int8_t* __restrict__ A, const int8_t* __restrict__ W, int M, int N,
                                         int K, int m0, int n0, int kt0, int kt1, uint8_t* smem,
                                         int32_t (&acc)[Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::MT]
                                                       [Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::NT][4]) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  constexpr int THREADS = T::THREADS, WM = T::WM, WN = T::WN, MT = T::MT, NT = T::NT, CH = T::CH;
  uint8_t* sA = smem;                         // [STAGES][BM * BK]
  uint8_t* sB = smem + STAGES * BM * BK;      // [STAGES][BN * BK]
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;

  auto load_stage = [&](int stage, int kt) {
    const int kb = kt * BK;
    uint8_t* a = sA + stage * BM * BK;
    uint8_t* b = sB + stage * BN * BK;
#pragma unroll
    for (int i = tid; i < BM * CH; i += THREADS) {
      int r = i / CH, c = i % CH;
      int gm = m0 + r;
      cp_async16(smem_u32(a + swz<BK>(r, c)), A + (size_t)min(gm, M - 1) * K + kb + c * 16, gm < M);
    }
#pragma unroll
    for (int i = tid; i < BN * CH; i += THREADS) {
      int r = i / CH, c = i % CH;
      int gn = n0 + r;
      cp_async16(smem_u32(b + swz<BK>(r, c)), W + (size_t)min(gn, N - 1) * K + kb + c * 16, gn < N);
    }
  };

  const int k_tiles = kt1 - kt0;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) load_stage(s, kt0 + s);
    cp_async_commit();
  }

  for (int kt = 0; kt < k_tiles; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {  // prefetch the tile STAGES-1 ahead into the slot freed last iteration
      int nk = kt + STAGES - 1;
      if (nk < k_tiles) load_stage(nk % STAGES, kt0 + nk);
      cp_async_commit();
    }
    const uint8_t* a = sA + (kt % STAGES) * BM * BK;
    const uint8_t* b = sB + (kt % STAGES) * BN * BK;
#pragma unroll
    for (int ks = 0; ks < BK / 32; ++ks) {  // k32 steps
      uint32_t af[MT][4], bf[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        int r = wm * WM + i * 16 + (lane % 16);
        int c = ks * 2 + lane / 16;
        ldmatrix_x4(smem_u32(a + swz<BK>(r, c)), af[i][0], af[i][1], af[i][2], af[i][3]);
      }
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        int r = wn * WN + j * 8 + (lane % 8) + (lane / 16) * 8;
        int c = ks * 2 + (lane / 8) % 2;
        ldmatrix_x4(smem_u32(b + swz<BK>(r, c)), bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1]);
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_s8(acc[i][j], af[i], bf[j]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // every warp is done with the stages before anyone refills them
}

// The same product with the shared-memory fragments software-pipelined (as CUTLASS's sm80 multistage loop): the
// fragments of the next k32 step load while this step's MMAs run - across k-tile boundaries too - so every ldmatrix
// latency hides behind tensor-core work; the barrier and the wait for the next stage sit at a tile's last k32 step, the
// copy of the stage STAGES-1 ahead is issued at its first.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void mainloop_sp(const int8_t* __restrict__ A, const int8_t* __restrict__ W, int M, int N,
                                            int K, int m0, int n0, int kt0, int kt1, uint8_t* smem,
                                            int32_t (&acc)[Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::MT]
                                                          [Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::NT][4]) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  constexpr int THREADS = T::THREADS, WM = T::WM, WN = T::WN, MT = T::MT, NT = T::NT, CH = T::CH;
  constexpr int KS = BK / 32;
  static_assert(KS % 2 == 0, "fragment double buffer alternates within a tile");
  uint8_t* sA = smem;
  uint8_t* sB = smem + STAGES * BM * BK;
  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;

  auto load_stage = [&](int stage, int kt) {
    const int kb = kt * BK;
    uint8_t* a = sA + stage * BM * BK;
    uint8_t* b = sB + stage * BN * BK;
#pragma unroll
    for (int i = tid; i < BM * CH; i += THREADS) {
      int r = i / CH, c = i % CH;
      int gm = m0 + r;
      cp_async16(smem_u32(a + swz<BK>(r, c)), A + (size_t)min(gm, M - 1) * K + kb + c * 16, gm < M);
    }
#pragma unroll
    for (int i = tid; i < BN * CH; i += THREADS) {
      int r = i / CH, c = i % CH;
      int gn = n0 + r;
      cp_async16(smem_u32(b + swz<BK>(r, c)), W + (size_t)min(gn, N - 1) * K + kb + c * 16, gn < N);
    }
  };
  uint32_t af[2][MT][4], bf[2][NT][2];
  auto load_frags = [&](int buf, int stage, int ks) {
    const uint8_t* a = sA + stage * BM * BK;
    const uint8_t* b = sB + stage * BN * BK;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      int r = wm * WM + i * 16 + (lane % 16);
      int c = ks * 2 + lane / 16;
      ldmatrix_x4(smem_u32(a + swz<BK>(r, c)), af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
    }
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      int r = wn * WN + j * 8 + (lane % 8) + (lane / 16) * 8;
      int c = ks * 2 + (lane / 8) % 2;
      ldmatrix_x4(smem_u32(b + swz<BK>(r, c)), bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    }
  };

  const int k_tiles = kt1 - kt0;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) load_stage(s, kt0 + s);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 2>();
  __syncthreads();
  if (k_tiles > 0) load_frags(0, 0, 0);
  int read_stage = 0, write_stage = STAGES - 1;
  for (int kt = 0; kt < k_tiles; ++kt) {
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      if (ks == KS - 1) {  // the next tile's stage must have landed before its first fragments load
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        read_stage = read_stage + 1 == STAGES ? 0 : read_stage + 1;
      }
      if (ks < KS - 1 || kt + 1 < k_tiles) load_frags((ks + 1) % 2, read_stage, (ks + 1) % KS);
      if (ks == 0) {  // refill the slot every warp finished reading before the last barrier
        int nk = kt + STAGES - 1;
        if (nk < k_tiles) load_stage(write_stage, kt0 + nk);
        cp_async_commit();
        write_stage = write_stage + 1 == STAGES ? 0 : write_stage + 1;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_s8(acc[i][j], af[ks % 2][i], bf[ks % 2][j]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // every warp is done with the stages before anyone refills them
}

template <bool SP, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__device__ __forceinline__ void run_mainloop(const int8_t* __restrict__ A, const int8_t* __restrict__ W, int M, int N,
                                             int K, int m0, int n0, int kt0, int kt1, uint8_t* smem,
                                             int32_t (&acc)[Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::MT]
                                                           [Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::NT][4]) {
  if constexpr (SP) {
    mainloop_sp<BM, BN, BK, WARPS_M, WARPS_N, STAGES>(A, W, M, N, K, m0, n0, kt0, kt1, smem, acc);
  } else {
    mainloop<BM, BN, BK, WARPS_M, WARPS_N, STAGES>(A, W, M, N, K, m0, n0, kt0, kt1, smem, acc);
  }
}

enum EpilogueMode { ADD = 0, ATOMIC = 1, STORE = 2, STORE_I32 = 3 };
// how the K splits of a tile combine (residual-add epilogue): atomic fp32 adds (order-dependent); int32 partial sums
// through a global workspace or over distributed shared memory in a cluster; or each split stores its int32 partial
// sums in its own slot and the next kernel that reads the residual adds them, scales the total and adds it (no sync).
// All but the atomic adds give the bits of an unsplit product.
enum SplitMode { SPLIT_ATOMIC = 0, SPLIT_WORKSPACE = 1, SPLIT_CLUSTER = 2, SPLIT_DEFER = 3 };

// Programmatic dependent launch: the GEMM is launched while the kernel that writes its activations (a norm or
// quantize kernel, which signals at its start) still runs, and waits for that grid to finish and its writes to be
// visible before anything reads A, its scales or the residual.  Weights do not depend on that kernel: with prefetch
// (k_end > k_begin) the CTA first streams bytes k_begin..k_end of its weight rows n0.. toward L2.
template <int BN, int THREADS>
__device__ __forceinline__ void pdl_prologue(const int8_t* W, int N, int K, int n0, int k_begin, int k_end) {
  const int lines = (k_end - k_begin + 127) / 128;
  for (int i = threadIdx.x; i < BN * lines; i += THREADS) {
    const int r = n0 + i / lines;
    if (r < N) asm volatile("prefetch.global.L2 [%0];" ::"l"(W + (size_t)r * K + k_begin + (i % lines) * 128));
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// pdl: 0 ordinary launch; 1 early launch, no prefetch; 2 the first M tile's CTAs prefetch the first PDL_PF k-tiles of
// their weight panel; 3 every CTA prefetches its whole panel
constexpr int PDL_PF = 4;
__device__ __forceinline__ int pdl_prefetch_end(int pdl, bool first_m_tile, int k_begin, int k_end, int bk) {
  if (pdl == 3) return k_end;
  if (pdl == 2 && first_m_tile) return min(k_end, k_begin + PDL_PF * bk);
  return k_begin;
}

// acc (row = groupID or groupID + 8, col = 2 * (lane % 4) + {0, 1} of each m16n8 tile) * sa[row] * sw[col] -> out
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int EPI>
__device__ __forceinline__ void epilogue(const int32_t (&acc)[Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::MT]
                                                             [Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>::NT][4],
                                         const float* __restrict__ sa, const float* __restrict__ sw, void* out,
                                         int ldo, int live, int N, int m0, int n0, int mode) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  constexpr int WM = T::WM, WN = T::WN, MT = T::MT, NT = T::NT;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int g = lane / 4, t4 = lane % 4;
#ifndef FLASHJEV_EPI_SKIP
  if constexpr (EPI != 0) {
    if (mode == ADD) {
      // residual add, a row slice (16-row tile i, half) at a time: all of the slice's residual loads are issued
      // before the first add, so their DRAM latencies overlap (one read-add-write after another, as the generic
      // path below does it, paid the full latency per element pair: 29% of a 4,096-row down projection)
#pragma unroll
      for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = m0 + wm * WM + i * 16 + g + half * 8;
          if (row >= live) continue;
          const float s_a = sa[row];
          float* orow = reinterpret_cast<float*>(out) + (size_t)row * ldo;
          float2 old[NT];
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * WN + j * 8 + 2 * t4;
            old[j] = col + 1 < N ? *reinterpret_cast<const float2*>(orow + col)
                                 : make_float2(col < N ? orow[col] : 0.f, 0.f);
          }
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * WN + j * 8 + 2 * t4;
            if (col >= N) continue;
            const float v0 = __fmul_rn(__fmul_rn((float)acc[i][j][half * 2], s_a), sw[col]);
            if (col + 1 < N) {
              const float v1 = __fmul_rn(__fmul_rn((float)acc[i][j][half * 2 + 1], s_a), sw[col + 1]);
              *reinterpret_cast<float2*>(orow + col) = make_float2(__fadd_rn(old[j].x, v0), __fadd_rn(old[j].y, v1));
            } else {
              orow[col] = __fadd_rn(old[j].x, v0);
            }
          }
        }
      }
      return;
    }
  }
#endif
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int row = m0 + wm * WM + i * 16 + g + half * 8;
      if (row >= live) continue;
      float s_a = sa[row];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        int col = n0 + wn * WN + j * 8 + 2 * t4;
        if (col >= N) continue;
        if constexpr (EPI != 0) {
          if (mode == STORE_I32) {  // a deferred split's raw partial sums: the reader adds the splits, then scales
            int32_t* o = reinterpret_cast<int32_t*>(out) + (size_t)row * ldo + col;
            if (col + 1 < N) {
              *reinterpret_cast<int2*>(o) = make_int2(acc[i][j][half * 2], acc[i][j][half * 2 + 1]);
            } else {
              o[0] = acc[i][j][half * 2];
            }
            continue;
          }
        }
        float v0 = __fmul_rn(__fmul_rn((float)acc[i][j][half * 2], s_a), sw[col]);
        float v1 = __fmul_rn(__fmul_rn((float)acc[i][j][half * 2 + 1], s_a), col + 1 < N ? sw[col + 1] : 0.f);
#ifdef FLASHJEV_EPI_SKIP  // measurement build (bench/epilogue_cost.py): no output writes; the product stays live
        if (v0 == 1.2345e30f && v1 == 1.2345e30f) reinterpret_cast<int32_t*>(out)[0] = 0;
        continue;
#endif
        if constexpr (EPI == 0) {
          __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)row * ldo + col;
          if (col + 1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(v0, v1);
          } else {
            o[0] = __float2bfloat16(v0);
          }
        } else {
          float* o = reinterpret_cast<float*>(out) + (size_t)row * ldo + col;
          if (mode == ADD) {
            o[0] = __fadd_rn(o[0], v0);
            if (col + 1 < N) o[1] = __fadd_rn(o[1], v1);
          } else if (mode == STORE) {
            if (col + 1 < N) {
              *reinterpret_cast<float2*>(o) = make_float2(v0, v1);
            } else {
              o[0] = v0;
            }
          } else {
            atomicAdd(o, v0);
            if (col + 1 < N) atomicAdd(o + 1, v1);
          }
        }
      }
    }
  }
}

template <bool SP, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int EPI>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32)
    w8a8_kernel(const int8_t* __restrict__ A, const float* __restrict__ sa, const int8_t* __restrict__ W,
                const float* __restrict__ sw, void* __restrict__ out, int ldo, const int* __restrict__ n_rows, int M,
                int N, int K, int k_per_split, int32_t* __restrict__ ws, int* __restrict__ counters, int cluster_k,
                int64_t slot_stride, int pdl) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  const int live = *n_rows;  // written by the batch's table upload, not by the kernel this one may overlap
  const int m0 = blockIdx.y * BM;
  if (m0 >= live) return;
  const int n0 = blockIdx.x * BN;
  const int kt0 = blockIdx.z * (k_per_split / BK);
  if (pdl)
    pdl_prologue<BN, T::THREADS>(W, N, K, n0, kt0 * BK,
                                 pdl_prefetch_end(pdl, blockIdx.y == 0, kt0 * BK, kt0 * BK + k_per_split, BK));
  extern __shared__ __align__(128) uint8_t smem[];
  int32_t acc[T::MT][T::NT][4];
#pragma unroll
  for (int i = 0; i < T::MT; ++i)
#pragma unroll
    for (int j = 0; j < T::NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
  run_mainloop<SP, BM, BN, BK, WARPS_M, WARPS_N, STAGES>(A, W, M, N, K, m0, n0, kt0, kt0 + k_per_split / BK, smem,
                                                        acc);
  if (EPI == 1 && gridDim.z > 1) {
    using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
    constexpr int THREADS = T::THREADS, MT = T::MT, NT = T::NT, ACC = T::ACC;
    const int tid = threadIdx.x;
    if (slot_stride > 0) {  // deferred: this split's int32 partial sums into its own slot; the residual's reader
                            // adds them (exact), scales the total and adds it into the residual
      epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw,
                                                          reinterpret_cast<int32_t*>(out) + blockIdx.z * slot_stride,
                                                          ldo, live, N, m0, n0, STORE_I32);
      return;
    }
    if (cluster_k) {
      // deterministic split-K in a thread block cluster: the K splits of this tile are one cluster; each leaves its
      // int32 partial sums in its own shared memory, and rank 0 adds them over distributed shared memory (integer
      // sums: exact in any order) and runs the one epilogue.  No global round trip.
      if constexpr (ACC * THREADS * 4 <= T::SMEM) {
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int32_t* part = reinterpret_cast<int32_t*>(smem);
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[((i * NT + j) * 4 + e) * THREADS + tid] = acc[i][j][e];
        cluster.sync();
        if (cluster.block_rank() == 0) {
          for (unsigned r = 1; r < cluster.num_blocks(); ++r) {
            const int32_t* remote = cluster.map_shared_rank(part, r);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
              for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] += remote[((i * NT + j) * 4 + e) * THREADS + tid];
          }
          epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ADD);
        }
        cluster.sync();  // the other ranks' shared memory must stay alive until rank 0 has read it
      }
      return;
    }
    if (ws == nullptr) {  // atomic adds: the fp32 sum depends on the order the splits arrive in
      epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ATOMIC);
      return;
    }
    // deterministic split-K through global memory: every split stores its int32 partial sums; the last of the
    // tile's splits to arrive adds the others' and runs the one epilogue
    __shared__ int s_prev;
    const int tile = blockIdx.y * gridDim.x + blockIdx.x, split = gridDim.z;
    int32_t* slots = ws + (size_t)tile * split * ACC * THREADS + tid;
    int32_t* mine = slots + (size_t)blockIdx.z * ACC * THREADS;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) __stcg(mine + ((i * NT + j) * 4 + e) * THREADS, acc[i][j][e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_prev = atomicAdd(counters + tile, 1);
    __syncthreads();
    if (s_prev != split - 1) return;
    __threadfence();
    for (int z = 0; z < split; ++z) {
      if (z == (int)blockIdx.z) continue;
      const int32_t* p = slots + (size_t)z * ACC * THREADS;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[i][j][e] += __ldcg(p + ((i * NT + j) * 4 + e) * THREADS);
    }
    epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ADD);
    if (tid == 0) counters[tile] = 0;
    return;
  }
  epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ADD);
}

template <bool SP, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int EPI>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32)
    w8a8_sk_kernel(const int8_t* __restrict__ A, const float* __restrict__ sa, const int8_t* __restrict__ W,
                   const float* __restrict__ sw, void* __restrict__ out, int ldo, const int* __restrict__ n_rows,
                   int M, int N, int K, int32_t* __restrict__ ws, int* __restrict__ counters, int sk_waves,
                   int pdl) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  constexpr int THREADS = T::THREADS, MT = T::MT, NT = T::NT, ACC = T::ACC;
  const int live = min(*n_rows, M);
  const int tiles_m = (live + BM - 1) / BM, tiles_n = (N + BN - 1) / BN, iters = K / BK;
  const long long tiles = (long long)tiles_m * tiles_n;
  const long long grid = gridDim.x;
  const int c = blockIdx.x;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_prev;
  const int tid = threadIdx.x;
  int32_t acc[MT][NT][4];
  auto zero = [&]() {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
  };
  // tiles in M-fastest order: CTAs running together share the weight columns, read from DRAM once
  auto tile_m0 = [&](long long t) { return (int)(t % tiles_m) * BM; };
  auto tile_n0 = [&](long long t) { return (int)(t / tiles_m) * BN; };
  if (pdl && tiles > 0)  // this CTA's first tile's panel
    pdl_prologue<BN, THREADS>(W, N, K, tile_n0(c % tiles), 0, pdl_prefetch_end(pdl, tile_m0(c % tiles) == 0, 0, K, BK));

  // data-parallel part: every full wave except the last sk_waves (all of them when the tiles fill whole waves)
  long long dp = tiles / grid * grid;
  if (dp != tiles) dp = max(0LL, dp - sk_waves * grid);
  for (long long t = c; t < dp; t += grid) {
    zero();
    run_mainloop<SP, BM, BN, BK, WARPS_M, WARPS_N, STAGES>(A, W, M, N, K, tile_m0(t), tile_n0(t), 0, iters, smem,
                                                          acc);
    epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, tile_m0(t), tile_n0(t),
                                                        ADD);
  }

  // stream-K part: tiles dp.. split evenly by iterations, at most one CTA per iteration
  const long long total = (tiles - dp) * iters;
  const long long G = min(grid, total);
  if (c >= G) return;
  auto start = [&](long long cta) { return total * cta / G; };                 // first iteration of a CTA
  auto owner = [&](long long x) { return (int)(((x + 1) * G - 1) / total); };  // the CTA holding iteration x
  const long long my_start = start(c), my_end = start(c + 1);
  for (long long x = my_start; x < my_end;) {
    const long long ts = x / iters;                  // stream-K tile index; the tile is dp + ts
    const int k0 = (int)(x - ts * iters);
    const int k1 = (int)min((long long)iters, k0 + (my_end - x));
    const long long t = dp + ts;
    const int m0 = tile_m0(t), n0 = tile_n0(t);
    zero();
    run_mainloop<SP, BM, BN, BK, WARPS_M, WARPS_N, STAGES>(A, W, M, N, K, m0, n0, k0, k1, smem, acc);
    if (k0 == 0 && k1 == iters) {
      epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ADD);
    } else {
      // slot 0: the segment that starts this CTA's range (it began inside the tile); slot 1: the one that ends it
      const long long t0 = ts * iters;
      int32_t* mine = ws + (size_t)(2 * c + (my_start >= t0 ? 0 : 1)) * ACC * THREADS + tid;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) __stcg(mine + ((i * NT + j) * 4 + e) * THREADS, acc[i][j][e]);
      __threadfence();
      __syncthreads();
      if (tid == 0) s_prev = atomicAdd(counters + t, 1);
      __syncthreads();
      const int lo = owner(t0), hi = owner(t0 + iters - 1);
      if (s_prev == hi - lo) {  // the last contributor to arrive: add the others' partial sums, write the tile
        __threadfence();
        for (int cc = lo; cc <= hi; ++cc) {
          if (cc == c) continue;
          const int32_t* p = ws + (size_t)(2 * cc + (start(cc) >= t0 ? 0 : 1)) * ACC * THREADS + tid;
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[i][j][e] += __ldcg(p + ((i * NT + j) * 4 + e) * THREADS);
        }
        epilogue<BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>(acc, sa, sw, out, ldo, live, N, m0, n0, ADD);
        if (tid == 0) counters[t] = 0;
      }
    }
    x += k1 - k0;
  }
}

template <bool SP, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int EPI>
void launch(const torch::Tensor& a, const torch::Tensor& sa, const torch::Tensor& w, const torch::Tensor& sw,
            torch::Tensor& out, const torch::Tensor& n_rows, int split, int mode, int32_t* ws, int64_t ws_numel,
            int* counters, int64_t slot_stride, int pdl) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  const int M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % (BK * split) == 0, "K must be a multiple of BK * split");
  auto kern = w8a8_kernel<SP, BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    configured = true;
  }
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, split);
  const int cluster_k = split > 1 && mode == SPLIT_CLUSTER;
  if (split > 1 && mode == SPLIT_WORKSPACE) {
    TORCH_CHECK((int64_t)grid.x * grid.y <= (1 << 16), "too many tiles for the split-K counters");
    TORCH_CHECK((int64_t)grid.x * grid.y * split * BM * BN <= ws_numel, "split-K workspace too small");
  }
  if (cluster_k) {
    TORCH_CHECK(T::ACC * T::THREADS * 4 <= T::SMEM, "cluster split-K: the tile's partial sums exceed its stages");
    TORCH_CHECK(split <= 8, "cluster split-K: at most 8 splits");
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(T::THREADS);
  config.dynamicSmemBytes = T::SMEM;
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  config.attrs = attr;
  config.numAttrs = 0;
  if (cluster_k) {
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = split;
    config.numAttrs = 1;
  } else if (pdl) {  // may start while the previous kernel runs (see pdl_prologue)
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kern, a.data_ptr<int8_t>(), sa.data_ptr<float>(), w.data_ptr<int8_t>(),
                                    sw.data_ptr<float>(), out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(),
                                    M, N, K, K / split, mode == SPLIT_WORKSPACE ? ws : nullptr,
                                    mode == SPLIT_WORKSPACE ? counters : nullptr, cluster_k,
                                    split > 1 && mode == SPLIT_DEFER ? slot_stride : (int64_t)0,
                                    cluster_k ? 0 : pdl));
}

template <bool SP, int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int EPI>
void launch_sk(const torch::Tensor& a, const torch::Tensor& sa, const torch::Tensor& w, const torch::Tensor& sw,
               torch::Tensor& out, const torch::Tensor& n_rows, torch::Tensor& ws, torch::Tensor& counters, int grid,
               int sk_waves, int pdl) {
  using T = Tile<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  const int M = a.size(0), K = a.size(1), N = w.size(0);
  TORCH_CHECK(K % BK == 0, "K must be a multiple of BK");
  TORCH_CHECK(ws.numel() >= (int64_t)2 * grid * T::ACC * T::THREADS, "stream-K workspace too small");
  TORCH_CHECK(counters.numel() >= (int64_t)((M + BM - 1) / BM) * ((N + BN - 1) / BN), "too few tile counters");
  auto kern = w8a8_sk_kernel<SP, BM, BN, BK, WARPS_M, WARPS_N, STAGES, EPI>;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    configured = true;
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(T::THREADS);
  config.dynamicSmemBytes = T::SMEM;
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kern, a.data_ptr<int8_t>(), sa.data_ptr<float>(), w.data_ptr<int8_t>(),
                                    sw.data_ptr<float>(), out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(),
                                    M, N, K, ws.data_ptr<int32_t>(), counters.data_ptr<int>(), (int)sk_waves, pdl));
}

}  // namespace

// cfg: 0 = 128x128x64 (8 warps, 2x4), 1 = 64x128x64 (4 warps, 2x2), 2 = 64x64x128 (4 warps, 2x2),
//      3 = 128x64x64 (4 warps, 2x2), 4 = 64x256x64 (8 warps, 2x4), 5 = 32x128x128 (4 warps, 1x4),
//      6 = 256x128x64 (8 warps, 4x2), 7 = 128x256x64 (8 warps, 2x4);
//      8..12 = 0, 3, 6, 7, 2 with the software-pipelined main loop, 13 = 128x128x128 (8 warps, 2x4) pipelined
#define W8A8_CONFIGS(X)           \
  X(0, false, 128, 128, 64, 2, 4)  \
  X(1, false, 64, 128, 64, 2, 2)   \
  X(2, false, 64, 64, 128, 2, 2)   \
  X(3, false, 128, 64, 64, 2, 2)   \
  X(4, false, 64, 256, 64, 2, 4)   \
  X(5, false, 32, 128, 128, 1, 4)  \
  X(6, false, 256, 128, 64, 4, 2)  \
  X(7, false, 128, 256, 64, 2, 4)  \
  X(8, true, 128, 128, 64, 2, 4)   \
  X(9, true, 128, 64, 64, 2, 2)    \
  X(10, true, 256, 128, 64, 4, 2)  \
  X(11, true, 128, 256, 64, 2, 4)  \
  X(12, true, 64, 64, 128, 2, 2)   \
  X(13, true, 128, 128, 128, 2, 4)

static void check_operands(const torch::Tensor& a, const torch::Tensor& w, const torch::Tensor& out, int64_t epi,
                           bool defer = false) {
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8, "int8 operands");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1 && a.stride(0) == a.size(1) && w.stride(0) == w.size(1),
              "contiguous K-major operands");
  TORCH_CHECK(defer ? out.dtype() == torch::kInt32
                    : (epi == 0 ? out.dtype() == torch::kBFloat16 : out.dtype() == torch::kFloat32), "output dtype");
}

void w8a8_gemm(torch::Tensor a, torch::Tensor sa, torch::Tensor w, torch::Tensor sw, torch::Tensor out,
               torch::Tensor n_rows, int64_t epi, int64_t cfg, int64_t split, int64_t stages, int64_t mode,
               std::optional<torch::Tensor> workspace, std::optional<torch::Tensor> counters, int64_t slot_stride,
               int64_t pdl) {
  check_operands(a, w, out, epi, mode == SPLIT_DEFER && split > 1);
  // split-K of the residual add (mode, SplitMode): with SPLIT_DEFER `out` is the first of `split` int32 slots,
  // slot_stride elements apart
  TORCH_CHECK(mode != SPLIT_DEFER || split == 1 || slot_stride >= out.size(0) * out.stride(0), "deferred slots overlap");
  TORCH_CHECK(mode != SPLIT_WORKSPACE || split == 1 || workspace.has_value(), "workspace split-K needs the workspace");
  int32_t* ws = workspace.has_value() ? workspace->data_ptr<int32_t>() : nullptr;
  int64_t ws_n = workspace.has_value() ? workspace->numel() : 0;
  int* cnt = counters.has_value() ? counters->data_ptr<int>() : nullptr;
  TORCH_CHECK((ws == nullptr) == (cnt == nullptr), "workspace and counters go together");
  const int mode_i = (int)mode;
#define W8A8_CASE(C, SP, BM, BN, BK, WMN, WNN)                                                   \
  case C:                                                                                         \
    if (epi == 0) {                                                                               \
      if (stages == 3) launch<SP, BM, BN, BK, WMN, WNN, 3, 0>(a, sa, w, sw, out, n_rows, split, mode_i, ws, ws_n, cnt, slot_stride, (int)pdl);  \
      else launch<SP, BM, BN, BK, WMN, WNN, 4, 0>(a, sa, w, sw, out, n_rows, split, mode_i, ws, ws_n, cnt, slot_stride, (int)pdl);              \
    } else {                                                                                      \
      if (stages == 3) launch<SP, BM, BN, BK, WMN, WNN, 3, 1>(a, sa, w, sw, out, n_rows, split, mode_i, ws, ws_n, cnt, slot_stride, (int)pdl);  \
      else launch<SP, BM, BN, BK, WMN, WNN, 4, 1>(a, sa, w, sw, out, n_rows, split, mode_i, ws, ws_n, cnt, slot_stride, (int)pdl);              \
    }                                                                                             \
    break;
  switch (cfg) {
    W8A8_CONFIGS(W8A8_CASE)
    default:
      TORCH_CHECK(false, "unknown cfg");
  }
#undef W8A8_CASE
}

void w8a8_gemm_sk(torch::Tensor a, torch::Tensor sa, torch::Tensor w, torch::Tensor sw, torch::Tensor out,
                  torch::Tensor n_rows, torch::Tensor ws, torch::Tensor counters, int64_t epi, int64_t cfg,
                  int64_t stages, int64_t grid, int64_t sk_waves, int64_t pdl) {
  check_operands(a, w, out, epi);
  TORCH_CHECK(ws.dtype() == torch::kInt32 && counters.dtype() == torch::kInt32, "int32 workspace and counters");
#define W8A8_SK_CASE(C, SP, BM, BN, BK, WMN, WNN)                                                          \
  case C:                                                                                                    \
    if (epi == 0) {                                                                                          \
      if (stages == 3)                                                                                       \
        launch_sk<SP, BM, BN, BK, WMN, WNN, 3, 0>(a, sa, w, sw, out, n_rows, ws, counters, grid, sk_waves, (int)pdl);  \
      else launch_sk<SP, BM, BN, BK, WMN, WNN, 4, 0>(a, sa, w, sw, out, n_rows, ws, counters, grid, sk_waves, (int)pdl); \
    } else {                                                                                                 \
      if (stages == 3)                                                                                       \
        launch_sk<SP, BM, BN, BK, WMN, WNN, 3, 1>(a, sa, w, sw, out, n_rows, ws, counters, grid, sk_waves, (int)pdl);  \
      else launch_sk<SP, BM, BN, BK, WMN, WNN, 4, 1>(a, sa, w, sw, out, n_rows, ws, counters, grid, sk_waves, (int)pdl); \
    }                                                                                                        \
    break;
  switch (cfg) {
    W8A8_CONFIGS(W8A8_SK_CASE)
    default:
      TORCH_CHECK(false, "unknown cfg");
  }
#undef W8A8_SK_CASE
}

TORCH_LIBRARY(flash_jev, m) {
  m.def("w8a8_gemm(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor(a!) out, Tensor n_rows, int epi, int cfg, "
        "int split, int stages, int mode=2, Tensor(b!)? workspace=None, Tensor(c!)? counters=None, "
        "int slot_stride=0, int pdl=0) -> ()");
  m.def("w8a8_gemm_sk(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor(a!) out, Tensor n_rows, Tensor(b!) ws, "
        "Tensor(c!) counters, int epi, int cfg, int stages, int grid, int sk_waves, int pdl=0) -> ()");
}
TORCH_LIBRARY_IMPL(flash_jev, CUDA, m) {
  m.impl("w8a8_gemm", &w8a8_gemm);
  m.impl("w8a8_gemm_sk", &w8a8_gemm_sk);
}
