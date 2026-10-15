// W4A8 GEMM on a Q4_K_M file's own weights (kernels/w4a8.py), for sm_80+ (built for sm_120): llama.cpp k-quant values
// as they are, int8 activations with one scale per 256 inputs, int8 mma.sync tensor cores.
//
//   Q4_K (KIND 0)  W packed two 4-bit values per byte, per 32 inputs 16 bytes, byte t = q[t] | q[t + 16] << 4;
//                  sc, mn uint8 [K / 256, N, 8] (per 32 inputs); d, dmin fp32 [K / 256, N].  w = d * sc * q - dmin * mn
//   Q6_K (KIND 1)  W int8 (q - 32); sc int8 [K / 256, N, 16] (per 16 inputs); d fp32 [K / 256, N].   w = d * sc * q
//   A int8 [M, K] row-major, a fp32 [K / 256, M], xs int8 [K / 256, M, 16] (Q4_K only: per 256 inputs the sums S of
//   the eight groups of 32, |S| <= 32 * 127, split as S = (S & 63) + 64 (S >> 6): the eight S & 63, then the eight
//   S >> 6).  The scales and sums are super-block-major, so a stage's copies of them read whole lines (row-major,
//   each column's few bytes sat a row apart: most of the copy time of the K = 9216 projections).
//
// The arithmetic is kernels/w4a8.py's, bit for bit (tests/test_w4a8.py): per 256-input super-block, in increasing
// order, the sub-blocks' int32 dot products times their integer scales (and Q4_K's mins times the activation sums) are
// summed exactly in int32, then t = d * float(acc), t = fma(-dmin, float(mins), t) and y = fma(a, t, y), each rounded
// once (explicit _rn intrinsics); the epilogue rounds y to bf16 or adds it into the fp32 residual.  So an output
// element has the same bits whichever tile, config or batch computes it (batch invariance), as the W8A8 path.
//
// Main loop as w8a8_gemm.cu's: cp.async 16-byte copies (and 4/8-byte ones for the scales) STAGES deep, 128 inputs per
// stage (two stages per super-block), XOR-swizzled 16-byte chunks read with ldmatrix.x4.  A Q4_K thread's B fragment
// for one k32 step is one 32-bit word of the packed tile (bytes 4i..4i+3 of its row: k = 4i..4i+3 in the low nibbles,
// 4i+16..4i+19 in the high ones), split by two masks; Q6_K runs each k32 step as two m16n8k16 MMAs (one scale per 16).
// A Q4_K thread reads its columns' sub-block scales once per stage (a word: four sub-blocks), a Q6_K one per step (two
// bytes).  The main loop runs a super-block (two stages) per iteration, so its first sub-block starts the integer sums
// and its floating-point step follows the last one without a branch; the super-block's second stage also brings the
// values only that step reads (d, dmin, mins, the activations' scales and split sums).  Q4_K's mins are not
// accumulated per step: the floating-point step takes the sum of mins x sums from two m16n8k16 MMAs per tile (the low
// and the high parts of the sums).  Tiles up to 128x128 with 32x32, 64x32 or 32x64 warp tiles (W4A8_CONFIGS; the
// minimum CTAs per SM sets the register cap: 128 or 255 per thread).
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
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 8 : 0));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 4 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
// c = a b (fresh accumulators: one sub-block's dot products)
__device__ __forceinline__ void mma_k32(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma_k16(int32_t* c, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// byte offset of 16-byte chunk c of row r in a tile of 64- or 128-byte rows (ldmatrix of 8 rows: distinct banks)
template <int ROW>
__device__ __forceinline__ int swz(int r, int c) {
  if constexpr (ROW == 64) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
  } else {
    return r * 128 + ((c ^ (r & 7)) << 4);
  }
}

constexpr int BK = 128;  // inputs per stage: half a super-block

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int KIND>
struct Cfg {
  static constexpr int THREADS = WARPS_M * WARPS_N * 32;
  static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, MT = WM / 16, NT = WN / 8;
  static constexpr int WROW = KIND == 0 ? BK / 2 : BK;  // weight bytes per row and stage
  static constexpr int SCB = KIND == 0 ? 4 : 8;         // scale bytes per weight row and stage
  static constexpr int OFF_W = BM * BK;
  static constexpr int OFF_SC = OFF_W + BN * WROW;
  static constexpr int OFF_MN = OFF_SC + BN * SCB;                      // Q4_K: the super-block's 8 mins per row
  static constexpr int OFF_XS = OFF_MN + (KIND == 0 ? BN * 8 : 0);      // Q4_K: its 16 split-sum bytes per row
  static constexpr int OFF_AS = OFF_XS + (KIND == 0 ? BM * 16 : 0);
  static constexpr int OFF_D = OFF_AS + BM * 4;
  static constexpr int OFF_DM = OFF_D + BN * 4;
  static constexpr int STAGE = (OFF_DM + (KIND == 0 ? BN * 4 : 0) + 127) / 128 * 128;
  static constexpr int SMEM = STAGES * STAGE;
  static_assert(NT % 4 == 0 || KIND == 1, "Q4_K B fragments load four n8 tiles at a time");
  static_assert(NT % 2 == 0, "B fragments load two n8 tiles at a time");
};

// Programmatic dependent launch (as csrc/w8a8_gemm.cu): the GEMM is launched while the kernel that writes its
// activations (a norm or quantize kernel, which signals at its start) still runs.  The weights do not depend on it:
// pdl 2 has the CTAs of the first row tile stream the first PDL_PF stages of their weight rows toward L2, pdl 3 every
// CTA its whole rows (pdl 1 none); then each CTA waits for that kernel before it reads anything (even the row count).
// The engine launches its W4A8 GEMMs without it: end to end it measured within noise (docs/optimizations.md).
constexpr int PDL_PF = 4;

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int KIND, int EPI, int MINB>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINB)
    w4a8_kernel(const int8_t* __restrict__ A, const float* __restrict__ as, const int8_t* __restrict__ xs,
                const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ wsc, const uint8_t* __restrict__ wmn,
                const float* __restrict__ wd, const float* __restrict__ wdm, void* __restrict__ out, int ldo,
                const int* __restrict__ n_rows, int M, int N, int K, int pdl) {
  using C = Cfg<BM, BN, WARPS_M, WARPS_N, STAGES, KIND>;
  constexpr int THREADS = C::THREADS, WM = C::WM, WN = C::WN, MT = C::MT, NT = C::NT;
  const int n0 = blockIdx.x * BN;
  if (pdl) {
    const int row_bytes = KIND == 0 ? K / 2 : K;
    const int end = pdl == 3 ? row_bytes : pdl == 2 && blockIdx.y == 0 ? min(row_bytes, PDL_PF * C::WROW) : 0;
    const int lines = (end + 127) / 128;
    for (int i = threadIdx.x; i < BN * lines; i += THREADS) {
      const int r = n0 + i / lines;
      if (r < N) asm volatile("prefetch.global.L2 [%0];" ::"l"(Wq + (size_t)r * row_bytes + (i % lines) * 128));
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900  // (before sm_90 no launch is early: sm90_or_later)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  }
  const int live = *n_rows;
  const int m0 = blockIdx.y * BM;
  if (m0 >= live) return;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int g = lane / 4, t4 = lane % 4;
  const int NB = K / 256;

  auto load_stage = [&](int stage, int kt) {
    uint8_t* st = smem + stage * C::STAGE;
    const int k0 = kt * BK, b = kt / 2;
#pragma unroll
    for (int i = tid; i < BM * (BK / 16); i += THREADS) {
      const int r = i / (BK / 16), c = i % (BK / 16), gm = m0 + r;
      cp_async16(smem_u32(st + swz<128>(r, c)), A + (size_t)min(gm, M - 1) * K + k0 + c * 16, gm < M);
    }
#pragma unroll
    for (int i = tid; i < BN * (C::WROW / 16); i += THREADS) {
      const int r = i / (C::WROW / 16), c = i % (C::WROW / 16), gn = n0 + r;
      cp_async16(smem_u32(st + C::OFF_W + swz<C::WROW>(r, c)),
                 Wq + (size_t)min(gn, N - 1) * (K / (KIND == 0 ? 2 : 1)) + k0 / (KIND == 0 ? 2 : 1) + c * 16,
                 gn < N);
    }
    // the stage's sub-block scales of the tile's columns (consecutive in memory); the super-block's second stage
    // also brings what its floating-point step reads: the columns' d (and Q4_K's dmin and mins), the rows' activation
    // scales (and Q4_K's split sums)
    const bool last = kt % 2 == 1;
    for (int i = tid; i < BN; i += THREADS) {
      const size_t gn = (size_t)b * N + min(n0 + i, N - 1);
      const bool ok = n0 + i < N;
      if constexpr (KIND == 0) {
        cp_async4(smem_u32(st + C::OFF_SC + i * 4), wsc + gn * 8 + (kt % 2) * 4, ok);
        if (last) {
          cp_async8(smem_u32(st + C::OFF_MN + i * 8), wmn + gn * 8, ok);
          cp_async4(smem_u32(st + C::OFF_DM + i * 4), wdm + gn, ok);
        }
      } else {
        cp_async8(smem_u32(st + C::OFF_SC + i * 8), wsc + gn * 16 + (kt % 2) * 8, ok);
      }
      if (last) cp_async4(smem_u32(st + C::OFF_D + i * 4), wd + gn, ok);
    }
    if (last) {
      for (int i = tid; i < BM; i += THREADS) {
        const size_t gm = (size_t)b * M + min(m0 + i, M - 1);
        const bool ok = m0 + i < M;
        if constexpr (KIND == 0) cp_async16(smem_u32(st + C::OFF_XS + i * 16), xs + gm * 16, ok);
        cp_async4(smem_u32(st + C::OFF_AS + i * 4), as + gm, ok);
      }
    }
  };

  float y[MT][NT][4];
  int32_t acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) y[i][j][e] = 0.f;

  const int k_tiles = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < k_tiles) load_stage(s, s);
    cp_async_commit();
  }
  for (int kb = 0; kb < NB; ++kb) {  // a super-block (two stages) per iteration
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int kt = 2 * kb + h;
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      {
        const int nk = kt + STAGES - 1;
        if (nk < k_tiles) load_stage(nk % STAGES, nk);
        cp_async_commit();
      }
      const uint8_t* st = smem + (kt % STAGES) * C::STAGE;
      // this thread's columns' sub-block scales of the stage: Q4_K one byte per sub-block (a word per column), Q6_K
      // eight int8 scales per column (read per step)
      uint32_t scw[KIND == 0 ? NT : 1][2];
      if constexpr (KIND == 0) {
#pragma unroll
        for (int jt = 0; jt < NT; ++jt)
#pragma unroll
          for (int c = 0; c < 2; ++c)
            scw[jt][c] = *reinterpret_cast<const uint32_t*>(st + C::OFF_SC + (wn * WN + jt * 8 + 2 * t4 + c) * 4);
      }
      // fragments double-buffered across the stage's k32 steps: step ks + 1's ldmatrix loads issue before step ks's
      // MMAs, so their latency hides behind tensor-core work (single-buffered Q6_K fragments spilled more, not less)
      uint32_t af[2][MT][4], bf[2][NT][2];
      auto load_frags = [&](uint32_t (&a)[MT][4], uint32_t (&b)[NT][2], int ks) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int r = wm * WM + i * 16 + (lane % 16);
          ldmatrix_x4(smem_u32(st + swz<128>(r, ks * 2 + lane / 16)), a[i][0], a[i][1], a[i][2], a[i][3]);
        }
        if constexpr (KIND == 0) {
#pragma unroll
          for (int jt = 0; jt < NT; jt += 4) {
            const int r = wn * WN + (jt + lane / 8) * 8 + (lane % 8);
            uint32_t w0, w1, w2, w3;
            ldmatrix_x4(smem_u32(st + C::OFF_W + swz<64>(r, ks)), w0, w1, w2, w3);
            b[jt][0] = w0 & 0x0F0F0F0Fu, b[jt][1] = (w0 >> 4) & 0x0F0F0F0Fu;
            b[jt + 1][0] = w1 & 0x0F0F0F0Fu, b[jt + 1][1] = (w1 >> 4) & 0x0F0F0F0Fu;
            b[jt + 2][0] = w2 & 0x0F0F0F0Fu, b[jt + 2][1] = (w2 >> 4) & 0x0F0F0F0Fu;
            b[jt + 3][0] = w3 & 0x0F0F0F0Fu, b[jt + 3][1] = (w3 >> 4) & 0x0F0F0F0Fu;
          }
        } else {
#pragma unroll
          for (int jt = 0; jt < NT; jt += 2) {
            const int r = wn * WN + jt * 8 + (lane % 8) + (lane / 16) * 8;
            ldmatrix_x4(smem_u32(st + C::OFF_W + swz<128>(r, ks * 2 + (lane / 8) % 2)), b[jt][0], b[jt][1],
                        b[jt + 1][0], b[jt + 1][1]);
          }
        }
      };
      load_frags(af[0], bf[0], 0);
#pragma unroll
      for (int ks = 0; ks < BK / 32; ++ks) {  // k32 steps: Q4_K sub-block 4 h + ks of the super-block
        if (ks + 1 < BK / 32) load_frags(af[(ks + 1) % 2], bf[(ks + 1) % 2], ks + 1);
        const bool first = h == 0 && ks == 0;  // the super-block's first sub-block starts the integer sums
#pragma unroll
        for (int jt = 0; jt < NT; ++jt) {
          if constexpr (KIND == 0) {
            int32_t p[MT][4];  // this column tile's MMAs (MT of them in flight), then their scaling
#pragma unroll
            for (int i = 0; i < MT; ++i) mma_k32(p[i], af[ks % 2][i], bf[ks % 2][jt][0], bf[ks % 2][jt][1]);
            const int sc0 = (scw[jt][0] >> (8 * ks)) & 0xFF, sc1 = (scw[jt][1] >> (8 * ks)) & 0xFF;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              acc[i][jt][0] = (first ? 0 : acc[i][jt][0]) + sc0 * p[i][0];
              acc[i][jt][1] = (first ? 0 : acc[i][jt][1]) + sc1 * p[i][1];
              acc[i][jt][2] = (first ? 0 : acc[i][jt][2]) + sc0 * p[i][2];
              acc[i][jt][3] = (first ? 0 : acc[i][jt][3]) + sc1 * p[i][3];
            }
          } else {  // sub-blocks 2 ks and 2 ks + 1 of the stage: bytes 2 ks, 2 ks + 1 of the column's eight
            const int c0 = wn * WN + jt * 8 + 2 * t4;
            const uint32_t v0 = *reinterpret_cast<const uint16_t*>(st + C::OFF_SC + c0 * 8 + 2 * ks);
            const uint32_t v1 = *reinterpret_cast<const uint16_t*>(st + C::OFF_SC + (c0 + 1) * 8 + 2 * ks);
            const int lo0 = (int8_t)v0, hi0 = (int8_t)(v0 >> 8), lo1 = (int8_t)v1, hi1 = (int8_t)(v1 >> 8);
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              int32_t pl[4], ph[4];
              mma_k16(pl, af[ks % 2][i][0], af[ks % 2][i][1], bf[ks % 2][jt][0]);
              mma_k16(ph, af[ks % 2][i][2], af[ks % 2][i][3], bf[ks % 2][jt][1]);
              acc[i][jt][0] = (first ? 0 : acc[i][jt][0]) + lo0 * pl[0] + hi0 * ph[0];
              acc[i][jt][1] = (first ? 0 : acc[i][jt][1]) + lo1 * pl[1] + hi1 * ph[1];
              acc[i][jt][2] = (first ? 0 : acc[i][jt][2]) + lo0 * pl[2] + hi0 * ph[2];
              acc[i][jt][3] = (first ? 0 : acc[i][jt][3]) + lo1 * pl[3] + hi1 * ph[3];
            }
          }
        }
      }
      if (h == 1) {  // the super-block's floating-point step
        const float* sa = reinterpret_cast<const float*>(st + C::OFF_AS);
        const float* sd = reinterpret_cast<const float*>(st + C::OFF_D);
        const float* sdm = reinterpret_cast<const float*>(st + C::OFF_DM);
        // Q4_K's mins, sum over the eight sub-blocks of min x activation sum, as two m16n8k16 MMAs per tile: A = a
        // row's 16 split-sum bytes (eight S & 63, then eight S >> 6), B = a column's eight mins against the first
        // half (S & 63) or the second (S >> 6); mins = P_lo + 64 P_hi, exact
        uint32_t ma[KIND == 0 ? MT : 1][2], mlo[KIND == 0 ? NT : 1], mhi[KIND == 0 ? NT : 1];
        if constexpr (KIND == 0) {
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const int r0 = wm * WM + i * 16 + g;
            ma[i][0] = *reinterpret_cast<const uint32_t*>(st + C::OFF_XS + r0 * 16 + t4 * 4);
            ma[i][1] = *reinterpret_cast<const uint32_t*>(st + C::OFF_XS + (r0 + 8) * 16 + t4 * 4);
          }
#pragma unroll
          for (int jt = 0; jt < NT; ++jt) {
            const uint32_t w = *reinterpret_cast<const uint32_t*>(st + C::OFF_MN + (wn * WN + jt * 8 + g) * 8 +
                                                                  (t4 & 1) * 4);
            mlo[jt] = t4 < 2 ? w : 0u;
            mhi[jt] = t4 < 2 ? 0u : w;
          }
        }
#pragma unroll
        for (int jt = 0; jt < NT; ++jt) {
          const int c0 = wn * WN + jt * 8 + 2 * t4;
          const float d0 = sd[c0], d1 = sd[c0 + 1];
          const float ndm0 = KIND == 0 ? -sdm[c0] : 0.f, ndm1 = KIND == 0 ? -sdm[c0 + 1] : 0.f;
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const int r0 = wm * WM + i * 16 + g;
            const float a0 = sa[r0], a1 = sa[r0 + 8];
            int32_t plo[4] = {0, 0, 0, 0}, phi[4] = {0, 0, 0, 0};
            if constexpr (KIND == 0) {
              mma_k16(plo, ma[i][0], ma[i][1], mlo[jt]);
              mma_k16(phi, ma[i][0], ma[i][1], mhi[jt]);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float d = (e & 1) ? d1 : d0, a = (e & 2) ? a1 : a0;
              float t = __fmul_rn(d, (float)acc[i][jt][e]);
              if constexpr (KIND == 0) t = __fmaf_rn((e & 1) ? ndm1 : ndm0, (float)(plo[e] + 64 * phi[e]), t);
              y[i][jt][e] = __fmaf_rn(a, t, y[i][jt][e]);
            }
          }
        }
      }
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = m0 + wm * WM + i * 16 + g + half * 8;
      if (row >= live) continue;
#pragma unroll
      for (int jt = 0; jt < NT; ++jt) {
        const int col = n0 + wn * WN + jt * 8 + 2 * t4;
        if (col >= N) continue;
        const float v0 = y[i][jt][half * 2], v1 = y[i][jt][half * 2 + 1];
        if constexpr (EPI == 0) {
          __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)row * ldo + col;
          if (col + 1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(v0, v1);
          } else {
            o[0] = __float2bfloat16_rn(v0);
          }
        } else {
          float* o = reinterpret_cast<float*>(out) + (size_t)row * ldo + col;
          o[0] = __fadd_rn(o[0], v0);
          if (col + 1 < N) o[1] = __fadd_rn(o[1], v1);
        }
      }
    }
  }
}

// Programmatic dependent launch exists from sm_90 (Hopper) on: before it the launches run in order (pdl ignored).
inline bool sm90_or_later() {
  static int known[64] = {};  // per device: 0 not asked yet, 1 before sm_90, 2 sm_90 or later
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!known[dev]) {
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    known[dev] = major >= 9 ? 2 : 1;
  }
  return known[dev] == 2;
}

// Dynamic shared memory a block may opt in to: 99 KB on the RTX 5090 and Ada, 163 KB on the A100 / A800, 227 KB on
// Hopper (a fixed 99 KB kept the larger tiles' deeper pipelines off the other two).
inline int smem_optin() {
  static int known[64] = {};  // per device: 0 not asked yet
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!known[dev]) C10_CUDA_CHECK(cudaDeviceGetAttribute(&known[dev], cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  return known[dev];
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int KIND, int EPI, int MINB>
void launch(const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs, const torch::Tensor& wq,
            const torch::Tensor& wsc, const torch::Tensor& wmn, const torch::Tensor& wd, const torch::Tensor& wdm,
            torch::Tensor& out, const torch::Tensor& n_rows, int pdl) {
  using C = Cfg<BM, BN, WARPS_M, WARPS_N, STAGES, KIND>;
  const int M = a.size(0), K = a.size(1), N = wd.size(1);
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(wd.size(0) == K / 256 && wd.is_contiguous() && wsc.is_contiguous() && wsc.size(1) == N &&
                  as.size(0) == K / 256 && as.size(1) == M && as.is_contiguous(),
              "super-block-major scales: d [K / 256, N], sc [K / 256, N, *], a [K / 256, M]");
  TORCH_CHECK(C::SMEM <= smem_optin(), "this tile config's stages exceed the shared memory of a block");
  auto kern = w4a8_kernel<BM, BN, WARPS_M, WARPS_N, STAGES, KIND, EPI, MINB>;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    configured = true;
  }
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
  config.blockDim = dim3(C::THREADS);
  config.dynamicSmemBytes = C::SMEM;
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;  // may start while the previous kernel runs
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  if (!sm90_or_later()) pdl = 0;
  config.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &config, kern, a.data_ptr<int8_t>(), as.data_ptr<float>(), KIND == 0 ? xs.data_ptr<int8_t>() : nullptr,
      reinterpret_cast<const uint8_t*>(wq.data_ptr()), reinterpret_cast<const uint8_t*>(wsc.data_ptr()),
      KIND == 0 ? wmn.data_ptr<uint8_t>() : nullptr, wd.data_ptr<float>(), KIND == 0 ? wdm.data_ptr<float>() : nullptr,
      out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(), M, N, K, pdl));
}

// ------------------------------------------------------------------------------------------- K groups, for few rows
// A GEMM of a few rows (a decode step's) reads its weights at a fraction of DRAM's rate in the tile kernel: a
// 5,120-wide projection has 80 tiles of 64 columns, one CTA of two warps each, too few warps to hide the MMAs' latency
// (deeper pipelines did not help).  Here a CTA computes BM (16 or 32) rows x 32 columns with KW warps: warp w takes the
// super-blocks b = w, w + KW, w + 2 KW, ... through a cp.async pipeline of its own (STAGES deep, two stages per
// super-block, the tile kernel's stage layout and arithmetic for a BM x 32 warp tile).  The t = d * acc - dmin * mins
// of a round (KW consecutive super-blocks) go to shared memory, and after one barrier every thread folds its elements'
// t in increasing b, y = fma(a, t, y), as the tile kernel does: the same bits, only the integer work is spread.
constexpr int KG_BN = 32;

template <int BM, int KW, int STAGES, int KIND>
struct KgCfg {
  using C = Cfg<BM, KG_BN, 1, 1, STAGES, KIND>;
  static constexpr int OFF_T = KW * STAGES * C::STAGE;          // [2][KW][BM][BN] fp32: the rounds' t
  static constexpr int OFF_A = OFF_T + 2 * KW * BM * KG_BN * 4;  // [2][KW][BM] fp32: their activation scales
  static constexpr int SMEM = OFF_A + 2 * KW * BM * 4;
};

template <int BM, int KW, int STAGES, int KIND, int EPI>
__global__ void __launch_bounds__(KW * 32, 1)
    w4a8_kgroups_kernel(const int8_t* __restrict__ A, const float* __restrict__ as, const int8_t* __restrict__ xs,
                        const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ wsc,
                        const uint8_t* __restrict__ wmn, const float* __restrict__ wd,
                        const float* __restrict__ wdm, void* __restrict__ out, int ldo,
                        const int* __restrict__ n_rows, int M, int N, int K) {
  constexpr int BN = KG_BN, MT = BM / 16, NT = BN / 8, THREADS = KW * 32, PER = BM * BN / THREADS;
  using KC = KgCfg<BM, KW, STAGES, KIND>;
  using C = typename KC::C;
  static_assert(BM == 16 || BM == 32, "a lane per row for the rows' scales");
  static_assert(BM * BN % THREADS == 0, "every thread folds the same number of elements");
  const int live = *n_rows;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  if (m0 >= live) return;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane / 4, t4 = lane % 4;
  const int NB = K / 256, rounds = (NB + KW - 1) / KW;
  uint8_t* wst = smem + warp * STAGES * C::STAGE;                // this warp's stages
  float* tsl = reinterpret_cast<float*>(smem + KC::OFF_T);
  float* asl = reinterpret_cast<float*>(smem + KC::OFF_A);
  const int nb_w = warp < NB ? (NB - warp + KW - 1) / KW : 0;   // this warp's super-blocks, two stages each

  auto load_stage = [&](int j) {  // this warp's stage j: half j % 2 of super-block warp + (j / 2) KW
    uint8_t* st = wst + (j % STAGES) * C::STAGE;
    const int b = warp + (j / 2) * KW, kt = 2 * b + j % 2, k0 = kt * BK;
    const bool last = kt % 2 == 1;
#pragma unroll
    for (int i = lane; i < BM * (BK / 16); i += 32) {
      const int r = i / (BK / 16), c = i % (BK / 16), gm = m0 + r;
      cp_async16(smem_u32(st + swz<128>(r, c)), A + (size_t)min(gm, M - 1) * K + k0 + c * 16, gm < M);
    }
#pragma unroll
    for (int i = lane; i < BN * (C::WROW / 16); i += 32) {
      const int r = i / (C::WROW / 16), c = i % (C::WROW / 16), gn = n0 + r;
      cp_async16(smem_u32(st + C::OFF_W + swz<C::WROW>(r, c)),
                 Wq + (size_t)min(gn, N - 1) * (K / (KIND == 0 ? 2 : 1)) + k0 / (KIND == 0 ? 2 : 1) + c * 16,
                 gn < N);
    }
    {  // a column per lane: its sub-block scales; with the second stage d (and Q4_K's dmin and mins)
      const size_t gn = (size_t)b * N + min(n0 + lane, N - 1);
      const bool ok = n0 + lane < N;
      if constexpr (KIND == 0) {
        cp_async4(smem_u32(st + C::OFF_SC + lane * 4), wsc + gn * 8 + (kt % 2) * 4, ok);
        if (last) {
          cp_async8(smem_u32(st + C::OFF_MN + lane * 8), wmn + gn * 8, ok);
          cp_async4(smem_u32(st + C::OFF_DM + lane * 4), wdm + gn, ok);
        }
      } else {
        cp_async8(smem_u32(st + C::OFF_SC + lane * 8), wsc + gn * 16 + (kt % 2) * 8, ok);
      }
      if (last) cp_async4(smem_u32(st + C::OFF_D + lane * 4), wd + gn, ok);
    }
    if (last && lane < BM) {  // a row per lane: its activation scale (and Q4_K's split sums)
      const size_t gm = (size_t)b * M + min(m0 + lane, M - 1);
      const bool ok = m0 + lane < M;
      if constexpr (KIND == 0) cp_async16(smem_u32(st + C::OFF_XS + lane * 16), xs + gm * 16, ok);
      cp_async4(smem_u32(st + C::OFF_AS + lane * 4), as + gm, ok);
    }
  };

  float y[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) y[i] = 0.f;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < 2 * nb_w) load_stage(s);
    cp_async_commit();
  }
  int32_t acc[MT][NT][4];
  for (int r = 0; r < rounds; ++r) {
    float* ts = tsl + ((r % 2) * KW + warp) * BM * BN;
    if (r < nb_w) {  // this warp's super-block warp + r KW
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int j = 2 * r + h;
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        if (j + STAGES - 1 < 2 * nb_w) load_stage(j + STAGES - 1);
        cp_async_commit();
        const uint8_t* st = wst + (j % STAGES) * C::STAGE;
        uint32_t scw[KIND == 0 ? NT : 1][2];
        if constexpr (KIND == 0) {
#pragma unroll
          for (int jt = 0; jt < NT; ++jt)
#pragma unroll
            for (int c = 0; c < 2; ++c)
              scw[jt][c] = *reinterpret_cast<const uint32_t*>(st + C::OFF_SC + (jt * 8 + 2 * t4 + c) * 4);
        }
        uint32_t af[2][MT][4], bf[2][NT][2];
        auto load_frags = [&](uint32_t (&a)[MT][4], uint32_t (&b)[NT][2], int ks) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
            ldmatrix_x4(smem_u32(st + swz<128>(i * 16 + lane % 16, ks * 2 + lane / 16)), a[i][0], a[i][1], a[i][2],
                        a[i][3]);
          if constexpr (KIND == 0) {
            uint32_t w0, w1, w2, w3;
            ldmatrix_x4(smem_u32(st + C::OFF_W + swz<64>(lane, ks)), w0, w1, w2, w3);
            b[0][0] = w0 & 0x0F0F0F0Fu, b[0][1] = (w0 >> 4) & 0x0F0F0F0Fu;
            b[1][0] = w1 & 0x0F0F0F0Fu, b[1][1] = (w1 >> 4) & 0x0F0F0F0Fu;
            b[2][0] = w2 & 0x0F0F0F0Fu, b[2][1] = (w2 >> 4) & 0x0F0F0F0Fu;
            b[3][0] = w3 & 0x0F0F0F0Fu, b[3][1] = (w3 >> 4) & 0x0F0F0F0Fu;
          } else {
#pragma unroll
            for (int jt = 0; jt < NT; jt += 2) {
              const int rr = jt * 8 + (lane % 8) + (lane / 16) * 8;
              ldmatrix_x4(smem_u32(st + C::OFF_W + swz<128>(rr, ks * 2 + (lane / 8) % 2)), b[jt][0], b[jt][1],
                          b[jt + 1][0], b[jt + 1][1]);
            }
          }
        };
        load_frags(af[0], bf[0], 0);
#pragma unroll
        for (int ks = 0; ks < BK / 32; ++ks) {  // k32 steps, as the tile kernel's
          if (ks + 1 < BK / 32) load_frags(af[(ks + 1) % 2], bf[(ks + 1) % 2], ks + 1);
          const bool first = h == 0 && ks == 0;
#pragma unroll
          for (int jt = 0; jt < NT; ++jt) {
            if constexpr (KIND == 0) {
              int32_t p[MT][4];
#pragma unroll
              for (int i = 0; i < MT; ++i) mma_k32(p[i], af[ks % 2][i], bf[ks % 2][jt][0], bf[ks % 2][jt][1]);
              const int sc0 = (scw[jt][0] >> (8 * ks)) & 0xFF, sc1 = (scw[jt][1] >> (8 * ks)) & 0xFF;
#pragma unroll
              for (int i = 0; i < MT; ++i) {
                acc[i][jt][0] = (first ? 0 : acc[i][jt][0]) + sc0 * p[i][0];
                acc[i][jt][1] = (first ? 0 : acc[i][jt][1]) + sc1 * p[i][1];
                acc[i][jt][2] = (first ? 0 : acc[i][jt][2]) + sc0 * p[i][2];
                acc[i][jt][3] = (first ? 0 : acc[i][jt][3]) + sc1 * p[i][3];
              }
            } else {
              const int c0 = jt * 8 + 2 * t4;
              const uint32_t v0 = *reinterpret_cast<const uint16_t*>(st + C::OFF_SC + c0 * 8 + 2 * ks);
              const uint32_t v1 = *reinterpret_cast<const uint16_t*>(st + C::OFF_SC + (c0 + 1) * 8 + 2 * ks);
              const int lo0 = (int8_t)v0, hi0 = (int8_t)(v0 >> 8), lo1 = (int8_t)v1, hi1 = (int8_t)(v1 >> 8);
#pragma unroll
              for (int i = 0; i < MT; ++i) {
                int32_t pl[4], ph[4];
                mma_k16(pl, af[ks % 2][i][0], af[ks % 2][i][1], bf[ks % 2][jt][0]);
                mma_k16(ph, af[ks % 2][i][2], af[ks % 2][i][3], bf[ks % 2][jt][1]);
                acc[i][jt][0] = (first ? 0 : acc[i][jt][0]) + lo0 * pl[0] + hi0 * ph[0];
                acc[i][jt][1] = (first ? 0 : acc[i][jt][1]) + lo1 * pl[1] + hi1 * ph[1];
                acc[i][jt][2] = (first ? 0 : acc[i][jt][2]) + lo0 * pl[2] + hi0 * ph[2];
                acc[i][jt][3] = (first ? 0 : acc[i][jt][3]) + lo1 * pl[3] + hi1 * ph[3];
              }
            }
          }
        }
        if (h == 1) {  // the super-block's t (the tile kernel's floating-point step up to its last fma), and a
          const float* sd = reinterpret_cast<const float*>(st + C::OFF_D);
          const float* sdm = reinterpret_cast<const float*>(st + C::OFF_DM);
          uint32_t ma[MT][2], mlo[NT], mhi[NT];
          if constexpr (KIND == 0) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              ma[i][0] = *reinterpret_cast<const uint32_t*>(st + C::OFF_XS + (i * 16 + g) * 16 + t4 * 4);
              ma[i][1] = *reinterpret_cast<const uint32_t*>(st + C::OFF_XS + (i * 16 + g + 8) * 16 + t4 * 4);
            }
#pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
              const uint32_t w = *reinterpret_cast<const uint32_t*>(st + C::OFF_MN + (jt * 8 + g) * 8 + (t4 & 1) * 4);
              mlo[jt] = t4 < 2 ? w : 0u;
              mhi[jt] = t4 < 2 ? 0u : w;
            }
          }
#pragma unroll
          for (int jt = 0; jt < NT; ++jt) {
            const int c0 = jt * 8 + 2 * t4;
            const float d0 = sd[c0], d1 = sd[c0 + 1];
            const float ndm0 = KIND == 0 ? -sdm[c0] : 0.f, ndm1 = KIND == 0 ? -sdm[c0 + 1] : 0.f;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              int32_t plo[4] = {0, 0, 0, 0}, phi[4] = {0, 0, 0, 0};
              if constexpr (KIND == 0) {
                mma_k16(plo, ma[i][0], ma[i][1], mlo[jt]);
                mma_k16(phi, ma[i][0], ma[i][1], mhi[jt]);
              }
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                float t = __fmul_rn((e & 1) ? d1 : d0, (float)acc[i][jt][e]);
                if constexpr (KIND == 0) t = __fmaf_rn((e & 1) ? ndm1 : ndm0, (float)(plo[e] + 64 * phi[e]), t);
                ts[(i * 16 + g + (e & 2) * 4) * BN + c0 + (e & 1)] = t;
              }
            }
          }
          if (lane < BM) asl[((r % 2) * KW + warp) * BM + lane] = reinterpret_cast<const float*>(st + C::OFF_AS)[lane];
        }
      }
    }
    // one barrier a round: the t slots alternate, so a warp writes round r + 2's only after every thread has passed
    // round r + 1's barrier, i.e. folded round r
    __syncthreads();
    const float* tr = tsl + (r % 2) * KW * BM * BN;
    const float* ar = asl + (r % 2) * KW * BM;
    const int nw = min(KW, NB - r * KW);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int e = tid + i * THREADS;
      for (int w = 0; w < nw; ++w) y[i] = __fmaf_rn(ar[w * BM + e / BN], tr[w * BM * BN + e], y[i]);
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int e = tid + i * THREADS, row = m0 + e / BN, col = n0 + e % BN;
    if (row >= live || col >= N) continue;
    if constexpr (EPI == 0) {
      reinterpret_cast<__nv_bfloat16*>(out)[(size_t)row * ldo + col] = __float2bfloat16_rn(y[i]);
    } else {
      float* o = reinterpret_cast<float*>(out) + (size_t)row * ldo + col;
      *o = __fadd_rn(*o, y[i]);
    }
  }
}

template <int BM, int KW, int STAGES, int KIND, int EPI>
void launch_kgroups(const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs, const torch::Tensor& wq,
                    const torch::Tensor& wsc, const torch::Tensor& wmn, const torch::Tensor& wd,
                    const torch::Tensor& wdm, torch::Tensor& out, const torch::Tensor& n_rows) {
  using KC = KgCfg<BM, KW, STAGES, KIND>;
  const int M = a.size(0), K = a.size(1), N = wd.size(1);
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(wd.size(0) == K / 256 && wd.is_contiguous() && wsc.is_contiguous() && wsc.size(1) == N &&
                  as.size(0) == K / 256 && as.size(1) == M && as.is_contiguous(),
              "super-block-major scales: d [K / 256, N], sc [K / 256, N, *], a [K / 256, M]");
  int dev = 0, optin = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(KC::SMEM <= optin, "this K-group config's stages exceed the shared memory of a block");
  auto kern = w4a8_kgroups_kernel<BM, KW, STAGES, KIND, EPI>;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, KC::SMEM));
    configured = true;
  }
  kern<<<dim3((N + KG_BN - 1) / KG_BN, (M + BM - 1) / BM), KW * 32, KC::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<int8_t>(), as.data_ptr<float>(), KIND == 0 ? xs.data_ptr<int8_t>() : nullptr,
      reinterpret_cast<const uint8_t*>(wq.data_ptr()), reinterpret_cast<const uint8_t*>(wsc.data_ptr()),
      KIND == 0 ? wmn.data_ptr<uint8_t>() : nullptr, wd.data_ptr<float>(), KIND == 0 ? wdm.data_ptr<float>() : nullptr,
      out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BM, int KW, int KIND, int EPI>
void kgroups_stages(int stages, const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs,
                    const torch::Tensor& wq, const torch::Tensor& wsc, const torch::Tensor& mn, const torch::Tensor& wd,
                    const torch::Tensor& dm, torch::Tensor& out, const torch::Tensor& n_rows) {
  switch (stages) {
    case 2: return launch_kgroups<BM, KW, 2, KIND, EPI>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
    case 3: return launch_kgroups<BM, KW, 3, KIND, EPI>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
    case 4: return launch_kgroups<BM, KW, 4, KIND, EPI>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
  }
  TORCH_CHECK(false, "w4a8: K-group configs take 2, 3 or 4 stages");
}

// ------------------------------------------------------------------------------------------------ GEMV, 1-8 rows
// The K-group kernel's four warps share a 32-column tile: a 5,120-wide projection is 160 CTAs, under five warps per SM
// of an H200, which stream its weights at 1.2-2 TB/s of 4.8.  Here a warp has columns of its own (COLS) for all of a
// decode step's rows (up to MR): at step i lane l reads piece p = 32 i + l of each column - 16 bytes, a Q4_K sub-block
// (32 values) or a Q6_K one (16) - and dp4a sums it against the rows' activations; the lanes of a super-block (8 or 16)
// add their integer sums with shuffles and compute its t (the tile kernel's steps, one rounding each), and the warp
// folds the step's super-blocks in increasing b, y = fma(a, t, y): the same bits.
constexpr int GV_WARPS = 4;   // warps per CTA
constexpr int GV_PIECES = 8;  // 16-byte weight loads in flight per lane (over its columns)

__device__ __forceinline__ uint4 ld_stream16(const void* p) {  // weights: read once, past L1
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <int KIND, int EPI, int MR, int COLS>
__global__ void __launch_bounds__(GV_WARPS * 32)
    w4a8_gemv_kernel(const int8_t* __restrict__ A, const float* __restrict__ as, const int8_t* __restrict__ xs,
                     const uint8_t* __restrict__ Wq, const uint8_t* __restrict__ wsc,
                     const uint8_t* __restrict__ wmn, const float* __restrict__ wd, const float* __restrict__ wdm,
                     void* __restrict__ out, int ldo, const int* __restrict__ n_rows, int M, int N, int K) {
  constexpr int SUBS = KIND == 0 ? 8 : 16;          // pieces (sub-blocks) per super-block
  constexpr int ITER = GV_PIECES / COLS > 1 ? GV_PIECES / COLS : 1;
  constexpr unsigned FULL = 0xffffffffu;
  const int live = min(*n_rows, MR);
  const int lane = threadIdx.x % 32;
  const int n0 = (blockIdx.x * GV_WARPS + (int)threadIdx.x / 32) * COLS;
  if (n0 >= N || live <= 0) return;
  const int NB = K / 256, pieces = NB * SUBS;
  const size_t wrow = KIND == 0 ? (size_t)K / 2 : (size_t)K;  // weight bytes of a column
  float y[COLS][MR];
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < MR; ++m) y[c][m] = 0.f;
  for (int p0 = 0; p0 < pieces; p0 += 32 * ITER) {
    uint4 w[ITER][COLS];           // the pieces, and with them their scales: sc | m << 8 (Q4_K), sc (Q6_K), d, dmin
    uint32_t sm[ITER][COLS];
    float dd[ITER][COLS], ddm[ITER][COLS];
#pragma unroll
    for (int i = 0; i < ITER; ++i) {
      const int p = p0 + 32 * i + lane, b = (p < pieces ? p : 0) / SUBS, g = p % SUBS;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const bool ok = p < pieces && n0 + c < N;
        const size_t sb = ((size_t)b * N + n0 + c) * SUBS + g, db = (size_t)b * N + n0 + c;
        w[i][c] = ok ? ld_stream16(Wq + (size_t)(n0 + c) * wrow + (size_t)p * 16) : make_uint4(0u, 0u, 0u, 0u);
        sm[i][c] = ok ? (KIND == 0 ? (uint32_t)__ldg(wsc + sb) | (uint32_t)__ldg(wmn + sb) << 8 : (uint32_t)__ldg(wsc + sb))
                      : 0u;
        dd[i][c] = ok ? __ldg(wd + db) : 0.f;
        ddm[i][c] = ok && KIND == 0 ? __ldg(wdm + db) : 0.f;
      }
    }
#pragma unroll
    for (int i = 0; i < ITER; ++i) {
      const int pb = p0 + 32 * i;
      if (pb >= pieces) break;                      // the same for the whole warp
      const int p = pb + lane;
      const bool on = p < pieces;
      const int b = (on ? p : pb) / SUBS, g = p % SUBS;
      float t[COLS][MR], av[MR];
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        av[m] = m < live ? as[(size_t)b * M + m] : 0.f;
        int xa[4] = {0, 0, 0, 0}, xb[4] = {0, 0, 0, 0}, S = 0;
        if (on && m < live) {
          const int8_t* x = A + (size_t)m * K + (size_t)p * (KIND == 0 ? 32 : 16);
          const uint4 u = __ldg(reinterpret_cast<const uint4*>(x));
          xa[0] = (int)u.x, xa[1] = (int)u.y, xa[2] = (int)u.z, xa[3] = (int)u.w;
          if constexpr (KIND == 0) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(x + 16));
            xb[0] = (int)v.x, xb[1] = (int)v.y, xb[2] = (int)v.z, xb[3] = (int)v.w;
            const int8_t* sp = xs + ((size_t)b * M + m) * 16;
            S = (int)sp[g] + 64 * (int)sp[8 + g];   // the sub-block's activation sum (quantize_ref's split)
          }
        }
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
          int acc = 0, mins = 0;
          if (on && m < live && n0 + c < N) {
            const uint32_t wv[4] = {w[i][c].x, w[i][c].y, w[i][c].z, w[i][c].w};
            int s = 0;
            if constexpr (KIND == 0) {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                s = __dp4a((int)(wv[j] & 0x0F0F0F0Fu), xa[j], s);
                s = __dp4a((int)((wv[j] >> 4) & 0x0F0F0F0Fu), xb[j], s);
              }
              acc = (int)(sm[i][c] & 0xFFu) * s;
              mins = (int)(sm[i][c] >> 8) * S;
            } else {
#pragma unroll
              for (int j = 0; j < 4; ++j) s = __dp4a((int)wv[j], xa[j], s);
              acc = (int)(int8_t)(sm[i][c] & 0xFFu) * s;
            }
          }
#pragma unroll
          for (int o = 1; o < SUBS; o <<= 1) {      // the super-block's lanes: exact integer sums
            acc += __shfl_xor_sync(FULL, acc, o);
            if constexpr (KIND == 0) mins += __shfl_xor_sync(FULL, mins, o);
          }
          float tv = __fmul_rn(dd[i][c], (float)acc);
          if constexpr (KIND == 0) tv = __fmaf_rn(-ddm[i][c], (float)mins, tv);
          t[c][m] = tv;
        }
      }
#pragma unroll
      for (int s = 0; s < 32 / SUBS; ++s) {         // the step's super-blocks, in increasing b
        if (pb + s * SUBS >= pieces) break;
#pragma unroll
        for (int m = 0; m < MR; ++m) {
          const float a_ = __shfl_sync(FULL, av[m], s * SUBS);
#pragma unroll
          for (int c = 0; c < COLS; ++c) y[c][m] = __fmaf_rn(a_, __shfl_sync(FULL, t[c][m], s * SUBS), y[c][m]);
        }
      }
    }
  }
#pragma unroll
  for (int c = 0; c < COLS; ++c)
#pragma unroll
    for (int m = 0; m < MR; ++m)
      if (lane == c * MR + m && m < live && n0 + c < N) {
        if constexpr (EPI == 0) {
          reinterpret_cast<__nv_bfloat16*>(out)[(size_t)m * ldo + n0 + c] = __float2bfloat16_rn(y[c][m]);
        } else {
          float* o = reinterpret_cast<float*>(out) + (size_t)m * ldo + n0 + c;
          *o = __fadd_rn(*o, y[c][m]);
        }
      }
}

template <int KIND, int EPI, int MR, int COLS>
void launch_gemv(const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs, const torch::Tensor& wq,
                 const torch::Tensor& wsc, const torch::Tensor& wmn, const torch::Tensor& wd, const torch::Tensor& wdm,
                 torch::Tensor& out, const torch::Tensor& n_rows) {
  const int M = a.size(0), K = a.size(1), N = wd.size(1);
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
  TORCH_CHECK(wd.size(0) == K / 256 && wd.is_contiguous() && wsc.is_contiguous() && wsc.size(1) == N &&
                  as.size(0) == K / 256 && as.size(1) == M && as.is_contiguous(),
              "super-block-major scales: d [K / 256, N], sc [K / 256, N, *], a [K / 256, M]");
  const int per_cta = GV_WARPS * COLS;
  w4a8_gemv_kernel<KIND, EPI, MR, COLS><<<(N + per_cta - 1) / per_cta, GV_WARPS * 32, 0,
                                          c10::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<int8_t>(), as.data_ptr<float>(), KIND == 0 ? xs.data_ptr<int8_t>() : nullptr,
      reinterpret_cast<const uint8_t*>(wq.data_ptr()), reinterpret_cast<const uint8_t*>(wsc.data_ptr()),
      KIND == 0 ? wmn.data_ptr<uint8_t>() : nullptr, wd.data_ptr<float>(), KIND == 0 ? wdm.data_ptr<float>() : nullptr,
      out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int KIND, int EPI, int COLS>
void gemv_rows(const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs, const torch::Tensor& wq,
               const torch::Tensor& wsc, const torch::Tensor& mn, const torch::Tensor& wd, const torch::Tensor& dm,
               torch::Tensor& out, const torch::Tensor& n_rows) {
  const int M = a.size(0);
  if (M <= 1) return launch_gemv<KIND, EPI, 1, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
  if (M <= 2) return launch_gemv<KIND, EPI, 2, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
  if (M <= 4) return launch_gemv<KIND, EPI, 4, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
  TORCH_CHECK(M <= 8, "w4a8: the GEMV configs take up to 8 rows");
  return launch_gemv<KIND, EPI, 8, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
}

}  // namespace

// cfg: (BM, BN, warps M x N, minimum CTAs per SM - the register cap: 2 holds an 8-warp CTA to 128 registers a thread,
// a 4-warp one to 255)  0 = 16x64 (1x2), 1 = 32x64 (1x2), 2 = 64x64 (2x2), 3 = 64x128 (2x4), 4 = 128x64 (4x2),
// 5 = 128x128 (4x4), 6 = 16x32 (1x1), 7 = 32x128 (1x4); 64x32 and 32x64 warp tiles: 8 = 128x128 (2x4), 9 = 128x128
// (4x2), 10 = 64x128 (1x4), 11 = 128x64 (2x2), 12 = 64x64 (1x2); K groups (w4a8_kgroups_kernel, few rows), the
// warps on the super-blocks: 13 = 16x32 with 8 warps, 14 = 16x32 with 4, 15 = 32x32 with 8, 16 = 32x32 with 4
#define W4A8_CONFIGS(X)  \
  X(0, 16, 64, 1, 2, 2)    \
  X(1, 32, 64, 1, 2, 2)    \
  X(2, 64, 64, 2, 2, 2)    \
  X(3, 64, 128, 2, 4, 2)   \
  X(4, 128, 64, 4, 2, 2)   \
  X(5, 128, 128, 4, 4, 1)  \
  X(6, 16, 32, 1, 1, 2)    \
  X(7, 32, 128, 1, 4, 2)   \
  X(8, 128, 128, 2, 4, 1)  \
  X(9, 128, 128, 4, 2, 1)  \
  X(10, 64, 128, 1, 4, 2)  \
  X(11, 128, 64, 2, 2, 2)  \
  X(12, 64, 64, 1, 2, 2)

void w4a8_gemm(torch::Tensor a, torch::Tensor as, torch::Tensor xs, torch::Tensor wq, torch::Tensor wsc,
               std::optional<torch::Tensor> wmn, torch::Tensor wd, std::optional<torch::Tensor> wdm, torch::Tensor out,
               torch::Tensor n_rows, int64_t kind, int64_t epi, int64_t cfg, int64_t stages, int64_t pdl) {
  TORCH_CHECK(a.dtype() == torch::kInt8 && a.stride(1) == 1 && a.stride(0) == a.size(1), "contiguous int8 A");
  TORCH_CHECK(epi == 0 ? out.dtype() == torch::kBFloat16 : out.dtype() == torch::kFloat32, "output dtype");
  TORCH_CHECK(kind == 1 || (wmn.has_value() && wdm.has_value()), "Q4_K needs mins and dmin");
  TORCH_CHECK(kind == 1 || (xs.dtype() == torch::kInt8 && xs.numel() == a.numel() / 16 && xs.is_contiguous() &&
                            wmn->is_contiguous() && wdm->is_contiguous()),
              "int8 split activation sums [K / 256, M, 16]");
  const torch::Tensor mn = wmn.has_value() ? *wmn : wsc, dm = wdm.has_value() ? *wdm : wd;
#define W4A8_STAGES(KIND, EPI, BM, BN, WMN, WNN, MB)                                                                  \
  if (stages == 2) launch<BM, BN, WMN, WNN, 2, KIND, EPI, MB>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows, (int)pdl); \
  else if (stages == 3) launch<BM, BN, WMN, WNN, 3, KIND, EPI, MB>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows,       \
                                                                   (int)pdl);                                       \
  else launch<BM, BN, WMN, WNN, 4, KIND, EPI, MB>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows, (int)pdl);
#define W4A8_CASE(C, BM, BN, WMN, WNN, MB)                     \
  case C:                                                      \
    if (kind == 0) {                                           \
      if (epi == 0) { W4A8_STAGES(0, 0, BM, BN, WMN, WNN, MB) } \
      else { W4A8_STAGES(0, 1, BM, BN, WMN, WNN, MB) }          \
    } else {                                                   \
      if (epi == 0) { W4A8_STAGES(1, 0, BM, BN, WMN, WNN, MB) } \
      else { W4A8_STAGES(1, 1, BM, BN, WMN, WNN, MB) }          \
    }                                                          \
    break;
  if (cfg >= 17 && cfg <= 19) {  // GEMV (up to 8 rows): a warp's own 1, 2 or 4 columns
    TORCH_CHECK(pdl == 0, "w4a8: no programmatic dependent launch for the GEMV configs");
#define W4A8_GV(COLS)                                                                              \
  if (kind == 0) {                                                                                 \
    if (epi == 0) gemv_rows<0, 0, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);              \
    else gemv_rows<0, 1, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);                       \
  } else {                                                                                         \
    if (epi == 0) gemv_rows<1, 0, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);              \
    else gemv_rows<1, 1, COLS>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);                       \
  }
    if (cfg == 17) {
      W4A8_GV(1)
    } else if (cfg == 18) {
      W4A8_GV(2)
    } else {
      W4A8_GV(4)
    }
#undef W4A8_GV
    return;
  }
  if (cfg >= 13 && cfg <= 16) {  // K groups (few rows): 8 or 4 warps on the super-blocks of a 16 or 32 x 32 tile
    TORCH_CHECK(pdl == 0, "w4a8: no programmatic dependent launch for the K-group configs");
#define W4A8_KG(BM, KW)                                                                                   \
  if (kind == 0) {                                                                                        \
    if (epi == 0) kgroups_stages<BM, KW, 0, 0>((int)stages, a, as, xs, wq, wsc, mn, wd, dm, out, n_rows); \
    else kgroups_stages<BM, KW, 0, 1>((int)stages, a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);          \
  } else {                                                                                                \
    if (epi == 0) kgroups_stages<BM, KW, 1, 0>((int)stages, a, as, xs, wq, wsc, mn, wd, dm, out, n_rows); \
    else kgroups_stages<BM, KW, 1, 1>((int)stages, a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);          \
  }
    if (cfg == 13) {
      W4A8_KG(16, 8)
    } else if (cfg == 14) {
      W4A8_KG(16, 4)
    } else if (cfg == 15) {
      W4A8_KG(32, 8)
    } else {
      W4A8_KG(32, 4)
    }
#undef W4A8_KG
    return;
  }
  switch (cfg) {
    W4A8_CONFIGS(W4A8_CASE)
    default:
      TORCH_CHECK(false, "unknown w4a8 cfg");
  }
#undef W4A8_CASE
#undef W4A8_STAGES
}

TORCH_LIBRARY_FRAGMENT(startlux_prism, m) {
  m.def("w4a8_gemm(Tensor a, Tensor a_s, Tensor xs, Tensor wq, Tensor wsc, Tensor? wmn, Tensor wd, Tensor? wdm, "
        "Tensor(a!) out, Tensor n_rows, int kind, int epi, int cfg, int stages, int pdl=0) -> ()");
}
TORCH_LIBRARY_IMPL(startlux_prism, CUDA, m) {
  m.impl("w4a8_gemm", &w4a8_gemm);
}
