// W8A8 GEMM on Hopper's warpgroup MMA (sm_90a): int8 x int8 -> int32 with wgmma.mma_async from shared memory, and the
// fused scale epilogue of w8a8_gemm.cu - the same bits.
//
//   C[m, n] = (sum_k A[m, k] W[n, k]) * sa[m] * sw[n]        A [M, K], W [N, K] int8, K-contiguous ("TN")
//   EPI 0 writes bf16 out[m * ldo + n]; EPI 1 adds into fp32 out[m * ldo + n] (the residual stream).
//
// On sm_90 the mma.sync instructions of w8a8_gemm.cu reach about half of the tensor cores' int8 rate (~510 TOPS on an
// H200 at 1,000-8,000 rows, where cuBLASLt's wgmma kernels reach 770-920); wgmma issues a 64 x N x 32 product per
// warpgroup from operands in shared memory, without the ldmatrix traffic through registers.
//
// A CTA computes a BM x BN tile, 64 rows (all BN columns: BN / 2 int32 accumulators per thread) per MMA warpgroup.  K
// advances 128 bytes per stage, in the 128-byte swizzle that wgmma's descriptors read (16-byte chunk c of row r at
// (c ^ (r % 8)) * 16, 8-row groups 1024 bytes apart - the layout w8a8_gemm.cu's BK = 128 tiles already use), STAGES
// deep.  Two pipelines:
//   w8a8_wgmma_kernel      every thread issues cp.async 16-byte copies; one wgmma group stays in flight while the next
//                          stage's copies are issued, so the copies run STAGES - 2 stages ahead (a stage is refilled
//                          only after every warpgroup's MMAs on it completed; a CTA barrier per stage)
//   w8a8_wgmma_tma_kernel  warp-specialized: a producer warpgroup (one thread) loads each stage with two TMA copies
//                          (cp.async.bulk.tensor, the tensor maps' 128-byte swizzle) that complete on the stage's
//                          "full" mbarrier; the MMA warpgroups wait on it, and release the stage on its "empty"
//                          mbarrier once their MMAs on it completed - STAGES - 1 stages in flight, no CTA barriers
// Tiles are visited in groups of GROUP_M row tiles (each group sweeps the column tiles), so a group's rows and the
// weight panels stay in L2.  Rows at or beyond *n_rows (the live rows of a padded bucket) are skipped by whole tiles.
//
// Batch invariance: integer sums are exact in any order; the epilogue rounds as w8a8_gemm.cu and the Triton kernel do
// (v = (float(acc) * sa) * sw, each product rounded; the residual add rounded once), so every kernel, tile and split
// gives an element the same bits.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <tuple>
#include <map>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define FLASHJEV_WGMMA_DEVICE 1
#endif

namespace {

constexpr int WG_BK = 128;          // bytes of K per stage (one 128-byte swizzle row)
constexpr int WG_GROUP_M = 8;       // row tiles per raster group
constexpr int WG_SMEM_MAX = 232448; // dynamic shared memory per block on sm_90 (227 KB)

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

#ifdef FLASHJEV_WGMMA_DEVICE
// generic-proxy writes (cp.async) to shared memory, made visible to the async proxy that wgmma reads through
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// keeps the compiler from moving accumulator reads or writes across the asynchronous MMAs
template <int R>
__device__ __forceinline__ void fence_operands(int32_t (&d)[R]) {
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+r"(d[i])::"memory");
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {  // until the phase with this parity completed
  asm volatile(  // labels are local to the { } scope, so every inlined copy may use the same names
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra.uni DONE;\n"
      "bra.uni LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
// box (128 bytes of K, rows) at (k, row) of a 2-D tensor map -> shared memory; completes on `bar` (transaction bytes)
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int k, int row, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(k), "r"(row), "r"(smem_u32(bar))
      : "memory");
}
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R)); }
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R)); }
#endif

// shared-memory matrix descriptor: K-major operand, 128-byte rows in the 128-byte swizzle, 8-row groups 1024 bytes
// apart (stride byte offset); the leading byte offset is unused for swizzled K-major layouts.  Advancing the start
// address by 32 bytes selects the next k32 slice (the swizzle is applied to the computed addresses).
__device__ __forceinline__ uint64_t sw128_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

// byte offset of 16-byte chunk `c` of row `r` in a tile of 128-byte rows (the 128-byte swizzle)
__device__ __forceinline__ int swz128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

template <int BN>
struct Wgmma;

#ifdef FLASHJEV_WGMMA_DEVICE
// d[64 x BN] += A (64 x 32, desc a) * B (BN x 32, desc b)^T; per thread BN / 2 accumulators, laid out per warp as
// mma.sync's m16n8 accumulators: warp w of the warpgroup holds rows 16w + lane / 4 (+ 8), columns 8j + 2 (lane % 4)
// (+ 1) in d[4j .. 4j + 3]
template <>
struct Wgmma<64> {
  static __device__ __forceinline__ void mma(int32_t (&d)[32], uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, "
        "%22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
          "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
          "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
          "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(da), "l"(db), "r"(1));
  }
};

template <>
struct Wgmma<128> {
  static __device__ __forceinline__ void mma(int32_t (&d)[64], uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, "
        "%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(da), "l"(db), "r"(1));
  }
};

template <>
struct Wgmma<256> {
  static __device__ __forceinline__ void mma(int32_t (&d)[128], uint64_t da, uint64_t db) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, "
        "%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, "
        "%67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, "
        "%89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, "
        "%109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, "
        "%127}, "
        "%128, %129, p;\n"
        "}\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
          "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]),
          "+r"(d[64]), "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]),
          "+r"(d[72]), "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]),
          "+r"(d[80]), "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]),
          "+r"(d[88]), "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]),
          "+r"(d[96]), "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]),
          "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]),
          "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]),
          "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]),
          "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(da), "l"(db), "r"(1));
  }
};
#endif

// grouped raster over the allocated tiles (the grid is sized on the host, which does not know the live rows)
template <int BM, int BN>
__device__ __forceinline__ void tile_origin(int M, int N, int& m0, int& n0) {
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int pid = blockIdx.x, width = WG_GROUP_M * tiles_n;
  const int first_m = (pid / width) * WG_GROUP_M;
  const int gm = min(tiles_m - first_m, WG_GROUP_M);
  m0 = (first_m + (pid % width) % gm) * BM;
  n0 = ((pid % width) / gm) * BN;
}

// a warpgroup's 64 x BN block (first row r0, columns n0..; t: the thread's index in the warpgroup) -> out:
// acc[4j + 2 half + e] is row 16 w + lane / 4 + 8 half, column 8j + 2 (lane % 4) + e
template <int BN, int EPI>
__device__ __forceinline__ void wg_epilogue(const int32_t (&acc)[BN / 2], const float* __restrict__ sa,
                                            const float* __restrict__ sw, void* __restrict__ out, int ldo, int live,
                                            int N, int r0, int n0, int t) {
  const int w = t / 32, lane = t % 32;
  const int g = lane / 4, t4 = lane % 4;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = r0 + w * 16 + g + half * 8;
    if (row >= live) continue;
    const float s_a = sa[row];
    if constexpr (EPI == 0) {
      __nv_bfloat16* orow = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)row * ldo;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int col = n0 + j * 8 + 2 * t4;
        if (col >= N) continue;
        const float v0 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half], s_a), sw[col]);
        if (col + 1 < N) {
          const float v1 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half + 1], s_a), sw[col + 1]);
          *reinterpret_cast<__nv_bfloat162*>(orow + col) = __floats2bfloat162_rn(v0, v1);
        } else {
          orow[col] = __float2bfloat16(v0);
        }
      }
    } else {
      // residual add, 8 column pairs at a time: their loads are issued before the first add, so the DRAM latencies
      // overlap
      float* orow = reinterpret_cast<float*>(out) + (size_t)row * ldo;
#pragma unroll
      for (int j0 = 0; j0 < BN / 8; j0 += 8) {
        float2 old[8];
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int col = n0 + (j0 + jj) * 8 + 2 * t4;
          old[jj] = col + 1 < N ? *reinterpret_cast<const float2*>(orow + col)
                                : make_float2(col < N ? orow[col] : 0.f, 0.f);
        }
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int j = j0 + jj, col = n0 + j * 8 + 2 * t4;
          if (col >= N) continue;
          const float v0 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half], s_a), sw[col]);
          if (col + 1 < N) {
            const float v1 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half + 1], s_a), sw[col + 1]);
            *reinterpret_cast<float2*>(orow + col) = make_float2(__fadd_rn(old[jj].x, v0), __fadd_rn(old[jj].y, v1));
          } else {
            orow[col] = __fadd_rn(old[jj].x, v0);
          }
        }
      }
    }
  }
}

// The same epilogue through shared memory, for outputs whose rows are 16-byte aligned: each warp writes its 16 rows of
// the warpgroup's block, scaled (bf16 for EPI 0, fp32 for EPI 1), row-major into `stage` (rows padded by 16 bytes, so
// the fragments' 8-row stores hit distinct banks), then copies those rows out in 16-byte pieces - EPI 1 reads the
// residual's piece, adds, writes it back.  wg_epilogue's stores are 4-byte pieces of 8 rows per instruction: on the
// H200 they took 30% of a 2,048-row bf16 projection's time.  Same arithmetic, same bits.
template <int BN, int EPI>
__device__ __forceinline__ void wg_epilogue_staged(const int32_t (&acc)[BN / 2], const float* __restrict__ sa,
                                                   const float* __restrict__ sw, void* __restrict__ out, int ldo,
                                                   int live, int N, int r0, int n0, int t, uint8_t* stage) {
  constexpr int ES = EPI == 0 ? 2 : 4;   // output element bytes
  constexpr int ROWB = BN * ES + 16;      // a staged row, padded
  constexpr int CH = BN * ES / 16;        // 16-byte pieces per row
  constexpr int PER = 16 / ES;            // elements per piece
  const int w = t / 32, lane = t % 32, g = lane / 4, t4 = lane % 4;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int rl = w * 16 + g + half * 8, row = r0 + rl;
    const float s_a = row < live ? sa[row] : 0.f;
    uint8_t* srow = stage + rl * ROWB;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int cl = j * 8 + 2 * t4, col = n0 + cl;
      const float w0 = col < N ? sw[col] : 0.f, w1 = col + 1 < N ? sw[col + 1] : 0.f;
      const float v0 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half], s_a), w0);
      const float v1 = __fmul_rn(__fmul_rn((float)acc[4 * j + 2 * half + 1], s_a), w1);
      if constexpr (EPI == 0) {
        *reinterpret_cast<__nv_bfloat162*>(srow + cl * 2) = __floats2bfloat162_rn(v0, v1);
      } else {
        *reinterpret_cast<float2*>(srow + cl * 4) = make_float2(v0, v1);
      }
    }
  }
  __syncwarp();  // a warp copies out the rows it staged
  if constexpr (EPI == 0) {
#pragma unroll 4
    for (int i = lane; i < 16 * CH; i += 32) {
      const int rl = w * 16 + i / CH, c = i % CH, row = r0 + rl, col = n0 + c * PER;
      if (row >= live || col >= N) continue;
      __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)row * ldo + col;
      if (col + PER <= N) {
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(stage + rl * ROWB + c * 16);
      } else {
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(stage + rl * ROWB + c * 16);
        for (int k = 0; k < N - col; ++k) o[k] = e[k];
      }
    }
  } else {
    // the residual: B pieces per lane at a time, their loads issued before the first add (one read-add-write after
    // another waited for DRAM each time)
    constexpr int B = 8;
#pragma unroll 1
    for (int i0 = lane; i0 < 16 * CH; i0 += 32 * B) {
      float4 r[B];
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int i = i0 + 32 * b, rl = w * 16 + i / CH, c = i % CH, row = r0 + rl, col = n0 + c * PER;
        if (i < 16 * CH && row < live && col + PER <= N)
          r[b] = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(out) + (size_t)row * ldo + col);
      }
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const int i = i0 + 32 * b, rl = w * 16 + i / CH, c = i % CH, row = r0 + rl, col = n0 + c * PER;
        if (i >= 16 * CH || row >= live || col >= N) continue;
        float* o = reinterpret_cast<float*>(out) + (size_t)row * ldo + col;
        const float4 v = *reinterpret_cast<const float4*>(stage + rl * ROWB + c * 16);
        if (col + PER <= N) {
          *reinterpret_cast<float4*>(o) = make_float4(__fadd_rn(r[b].x, v.x), __fadd_rn(r[b].y, v.y),
                                                      __fadd_rn(r[b].z, v.z), __fadd_rn(r[b].w, v.w));
        } else {
          const float* e = reinterpret_cast<const float*>(stage + rl * ROWB + c * 16);
          for (int k = 0; k < N - col; ++k) o[k] = __fadd_rn(o[k], e[k]);
        }
      }
    }
  }
}

// A K split's raw int32 sums of a warpgroup's 64 x BN block (deferred split-K: the residual's next reader adds the
// splits' slots - exact - then scales the total and adds it, as the unsplit epilogue would; quant.norm_q8 / fold)
template <int BN>
__device__ __forceinline__ void wg_store_i32(const int32_t (&acc)[BN / 2], int32_t* __restrict__ out, int ldo, int live,
                                             int N, int r0, int n0, int t) {
  const int w = t / 32, lane = t % 32, g = lane / 4, t4 = lane % 4;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = r0 + w * 16 + g + half * 8;
    if (row >= live) continue;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int col = n0 + j * 8 + 2 * t4;
      if (col >= N) continue;
      int32_t* o = out + (size_t)row * ldo + col;
      if (col + 1 < N) {
        *reinterpret_cast<int2*>(o) = make_int2(acc[4 * j + 2 * half], acc[4 * j + 2 * half + 1]);
      } else {
        o[0] = acc[4 * j + 2 * half];
      }
    }
  }
}

template <int BM, int BN, int STAGES>
struct WgTile {
  static constexpr int WGS = BM / 64;                       // warpgroups (64 rows each)
  static constexpr int THREADS = WGS * 128;
  static constexpr int NACC = BN / 2;                       // int32 accumulators per thread
  static constexpr int SMEM = STAGES * (BM + BN) * WG_BK + 1024;   // + slack to align the stages to 1024 bytes
  static_assert(BM % 64 == 0 && (BN == 64 || BN == 128 || BN == 256) && STAGES >= 3, "tile");
};

template <int BM, int BN, int STAGES, int EPI>
__global__ void __launch_bounds__(WgTile<BM, BN, STAGES>::THREADS, 1)
    w8a8_wgmma_kernel(const int8_t* __restrict__ A, const float* __restrict__ sa, const int8_t* __restrict__ W,
                      const float* __restrict__ sw, void* __restrict__ out, int ldo, const int* __restrict__ n_rows,
                      int M, int N, int K) {
#ifdef FLASHJEV_WGMMA_DEVICE
  using T = WgTile<BM, BN, STAGES>;
  constexpr int THREADS = T::THREADS, NACC = T::NACC, CH = WG_BK / 16;
  const int live = min(*n_rows, M);  // written by the batch's table upload, not by the kernel before this one
  int m0, n0;
  tile_origin<BM, BN>(M, N, m0, n0);
  if (m0 >= live) return;

  extern __shared__ uint8_t smem_raw[];
  const uint32_t base_u32 = (smem_u32(smem_raw) + 1023) & ~1023u;
  uint8_t* smem = smem_raw + (base_u32 - smem_u32(smem_raw));
  uint8_t* sA = smem;                              // [STAGES][BM * 128]
  uint8_t* sB = smem + STAGES * BM * WG_BK;        // [STAGES][BN * 128]
  const int tid = threadIdx.x, wg = tid / 128;

  auto load_stage = [&](int stage, int kt) {
    const int kb = kt * WG_BK;
    uint8_t* a = sA + stage * BM * WG_BK;
    uint8_t* b = sB + stage * BN * WG_BK;
#pragma unroll
    for (int i = tid; i < BM * CH; i += THREADS) {
      const int r = i / CH, c = i % CH, gm_ = m0 + r;
      cp_async16(smem_u32(a + swz128(r, c)), A + (size_t)min(gm_, M - 1) * K + kb + c * 16, gm_ < M);
    }
#pragma unroll
    for (int i = tid; i < BN * CH; i += THREADS) {
      const int r = i / CH, c = i % CH, gn = n0 + r;
      cp_async16(smem_u32(b + swz128(r, c)), W + (size_t)min(gn, N - 1) * K + kb + c * 16, gn < N);
    }
  };

  int32_t acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0;

  const int k_tiles = K / WG_BK;
#pragma unroll
  for (int s = 0; s < STAGES - 2; ++s) {
    if (s < k_tiles) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < k_tiles; ++kt) {
    cp_async_wait<STAGES - 3>();  // this thread's copies of stage kt landed
    fence_async_shared();
    __syncthreads();              // everyone's copies of stage kt landed; every warpgroup's MMAs on kt - 2 completed
    {
      const int nk = kt + STAGES - 2;  // refill the stage the MMAs of kt - 2 read
      if (nk < k_tiles) load_stage(nk % STAGES, nk);
      cp_async_commit();
    }
    const int st = kt % STAGES;
    const uint64_t da = sw128_desc(smem_u32(sA + st * BM * WG_BK + wg * 64 * WG_BK));
    const uint64_t db = sw128_desc(smem_u32(sB + st * BN * WG_BK));
    fence_operands(acc);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < WG_BK / 32; ++kk) Wgmma<BN>::mma(acc, da + 2 * kk, db + 2 * kk);  // +32 bytes per k32 slice
    wg_commit();
    wg_wait<1>();                 // the previous stage's MMAs completed (this warpgroup)
    fence_operands(acc);
  }
  wg_wait<0>();
  fence_operands(acc);

  wg_epilogue<BN, EPI>(acc, sa, sw, out, ldo, live, N, m0 + wg * 64, n0, tid % 128);
#else
  __trap();  // built without sm_90a (the host launcher refuses before this)
#endif
}

template <int BM, int BN, int STAGES>
struct TmaTile {
  static constexpr int CWGS = BM / 64;                        // MMA warpgroups (64 rows each)
  static constexpr int THREADS = (CWGS + 1) * 128;            // + the producer warpgroup
  static constexpr int STAGE_BYTES = (BM + BN) * WG_BK;
  static constexpr int SMEM = STAGES * STAGE_BYTES + 1024 + 2 * STAGES * 8;   // stages (1024-aligned), mbarriers
  static_assert(BM % 64 == 0 && BM <= 128 && (BN == 64 || BN == 128 || BN == 256) && STAGES >= 2, "tile");
};

template <int BM, int BN, int STAGES, int EPI>
__global__ void __launch_bounds__(TmaTile<BM, BN, STAGES>::THREADS, 1)
    w8a8_wgmma_tma_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                          const float* __restrict__ sa, const float* __restrict__ sw, void* __restrict__ out, int ldo,
                          const int* __restrict__ n_rows, int M, int N, int K, int vec, int kt_per_split,
                          int64_t slot_stride) {
#ifdef FLASHJEV_WGMMA_DEVICE
  using T = TmaTile<BM, BN, STAGES>;
  constexpr int ROWB = BN * (EPI == 0 ? 2 : 4) + 16;   // wg_epilogue_staged's rows, in the stages' memory
  static_assert(STAGES * T::STAGE_BYTES >= T::CWGS * 64 * ROWB, "the staged epilogue does not fit in the stages");
  const int live = min(*n_rows, M);
  int m0, n0;
  tile_origin<BM, BN>(M, N, m0, n0);
  if (m0 >= live) return;

  extern __shared__ uint8_t smem_raw[];
  const uint32_t base_u32 = (smem_u32(smem_raw) + 1023) & ~1023u;
  uint8_t* smem = smem_raw + (base_u32 - smem_u32(smem_raw));
  uint8_t* sA = smem;                              // [STAGES][BM * 128]
  uint8_t* sB = smem + STAGES * BM * WG_BK;        // [STAGES][BN * 128]
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * T::STAGE_BYTES);   // loaded (TMA transaction bytes)
  uint64_t* empty = full + STAGES;                 // released by every MMA warpgroup
  const int tid = threadIdx.x, wg = tid / 128;
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], T::CWGS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int k_tiles = kt_per_split, kt0 = blockIdx.y * kt_per_split;   // this K split's 128-byte steps

  if (wg == 0) {  // producer: one thread keeps STAGES - 1 stages of TMA copies in flight
    if constexpr (T::CWGS == 2) setmaxnreg_dec<40>();
    if (tid == 0) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&tmA) : "memory");
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&tmW) : "memory");
      for (int kt = 0; kt < k_tiles; ++kt) {
        const int s = kt % STAGES;
        mbar_wait(&empty[s], ((kt / STAGES) & 1) ^ 1);   // a fresh barrier's preceding phase counts as completed
        mbar_expect_tx(&full[s], T::STAGE_BYTES);
        tma_load_2d(sA + s * BM * WG_BK, &tmA, (kt0 + kt) * WG_BK, m0, &full[s]);
        tma_load_2d(sB + s * BN * WG_BK, &tmW, (kt0 + kt) * WG_BK, n0, &full[s]);
      }
    }
  } else {        // MMA warpgroups: 64 rows each
    if constexpr (T::CWGS == 2) setmaxnreg_inc<232>();
    const int cw = wg - 1;
    int32_t acc[BN / 2];
#pragma unroll
    for (int i = 0; i < BN / 2; ++i) acc[i] = 0;
    for (int kt = 0; kt < k_tiles; ++kt) {
      const int s = kt % STAGES;
      mbar_wait(&full[s], (kt / STAGES) & 1);
      const uint64_t da = sw128_desc(smem_u32(sA + s * BM * WG_BK + cw * 64 * WG_BK));
      const uint64_t db = sw128_desc(smem_u32(sB + s * BN * WG_BK));
      fence_operands(acc);
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < WG_BK / 32; ++kk) Wgmma<BN>::mma(acc, da + 2 * kk, db + 2 * kk);
      wg_commit();
      wg_wait<1>();               // the previous stage's MMAs completed: release it
      fence_operands(acc);
      if (kt > 0 && tid % 128 == 0) mbar_arrive(&empty[(kt - 1) % STAGES]);
    }
    wg_wait<0>();
    fence_operands(acc);
    if (slot_stride > 0) {
      wg_store_i32<BN>(acc, reinterpret_cast<int32_t*>(out) + blockIdx.y * slot_stride, ldo, live, N, m0 + cw * 64,
                       n0, tid % 128);
    } else if (vec) {
      // the staging rows overwrite stages the other warpgroup's last MMAs may still read
      if constexpr (T::CWGS > 1) asm volatile("bar.sync 1, %0;\n" ::"n"(T::CWGS * 128) : "memory");
      wg_epilogue_staged<BN, EPI>(acc, sa, sw, out, ldo, live, N, m0 + cw * 64, n0, tid % 128,
                                  smem + cw * 64 * ROWB);
    } else {
      wg_epilogue<BN, EPI>(acc, sa, sw, out, ldo, live, N, m0 + cw * 64, n0, tid % 128);
    }
  }
#else
  __trap();
#endif
}

// TMA descriptors: a 2-D tensor map (rows x k int8, 128-byte rows of K per box, 128-byte swizzle) per (address, rows,
// k, box rows), encoded once.  Kernel arguments, so a captured CUDA graph keeps the ones of its capture.
using MapKey = std::tuple<const void*, int64_t, int64_t, int>;

static const CUtensorMap& tensor_map(const void* ptr, int64_t rows, int64_t k, int box_rows) {
  static std::map<MapKey, CUtensorMap> cache;
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  auto key = MapKey(ptr, rows, k, box_rows);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  if (encode == nullptr) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess && fn != nullptr, "cuTensorMapEncodeTiled not available");
    encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  CUtensorMap map;
  cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)k};
  cuuint32_t box[2] = {(cuuint32_t)WG_BK, (cuuint32_t)box_rows};
  cuuint32_t elem[2] = {1, 1};
  CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, elem,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return cache.emplace(key, map).first->second;
}

template <int BM, int BN, int STAGES, int EPI>
void launch_tma(const torch::Tensor& a, const torch::Tensor& sa, const torch::Tensor& w, const torch::Tensor& sw,
                torch::Tensor& out, const torch::Tensor& n_rows, int split, int64_t slot_stride) {
  using T = TmaTile<BM, BN, STAGES>;
  if constexpr (T::SMEM > WG_SMEM_MAX) {
    TORCH_CHECK(false, "w8a8_wgmma: the stages do not fit in shared memory");
  } else {
    const int M = a.size(0), K = a.size(1), N = w.size(0);
    TORCH_CHECK(K % (WG_BK * split) == 0, "w8a8_wgmma: K must be a multiple of 128 * split");
    const auto aligned = [](const torch::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0; };
    TORCH_CHECK(aligned(a) && aligned(w), "w8a8_wgmma: TMA needs 16-byte aligned operands");
    const CUtensorMap& ma = tensor_map(a.data_ptr(), M, K, BM);
    const CUtensorMap& mw = tensor_map(w.data_ptr(), N, K, BN);
    auto kern = w8a8_wgmma_tma_kernel<BM, BN, STAGES, EPI>;
    static bool configured = false;
    if (!configured) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
      configured = true;
    }
    const int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    // 16-byte aligned output rows: the staged epilogue's vector stores
    const int vec = aligned(out) && out.stride(0) * out.element_size() % 16 == 0;
    kern<<<dim3(tiles, split), T::THREADS, T::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
        ma, mw, sa.data_ptr<float>(), sw.data_ptr<float>(), out.data_ptr(), (int)out.stride(0),
        n_rows.data_ptr<int>(), M, N, K, vec, K / WG_BK / split, slot_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int BM, int BN, int STAGES, int EPI>
void launch_wg(const torch::Tensor& a, const torch::Tensor& sa, const torch::Tensor& w, const torch::Tensor& sw,
               torch::Tensor& out, const torch::Tensor& n_rows, int split, int64_t slot_stride) {
  using T = WgTile<BM, BN, STAGES>;
  if constexpr (T::SMEM > WG_SMEM_MAX) {
    TORCH_CHECK(false, "w8a8_wgmma: the stages do not fit in shared memory");
  } else {
    TORCH_CHECK(split == 1 && slot_stride == 0, "w8a8_wgmma: split-K on the TMA pipeline only");
    const int M = a.size(0), K = a.size(1), N = w.size(0);
    TORCH_CHECK(K % WG_BK == 0, "w8a8_wgmma: K must be a multiple of 128");
    auto kern = w8a8_wgmma_kernel<BM, BN, STAGES, EPI>;
    static bool configured = false;
    if (!configured) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
      configured = true;
    }
    const int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    kern<<<tiles, T::THREADS, T::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<int8_t>(), sa.data_ptr<float>(), w.data_ptr<int8_t>(), sw.data_ptr<float>(), out.data_ptr(),
        (int)out.stride(0), n_rows.data_ptr<int>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

// cfg: 0 = 128x256 (2 warpgroups), 1 = 128x128 (2), 2 = 64x256 (1), 3 = 64x128 (1) on the cp.async pipeline;
//      4..7 = the same tiles on the TMA pipeline (+ the producer warpgroup); 8 = 64x64, 9 = 128x64 on the TMA
//      pipeline (small batches: four times the tiles of 256-wide ones, so the SMs fill); K 128 bytes per stage
#define W8A8_WG_CONFIGS(X) \
  X(0, false, 128, 256)    \
  X(1, false, 128, 128)    \
  X(2, false, 64, 256)     \
  X(3, false, 64, 128)     \
  X(4, true, 128, 256)     \
  X(5, true, 128, 128)     \
  X(6, true, 64, 256)      \
  X(7, true, 64, 128)      \
  X(8, true, 64, 64)       \
  X(9, true, 128, 64)

void w8a8_wgmma(torch::Tensor a, torch::Tensor sa, torch::Tensor w, torch::Tensor sw, torch::Tensor out,
                torch::Tensor n_rows, int64_t epi, int64_t cfg, int64_t stages, int64_t split, int64_t slot_stride) {
#ifndef FLASHJEV_SM90A
  TORCH_CHECK(false, "w8a8_wgmma: the extension was built without sm_90a (FLASHJEV_CUDA_ARCH=90a)");
#endif
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9, "w8a8_wgmma: Hopper (sm_90) only");
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8, "int8 operands");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1 && a.stride(0) == a.size(1) && w.stride(0) == w.size(1),
              "contiguous K-major operands");
  // split > 1 (residual add only): `out` is the first of `split` int32 slots, slot_stride elements apart
  // (one split with slot_stride > 0: int32 sums out, gemm_i32 of a tensor-parallel shard)
  TORCH_CHECK(slot_stride == 0 ? split == 1 : (epi == 1 && slot_stride >= out.size(0) * out.stride(0)), "split slots");
  TORCH_CHECK(slot_stride > 0 ? out.dtype() == torch::kInt32
                              : (epi == 0 ? out.dtype() == torch::kBFloat16 : out.dtype() == torch::kFloat32),
              "output dtype");
  TORCH_CHECK(stages >= 3 && stages <= 6, "stages: 3..6");
  const int split_i = (int)split;
#define W8A8_WG_STAGES(L, BM, BN, E)                                                   \
  switch (stages) {                                                                    \
    case 3: L<BM, BN, 3, E>(a, sa, w, sw, out, n_rows, split_i, slot_stride); break;   \
    case 4: L<BM, BN, 4, E>(a, sa, w, sw, out, n_rows, split_i, slot_stride); break;   \
    case 5: L<BM, BN, 5, E>(a, sa, w, sw, out, n_rows, split_i, slot_stride); break;   \
    default: L<BM, BN, 6, E>(a, sa, w, sw, out, n_rows, split_i, slot_stride); break;  \
  }
#define W8A8_WG_CASE(C, TMA, BM, BN)                                       \
  case C:                                                                  \
    if (TMA) {                                                             \
      if (epi == 0) {                                                      \
        W8A8_WG_STAGES(launch_tma, BM, BN, 0)                              \
      } else {                                                             \
        W8A8_WG_STAGES(launch_tma, BM, BN, 1)                              \
      }                                                                    \
    } else {                                                               \
      if (epi == 0) {                                                      \
        W8A8_WG_STAGES(launch_wg, BM, BN, 0)                               \
      } else {                                                             \
        W8A8_WG_STAGES(launch_wg, BM, BN, 1)                               \
      }                                                                    \
    }                                                                      \
    break;
  switch (cfg) {
    W8A8_WG_CONFIGS(W8A8_WG_CASE)
    default:
      TORCH_CHECK(false, "unknown wgmma cfg");
  }
#undef W8A8_WG_CASE
#undef W8A8_WG_STAGES
}

TORCH_LIBRARY_FRAGMENT(flash_jev, m) {
  m.def("w8a8_wgmma(Tensor a, Tensor sa, Tensor w, Tensor sw, Tensor(a!) out, Tensor n_rows, int epi, int cfg, "
        "int stages, int split=1, int slot_stride=0) -> ()");
}
TORCH_LIBRARY_IMPL(flash_jev, CUDA, m) { m.impl("w8a8_wgmma", &w8a8_wgmma); }
