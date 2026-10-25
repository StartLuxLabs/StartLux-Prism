// W4A8 GEMM on Hopper's warpgroup MMA (sm_90a): a Q4_K_M file's own k-quant weights with int8 activations, the
// arithmetic of kernels/w4a8.py bit for bit (as csrc/w4a8_gemm.cu), its products on wgmma.
//
// csrc/w4a8_gemm.cu applies the sub-blocks' integer scales on the CUDA cores - an integer multiply-add per output after
// every 32-input MMA - so on the H200 its MMAs wait for issue slots (200-300 TOPS; W8A8 runs at 1,250 on wgmma).  Here
// the scales go into the weights: each 128-input stage of the packed weights is expanded in shared memory into two
// int8 planes whose int32 dot products with the activations give the super-block's sum exactly:
//   Q4_K  sc = s0 + 8 s1 (3 bits each): planes s0 q and s1 q (<= 105), acc = sum x (s0 q) + 8 sum x (s1 q);
//         one 32-bit multiply scales four bytes (no carry between them)
//   Q6_K  w' = sc q (-4064..4096) split lo = w' & 63, hi = w' >> 6: acc = sum x lo + 64 sum x hi; one multiply-add
//         per two values in 16-bit lanes (expand_q6)
// so a 32-input step is one m64n256k32 wgmma (both planes side by side as its B) and nothing per output until the
// super-block ends.  Q4_K's mins, sum over the 8 sub-blocks of m * S with the activations' sums split S = S_lo + 64
// S_hi, are four small wgmmas per super-block.  Then the reference's floating-point step, rounded where it rounds:
// the exact int32 sum converted once (I2F, the reference's float(acc)), t = d * float(acc), Q4_K
// t = fma(-dmin, float(mins), t) (|mins| < 2^24: exact), y = fma(a, t, y).
//
//   Operands as csrc/w4a8_gemm.cu takes them: A int8 [M, K], a fp32 [K / 256, M], xs int8 [K / 256, M, 16] (Q4_K); W
//   packed [N, K / 2] (Q4_K) or int8 q - 32 [N, K] (Q6_K); sc, mn, d, dmin super-block-major.
//
// A CTA: a producer warpgroup and BM / 64 MMA warpgroups (64 rows each), a BM x 128 tile, 128 inputs per stage (two
// per super-block), three stages.  The producer's thread 0 keeps TMA copies of the raw stages ahead in flight (the
// packed weights and the stage's sub-block scales; per super-block d, dmin, mins, a and the activation sums); each MMA
// warpgroup loads its own 64 rows of activations (128-byte swizzle) as soon as its MMAs are done with a slot.  Every
// warp expands a share of each stage, writing the planes in the 128-byte swizzle wgmma reads (chunk c of row r at
// (c ^ (r % 8)) * 16): the producer warpgroup its rows of the current stage, the MMA warpgroups theirs of the next one
// while their MMAs run.  The main loop runs a super-block per iteration: its first wgmma writes the accumulators (no
// reset), so they are not live across the floating-point step.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <tuple>
#include <map>
#include <optional>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define STARTLUX_PRISM_W4WG_DEVICE 1
#endif

namespace {

constexpr int W4_BK = 128;          // inputs per stage (half a super-block)
constexpr int W4_BN = 128;          // weight rows (output columns) per tile
constexpr int W4_GROUP_M = 8;       // row tiles per raster group
constexpr int W4_SMEM_MAX = 232448; // dynamic shared memory per block on sm_90 (227 KB)
constexpr int MIN_SBO = 256;        // mins operands: 8-row core-matrix groups 256 bytes apart,
constexpr int MIN_LBO = 128;        // the k16..31 core matrices 128 bytes after k0..15 (no swizzle)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

#ifdef STARTLUX_PRISM_W4WG_DEVICE
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
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
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
  asm volatile(
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
// box of a 2-D tensor map at (inner, outer) -> shared memory; completes on `bar` (transaction bytes)
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int inner, int outer, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(inner), "r"(outer), "r"(smem_u32(bar))
      : "memory");
}
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R)); }
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R)); }
#endif

// K-major operand descriptors: 128-byte rows in the 128-byte swizzle (8-row groups 1024 bytes apart; +32 bytes is the
// next k32), or no swizzle in 8 x 16-byte core matrices (LBO: the next 16 bytes of K, SBO: the next 8 rows)
__device__ __forceinline__ uint64_t sw128_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ uint64_t core_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(MIN_LBO >> 4) << 16) | ((uint64_t)(MIN_SBO >> 4) << 32);
}
__device__ __forceinline__ int swz128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

#ifdef STARTLUX_PRISM_W4WG_DEVICE
// d[64 x 256] (+)= A (64 x 32 int8) * B (256 x 32 int8)^T: B is an expanded stage's lo tile then its hi tile (128
// rows each, adjacent), so d[0..63] are the lo sums and d[64..127] the hi sums of the 64 x 128 block - one product
// reads the activations once for both.  Per thread 128 accumulators, laid out per warp as mma.sync's m16n8 ones: warp
// w of the warpgroup holds rows 16w + lane / 4 (+ 8), columns 8j + 2 (lane % 4) (+ 1) in d[4j .. 4j + 3].  first:
// the accumulators are written, not read (a super-block's first product).
__device__ __forceinline__ void mma_n256(int32_t (&d)[128], uint64_t da, uint64_t db, bool first) {
  if (first) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, "
        "%124, %125, %126, %127}, "
        "%128, %129, 0;\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]), "=r"(d[8]),
          "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]), "=r"(d[16]),
          "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]),
          "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31]), "=r"(d[32]),
          "=r"(d[33]), "=r"(d[34]), "=r"(d[35]), "=r"(d[36]), "=r"(d[37]), "=r"(d[38]), "=r"(d[39]), "=r"(d[40]),
          "=r"(d[41]), "=r"(d[42]), "=r"(d[43]), "=r"(d[44]), "=r"(d[45]), "=r"(d[46]), "=r"(d[47]), "=r"(d[48]),
          "=r"(d[49]), "=r"(d[50]), "=r"(d[51]), "=r"(d[52]), "=r"(d[53]), "=r"(d[54]), "=r"(d[55]), "=r"(d[56]),
          "=r"(d[57]), "=r"(d[58]), "=r"(d[59]), "=r"(d[60]), "=r"(d[61]), "=r"(d[62]), "=r"(d[63]), "=r"(d[64]),
          "=r"(d[65]), "=r"(d[66]), "=r"(d[67]), "=r"(d[68]), "=r"(d[69]), "=r"(d[70]), "=r"(d[71]), "=r"(d[72]),
          "=r"(d[73]), "=r"(d[74]), "=r"(d[75]), "=r"(d[76]), "=r"(d[77]), "=r"(d[78]), "=r"(d[79]), "=r"(d[80]),
          "=r"(d[81]), "=r"(d[82]), "=r"(d[83]), "=r"(d[84]), "=r"(d[85]), "=r"(d[86]), "=r"(d[87]), "=r"(d[88]),
          "=r"(d[89]), "=r"(d[90]), "=r"(d[91]), "=r"(d[92]), "=r"(d[93]), "=r"(d[94]), "=r"(d[95]), "=r"(d[96]),
          "=r"(d[97]), "=r"(d[98]), "=r"(d[99]), "=r"(d[100]), "=r"(d[101]), "=r"(d[102]), "=r"(d[103]),
          "=r"(d[104]), "=r"(d[105]), "=r"(d[106]), "=r"(d[107]), "=r"(d[108]), "=r"(d[109]), "=r"(d[110]),
          "=r"(d[111]), "=r"(d[112]), "=r"(d[113]), "=r"(d[114]), "=r"(d[115]), "=r"(d[116]), "=r"(d[117]),
          "=r"(d[118]), "=r"(d[119]), "=r"(d[120]), "=r"(d[121]), "=r"(d[122]), "=r"(d[123]), "=r"(d[124]),
          "=r"(d[125]), "=r"(d[126]), "=r"(d[127])
        : "l"(da), "l"(db));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, "
        "%124, %125, %126, %127}, "
        "%128, %129, 1;\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
          "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
          "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
          "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]), "+r"(d[32]),
          "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]), "+r"(d[40]),
          "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]), "+r"(d[48]),
          "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]), "+r"(d[56]),
          "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]), "+r"(d[64]),
          "+r"(d[65]), "+r"(d[66]), "+r"(d[67]), "+r"(d[68]), "+r"(d[69]), "+r"(d[70]), "+r"(d[71]), "+r"(d[72]),
          "+r"(d[73]), "+r"(d[74]), "+r"(d[75]), "+r"(d[76]), "+r"(d[77]), "+r"(d[78]), "+r"(d[79]), "+r"(d[80]),
          "+r"(d[81]), "+r"(d[82]), "+r"(d[83]), "+r"(d[84]), "+r"(d[85]), "+r"(d[86]), "+r"(d[87]), "+r"(d[88]),
          "+r"(d[89]), "+r"(d[90]), "+r"(d[91]), "+r"(d[92]), "+r"(d[93]), "+r"(d[94]), "+r"(d[95]), "+r"(d[96]),
          "+r"(d[97]), "+r"(d[98]), "+r"(d[99]), "+r"(d[100]), "+r"(d[101]), "+r"(d[102]), "+r"(d[103]),
          "+r"(d[104]), "+r"(d[105]), "+r"(d[106]), "+r"(d[107]), "+r"(d[108]), "+r"(d[109]), "+r"(d[110]),
          "+r"(d[111]), "+r"(d[112]), "+r"(d[113]), "+r"(d[114]), "+r"(d[115]), "+r"(d[116]), "+r"(d[117]),
          "+r"(d[118]), "+r"(d[119]), "+r"(d[120]), "+r"(d[121]), "+r"(d[122]), "+r"(d[123]), "+r"(d[124]),
          "+r"(d[125]), "+r"(d[126]), "+r"(d[127])
        : "l"(da), "l"(db));
  }
}

// d[64 x 64] = A (64 x 32 int8) * B (64 x 32 int8)^T, written (the mins: one product per super-block and half)
__device__ __forceinline__ void mma_n64_first(int32_t (&d)[32], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, "
      "%22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, 0;\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]), "=r"(d[8]),
        "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]), "=r"(d[16]),
        "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]),
        "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "l"(da), "l"(db));
}

// Q4_K: the low (s = 0) or high (s = 4) nibbles of a word of packed bytes (byte t: q[t] | q[t + 16] << 4), q of
// k = 4i..4i+3 or 16 + 4i.., already in place as bytes, times the two halves of their sub-block's scale sc = s0 + 8 s1
// (3 bits each): s0 q and s1 q are at most 105, so one multiply takes four bytes without a carry between them.
__device__ __forceinline__ void expand_q4(uint32_t w, int s, uint32_t s0, uint32_t s1, uint32_t& p0, uint32_t& p1) {
  const uint32_t q = (w >> s) & 0x0F0F0F0Fu;
  p0 = q * s0;
  p1 = q * s1;
}

// Q6_K: a word of int8 q (-32..31) and its 16-group's scale sc (int8), with c1 = (16384 - 32 sc) * 0x10001 -> the lo
// and hi bytes of its four values.  With q' = q + 32 (0..63) two to a 16-bit lane, q' sc + 16384 - 32 sc =
// sc q + 4096 + 12288 = u lies in 12320..20480 in each lane: one multiply-add, whose final lanes need no borrow
// (exact modulo 2^32, whatever the intermediate signs).  lo = u & 63 (12288 = 192 * 64 changes no low bits), and the
// low byte of u >> 6 = (sc q >> 6) + 64 + 192 is hi as an int8.
__device__ __forceinline__ void expand_q6(uint32_t w, uint32_t sc, uint32_t c1, uint32_t& lo, uint32_t& hi) {
  const uint32_t q = (w & 0x3F3F3F3Fu) ^ 0x20202020u;
  const uint32_t ue = (q & 0x00FF00FFu) * sc + c1, uo = ((q >> 8) & 0x00FF00FFu) * sc + c1;
  lo = __byte_perm(ue, uo, 0x6240) & 0x3F3F3F3Fu;
  hi = __byte_perm(ue >> 6, uo >> 6, 0x6240);
}

// one item of a stage, row r's k32 j (32 weights) -> its lo and hi bytes in the expanded tile ex (lo, then hi), one
// 16-byte chunk at a time: few registers, so the MMA warpgroups expand next to their live accumulators.  rs: the raw
// stage (Q4_K: 64 bytes a row; Q6_K: 128 in the 128-byte swizzle), then its sub-block scales.
template <int KIND>
__device__ __forceinline__ void expand_item(const uint8_t* rs, uint8_t* ex, int r, int j, int half) {
  constexpr int RAW_W = W4_BN * (KIND == 0 ? W4_BK / 2 : W4_BK);
  if constexpr (KIND == 0) {
    const uint4 raw = *reinterpret_cast<const uint4*>(rs + r * 64 + j * 16);
    const uint32_t sc = rs[RAW_W + r * 8 + half * 4 + j], s0 = sc & 7, s1 = sc >> 3;
#pragma unroll
    for (int c = 0; c < 2; ++c) {   // low nibbles: k 0..15 of the k32 (chunk 2j); high nibbles: k 16..31 (2j + 1)
      uint32_t lo[4], hi[4];
      expand_q4(raw.x, 4 * c, s0, s1, lo[0], hi[0]);
      expand_q4(raw.y, 4 * c, s0, s1, lo[1], hi[1]);
      expand_q4(raw.z, 4 * c, s0, s1, lo[2], hi[2]);
      expand_q4(raw.w, 4 * c, s0, s1, lo[3], hi[3]);
      *reinterpret_cast<uint4*>(ex + swz128(r, 2 * j + c)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
      *reinterpret_cast<uint4*>(ex + W4_BN * W4_BK + swz128(r, 2 * j + c)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    }
  } else {
#pragma unroll
    for (int c = 0; c < 2; ++c) {   // the k32's two 16-groups, one scale each
      const uint4 raw = *reinterpret_cast<const uint4*>(rs + swz128(r, 2 * j + c));
      const int sc = reinterpret_cast<const int8_t*>(rs + RAW_W)[r * 16 + half * 8 + 2 * j + c];
      const uint32_t c1 = (uint32_t)(16384 - 32 * sc) * 0x10001u;
      uint32_t lo[4], hi[4];
      expand_q6(raw.x, (uint32_t)sc, c1, lo[0], hi[0]);
      expand_q6(raw.y, (uint32_t)sc, c1, lo[1], hi[1]);
      expand_q6(raw.z, (uint32_t)sc, c1, lo[2], hi[2]);
      expand_q6(raw.w, (uint32_t)sc, c1, lo[3], hi[3]);
      *reinterpret_cast<uint4*>(ex + swz128(r, 2 * j + c)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
      *reinterpret_cast<uint4*>(ex + W4_BN * W4_BK + swz128(r, 2 * j + c)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    }
  }
}
#endif

// Shared memory: per MMA warpgroup a ring of C activation tiles (its 64 rows; it loads them itself, as soon as its
// MMAs are done with a slot); a ring of E expanded weight tiles (lo, hi); a ring of P raw stages (packed weights and
// the stage's sub-block scales), which the producer frees itself once expanded, so their copies run P - 1 stages
// ahead whatever the MMA warpgroups do; per super-block parity the floating-point step's values (d, a, and Q4_K's
// dmin, mins, activation sums and mins operands), freed after that step; the mbarriers.
template <int KIND, int BM, int STAGES>
struct W4Tile {
  static constexpr int CWGS = BM / 64;                        // MMA warpgroups
  static constexpr int THREADS = (CWGS + 1) * 128;            // + the producer warpgroup
  static constexpr int BN = W4_BN;
  static constexpr int C = STAGES, E = STAGES, P = STAGES + 1;
  static constexpr int WARPS = THREADS / 32;                  // every warp expands a share of each stage:
  static constexpr int PROD_ROWS = BN - 32 * CWGS;            // the producer's rows, then 32 per MMA warpgroup
  static constexpr int A_BYTES = 64 * W4_BK;                  // a warpgroup's rows of a stage
  static constexpr int A_OFF = 0;                             // [CWGS][C]
  static constexpr int EXP_BYTES = 2 * BN * W4_BK;            // lo, then hi
  static constexpr int EXP_OFF = A_OFF + CWGS * C * A_BYTES;  // [E]
  static constexpr int RAW_W = BN * (KIND == 0 ? W4_BK / 2 : W4_BK), RAW_SC = BN * (KIND == 0 ? 8 : 16);
  static constexpr int RAW_BYTES = (RAW_W + RAW_SC + 1023) / 1024 * 1024;
  static constexpr int RAW_OFF = EXP_OFF + E * EXP_BYTES;     // [P]: values, then scales
  static constexpr int SB_D = 0, SB_AS = SB_D + BN * 4, SB_DM = SB_AS + BM * 4;
  static constexpr int SB_MN = SB_DM + (KIND == 0 ? BN * 4 : 0), SB_XS = SB_MN + (KIND == 0 ? BN * 8 : 0);
  static constexpr int SB_MIN = (SB_XS + (KIND == 0 ? BM * 16 : 0) + 127) / 128 * 128;
  // Q4_K's mins operands: A1 (S_lo), A2 (S_hi) [BM rows], B (m) [BN rows], 32 bytes of K per row in core matrices
  static constexpr int MIN_A = BM * 32, MIN_B = BN * 32;
  static constexpr int SB_BYTES = (SB_MIN + (KIND == 0 ? 2 * MIN_A + MIN_B : 0) + 127) / 128 * 128;
  static constexpr int SB_OFF = RAW_OFF + P * RAW_BYTES;      // [2]
  static constexpr int BAR_OFF = SB_OFF + 2 * SB_BYTES;
  static constexpr int N_BARS = 2 * P + 2 * E + CWGS * C + 4;
  static constexpr int SMEM = BAR_OFF + N_BARS * 8 + 1024;    // + slack to align the base to 1024 bytes
  static constexpr int TX_RAW = RAW_W + RAW_SC;
  static constexpr int TX_SB = BN * 4 + BM * 4 + (KIND == 0 ? BN * 4 + BN * 8 + BM * 16 : 0);
  static_assert(BM == 64 || BM == 128, "tile");
  static_assert(STAGES >= 3, "the MMA warpgroups expand a stage ahead: three expanded tiles at least");
};

template <int BM>
__device__ __forceinline__ void w4_tile_origin(int M, int N, int& m0, int& n0) {
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + W4_BN - 1) / W4_BN;
  const int pid = blockIdx.x, width = W4_GROUP_M * tiles_n;
  const int first_m = (pid / width) * W4_GROUP_M;
  const int gm = min(tiles_m - first_m, W4_GROUP_M);
  m0 = (first_m + (pid % width) % gm) * BM;
  n0 = ((pid % width) / gm) * W4_BN;
}

// a warpgroup's 64 x 128 block of y -> out (bf16 store, EPI 0, or fp32 residual add, EPI 1), through shared memory
// when the output rows are 16-byte aligned (as csrc/w8a8_wgmma.cu's staged epilogue); y[4j + 2 half + e] is row
// 16 w + lane / 4 + 8 half, column 8j + 2 (lane % 4) + e
template <int EPI>
__device__ __forceinline__ void w4_store(const float (&y)[64], void* __restrict__ out, int ldo, int live, int N, int r0,
                                         int n0, int t, uint8_t* stage, bool vec) {
  constexpr int BN = W4_BN;
  const int w = t / 32, lane = t % 32, g = lane / 4, t4 = lane % 4;
  if (!vec) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = r0 + w * 16 + g + half * 8;
      if (row >= live) continue;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int col = n0 + j * 8 + 2 * t4;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          if (col + e >= N) continue;
          const float v = y[4 * j + 2 * half + e];
          if constexpr (EPI == 0) {
            reinterpret_cast<__nv_bfloat16*>(out)[(size_t)row * ldo + col + e] = __float2bfloat16(v);
          } else {
            float* o = reinterpret_cast<float*>(out) + (size_t)row * ldo + col + e;
            *o = __fadd_rn(*o, v);
          }
        }
      }
    }
    return;
  }
  constexpr int ES = EPI == 0 ? 2 : 4;
  constexpr int ROWB = BN * ES + 16;
  constexpr int CH = BN * ES / 16;
  constexpr int PER = 16 / ES;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint8_t* srow = stage + (w * 16 + g + half * 8) * ROWB;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int cl = j * 8 + 2 * t4;
      const float v0 = y[4 * j + 2 * half], v1 = y[4 * j + 2 * half + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<__nv_bfloat162*>(srow + cl * 2) = __floats2bfloat162_rn(v0, v1);
      } else {
        *reinterpret_cast<float2*>(srow + cl * 4) = make_float2(v0, v1);
      }
    }
  }
  __syncwarp();
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

template <int KIND, int BM, int STAGES, int EPI>
__global__ void __launch_bounds__(W4Tile<KIND, BM, STAGES>::THREADS, 1)
    w4a8_wgmma_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                      const __grid_constant__ CUtensorMap tmSc, const __grid_constant__ CUtensorMap tmMn,
                      const __grid_constant__ CUtensorMap tmD, const __grid_constant__ CUtensorMap tmDm,
                      const __grid_constant__ CUtensorMap tmAs, const __grid_constant__ CUtensorMap tmXs,
                      void* __restrict__ out, int ldo, const int* __restrict__ n_rows, int M, int N, int K, int vec) {
#ifdef STARTLUX_PRISM_W4WG_DEVICE
  using T = W4Tile<KIND, BM, STAGES>;
  constexpr int BN = T::BN, C = T::C, E = T::E, P = T::P;
  const int live = min(*n_rows, M);
  int m0, n0;
  w4_tile_origin<BM>(M, N, m0, n0);
  if (m0 >= live) return;

  extern __shared__ uint8_t smem_raw[];
  const uint32_t base_u32 = (smem_u32(smem_raw) + 1023) & ~1023u;
  uint8_t* smem = smem_raw + (base_u32 - smem_u32(smem_raw));
  uint64_t* raw_full = reinterpret_cast<uint64_t*>(smem + T::BAR_OFF);   // [P] TMA bytes
  uint64_t* raw_empty = raw_full + P;                 // [P] every warp, once it expanded its share of the stage
  uint64_t* exp_full = raw_empty + P;                 // [E] every warp
  uint64_t* exp_empty = exp_full + E;                 // [E] each MMA warpgroup, once its MMAs read the tile
  uint64_t* a_full = exp_empty + E;                   // [CWGS][C] TMA bytes
  uint64_t* sb_full = a_full + T::CWGS * C;           // [2] TMA bytes
  uint64_t* sb_empty = sb_full + 2;                   // [2] each MMA warpgroup, after the floating-point step
  const int tid = threadIdx.x, wg = tid / 128;
  const int k_tiles = K / W4_BK, n_sb = K / 256;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < P; ++i) {
      mbar_init(&raw_full[i], 1);
      mbar_init(&raw_empty[i], T::WARPS);
    }
#pragma unroll
    for (int i = 0; i < E; ++i) {
      mbar_init(&exp_full[i], T::WARPS);
      mbar_init(&exp_empty[i], T::CWGS);
    }
#pragma unroll
    for (int i = 0; i < T::CWGS * C; ++i) mbar_init(&a_full[i], 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      mbar_init(&sb_full[i], 1);
      mbar_init(&sb_empty[i], T::CWGS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if constexpr (KIND == 0) {   // the mins operands' k8..31 stay zero
    constexpr int Z = (2 * T::MIN_A + T::MIN_B) / 16;
    for (int i = tid; i < 2 * Z; i += T::THREADS)
      reinterpret_cast<uint4*>(smem + T::SB_OFF + (i / Z) * T::SB_BYTES + T::SB_MIN)[i % Z] = make_uint4(0, 0, 0, 0);
    fence_async_shared();
  }
  __syncthreads();

  // a warp's share of stage kt: `iters` x 8 rows from r0 (lane / 4 the row, lane % 4 the k32: conflict-free reads and
  // writes), into the stage's expanded slot once the MMAs released it; then, after any other writes, its marks
  auto expand_rows = [&](int kt, int r0, int iters, int lane) {
    const int p = kt % P, e = kt % E;
    const uint8_t* rs = smem + T::RAW_OFF + p * T::RAW_BYTES;
    uint8_t* ex = smem + T::EXP_OFF + e * T::EXP_BYTES;
    mbar_wait(&raw_full[p], (kt / P) & 1);
    mbar_wait(&exp_empty[e], ((kt / E) & 1) ^ 1);   // a fresh barrier's preceding phase counts as completed
#pragma unroll 1
    for (int it = 0; it < iters; ++it) expand_item<KIND>(rs, ex, r0 + it * 8 + lane / 4, lane % 4, kt & 1);
  };
  auto expanded = [&](int kt, int lane) {
    fence_async_shared();   // the generic-proxy writes, seen by wgmma
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(&exp_full[kt % E]);
      mbar_arrive(&raw_empty[kt % P]);
    }
  };

  if (wg == 0) {  // ------------------------------------------------------------------------------------ producer
    if constexpr (T::CWGS == 2) setmaxnreg_dec<40>();
    const int pt = tid, pw = pt / 32, lane = pt % 32;
    auto issue_raw = [&](int kt) {   // thread 0: the packed weights and sub-block scales of stage kt
      const int p = kt % P;
      uint8_t* rs = smem + T::RAW_OFF + p * T::RAW_BYTES;
      mbar_wait(&raw_empty[p], ((kt / P) & 1) ^ 1);   // a fresh barrier's preceding phase counts as completed
      mbar_expect_tx(&raw_full[p], T::TX_RAW);
      tma_load_2d(rs, &tmW, KIND == 0 ? kt * (W4_BK / 2) : kt * W4_BK, n0, &raw_full[p]);
      tma_load_2d(rs + T::RAW_W, &tmSc, KIND == 0 ? n0 : 2 * n0, kt >> 1, &raw_full[p]);
    };
    auto issue_sb = [&](int b) {     // thread 0: super-block b's values for the floating-point step
      const int q = b & 1;
      uint8_t* sb = smem + T::SB_OFF + q * T::SB_BYTES;
      mbar_wait(&sb_empty[q], ((b >> 1) & 1) ^ 1);
      mbar_expect_tx(&sb_full[q], T::TX_SB);
      tma_load_2d(sb + T::SB_D, &tmD, n0, b, &sb_full[q]);
      tma_load_2d(sb + T::SB_AS, &tmAs, m0, b, &sb_full[q]);
      if constexpr (KIND == 0) {
        tma_load_2d(sb + T::SB_DM, &tmDm, n0, b, &sb_full[q]);
        tma_load_2d(sb + T::SB_MN, &tmMn, n0, b, &sb_full[q]);
        tma_load_2d(sb + T::SB_XS, &tmXs, 2 * m0, b, &sb_full[q]);
      }
    };
    if (pt == 0) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&tmW) : "memory");
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&tmSc) : "memory");
      for (int kt = 0; kt < P - 1 && kt < k_tiles; ++kt) issue_raw(kt);
      issue_sb(0);
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
      const int half = kt & 1, b = kt >> 1;
      expand_rows(kt, pw * (T::PROD_ROWS / 4), T::PROD_ROWS / 32, lane);
      if constexpr (KIND == 0) {
        if (half) {   // the super-block's mins operands: A1 / A2 rows (S_lo / S_hi of a token), B rows (m of a column)
          uint8_t* sb = smem + T::SB_OFF + (b & 1) * T::SB_BYTES;
          uint8_t* mo = sb + T::SB_MIN;
          mbar_wait(&sb_full[b & 1], (b >> 1) & 1);
          const int cm = (pt / 8) * MIN_SBO + (pt % 8) * 16;
          if (pt < BM) {
            const uint4 xs = *reinterpret_cast<const uint4*>(sb + T::SB_XS + pt * 16);
            *reinterpret_cast<uint2*>(mo + cm) = make_uint2(xs.x, xs.y);
            *reinterpret_cast<uint2*>(mo + T::MIN_A + cm) = make_uint2(xs.z, xs.w);
          }
          *reinterpret_cast<uint2*>(mo + 2 * T::MIN_A + cm) = *reinterpret_cast<const uint2*>(sb + T::SB_MN + pt * 8);
        }
      }
      expanded(kt, lane);
      if (pt == 0) {
        if (kt + P - 1 < k_tiles) issue_raw(kt + P - 1);
        if (half && b + 1 < n_sb) issue_sb(b + 1);
      }
    }
  } else {  // ----------------------------------------------------------------------------------- MMA warpgroups
    if constexpr (T::CWGS == 2) setmaxnreg_inc<232>();
    const int cw = wg - 1, t = tid % 128;
    const int w = t / 32, lane = t % 32, g = lane / 4, t4 = lane % 4;
    uint8_t* abase = smem + T::A_OFF + cw * C * T::A_BYTES;
    uint64_t* af = a_full + cw * C;
    auto issue_a = [&](int kt) {   // thread 0: this warpgroup's rows of stage kt (the slot's MMAs completed)
      const int s = kt % C;
      mbar_expect_tx(&af[s], T::A_BYTES);
      tma_load_2d(abase + s * T::A_BYTES, &tmA, kt * W4_BK, m0 + cw * 64, &af[s]);
    };
    if (t == 0) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&tmA) : "memory");
      for (int kt = 0; kt < C && kt < k_tiles; ++kt) issue_a(kt);
    }
    const int rows = T::PROD_ROWS + 32 * cw + 8 * w;   // its warps' share of every stage: 8 rows each
    expand_rows(0, rows, 1, lane);
    expanded(0, lane);
    float y[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) y[i] = 0.f;
    int32_t acc[128];   // lo sums, then hi sums
    for (int kt = 0; kt < k_tiles; kt += 2) {   // a super-block: two stages
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int k = kt + h, s = k % C, e = k % E;
        mbar_wait(&af[s], (k / C) & 1);
        mbar_wait(&exp_full[e], (k / E) & 1);
        const uint64_t da = sw128_desc(smem_u32(abase + s * T::A_BYTES));
        const uint64_t db = sw128_desc(smem_u32(smem + T::EXP_OFF + e * T::EXP_BYTES));
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < W4_BK / 32; ++kk) mma_n256(acc, da + 2 * kk, db + 2 * kk, h == 0 && kk == 0);
        wg_commit();
        if (k + 1 < k_tiles) {   // its share of the next stage, while the tensor cores work
          expand_rows(k + 1, rows, 1, lane);
          expanded(k + 1, lane);
        }
      }
      wg_wait<1>();   // the first stage's products are done: its tiles go back
      fence_operands(acc);
      if (t == 0) {
        mbar_arrive(&exp_empty[kt % E]);
        if (kt + C < k_tiles) issue_a(kt + C);
      }
      wg_wait<0>();
      fence_operands(acc);
      if (t == 0) {
        mbar_arrive(&exp_empty[(kt + 1) % E]);
        if (kt + 1 + C < k_tiles) issue_a(kt + 1 + C);
      }
      // the floating-point step: per output t = d * float(acc) [Q4_K: t = fma(-dmin, float(mins), t)], y = fma(a, t, y)
      const int b = kt >> 1, q = b & 1;
      uint8_t* sb = smem + T::SB_OFF + q * T::SB_BYTES;
      mbar_wait(&sb_full[q], (b >> 1) & 1);
      const float* sd = reinterpret_cast<const float*>(sb + T::SB_D);
      const float* sa = reinterpret_cast<const float*>(sb + T::SB_AS);
      const float a0 = sa[cw * 64 + w * 16 + g], a1 = sa[cw * 64 + w * 16 + g + 8];
      float tv[64];
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const float2 dd = *reinterpret_cast<const float2*>(sd + j * 8 + 2 * t4);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int i = 4 * j + r;
          const float f = __int2float_rn(acc[i] + (acc[64 + i] << (KIND == 0 ? 3 : 6)));   // the exact int32 sum
          tv[i] = __fmul_rn(r & 1 ? dd.y : dd.x, f);
        }
      }
      if constexpr (KIND == 0) {
        const float* sdm = reinterpret_cast<const float*>(sb + T::SB_DM);
        uint8_t* mo = sb + T::SB_MIN;
        const uint64_t a1d = core_desc(smem_u32(mo + cw * 8 * MIN_SBO));
        const uint64_t a2d = core_desc(smem_u32(mo + T::MIN_A + cw * 8 * MIN_SBO));
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {   // 64 columns at a time
          const uint64_t bd = core_desc(smem_u32(mo + 2 * T::MIN_A + nh * 8 * MIN_SBO));
          int32_t ml[32], mh[32];
          wg_fence();
          mma_n64_first(ml, a1d, bd);
          mma_n64_first(mh, a2d, bd);
          wg_commit();
          wg_wait<0>();
          fence_operands(ml);
          fence_operands(mh);
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) {
            const float2 dm = *reinterpret_cast<const float2*>(sdm + (nh * 8 + jj) * 8 + 2 * t4);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int i = 32 * nh + 4 * jj + r;
              const float mn = __int2float_rn(ml[4 * jj + r] + (mh[4 * jj + r] << 6));   // exact: below 2^24
              tv[i] = __fmaf_rn(-(r & 1 ? dm.y : dm.x), mn, tv[i]);
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 64; ++i) y[i] = __fmaf_rn((i & 2) ? a1 : a0, tv[i], y[i]);
      asm volatile("bar.sync %0, 128;\n" ::"r"(2 + cw) : "memory");   // the warpgroup read the super-block's values
      if (t == 0) mbar_arrive(&sb_empty[q]);
    }
    // the staging rows overwrite tiles the other warpgroup's last steps may still read
    if constexpr (T::CWGS > 1) asm volatile("bar.sync 1, %0;\n" ::"n"(T::CWGS * 128) : "memory");
    constexpr int ROWB = W4_BN * (EPI == 0 ? 2 : 4) + 16;
    w4_store<EPI>(y, out, ldo, live, N, m0 + cw * 64, n0, t, smem + cw * 64 * ROWB, vec != 0);
  }
#else
  __trap();
#endif
}

// TMA descriptors, encoded once per (address, element type, inner, outer, row bytes, box, swizzle); kernel arguments,
// so a captured CUDA graph keeps the ones of its capture
using W4MapKey = std::tuple<const void*, int, int64_t, int64_t, int64_t, int, int, int>;

static const CUtensorMap& w4_map(const void* ptr, CUtensorMapDataType type, int elem, int64_t inner, int64_t outer,
                                 int64_t row_bytes, int box_inner, int box_outer, CUtensorMapSwizzle swz) {
  static std::map<W4MapKey, CUtensorMap> cache;
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  auto key = W4MapKey(ptr, (int)type, inner, outer, row_bytes, box_inner, box_outer, (int)swz);
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
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {(cuuint32_t)box_inner, (cuuint32_t)box_outer};
  cuuint32_t es[2] = {1, 1};
  CUresult r = encode(&map, type, 2, const_cast<void*>(ptr), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                      swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "w4a8_wgmma: cuTensorMapEncodeTiled failed: ", (int)r, " (elem ", elem, ")");
  return cache.emplace(key, map).first->second;
}

template <int KIND, int BM, int STAGES, int EPI>
void w4_launch(const torch::Tensor& a, const torch::Tensor& as, const torch::Tensor& xs, const torch::Tensor& wq,
               const torch::Tensor& wsc, const torch::Tensor& wmn, const torch::Tensor& wd, const torch::Tensor& wdm,
               torch::Tensor& out, const torch::Tensor& n_rows) {
  using T = W4Tile<KIND, BM, STAGES>;
  constexpr int SMEM = T::SMEM;
  if constexpr (SMEM > W4_SMEM_MAX) {
    TORCH_CHECK(false, "w4a8_wgmma: the stages do not fit in shared memory");
  } else {
    const int M = a.size(0), K = a.size(1), N = wq.size(0), NB = K / 256;
    const auto U8 = CU_TENSOR_MAP_DATA_TYPE_UINT8, U64 = CU_TENSOR_MAP_DATA_TYPE_UINT64,
               F32 = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    const auto NONE = CU_TENSOR_MAP_SWIZZLE_NONE, SW128 = CU_TENSOR_MAP_SWIZZLE_128B;
    const CUtensorMap& mA = w4_map(a.data_ptr(), U8, 1, K, M, K, W4_BK, 64, SW128);   // a warpgroup's rows
    const CUtensorMap& mW = KIND == 0 ? w4_map(wq.data_ptr(), U8, 1, K / 2, N, K / 2, W4_BK / 2, W4_BN, NONE)
                                      : w4_map(wq.data_ptr(), U8, 1, K, N, K, W4_BK, W4_BN, SW128);
    const CUtensorMap& mSc = KIND == 0 ? w4_map(wsc.data_ptr(), U64, 8, N, NB, (int64_t)N * 8, W4_BN, 1, NONE)
                                       : w4_map(wsc.data_ptr(), U64, 8, 2 * N, NB, (int64_t)N * 16, 2 * W4_BN, 1, NONE);
    const CUtensorMap& mD = w4_map(wd.data_ptr(), F32, 4, N, NB, (int64_t)N * 4, W4_BN, 1, NONE);
    const CUtensorMap& mAs = w4_map(as.data_ptr(), F32, 4, M, NB, (int64_t)M * 4, BM, 1, NONE);
    const CUtensorMap& mMn = KIND == 0 ? w4_map(wmn.data_ptr(), U64, 8, N, NB, (int64_t)N * 8, W4_BN, 1, NONE) : mD;
    const CUtensorMap& mDm = KIND == 0 ? w4_map(wdm.data_ptr(), F32, 4, N, NB, (int64_t)N * 4, W4_BN, 1, NONE) : mD;
    const CUtensorMap& mXs = KIND == 0 ? w4_map(xs.data_ptr(), U64, 8, 2 * M, NB, (int64_t)M * 16, 2 * BM, 1, NONE)
                                       : mD;
    auto kern = w4a8_wgmma_kernel<KIND, BM, STAGES, EPI>;
    static bool configured = false;
    if (!configured) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      configured = true;
    }
    const int tiles = ((M + BM - 1) / BM) * ((N + W4_BN - 1) / W4_BN);
    const auto aligned = [](const torch::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0; };
    const int vec = aligned(out) && out.stride(0) * out.element_size() % 16 == 0;
    kern<<<tiles, T::THREADS, SMEM, c10::cuda::getCurrentCUDAStream()>>>(
        mA, mW, mSc, mMn, mD, mDm, mAs, mXs, out.data_ptr(), (int)out.stride(0), n_rows.data_ptr<int>(), M, N, K, vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

// bm: 128 (two MMA warpgroups) or 64 (one); stages 3
void w4a8_wgmma(torch::Tensor a, torch::Tensor as, torch::Tensor xs, torch::Tensor wq, torch::Tensor wsc,
                std::optional<torch::Tensor> wmn, torch::Tensor wd, std::optional<torch::Tensor> wdm, torch::Tensor out,
                torch::Tensor n_rows, int64_t kind, int64_t epi, int64_t bm, int64_t stages) {
#ifndef STARTLUX_PRISM_SM90A
  TORCH_CHECK(false, "w4a8_wgmma: the extension was built without sm_90a (STARTLUX_PRISM_CUDA_ARCH=90a)");
#endif
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9, "w4a8_wgmma: Hopper (sm_90) only");
  const int64_t M = a.size(0), K = a.size(1), N = wq.size(0);
  TORCH_CHECK(a.dtype() == torch::kInt8 && a.stride(1) == 1 && a.stride(0) == K, "contiguous int8 A");
  TORCH_CHECK(K % 256 == 0 && M % 4 == 0 && N % 4 == 0, "w4a8_wgmma: K a multiple of 256, M and N of 4 (TMA rows)");
  TORCH_CHECK(kind == 1 || (wmn.has_value() && wdm.has_value()), "Q4_K needs mins and dmin");
  TORCH_CHECK(epi == 0 ? out.dtype() == torch::kBFloat16 : out.dtype() == torch::kFloat32, "output dtype");
  const auto aligned = [](const torch::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0; };
  TORCH_CHECK(aligned(a) && aligned(as) && aligned(wq) && aligned(wsc) && aligned(wd) &&
                  (kind == 1 || (aligned(xs) && aligned(*wmn) && aligned(*wdm))),
              "w4a8_wgmma: TMA needs 16-byte aligned operands");
  TORCH_CHECK(wq.is_contiguous() && wsc.is_contiguous() && wd.is_contiguous() && as.is_contiguous(),
              "w4a8_wgmma: contiguous operands");
  const torch::Tensor& mn = kind == 0 ? *wmn : wd;
  const torch::Tensor& dm = kind == 0 ? *wdm : wd;
#define W4WG_STAGES(KD, BM, E) w4_launch<KD, BM, 3, E>(a, as, xs, wq, wsc, mn, wd, dm, out, n_rows);
#define W4WG_BM(KD, E)                 \
  if (bm == 128) {                     \
    W4WG_STAGES(KD, 128, E)            \
  } else {                             \
    W4WG_STAGES(KD, 64, E)             \
  }
  TORCH_CHECK(bm == 64 || bm == 128, "w4a8_wgmma: bm 64 or 128");
  TORCH_CHECK(stages == 3, "w4a8_wgmma: 3 stages (the MMA warpgroups expand a stage ahead; 4 exceed the shared "
                           "memory)");
  if (kind == 0) {
    if (epi == 0) { W4WG_BM(0, 0) } else { W4WG_BM(0, 1) }
  } else {
    if (epi == 0) { W4WG_BM(1, 0) } else { W4WG_BM(1, 1) }
  }
#undef W4WG_BM
#undef W4WG_STAGES
}

TORCH_LIBRARY_FRAGMENT(startlux_prism, m) {
  m.def("w4a8_wgmma(Tensor a, Tensor a_s, Tensor xs, Tensor wq, Tensor wsc, Tensor? wmn, Tensor wd, Tensor? wdm, "
        "Tensor(a!) out, Tensor n_rows, int kind, int epi, int bm, int stages) -> ()");
}
TORCH_LIBRARY_IMPL(startlux_prism, CUDA, m) { m.impl("w4a8_wgmma", &w4a8_wgmma); }
