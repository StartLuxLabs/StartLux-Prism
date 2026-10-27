// MXFP4 GEMM on Blackwell (SM120/121) through CUTLASS 4.8's block-scaled mma.sync, callable from
// Python by ctypes.  A and B data are row-major and K-contiguous, as stored by kernels/mxfp4.py;
// only the e8m0 scales move to CUTLASS's interleaved layout (B once, A per call).
#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace mxfp4_launcher {

using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentAB = 32;

template <class ElementC, class ElementD>
struct Kernel {
  using ThreadBlockShape = Shape<_128, _128, _128>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ThreadBlockShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutDTag, 128 / cutlass::sizeof_bits<ElementD>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentAB,
      ElementB, LayoutBTag, AlignmentAB,
      ElementAccumulator, ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using KernelT = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                      CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<KernelT>;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

template <class T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

__device__ __forceinline__ float e2m1_value(int c) {
  const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float x = v[c & 7];
  return (c & 8) ? -x : x;
}

template <class TensorSF>
__global__ void repack_sf_kernel(TensorSF t, const uint8_t* s, int MN, int KS) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= MN * KS) return;
  int mn = idx / KS, ks = idx % KS;
  t(mn, ks * 32, 0) = typename TensorSF::value_type(ldexpf(1.0f, int(s[idx]) - 127));
}

template <class ElementC, class ElementD, class KernelT>
int run(const void* aq, const void* as, const void* bq, const void* bs, void* out,
        int M, int N, int K, float alpha, float beta, cudaStream_t stream) {
  using Gemm = typename KernelT::Gemm;
  using LayoutSFA = typename KernelT::LayoutSFA;
  using LayoutSFB = typename KernelT::LayoutSFB;
  using Sm1xx = typename KernelT::Sm1xxBlkScaledConfig;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  LayoutSFA layout_SFA = Sm1xx::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA::DataType const*>(aq), stride_A,
       reinterpret_cast<ElementB::DataType const*>(bq), stride_B,
       reinterpret_cast<ElementA::ScaleFactorType const*>(as), layout_SFA,
       reinterpret_cast<ElementB::ScaleFactorType const*>(bs), layout_SFB},
      {{alpha, beta}, reinterpret_cast<ElementC*>(out), stride_C, reinterpret_cast<ElementD*>(out), stride_D}};

  static void* workspace = nullptr;
  static size_t workspace_size = 0;
  size_t need = Gemm::get_workspace_size(args);
  if (need > workspace_size) {
    if (workspace) cudaFree(workspace);
    cudaMalloc(&workspace, need);
    workspace_size = need;
  }
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 2;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return 3;
  return 0;
}

}  // namespace mxfp4_launcher

using namespace mxfp4_launcher;

extern "C" {

long mxfp4_sfb_bytes(int N, int K) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  auto layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
  return (long)size(filter_zeros(layout_SFB));
}

void mxfp4_prepare_sfa(const void* s, void* dst, int M, int K, cudaStream_t stream) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xx::tile_atom_to_shape_SFA(make_shape(M, 1, K, 1));
  auto tS = make_tensor(reinterpret_cast<ElementA::ScaleFactorType*>(dst), layout_SFA);
  int n = M * (K / 32);
  repack_sf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(tS, (const uint8_t*)s, M, K / 32);
}

long mxfp4_sfa_bytes(int M, int K) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xx::tile_atom_to_shape_SFA(make_shape(M, 1, K, 1));
  return (long)size(filter_zeros(layout_SFA));
}

void mxfp4_prepare_sfb(const void* s, void* dst, int N, int K, cudaStream_t stream) {
  using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t>;
  using Sm1xx = typename K0::Sm1xxBlkScaledConfig;
  auto layout_SFB = Sm1xx::tile_atom_to_shape_SFB(make_shape(128, N, K, 1));
  auto tS = make_tensor(reinterpret_cast<ElementB::ScaleFactorType*>(dst), layout_SFB);
  int n = N * (K / 32);
  repack_sf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(tS, (const uint8_t*)s, N, K / 32);
}

int mxfp4_gemm(const void* aq, const void* as, const void* bq, const void* bs, void* out,
                int M, int N, int K, int epi, cudaStream_t stream) {
  if (epi == 0) {
    using K0 = Kernel<cutlass::bfloat16_t, cutlass::bfloat16_t>;
    return run<cutlass::bfloat16_t, cutlass::bfloat16_t, K0>(aq, as, bq, bs, out, M, N, K, 1.f, 0.f, stream);
  }
  if (epi == 1) {
    using K1 = Kernel<float, float>;
    return run<float, float, K1>(aq, as, bq, bs, out, M, N, K, 1.f, 1.f, stream);
  }
  return 4;
}

}  // extern "C"
