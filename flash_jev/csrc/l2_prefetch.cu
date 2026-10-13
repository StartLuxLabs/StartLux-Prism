// Weight prefetch into L2: up to four byte ranges, every 128-byte line touched once with `prefetch.global.L2::evict_last`
// (fire-and-forget: no register result, no L1 allocation).  Run on a side stream while latency-bound kernels leave
// DRAM idle, so the projections that follow read their weights from L2 instead of DRAM.
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__global__ void l2_prefetch_kernel(const char* p0, int64_t n0, const char* p1, int64_t n1, const char* p2, int64_t n2,
                                   const char* p3, int64_t n3) {
  const int64_t l0 = (n0 + 127) >> 7, l1 = (n1 + 127) >> 7, l2 = (n2 + 127) >> 7, l3 = (n3 + 127) >> 7;
  const int64_t total = l0 + l1 + l2 + l3;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const char* p;
    if (i < l0) {
      p = p0 + (i << 7);
    } else if (i < l0 + l1) {
      p = p1 + ((i - l0) << 7);
    } else if (i < l0 + l1 + l2) {
      p = p2 + ((i - l0 - l1) << 7);
    } else {
      p = p3 + ((i - l0 - l1 - l2) << 7);
    }
    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
  }
}

}  // namespace

// tensors: up to four (contiguous) tensors whose bytes go to L2; ctas x 256 threads issue the prefetches
void l2_prefetch(std::vector<torch::Tensor> tensors, int64_t ctas) {
  TORCH_CHECK(tensors.size() <= 4, "at most four ranges");
  const char* p[4] = {nullptr, nullptr, nullptr, nullptr};
  int64_t n[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < tensors.size(); ++i) {
    TORCH_CHECK(tensors[i].is_contiguous(), "contiguous tensors");
    p[i] = static_cast<const char*>(tensors[i].data_ptr());
    n[i] = tensors[i].numel() * tensors[i].element_size();
  }
  l2_prefetch_kernel<<<ctas, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(p[0], n[0], p[1], n[1], p[2], n[2], p[3],
                                                                           n[3]);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(flash_jev, m) { m.def("l2_prefetch(Tensor[] tensors, int ctas) -> ()"); }
TORCH_LIBRARY_IMPL(flash_jev, CUDA, m) { m.impl("l2_prefetch", &l2_prefetch); }
