// Read bandwidth of L2 (a buffer that fits in it, read over and over) and of DRAM (one much larger than L2), with
// 16-byte loads from every SM.  Bounds what a GEMM can stream into shared memory per tile byte.
//
//   nvcc -O3 -gencode=arch=compute_120,code=sm_120 bench/csrc/l2_bw.cu -o l2_bw && ./l2_bw
#include <cstdio>
#include <cuda_runtime.h>

__global__ void reader(const int4* __restrict__ p, size_t n16, int reps, int* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < reps; ++r)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += stride) {
      int4 v = __ldcg(p + i);  // L2 only (no L1 allocation)
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7fffffff) out[0] = 1;
}

int main() {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int* out;
  cudaMalloc(&out, 4);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (size_t mb : {8, 16, 32, 48, 64, 1024}) {
    size_t bytes = mb << 20;
    int4* p;
    cudaMalloc(&p, bytes);
    cudaMemset(p, 1, bytes);
    int reps = mb >= 1024 ? 2 : 50;
    for (int blocks_per_sm : {2, 4, 8}) {
      dim3 grid(sms * blocks_per_sm), block(512);
      reader<<<grid, block>>>(p, bytes / 16, 1, out);
      cudaEventRecord(e0);
      reader<<<grid, block>>>(p, bytes / 16, reps, out);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      printf("%5zu MB buffer, %d CTAs/SM: %7.2f TB/s\n", mb, blocks_per_sm, (double)bytes * reps / (ms * 1e-3) / 1e12);
    }
    cudaFree(p);
  }
  return 0;
}
