// Tensor-core peak of mma.sync on this GPU: every warp issues independent MMAs on register operands (no memory
// traffic), 8 accumulator chains per warp.  Reports TOPS for int8 (m16n8k32, int32 acc), fp8 e4m3 (m16n8k32, fp32
// acc), bf16 (m16n8k16, fp32 acc) and fp16 (m16n8k16, fp16 acc).
//
//   nvcc -O3 -gencode=arch=compute_120,code=sm_120 bench/csrc/mma_peak.cu -o mma_peak && ./mma_peak
//   ./mma_peak sustain 5        # int8 only, for 5 s: the rate at the clock the GPU holds under load
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

constexpr int CHAINS = 8, ITERS = 4096;

__global__ void k_int8(int* out) {
  int c[CHAINS][4] = {};
  unsigned a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u}, b[2] = {threadIdx.x * 11u, 13u};
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  int s = 0;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 0x12345) out[0] = s;
}

__global__ void k_fp8(float* out) {
  float c[CHAINS][4] = {};
  unsigned a[4] = {0x3c3c3c3cu, 0x3c3c3c3cu, 0x3c3c3c3cu, 0x3c3c3c3cu}, b[2] = {0x3c3c3c3cu, 0x3c3c3c3cu};
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  float s = 0;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1.2345f) out[0] = s;
}

__global__ void k_bf16(float* out) {
  float c[CHAINS][4] = {};
  unsigned a[4] = {0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u}, b[2] = {0x3f803f80u, 0x3f803f80u};
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  float s = 0;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1.2345f) out[0] = s;
}

__global__ void k_fp16acc(float* out) {
  unsigned c[CHAINS][2] = {};
  unsigned a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u}, b[2] = {0x3c003c00u, 0x3c003c00u};
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                   : "+r"(c[j][0]), "+r"(c[j][1])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  unsigned s = 0;
  for (int j = 0; j < CHAINS; ++j) s += c[j][0] + c[j][1];
  if (s == 0x12345u) out[0] = (float)s;
}

int main(int argc, char** argv) {
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int *oi;
  float* of;
  cudaMalloc(&oi, 64);
  cudaMalloc(&of, 64);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  if (argc > 2 && std::string(argv[1]) == "sustain") {  // int8 at 8 warps/block for argv[2] seconds: the sustained rate
    dim3 grid(sms * 8), block(32 * 8);
    double mmas = (double)grid.x * 8 * ITERS * CHAINS, total_ms = 0, n = 0;
    while (total_ms < atof(argv[2]) * 1000) {
      float ms;
      cudaEventRecord(e0);
      for (int i = 0; i < 20; ++i) k_int8<<<grid, block>>>(oi);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      total_ms += ms;
      n += 20;
    }
    printf("sustained int8 %.1f TOPS over %.1f s\n", mmas * n * 16 * 8 * 32 * 2 / (total_ms * 1e-3) / 1e12,
           total_ms / 1000);
    return 0;
  }
  for (int warps : {4, 8, 16}) {
    dim3 grid(sms * 8), block(32 * warps);
    double mmas = (double)grid.x * warps * ITERS * CHAINS;
    for (int rep = 0; rep < 2; ++rep) {
      float ms;
      k_int8<<<grid, block>>>(oi);
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) k_int8<<<grid, block>>>(oi);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      double t_int8 = mmas * 5 * 16 * 8 * 32 * 2 / (ms * 1e-3) / 1e12;
      k_fp8<<<grid, block>>>(of);
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) k_fp8<<<grid, block>>>(of);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      double t_fp8 = mmas * 5 * 16 * 8 * 32 * 2 / (ms * 1e-3) / 1e12;
      k_bf16<<<grid, block>>>(of);
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) k_bf16<<<grid, block>>>(of);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      double t_bf16 = mmas * 5 * 16 * 8 * 16 * 2 / (ms * 1e-3) / 1e12;
      k_fp16acc<<<grid, block>>>(of);
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) k_fp16acc<<<grid, block>>>(of);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&ms, e0, e1);
      double t_f16 = mmas * 5 * 16 * 8 * 16 * 2 / (ms * 1e-3) / 1e12;
      int clk;
      cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
      printf("warps/block %2d: int8 %6.1f TOPS  fp8(f32 acc) %6.1f  bf16(f32 acc) %6.1f  fp16(f16 acc) %6.1f TFLOPS\n",
             warps, t_int8, t_fp8, t_bf16, t_f16);
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
  return 0;
}
