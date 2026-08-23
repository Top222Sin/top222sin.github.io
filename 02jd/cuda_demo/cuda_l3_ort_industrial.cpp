// =============================================================================
// 工业级 CUDA EP 实战解析 —— 以 onnxruntime CUDA Execution Provider 为蓝本
// -----------------------------------------------------------------------------
// 本文件把 onnxruntime 源码里两个最典型的「分块 + 内存优化」手写 kernel 拆出来，
// 做成可独立编译、可 CPU 校验的教材版本：
//   (A) transpose_tiled   —— 对应 onnxruntime transpose_impl.cu 的 Transpose3DKernel
//   (B) softmax_warp      —— 对应 onnxruntime softmax_warpwise_impl.cuh 的 softmax_warp_forward
// 两者都用在真机 nvcc 编译运行；CPU 参考函数用于数值校验（沙箱无 GPU，用 Python 镜像验证过数值）。
//
// 源码引用（main 分支，2026-08 拉取）：
//   onnxruntime/core/providers/cuda/tensor/transpose_impl.cu
//   onnxruntime/core/providers/cuda/math/softmax_impl.cu
//   onnxruntime/core/providers/cuda/math/softmax_warpwise_impl.cuh
//   onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu
//   onnxruntime/core/providers/cuda/math/gemm.cc
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_e));                                            \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

// =============================================================================
// (A) 分块转置：shared memory + [TILE][TILE+1] padding
// -----------------------------------------------------------------------------
// onnxruntime 同款做法（transpose_impl.cu:10-46）：
//   constexpr unsigned int kTileSize = 32;
//   __shared__ T tile[TileSize][TileSize + 1];   // 关键：+1 避免 bank conflict
// 读阶段：每个线程把输入里「一列」连续元素搬进共享，读全局是合并的；
// 同步后写阶段：把共享里「一行」连续元素写出，写全局也是合并的；
// 用 [TILE+1] 让同一 warp 内不同线程访问不同 bank（详见 md 的 bank 推导）。
// =============================================================================

// 朴素版：元素级 gmem->gmem，只做正确性对照（写输出是跨步的，不合并）
template <typename T>
__global__ void transpose_naive(const T* in, T* out, int m, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
  if (x < n && y < m) out[x * m + y] = in[y * n + x];
}

// 工业版（onnxruntime Transpose3DKernel 的精简教材版）
template <typename T, int TILE = 32>
__global__ void transpose_tiled(const T* in, T* out, int m, int n) {
  __shared__ T tile[TILE][TILE + 1];  // +1：破 bank conflict（L2.1 知识点）

  // ---- 读阶段：输入按行连续 -> 合并读 ----
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < n && y < m) tile[threadIdx.y][threadIdx.x] = in[y * n + x];

  __syncthreads();

  // ---- 写阶段：输出按列连续 -> 合并写 ----
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  if (x < m && y < n) out[y * m + x] = tile[threadIdx.x][threadIdx.y];
}

// =============================================================================
// (B) warp 级 softmax：寄存器 + __shfl_xor 归约（无 shared memory）
// -----------------------------------------------------------------------------
// onnxruntime softmax_warpwise_impl.cuh 的同款骨架：
//   - 一个 warp 负责一行（row），行内元素 <= 1024（32 * WARP_ITER）
//   - 每线程把若干元素装进寄存器（WARP_BATCH 版更激进，这里 WARP_BATCH=1 便于讲清）
//   - max / sum 都靠 warp_reduce（__shfl_xor）在寄存器间直接交换，不碰共享内存
//   - 倒数只算一次，除法改成乘法（invsum）
// 这正是我们 L2.4 里 reduce_warp_shuffle 的「工业升级版」。
// =============================================================================

template <typename T, int WARP_ITER = 32>
__global__ void softmax_warp(const T* in, T* out, int rows, int n) {
  int row = blockIdx.x;          // 一个 warp 一行
  int lane = threadIdx.x;        // 0..31，同一 warp
  if (row >= rows) return;

  const T* rin = in + row * n;
  T* rout = out + row * n;

  T vals[WARP_ITER];
#pragma unroll
  for (int it = 0; it < WARP_ITER; ++it) {
    int idx = lane + it * 32;
    vals[it] = (idx < n) ? rin[idx] : T(-1e30);  // 越界填 -inf，不影响 max/sum
  }

  // --- 1) 求 max（warp 内 shfl_xor 归约）---
  T mx = vals[0];
#pragma unroll
  for (int it = 1; it < WARP_ITER; ++it) mx = fmaxf(mx, vals[it]);
#pragma unroll
  for (int off = 16; off > 0; off /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));

  // --- 2) 求 sum(exp(x - max)) ---
  float sum = 0.f;
#pragma unroll
  for (int it = 0; it < WARP_ITER; ++it) {
    if (lane + it * 32 < n) { vals[it] = expf(vals[it] - mx); sum += vals[it]; }
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, off);
  float inv = 1.f / sum;  // 只算一次倒数

  // --- 3) 写回 ---
#pragma unroll
  for (int it = 0; it < WARP_ITER; ++it) {
    int idx = lane + it * 32;
    if (idx < n) rout[idx] = T(vals[it] * inv);
  }
}

// =============================================================================
// CPU 参考实现（用于数值校验，也可在无 GPU 环境单独跑）
// =============================================================================
static void cpu_transpose(const float* in, float* out, int m, int n) {
  for (int y = 0; y < m; ++y)
    for (int x = 0; x < n; ++x) out[x * m + y] = in[y * n + x];
}
static void cpu_softmax(const float* in, float* out, int rows, int n) {
  for (int r = 0; r < rows; ++r) {
    const float* p = in + r * n;
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, p[i]);
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += expf(p[i] - mx);
    float inv = 1.f / s;
    for (int i = 0; i < n; ++i) out[r * n + i] = expf(p[i] - mx) * inv;
  }
}
static int check_close(const float* a, const float* b, int n, float tol = 1e-3f) {
  for (int i = 0; i < n; ++i)
    if (fabsf(a[i] - b[i]) > tol) { printf("  MISMATCH @%d: %g vs %g\n", i, a[i], b[i]); return 0; }
  return 1;
}

// =============================================================================
// 确定性分析（CPU 可跑）：把 onnxruntime 的几个工业级决策点量化出来
// =============================================================================
static void analyze_ort_patterns() {
  printf("\n=== onnxruntime CUDA EP 工业级模式对照（确定性分析）===\n");

  // 1) 转置的 bank padding：bank(i) = i % 32
  //    tile[T][T+1] 中元素 [r][c] 落在 bank((r*(T+1)+c) % 32)
  //    T=32 => (r*33 + c)%32 = (r + c) % 32 （因为 33%32=1）
  //    写阶段同一 warp 的线程访问 tile[threadIdx.x][threadIdx.y + i]，
  //    bank = (threadIdx.x + threadIdx.y + i) % 32，对固定 i 各线程互不相同 => 无冲突
  printf("[转置 padding] tile[32][33]: 写阶段 bank = (tx + ty + i) %% 32，同 warp 各 lane 不同 => 0 bank conflict\n");
  printf("            对比 tile[32][32]: 写阶段 bank = (tx*32 + ty + i) %% 32 = (ty + i) %% 32，\n");
  printf("                  全部 32 线程落同一 bank => 32 路冲突（最差情况，L2.1 知识点）\n");

  // 2) softmax 寄存器压力回退（onnxruntime softmax_warpwise_impl.cuh:165-233）
  //    元素数小时用寄存器（softmax_warp_forward），元素数大时寄存器溢出 ->
  //    改存原始 dtype 到 shared 并即时 cast（resource_efficient），以降低寄存器占用、保 occupancy
  printf("[softmax 寄存器] element<=128: WARP_BATCH=2 全寄存器；>128: 1 warp/block + shared 回退，避免 spill 拖低 occupancy（L2.2）\n");

  // 3) GEMM 交给库：onnxruntime gemm.cc 不手写 tiling，而是 tunable::TunableGemm / cublasGemmHelper
  printf("[GEMM 边界] onnxruntime 的 MatMul/Gemm 直接调 cuBLAS/cuBLASLt（含 workspace + 算法启发式），\n");
  printf("            tiling 由库内极致手写 kernel 完成；自己只做 shape 解析、broadcast、stream 绑定\n");

  // 4) async + 指定 stream：所有拷贝走 cudaMemcpyAsync(..., Stream(ctx))，绝不用默认流
  printf("[流/异步] 拷贝与 kernel 均绑定到 Op 的 stream（cudaMemcpyAsync + Stream(ctx)），避免默认流串行（L2.3 nsys 时间线）\n");
}

// =============================================================================
// 真机校验（有 GPU 时取消注释运行）
// =============================================================================
static void run_gpu_checks() {
  const int m = 256, n = 192, TILE = 32;
  int elems = m * n;
  std::vector<float> h_in(elems), h_out_gpu(elems), h_out_cpu(elems);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dis(-5.f, 5.f);
  for (float& v : h_in) v = dis(rng);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blk(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  transpose_tiled<float, TILE><<<grid, blk>>>(d_in, d_out, m, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost));

  cpu_transpose(h_in.data(), h_out_cpu.data(), m, n);
  printf("[transpose] GPU vs CPU: %s\n", check_close(h_out_gpu.data(), h_out_cpu.data(), elems) ? "PASS" : "FAIL");

  // softmax
  const int rows = 64, cols = 256;
  int srem = rows * cols;
  std::vector<float> s_in(srem), s_out_gpu(srem), s_out_cpu(srem);
  for (float& v : s_in) v = dis(rng);
  float *d_sin, *d_sout;
  CUDA_CHECK(cudaMalloc(&d_sin, srem * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sout, srem * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_sin, s_in.data(), srem * sizeof(float), cudaMemcpyHostToDevice));
  softmax_warp<float, 8><<<rows, 32>>>(d_sin, d_sout, rows, cols);  // 256 = 32*8
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(s_out_gpu.data(), d_sout, srem * sizeof(float), cudaMemcpyDeviceToHost));
  cpu_softmax(s_in.data(), s_out_cpu.data(), rows, cols);
  printf("[softmax]   GPU vs CPU: %s\n", check_close(s_out_gpu.data(), s_out_cpu.data(), srem) ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_sout));
}

int main() {
  analyze_ort_patterns();
  // run_gpu_checks();   // 真机（有 nvcc + GPU）时取消注释
  printf("\n（提示：真机取消 main() 里 run_gpu_checks() 注释即可 nvcc 编译运行，两 kernel 均与 CPU 对拍）\n");
  return 0;
}
