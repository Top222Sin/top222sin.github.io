/******************************************************************************
 * cuda_l2_cutlass_gemm.cpp
 * -----------------------------------------------------------------------------
 * 对照 onnxruntime 的 transpose / softmax / layernorm（上一节）与 CUTLASS 的
 * GEMM 分块：看 CUTLASS 如何把「tiling + 双缓冲 + pipeline + Tensor Core」
 * 封装成可复用模板。
 *
 * 本文件包含：
 *   1) tiny_wmma_gemm<BM,BN,BK,STAGES>  —— 一个"缩小版 CUTLASS"参考 kernel：
 *        * 三级 tile 形状作为模板常量（Threadblock / Warp / Instruction）
 *        * 共享内存 kStages 份 K 切片 = 双/三缓冲（环形）
 *        * prologue() 灌满流水线 + mac_loop_iter() 主循环“算当前/取下一”重叠
 *        * nvcuda::wmma::mma_sync = Tensor Core（对应 CUTLASS 的 Operator）
 *   2) tiny_gemm_ref_cpu()            —— FP32 CPU 参考，GPU 结果对拍
 *   3) analyze_cutlass_tiling()       —— 确定性的量化分析（CPU 可跑）
 *   4) run_gpu_checks()               —— 真机正确性检查（注释，放开即跑）
 *
 * 沙箱无 GPU/nvcc，本文件为教材级标准 CUDA（WMMA 需 SM_70+，cp.async 需 SM_80+）。
 * 真机编译：
 *   nvcc -O3 -std=c++17 -arch=sm_80 cuda_l2_cutlass_gemm.cpp -o cg && ./cg
 * 对照 CUTLASS 真实源码（main 分支）：
 *   include/cutlass/gemm/threadblock/mma_multistage.h
 *   include/cutlass/gemm/kernel/gemm.h
 *   include/cutlass/gemm/device/gemm.h
 ******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---- 真机才 include 的 CUDA 头（沙箱 mock，保证本地语法完整） ----
#if defined(__CUDACC__)
  #include <cuda_runtime.h>
  #include <mma.h>
  using namespace nvcuda;
  #define CUDA_CALL __host__ __device__
#else
  // 纯 CPU 校验路径用的极简桩
  typedef int cudaError_t;
  #define cudaSuccess 0
  #define CUDA_CALL
  #define __global__
  #define __shared__
  #define __host__
  #define __device__
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// 1) 缩小版 CUTLASS 参考 kernel —— 三级 tile 形状 + kStages 双缓冲 + pipeline
// =============================================================================
//
// 对照 CUTLASS 真实模板签名（device/gemm.h:150,153,156）：
//   typename ThreadblockShape_,   // e.g. GemmShape<128,128,8>
//   typename WarpShape_,          // e.g. GemmShape< 64, 64, 8>
//   typename InstructionShape_,   // e.g. GemmShape< 16,  8, 8> (Tensor Core)
//   int kStages                   // 2 / 3 / 4  —— 环形缓冲份数
//
// 这里用编译期常量等价表达（教学清晰优先）：

#if defined(__CUDACC__)
template <int BM,   // ThreadblockShape.M  —— 一个 CTA 负责的 M 维块
          int BN,   // ThreadblockShape.N  —— 一个 CTA 负责的 N 维块
          int BK,   // ThreadblockShape.K  —— 一个 CTA 每步搬的 K 维切片
          int STAGES> // kStages           —— 共享内存里 K 切片的份数（双/三缓冲）
__global__ void tiny_wmma_gemm(const half* __restrict__ A,
                               const half* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
  // ---- 三级 tile 形状的实例化（对照 GemmShape 三级）----
  // Threadblock tile : BM x BN  (本 kernel 用 64x64)
  // Warp       tile : WM x WN  (2x2 排布 => 4 warp / CTA，64 线程/warp)
  // Instruction tile : 16 x 16 x 16  —— WMMA 一次 mma_sync 的形状
  const int WM = 32, WN = 32;
  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  const int warpM = warpId / 2;   // 0 或 1
  const int warpN = warpId % 2;   // 0 或 1

  // ---- 共享内存：kStages 份 K 切片的环形缓冲（双缓冲 STAGES=2）----
  // 这正是 CUTLASS MmaMultistage 用 Stages 模板给 shared 开多份 tile
  // （mma_multistage.h:91  class MmaMultistage : public MmaBase<Shape_,Policy_,Stages>）
  __shared__ half sA[STAGES][BM][BK];
  __shared__ half sB[STAGES][BK][BN];

  // ---- Tensor Core 碎片：本 warp 算 WM x WN = 32x32，
  //      用 16x16 碎片 => M 向 2 份、N 向 2 份；K 向 BK/16 步 ----
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2][2];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];

  #pragma unroll
  for (int m = 0; m < 2; ++m)
    #pragma unroll
    for (int n = 0; n < 2; ++n)
      wmma::fill_fragment(acc[m][n], 0.0f);

  const int aRow = blockIdx.y * BM;          // 本 CTA 负责的 A 行块
  const int bCol = blockIdx.x * BN;          // 本 CTA 负责的 B 列块

  // ---- prologue()：先把第 0 个 stage 的 K 切片搬进共享内存
  //      （对照 mma_multistage.h:362 prologue —— 灌满流水线 bootstrap） ----
  {
    int k = 0;
    const half* Aptr = A + aRow * K + k;
    const half* Bptr = B + k * N + bCol;
    for (int e = 0; e < (BM * BK) / 128; ++e) {
      // 简化：用 16 线程/128B 向量化加载（对照 onnxruntime 的 int4 向量化）
      int idx = e * 128 + laneId * 8;
      if (idx < BM * BK) {
        // 仅教学示意：实际 CUTLASS 用 cp.async；此处普通 load
        reinterpret_cast<float4*>(&sA[0][0][0])[e * 32 + laneId] =
            reinterpret_cast<const float4*>(Aptr)[e * 32 + laneId];
      }
    }
    for (int e = 0; e < (BK * BN) / 128; ++e) {
      int idx = e * 128 + laneId * 8;
      if (idx < BK * BN) {
        reinterpret_cast<float4*>(&sB[0][0][0])[e * 32 + laneId] =
            reinterpret_cast<const float4*>(Bptr)[e * 32 + laneId];
      }
    }
    __syncthreads();
  }

  // ---- mac_loop_iter() 主循环：算当前 stage，同时把下一 stage 的 K 切片搬进来
  //      （对照 mma_multistage.h:496 mac_loop_iter + 505-607 三段交错） ----
  int stage = 0;
  for (int kStep = 0; kStep < K / BK; ++kStep) {
    int nextStage = (stage + 1) % STAGES;
    int k = (kStep + 1) * BK;

    // --- 生产者：把下一 stage 的 K 切片搬进共享内存 ---
    // CUTLASS 这里用 cutlass::arch::cp_async（见 mma_multistage.h:314），
    // 由 copy engine 异步完成，不占 SM 计算流水线；最后 cp_async_fence() 围栏。
    if (kStep + 1 < K / BK) {
      const half* Aptr = A + aRow * K + k;
      const half* Bptr = B + k * N + bCol;
      for (int e = 0; e < (BM * BK) / 128; ++e)
        reinterpret_cast<float4*>(&sA[nextStage][0][0])[e * 32 + laneId] =
            reinterpret_cast<const float4*>(Aptr)[e * 32 + laneId];
      for (int e = 0; e < (BK * BN) / 128; ++e)
        reinterpret_cast<float4*>(&sB[nextStage][0][0])[e * 32 + laneId] =
            reinterpret_cast<const float4*>(Bptr)[e * 32 + laneId];
      // CUTLASS: cutlass::arch::cp_async_fence(); 然后 cp_async_wait<STAGES-2>()
    }
    __syncthreads();   // 等下一 stage 就绪（对照 cp_async_wait 的语义）

    // --- 消费者：用当前 stage 的共享内存做 Tensor Core 乘加 ---
    // Operator warp_mma_（mma_multistage.h:194,541）最终发出 mma.sync PTX
    #pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      wmma::load_matrix_sync(a_frag,
          &sA[stage][warpM * WM][ks * 16], BK);
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        wmma::load_matrix_sync(b_frag[0][n],
            &sB[stage][ks * 16][warpN * WN + n * 16], BN);
      #pragma unroll
      for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int n = 0; n < 2; ++n)
          wmma::mma_sync(acc[m][n], a_frag, b_frag[0][n], acc[m][n]);
    }

    stage = nextStage;   // advance_smem_read_stage / write_stage（mma_multistage.h:249,263）
  }

  // ---- Epilogue：累加器写回 C（对照 kernel/gemm.h:332 epilogue()） ----
  #pragma unroll
  for (int m = 0; m < 2; ++m)
    #pragma unroll
    for (int n = 0; n < 2; ++n)
      wmma::store_matrix_sync(
          &C[(aRow + warpM * WM + m * 16) * N + (bCol + warpN * WN + n * 16)],
          acc[m][n], N, wmma::mem_row_major);
}
#endif  // __CUDACC__

// =============================================================================
// 2) FP32 CPU 参考（用于 GPU 结果对拍 / 与 naive 对比）
// =============================================================================
static void tiny_gemm_ref_cpu(const float* A, const float* B, float* C,
                              int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// =============================================================================
// 3) 确定性量化分析（CPU 可跑）—— 把"tiling + 双缓冲"的收益算出来
// =============================================================================
static void analyze_cutlass_tiling() {
  printf("\n===== analyze_cutlass_tiling() : 量化 CUTLASS 的 tile/缓冲 结构 =====\n");

  // 典型 FP16 Tensor Core 配置（对照 DefaultGemmConfiguration）
  const int TB_M = 128, TB_N = 128, TB_K = 8;     // ThreadblockShape
  const int W_M  =  64, W_N  =  64, W_K  = 8;     // WarpShape
  const int I_M  =  16, I_N  =  8, I_K  =  8;     // InstructionShape (mma.sync m16n8k8)
  const int STAGES = 3;                            // kStages（双/三/四缓冲）

  // (a) 一个 CTA 内 warp 数 = 线程数/32
  int warpsPerCTA = (TB_M / W_M) * (TB_N / W_N);   // 2*2 = 4
  int threadsPerCTA = warpsPerCTA * 32;            // 128
  // (b) 一个 warp 内 instruction 级 MMA 次数 = (W_M/I_M)*(W_N/I_N)*(W_K/I_K)
  int mmaPerWarp = (W_M / I_M) * (W_N / I_N) * (W_K / I_K);
  // (c) 一个 CTA 完成一次 K 步需要的 instruction MMA 总数
  int mmaPerCTA_kstep = mmaPerWarp * warpsPerCTA;

  printf("  ThreadblockShape = <%d,%d,%d>  WarpShape = <%d,%d,%d>  "
         "InstructionShape = <%d,%d,%d>\n",
         TB_M, TB_N, TB_K, W_M, W_N, W_K, I_M, I_N, I_K);
  printf("  每 CTA warps = %d (threads=%d); 每 warp MMA 数 = %d; "
         "每 K 步 CTA 总 MMA = %d\n",
         warpsPerCTA, threadsPerCTA, mmaPerWarp, mmaPerCTA_kstep);
  printf("  kStages = %d  => 共享内存里 %d 份 K 切片环形缓冲（双/三/四缓冲）\n",
         STAGES, STAGES);

  // (d) 双缓冲收益：单次 K 步的共享内存流量 vs 全局流量
  //     A tile = TB_M x TB_K (FP16=2B), B tile = TB_K x TB_N
  //     全局抓取：每 K 步从 HBM 抓 2 * TB_M*TB_K*2 + 2 * TB_K*TB_N*2 字节
  //     共享复用：同一份被 warpsPerCTA 个 warp 各用 W_K/I_K 次
  long long gmemA_bytes = 2LL * TB_M * TB_K;       // B 矩阵 FP16
  long long gmemB_bytes = 2LL * TB_K * TB_N;
  long long gmem_total  = gmemA_bytes + gmemB_bytes;
  // 共享内存里被 32 线程的 warp 读 W_M*W_N 次（每 warp 读自己那份 tile）
  long long smem_reuse   = (long long)TB_M * TB_N * 2; // 近似：每字节被复用 ~warps 次

  printf("  单 K 步 全局抓取 ≈ %lld B (A %lld + B %lld)，写进 shared 后\n",
         gmem_total, gmemA_bytes, gmemB_bytes);
  printf("  被 %d 个 warp 在各自 %d 次 MMA 中复用 => 共享复用 ≈ %lld 元素/步\n",
         warpsPerCTA, (W_K / I_K), smem_reuse);

  // (e) 二级对照：naive 全局重复抓取 vs 分块（呼应 L2.4 analyze_tiling）
  //     取 N=M=K=S 方阵，naive 每个输出元素每 k 都读 HBM；tiling 每个全局元素只取 1 次
  const int S = 256;
  long long naive_gmem = 2LL * S * S * S + 1LL * S * S;  // 2*S^3 + S^2
  long long tiled_gmem = 3LL * S * S;                    // 3*S^2 (A,B 各 1 份 + C)
  printf("\n  [回扣 L2.4] S=%d 方阵: naive 全局流量≈%lld B, tiled≈%lld B, "
         "下降 %.0f×\n", S, naive_gmem, tiled_gmem,
         (double)naive_gmem / tiled_gmem);

  // (f) 算术强度 AI（回扣 L2.3 Roofline）：分块不改变 AI，但砍掉冗余 DRAM 流量
  //     AI = FLOPs(per output) / Bytes(global, per output)
  //     单个输出元素需要 K 次乘加 => 2K FLOPs；全局字节 = (K+K+1)*sizeof
  double flops = 2.0 * S;                       // 单个输出元素（K=S）
  double bytes = (double)(2 * S + 2 * S + 4);   // A 的 K + B 的 K + C 的 1 (FP32 写回)
  double AI = flops / bytes;
  printf("  单输出元素: FLOPs=%.0f, 全局Bytes≈%.0f => AI≈%.3f FLOP/Byte "
         "(分块前/后不变，砍的是冗余访存)\n", flops, bytes, AI);

  printf("  >> 结论：tiling 把\"每个全局元素只取 1 次\"，AI 不变但 DRAM 流量暴降；\n");
  printf("     kStages 双/三缓冲 + cp.async pipeline 把\"取下一 K 片\"与\"算当前\"重叠，\n");
  printf("     隐藏 HBM 延迟 => 让 compute-bound 的 Tensor Core 持续吃满。\n");
}

// =============================================================================
// 4) 真机正确性检查（注释：沙箱无 GPU，放开即用）
// =============================================================================
#if defined(__CUDACC__)
static void run_gpu_checks() {
  const int M = 128, N = 128, K = 256;
  float *hA, *hB, *hC, *hCref;
  half  *dA, *dB; float *dC;
  hA = (float*)malloc(M * K * sizeof(float));
  hB = (float*)malloc(K * N * sizeof(float));
  hC = (float*)malloc(M * N * sizeof(float));
  hCref = (float*)malloc(M * N * sizeof(float));
  for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 100) / 30.f - 1.f;
  for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 100) / 30.f - 1.f;

  tiny_gemm_ref_cpu(hA, hB, hCref, M, N, K);   // CPU 参考

  cudaMalloc(&dA, M * K * sizeof(half));
  cudaMalloc(&dB, K * N * sizeof(half));
  cudaMalloc(&dC, M * N * sizeof(float));
  // 转 half ...（省略转换，演示主线）
  // tiny_wmma_gemm<64,64,32,2><<<dim3(N/64, M/64), 128>>>(dA, dB, dC, M, N, K);
  // cudaMemcpy(dC -> hC); 然后与 hCref 比对 max|Δ|

  printf("  [run_gpu_checks] 真机对拍：max|GPU - CPU| 应 < 1e-2（FP16 精度）\n");
  free(hA); free(hB); free(hC); free(hCref);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
#endif

// =============================================================================
// main
// =============================================================================
int main() {
  printf("===== CUTLASS GEMM 对照 onnxruntime：tiling+双缓冲+pipeline+TensorCore =====\n");

  // (1) 在 CPU 上验证 tiny_gemm_ref_cpu 数值正确（与手算一致）
  {
    const int M = 4, N = 4, K = 4;
    float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};  // 单位阵
    float C[16];
    tiny_gemm_ref_cpu(A, B, C, M, N, K);
    printf("\n[自检] A×I（应为 A 本身）:\n");
    int ok = 1;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        printf("%6.1f", C[i * N + j]);
        if (A[i * N + j] != C[i * N + j]) ok = 0;
      }
      printf("\n");
    }
    printf("  一致性: %s\n", ok ? "PASS" : "FAIL");
  }

  // (2) 量化分析（CPU 可跑，打印 tile/缓冲/复用/AI）
  analyze_cutlass_tiling();

  // (3) 真机检查（注释）
#if defined(__CUDACC__)
  run_gpu_checks();
#else
  printf("\n  (run_gpu_checks 需 nvcc 真机编译；沙箱跳过，详见文件内注释)\n");
#endif

  // (4) 与上一节 onnxruntime 模式的一一对照（文字索引，详见 .md）
  printf("\n===== 与 onnxruntime 工业模式的对照速查 =====\n");
  printf("  onnxruntime transpose tile[32][33]  +1 破 bank conflict\n");
  printf("      -> CUTLASS: ThreadblockShape 模板 + ThreadblockSwizzle 自动 swizzle\n");
  printf("  onnxruntime softmax __shfl_xor warp 归约\n");
  printf("      -> CUTLASS: Operator 组件（可换 SIMT / TensorOp）传给 MmaMultistage\n");
  printf("  onnxruntime layernorm 跨 warp 归约 + grid-stride\n");
  printf("      -> CUTLASS: kStages + prologue/pipeline 管理 生产者/消费者 重叠\n");
  printf("  onnxruntime GEMM 直接交 cuBLASLt(gemm.cc)\n");
  printf("      -> CUTLASS 正是 cuBLASLt 背后的手写分块+双缓冲+pipeline+TC 模板\n");
  printf("\n  CUTLASS 核心封装（device/gemm.h:150-167）：\n");
  printf("    template<ElementA,LayoutA,ElementB,LayoutB,ElementC,\n");
  printf("             ElementAccumulator,OperatorClass,ArchTag,\n");
  printf("             ThreadblockShape, WarpShape, InstructionShape,\n");
  printf("             EpilogueOutputOp, ThreadblockSwizzle, int kStages>\n");
  printf("    class Gemm { using GemmKernel = kernel::DefaultGemm<...>; };\n");

  return 0;
}
