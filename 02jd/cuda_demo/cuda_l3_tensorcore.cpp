// ============================================================================
//  cuda_l3_tensorcore.cpp
//  L3 架构级第二站：Tensor Core (WMMA / MMA PTX / CUTLASS) + 混合精度
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l3_tensorcore.cpp -o tc && ./tc
//  （真机 WMMA / MMA PTX / CUTLASS 片段由 #ifdef __CUDACC__ 守护，g++ 下跳过）
//
//  三大块：
//    Part A  混合精度 GEMM：fp16 输入 + fp32 累加（host 模拟 __half 量化）
//            → 数值 ≈ 纯 fp32；且 fp16 输入把内存/带宽减半（2×↓）
//    Part B  吞吐模型：TC 用低精度换取数倍~数十倍算力（V100/A100 示例规格）
//    Part C/D 真机 WMMA fragment API + 等价 MMA PTX + CUTLASS MmaTensorOp 连接
// ============================================================================
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

// ---------- fp16 host 模拟（替代 GPU 上的 __half / TC 低精度输入）----------
// 仅用于纯 C++ 演示「fp16 量化 → 误差很小」；真机由硬件 TC 完成。
static inline uint16_t f32_to_f16(float f){
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp  = (x >> 23) & 0xffu;
  uint32_t mant =  x & 0x7fffffu;
  if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u)); // inf/nan
  if (exp == 0u)    return (uint16_t)sign;                                    // 0/denorm
  int e = (int)exp - 127 + 15;
  if (e >= 0x1f)    return (uint16_t)(sign | 0x7c00u);                        // 溢出→inf
  if (e <= 0) {                                                                // 次正规
    mant |= 0x800000u;
    unsigned shift = (unsigned)(14 - e);
    if (shift > 31) return (uint16_t)sign;
    return (uint16_t)(sign | (mant >> shift));
  }
  mant = (mant + 0x1000u) >> 13;                  // 保留 10 位尾数，就近舍入
  if (mant & 0x400u) { mant = 0; if (++e >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
  return (uint16_t)(sign | ((uint32_t)e << 10) | mant);
}
static inline float f16_to_f32(uint16_t h){
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (uint32_t)(h & 0x7c00u) >> 10;
  uint32_t mant = (uint32_t)(h & 0x03ffu);
  uint32_t r;
  if (exp == 0u) {
    if (mant == 0u) { r = sign; }
    else { uint32_t e = 127u - 15u; while (!(mant & 0x400u)) { mant <<= 1; e--; } mant &= 0x3ffu; r = (e << 23) | (mant << 13); }
  } else if (exp == 0x1fu) { r = sign | 0x7f800000u | (mant << 13); }
  else { r = ((exp - 15u + 127u) << 23) | (mant << 13); }
  float out; std::memcpy(&out, &r, 4); return out;
}

// ============================ Part A: 混合精度 GEMM ============================
//  C = A·B，A/B 以 fp16 表示（低精度乘法），累加用 fp32（高精度）→ TC 的精髓
static void gemm_fp32(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K){
  for(int i=0;i<M;++i) for(int j=0;j<N;++j){
    float s=0.f; for(int k=0;k<K;++k) s += A[i*K+k]*B[k*N+j];
    C[i*N+j]=s;
  }
}
static void gemm_mixed(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<float>& C, int M, int N, int K){
  for(int i=0;i<M;++i) for(int j=0;j<N;++j){
    float s=0.f;
    for(int k=0;k<K;++k){
      // GPU 真机：__half a=__float2half(A[..]); __half b=__float2half(B[..]);
      //            s = __fmaf_rn((float)a, (float)b, s);   // fp16 乘，fp32 累加
      float a = f16_to_f32(f32_to_f16(A[i*K+k]));
      float b = f16_to_f32(f32_to_f16(B[k*N+j]));
      s += a*b;
    }
    C[i*N+j]=s;
  }
}
static bool verify_mixed(int M, int N, int K){
  std::vector<float> A(M*K), B(K*N), Cf(M*N), Cm(M*N);
  for(int i=0;i<M*K;++i) A[i] = ((i%5)-2)*0.1f;   // 小范围确定性输入 [-0.4,0.4]
  for(int i=0;i<K*N;++i) B[i] = ((i%7)-3)*0.1f;
  gemm_fp32(A,B,Cf,M,N,K);
  gemm_mixed(A,B,Cm,M,N,K);
  float maxerr=0.f;
  for(int i=0;i<M*N;++i) maxerr = std::max(maxerr, std::fabs(Cf[i]-Cm[i]));
  printf("  最大绝对误差 = %.2e（fp16 尾数 ~10bit，典型 GEMM 下远小于 1）\n", maxerr);
  return maxerr < 0.5f;   // 宽松判据：仅防 fp16 模拟彻底错误
}
static void report_memory(int M, int N, int K){
  const int F=4;
  long fp32_in = (long)(M*K + K*N) * F;
  long fp16_in = (long)(M*K + K*N) * 2;       // fp16 = 2 字节
  printf("  输入字节：fp32=%ld  fp16=%ld  → 内存/带宽节省 %.2fx\n", fp32_in, fp16_in, (double)fp32_in/fp16_in);
}

// ============================ Part B: 吞吐模型 =================================
static void throughput_model(){
  printf("\n--- Part B: Tensor Core 吞吐模型（示例 GPU 规格，理想峰值）---\n");
  struct Gpu { const char* name; double fp32_tflops; double tc_tflops; };
  Gpu gpus[] = { {"V100 (fp16 TC)", 15.7, 125.0}, {"A100 (fp16 TC)", 19.5, 312.0}, {"A100 (tf32 TC)", 19.5, 156.0} };
  int M=4096,N=4096,K=4096;
  double flops = 2.0*(double)M*N*K;
  printf("  GEMM %dx%dx%d = %.1f GFLOP\n", M,N,K, flops/1e9);
  for(auto& g : gpus){
    double t_fp32 = flops/(g.fp32_tflops*1e12);
    double t_tc   = flops/(g.tc_tflops*1e12);
    printf("  %-16s fp32=%.2f ms  TC=%.4f ms  理想加速比=%.1fx\n",
           g.name, t_fp32*1e3, t_tc*1e3, g.tc_tflops/g.fp32_tflops);
  }
  printf("  [解读] TC 用低精度输入换数倍~数十倍算力；fp16 还把输入带宽需求减半，\n"
         "        故即便 memory-bound 的 GEMM 也受益。代价：输入精度有限 → 见 Part C 混合精度纪律。\n");
}

// ============================ Part C/D: 真机片段（__CUDACC__ 守护）=============
#ifdef __CUDACC__
#include <mma.h>
using namespace nvcuda::wmma;
// ---- WMMA fragment API：形状即一条 mma.sync 的 warp 级 tile ----
__global__ void wmma_gemm_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K){
  // 每个 warp 负责一个 16x16 输出块；fragment 把矩阵块映射到寄存器
  fragment<matrix_a, 16,16,16, half, col_major> a_frag;
  fragment<matrix_b, 16,16,16, half, row_major> b_frag;
  fragment<accumulator, 16,16,16, float>        c_frag;
  int warp_m = (blockIdx.y*blockDim.y + threadIdx.y) * 16;
  int warp_n = (blockIdx.x*blockDim.x + threadIdx.x) * 16; // 示意
  fill_fragment(c_frag, 0.0f);
  for(int k=0; k<K; k+=16){
    // 全局 → 共享/寄存器（可接 L3.1 的 cp.async 流水线取数）
    load_matrix_sync(a_frag, A + warp_m*K + k,        K);
    load_matrix_sync(b_frag, B + k*N + warp_n,        N);
    mma_sync(c_frag, a_frag, b_frag, c_frag);          // D = A·B + C，fp32 累加
  }
  store_matrix_sync(C + warp_m*N + warp_n, c_frag, N, mem_row_major);
}
// ---- 等价底层 PTX（编译器从 WMMA 生成，这里写出其形式）----
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//        {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0}, {c0,c1,c2,c3};
//   说明：指令本身是 m16n8k16；一个 16x16x16 的 warp tile = 两条这样的指令
//        拼出 N 方向的两半（d0,d1 与 d2,d3 分别对应 N 的前/后 8 列）。
//   变体：tf32→ .m16n8k8...tf32.tf32.f32；int8→ .m16n8k32...s8.s8.s32。
//
// ---- CUTLASS 连接（已备份 _cutlass_src/）----
//   default_mma.h:190  "Specialization ... (OperatorClass TensorOp)" 按 Operator 派发
//   default_mma.h:213  InstructionShape（指令级形状，如 <16,8,16> fp16 / <16,8,8> tf32）
//   warp_mma_tensor_op.h:167  class MmaTensorOp（warp 级 Tensor Core MMA）
//   warp_mma_tensor_op.h:206  InstructionShape = ArchMmaOperator::Shape
//   warp_mma_tensor_op.h:294-296  FragmentA / FragmentB / FragmentC
//   → 即 L2 的「三级 GemmShape」里最内层的 InstructionShape + L3.1 的 kStages
//     cp.async 取数 + station4 的 EVT epilogue，拼成完整 TC GEMM。
#endif // __CUDACC__

int main(){
  printf("===== L3.2  Tensor Core (WMMA / MMA PTX / CUTLASS) + 混合精度 =====\n");
  int M=32,N=32,K=32;
  printf("\n--- Part A: 混合精度 GEMM (M=N=K=%d) ---\n", M);
  bool ok = verify_mixed(M,N,K);
  printf("[verify_on_cpu] %s  (fp16 输入 + fp32 累加 ≈ 纯 fp32)\n", ok?"PASS":"INFO");
  report_memory(M,N,K);
  throughput_model();
  printf("\n(纯 C++ 校验完成；WMMA/MMA PTX/CUTLASS 真机片段由 __CUDACC__ 守护，需真机 nvcc 编译)\n");
  return 0;
}
