// ============================================================================
//  cuda_l3_fusion_pipeline.cpp
//  L3 架构级第一站：kernel fusion + 异步拷贝(cp.async / memcpy_async) + pipeline
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l3_fusion_pipeline.cpp -o l3 && ./l3
//  （真机 CUDA kernel 片段由 #ifdef __CUDACC__ 守护，g++ 下跳过，仅作教材）
//
//  三大块：
//    Part A  kernel fusion：全局流量量化 + CPU 数值对拍
//            （为什么融：避免中间张量落全局内存做无意义往返）
//    Part B  cp.async / memcpy_async：全局->共享的「异步」拷贝语义
//            （真机 PTX / C++ 片段，运行期在 GPU 上）
//    Part C  producer/consumer 软件流水线：时间线仿真，量化重叠加速比
//            （为什么需要 pipeline：把 Part B 的异步取数与计算重叠，藏掉访存延迟）
// ============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ----------------------------- 小工具 ----------------------------------------
static float gelu(float x){
  // GELU 近似（CUDA 常用 tanh 近似）：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  return 0.5f * x * (1.0f + std::tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ============================ Part A: kernel fusion ============================
// 目标算子（逐元素，size N）：   y = GELU( scale * x + bias )
//
//  naive 写法：拆成 3 个独立 kernel
//      k1: t1 = x * scale        （读 x，写 t1）
//      k2: t2 = t1 + bias        （读 t1，写 t2）
//      k3: y  = gelu(t2)         （读 t2，写 y）
//      → 每个中间张量都「写全局 + 下一 kernel 再读全局」= 多次全局往返
//
//  fused 写法：1 个 kernel，全程在寄存器，只 1 读(x) + 1 写(y)
//      → 中间 t1/t2 从不落全局

struct Traffic { long reads; long writes; long total() const { return reads + writes; } };

static Traffic fusion_traffic_naive(int N, int stages){
  const int FLOAT = 4;
  // stages 个 kernel，每个读前一 stage 输出 + 写自身输出（第一 stage 读 x）
  Traffic t;
  t.reads  = (long)stages * N * FLOAT;   // x, t1, t2, ...
  t.writes = (long)stages * N * FLOAT;   // t1, t2, y, ...
  return t;
}
static Traffic fusion_traffic_fused(int N){
  const int FLOAT = 4;
  Traffic t;
  t.reads  = (long)1 * N * FLOAT;        // 读 x 一次
  t.writes = (long)1 * N * FLOAT;        // 写 y 一次
  return t;
}

// CPU 数值对拍：fused 结果必须 == naive 逐 stage 串联结果
static bool verify_fusion(int N, float scale, float bias){
  std::vector<float> x(N), t1(N), t2(N), y_naive(N), y_fused(N);
  for(int i=0;i<N;++i) x[i] = (float)((i % 7) - 3) * 0.37f + 0.1f; // 确定性输入
  for(int i=0;i<N;++i) t1[i]      = x[i] * scale;
  for(int i=0;i<N;++i) t2[i]      = t1[i] + bias;
  for(int i=0;i<N;++i) y_naive[i] = gelu(t2[i]);
  for(int i=0;i<N;++i) y_fused[i] = gelu(scale * x[i] + bias);
  float maxerr = 0.f;
  for(int i=0;i<N;++i) maxerr = std::max(maxerr, std::fabs(y_naive[i] - y_fused[i]));
  return maxerr < 1e-5f;
}

// ============================ Part B: cp.async / memcpy_async =================
// （纯文档 + 真机片段；运行期由 __CUDACC__ 守护，g++ 下不编译）
// -----------------------------------------------------------------------------
//  (1) PTX cp.async  ——  Ampere(SM_80) 引入，Hopper(SM_90) 由 TMA 取代但语义一致
//      普通 __global__ load 会把数据经「寄存器」搬进共享内存（ld.global -> st.shared）。
//      cp.async 则让**专门的异步拷贝引擎**直接把数据从全局内存搬到共享内存，
//      **不经过任何线程的寄存器**，且**发射线程立刻继续**，不阻塞等待。
//      语法（每线程搬 16 字节为例）：
//          cp.async.ca.shared.global  [smem_dst], [gmem_src], 16;
//          //  .ca = cache at all levels（L2/L1 都缓存），.cg 仅 L2
//          //  第 3 操作数必须是 4 / 8 / 16 之一
//          cp.async.commit_group;          // 把之前若干 cp.async 归为一组
//          cp.async.wait_group    0;       // 等该组全部完成（0 = 所有组）
//      还有 cp.async.bulk（Hopper TMA，一次搬一整块并自动处理边界/padding）。
//
//  (2) C++ memcpy_async + cuda::pipeline（CUDA 11.6+，头文件 <cuda/pipeline>）
//      是 cp.async 的类型安全封装，用 pipeline 管理「生产者/消费者」的顺序与可见性：
//          __shared__ alignas(16) extern char smem[];
//          auto& pipe = *reinterpret_cast<cuda::pipeline<cuda::thread_scope_block>*>(smem);
//          // 生产者（一个线程/warp）：
//          auto tok = pipe.producer_acquire();
//          cuda::memcpy_async(smem_stage, gmem_src, bytes, pipe);
//          pipe.producer_commit(tok);
//          // 消费者：
//          auto tok = pipe.consumer_wait();          // 等本 stage 拷贝完
//          // ... 用 smem_stage 计算 ...
//          pipe.consumer_release();
//
//  (3) 这正是 CUTLASS MmaMultistage 的底层机制（已备份 _cutlass_src/）：
//        mma_multistage.h:125  "Minimum architecture is Sm80 to support cp.async"
//        mma_multistage.h:311 / :349  cutlass::arch::cp_async<...> 搬运 A/B 的每一 stage
//        mma_multistage.h:360-370  prologue 预取前 kStages-1 个 stage 的全局片段
//        mma_pipelined.h:136-137    static_assert(kStages==2, "Double-buffered pipeline")
//      即：CUTLASS 用「kStages 个共享缓冲 + cp.async 流水线」把取操作数与 MMA 重叠，
//      这正是 Part C 要量化的「重叠」收益。
// -----------------------------------------------------------------------------
#ifdef __CUDACC__
// —— 真机片段 A：fused elementwise kernel（一个 kernel 干完 mul+add+gelu）——
__global__ void fused_gelu_kernel(const float* __restrict__ x, float* __restrict__ y,
                                  float scale, float bias, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) y[i] = gelu(scale * x[i] + bias);   // 全程寄存器，t1/t2 不落全局
}

// —— 真机片段 B：cp.async prologue（节选自 CUTLASS MmaMultistage 思路）——
//  生产者 warp 把全局 A/B 块异步搬进共享缓冲 stage，发射后立即回去发下一个 stage，
//  消费者 warp 在 cp.async.wait_group 后直接用共享里的操作数做 MMA。
//  （完整实现见 _cutlass_src/mma_multistage.h；此处只示意骨架）
__device__ void cp_async_prologue_skeleton(){
  // extern __shared__ char smem[];
  // for(int stage=0; stage < kStages-1; ++stage){
  //   cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(dst, src);
  //   cp_async_commit_group();        // 见 mma_multistage.h:433 "boundary of a stage"
  // }
  // cp_async_wait_group<kStages-2>(); // 等前面组，保留若干在途
}
#endif // __CUDACC__

// ============================ Part C: pipeline 时间线仿真 ======================
//  模型：N 个「互相独立」的 tile（GEMM 输出 tile / elementwise 分块都独立，
//        不依赖前一个 tile 的计算结果）。
//  每个 tile：copy 延迟 Tc（全局->共享，即 cp.async 的飞行时间），compute 延迟 Tm。
//  k 个共享缓冲（= CUTLASS 的 kStages）。单拷贝引擎顺序发射拷贝。
//  约束：拷贝 tile i 需要缓冲 (i % k)，该缓冲在 tile (i-k) 计算完成后才释放。
//  消费者 tile i 只要「自己那份拷贝完成」即可开工（tile 间独立 → 不等前一个 compute）。
static double pipeline_time(int N, int k, double Tc, double Tm){
  std::vector<double> ce(N), cend(N);            // copy-end, compute-end
  for(int i=0;i<N;++i){
    double prev_copy_end = (i==0) ? 0.0 : ce[i-1];
    double buf_release   = (i<k) ? 0.0 : cend[i-k];
    double cs  = std::max(prev_copy_end, buf_release);  // copy start
    ce[i]      = cs + Tc;
    double cstart = ce[i];                                // 独立 tile：只需本 tile 拷贝完
    cend[i]    = cstart + Tm;
  }
  return cend[N-1];
}
static double naive_time(int N, double Tc, double Tm){ return (double)N * (Tc + Tm); }

static void run_sims(){
  printf("\n--- Part C: 软件流水线时间线仿真 (N=%d 个独立 tile) ---\n", 200);
  struct Case { const char* name; double Tc; double Tm; };
  Case cases[] = {
    {"balanced     Tc=Tm=1.0", 1.0, 1.0},   // 均衡
    {"compute-bound Tc=1 Tm=3", 1.0, 3.0},   // 计算主导（Tensor Core 很快时常见）
    {"memory-bound  Tc=3 Tm=1", 3.0, 1.0},   // 取数主导（大算子、带宽受限时）
  };
  int N = 200;
  int ks[] = {1, 2, 4, 8};
  for(auto& c : cases){
    double t_naive = naive_time(N, c.Tc, c.Tm);
    printf("\n[%s]\n  naive_total = %.0f\n", c.name, t_naive);
    for(int k : ks){
      double t_pipe = pipeline_time(N, k, c.Tc, c.Tm);
      printf("  kStages=%-2d  pipeline_total = %6.0f   加速比 = %.2fx\n",
             k, t_pipe, t_naive / t_pipe);
    }
  }
  printf("\n[解读] 收益随 stages 增大而逼近上限：均衡/访存主导时接近 2x；\n"
         "       计算已占主导(compute-bound)时收益趋近 1x —— 因为瓶颈本就不是访存，\n"
         "       本来就没多少延迟可藏。这解释了为什么 fusion/pipeline 对「带宽受限」算子最划算。\n");
}

int main(){
  printf("===== L3.1  kernel fusion + async copy + pipeline =====\n");

  // ---------- Part A ----------
  int N = 1 << 20;                       // 1M 元素
  Traffic tn = fusion_traffic_naive(N, 3);
  Traffic tf = fusion_traffic_fused(N);
  printf("\n--- Part A: kernel fusion 全局流量 (N=%d, float=4B) ---\n", N);
  printf("naive(3 kernel): read=%ld write=%ld total=%ld 字节\n", tn.reads, tn.writes, tn.total());
  printf("fused(1 kernel): read=%ld write=%ld total=%ld 字节\n", tf.reads, tf.writes, tf.total());
  printf("全局流量降低: %.2fx\n", (double)tn.total() / (double)tf.total());
  bool ok = verify_fusion(1024, 0.5f, 0.25f);
  printf("[verify_on_cpu] %s  (fused 数值 == naive 串联, N=1024)\n", ok ? "PASS" : "FAIL");

  // ---------- Part C ----------
  run_sims();

  printf("\n(纯 C++ 校验完成；CUDA kernel 片段由 __CUDACC__ 守护，需真机 nvcc 编译)\n");
  return 0;
}
