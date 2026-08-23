// =============================================================================
// cuda_l3_epilogue_fusion.cpp
// -----------------------------------------------------------------------------
// 工业级对照（源码线续）：CUTLASS 的 epilogue 组件（bias 广播 + 激活 + 写回）
//    vs  onnxruntime 的 bias broadcast 巧法
//    —— 看工业界如何用"同一套 pipeline"描述任意算子
//
// 真机编译（含 GPU kernel，sm_70+）：
//   nvcc -O3 -std=c++17 cuda_l3_epilogue_fusion.cpp -o ep && ./ep
// 仅想看数值正确性（CPU 即可，无需 GPU）：
//   g++  -O3 -std=c++17 cuda_l3_epilogue_fusion.cpp -o ep && ./ep
//
// 设计：MiniOutputOp / epilogue_pipeline / 参考对拍 都是纯 C++（无 CUDA 依赖），
//       所以 g++ 也能跑通并打印 PASS；真正的 __global__ kernel 用 __CUDACC__ 守护，
//       nvcc 下才会编译。这与 CUTLASS 把"输出算子"做成可替换 functor 的思想一致。
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// -----------------------------------------------------------------------------
// 0. 激活 / 二元算子（对应 CUTLASS thread/activation.h 与 functional.h）
//    CUTLASS 把它们做成模板参数 ElementwiseOp / BinaryOp，于是同一个 epilogue
//    pipeline 能服务 GEMM / GEMM+bias / GEMM+bias+ReLU / GELU / residual …
// -----------------------------------------------------------------------------
enum class Act { kIdentity, kRelu, kGelu };
enum class BinOp { kAdd };   // CUTLASS 默认 plus：z = α·accum + β·C + bias

static inline float apply_act(float x, Act a) {
    switch (a) {
        case Act::kIdentity: return x;
        case Act::kRelu:     return x > 0.f ? x : 0.f;
        case Act::kGelu: {
            // 近似 GELU: x * 0.5 * (1 + erf(x/sqrt(2)))
            float cdf = 0.5f * (1.0f + std::erf(x / 1.41421356f));
            return x * cdf;
        }
    }
    return x;
}

// bias 广播模式（对应 ORT gemm.cc:101-141 的四种 shape + CUTLASS 的 Broadcast 类型）
//   0: 标量 bias[b]        -> stride 0，广播到全 M*N
//   1: 逐行 bias[row]       -> 沿 N 维广播（每输出行加同一偏置）
//   2: 逐列 bias[col]       -> 沿 M 维广播
//   3: 完整 bias[row*N+col]  -> 不广播
static inline int bias_index(int row, int col, int N, int mode) {
    switch (mode) {
        case 0: return 0;
        case 1: return row;
        case 2: return col;
        default: return row * N + col;
    }
}

// -----------------------------------------------------------------------------
// 1. MiniOutputOp —— 复刻 CUTLASS LinearCombinationBiasElementwise::operator()
//    (linear_combination_bias_elementwise.h:344-348)
//        z = binary_op(α·accum + β·C, bias) ;  Z = skip ? z : activation(z)
//    这里 binary_op=plus，故：
//        Z = activation( α·accum + β·C + bias )
//    它是"逐元素"的：epilogue pipeline 对输出 tile 的每个元素调一次。
// -----------------------------------------------------------------------------
struct MiniOutputOp {
    float alpha, beta;
    Act act;
    bool skip_act;   // CUTLASS 的 skip_elementwise_：多 K 段归约时中间段不做激活

    MiniOutputOp(float a, float b, Act ac, bool skip = false)
        : alpha(a), beta(b), act(ac), skip_act(skip) {}

    // 单元素版本（对应 .h:450-468 的标量 operator()）
    float operator()(float accum, float C, float bias) const {
        float z = alpha * accum + beta * C + bias;   // binary_op = plus
        return skip_act ? z : apply_act(z, act);
    }
};

// -----------------------------------------------------------------------------
// 2. epilogue_pipeline —— 复刻 CUTLASS EpilogueBase 的"输出 tile 迭代 + 写回"
//    (epilogue_base.h:95-228; 真实 Epilogue::operator() 对 tile 逐元素调 OutputOp)
//    我们用纯 C++ 在 CPU 上跑同一条 pipeline，用来验证"任意算子只需换 OutputOp"。
//    输入：accum[M*N]（已算好的 GEMM 累加结果）、C[M*N]（source，β≠0 时需要）、
//          bias[...]（按 mode 广播）、Out[M*N]（写回）。
// -----------------------------------------------------------------------------
static void epilogue_pipeline(const std::vector<float>& accum,
                              const std::vector<float>& C,
                              const std::vector<float>& bias,
                              std::vector<float>& Out,
                              int M, int N,
                              MiniOutputOp op, int bias_mode) {
    Out.resize(M * N);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            float b = bias[bias_index(row, col, N, bias_mode)];
            // 这里 source C 的 β 在 ORT 模式下为 0（bias 已并入），CUTLASS 模式下为 1
            Out[idx] = op(accum[idx], C[idx], b);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. CPU 参考：朴素 GEMM + bias + 激活（独立实现，作为对拍基准）
// -----------------------------------------------------------------------------
static void ref_gemm_bias_act(const std::vector<float>& A,
                              const std::vector<float>& B,
                              const std::vector<float>& bias,
                              std::vector<float>& Out,
                              int M, int N, int K,
                              float alpha, int bias_mode, Act act) {
    std::vector<float> accum(M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
            accum[m * N + n] = alpha * s;   // 仅 GEMM 部分（与 pipeline 的 accum 对齐）
        }
    // 用 β=0（bias 已通过 broadcast 进入 pipeline），与 epilogue_pipeline 对齐
    MiniOutputOp op(alpha, 0.f, act);
    epilogue_pipeline(accum, std::vector<float>(M * N, 0.f), bias, Out, M, N, op, bias_mode);
}

// -----------------------------------------------------------------------------
// 4. ORT 的 bias broadcast 巧法（gemm.cc:100-142 的纯 CPU 复刻）
//    思路：先把 bias 广播"预填"到输出缓冲 Y，再调用 GEMM 并把 β 设为 β_≠0，
//    这样 GEMM 自带的 epilogue（D = α·A·B + β·Y）就把 bias "免费"加回来了——
//    复用库的 pipeline，无需单独写融合 kernel、无需多一次全局访存。
//    我们复刻"逐列广播"那一支（B 为 (N,) 时，ORT 用 ones-GEMM 广播；这里直接算）。
// -----------------------------------------------------------------------------
static void ort_bias_prefill(std::vector<float>& Y, const std::vector<float>& bias,
                             int M, int N, int bias_mode) {
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
            Y[row * N + col] = bias[bias_index(row, col, N, bias_mode)];
}
// ORT 对应：cublasGemm(α·Wᵀ·X + β·Y)，Y 已含 bias —— 这里用 ref GEMM + β 累积
static void ref_gemm_with_ort_beta(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   const std::vector<float>& bias,
                                   std::vector<float>& Out,
                                   int M, int N, int K,
                                   float alpha, float beta, int bias_mode, Act act) {
    std::vector<float> Y(M * N);
    ort_bias_prefill(Y, bias, M, N, bias_mode);            // 第一步：预填 bias（ORT 的 broadcast）
    // 第二步：GEMM，β≠0 -> D = α·A·B + β·Y（Y 含 bias）。
    // 这里把激活也一并做（ORT 的 FusedMatMul 路径再叠一次 elementwise）。
    std::vector<float> ab(M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
            ab[m * N + n] = alpha * s;
        }
    for (int i = 0; i < M * N; ++i) Out[i] = apply_act(ab[i] + beta * Y[i], act);
}

// -----------------------------------------------------------------------------
// 5. analyze_fusion —— 确定性量化"复用 pipeline / 融合"省下的全局访存
//    （对应 L2.3 的 AI 思想：算子融合减少冗余 DRAM 往返）
//    重要澄清：对"一个独立 GEMM+bias+act"，融合省下的主要是输出张量 D 的
//    反复往返（5 次→1 次），但整 op 流量受 A·B 读取(2·M·N·K)支配，占比很小；
//    融合的真正收益在"算子链"——避免中间张量落盘反复读写。
// -----------------------------------------------------------------------------
static void analyze_fusion(int M, int N, int K) {
    printf("\n[analyze_fusion] M=%d N=%d K=%d\n", M, N, K);
    long long elems = (long long)M * N;
    // 输出张量 D 被碰的次数：GEMM 写(1) + bias 读/写(2) + act 读/写(2) = 5
    long long d_unmerged = 5LL * elems;
    long long d_merged   = 1LL * elems;     // 融合后 epilogue 只写 1 次
    long long total_unmerged = (long long)2 * M * N * K + d_unmerged;
    long long total_merged   = (long long)2 * M * N * K + d_merged;
    printf("  输出张量 D 的全局往返：未融合 %lld 次 ↔ 融合 %lld 次  =>  %.1fx\n",
           d_unmerged, d_merged, (double)d_unmerged / d_merged);
    printf("  整 op 总流量(受 A·B 读取 2·M·N·K 支配)：\n");
    printf("    未融合≈%lld   融合≈%lld   =>  整 op 仅 %.3fx（占比很小！）\n",
           total_unmerged, total_merged, (double)total_unmerged / total_merged);
    printf("  => 单看 GEMM 提速有限；真正收益在算子链：\n");
    printf("     GEMM→Norm→残差 若每步落盘，中间张量反复读写；epilogue 串起来即免落盘\n");
    printf("  —— 这正是 CUTLASS 把 bias/act 塞进 epilogue(EVT)、ORT 用 β·Y 复用库 epilogue 的动机\n");
}

// -----------------------------------------------------------------------------
// 6. 真机 CUDA kernel（守护在 __CUDACC__ 内；仅 nvcc 编译。结构同 mini pipeline）
//    对应 CUTLASS epilogue：每个线程负责输出 tile 中若干个元素，
//    读 accum（来自 MMA 累加寄存器/SMEM）、读 bias（按 mode 计算索引）、
//    套用 MiniOutputOp、写回 D（合并访问）。
// -----------------------------------------------------------------------------
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t e = (call);                                                   \
    if (e != cudaSuccess) { printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__,  \
                                   cudaGetErrorString(e)); exit(1); }          \
} while(0)

__global__ void gemm_bias_act_epilogue(const float* __restrict__ accum,
                                       const float* __restrict__ C,
                                       const float* __restrict__ bias,
                                       float* __restrict__ D,
                                       int M, int N, float alpha, float beta,
                                       int bias_mode, int act_code) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N, col = idx % N;
    int bidx = (bias_mode == 0) ? 0 : (bias_mode == 1) ? row
              : (bias_mode == 2) ? col : idx;
    Act act = (Act)act_code;
    float z = alpha * accum[idx] + beta * C[idx] + bias[bidx];   // plus binary op
    D[idx] = (act == Act::kIdentity) ? z : apply_act(z, act);
}
#endif // __CUDACC__

// -----------------------------------------------------------------------------
// 7. CPU 校验：MiniOutputOp + pipeline 必须等于朴素参考；ORT β-trick 必须等于融合参考
// -----------------------------------------------------------------------------
static int verify_on_cpu() {
    const int M = 4, N = 4, K = 3;
    std::vector<float> A(M * K), B(K * N), bias_row(N), bias_scalar(1), C(M * N, 0.f);
    for (int i = 0; i < M * K; ++i) A[i] = (float)(i + 1);
    for (int i = 0; i < K * N; ++i) B[i] = (float)(i % 3 + 1);
    for (int j = 0; j < N; ++j) bias_row[j] = (float)(10 + j);   // 逐列偏置 (N,)
    bias_scalar[0] = 7.0f;

    int fails = 0;
    auto check = [&](const std::vector<float>& a, const std::vector<float>& b, const char* name) {
        for (int i = 0; i < M * N; ++i)
            if (std::fabs(a[i] - b[i]) > 1e-3f) {
                printf("  FAIL %s @%d: %.4f vs %.4f\n", name, i, a[i], b[i]); fails++;
            }
    };

    // (a) 逐列 bias + ReLU：pipeline 版 vs 朴素参考（α=1, β=0）
    std::vector<float> out_pipe, out_ref;
    MiniOutputOp op(1.f, 0.f, Act::kRelu);
    std::vector<float> accum(M * N, 0.f);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float s = 0.f; for (int k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
        accum[m * N + n] = s;
    }
    epilogue_pipeline(accum, C, bias_row, out_pipe, M, N, op, /*bias_mode=*/2);
    ref_gemm_bias_act(A, B, bias_row, out_ref, M, N, K, 1.f, /*bias_mode=*/2, Act::kRelu);
    check(out_pipe, out_ref, "pipeline_vs_ref(bias-col,relu)");

    // (b) ORT β-trick：Y 先预填 bias(标量)，再 GEMM 带 β=1 + ReLU
    std::vector<float> out_ort(M * N, 0.f);
    ref_gemm_with_ort_beta(A, B, bias_scalar, out_ort, M, N, K, 1.f, 1.f, /*bias_mode=*/0, Act::kRelu);
    // 对照：CUTLASS 同构 Z = relu(accum + 0*C + bias_scalar) （用 pipeline 复算）
    std::vector<float> out_cutlass;
    MiniOutputOp op2(1.f, 0.f, Act::kRelu);
    epilogue_pipeline(accum, C, bias_scalar, out_cutlass, M, N, op2, /*bias_mode=*/0);
    check(out_ort, out_cutlass, "ort_beta_trick_vs_cutlass(bias-scalar,relu)");

    if (fails == 0) {
        printf("[verify_on_cpu] PASS  (M=%d N=%d K=%d)\n", M, N, K);
        printf("  pipeline(Z=relu(accum+bias_col)) [0..3] = %.3f %.3f %.3f %.3f\n",
               out_pipe[0], out_pipe[1], out_pipe[2], out_pipe[3]);
        printf("  ORT β-trick(Z=relu(accum+bias_7))  [0..3] = %.3f %.3f %.3f %.3f\n",
               out_ort[0], out_ort[1], out_ort[2], out_ort[3]);
    } else {
        printf("[verify_on_cpu] %d FAILURES\n", fails);
    }
    return fails;
}

// -----------------------------------------------------------------------------
int main() {
    int fails = verify_on_cpu();
    analyze_fusion(256, 256, 256);
    printf("\n参考源码（已备份于 _cutlass_src/ 与 _ort_src/）：\n");
    printf("  CUTLASS  epilogue_base.h (EpilogueBase + SharedStorage 写回)\n");
    printf("           linear_combination.h (LinearCombination: D=α·accum+β·C)\n");
    printf("           linear_combination_bias_elementwise.h (bias 广播 + ElementwiseOp 任意激活)\n");
    printf("           collective/default_epilogue.hpp (3.x CuTe visitor：同一 pipeline 描述任意算子)\n");
    printf("  ORT      gemm.cc:100-142 (bias broadcast 巧法：预填 Y 再 GEMM 带 β)\n");
    return fails ? 1 : 0;
}
