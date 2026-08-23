// =============================================================================
// L2.3  Nsight Compute / Nsight Systems profiling —— 区分 compute-bound / memory-bound
// =============================================================================
// 配套文档：cuda_l2_profiling_bound.md
// 对应笔记本：01CUDA基础学习.docx → L2 能写高效 kernel → 3、Nsight profiling
//
// 本文件做两件事：
//  (A) 在 CPU 上**确定性**地算出两个对照 kernel 的「算术强度 AI = FLOPs / Byte」，
//      并用一个参考 GPU 的「平衡点 ridge point」判断它到底被算力还是带宽卡住。
//      —— 这一步不依赖 GPU，沙箱里也能跑通、能教学。
//  (B) 给出两个对照 kernel（内存密集 saxpy / 计算密集 ALU-heavy），真机用
//      ncu / nsys 实测时的命令与「该看哪些指标」。kernel 启动部分需真机 nvcc+cuda。
//
// 为什么这样设计：profiler 必须在真机跑；但「compute-bound vs memory-bound」的
// 判据（算术强度 + Roofline）本质是数学，可以先用 CPU 算清楚，再去真机用 ncu 验证。
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ---- CUDA 头（真机编译需要；沙箱只跑 main 里的 CPU 分析部分，编译用 nvcc 即可）----
#include <cuda_runtime.h>

// ==== 错误检查宏（沿用本目录一贯风格）====
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// =============================================================================
// 参考 GPU 规格（用 A100 量级做"方法"演示，换成你的卡改这两个数即可）
// =============================================================================
struct GpuSpec {
    const char* name;
    double fp32_tflops;   // 峰值 FP32 算力 (TFLOP/s)
    double mem_bw_tbs;    // 峰值显存带宽 (TB/s)
};

// A100：~19.5 TFLOP/s FP32，~1.555 TB/s HBM2e
static const GpuSpec kA100 = { "NVIDIA A100 (演示用)", 19.5, 1.555 };

// 平衡点（ridge point）= 峰值算力 / 峰值带宽，单位 FLOP/Byte。
// AI < ridge  → 被带宽卡住（memory-bound）
// AI > ridge  → 被算力卡住（compute-bound）
static double ridge_point(const GpuSpec& g) {
    return (g.fp32_tflops * 1e12) / (g.mem_bw_tbs * 1e12);  // TFLOP/s ÷ TB/s = FLOP/Byte
}

// =============================================================================
// 两个对照 kernel
// =============================================================================

// ---- ① 内存密集（memory-bound）：SAXPY ----
//   y[i] = a * x[i] + y[i]
//   每元素：读 x(4B) + 读 y(4B) + 写 y(4B) = 12 字节；算术 = 1 个 FMA = 2 FLOP
//   → AI = 2 / 12 ≈ 0.167 FLOP/Byte（极低 → 典型 memory-bound）
__global__ void saxpy_memory_bound(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// ---- ② 计算密集（compute-bound）：每线程只取 1 个 float，做 ITERS 轮纯 ALU ----
//   v = out[i]; s = Σ_{k=1..ITERS} (s + v*(v+k))
//   每元素：读 out(4B) + 写 out(4B) = 8 字节；算术 = 2 FLOP/迭代 × ITERS
//   → AI = (2*ITERS) / 8 = ITERS/4 FLOP/Byte（ITERS 越大越 compute-bound）
//   用纯 ALU（无超越函数）→ CPU 参考可对拍到逐位一致
__global__ void compute_bound(int n, float* out, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = out[i];
        float s = 0.f;
        for (int k = 1; k <= iters; ++k) {
            s = s + v * (v + (float)k);   // 2 FLOP：一次乘、一次加
        }
        out[i] = s;
    }
}

// =============================================================================
// CPU 参考（用于对拍正确性）
// =============================================================================
static void cpu_saxpy(int n, float a, const float* x, float* y) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}
static void cpu_compute_bound(int n, float* out, int iters) {
    for (int i = 0; i < n; ++i) {
        float v = out[i];
        float s = 0.f;
        for (int k = 1; k <= iters; ++k) s = s + v * (v + (float)k);
        out[i] = s;
    }
}

// =============================================================================
// (A) 确定性分析：算术强度 + Roofline 判定（CPU 即可运行）
// =============================================================================
static void analyze_kernels() {
    printf("==================== (A) 确定性分析：算术强度 AI 与 Roofline ====================\n");
    const int N = 1 << 20;          // 1048576 元素，仅用于"比率"，与具体规模无关
    const int ITERS = 64;           // compute_bound 每元素迭代次数

    double flops_saxpy = 2.0 * N;               // FMA = 2 FLOP/元素
    double bytes_saxpy = 12.0 * N;              // 读x+读y+写y = 12B/元素
    double ai_saxpy = flops_saxpy / bytes_saxpy;

    double flops_cb = 2.0 * ITERS * N;          // 2 FLOP/迭代 × ITERS × N
    double bytes_cb = 8.0 * N;                  // 读out+写out = 8B/元素
    double ai_cb = flops_cb / bytes_cb;

    double ridge = ridge_point(kA100);

    printf("参考 GPU：%s\n", kA100.name);
    printf("  峰值 FP32 = %.2f TFLOP/s，峰值显存带宽 = %.3f TB/s\n", kA100.fp32_tflops, kA100.mem_bw_tbs);
    printf("  ▶ 平衡点 ridge point = 算力/带宽 = %.2f FLOP/Byte\n", ridge);
    printf("    （AI < %.2f → memory-bound；AI > %.2f → compute-bound）\n\n", ridge, ridge);

    printf("① saxpy（内存密集）：\n");
    printf("    FLOPs = 2*N = %.3e；Bytes = 12*N = %.3e\n", flops_saxpy, bytes_saxpy);
    printf("    AI = FLOPs/Bytes = %.4f FLOP/Byte  → %s\n",
           ai_saxpy, (ai_saxpy < ridge ? "** memory-bound **" : "compute-bound"));
    printf("    Roofline 预言：受带宽限制，理论峰值 ≈ BW × AI = %.3e × %.4f ≈ %.2f GFLOP/s\n\n",
           kA100.mem_bw_tbs * 1e12, ai_saxpy, kA100.mem_bw_tbs * 1e12 * ai_saxpy / 1e9);

    printf("② compute_bound(iters=%d)（计算密集）：\n", ITERS);
    printf("    FLOPs = 2*%d*N = %.3e；Bytes = 8*N = %.3e\n", ITERS, flops_cb, bytes_cb);
    printf("    AI = FLOPs/Bytes = %.2f FLOP/Byte  → %s\n",
           ai_cb, (ai_cb < ridge ? "memory-bound" : "** compute-bound **"));
    printf("    Roofline 预言：AI > ridge，理论由算力主导 ≈ up to %.2f TFLOP/s\n\n",
           kA100.fp32_tflops);

    printf("结论：两个 kernel 的『bound 属性』由算术强度决定，与具体规模无关。\n");
    printf("      真机上用 ncu 测到的 DRAM 带宽利用率 / SM 算力利用率会印证这一判定。\n");
    printf("=============================================================================\n\n");
}

// =============================================================================
// (B) profiler 命令速查（打印出来，真机照抄即可）
// =============================================================================
static void print_profiler_commands() {
    printf("==================== (B) Nsight 实测命令速查 ====================\n");
    printf("# 1) Nsight Compute（逐 kernel 深层指标）—— 看 bound 属性首选\n");
    printf("ncu --set full -k saxpy_memory_bound \\\n");
    printf("    -o report_mem ./prof_bound    # 生成 report_mem.ncu-rep，GUI 打开\n");
    printf("ncu --set full -k compute_bound -o report_compute ./prof_bound\n\n");

    printf("# 2) 只看几个关键 bound 指标（快、不用 full）\n");
    printf("ncu --metrics \\\n");
    printf("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");   // compute (SM) throughput
    printf("  dram__throughput.avg.pct_of_peak_sustained_elapsed,\\\n");  // DRAM bandwidth util
    printf("  l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\\\n"); // L2/L1 tex throughput
    printf("  sm__warps_active.avg.pct_of_peak_sustained_active,\\\n");   // achieved occupancy
    printf("  gpu__time_duration \\\n");
    printf("  ./prof_bound\n\n");

    printf("# 3) Nsight Systems（系统级时间线）—— 看 CPU/GPU 重叠、空隙、API 开销\n");
    printf("nsys profile -o timeline ./prof_bound\n");
    printf("  # 打开 timeline.qdrep：关注 kernel 持续时长、kernel 之间是否有空隙、\n");
    printf("  # host 侧是否有长 CPU 段导致 GPU 饿着（绿条断开 = GPU 在等 host）\n");
    printf("==============================================================\n\n");
}

// =============================================================================
// (C) 真机 GPU 校验（需 CUDA Toolkit；沙箱里这段不会被执行/编译后运行需 GPU）
// =============================================================================
static int run_gpu_checks() {
    const int N = 1 << 20;
    const int ITERS = 64;
    const float a = 2.0f;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float *h_x, *h_y, *h_y_ref, *d_x, *d_y, *d_out;
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));
    h_y_ref = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) { h_x[i] = (float)i; h_y[i] = (float)(2 * i); }

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // kernel ① saxpy
    saxpy_memory_bound<<<blocks, threads>>>(N, a, d_x, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_saxpy(N, a, h_x, h_y_ref);
    int ok1 = 1; for (int i = 0; i < N; ++i) if (fabsf(h_y[i] - h_y_ref[i]) > 1e-3f) { ok1 = 0; break; }
    printf("[GPU] saxpy_memory_bound 正确性：%s\n", ok1 ? "PASS" : "FAIL");

    // kernel ② compute_bound
    float *h_out = (float*)malloc(N * sizeof(float));
    float *h_out_ref = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) { h_out[i] = (float)i; h_out_ref[i] = (float)i; }
    CUDA_CHECK(cudaMemcpy(d_out, h_out, N * sizeof(float), cudaMemcpyHostToDevice));
    compute_bound<<<blocks, threads>>>(N, d_out, ITERS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_compute_bound(N, h_out_ref, ITERS);
    int ok2 = 1; for (int i = 0; i < N; ++i) if (fabsf(h_out[i] - h_out_ref[i]) > 1e-3f) { ok2 = 0; break; }
    printf("[GPU] compute_bound 正确性：%s\n", ok2 ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_y_ref); free(h_out); free(h_out_ref);
    return (ok1 && ok2) ? 0 : 1;
}

// =============================================================================
int main() {
    analyze_kernels();          // (A) 确定性：AI + Roofline（CPU 可跑，沙箱也能教学）
    print_profiler_commands();  // (B) 真机 ncu/nsys 命令速查
    // (C) 真机 GPU 校验：需 nvcc+cuda+GPU；沙箱无 GPU 时本段编译通过但运行会停在 cudaMalloc。
    //     直接放开下面这行即可在真机完整对拍：
    // return run_gpu_checks();
    printf("[提示] 真机运行请取消 main 末尾 run_gpu_checks() 的注释，用：\n");
    printf("        nvcc -O3 -std=c++17 cuda_l2_profiling_bound.cpp -o prof && ./prof\n");
    return 0;
}
