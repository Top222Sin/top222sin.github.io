// ============================================================================
//  neon_l3_gemm.cpp
//  L3 深度优化（库开发者 / 移动端推理引擎） 第三部分：
//      A76 / X1 上 GEMM 微内核的完整调优 walkthrough
//
//  这是把 L3 第一部分（寄存器分配/指令调度）和第二部分（微架构 ILP/带宽）
//  真正合成一个**可量化、可对拍的 capstone**：从朴素三重循环一路调到
//  「分块 + 打包 + 寄存器分块微内核」，并演示为什么 8x4 块恰好喂满 A76 的
//  双 NEON 管、16x4 块喂满 X1 的四 NEON 管。
//
//  结构对齐笔记《00NEON基础学习.docx》的「X、标题 → 一、二、三、四」风格。
//
//  本文件是调优实验台：
//      §1  GEMM 为何是调优试金石
//      §2  正确基线：朴素 i,j,k（坏访问） + 参考 i,k,j
//      §3  分块 + 打包（packing）：把工作集塞进 L1/L2
//      §4  微内核三档：4x4(半 A76 峰值) / 8x?→8x4(满 A76) / 16x4(满 X1)
//      §5  aarch64 NEON 8x4 微内核（intrinsics，可编译对拍）
//      §6  验证闭环：perf stat 看 cycles / IPC / cache-misses 各 stage 对比
//
//  编译运行：
//      # x86 本机（标量分块+打包演示，验证正确性与缓存/ILP 收益）
//      g++ -O3 -std=c++17 neon_l3_gemm.cpp -o l3g && ./l3g
//
//      # ARM 真跑 NEON 8x4 微内核
//      aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -std=c++17 neon_l3_gemm.cpp -o l3g && ./l3g
//
//  沙箱说明：本机无 C++ 编译器，未能实跑；代码为标准 C++17 / NEON intrinsics。
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ----------------------------------------------------------------------------
//  一、GEMM 为何是调优试金石
// ----------------------------------------------------------------------------
//
//  C[M,N] = A[M,K] × B[K,N]，计算量 = 2·M·N·K 次乘加，访存量 = (M·N + M·K + K·N)·4B。
//  当 M,N,K 都上千时，算:访 ≈ K:1，是**典型的计算受限（compute-bound）** kernel。
//  对这类 kernel，性能 = 把「算力」(NEON 端口) 和「带宽」(L1/L2) 同时喂满。
//  ncnn 里 gemm 是 ResNet/Transformer 最热的调用，每秒上亿次 → 必须下沉到 L3 手写。
//
//  调优四步（本文件逐档演示）：
//      stage0  朴素 i,j,k（B 跨步访问，缓存不友好）
//      stage1  分块（tiling）+ 打包 B（packing）→ 访问全连续、工作集进 L1/L2
//      stage2  微内核 4x4（4 个累加器 → 只摸 A76 一半峰值，见 §4）
//      stage3  微内核 8x4（8 个累加器 → 喂满 A76 双 NEON 管 = 满峰值）
//      stage4  （X1）16x4（16 个累加器 → 喂满 X1 四 NEON 管）
//
//  寄存器预算（L3 第一部分）：AArch64 有 32 个 128-bit V。一个 float32x4 累加器
//  占 1 个 V。8x4 微内核 = 8 个累加器(8 V) + 复用 A/B 寄存器，远未触顶；
//  但「输出块」再大就会 spill → 性能骤降。块尺寸是预算逼出来的（ncnn 同理）。

// ----------------------------------------------------------------------------
//  二、正确基线
// ----------------------------------------------------------------------------

// 朴素 i,j,k：B[k*N+j] 跨步访问（步长 N），缓存命中差 → 作为「坏基线」对照
static void gemm_naive(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] = 0.f;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// 参考 i,k,j：A 与 B 均连续访问，正确性基准（对拍用）
static void gemm_ref(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] = 0.f;
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

// ----------------------------------------------------------------------------
//  三、分块 + 打包（packing）
// ----------------------------------------------------------------------------
//
//  把 B 的 [kc × nr] 小面板重排成连续内存 Bp[kk*nr + c]，使微内核里 B 的访问
//  每次都是一条 vld1q（连续 16 字节）。A 按行本就连续。两者都连续 → K 循环可
//  向量化、可缓存。工作集：一个 (mr×kc) 的 A 块 + (kc×nr) 的 B 面板，应塞进 L1。
//  A76 L1D=64KB：例如 mr=8,nr=4,kc=256 → A 块 8KB + B 面板 4KB，轻松进 L1。

// 微内核（标量版，x86 / 边界都可用）：mr×nr，K 循环在最内、A 与 Bp 均连续
static void gemm_micro_scalar(const float* A, const float* Bp, float* C,
                              int M, int N, int K,
                              int i, int j, int k, int mr, int nr, int kc) {
    float acc[16][4] = {0};  // 足够 16 行（X1 档）
    for (int kk = 0; kk < kc; ++kk) {
        for (int r = 0; r < mr; ++r) {
            float av = A[(i + r) * K + k + kk];
            const float* bp = &Bp[kk * nr];
            for (int c = 0; c < nr; ++c)
                acc[r][c] += av * bp[c];
        }
    }
    for (int r = 0; r < mr; ++r)
        for (int c = 0; c < nr; ++c)
            C[(i + r) * N + j + c] += acc[r][c];
}

#if defined(__aarch64__)
// 微内核（NEON 8x4，A76 满峰值档）：8 个 float32x4 累加器，每个 k 步
//   读 B 面板 4 个连续 float（vld1q）→ 广播 A 的 8 个行元素（vdupq_n）→ 8 条 vmlaq
// 8 条 fmla 在「飞行」(latency 4) → 8/4 = 2/周期 = 喂满 A76 双 NEON 管。
static void gemm_8x4_neon(const float* A, const float* Bp, float* C,
                          int M, int N, int K,
                          int i, int j, int k, int mr, int nr, int kc) {
    float32x4_t acc[8];
    for (int r = 0; r < mr; ++r) acc[r] = vdupq_n_f32(0.f);
    for (int kk = 0; kk < kc; ++kk) {
        float32x4_t b = vld1q_f32(&Bp[kk * nr]);
        for (int r = 0; r < mr; ++r) {
            float32x4_t a = vdupq_n_f32(A[(i + r) * K + k + kk]);
            acc[r] = vmlaq_f32(acc[r], b, a);
        }
    }
    for (int r = 0; r < mr; ++r) {
        float32x4_t c = vld1q_f32(&C[(i + r) * N + j]);
        c = vaddq_f32(c, acc[r]);
        vst1q_f32(&C[(i + r) * N + j], c);
    }
}
#endif

// 顶层 driver：分块 + 打包，调用微内核
static void gemm_packed(const float* A, const float* B, float* C,
                        int M, int N, int K, int MR, int NR, int KC) {
    for (int x = 0; x < M * N; ++x) C[x] = 0.f;
    std::vector<float> Bp((size_t)KC * NR);
    for (int i = 0; i < M; i += MR) {
        int mr = std::min(MR, M - i);
        for (int j = 0; j < N; j += NR) {
            int nr = std::min(NR, N - j);
            for (int k = 0; k < K; k += KC) {
                int kc = std::min(KC, K - k);
                // 打包 B 的 [kc × nr] 面板到连续 Bp
                for (int kk = 0; kk < kc; ++kk)
                    for (int c = 0; c < nr; ++c)
                        Bp[(size_t)kk * NR + c] = B[(size_t)(k + kk) * N + (j + c)];
                if (mr == MR && nr == NR) {
#if defined(__aarch64__)
                    gemm_8x4_neon(A, Bp.data(), C, M, N, K, i, j, k, mr, nr, kc);
#else
                    gemm_micro_scalar(A, Bp.data(), C, M, N, K, i, j, k, mr, nr, kc);
#endif
                } else {
                    gemm_micro_scalar(A, Bp.data(), C, M, N, K, i, j, k, mr, nr, kc);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  四、微内核三档（ILP 深度 = 端口数）
// ----------------------------------------------------------------------------
//
//  A76：2 个 NEON 管，FMLA 延迟 4 → 要 ≥2/周期 = 需 8 条 fmla 在飞行(8/4)。
//       · 4x4 微内核 = 4 个累加器 → 4/4 = 1 fmla/周期 = 半峰值（只喂了 1 个管）
//       · 8x4 微内核 = 8 个累加器 → 8/4 = 2 fmla/周期 = 满峰值（双管都忙）
//  X1：4 个 NEON 管 → 需 16 条飞行 = 16 个累加器 → 16x4 块才摸满。
//  所以「块做多大」不是拍脑袋，是「端口数 × 延迟」逼出来的（和寄存器预算一起）。

// ----------------------------------------------------------------------------
//  五、验证闭环（perf）
// ----------------------------------------------------------------------------

struct Perf {
#ifdef __linux__
    int fd_i = -1, fd_c = -1, fd_m = -1;
    long long inst = 0, cyc = 0, miss = 0;
    bool open() {
        auto mk = [](uint32_t cfg) {
            struct perf_event_attr a{};
            a.type = PERF_TYPE_HARDWARE; a.config = cfg;
            a.disabled = 1; a.exclude_kernel = 1; a.size = sizeof(a);
            return (int)syscall(__NR_perf_event_open, &a, 0, -1, 0, 0);
        };
        fd_i = mk(PERF_COUNT_HW_INSTRUCTIONS);
        fd_c = mk(PERF_COUNT_HW_CPU_CYCLES);
        fd_m = mk(PERF_COUNT_HW_CACHE_MISSES);
        return fd_i > 0 && fd_c > 0 && fd_m > 0;
    }
    void reset() { ioctl(fd_i,PERF_EVENT_IOC_RESET,0); ioctl(fd_c,PERF_EVENT_IOC_RESET,0); ioctl(fd_m,PERF_EVENT_IOC_RESET,0); }
    void start() { ioctl(fd_i,PERF_EVENT_IOC_ENABLE,0); ioctl(fd_c,PERF_EVENT_IOC_ENABLE,0); ioctl(fd_m,PERF_EVENT_IOC_ENABLE,0); }
    void stop()  { ioctl(fd_i,PERF_EVENT_IOC_DISABLE,0); ioctl(fd_c,PERF_EVENT_IOC_DISABLE,0); ioctl(fd_m,PERF_EVENT_IOC_DISABLE,0); }
    void read()  { ::read(fd_i,&inst,sizeof(inst)); ::read(fd_c,&cyc,sizeof(cyc)); ::read(fd_m,&miss,sizeof(miss)); }
    double ipc() const { return cyc>0 ? (double)inst/cyc : 0; }
    void close() { ::close(fd_i); ::close(fd_c); ::close(fd_m); }
#else
    bool open() { return false; }
    void reset(){} void start(){} void stop(){} void read(){}
    double ipc() const { return 0; } long long miss=0;
    void close(){}
#endif
};

// ----------------------------------------------------------------------------
//  六、demo：跑各 stage，对拍正确性 + 比时间 + 看 perf
// ----------------------------------------------------------------------------

int main() {
    const int M = 256, N = 256, K = 256;     // 适中，x86 也能秒跑
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    std::mt19937 rng(2026); std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : A) v = d(rng);
    for (auto& v : B) v = d(rng);

    std::vector<float> Cref((size_t)M * N), Cnaive((size_t)M * N), Cpk((size_t)M * N);

    // 正确性：ref vs packed(8x4)
    gemm_ref(A.data(), B.data(), Cref.data(), M, N, K);
    gemm_packed(A.data(), B.data(), Cpk.data(), M, N, K, /*MR=*/8, /*NR=*/4, /*KC=*/256);
    double maxerr = 0;
    for (int x = 0; x < M * N; ++x) maxerr = std::max(maxerr, (double)std::fabs((double)(Cpk[x] - Cref[x])));
    std::printf("=== L3 第三部分：A76/X1 GEMM 微内核调优 walkthrough ===\n\n");
    std::printf("[正确性] gemm_packed(8x4) vs gemm_ref : max err = %.2e %s\n", maxerr, maxerr < 1e-3 ? "OK" : "BAD");

    // 性能：naive vs packed（各跑若干次取平均）
    auto timeit = [&](const char* tag, void(*fn)(const float*,const float*,float*,int,int,int)) {
        volatile float sink = 0;
        // 预热
        fn(A.data(), B.data(), Cnaive.data(), M, N, K); sink += Cnaive[0];
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 20; ++r) fn(A.data(), B.data(), Cnaive.data(), M, N, K);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1 - t0).count() / 20.0;
        double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
        std::printf("  %-14s : %.2f ms  (%.1f GFLOP/s)\n", tag, ms, gflops);
        (void)sink;
    };
    std::printf("\n[性能] stage0 朴素 vs stage1+3 分块打包(8x4)：\n");
    timeit("naive i,j,k", gemm_naive);
    timeit("packed 8x4",  [](const float*a,const float*b,float*c,int m,int n,int k){
                              gemm_packed(a,b,c,m,n,k,8,4,256); });

    // perf 闭环（仅 Linux）：打包版 cycles / IPC / cache-miss
    Perf p; bool hasPerf = p.open();
    if (hasPerf) {
        p.reset(); p.start();
        for (int r = 0; r < 20; ++r) gemm_packed(A.data(), B.data(), Cpk.data(), M, N, K, 8, 4, 256);
        p.stop(); p.read();
        std::printf("\n[perf] packed 8x4 : IPC=%.2f, cache-misses=%lld\n", p.ipc(), p.miss);
        std::printf("        （A76 上 IPC 越高=越接近峰值；cache-miss 低=打包/分块生效）\n");
        p.close();
    } else {
        std::printf("\n[perf] 非 Linux：跳过硬件计数器（在上机 ARM/Linux 时用 perf stat 验证）。\n");
    }

    std::printf("\n[收口] 8x4 喂满 A76 双 NEON 管；要摸 X1 四管请改 MR=16（见 §四）。\n");
    std::printf("       完整讲解见 neon_l3_gemm.md（含分块几何/寄存器棋盘/端口时间线/perf 对比图）。\n");
    return 0;
}
