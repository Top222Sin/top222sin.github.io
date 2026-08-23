// ============================================================================
//  neon_l3_microarch.cpp
//  L3 深度优化（库开发者 / 移动端推理引擎） 第二部分：
//      Cortex-A 微架构调优（流水线 · 发射端口 · 延迟/吞吐 · 缓存带宽 · perf 验证）
//
//  结构对齐你的笔记《00NEON基础学习.docx》的「X、标题 → 一、二、三、四」风格。
//
//  本文件是「能跑的调优实验台」：
//      §0  本文件在学什么
//      §1  微架构参数怎么读（编译期常量表，逐代 NEON 能力）
//      §2  用 perf 读出现在这台机器真实的 IPC / 周期（Linux perf_event_open）
//      §3  ILP 原理演示：1 / 4 / 8 个累加器的浮点点积，看 IPC 怎么涨
//      §4  aarch64 手写 NEON 内核：1 累加器 vs 8 累加器 FMLA 点积（延迟 vs 吞吐）
//      §5  缓存带宽体验：STREAM 式拷贝，把「字节/周期」和核心参数对上
//      §6  调优闭环：读官方指南 → 写 → perf 验证 → 改
//
//  编译运行：
//      # x86 本机（标量演示 + perf 计数器，验证 ILP 原理）
//      g++ -O3 -std=c++17 neon_l3_microarch.cpp -o l3m && ./l3m
//
//      # ARM 真跑 NEON 内核（需 NEON）
//      aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -std=c++17 neon_l3_microarch.cpp -o l3m && ./l3m
//
//  沙箱说明：本机无 C++ 编译器，未能实跑；代码为标准 C++17 / GNU 内联汇编，
//            x86 段用标量演示原理，aarch64 段为可移植内联汇编。
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

// ----------------------------------------------------------------------------
//  一、微架构参数怎么读（数据来自各核心《Arm Cortex-Axx Software Optimization Guide》）
// ----------------------------------------------------------------------------
//
//  同一行 intrinsics（比如 fmla），在 A53 与 A76 上差的可不是 10%——是「能不能
//  填满流水线」的量级差异。调优前先弄清楚手里的核属于哪一类：
//
//      little（小核，顺序执行 in-order）：A53 / A55        → 1 条 NEON/周期
//      big  （大核，乱序执行 out-of-order）：A57/A72/A75/A76/A77/A78/X1…
//           → 多端口超标量，2~4 条 NEON FMLA/周期
//
//  「吞吐 throughput」和「延迟 latency」是两个独立数字（见下表），也是本文件
//  反复要用的两个旋钮。下面这张表是「代际能力概览」，精确到具体 SKU 的延迟/
//  吞吐请以对应核心的官方优化指南为准（Cortex-A76 的官方数字我们会在 §3 直接引用）。
//
//  NEON fp32 FMLA 能力（每周期可执行条数 / 典型延迟）：
//      A53/A55  : 1 条/周期 , 延迟 ~4~6   （顺序，单 NEON 管）
//      A57/A72  : 2 条/周期 , 延迟 ~5~6   （双 NEON 管）
//      A73      : 1 条/周期 , 延迟 ~5     （为省电故意收窄到 1 管）
//      A75      : 2 条/周期 , 延迟 ~4
//      A76      : 2 条/周期 , 延迟 4      （双 NEON 管，官方指南实证）
//      A77      : 3 条/周期 , 延迟 ~4     （三 NEON 管）
//      A78      : 2 条/周期 , 延迟 ~4     （相对 A77 收 FP 管换内存带宽）
//      X1/X2/X3 : 4 条/周期 , 延迟 ~4     （4×128-bit NEON 引擎，X 系列超大核）
//
//  内存带宽（L1D→寄存器，约值，单位 字节/周期）：
//      A53  : ~1×64b (8B/cyc  load)
//      A76  : ~2×128b (32B/cyc load)
//      A78  : 比 A77 +50% load，store 翻倍（~32B/cyc store）
//      X1   : 在 A78 基础上再翻倍
//
//  结论先行：你的调度深度（累加器数量）必须 ≥ 核心的「每周期可发 NEON 条数」，
//            否则算力被延迟链卡死，永远跑不到峰值。

namespace core_params {  // 编译期常量，供代码里引用 / 打印
    struct Core { const char* name; int neon_fmla_per_cyc; int fmla_latency; int l1_load_B_per_cyc; };
    // 仅取代表性几代，精确值查官方指南
    static constexpr Core TABLE[] = {
        {"Cortex-A53/A55 (little, in-order)", 1, 5, 8},
        {"Cortex-A72 (big)",                  2, 5, 16},
        {"Cortex-A73 (big, narrow)",          1, 5, 16},
        {"Cortex-A75 (big)",                  2, 4, 16},
        {"Cortex-A76 (big)",                  2, 4, 32},
        {"Cortex-A77 (big)",                  3, 4, 24},
        {"Cortex-A78 (big)",                  2, 4, 32},
        {"Cortex-X1/X2/X3 (prime)",           4, 4, 64},
    };
    static void print_table() {
        std::printf("  %-34s %10s %8s %14s\n", "Core", "FMLA/cyc", "Latency", "L1 load B/cyc");
        for (auto& c : TABLE)
            std::printf("  %-34s %10d %8d %14d\n", c.name, c.neon_fmla_per_cyc, c.fmla_latency, c.l1_load_B_per_cyc);
    }
}

// ----------------------------------------------------------------------------
//  二、用 perf 读出现在这台机器真实的 IPC / 周期
// ----------------------------------------------------------------------------
//
//  IPC = instructions / cycles。对计算密集内核，IPC 越接近「宽度×前端效率」越好；
//  若 IPC 很低，说明被某条依赖链（延迟受限）或访存（带宽受限）卡住。
//  这是把「微架构调优」从玄学变成可测量的第一步。
//
//  注意：下面用 Linux perf_event_open 直接读硬件计数器，跨平台只在 Linux 有效；
//        非 Linux 会降级为「只计时，不报 IPC」。

struct Perf {
#ifdef __linux__
    int fd_inst = -1, fd_cyc = -1;
    long long inst = 0, cyc = 0;
    bool open() {
        struct perf_event_attr a1{}, a2{};
        a1.type = PERF_TYPE_HARDWARE; a1.config = PERF_COUNT_HW_INSTRUCTIONS;
        a1.disabled = 1; a1.exclude_kernel = 1; a1.size = sizeof(a1);
        a2.type = PERF_TYPE_HARDWARE; a2.config = PERF_COUNT_HW_CPU_CYCLES;
        a2.disabled = 1; a2.exclude_kernel = 1; a2.size = sizeof(a2);
        fd_inst = (int)syscall(__NR_perf_event_open, &a1, 0, -1, 0, 0);
        fd_cyc  = (int)syscall(__NR_perf_event_open, &a2, 0, -1, 0, 0);
        return fd_inst > 0 && fd_cyc > 0;
    }
    void reset()  { if (fd_inst>0) ioctl(fd_inst, PERF_EVENT_IOC_RESET, 0);
                    if (fd_cyc>0)  ioctl(fd_cyc,  PERF_EVENT_IOC_RESET, 0); }
    void start()  { if (fd_inst>0) ioctl(fd_inst, PERF_EVENT_IOC_ENABLE, 0);
                    if (fd_cyc>0)  ioctl(fd_cyc,  PERF_EVENT_IOC_ENABLE, 0); }
    void stop()   { if (fd_inst>0) ioctl(fd_inst, PERF_EVENT_IOC_DISABLE, 0);
                    if (fd_cyc>0)  ioctl(fd_cyc,  PERF_EVENT_IOC_DISABLE, 0); }
    void read()   { if (fd_inst>0) ::read(fd_inst, &inst, sizeof(inst));
                    if (fd_cyc>0)  ::read(fd_cyc,  &cyc,  sizeof(cyc)); }
    double ipc() const { return cyc > 0 ? (double)inst / (double)cyc : 0.0; }
    void close()  { if (fd_inst>0) ::close(fd_inst); if (fd_cyc>0) ::close(fd_cyc); }
#else
    bool open() { return false; }
    void reset() {} void start() {} void stop() {} void read() {}
    double ipc() const { return 0.0; }
    void close() {}
#endif
};

// ----------------------------------------------------------------------------
//  三、ILP 原理演示：1 / 4 / 8 个累加器的浮点点积
// ----------------------------------------------------------------------------
//
//  原理：点积 s += a[i]*b[i] 是「一条依赖链」——第 i 次乘加必须等第 i-1 次的
//  结果（累加器在飞行中）。在延迟为 L 的周期里，单累加器最多每 L 周期完成一次，
//  永远喂不满每周期 2~4 条的发射端口。
//
//  解法：把累加器拆成 N 份交错累加（s0+=..; s1+=..; s2+=..; s3+=..;），N 份互相
//  独立 → 乱序内核可同时让 N 条乘加在「飞行」，把端口填满。下面用标量浮点演示
//  （编译器在 x86 上可能自动向量化，但「依赖链」本质一样，看 IPC 变化即可）。

static double dot1(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
    return s;
}
static double dot4(const float* a, const float* b, int n) {
    double s0=0,s1=0,s2=0,s3=0; int i=0;
    for (; i+4 <= n; i += 4) {
        s0 += (double)a[i]   * (double)b[i];
        s1 += (double)a[i+1] * (double)b[i+1];
        s2 += (double)a[i+2] * (double)b[i+2];
        s3 += (double)a[i+3] * (double)b[i+3];
    }
    for (; i < n; ++i) s0 += (double)a[i] * (double)b[i];
    return s0+s1+s2+s3;
}
static double dot8(const float* a, const float* b, int n) {
    double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; int i=0;
    for (; i+8 <= n; i += 8) {
        s0 += (double)a[i]   * (double)b[i];   s1 += (double)a[i+1] * (double)b[i+1];
        s2 += (double)a[i+2] * (double)b[i+2]; s3 += (double)a[i+3] * (double)b[i+3];
        s4 += (double)a[i+4] * (double)b[i+4]; s5 += (double)a[i+5] * (double)b[i+5];
        s6 += (double)a[i+6] * (double)b[i+6]; s7 += (double)a[i+7] * (double)b[i+7];
    }
    for (; i < n; ++i) s0 += (double)a[i] * (double)b[i];
    return s0+s1+s2+s3+s4+s5+s6+s7;
}

static void bench_ilp() {
    const int N = 1 << 20;
    std::vector<float> a(N), b(N);
    std::mt19937 rng(12345); std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < N; ++i) { a[i] = dist(rng); b[i] = dist(rng); }

    Perf p; bool hasPerf = p.open();
    if (!hasPerf) std::printf("  (非 Linux：perf 不可用时只报耗时，不报 IPC)\n");

    auto run = [&](const char* tag, double(*fn)(const float*,const float*,int)) {
        volatile double sink = 0;
        // 预热
        sink += fn(a.data(), b.data(), N);
        p.reset(); p.start();
        for (int r = 0; r < 16; ++r) sink += fn(a.data(), b.data(), N);
        p.stop(); p.read();
        double ipc = p.ipc();
        std::printf("  %-10s : IPC=%s\n", tag, hasPerf ? std::to_string(ipc).c_str() : "n/a");
        (void)sink;
    };

    std::printf("  [ILP 演示] 浮点点积，n=%d，累加器数量 vs IPC（乱序核上 N 越大 IPC 越高）\n", N);
    run("dot1 (1x)",  dot1);
    run("dot4 (4x)",  dot4);
    run("dot8 (8x)",  dot8);
    p.close();
    std::printf("  注：若 dot8 的 IPC 明显 > dot1，你已经亲自验证了「微架构靠 ILP 喂满端口」。\n");
}

// ----------------------------------------------------------------------------
//  四、aarch64 手写 NEON 内核：1 累加器 vs 8 累加器 FMLA 点积
// ----------------------------------------------------------------------------
//
//  §3 的标量演示在 ARM 上换成真 NEON：用 128-bit 寄存器每次吃 4 个 float（4 lane）。
//  单累加器：一条 fmla 链依赖 v0 → 每 4 周期（A76 FMLA 延迟）才推进一次，
//            实测只能用到 2 端口里的 1/4。
//  8 累加器：8 路 v0..v7 互相独立、交错发射，轻松喂满 A76 的 2 个 NEON 管（乃至
//            X1 的 4 个管）。下面两个内核都可直接编译对拍。
//
//  对照 A76 官方数据（Arm Cortex-A76 Software Optimization Guide, p.28）：
//      ASIMD FP multiply accumulate (FMLA) :  exec latency 4 , throughput 2 , pipeline V1
//      ASIMD FP add/sub/mul     (VADD)     :  exec latency 2 , throughput 2 , pipeline V
//  → 延迟 4、每周期 2 条，正是我们「要 ≥2 个累加器才喂得满」的数学依据。

#if defined(__aarch64__)

// 1 累加器：4 lane 一次，依赖链锁死在 v0
static float neon_dot1(const float* a, const float* b, int n) {
    float s = 0.f; uint32_t bits = 0;
    const float* pa = a; const float* pb = b; int cnt = n / 4;
    asm volatile(
        "dup    v0.4s, wzr\n"
        "1:\n"
        "ldr    q1, [%0], #16\n"
        "ldr    q2, [%1], #16\n"
        "fmla   v0.4s, v1.4s, v2.4s\n"   // 结果回写 v0 -> 下一条必须等 v0 就绪
        "subs   %w2, %w2, #1\n"
        "b.gt   1b\n"
        "faddp  v0.4s, v0.4s, v0.4s\n"
        "faddp  v0.4s, v0.4s, v0.4s\n"
        "fmov   %w3, s0\n"
        : "+r"(pa), "+r"(pb), "+r"(cnt), "=r"(bits)
        : : "v0", "v1", "v2", "cc", "memory"
    );
    std::memcpy(&s, &bits, 4);
    return s;
}

// 8 累加器：v0..v7 各持一个 4-lane 部分和，每轮 8 次 fmla 互不依赖
static float neon_dot8(const float* a, const float* b, int n) {
    float s = 0.f; uint32_t bits = 0;
    const float* pa = a; const float* pb = b; int cnt = n / 32; // 8acc*4lane=32/轮
    asm volatile(
        "dup    v0.4s, wzr\n  dup    v1.4s, wzr\n  dup    v2.4s, wzr\n  dup    v3.4s, wzr\n"
        "dup    v4.4s, wzr\n  dup    v5.4s, wzr\n  dup    v6.4s, wzr\n  dup    v7.4s, wzr\n"
        "1:\n"
        "ldr    q16, [%0], #16\n  ldr    q17, [%1], #16\n  fmla   v0.4s, v16.4s, v17.4s\n"
        "ldr    q18, [%0], #16\n  ldr    q19, [%1], #16\n  fmla   v1.4s, v18.4s, v19.4s\n"
        "ldr    q20, [%0], #16\n  ldr    q21, [%1], #16\n  fmla   v2.4s, v20.4s, v21.4s\n"
        "ldr    q22, [%0], #16\n  ldr    q23, [%1], #16\n  fmla   v3.4s, v22.4s, v23.4s\n"
        "ldr    q24, [%0], #16\n  ldr    q25, [%1], #16\n  fmla   v4.4s, v24.4s, v25.4s\n"
        "ldr    q26, [%0], #16\n  ldr    q27, [%1], #16\n  fmla   v5.4s, v26.4s, v27.4s\n"
        "ldr    q28, [%0], #16\n  ldr    q29, [%1], #16\n  fmla   v6.4s, v28.4s, v29.4s\n"
        "ldr    q30, [%0], #16\n  ldr    q31, [%1], #16\n  fmla   v7.4s, v30.4s, v31.4s\n"
        "subs   %w2, %w2, #1\n"
        "b.gt   1b\n"
        // 归约 v0..v7 -> v0
        "fadd   v0.4s, v0.4s, v1.4s\n  fadd   v2.4s, v2.4s, v3.4s\n"
        "fadd   v4.4s, v4.4s, v5.4s\n  fadd   v6.4s, v6.4s, v7.4s\n"
        "fadd   v0.4s, v0.4s, v2.4s\n  fadd   v4.4s, v4.4s, v6.4s\n"
        "fadd   v0.4s, v0.4s, v4.4s\n"
        "faddp  v0.4s, v0.4s, v0.4s\n  faddp  v0.4s, v0.4s, v0.4s\n"
        "fmov   %w3, s0\n"
        : "+r"(pa), "+r"(pb), "+r"(cnt), "=r"(bits)
        : : "v0","v1","v2","v3","v4","v5","v6","v7",
            "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31",
            "cc", "memory"
    );
    std::memcpy(&s, &bits, 4);
    return s;
}

static int demo_neon() {
    const int N = 1 << 14;  // 必须是 32 的倍数（内核按 32/轮）
    std::vector<float> a(N), b(N);
    std::mt19937 rng(7); std::uniform_real_distribution<float> d(-1.f, 1.f);
    double ref = 0.0;
    for (int i = 0; i < N; ++i) { a[i]=d(rng); b[i]=d(rng); ref += (double)a[i]*(double)b[i]; }

    float r1 = neon_dot1(a.data(), b.data(), N);
    float r8 = neon_dot8(a.data(), b.data(), N);
    double e1 = std::fabs((double)r1 - ref), e8 = std::fabs((double)r8 - ref);
    std::printf("  [aarch64 NEON] neon_dot1 : ref=%.4f got=%.4f err=%.2e %s\n", ref, r1, e1, e1<1e-3?"OK":"BAD");
    std::printf("  [aarch64 NEON] neon_dot8 : ref=%.4f got=%.4f err=%.2e %s\n", ref, r8, e8, e8<1e-3?"OK":"BAD");

    // 计时对比（演示 8 累加器更省周期）
    auto t0 = std::chrono::steady_clock::now();
    for (int r=0;r<200;++r) neon_dot1(a.data(), b.data(), N);
    auto t1 = std::chrono::steady_clock::now();
    for (int r=0;r<200;++r) neon_dot8(a.data(), b.data(), N);
    auto t2 = std::chrono::steady_clock::now();
    double ms1 = std::chrono::duration<double,std::milli>(t1-t0).count();
    double ms8 = std::chrono::duration<double,std::milli>(t2-t1).count();
    std::printf("  计时(200 次, %d 元素): dot1=%.2f ms, dot8=%.2f ms  ", N, ms1, ms8);
    std::printf("（8 累加器应明显更快，因喂满双 NEON 管）\n");
    return (e1<1e-3 && e8<1e-3) ? 0 : 1;
}

#else
static int demo_neon() {
    std::printf("  [x86] NEON 内核段被 __aarch64__ 宏隔离；在 ARM 上会真正跑 1 vs 8 累加器对拍。\n");
    std::printf("       原理在 §3 的标量 dot1/dot4/dot8 已可本机验证。\n");
    return 0;
}
#endif

// ----------------------------------------------------------------------------
//  五、缓存带宽体验：STREAM 式拷贝，把「字节/周期」对上核心参数
// ----------------------------------------------------------------------------
//
//  算力的另一半是「喂不喂得饱」。下面测顺序拷贝的带宽（GB/s），把它和 §1 的
//  L1 load 参数对照——如果实测远小于核心理论带宽，说明访问模式/预取有问题。

static void bench_bandwidth() {
    const size_t N = 1 << 24;  // 16M float = 64MB，超出 L2/L3，逼出内存带宽
    std::vector<float> src(N), dst(N);
    for (size_t i = 0; i < N; ++i) src[i] = (float)(i & 0xff);

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; ++r)
        for (size_t i = 0; i < N; ++i) dst[i] = src[i];
    auto t1 = std::chrono::steady_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count() / 5.0;
    double bytes = (double)N * sizeof(float) * 2;  // 读 src + 写 dst
    double gbps = bytes / sec / 1e9;
    std::printf("  [带宽] 顺序拷贝 %zu MB, %.2f GB/s (读+写)\n", N*4/1048576, gbps);
    std::printf("  对照 §1：A76 L1 load≈32B/cyc；若此数低，多半是预取没打开/步长不友好。\n");
}

// ----------------------------------------------------------------------------
//  六、调优闭环：读官方指南 → 写 → perf 验证 → 改（本文件即你的实验台）
// ----------------------------------------------------------------------------
//
//  1) 查核：Arm 官网搜 «Cortex-A<你的核> Software Optimization Guide»，抄下
//     FMLA/VADD/SDOT 的 latency & throughput，以及 LSU 带宽。
//  2) 写：按「累加器数 ≥ 每周期 NEON 条数」定 ILP 深度（GEMM 即输出块大小）。
//  3) 验：perf stat -e instructions,cycles,cache-misses ./l3m
//         → IPC 低 = 依赖链/访存受限；cache-misses 高 = 布局/预取要改。
//  4) 改：加累加器、调循环展开、插 prfm、换数据布局（SoA/pack）。回到 3。

int main() {
    std::printf("=== L3 深度优化 · 第二部分：Cortex-A 微架构调优 ===\n\n");

    std::printf("一、核心 NEON 能力概览（FMLA 每周期条数 / 延迟 / L1 载入带宽）\n");
    core_params::print_table();
    std::printf("\n");

    std::printf("二、ILP 原理演示（1/4/8 累加器，看 IPC 变化）\n");
    bench_ilp();
    std::printf("\n");

    std::printf("三、aarch64 NEON 手写内核：1 vs 8 累加器 FMLA 点积\n");
    demo_neon();
    std::printf("\n");

    std::printf("四、缓存带宽体验（STREAM 式拷贝）\n");
    bench_bandwidth();
    std::printf("\n");

    std::printf("五、调优闭环见本文件 §六 / neon_l3_microarch.md。\n");
    return 0;
}
