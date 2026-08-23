// ============================================================================
//  NEON 性能剖析 (profiling) 独立手册 + 可运行基准
//  本文件与 neon_l1_lut.cpp 分开，专讲"写对了更要量得对"。
//
//  编译:
//    ARM Linux : g++ -O3 -march=armv8-a+simd -std=c++17 -DRUN_BENCH neon_l2_profiling.cpp -o prof
//    x86 Win   : g++ -O3 -mssse3        -std=c++17 -DRUN_BENCH neon_l2_profiling.cpp -o prof.exe
//  运行: ./prof          (默认只打印用法)
//        ./prof bench    (或编译带 -DRUN_BENCH) 跑 add 带宽/加速比基准
//
//  配套 perf / simpleperf 命令见 §C~§I 注释。所有讲解对应你工作区 ncnn/src/layer/arm/ 的实现思路。
// ============================================================================


// ============================================================================
//  §A. 核心心法 —— 一个迭代闭环
// ============================================================================
//  写出来的 intrinsics ≠ 快。必须"量": 测墙钟 -> perf stat 分类瓶颈 -> 优化 ->
//  确认向量化真生效 -> 再测。对应流程图(见对话): 测 -> 分类 -> 优化 -> 确认 -> 迭代。

// ============================================================================
//  §B. 先会"正确测量" —— 避免假数据
// ============================================================================
//  * 用单调时钟 + 多次重复取中位数，不要单次 time()。CPU 频率会跳、调度会抖:
//        auto t0 = std::chrono::steady_clock::now();  /* 热循环 */  auto t1=...;
//        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
//  * 必须 warm-up: 正式计前先跑一遍把代码/数据塞进 I$/D$/L1，否则首轮含冷启动。
//  * 单位换算:
//        - 带宽 = 读写字节数 / 秒 (GB/s)，与内存理论带宽比，看访存是否打满。
//        - 吞吐 = 算术次数 / 秒 (GFLOP/s)，与峰值算力比，看算力是否打满。
//  * 陷阱 1: 编译器优化掉"结果没被使用"的计算 -> 把结果 print 出来或当 volatile 副作用。
//  * 陷阱 2: 小数据塞进 cache 后测的是 cache 速度不是内存速度 -> 数组要远大于 L3(本文件用 256MB)。
//  * 陷阱 3: 频率调节/后台进程/睿频 -> taskset 绑核、cpupower frequency-set 锁频更稳。

// ============================================================================
//  §C. perf stat —— 看硬件计数器 (Linux/ARM 必备)
// ============================================================================
//    perf stat -r 5 ./prof
//    关键看:
//      - instructions        : 总指令数。配合 cycles 算 IPC。
//      - cycles              : 总周期。
//      - IPC = instructions/cycles。NEON 友好代码应明显高于标量(标量常 <1 受访存/分支拖)；
//        但 IPC 高 ≠ 快(可能都在等内存)。
//      - cache-misses / cache-references : 缺失率。高 -> memory-bound，回看对齐/SoA。
//      - branch-misses      : 分支预测失败。高 -> 数据相关分支，用 mask 或改算法去分支。
//    NEON 专属 ARM PMU 事件:
//      perf stat -e armv8_pmuv3/instructions/ \
//                  -e armv8_pmuv3/l1d_cache/ -e armv8_pmuv3/l1d_cache_refill/ \
//                  -e armv8_pmuv3/br_mispredict/ ./prof
//      部分内核暴露 neon 类事件(取决于 Cortex 型号)；`perf list | grep -i neon` 查本机可用。
//    对比法: 同一算法跑标量版 vs NEON 版，看 instructions 是否降到 ~1/4、cycles 是否下降、
//            cache-misses 是否变化——若 instructions 没降，说明向量化没真发生。

// ============================================================================
//  §D. 确认向量化"真生效" —— 看汇编
// ============================================================================
//    objdump -d ./prof | grep -E 'fadd|fmla|v(add|mla|tbl)'
//    应出现 v 前缀的 NEON 指令(如 fadd v0.4s, v1.4s, v2.4s；fmla v6.4s, v8.4s, v0.s[0]；
//    vqtbl1q)。若只看到 s 前缀标量指令(s0/s1)，说明编译器没向量化或手写没进热路径。
//    另可用编译器报告: -fopt-info-vec / -Rpass=loop-vectorize 确认循环已向量化。

// ============================================================================
//  §E. 找热点 —— perf record / report / annotate
// ============================================================================
//   §C 的 perf stat 只告诉你"整体什么瓶颈"；要定位"具体哪个函数/哪行最热"，必须
//   用 perf record 采样 + perf report/annotate 钻取。这才是"找热点"的正解。
//
//   1) 采样抓热点 (带调用栈 -g，频率 -F 约 1 万次/秒):
//        perf record -g -F 9999 -o perf.data -- ./your_binary
//        perf report -i perf.data --stdio          # 函数按采样占比排序
//        perf report -i perf.data                  # 交互 TUI，回车展开调用链
//
//   2) 钻到汇编级 (NEON 验证关键一步):
//        perf annotate -i perf.data                # 汇编与源码混排，热点指令高亮
//      选中热点函数后看其热循环里是:
//        - v 前缀 (fmla v6.4s, v8.4s, v0.s[0] / vld1q)  -> 已在跑 NEON，向量化生效；
//        - s 前缀 (fadd s0, s1, s2 / ldr s0)            -> 还是标量，这就是该下手处。
//      (与 §D 的 objdump 互相印证: annotate 看"运行时热在哪行"，objdump 看"编出了什么指令")
//
//   3) 火焰图 (最直观的热点视图):
//        perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > f.svg
//      横轴栈深度、宽度=采样占比，一眼看出最宽的函数块。
//
//   4) NEON 专属计数器 (直接证明 SIMD 真的在执行):
//        perf list | grep -iE 'neon|simd|fp'
//        perf stat -e armv8_pmuv3/simd_inst_retired/ -e armv8_pmuv3/fp_spec/ ./bin
//      simd_inst_retired ≈ 0 说明你以为的向量化压根没发生；带 DOT 的核还有 dotprod 事件可数 sdot。
//
//   5) 热点指令判型:
//        大量时间花在 load/store -> memory-bound；
//        花在 fmla 排队等算子    -> 寄存器压力/依赖链；
//        花在分支                -> branch-bound。

// ============================================================================
//  §F. 分类决策表 (接回前面的课)
// ============================================================================
//    | 现象                            | 瓶颈类型      | 对策                 |
//    |---------------------------------+---------------+----------------------|
//    | IPC 低 + cache-misses 高        | Memory-bound  | 对齐/SoA/合并访问/prfm|
//    | IPC 中 + 时间花在 fmla 排队     | Compute-bound | fmla/寄存器分块/摊薄  |
//    | branch-misses 高                | Branch-bound  | mask 替代/去分支改算法|
//    | instructions 没降到 ~1/4        | 没向量化      | 查 -fopt-info 报告   |

// ============================================================================
//  §G. 工具箱一览
// ============================================================================
//    ARM/Linux : perf(万能)、Arm Streamline / Arm MAP(图形化、含 NEON 流水线视图)、
//                cachegrind(缓存模拟)、valgrind --tool=callgrind。
//    x86/Win   : Intel VTune(微架构级、内存/分支/向量化洞察极强)、WPT/ETW、
//                Visual Studio 内置 CPU 采样。
//    通用      : 自写 benchmark(见下方, 编译加 -DRUN_BENCH) + perf。


// ============================================================================
//  §H. Android simpleperf —— 设备端做同样的事
// ============================================================================
//  simpleperf 是 NDK 自带的 perf 移植版，底层同样是 perf_event_open() 内核接口；
//  语法与 §E 的 perf 几乎一致，只是运行在 Android 设备端，并多了一组自动取符号的 python 脚本。
//  位置: $ANDROID_NDK/simpleperf/ (含 simpleperf 二进制 + app_profiler.py / report_html.py /
//        binary_cache_builder.py)。
//
//   1) 一键录制 + 出 HTML 报告 (推荐):
//        python $ANDROID_NDK/simpleperf/app_profiler.py \
//            -p com.your.app \                          # 包名，或 -a 指定 activity
//            --native_lib_dir obj/local/arm64-v8a/ \    # 放带符号的 .so (NDK_DEBUG=1 编的)
//            -r "-g -f 9999 --duration 10"              # 透传给 simpleperf record 的参数
//        python $ANDROID_NDK/simpleperf/report_html.py  # 生成 report.html，浏览器打开
//      report_html.py 内含火焰图 + 函数列表 + 源码/汇编标注，等价于 §E 的 annotate/火焰图。
//
//   2) 手动流程 (理解原理):
//        adb shell simpleperf record -g -f 9999 -o /data/local/tmp/perf.data -- ./native_bin
//        adb shell simpleperf record -g -p <pid> --duration 10 -o /data/local/tmp/perf.data  # attach
//        adb pull /data/local/tmp/perf.data .
//        python $ANDROID_NDK/simpleperf/binary_cache_builder.py -i perf.data -lib <带符号so目录>
//        simpleperf report -i perf.data -g --stdio          # 命令行调用图
//        simpleperf report -i perf.data --children          # 含子函数开销
//
//   3) 与 perf 一一对应:
//        | 目的          | Linux perf                      | Android simpleperf                 |
//        |----------------+--------------------------------+-------------------------------------|
//        | 抓热点带调用栈 | perf record -g                 | simpleperf record -g              |
//        | 看函数占比     | perf report                    | simpleperf report -g              |
//        | 钻源码/汇编    | perf annotate                  | report_html.py 的 HTML 标注       |
//        | 火焰图         | perf script|flamegraph.pl      | report_html.py 内置              |
//        | 数 NEON 指令   | perf stat -e simd_inst_retired | simpleperf stat -e simd_inst_retired |
//        | 取符号         | 编译带 -g 即可                 | 必须额外给带符号 .so (脚本处理)   |
//
//   4) Android 两个特有坑:
//        - 符号被 strip: release 的 .so 去掉符号表，report 里只有地址。修法: 用 NDK_DEBUG=1
//          或保留 obj/local/arm64-v8a/ 下带符号的 .so 喂给脚本。这正是 simpleperf 要配 .py 的原因。
//        - 权限: 用户态采样一般不需 root；要数内核 PMU 事件或读 /sys/kernel/debug 才需 adb root。
//          多数 NEON 优化场景用非 root 用户态采样足够。

// ============================================================================
//  §I. NEON 局部加速闭环 —— 把三件工具串起来
// ============================================================================
//  perf/simpleperf 是"找该优化哪"的眼睛；-fopt-info 是"编译器为什么没优化"的病历
//  (见 neon_l1_optinfo.cpp)；<arm_neon.h> 是"动手优化"的手术刀。三者合起来就是你分级里
//  L2「会用 perf 找热点并局部加速」的完整闭环:
//
//    perf/simpleperf record -g  ->  定位热点函数 (看占比/火焰图)
//            |
//            v
//    perf/simpleperf annotate/HTML  ->  看那行是 v 前缀(NEON)还是 s 前缀(标量)
//            |
//            v
//    若是标量: -fopt-info-vec-missed 看报告 (neon_l1_optinfo.cpp)
//              -> 改 SoA(§3) / alignas(16)(面试要求) / __restrict(去别名) / 固定步长+补尾
//    若是 NEON 仍慢: 看 IPC/cache-miss (§C)
//              -> 查 对齐/合并/gather/寄存器分块/prfm/int8 SDOT (neon_l2_int8_gemm.cpp)
//            |
//            v
//    再 record 一遍  ->  确认热点下降，闭环
//
//  一句话记忆: 写对 ≠ 快，必须"量"。record 找热点、stat 分类瓶颈、annotate 验证向量化、
//  -fopt-info 查失败原因、intrinsics 动手——这条线走通，L2 的"局部加速"就达标了。

// ----------------------------------------------------------------------------
//  可运行基准 (自包含: 自带 add_scalar / add_simd + RUN_BENCH 开关)
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON) || \
    defined(_M_ARM) || defined(_M_ARM64)
  #include <arm_neon.h>
  #define BACKEND "NEON"
  #define HAVE_NEON 1
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || _M_IX86_FP >= 2))
    #include <emmintrin.h>
    #define BACKEND "SSE2"
    #define HAVE_SSE2 1
  #endif
#endif
#ifndef BACKEND
  #define BACKEND "scalar"
#endif

static void add_scalar(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
#if defined(HAVE_NEON)
static void add_simd(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
#elif defined(HAVE_SSE2)
static void add_simd(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
#else
static void add_simd(const float* a, const float* b, float* c, int n) { add_scalar(a,b,c,n); }
#endif

#ifdef RUN_BENCH
static void bench() {
    const size_t N = 1 << 26;                 // ~64M floats = 256MB，远超 L3
    std::vector<float> a(N), b(N), c(N);
    std::mt19937 rng(1);
    for (size_t i = 0; i < N; ++i) { a[i] = rng()*1e-3f; b[i] = rng()*1e-3f; }
    const int REPEAT = 20;
    auto time_it = [&](void(*fn)(const float*,const float*,float*,int)) {
        fn(a.data(), b.data(), c.data(), (int)N);   // warm-up
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < REPEAT; ++r) fn(a.data(), b.data(), c.data(), (int)N);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count() / REPEAT;
        double bytes = 3.0 * N * sizeof(float);     // 读 2N + 写 N 个 float
        return bytes / sec / 1e9;                   // GB/s
    };
    double gb_simd   = time_it(add_simd);
    double gb_scalar = time_it(add_scalar);
    printf("[基准] backend=%s  N=%zu (%.0f MB)\n", BACKEND, N, (double)N*4/1e6);
    printf("  add_simd   带宽: %.1f GB/s\n", gb_simd);
    printf("  add_scalar 带宽: %.1f GB/s\n", gb_scalar);
    printf("  加速比: %.2fx\n", gb_scalar / gb_simd);
}
#endif

int main(int argc, char** argv) {
    (void)argv;
#ifdef RUN_BENCH
    bench();
#else
    bool run = (argc > 1 && argv[1] && argv[1][0] == 'b'); // ./prof bench
    if (run) {
        // 编译时未开 -DRUN_BENCH 也能跑: 直接调用(但 bench 只在 RUN_BENCH 下定义)
        printf("未编译基准。请用 -DRUN_BENCH 重新编译，或运行 ./prof bench 且编译时带 -DRUN_BENCH。\n");
    } else {
        printf("NEON 性能剖析手册 (profiling)。\n");
        printf("  - 编译加 -DRUN_BENCH 再运行，可测 add 带宽与加速比。\n");
        printf("  - 配套 perf / simpleperf 命令见文件顶部 §C~§I 注释。\n");
        printf("  backend 探测 = %s\n", BACKEND);
    }
#endif
    return 0;
}
