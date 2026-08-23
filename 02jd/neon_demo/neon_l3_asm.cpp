// ============================================================================
//  neon_l3_asm.cpp
//  L3 深度优化（库开发者 / 移动端推理引擎） 第一部分：
//      手写 NEON 汇编 · 指令调度 · 寄存器分配
//
//  结构对齐你的笔记《00NEON基础学习.docx》的「X、标题 → 一、二、三、四」风格。
//  所有引用都指向你工作区里真实的 ncnn 源码：
//      ncnn/src/layer/arm/gemm_arm.cpp        (fp32 GEMM 微内核, 约 2455 行)
//      ncnn/src/layer/arm/gemm_int8.h         (int8 SDOT 微内核, 约 10018 行)
//      ncnn/src/layer/arm/gemm_arm_asimddp.cpp (薄包装, 转交 gemm_int8.h)
//
//  编译运行：
//      # ARM (真·跑手写汇编, 含最小内核 vadd4_asm 对拍)
//      aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -std=c++17 neon_l3_asm.cpp -o l3
//      ./l3
//      # x86 上本文件只打印 "aarch64 only" 提示（手写汇编段被宏隔离），但注释全文可查阅
//
//  沙箱说明：本机无 C++ 编译器，未能实跑；aarch64 段为标准 GNU 内联汇编，可移植。
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// ----------------------------------------------------------------------------
//  一、为什么不在 intrinsics —— 分水岭在哪
// ----------------------------------------------------------------------------
//
//  L2 用 <arm_neon.h> intrinsics：你「告诉编译器我想干什么」，寄存器分配、
//  指令顺序、是否 spill 全由编译器决定。对绝大多数热点这够了。
//
//  L3 手写汇编：你「自己决定每条指令、每个寄存器、每个周期」。
//  分水岭 = 微内核被调用的频率。一个 GEMM / Conv 微内核每秒被调用上亿次，
//  编译器保守分配带来的每一次 spill（寄存器溢出到栈）、每一条多余的 vst、
//  每一个没排好的依赖停顿，都是实打实的性能损失，且无法靠 -O3 消除。
//
//  证据：ncnn 同一份 GEMM 同时保留了 intrinsics 版和手写汇编版，用宏切换：
//
//      // ncnn/src/layer/arm/gemm_int8.h 风格（伪代码示意）
//      #if NCNN_GNU_INLINE_ASM
//          asm volatile("sdot v16.4s, v0.16b, v4.4b[0] ...");   // ← L3 手写汇编
//      #else
//          _sum0 = vdotq_lane_s32(_sum0, va, vb, 0);             // ← L2 intrinsics
//      #endif
//
//  结论：先 intrinsics 跑通正确性，再对「真·热」的微内核下沉到汇编微调。
//        不是炫技，是账单逼的。

// ----------------------------------------------------------------------------
//  二、寄存器分配（Register Allocation）
// ----------------------------------------------------------------------------
//
//  AArch64 棋盘：32 个 128-bit V 寄存器（V0–V31）。
//  AArch32 只有 32 个 64-bit D（或 16 个 128-bit Q0–Q15）—— 这正是
//  gemm_arm.cpp 里那份 aarch32 内联汇编用 q0–q2 / d0–d7 的原因，架构不同棋盘更小。
//
//  AAPCS64 调用约定决定的「免保存区」（ncnn 故意利用）：
//      V0–V7    调用方保存（参数/返回值） → 叶子函数里随便用
//      V8–V15   被调方保存               → 用了就得 stp/ldp 压栈恢复（有开销）
//      V16–V31  调用方保存               → 叶子函数里随便用
//
//  ncnn 的分配账本（见 neon_l3_asm_diagrams.html 图一）：
//      累加器    → 全用 v16–v31（16 个 int32x4 = 64 个部分和 → 8×8 输出块的一个切分）
//      数据块 A/B → 用 v0–v7（8 个 128-bit）
//      v8–v15    → 故意留空！不是忘了，是为「不付保存代价」主动让出的 8 个槽
//  → 整个微内核零寄存器保存/恢复，纯叶子效率。
//
//  ★ 输出块尺寸 = 寄存器账单逼出来的：块再大 → 累加器数 > 32 → spill 到栈
//    → 直接慢一个数量级。所以「这个 GEMM 算 8×几的块」不是数学选择，是预算决策。
//
//  内联汇编里「钉」寄存器（gemm_arm.cpp:2455 约束表）：
//      : "=w"(_sum0), ..."=w"(_sum7)        // 输出: 钉到某 w(向量)寄存器
//      : "0"(pA), "1"(pB), "2"(_sum0), ..."9"(_sum7)
//                                          // "2"(..) = 该输入复用输出操作数 #2 的同寄存器
//      : "memory", "v0", "v1", "v2", "v3"   // 破坏列表: 内存被改 / 这些寄存器被踩
//  "2"(_sum0) 的含义：让这个输入操作数使用「和输出操作数 #2 相同的物理寄存器」，
//  从而跨 asm 块把累加器钉死、不挪窝。

// ----------------------------------------------------------------------------
//  三、指令调度（Instruction Scheduling）
// ----------------------------------------------------------------------------
//
//  两个独立指标（Cortex-A 是超标量、乱序内核）：
//      延迟 latency  ：结果多久后能被下一条指令使用（FMLA 约 4 周期）
//      吞吐 throughput：每周期最多发几条同类指令（FMLA 约 2/周期）
//  陷阱：依赖链 a=fmla(...); b=fmla(a,...) 必须等 4 周期 → 吞吐被压到 0.25×。
//
//  解法 = 更多累加器（ILP 深度）：让独立的 sdot 彼此不依赖，填满流水线。
//  gemm_int8.h:10021 那一串 16 条 sdot 互相独立，v16 的结果还在「飞行」中，
//  后面 15 条已经排队发射 → 吞吐拉满。
//
//  预取调度（gemm_arm.cpp:2456）：
//      prfm  pldl1keep, [%0, #128]     // 预取 A，提前 128 字节
//      ld1   {v2.4s},   [%0], #16      // 真正载入 A（此时数据已在 L1）
//      prfm  pldl1keep, [%1, #256]     // 预取 B
//      ld1   {v0.4s,v1.4s},[%1], #32
//  prfm 在 ld1 之前发出且地址提前 → 藏住访存延迟。这是手工调度才精确可控的。
//
//  循环尾调度技巧（gemm_int8.h:10037）：subs 与 bne 本可紧挨，作者却把一批
//  独立 sdot 插在 subs 之后、bne 之前 → 避免循环尾部流水线排空：
//      sdot  v31.4s, v1.16b, v7.4b[3]   // 第一批 sdot 末尾
//      subs  w4, w4, #1                 // K 计数器减 1（置标志，1 周期就绪）
//      sdot  v16.4s, v2.16b, v6.4b[0]   // ← 在 subs 之后继续发射 sdot
//      ...
//      bne   2b                         // 才消费标志、跳回
//
//  手写汇编两种形态（对照 CUDA 的 PTX/SASS）：
//      · 内联汇编（ncnn 选的）：asm volatile 嵌在 C++ 里，编译器管 ABI，你控热块；
//        代价是约束繁琐，且周围可能被编译器微调。
//      · 纯 .S 文件：整函数手写，ABI/栈/保存全自控（本目录 ncnn 未用）。

// ----------------------------------------------------------------------------
//  四、验证方法（落回你的工具链）
// ----------------------------------------------------------------------------
//
//  1) 先 intrinsics 后 asm：intrinsics 写通 → objdump -d 看编译器生成了什么
//     （有无 spill、顺序好不好）→ 把热块改成内联汇编微调。
//  2) objdump -d ./bin | grep -E 'fadd|fmla|sdot|vadd'   确认 v 前缀指令、无 spill。
//  3) perf stat -e instructions,cycles ./bin   → 调度好了 = 同工作量下 cycles 更少、IPC 更高。
//  4) perf annotate                            → 确认手调块真的以你写的顺序进了二进制。
//  5) 查微架构手册：ARM《Cortex-A76 Software Optimization Guide》逐条列 latency/throughput，
//     是调度的「圣经」。

// ============================================================================
//  可运行示范：最小手写汇编内核（aarch64）
//  仅用于教学：4 路 float 向量加，全部手写内联汇编，并与标量对拍。
//  （真正的 GEMM 微内核见上方注释引用的 ncnn 源码，过于复杂不在此编译。）
// ============================================================================
#if defined(__aarch64__)

static void vadd4_asm(const float* a, const float* b, float* c) {
    // 累加器/数据全用 V0–V2（AAPCS 中调用方保存，叶子函数自由使用）
    asm volatile(
        "ld1   {v0.4s}, [%0]\n"
        "ld1   {v1.4s}, [%1]\n"
        "fadd  v2.4s, v0.4s, v1.4s\n"
        "st1   {v2.4s}, [%2]\n"
        :
        : "r"(a), "r"(b), "r"(c)
        : "v0", "v1", "v2", "memory"
    );
}

static void vadd4_scalar(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

static int demo_asm() {
    const int N = 4;
    alignas(16) float a[4] = {1.f, 2.f, 3.f, 4.f};
    alignas(16) float b[4] = {10.f, 20.f, 30.f, 40.f};
    alignas(16) float c[4] = {0};
    alignas(16) float ref[4] = {0};

    vadd4_asm(a, b, c);
    vadd4_scalar(a, b, ref, N);

    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (std::fabs(c[i] - ref[i]) > 1e-6f) ok = false;

    std::printf("  [aarch64] vadd4_asm 手写汇编内核 : %s\n", ok ? "OK (与标量对拍一致)" : "MISMATCH");
    return ok ? 0 : 1;
}

#else
static int demo_asm() {
    std::printf("  [x86] 手写 NEON 汇编段被 __aarch64__ 宏隔离，仅 ARM 上运行。\n");
    std::printf("         完整讲解与 ncnn 真代码见本文件注释 + neon_l3_asm.md。\n");
    return 0;
}
#endif

int main() {
    std::printf("=== L3 手写 NEON 汇编 · 寄存器分配 · 指令调度 ===\n");
    std::printf("附：最小手写汇编内核示范（4 路 float 向量加）\n\n");
    return demo_asm();
}
