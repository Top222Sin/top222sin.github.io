// ============================================================================
//  NEON 学习手册 (自包含参考, 精简版 §0~§6)
//  性能剖析部分已拆分到独立文件: neon_l2_profiling.cpp (含可运行基准)。
//
//  本文件附带两段可直接编译运行的 demo:
//    1) 基础 intrinsics  (add / saxpy / dot)     —— demo_basics()
//    2) 查表 LUT          (vqtbl1q_u8)            —— demo_lut()
//  所有讲解均对应你工作区 ncnn/src/layer/arm/ 的真实实现，可对照开读。
//
//  编译:
//    ARM Linux : g++ -O3 -march=armv8-a+simd -std=c++17 neon_l1_lut.cpp -o neon_demo
//    x86 Win   : g++ -O3 -mssse3        -std=c++17 neon_l1_lut.cpp -o neon_demo.exe  (MinGW)
//                cl /O2 /std:c++17 /EHsc /arch:AVX2 neon_l1_lut.cpp                    (MSVC)
//  运行: ./neon_demo
// ============================================================================


// ============================================================================
//  §0. 先定位你的场景 (NEON vs CUDA 能力矩阵)
// ============================================================================
//  结论: NEON 和 CUDA 的"深度"取决于目标硬件与业务域，不是越深越好，而是够用就好、按需深挖。
//
//  | 目标平台                          | NEON | CUDA |
//  |-----------------------------------|------|------|
//  | NVIDIA GPU 科学计算/深度学习/渲染 | 基本用不到 | 核心硬技能 |
//  | ARM 服务器(Graviton)/移动端/边缘  | 核心手段   | 用不到     |
//  | 桌面/服务器 x86 HPC               | 对应物是 AVX/SSE | 有 GPU 才需要 |
//
//  常见组合: 一个 SIMD(NEON 或 AVX) + 一个 GPU 框架(CUDA 或 ROCm)。
//
//  NEON 三档掌握度:
//    L1 了解    —— 懂 SIMD 原理；会用 -O3 -march=armv8-a+simd 让编译器自动向量化；
//                 写对内存对齐、循环展开友好的代码。
//    L2 会用 intrinsics (多数 HPC 工程师目标线) —— 用 <arm_neon.h> 手写向量化函数
//                 (向量加减乘、点积、矩阵小块)；理解 AoS vs SoA、对齐；用 perf 找热点局部加速。
//    L3 深度优化 (库开发者/移动端推理引擎) —— 手写 NEON 汇编、指令调度、寄存器分配；
//                 针对 Cortex-A 微架构调优；结合 SVE 做 FFT/矩阵乘/编解码极致优化
//                 (NCNN、MNN 的 ARM 后端就是这层)。
//  => 除非做移动端/ARM 推理引擎或 ARM 服务器 HPC 库，否则 L2 足够，剩下交给 ACL/Eigen。
//
//  CUDA 三档掌握度:
//    L1 能跑    —— 理解 kernel/grid/block/thread/SIMT；会写简单 kernel；会调 cuBLAS/cuDNN。
//    L2 能写高效 kernel (HPC 工程师基准线) —— 内存层次 global/shared/register/constant；
//                 合并访问、避免 bank conflict、shared 双缓冲；warp/分支发散/occupancy；
//                 用 Nsight Compute/Systems profiling；共享内存归约、矩阵乘分块(tiling)。
//    L3 架构级极致优化 (AI 框架/算子库/超算) —— kernel fusion、异步拷贝(cp.async)、
//                 Tensor Core(WMMA/MMA PTX/CUTLASS)、cooperative groups、多 GPU/NVLink/NCCL。
//  => HPC 工程师 CUDA 至少 L2；做 AI 框架/算子库要到 L3。
//
//  你工作区里已躺着 NCNN、MNN、onnxruntime 三个仓库——它们本身就是 NEON(ARM 后端)
//  与 CUDA 优化的真实教科书:
//    - 读 ncnn/src/layer/arm/        看 NEON 怎么手写卷积/矩阵乘；
//    - 读 onnxruntime 的 CUDA EP     看 kernel 分块与内存优化。


// ============================================================================
//  §1. SIMD 原理 —— 向量化为什么比标量快
// ============================================================================
//  SIMD = Single Instruction, Multiple Data (单指令多数据)，Flynn 分类法的一种并行模型。
//  核心思想: 一条指令，对一组数据(一个向量)施加同一个操作。
//
//  标量(SISD):  r0+1, r1+1, r2+1, r3+1 是 4 条独立指令，CPU 分别取指/译码/执行 4 次。
//  SIMD:       把 4 个数塞进一个 128-bit 向量寄存器，发一条"向量 ADD"，硬件内 4 个算术
//              通道(lane)在同一周期内各自算一个，一次吐出 4 个结果。
//
//  NEON 寄存器 128-bit，按数据类型切成 lane:
//    float32 : 每 lane 32 bit  -> 1 个寄存器容纳 4 个
//    int16   : 每 lane 16 bit  -> 容纳 8 个
//    int8/u8 : 每 lane  8 bit  -> 容纳 16 个
//    float64 : 每 lane 64 bit  -> 容纳 2 个
//  => 处理 int8 时 NEON 一次并行 16 个数——这就是量化推理(int8)在 ARM 上比 float32
//     快近 4 倍的理论来源。
//
//  向量化"快在哪里"(4 个真实收益来源):
//    1) 指令开销被摊薄: 取指/译码/派发按"指令条数"算，4 条变 1 条 -> 固定开销降为 1/4。
//    2) 每周期算术量翻倍: 标量 ALU 一周期出 1 个结果；128-bit SIMD 一周期出 4(float32)/
//       16(int8) 个，吞吐 = lane 数。
//    3) 内存搬运更"粗": 一次 load 搬 128 bit(16 字节)连续数据进向量寄存器，相当于一次
//       搬 4 个 float32，减少访存指令条数与 cache line 浪费——带宽受限场景比算力更重要。
//    4) 循环次数变少: 步长从 1 变成 4/8/16，分支/计数/回边开销跟着降。
//
//  反直觉: 加速 ≠ lane 数 (现实达不到 4×/16×)，原因心里要有数:
//    a) 数据得"排好队": SIMD 要连续、对齐的内存。AoS/链表/稀疏结构要额外 gather/scatter。
//    b) 不是所有运算都能并行: 带分支逻辑要用 mask 模拟，分支发散时部分 lane 空转。
//    c) 内存带宽可能先到顶: 数据喂不过来，计算再快也白搭。合并访问/预取比"多算几个"更关键。
//    d) 启动/对齐/余数开销: N 不能被 lane 数整除时余数要标量补；循环太短收益被 amortize 不掉。
//
//  串联分级:
//    L1(让编译器干): 写 -O3 -march=...，编译器自动把"连续数组上的同构循环"向量化，
//                   你只需保证数据连续、循环简单。
//    L2(手写 intrinsics): 当编译器 vectorize 不出来(复杂分支、特定布局、特定指令如查表/fma)
//                   时，用 <arm_neon.h> 手动控制 lane，把数据摆成 SIMD 友好形状。
//    L3(手调汇编): 精确到"这条指令延迟 2 周期/吞吐 1/周期/占几个寄存器"，喂满 lane 不空转、
//                   不踩 bank conflict。


// ============================================================================
//  §2. 编译器自动向量化 —— 能做多少，何时失败 (决定 L1 够不够用)
// ============================================================================
//  现代编译器(GCC/Clang -O3, 或 -O2 -ftree-vectorize)的自动向量化器已相当强。对
//  "连续数组 + 同构循环体 + 迭代相互独立"的理想形态，几乎总能做对，并自动处理脏活:
//    - 余数(tail)自动补: N 不能被 lane 数整除时，生成"向量主循环 + 标量尾巴"。
//    - 简单归约: sum += a[i] 配合 -ffast-math 可向量化(多 lane 各自累加再合并)。
//    - 简单分支转无分支: if/else 被编译成 mask 选择，不破坏向量流。
//    - 循环展开 + 指令调度: 自动合并多轮，提高 ILP、隐藏延迟。
//  => 若热点都是"数组上跑同构循环"，L1(只靠编译器 + -O3 -march)完全够用，不需手写 NEON。
//
//  失败点(= L1 不够用的信号，每类都对应 L2/L3 要手动解决的事):
//    1) 循环依赖/递推: a[i] 依赖 a[i-1]，串行，向量化器放弃——改算法(如前缀和并行化)。
//    2) 指针别名 aliasing: 两个裸指针 x,y，编译器不敢假设不重叠，为安全只能串行 -> 加 restrict。
//    3) 非连续/间接访问: AoS(struct{x,y,z} 取 x 隔着 y,z)、指针数组、链表、稀疏下标 ->
//       内存不连续，一条 vector load 读不出整齐的 4 个数 -> 改 SoA，或手写 vld1/gather。
//    4) 函数调用看不到函数体: 循环里调未 inline 的外部函数，无法跨调用分析，通常卡住。
//    5) 复杂控制流: 虚函数、异常、goto、数据相关分支，分析不动。
//    6) 顺序敏感重排: 浮点累加默认不能随意重排(误差变)，不开 -ffast-math 时归约被禁。
//  一句话记忆: 编译器怕两件事——"看不看得透"(别名/调用/分支) 和 "排得整不整齐"(连续/独立/可重排)。
//             它放弃的地方，就是你该上 L2 intrinsics 的地方。
//
//  怎么确认编译器到底做没做(别靠猜):
//    - GCC  : -fopt-info-vec 看哪些循环向量化；-fopt-info-vec-missed 看哪些没做、为什么
//             (会直接告诉 "loop carried dependency" / "possible aliasing")。
//    - Clang: -Rpass=loop-vectorize(成功)、-Rpass-missed=loop-vectorize(失败原因)。
//    - 看汇编: objdump -d 找向量指令——NEON 下是 fadd v0.4s, v1.4s, v2.4s(带 v 和 .4s)；
//             标量是 fadd s0, s1, s2。
//    - 常用组合: -O3 -march=native -ffast-math。注意 -ffast-math 放开重排与 NaN/无穷假设，
//              数值语义会变，生产环境要评估能否接受。
//
//  高效工作流(别一上来就撸 NEON):
//    1) 先 -O3 -march=... 跑，用 -fopt-info-vec-missed 看报告；
//    2) 若瓶颈循环都"已向量化" -> L1 足够，收工；
//    3) 若反复出现别名/AoS/依赖/调用且 profiler 确认是热点 -> 正是上 L2 手写 intrinsics 的时机。


// ============================================================================
//  §3. 内存对齐 / AoS vs SoA —— 向量化的命门
// ============================================================================
//  接上 §1 反直觉第 1 点(排好队)与第 3 点(带宽先到顶)。很多人误以为向量化快纯靠 ALU 并行，
//  但真实负载里往往"内存喂不饱 ALU"。所以"数据怎么摆"比"怎么算"更决定快慢。
//
//  AoS = struct Point{ float x,y,z; } pts[N];  内存:
//      x0 y0 z0 | x1 y1 z1 | x2 y2 z2 | x3 y3 z3 ...
//    想"给所有点的 x 加 d"时，目标 x0 x1 x2 x3 之间隔着 y,z，每 12 字节才一个。
//    NEON 的 vld1q_f32 是一条连续 128-bit load，只能从某地址一口气抓 16 连续字节。
//    这 4 个 x 不连续 -> 一条抓不到，要么 4 次零散窄 load 再用 vzip/vtrn/vst1_lane 拼回向量，
//    要么放弃向量化。拼装 + 可能跨多个 cache line 的代价，常把并行收益吃光甚至不如标量。
//
//  SoA = float xs[N]; float ys[N]; float zs[N];  内存:
//      x0 x1 x2 x3 | y0 y1 y2 y3 | z0 z1 z2 z3 ...
//    4 个 x 首尾相连正好 16 字节: 一条 vld1q_f32(xs+i) 直接拿到；vaddq 加 d；vst1q 存回。
//    全程无 shuffle、无零散 load，向量化并行收益 100% 落袋。
//
//  内存对齐为什么也是命门:
//    - NEON 的 128-bit load/store 对地址有自然对齐期望(16 字节边界最干净)。
//    - 对齐好处: ①一次内存事务取满 128 位，不跨 cache line，不浪费带宽；
//               ②未对齐 load 在不少架构上更慢(ARM 一般能容忍未对齐但性能下降，旧架构甚至 fault)。
//    - SoA 数组只需 alignas(16)(或 16 字节对齐的 malloc)，首地址 + 每步 +16 永远对齐 -> 全程干净 load。
//    - AoS: 单个 Point 是 12 字节(3×float32)，x 地址是 12 的倍数而非 16 的倍数 -> 对齐别扭，
//      进一步劝退向量化。
//
//  工程怎么选(别走极端):
//    - 热点密集数值计算(物理仿真/矩阵/图像/推理算子): 优先 SoA，或做 AoS→SoA 转换后再算
//      (正是 L2 手写 intrinsics 天天干的事)。
//    - 非热点/面向对象逻辑: AoS 更自然，别为优化牺牲可读性，先 profile 再说。
//  => L1(编译器)对 SoA 连续数组基本全自动；一旦数据 AoS/链表，L1 立刻失效，必须 L2 手动改布局 + 写 NEON。
//
//  串起全局:
//    - 第 1 点(排好队) = SoA + 对齐 -> 一条向量 load 解决；
//    - 第 3 点(内存带宽) = 对齐连续的 128-bit load 不浪费带宽、不跨 cache line。
//  "连续对齐访存"这条铁律，在 CUDA 里叫"合并访问(coalesced access)"，是 L2 CUDA 的第一条军规。


// ============================================================================
//  §4. 基础 intrinsics 实战 —— vld1q / vst1q / vaddq / 融合乘加 / 水平归约
// ============================================================================
//  命名规律 v<操作>q_<类型> 记一次就通: q=128-bit(quadword), f32=float32, 4=4 个 lane。
//
//  (1) 向量加法 c = a + b  (§1 那张"标量 vs 向量"图的直接落地):
//        float32x4_t va = vld1q_f32(a + i);   // 连续地址一次载入 4 个 float 进 128 位寄存器
//        float32x4_t vb = vld1q_f32(b + i);
//        float32x4_t vc = vaddq_f32(va, vb);  // 4 个 lane 同时加: lane_k = va_k + vb_k
//        vst1q_f32(c + i, vc);                // 一次存回 4 个 float
//
//  (2) SAXPY y = alpha*x + y  (融合乘加):
//        float32x4_t va = vdupq_n_f32(alpha); // 把标量广播到 4 个 lane (n = "from scalar")
//        vy = vmlaq_f32(vy, va, vx);          // vy = vy + va*vx，一条指令完成乘+加
//      vdupq_n_f32 对应 §1"同一操作作用于多个数据"——把同一 alpha 复制到每个 lane。
//
//  (3) 点积 + 水平归约 (NEON 最易踩坑):
//        向量只能做"垂直"运算(lane 对 lane)，但点积最后要把 4 个 lane 加成一个标量，
//        这叫"水平归约":
//        acc = vmlaq_f32(acc, va, vb);        // 4 个 lane 各自累乘加
//        float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc)); // 拆 2+2 再相加
//        float32x2_t s1 = vpadd_f32(s2, s2);  // 两两加: 2 个 -> 1 个
//        float s = vget_lane_f32(s1, 0);      // 取出标量
//        vpadd 是 NEON 特有的"两两打包相加"，做水平归约的标配。
//        (SSE 那边更省事: 直接 _mm_storeu_ps 到数组再 tmp[0..3] 相加。)
//
//  与前面概念的对应:
//    - SoA 命门: 本 demo 数据就是 SoA(float* a,b,c 各自连续)，vld1q 一条干净取到 4 个，
//      正是 §3"右边紫色聚成块"的代码版。若换成 AoS(struct{float x,y,z})，vld1q 抓不到整齐的 4 个数。
//    - 对齐: vld1q_f32 在 AArch64 上对未对齐地址也安全(同 x86 _mm_loadu_ps 用 u 表示 unaligned)；
//      但为性能应让数组 16 字节对齐(SoA 数组加 alignas(16))。x86 的 _mm_load_ps(不带 u)才强制对齐。
//    - 余数处理: for(; i+4<=n; i+=4) 主循环吃满 4 的倍数，后面 for(; i<n; ++i) 标量补尾——
//      这正是编译器自动向量化也会做的 tail。N 不是 4 的倍数时不能漏。


// ----------------------------------------------------------------------------
//  §4 配套可运行代码: add / saxpy / dot —— 三种后端 (NEON 真跑 / SSE2 x86 兜底 / 标量对拍)
// ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON) || \
    defined(_M_ARM) || defined(_M_ARM64)
  #include <arm_neon.h>
  #define BACKEND "NEON"
  #define HAVE_NEON 1
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || _M_IX86_FP >= 2))
    #include <emmintrin.h>   // SSE2
    #define BACKEND "SSE2"
    #define HAVE_SSE2 1
  #endif
#endif
#ifndef BACKEND
  #define BACKEND "scalar"
#endif

// ---- 向量加法 ----
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

// ---- SAXPY: y = alpha*x + y ----
#if defined(HAVE_NEON)
static void saxpy_simd(float alpha, const float* x, float* y, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vst1q_f32(y + i, vmlaq_f32(vld1q_f32(y + i), va, vx));
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
#elif defined(HAVE_SSE2)
static void saxpy_simd(float alpha, const float* x, float* y, int n) {
    int i = 0;
    __m128 va = _mm_set1_ps(alpha);
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vy = _mm_loadu_ps(y + i);
        _mm_storeu_ps(y + i, _mm_add_ps(_mm_mul_ps(va, vx), vy));
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
#else
static void saxpy_simd(float alpha, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
#endif

// ---- 点积 + 水平归约 ----
#if defined(HAVE_NEON)
static float dot_simd(const float* a, const float* b, int n) {
    int i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float32x2_t s1 = vpadd_f32(s2, s2);
    float s = vget_lane_f32(s1, 0);
    for (; i < n; ++i) s += a[i] * b[i];   // 标量补尾
    return s;
}
#elif defined(HAVE_SSE2)
static float dot_simd(const float* a, const float* b, int n) {
    int i = 0;
    __m128 acc = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float tmp[4]; _mm_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
#else
static float dot_simd(const float* a, const float* b, int n) {
    float s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s;
}
#endif

static int demo_basics() {
    const int N = 1 << 12;
    std::vector<float> a(N), b(N), c(N), y(N);
    std::mt19937 rng(2024);
    for (int i = 0; i < N; ++i) {
        a[i] = (float)(rng() & 1023) / 64.0f - 8.0f;
        b[i] = (float)(rng() & 1023) / 64.0f - 8.0f;
        y[i] = (float)(rng() & 1023) / 64.0f - 8.0f;
    }
    printf("\n[§4 基础 intrinsics] backend = %s\n", BACKEND);

    // add
    std::vector<float> cref(N);
    add_scalar(a.data(), b.data(), cref.data(), N);
    add_simd(a.data(), b.data(), c.data(), N);
    bool ok_add = true;
    for (int i = 0; i < N; ++i) if (std::fabs(c[i] - cref[i]) > 1e-5f) { ok_add = false; break; }

    // saxpy
    std::vector<float> yref = y;
    float alpha = 1.5f;
    for (int i = 0; i < N; ++i) yref[i] = alpha * a[i] + yref[i];
    saxpy_simd(alpha, a.data(), y.data(), N);
    bool ok_saxpy = true;
    for (int i = 0; i < N; ++i) if (std::fabs(y[i] - yref[i]) > 1e-4f) { ok_saxpy = false; break; }

    // dot
    float dref = 0; for (int i = 0; i < N; ++i) dref += a[i] * b[i];
    float d = dot_simd(a.data(), b.data(), N);
    bool ok_dot = std::fabs(d - dref) < 1e-3f * std::fabs(dref) + 1e-3f;

    printf("  add   : %s\n", ok_add ? "OK" : "FAIL");
    printf("  saxpy : %s\n", ok_saxpy ? "OK" : "FAIL");
    printf("  dot   : %s  (simd=%.4f  scalar=%.4f)\n", ok_dot ? "OK" : "FAIL", d, dref);
    return (ok_add && ok_saxpy && ok_dot) ? 0 : 1;
}


// ============================================================================
//  §5. 查表 LUT —— vqtbl1q_u8 实战 (可运行 demo)
// ============================================================================
//  先说实情: grep 了整个 ncnn/src, 没有任何文件用 vqtbl/vqtbl1q。ncnn ARM 后端偏好用
//  ext/vtrn/vzip 这类"固定模式重排" + fmla 计算，而不是运行时查表。但 LUT 本身仍是 NEON
//  的杀手锏指令，值得掌握，且本段代码可直接编译运行对拍。
//
//  NEON 查表指令: vqtbl1q_u8(table, idx) —— 用 idx 的 16 个字节作索引，从一张 16 字节表里
//  并行查出 16 个结果，一条指令完成。表是 256 项时用 vqtbl2q_u8(两张 16 字节表)，更大用 vqtbl4q_u8。
//  典型用途: int8 反量化/重映射表、通道重排、解交织、阈值化。
//
//  顺带点破: §6 里 transpose8x8_ps 用的 vtrn/vzip/vext 和 vqtbl1q_u8 是同一家族——都是"按某种
//  模式把字节重新排列"。区别仅是前者模式固定(编译期)，后者模式来自一张运行时表。理解了这点，
//  NEON 的"数据重排"手法就打通了。
#include <cstring>

#if defined(HAVE_NEON)
  // NEON 已在 §4 头部 include；这里直接复用
  #define LUT_BACKEND "NEON vqtbl1q_u8"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(__SSSE3__)
    #include <tmmintrin.h>   // _mm_shuffle_epi8
    #define LUT_BACKEND "SSSE3 _mm_shuffle_epi8"
    #define HAVE_SSSE3 1
  #endif
#endif
#ifndef LUT_BACKEND
  #define LUT_BACKEND "scalar (no SIMD)"
#endif

// 16 字节查表: out[i] = table[in[i] & 15]
static void lut_scalar(const uint8_t* in, const uint8_t table[16], uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = table[in[i] & 15];
}

#if defined(HAVE_NEON)
// vqtbl1q_u8(tbl, idx): 对 idx 的 16 个字节并行查表, 结果字节 = tbl[idx字节]
static void lut_neon(const uint8_t* in, const uint8_t table[16], uint8_t* out, size_t n) {
    uint8x16_t tbl = vld1q_u8(table);            // 载入 16 字节的表
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t idx  = vld1q_u8(in + i);       // 16 个待查索引
        uint8x16_t res  = vqtbl1q_u8(tbl, idx);   // 一条指令完成 16 路查表
        vst1q_u8(out + i, res);
    }
    for (; i < n; ++i) out[i] = table[in[i] & 15];
}
// 注: 若表是 256 项, 用 vqtbl2q_u8 (两张 16 字节表) / vqtbl4q_u8

#elif defined(HAVE_SSSE3)
static void lut_ssse3(const uint8_t* in, const uint8_t table[16], uint8_t* out, size_t n) {
    __m128i tbl = _mm_loadu_si128((const __m128i*)table);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i idx = _mm_loadu_si128((const __m128i*)(in + i));
        __m128i res = _mm_shuffle_epi8(tbl, idx); // 与 vqtbl1q_u8 等价
        _mm_storeu_si128((__m128i*)(out + i), res);
    }
    for (; i < n; ++i) out[i] = table[in[i] & 15];
}
#endif

static int demo_lut() {
    // 构造一张示例表: v -> (v*3) % 16
    uint8_t table[16];
    for (int v = 0; v < 16; ++v) table[v] = (uint8_t)((v * 3) % 16);

    const size_t N = 1 << 14;
    std::vector<uint8_t> in(N), out(N), ref(N);
    std::mt19937 rng(7);
    for (size_t i = 0; i < N; ++i) in[i] = (uint8_t)(rng() & 15);  // 索引限制在 0..15

    printf("\n[§5 查表 LUT] backend = %s\n", LUT_BACKEND);
    lut_scalar(in.data(), table, ref.data(), N);

#if defined(HAVE_NEON)
    lut_neon(in.data(), table, out.data(), N);
#elif defined(HAVE_SSSE3)
    lut_ssse3(in.data(), table, out.data(), N);
#else
    lut_scalar(in.data(), table, out.data(), N);
#endif

    bool ok = true;
    for (size_t i = 0; i < N; ++i) if (out[i] != ref[i]) { ok = false; break; }
    printf("  correctness: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}


// ============================================================================
//  §6. 三个算子实战 —— 矩阵乘分块(GEMM tiling) / 卷积 / 查表  (对照 ncnn 真代码)
// ============================================================================
//  以下描述均对应你工作区 ncnn/src/layer/arm/ 的真实实现。

//  (一) 矩阵乘分块 (GEMM tiling) —— ncnn 怎么做的
//  * 为什么分块: 朴素 GEMM C[M][N]+=A[M][K]*B(inverse K)... 每个元素扫 K 次，但内存行优先，
//    B 的列访问跳着走，cache 命中极低。分块把大矩阵切成 tile，让一个 tile 的 A/B/C 都塞进
//    寄存器/L1/L2，K 维被反复复用——这就是 tiling 的本质。
//  * ncnn 路线: 卷积先 im2col 展开成矩阵，再走 GEMM。convolution_im2col_gemm.h 先把 A 打包并
//    转置成 tile:
//        transpose8x8_ps(_r0l,_r0h,...,_r7l,_r7h);   // 转置 8x8
//        vst1q_f32(pp, _r0l); vst1q_f32(pp + 4, _r0h); ...
//    这正是 §3"SoA + 对齐"的工业版: 把数据重排成"K 维连续"，热循环里纯 vld1q 连续取数。
//    转置本身在 arm_usability.h:596 的 transpose8x8_ps，用 vtrn/vzip/vext 实现——
//    它和"查表"是同一类"按模式重排字节"的手法。
//  * 微内核 (核心, gemm_arm.cpp:2455) —— 寄存器分块的真容 (汇编写法):
//        ld1    {v2.4s}, [%0], #16          // A tile 的一列(4 个 lane = 4 行)
//        ld1    {v0.4s, v1.4s}, [%1], #32   // B tile 的 4 列(拆成 v0/v1 各 4 lane)
//        fmla   %2.4s, v2.4s, v0.s[0]       // _sum0 += A列 * B[0]
//        fmla   %3.4s, v2.4s, v0.s[1]       // _sum1 += A列 * B[1]
//        ... (共 8 条 fmla) ...
//        fmla   %9.4s, v2.4s, v1.s[3]       // _sum7 += A列 * B[7]
//    - _sum0.._sum7 是 8 个 float32x4_t 累加器 -> 一次算 8 行 × 4 列 = 8×4 输出块。
//    - v2 是 A 的一列(4 lane = 4 行)，v0.s[k] 把 B 的某个标量广播到 4 个 lane，8 条 fmla
//      完成一次 rank-1 更新。
//    - prfm pldl1keep 是预取，藏内存延迟——这是 L3 才有的手工调度。
//    - vfmaq_laneq_f32 (gemm_arm.cpp:2531) 是 intrinsics 写法，等价于上面那段汇编。
//  => 这就是分级里的 L3: 把输出块锁在寄存器、K 维全复用、用 fmla 融合乘加、用 prfm 掩盖延迟。
//     编译器自动向量化绝对写不出这种。

//  (二) 卷积 —— ncnn 里的两条路线
//  * 路线 A: im2col + GEMM (上面那套)。通用、对任意 kernel 尺寸都工作，但 im2col 会膨胀内存。
//  * 路线 B: direct NEON (conv3x3s1_neon, convolution_3x3.h:75) —— 小核(3×3)更快，因为它不展开
//    内存，直接在寄存器里滑窗口:
//        ld1    {v8.4s, v9.4s}, [%5]        // 载入第 0 行 r0 的 8 个 float (v8=前4, v9=后4)
//        ext    v10.16b, v8.16b, v9.16b, #4  // 窗口向右滑 4 字节(=1 个 float)
//        ext    v11.16b, v14.16b, v15.16b, #8
//        fmla   v6.4s, v8.4s, %18.s[0]      // 窗口 tap0 × 权重k0 -> 累加到 out0
//        fmla   v7.4s, v8.4s, %21.s[0]      // 同时算 out1 (双行并行，提升 ILP)
//    - ext v10.16b, v8.16b, v9.16b, #4 是精髓: 把 v8|v9 这 8 个 lane 组成的窗口按字节右移 4 字节，
//      得到 3×3 窗口的"下一列"tap。三次 ext 取出 3 个水平 tap，配合 r0/r1/r2 三行，凑齐 9 个 tap，
//      每个 fmla 乘对应权重累加。
//    - 同时算两行输出(out0/out1)，喂满 NEON 的 Multiple Issue，避免流水线饿着。
//  => 这条路线复用了前面所有知识点: 连续 vld1q 载入(SoA 友好)、fmla 融合乘加、ext 数据重排
//     (AoS 图像布局 -> 向量友好的窗口)、prfm 预取。

//  (三) 查表 (LUT) —— 实情 + 可运行示例
//  * 实情: 如前所述，ncnn ARM 端几乎不用 vqtbl(改用 ext/vtrn/vzip 做数据重排)，LUT 不是主菜。
//    但这不妨碍它本身是 NEON 杀手锏，见 §5 的演示。
//  * NEON 查表指令 vqtbl1q_u8(table, idx): 用 idx 的 16 字节作索引，从 16 字节表并行查 16 结果
//    (§5 已给可运行代码)。典型用途: int8 反量化/重映射表、通道重排、解交织、阈值化。

//  (四) 三块串起来
//    | 算子           | 核心 NEON 手法                       | 对应前面哪课              |
//    |----------------|--------------------------------------|---------------------------|
//    | GEMM 分块      | 寄存器分块 + fmla rank-1 更新 + prfm | 寄存器复用=极致版吞吐/访存|
//    | 卷积 direct    | ext 滑动窗口 + fmla 双行并行          | SoA/连续载入 + 数据重排    |
//    | LUT 查表       | vqtbl1q_u8 向量查表                  | 数据重排家族(与 vtrn/vext 同类)|
//  反复出现的"肌肉记忆"指令: vld1q(连续载入，靠 SoA/对齐)、fmla/vmlaq(融合乘加)、
//  ext/vtrn/vzip(数据重排)、prfm(预取)、vfmaq_laneq(跨 lane 广播乘加)。

//  后续可深挖:
//    1) "CUDA 合并访问为什么是铁律" —— 把"连续对齐访存"放大到 GPU 的 warp(32 线程协同)，
//       不合并的访存直接掉一个数量级，与 NEON 的 vld1q 连续载入是同一直觉。
//    2) "ncnn int8 量化 GEMM" —— 读 gemm_arm_asimddp.cpp，看 ARMv8.2 的 dot-product 指令
//       (sdot/udot) 怎么把 int8 矩阵乘做成 L3 终极形态，正好接上"查表/量化"那块。
//    (性能剖析部分已拆分到独立文件 neon_l2_profiling.cpp。)


int main() {
    printf("==================================================\n");
    printf("  NEON 学习手册 / 实战 demo (精简版 §0~§6)\n");
    printf("==================================================\n");
    int r1 = demo_basics();   // §4 基础 intrinsics (add/saxpy/dot)
    int r2 = demo_lut();      // §5 查表 LUT
    printf("\n结果: basics=%s  lut=%s\n", r1 ? "FAIL" : "OK", r2 ? "FAIL" : "OK");
    return (r1 | r2) ? 1 : 0;
}
