// ============================================================================
//  L3 深度优化（库开发者 / 推理引擎后端）· 第四部分：SVE / SVE2 迁移
//  「让一份微内核自适应 A76/X1/X2，免逐代手写汇编」
//
//  ⚠️ 前置事实校正（这是本部分最重要的认知）：
//     A76 / A78 / X1 这一代是 Armv8.2~8.4，**没有 SVE 单元**，只能用 NEON。
//     —— 但「SVE 是 Armv9 专属」是错的：基础 SVE 从 **Armv8.2-A 起即作为可选扩展**
//        存在；Neoverse-V1 = **Armv8.4-A + SVE（第一代，非 SVE2）**，是首款带 SVE 的核。
//     —— SVE2 才是 **Armv9** 的标配超集（X2/X3/X4、A710/A715、Neoverse N2/V2）。
//     → 诚实的迁移结论（三路，不是两路）：
//         * NEON(ASIMD) 守住 A76/A78/X1 这一代（固定 128-bit，需手写每代汇编）；
//         * 基础 SVE 一份内核，守住 **Neoverse-V1**（Armv8.4-A+SVE，256-bit，VL=8）；
//         * SVE2 一份内核，自适应 X2→X4 及 Neoverse V2/N2（128~512-bit 不等）。
//     → 真正的生产力收益不是"更宽"，而是：
//         (1) VLA 编程模型：向量长度编译期未知，运行时用 svcntw() 发现；
//         (2) 谓词执行：svwhilelt_b32 直接掩掉尾部，免写标量 remainder 循环；
//         (3) 一份 .c 二进制，跑在任意 VL 的核上都不用重新汇编。
//     → 本部分的 FP32 VLA 微内核只用**基础 SVE** 指令（svld1/svmla/svwhilelt/svdup），
//        所以 V1（基础 SVE）和 X2/V2/X4（SVE2）**都能跑同一份**；SVE2 作为超集，
//        额外提供的是 **int8 `svdot` 矩阵乘**等（见第五部分 int8 GEMM/量化路径）。
//
//  本文件结构（对齐笔记 §九(终章)/§十）：
//   §一  核心事实：哪些核有 SVE / SVE2、向量长度怎么定
//   §二  运行时向量长度发现（svcntw 仿真 + 真 HWCAP 三路分发）
//   §三  VLA GEMM 单代码路径：一份内核自适应 VL=4/8/16（x86 可跑的仿真版）
//   §四  谓词消除尾部循环：NEON 需要 remainder，SVE 一行 svwhilelt 搞定
//   §五  真 SVE/SVE2 微内核（#ifdef __ARM_FEATURE_SVE 守卫，ARM 真编译；SVE2 兼容）
//   §六  迁移清单：从 NEON 8x4 到 SVE VLA 内核的逐条改动
//
//  编译（注意：基础 SVE 需 Armv8.x 工具链 +sve，SVE2 需 Armv9 或 +sve2 扩展）：
//    x86 本机（跑仿真 + 标量对拍）：
//        g++ -O3 -std=c++17 neon_l3_sve2.cpp -o l3s && ./l3s
//    ARM 真跑基础 SVE（Neoverse-V1，Armv8.4-A+SVE）：
//        aarch64-linux-gnu-g++ -O3 -march=armv8.4-a+sve -std=c++17 neon_l3_sve2.cpp -o l3s && ./l3s
//    ARM 真跑 SVE2（X2/X3/X4、A710、Neoverse V2/N2）：
//        aarch64-linux-gnu-g++ -O3 -march=armv9-a+sve2 -std=c++17 neon_l3_sve2.cpp -o l3s && ./l3s
//    ARM 三路分发（NEON 守 A76；SVE 守 V1；SVE2 守 X2+，各自独立 TU 或 target 属性）：
//        aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd   -std=c++17 neon_l3_sve2_neon.cpp -c
//        aarch64-linux-gnu-g++ -O3 -march=armv8.4-a+sve  -std=c++17 neon_l3_sve2_sve.cpp  -c
//        aarch64-linux-gnu-g++ -O3 -march=armv9-a+sve2  -std=c++17 neon_l3_sve2_sve2.cpp -c
//        aarch64-linux-gnu-g++ -O3 *.o main.o -o l3s
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

// ---- 真 SVE/SVE2 内核需要的头与检测（仅 ARM 工具链可见）-----------------------
#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2)
#  include <arm_sve.h>
#endif

// ============================================================================
//  §一  核心事实：SVE / SVE2 在哪些核上、向量长度怎么定
// ============================================================================
//  核              架构          SVE?   SVE2?   实现向量长度(典型)   对应向量路
//  --------------- ------------ ------ ------- ------------------  ------------
//  Cortex-A76      Armv8.2      无     无       — (固定 128b)       NEON 8x4
//  Cortex-A78      Armv8.2      无     无       —                   NEON 8x4
//  Cortex-X1       Armv8.2      无     无       —                   NEON 8x4
//  Neoverse-V1     Armv8.4-A    ✅ SVE  无      256-bit             SVE(VL=8, 第一代)
//  Cortex-X2       Armv9.0      无     ✅       128-bit             SVE2(VL=4)
//  Cortex-A710     Armv9.0      无     ✅       128-bit             SVE2(VL=4)
//  Neoverse-N2     Armv9.0      无     ✅       128-bit             SVE2(VL=4)
//  Cortex-X3/X4    Armv9.x      无     ✅       128/256-bit         SVE2
//  Neoverse-V2     Armv9.2      无     ✅       256-bit             SVE2(VL=8)
//
//  关键：
//   * 基础 SVE 从 Armv8.2 起就是可选扩展（V1 = Armv8.4-A+SVE），并非 Armv9 专属；
//     SVE2 才是 Armv9 标配的超集。
//   * 本部分 FP32 VLA 内核用的是基础 SVE 指令（svld1/svmla/svwhilelt/svdup），
//     **V1（基础 SVE）与 X2/V2/X4（SVE2）都能跑同一份**；SVE2 额外给 int8 svdot。
//   * "向量长度"由硅决定、编译期未知。一份内核用 svcntw() 在运行时得知
//     "本核一向量能装几个 float"，据此动态决定累加器列数。
//   * A76/A78/X1 不存在"自适应"，它们根本没有 SVE；自适应发生在 V1→X4 之间。

// 仿真用的"核画像"：在 x86 上没有真 SVE，我们用它演示 VLA 编程模型
// （注意：V1 是"基础 SVE"路，不是 SVE2 路）
enum class CoreSim { NEON_A76, SVE_V1_256, SVE2_X2_128, SVE2_X4_512 };

// 返回"每个向量能装几个 32-bit 元素"（即 svcntw() 的语义）
static int vl_of(CoreSim c) {
    switch (c) {
        case CoreSim::NEON_A76:    return 4;   // 128-bit NEON：固定 4 个 float
        case CoreSim::SVE_V1_256:  return 8;   // Neoverse-V1 基础 SVE：256-bit
        case CoreSim::SVE2_X2_128: return 4;   // X2 的 SVE2 实际也是 128-bit
        case CoreSim::SVE2_X4_512: return 16;  // 假设某 512-bit 实现（如 A64FX 思路）
    }
    return 4;
}

// ============================================================================
//  §二  运行时向量长度发现 + HWCAP 三路分发骨架
// ============================================================================
//  真机检测：读 auxv。
//   * AT_HWCAP 的 HWCAP_SVE 位：基础 SVE 或 SVE2 置位（V1 与 X2+ 都置位）；
//   * AT_HWCAP2 的 HWCAP2_SVE2 位：仅 SVE2 置位（V1 不置位）。
//  → 三路判断：HWCAP2_SVE2 置位 → SVE2；仅 HWCAP_SVE 置位 → 基础 SVE(V1)；
//              两者皆无 → NEON(A76/A78/X1)。
//  （x86 沙箱没有这些符号，所以整段用宏守卫，保证 x86 也能编译运行。）
#if defined(__linux__) && (defined(__aarch64__) || defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SVE2))
#  include <sys/auxv.h>
#  ifndef AT_HWCAP
#    include <asm/hwcap.h>
#  endif
static bool host_has_sve() {
    unsigned long hw = getauxval(AT_HWCAP);
    return (hw & HWCAP_SVE) != 0;          // 基础 SVE 或 SVE2 都置位
}
static bool host_has_sve2() {
#  ifdef HWCAP2_SVE2
    unsigned long hw2 = getauxval(AT_HWCAP2);
    return (hw2 & HWCAP2_SVE2) != 0;       // 仅 SVE2 置位（V1 不会置位）
#  else
    return false;
#  endif
}
#else
static bool host_has_sve()  { return false; } // x86 沙箱：无 SVE
static bool host_has_sve2() { return false; } // x86 沙箱：无 SVE2
#endif

// 真机 svcntw() 直接返回本核 VL 的 32-bit 元素数；下面给一个可编译的引用写法：
//   int vl = svcntw();   // 需要 <arm_sve.h> 且开启对应 SVE/SVE2 扩展

// ============================================================================
//  §三  VLA GEMM 单代码路径（x86 可跑的仿真版）
// ----------------------------------------------------------------------------
//  思路：把"向量"抽象成固定容量的小缓冲（cap = 最大 VL）。写一份内核，
//       外循环按"当前核的 VL"切 N 维；尾部用"谓词掩码"emul 掉——这正是
//       SVE 在真机上用 svwhilelt_b32 做的事。一份代码，换 CoreSim 即可
//       自适应 4/8/16，无需任何 per-core 汇编。
// ============================================================================
static void gemm_vla_sim(const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         int M, int N, int K,
                         CoreSim core, int MR) {
    const int VL = vl_of(core);
    // 每个 MR×VL 小块用 MR*VL 个标量累加器"伪向量"表达
    std::vector<float> acc(MR * VL, 0.f);
    for (int i = 0; i < M; i += MR) {
        int mr = std::min(MR, M - i);
        for (int j = 0; j < N; j += VL) {
            int nr = std::min(VL, N - j);            // 尾部元素数（谓词长度）
            // 清零累加器（仅本小块）
            for (int r = 0; r < mr; ++r)
                for (int c = 0; c < nr; ++c)
                    acc[r * VL + c] = 0.f;
            // K 维归约
            for (int k = 0; k < K; ++k) {
                for (int r = 0; r < mr; ++r) {
                    float a = A[(i + r) * K + k];    // 一次"广播"（svdup_n）
                    for (int c = 0; c < nr; ++c) {   // c<nr 即"谓词"：尾部不越界
                        acc[r * VL + c] += a * B[k * N + (j + c)];
                    }
                }
            }
            // 写回（谓词掩码写：只写 nr 个）
            for (int r = 0; r < mr; ++r)
                for (int c = 0; c < nr; ++c)
                    C[(i + r) * N + (j + c)] = acc[r * VL + c];
        }
    }
}

// NEON 风格固定 128-bit(4 lane)内核：必须手写 remainder 循环处理 N%4
// —— 这就是"逐代手写汇编"的痛点：每个不对称维度都要补一段标量。
static void gemm_neon_fixed4(const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int M, int N, int K, int MR) {
    const int VL = 4; // 固定，写死
    std::vector<float> acc(MR * VL, 0.f);
    for (int i = 0; i < M; i += MR) {
        int mr = std::min(MR, M - i);
        // 主循环：严格按 4 走
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            for (int r = 0; r < mr; ++r) for (int c = 0; c < 4; ++c) acc[r*4+c] = 0.f;
            for (int k = 0; k < K; ++k) {
                for (int r = 0; r < mr; ++r) {
                    float a = A[(i+r)*K+k];
                    for (int c = 0; c < 4; ++c) acc[r*4+c] += a * B[k*N+(j+c)];
                }
            }
            for (int r = 0; r < mr; ++r) for (int c = 0; c < 4; ++c)
                C[(i+r)*N+(j+c)] = acc[r*4+c];
        }
        // 尾部 remainder：标量补 N%4（SVE 用 svwhilelt 一行取代这段）
        for (; j < N; ++j) {
            for (int r = 0; r < mr; ++r) {
                float s = 0.f;
                for (int k = 0; k < K; ++k) s += A[(i+r)*K+k] * B[k*N+j];
                C[(i+r)*N+j] = s;
            }
        }
    }
}

// ============================================================================
//  §五  真 SVE / SVE2 微内核（守卫：基础 SVE 即可，SVE2 自然兼容）
// ----------------------------------------------------------------------------
//  要点：svcntw() 运行时决定 NR（列数=向量元素数）；svwhilelt_b32 生成尾部
//        谓词，使同一份内核处理任意 N 而不写 remainder。MR 仍可取 8（行数
//        与 VL 正交，便于把工作集压进 L1，呼应第三部分）。
//  注意：这个 FP32 内核只用基础 SVE 指令（svld1/svmla/svwhilelt/svdup），
//        **V1（基础 SVE）和 X2/V2/X4（SVE2）都能编译运行同一份**；
//        SVE2 作为超集额外提供 int8 `svdot` 等（主要用于 int8 GEMM/量化）。
// ============================================================================
#if defined(__ARM_FEATURE_SVE)   // 基础 SVE 即可（SVE2 满足此宏，故也兼容）
// 8×VL fp32 GEMM 微内核，VL 由运行时 svcntw() 决定
static void gemm_sve_8xvl(const float* A, const float* B, float* C,
                           int M, int N, int K, int ldc) {
    const int VL = svcntw();           // 运行时向量长度（V1→8, X2→4, ...）
    for (int i = 0; i < M; i += 8) {
        int mr = (M - i < 8) ? (M - i) : 8;
        for (int j = 0; j < N; j += VL) {
            int nr = (N - j < VL) ? (N - j) : VL;
            svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)nr); // 尾部谓词
            // 8 个向量累加器（每行一个）
            svfloat32_t c0 = svdup_n_f32(0.f), c1 = svdup_n_f32(0.f);
            svfloat32_t c2 = svdup_n_f32(0.f), c3 = svdup_n_f32(0.f);
            svfloat32_t c4 = svdup_n_f32(0.f), c5 = svdup_n_f32(0.f);
            svfloat32_t c6 = svdup_n_f32(0.f), c7 = svdup_n_f32(0.f);
            for (int k = 0; k < K; ++k) {
                svfloat32_t b = svld1_f32(pg, &B[k * N + j]);     // 谓词载入 B 面板
                c0 = svmla_f32_x(pg, c0, svdup_n_f32(A[(i+0)*K+k]), b); // 广播 A 行 × B
                c1 = svmla_f32_x(pg, c1, svdup_n_f32(A[(i+1)*K+k]), b);
                c2 = svmla_f32_x(pg, c2, svdup_n_f32(A[(i+2)*K+k]), b);
                c3 = svmla_f32_x(pg, c3, svdup_n_f32(A[(i+3)*K+k]), b);
                c4 = svmla_f32_x(pg, c4, svdup_n_f32(A[(i+4)*K+k]), b);
                c5 = svmla_f32_x(pg, c5, svdup_n_f32(A[(i+5)*K+k]), b);
                c6 = svmla_f32_x(pg, c6, svdup_n_f32(A[(i+6)*K+k]), b);
                c7 = svmla_f32_x(pg, c7, svdup_n_f32(A[(i+7)*K+k]), b);
            }
            if (mr >= 1) svst1_f32(pg, &C[(i+0)*ldc+j], c0);
            if (mr >= 2) svst1_f32(pg, &C[(i+1)*ldc+j], c1);
            if (mr >= 3) svst1_f32(pg, &C[(i+2)*ldc+j], c2);
            if (mr >= 4) svst1_f32(pg, &C[(i+3)*ldc+j], c3);
            if (mr >= 5) svst1_f32(pg, &C[(i+4)*ldc+j], c4);
            if (mr >= 6) svst1_f32(pg, &C[(i+5)*ldc+j], c5);
            if (mr >= 7) svst1_f32(pg, &C[(i+6)*ldc+j], c6);
            if (mr >= 8) svst1_f32(pg, &C[(i+7)*ldc+j], c7);
        }
    }
}

// 三路分发：按运行时能力选内核（基础 SVE 守卫下，V1 与 SVE2 都走 gemm_sve_8xvl）
static void gemm_dispatch(const float* A, const float* B, float* C,
                          int M, int N, int K, int ldc) {
    if (host_has_sve()) {                 // 基础 SVE 或 SVE2 置位（V1 / X2+）
        gemm_sve_8xvl(A, B, C, M, N, K, ldc);
    } else {                              // 仅 NEON（A76/A78/X1）
        // 真机这里调用 NEON 8×4 内核；x86 下不编译此分支，仅演示分发逻辑
        (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)ldc;
    }
}
#endif // __ARM_FEATURE_SVE

// ============================================================================
//  §六  迁移清单（NEON 8x4 → SVE VLA 内核 的逐条改动）
// ----------------------------------------------------------------------------
//   1. 头文件：<arm_neon.h> → 追加 <arm_sve.h>
//   2. 编译：   -march=armv8-a+simd → 基础 SVE: -march=armv8.4-a+sve（V1 用）
//                                        SVE2:   -march=armv9-a+sve2（X2/V2/X4 用）
//   3. 类型：   float32x4_t → svfloat32_t（不透明、长度未知）
//   4. 块尺寸： NR 写死 4 → NR = svcntw()（运行时）
//   5. 载入：   vld1q_f32 → svld1_f32(pg, ...)（pg 谓词）
//   6. 乘加：   vmlaq_f32 → svmla_f32_x(pg, ...)（同谓词）
//   7. 广播：   vdupq_n_f32 → svdup_n_f32（语义一致）
//   8. 尾部：   手写 for 补 N%4 → svwhilelt_b32 生成 pg，主循环一体化
//   9. 分发：   加 getauxval 三路判定（HWCAP2_SVE2→SVE2 / HWCAP_SVE→基础SVE / 否→NEON）
//  10. 收益：   一份 VLA 内核（基础 SVE 指令）覆盖 V1(8)/X2(4)/X4(16)；
//              SVE2 额外提供 int8 svdot 矩阵乘（int8 GEMM/量化见第五部分）
// ============================================================================

// ---- 工具：标量参考 + 对拍 --------------------------------------------------
static void gemm_ref(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

static bool check_close(const float* a, const float* b, int n, float tol=1e-3f) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i]-b[i]) > tol) { printf("  mismatch @%d: %g vs %g\n", i, a[i], b[i]); return false; }
    return true;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- main：跑仿真 + 对拍，直观看到"一份代码自适应多 VL"----------------------
int main() {
    const int M = 64, N = 70, K = 64;   // N=70：故意非 4 的倍数，逼出尾部
    std::vector<float> A(M*K), B(K*N), C(M*N), Cref(M*N), Cneon(M*N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    gemm_ref(A.data(), B.data(), Cref.data(), M, N, K);

    printf("=== §三 VLA 单代码路径：一份 gemm_vla_sim 自适应不同核 ===\n");
    const char* names[] = {"NEON_A76(4 lane)", "SVE_V1(256b,VL=8)",
                           "SVE2_X2(128b,VL=4)", "SVE2_X4(512b,VL=16)"};
    CoreSim cores[] = {CoreSim::NEON_A76, CoreSim::SVE_V1_256,
                       CoreSim::SVE2_X2_128, CoreSim::SVE2_X4_512};
    for (int t = 0; t < 4; ++t) {
        double t0 = now_ms();
        gemm_vla_sim(A, B, C, M, N, K, cores[t], 8);
        double dt = now_ms() - t0;
        bool ok = check_close(C.data(), Cref.data(), M*N);
        printf("  [%s] VL=%d  结果%s  用时 %.2f ms\n",
               names[t], vl_of(cores[t]), ok ? "一致" : "错", dt);
    }

    printf("\n=== §四 谓词消除尾部：NEON 固定 4 lane 需要 remainder 循环 ===\n");
    double t0 = now_ms();
    gemm_neon_fixed4(A, B, Cneon, M, N, K, 8);
    double dt = now_ms() - t0;
    bool ok = check_close(Cneon.data(), Cref.data(), M*N);
    printf("  [NEON 固定4] 结果%s  用时 %.2f ms  （尾部 %d 列走标量 remainder）\n",
           ok ? "一致" : "错", dt, N % 4);

    printf("\n=== §二 真机 SVE / SVE2 三路检测 ===\n");
    if (host_has_sve2())
        printf("  本机 HWCAP2_SVE2 置位 → SVE2 内核（X2/V2/X4；同份 FP32 内核也能跑，额外有 int8 svdot）\n");
    else if (host_has_sve())
        printf("  本机 HWCAP_SVE 置位（无 SVE2）→ 基础 SVE 内核（V1；svcntw 运行时决定 VL=8）\n");
    else
        printf("  本机无 SVE（x86 沙箱/老核）→ 仅演示仿真与代码形态\n");

#if defined(__ARM_FEATURE_SVE)
    printf("\n=== §五 真 SVE/SVE2 8×VL 内核编译通过，运行时 VL=%d ===\n", svcntw());
    // 真机可在此调用 gemm_sve_8xvl(A.data(), B.data(), C.data(), M, N, K, N);
#endif

    printf("\n=== §六 迁移清单（见文件顶部注释，共 10 步）===\n");
    printf("  NEON 8x4  →  SVE VLA：类型不透明化 / NR=svcntw() / 谓词取代 remainder\n");
    printf("  基础 SVE(-march=armv8.4-a+sve, V1) 与 SVE2(-march=armv9-a+sve2, X2+) 共用同一份 FP32 内核\n");
    return 0;
}
