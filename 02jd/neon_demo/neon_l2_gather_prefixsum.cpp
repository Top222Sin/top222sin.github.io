// ============================================================
//  NEON 进阶两坑：gather/scatter 与 前缀和并行化
//
//  配套《00NEON基础学习.docx》
//   - §11「什么是 gather/scatter 操作？」—— 代码落地版
//   - §12「什么是前缀和并行化？」     —— 原理 + 代码版
//
//  结构对齐笔记：一、定义 / 二、为什么（接回前面知识）
//                / 三、代码 / 四、一句话收口
//
//  编译：
//   ARM: g++ -O3 -march=armv8-a+simd -std=c++17 neon_l2_gather_prefixsum.cpp -o pits
//   x86: g++ -O3 -mavx2 -std=c++17 neon_l2_gather_prefixsum.cpp -o pits.exe
//  运行：./pits      （全部打印 OK 即正确，并与标量对拍）
//
//  注意：本沙箱无 C++ 编译器，代码为完整可编译版，需在你本机运行。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

// ============================================================
//  §一  gather / scatter（SIMD 处理非连续/间接访问的逃生通道）
//
//  一、定义（一句话）
//    gather：拿一组索引，从散布在各处的内存地址取数，按顺序塞进一个向量的连续 lane。
//    scatter：反过来，把向量的每个 lane 写回到一组索引指向的散布地址。
//
//  二、为什么存在（接回前面知识）
//    SIMD 的 vld1q 是「连续」载入——一次抓 16 个连续字节。但真实数据常常不连续：
//    AoS(struct{float x,y,z})、指针数组、稀疏下标、通道重排。这些一条 vld1q 抓不到
//    整齐的 4 个数——正是笔记 §4「编译器处理失败·卡4 非连续/间接访问」那个点。
//    解法两条路：① AoS→SoA 让数据连续（正道）；② gather/scatter 不改布局直接按索引取（兜底）。
//
//  三、代价
//    gather 比连续 vld1q 慢得多：每个 lane 地址可能落在不同缓存行，硬件要发起多次
//    独立访存、易触发 cache miss。GPU 上 warp 各线程 gather 若散开，吃不到「合并访问」
//    收益，直接掉一个数量级（接 §CUDA 合并访问）。铁律：能连续就别 gather。
//
//  四、和指令集的对应（别记错）
//    x86 AVX2：真·任意地址 gather（_mm256_i32gather_ps）；AVX-512 另有 scatter。
//    ARM SVE/SVE2：真·内存 gather/scatter（ld1w/st1w 配索引向量）。
//    经典 NEON(ARMv8)：没有任意地址 gather。但有「结构化载入/存储」vld3q/vst3q——
//      对固定步长 AoS 是天然 gather/scatter（下面的 NEON 示例就用它）。
//      vqtbl1q_u8 只是寄存器内查表，不是访存。ncnn 用 ext/vtrn/vzip 做固定重排也是同理。
// ============================================================

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

// NEON 结构化载入：从 AoS(struct{float x,y,z}) 一次 gather 出 3 个 SoA 向量
// vld3q_f32 读 48 字节（4 个 struct），把每第 0/1/2 个 float 分别聚成向量 —— 这是
// 固定步长 AoS 的「gather」。vst3q_f32 反之，是「scatter」。
void aos_to_soa_neon(const float* pts, float* xs, float* ys, float* zs, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x3_t v = vld3q_f32(pts + i * 3);  // 结构化 gather：x/y/z 各自成向量
        vst1q_f32(xs + i, v.val[0]);
        vst1q_f32(ys + i, v.val[1]);
        vst1q_f32(zs + i, v.val[2]);
    }
    for (; i < n; ++i) {                 // 标量补尾（N 非 4 倍数时不能漏）
        xs[i] = pts[i * 3 + 0];
        ys[i] = pts[i * 3 + 1];
        zs[i] = pts[i * 3 + 2];
    }
}

#else

// 非 NEON 平台（如你这台 x86）用标量版对拍，保证文件可编译运行
void aos_to_soa_neon(const float* pts, float* xs, float* ys, float* zs, int n) {
    for (int i = 0; i < n; ++i) {
        xs[i] = pts[i * 3 + 0];
        ys[i] = pts[i * 3 + 1];
        zs[i] = pts[i * 3 + 2];
    }
}

#endif

// x86 AVX2 真·任意地址 gather：按 idx 把 8 个散布 float 聚成一个向量（scale=4=sizeof float）
// 这是笔记里说的「不改布局、直接按索引取」的标准写法；AVX2 有 gather 无 scatter。
#ifdef __AVX2__
#include <immintrin.h>
__m256 gather_avx2(const float* base, const int* idx) {
    return _mm256_i32gather_ps(base, _mm_loadu_si256((const __m256i*)idx), 4);
}
#endif

// ============================================================
//  §二  前缀和并行化（打破「循环依赖/递推」让向量化/GPU 能做）
//
//  一、定义（一句话）
//    前缀和（prefix sum / scan）：out[i] = in[0]+in[1]+...+in[i]（inclusive）。
//    「并行化」指用并行算法替代朴素串行递推，让每一轮加法彼此独立、可被 SIMD/GPU 打满。
//
//  二、为什么需要（接回前面知识）
//    朴素前缀和是递推：out[i] = out[i-1] + in[i]。out[i] 依赖 out[i-1]，本质是串行——
//    这正是笔记 §4「编译器处理失败·卡2 循环依赖/递推」那个点，向量化器直接放弃，
//    且「手写 intrinsics 也救不了，得改算法」。前缀和并行化就是那个「改算法」的解法。
//
//  三、两种经典并行 scan 算法（代码见下）
//    (A) Hillis-Steele（相邻两两加，log 轮）：每轮做 stride=2^k 的「独立」加法，
//        整轮数据并行 → 可向量化。代价 O(n log n)，实现简单。
//    (B) Blelloch（上行归约 + 下行扫描，work-efficient）：O(n)，每阶段内层循环同样
//        数据并行。生产级 scan 多用它（GPU 经典原语）。要求长度为 2 的幂（否则 pad）。
//    关键技巧：用双缓冲 / 快照，让「本轮读取」全部来自上一轮已定稿的值，消除轮内依赖。
//
//  四、一句话收口
//    串行递推不可向量化；Hillis-Steele / Blelloch 把依赖链打散成「每轮独立加法」，
//    既能被 SIMD 向量化，也是 CUDA 里 scan 原语的基石。遇到 a[i] 依赖 a[i-1] 这类
//    递推，先想「能不能改成并行 scan」，而不是冲去手写 intrinsics。
// ============================================================

// 串行 inclusive 前缀和（递推，不可向量化——作为正确性对拍基准）
void prefix_sum_serial_inclusive(const std::vector<float>& in, std::vector<float>& out) {
    int n = (int)in.size();
    out.assign(n, 0.f);
    float acc = 0.f;
    for (int i = 0; i < n; ++i) { acc += in[i]; out[i] = acc; }
}

// 串行 exclusive 前缀和（对拍 Blelloch 用）
void prefix_sum_serial_exclusive(const std::vector<float>& in, std::vector<float>& out) {
    int n = (int)in.size();
    out.assign(n, 0.f);
    float acc = 0.f;
    for (int i = 0; i < n; ++i) { out[i] = acc; acc += in[i]; }
}

// (A) Hillis-Steele inclusive scan：双缓冲消除轮内依赖，每轮完全数据并行
void hillis_steele_inclusive(const std::vector<float>& in, std::vector<float>& out) {
    int n = (int)in.size();
    std::vector<float> a = in;   // 上一轮快照（读取源）
    std::vector<float> b(n);     // 本轮结果（写入目标）
    for (int stride = 1; stride < n; stride <<= 1) {
        for (int i = 0; i < n; ++i) {            // 轮内各 i 互不依赖 → 可向量化
            b[i] = a[i] + (i >= stride ? a[i - stride] : 0.f);
        }
        std::swap(a, b);                          // a 持有最新一轮
    }
    out = std::move(a);
}

// (B) Blelloch exclusive scan（要求 n 为 2 的幂，否则需 pad）
void blelloch_exclusive(const std::vector<float>& in, std::vector<float>& out) {
    int n = (int)in.size();   // 调用方需保证 n 是 2 的幂
    std::vector<float> a = in;
    // up-sweep（归约）
    for (int d = 1; d < n; d <<= 1) {
        for (int k = 0; k < n; k += 2 * d) {      // 各 k 区间互不重叠 → 数据并行
            a[k + 2 * d - 1] += a[k + d - 1];
        }
    }
    a[n - 1] = 0.f;                                // 清除总根，进入下行
    // down-sweep（扫描）
    for (int d = n >> 1; d >= 1; d >>= 1) {
        for (int k = 0; k < n; k += 2 * d) {      // 各 k 区间互不重叠 → 数据并行
            float t = a[k + d - 1];
            a[k + d - 1] = a[k + 2 * d - 1];
            a[k + 2 * d - 1] += t;
        }
    }
    out = std::move(a);
}

// ============================================================
//  验证：与串行基准对拍，全部 OK 即正确
// ============================================================
static int g_fail = 0;
static void check(const char* name, const std::vector<float>& got,
                  const std::vector<float>& ref) {
    if (got.size() != ref.size()) { printf("  [FAIL] %s: size\n", name); g_fail++; return; }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-4f) {
            printf("  [FAIL] %s @%zu: %.4f vs %.4f\n", name, i, got[i], ref[i]);
            g_fail++; return;
        }
    }
    printf("  [OK]   %s\n", name);
}

int main() {
    printf("=== §二 前缀和并行化 ===\n");
    const int n = 16;  // 2 的幂，方便 Blelloch
    std::vector<float> in(n), sincl(n), sexcl(n), hincl(n), bexcl(n);
    for (int i = 0; i < n; ++i) in[i] = (float)(i + 1);  // 1..16，确定性

    prefix_sum_serial_inclusive(in, sincl);
    prefix_sum_serial_exclusive(in, sexcl);
    hillis_steele_inclusive(in, hincl);
    blelloch_exclusive(in, bexcl);

    check("Hillis-Steele vs serial(inclusive)", hincl, sincl);
    check("Blelloch     vs serial(exclusive)", bexcl, sexcl);

    printf("\n=== §一 gather（AoS->SoA，vld3q 结构化载入）===\n");
    std::vector<float> pts(n * 3);
    for (int i = 0; i < n * 3; ++i) pts[i] = (float)(i + 1);
    std::vector<float> xs(n), ys(n), zs(n);
    aos_to_soa_neon(pts.data(), xs.data(), ys.data(), zs.data(), n);
    // 对拍：xs[i] 应等于 pts[3i]
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
        if (xs[i] != pts[3 * i] || ys[i] != pts[3 * i + 1] || zs[i] != pts[3 * i + 2])
            ok = false;
    printf("  [%s]   AoS->SoA 字段提取\n", ok ? "OK" : "FAIL");
    if (!ok) g_fail++;

    printf("\n%s\n", g_fail == 0 ? "ALL OK" : "HAS FAILURE");
    return g_fail == 0 ? 0 : 1;
}
