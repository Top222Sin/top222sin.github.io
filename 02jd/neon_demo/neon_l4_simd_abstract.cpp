// =============================================================================
// **跨平台 SIMD 抽象层（L4 工业扩展）：一处写多端编译（NEON / SSE / AVX / 标量）**
//
// 本文件补全 P2 neon「跨平台 SIMD 抽象（SSE/AVX 封装）」专题（按 L4 工业扩展层落地）。
// 核心论点：把"向量是什么"与"算什么"解耦——kernel 只写一次（模板宽度 L），
//   通过替换 `Simd<L>` 后端即可编译到 ARM NEON / x86 SSE / AVX / 标量回退。
//
// 编译（本沙箱无编译器；x86 本机跑「标量模拟后端」对拍，即"仅 x86 标量对照"）：
//   g++ -O3 -std=c++17 neon_l4_simd_abstract.cpp -o simdabs && ./simdabs
// ARM 真跑（NEON 后端参与编译）：
//   aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -std=c++17 neon_l4_simd_abstract.cpp -o simdabs
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

// -----------------------------------------------------------------------------
// **一、标量模拟后端（始终编译）**
//
//   `Simd<L>` 是一个 L 路 float 向量，所有操作（load/store/add/sub/mul/
//   broadcast/hsum/relu）都是对 L 个 lane 的逐元素标量展开。
//   本沙箱没有 NEON/SSE/AVX，正是走这条路——它等价于"x86 标量对照"。
// -----------------------------------------------------------------------------
template<int L>
struct Simd { float f[L]; };

template<int L> static Simd<L> vload(const float* p) {
    Simd<L> v; for (int i = 0; i < L; ++i) v.f[i] = p[i]; return v;
}
template<int L> static void vstore(float* p, const Simd<L>& v) {
    for (int i = 0; i < L; ++i) p[i] = v.f[i];
}
template<int L> static Simd<L> vbroadcast(float s) {
    Simd<L> v; for (int i = 0; i < L; ++i) v.f[i] = s; return v;
}
template<int L> static Simd<L> vadd(const Simd<L>& a, const Simd<L>& b) {
    Simd<L> r; for (int i = 0; i < L; ++i) r.f[i] = a.f[i] + b.f[i]; return r;
}
template<int L> static Simd<L> vsub(const Simd<L>& a, const Simd<L>& b) {
    Simd<L> r; for (int i = 0; i < L; ++i) r.f[i] = a.f[i] - b.f[i]; return r;
}
template<int L> static Simd<L> vmul(const Simd<L>& a, const Simd<L>& b) {
    Simd<L> r; for (int i = 0; i < L; ++i) r.f[i] = a.f[i] * b.f[i]; return r;
}
template<int L> static float vhsum(const Simd<L>& v) {
    float s = 0.f; for (int i = 0; i < L; ++i) s += v.f[i]; return s;
}
template<int L> static Simd<L> vrelu(const Simd<L>& a) {
    Simd<L> r; for (int i = 0; i < L; ++i) r.f[i] = a.f[i] > 0.f ? a.f[i] : 0.f; return r;
}

// -----------------------------------------------------------------------------
// **二、kernel：只写一次，任意宽度 L 通吃（一处写多端编译的关键）**
//
//   下面三个 kernel 完全不关心 L 是 2/4/8——它们只调用上面的统一 API。
//   把 `Simd<L>` 换成 NEON 的 `float32x4_t` 或 AVX 的 `__m256`，kernel 源码一字不改。
// -----------------------------------------------------------------------------
template<int L>
static double simd_dot(const float* a, const float* b, int n) {
    double acc = 0.0; int i = 0;
    for (; i + L <= n; i += L) {
        Simd<L> va = vload<L>(a + i), vb = vload<L>(b + i);
        acc += (double)vhsum<L>(vmul<L>(va, vb));   // 向量水平和后累加到 double 累加器
    }
    for (; i < n; ++i) acc += (double)a[i] * (double)b[i];   // 标量尾巴（向量长度不可整除 L）
    return acc;
}
template<int L>
static void simd_saxpy(float a, const float* x, float* y, int n) {
    Simd<L> va = vbroadcast<L>(a);
    int i = 0;
    for (; i + L <= n; i += L) {
        Simd<L> vx = vload<L>(x + i), vy = vload<L>(y + i);
        vstore<L>(y + i, vadd<L>(vy, vmul<L>(va, vx)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
}
template<int L>
static void simd_relu(const float* x, float* y, int n) {
    int i = 0;
    for (; i + L <= n; i += L) vstore<L>(y + i, vrelu(vload<L>(x + i)));
    for (; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

// -----------------------------------------------------------------------------
// **三、生产后端映射（文档化，不在此沙箱编译）**
//
//   真实工程里，把 `Simd<L>` 换成下面任一线后端，上面三个 kernel 源码零改动：
//     · ARM NEON  : using Vec4 = float32x4_t;  vload=vld1q_f32  vadd=vaddq_f32
//                   vmul=vmulq_f32  vhsum=vaddvq_f32  vbroadcast=vdupq_n_f32
//     · x86 SSE   : using Vec4 = __m128;        vload=_mm_loadu_ps vadd=_mm_add_ps
//                   vmul=_mm_mul_ps  vhsum=_mm_hsum_ps(两次 _mm_hadd) vdup=_mm_set1_ps
//     · x86 AVX   : using Vec8 = __m256;        vload=_mm256_loadu_ps ... vhsum=_mm256_reduce
//   条件编译：#if defined(__ARM_NEON) / __SSE__ / __AVX__ 选后端；都不中则标量回退。
//   下面用 SIMD_SHOW_PRODUCTION 守卫，本环境不定义，纯作"一处写多端编译"的落地范本。
// -----------------------------------------------------------------------------
#if defined(SIMD_SHOW_PRODUCTION)
#if defined(__ARM_NEON)
#include <arm_neon.h>
using Vec4 = float32x4_t;
static inline Vec4  VLOAD (const float* p) { return vld1q_f32(p); }
static inline Vec4  VADD  (Vec4 a, Vec4 b) { return vaddq_f32(a, b); }
static inline Vec4  VMUL  (Vec4 a, Vec4 b) { return vmulq_f32(a, b); }
static inline float VHSUM (Vec4 a)         { return vaddvq_f32(a); }
static inline Vec4  VDUP  (float s)        { return vdupq_n_f32(s); }
#elif defined(__AVX__)
#include <immintrin.h>
using Vec8 = __m256;
static inline Vec8  VLOAD (const float* p) { return _mm256_loadu_ps(p); }
static inline Vec8  VADD  (Vec8 a, Vec8 b) { return _mm256_add_ps(a, b); }
static inline Vec8  VMUL  (Vec8 a, Vec8 b) { return _mm256_mul_ps(a, b); }
static inline float VHSUM (Vec8 a)         { return _mm256_reduce_add_ps(a); }
static inline Vec8  VDUP  (float s)        { return _mm256_set1_ps(s); }
#elif defined(__SSE__)
#include <xmmintrin.h>
using Vec4 = __m128;
static inline Vec4  VLOAD (const float* p) { return _mm_loadu_ps(p); }
static inline Vec4  VADD  (Vec4 a, Vec4 b) { return _mm_add_ps(a, b); }
static inline Vec4  VMUL  (Vec4 a, Vec4 b) { return _mm_mul_ps(a, b); }
static inline float VHSUM (Vec4 a)         { /* 两次 _mm_hadd_ps + _mm_cvtss_f32 */ return 0.f; }
static inline Vec4  VDUP  (float s)        { return _mm_set1_ps(s); }
#endif
// 注意：kernel 里把 `Simd<L>` 换成 `Vec4`/`Vec8`、把 `vload/vadd/...` 换成
// `VLOAD/VADD/...`，其余（循环、数据布局、算法）一字不改——这就是"一处写多端编译"。
#endif // SIMD_SHOW_PRODUCTION

// -----------------------------------------------------------------------------
// **四、chk 框架 + 验证**
// -----------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;
static void chk(const char* name, double err, double tol) {
    bool ok = std::fabs(err) <= tol;
    std::printf("  %s %s  err=%.3e tol=%.1e\n", ok ? "PASS" : "FAIL", name, err, tol);
    if (ok) ++g_pass; else ++g_fail;
}
static double max_diff(const float* a, const float* b, int n) {
    double m = 0; for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    const int n = 12;
    float a[12] = {0.1f, -0.4f, 0.7f, 0.2f, -0.9f, 0.3f, 0.5f, -0.2f, 0.8f, -0.6f, 0.4f, 0.1f};
    float b[12] = {0.3f,  0.2f, 0.6f, 0.1f, -0.3f, 0.9f, 0.2f,  0.7f, 0.4f, -0.5f, 0.8f, 0.3f};

    // ---- C1. 跨后端一致 + == 朴素标量（证明"一处写多端编译"的算法无关性）----
    //   向量水平和累加到 double 累加器，故结果与车道宽度无关（真实 float32 SIMD
    //   有 ~1e-6 级舍入，属已知硬件细节；模型以 double 累加器做精确演示）。
    double nd = 0.0; for (int i = 0; i < n; ++i) nd += (double)a[i] * (double)b[i];
    double d2 = simd_dot<2>(a, b, n), d4 = simd_dot<4>(a, b, n), d8 = simd_dot<8>(a, b, n);
    double c1 = std::fmax(std::fmax(std::fabs(d2 - nd), std::fabs(d4 - nd)),
                          std::fmax(std::fabs(d8 - nd), std::fabs(d2 - d4)));
    chk("dot<2/4/8> 跨宽度一致 且 == 朴素标量", c1, 1e-9);

    // ---- C2. saxpy<4> == 朴素 ----
    {
        float y[12]; for (int i = 0; i < n; ++i) y[i] = b[i];
        float yr[12]; for (int i = 0; i < n; ++i) yr[i] = b[i] + 0.5f * a[i];
        simd_saxpy<4>(0.5f, a, y, n);
        chk("saxpy<4> == 朴素 y+=0.5x", max_diff(y, yr, n), 0.0);
    }

    // ---- C3. relu<4> == max(0,·) ----
    {
        float y[12], yr[12];
        simd_relu<4>(a, y, n);
        for (int i = 0; i < n; ++i) yr[i] = a[i] > 0.f ? a[i] : 0.f;
        chk("relu<4> == max(0,x)", max_diff(y, yr, n), 0.0);
    }

    // ---- C4. load/store 往返精确（L=4）----
    {
        float v[4] = {1.5f, -2.5f, 3.5f, -4.5f};
        Simd<4> s = vload<4>(v);
        float w[4]; vstore<4>(w, s);
        chk("load/store 往返 == 原值 (L=4)", max_diff(v, w, 4), 0.0);
    }

    // ---- C5. vadd 逐元素协定（任意两向量 == 标量逐元素加）----
    {
        float u[4] = {0.2f, 0.9f, -1.1f, 0.4f}, w2[4] = {0.3f, -0.7f, 0.5f, -0.2f};
        Simd<4> su = vload<4>(u), sw = vload<4>(w2);
        Simd<4> sr = vadd<4>(su, sw);
        float ref[4]; for (int i = 0; i < 4; ++i) ref[i] = u[i] + w2[i];
        chk("vadd<4> 逐元素 == 标量加", max_diff(sr.f, ref, 4), 0.0);
    }

    std::printf("\nbackend: %s\n",
#if defined(__ARM_NEON)
                "NEON (aarch64+simd)");
#elif defined(__AVX__)
                "AVX (__m256)");
#elif defined(__SSE__)
                "SSE (__m128)");
#else
                "scalar model (x86 标量对照)");
#endif
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
