// ============================================================================
//  NEON intrinsics 实战 demo（跨平台：ARM 走 NEON，x86 走 SSE2，否则标量）
//  演示的核心 intrinsic: vld1q_f32 / vst1q_f32 / vaddq_f32 /
//                        vdupq_n_f32 / vmlaq_f32 / 以及水平归约(vpadd)
//  编译:
//    ARM Linux : g++ -O3 -march=armv8-a+simd -std=c++17 neon_demo.cpp -o neon_demo
//    x86 MinGW : g++ -O3 -std=c++17 neon_demo.cpp -o neon_demo.exe
//    x86 MSVC  : cl /O2 /std:c++17 /EHsc neon_demo.cpp
//  运行: ./neon_demo
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// 后端选择：先判 ARM(NEON) 再判 x86(SSE2)，都没有就退化标量
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || \
    defined(_M_ARM) || defined(__ARM_NEON)
  #include <arm_neon.h>
  #define SIMD_BACKEND "NEON (ARM)"
  #define HAVE_NEON 1
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
      defined(_M_IX86)
  #include <emmintrin.h>   // SSE2
  #define SIMD_BACKEND "SSE2 (x86)"
  #define HAVE_SSE 1
#else
  #define SIMD_BACKEND "scalar (no SIMD)"
#endif

// ===========================================================================
// 一、标量参考实现（用于正确性对拍）
// ===========================================================================
static void add_scalar(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
static void saxpy_scalar(float alpha, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
static float dot_scalar(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ===========================================================================
// 二、NEON 实现（ARM 平台真实执行）
// ===========================================================================
#if defined(HAVE_NEON)

// --- 向量加法: c = a + b ---
// vld1q_f32 : 从连续地址加载 4 个 float 到 128 位寄存器(4 个 lane)
// vaddq_f32 : 4 个 lane 同时相加 (lane0+0, lane1+1, ...)
// vst1q_f32 : 把 128 位寄存器存回连续地址
// 余数(tail): N 不能被 4 整除时，标量补完剩余元素
static void add_neon(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

// --- SAXPY: y = alpha * x + y (融合乘加) ---
// vdupq_n_f32 : 把标量 alpha 广播(broadcast)到 4 个 lane
// vmlaq_f32   : 融合乘加, vy = vy + va * vx  (一条指令完成乘+加)
static void saxpy_neon(float alpha, const float* x, float* y, size_t n) {
    float32x4_t va = vdupq_n_f32(alpha);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vmlaq_f32(vy, va, vx);
        vst1q_f32(y + i, vy);
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

// --- 点积: sum(a[i]*b[i])，含水平归约(horizontal reduction) ---
// 向量乘加累加到 acc 的 4 个 lane 后，需要把 4 个 lane 加起来得到标量
// vget_low/high : 把 128 位拆成两个 64 位(各 2 个 float)
// vpadd_f32     : 两两相加, 用于把 2 个 float 归约成 1 个
static float dot_neon(const float* a, const float* b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t s2 = vadd_f32(lo, hi);   // 4 个 lane -> 2 个
    float32x2_t s1 = vpadd_f32(s2, s2);  // 2 个 -> 1 个(复制后两两加)
    float s = vget_lane_f32(s1, 0);
    for (; i < n; ++i) s += a[i] * b[i]; // 余数
    return s;
}

#endif // HAVE_NEON

// ===========================================================================
// 三、SSE2 实现（x86 平台，使 demo 在你的 Windows 上也能跑）
//   与 NEON 一一对应，方便对照学习
// ===========================================================================
#if defined(HAVE_SSE)

static void add_sse(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
static void saxpy_sse(float alpha, const float* x, float* y, size_t n) {
    __m128 va = _mm_set1_ps(alpha);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vy = _mm_loadu_ps(y + i);
        vy = _mm_add_ps(vy, _mm_mul_ps(va, vx));
        _mm_storeu_ps(y + i, vy);
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}
static float dot_sse(const float* a, const float* b, size_t n) {
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

#endif // HAVE_SSE

// ===========================================================================
// 四、统一入口（按后端选择实现）
// ===========================================================================
static void add_simd(const float* a, const float* b, float* c, size_t n) {
#if defined(HAVE_NEON)
    add_neon(a, b, c, n);
#elif defined(HAVE_SSE)
    add_sse(a, b, c, n);
#else
    add_scalar(a, b, c, n);
#endif
}
static void saxpy_simd(float alpha, const float* x, float* y, size_t n) {
#if defined(HAVE_NEON)
    saxpy_neon(alpha, x, y, n);
#elif defined(HAVE_SSE)
    saxpy_sse(alpha, x, y, n);
#else
    saxpy_scalar(alpha, x, y, n);
#endif
}
static float dot_simd(const float* a, const float* b, size_t n) {
#if defined(HAVE_NEON)
    return dot_neon(a, b, n);
#elif defined(HAVE_SSE)
    return dot_sse(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

// ===========================================================================
// 五、main: 正确性对拍 + 简单基准
// ===========================================================================
static bool approx_eq(const float* a, const float* b, size_t n, float tol = 1e-4f) {
    for (size_t i = 0; i < n; ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    const size_t N = 1 << 16;            // 65536
    std::vector<float> a(N), b(N), y(N), c(N), c_ref(N), y_ref(N);

    std::srand(42);
    for (size_t i = 0; i < N; ++i) {
        a[i] = (float)std::rand() / RAND_MAX;
        b[i] = (float)std::rand() / RAND_MAX;
        y[i] = (float)std::rand() / RAND_MAX;
        c[i] = c_ref[i] = 0.0f;
        y_ref[i] = y[i];
    }
    const float alpha = 2.5f;

    printf("backend : %s\n", SIMD_BACKEND);
    printf("N       : %zu\n\n", N);

    // --- 正确性对拍 ---
    add_scalar(a.data(), b.data(), c_ref.data(), N);
    add_simd  (a.data(), b.data(), c.data(),     N);
    printf("[add ] SIMD vs scalar : %s\n", approx_eq(c.data(), c_ref.data(), N) ? "OK" : "FAIL");

    saxpy_scalar(alpha, a.data(), y_ref.data(), N);
    saxpy_simd  (alpha, a.data(), y.data(),     N);
    printf("[saxp] SIMD vs scalar : %s\n", approx_eq(y.data(), y_ref.data(), N) ? "OK" : "FAIL");

    float d_ref  = dot_scalar(a.data(), b.data(), N);
    float d_simd = dot_simd  (a.data(), b.data(), N);
    printf("[dot ] SIMD=%.6f scalar=%.6f : %s\n\n",
           d_simd, d_ref, (std::fabs(d_simd - d_ref) < 1e-3f) ? "OK" : "FAIL");

    // --- 简单基准 (dot, 重复多次) ---
    const int REPS = 2000;
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile float sink = 0.0f;
    for (int r = 0; r < REPS; ++r) sink += dot_simd(a.data(), b.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double bytes = (double)REPS * N * 2 * sizeof(float);
    printf("dot benchmark: %.2f ms total, %.1f MB/s\n", ms, bytes / ms / 1e6);
    (void)sink;

    return 0;
}
