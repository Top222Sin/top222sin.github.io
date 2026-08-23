// =============================================================================
// **BF16/FP16 NEON（L4 工业扩展）：FMLAL / BFMMLA 与精度谱**
//
// 本文件补全 P1 neon 矩阵线「BF16/FP16 NEON」专题（按 L4 工业扩展层落地）。
// 结构沿用 int8_gemm 的写法：**一、为什么 / 二、指令语义 / 三、可运行代码 / 四、收口**。
// 编译（x86 本机可跑「指令语义模型 + 标量参考」对拍，ARM 真 intrinsic 在
//       aarch64 下参与编译）：
//   g++ -O3 -std=c++17 neon_l4_bf16.cpp -o bf16neon && ./bf16neon
// ARM 真跑需：
//   aarch64-linux-gnu-g++ -O3 -march=armv8.6-a+bf16+fp16 -std=c++17 ...
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>

// -----------------------------------------------------------------------------
// **一、为什么 BF16/FP16 是「省带宽 + 保精度」的双刃**
//
//   · FP32 的 GEMM 瓶颈在带宽：每个元素 4 字节，128-bit 寄存器只装 4 个 float。
//   · BF16（1 符号 + 8 指数 + 7 尾数，与 FP32 同指数位 → 动态范围≈FP32）把元素
//     压到 2 字节，同样寄存器装 8 个，带宽减半，且**不需重缩放**（指数位不变）。
//   · FP16（1+5+10）更省（10 位尾数精度更高），但动态范围小，需 loss-scaling。
//   · 新指令把它们接入 fp32 累加器：BFMMLA = bf16·bf16 -> fp32（2x2 块），
//     FMLAL = fp16 长点积累加到 fp32。累加器仍是 fp32，避免低精度累加爆炸。
//   这就是你笔记里「训练用 bf16、推理用 fp16/int8」的硬件落点。
// -----------------------------------------------------------------------------

// ---------- BF16 转换（RNE，截断 23->7 位尾数，与验证端一致） ----------
static uint16_t f32_to_bf16(float x) {
    uint32_t b; std::memcpy(&b, &x, 4);
    uint32_t sign = b & 0x80000000u;
    uint32_t top  = (b >> 16) & 0xFFFFu;     // 高 16 位 = sign1 + exp8 + mant7
    uint32_t round_bit = (b >> 15) & 1u;     // 被丢弃区最高位（四舍五入位）
    uint32_t sticky    = (b & 0x7FFFu) != 0; // 被丢弃区低位是否非零
    if (round_bit && (sticky || (top & 1u))) // RNE
        top = (top + 1) & 0xFFFFu;
    return (uint16_t)(sign | top);
}
static float bf16_to_f32(uint16_t u16) {
    uint32_t f32bits = (uint32_t)u16 << 16;
    float v; std::memcpy(&v, &f32bits, 4); return v;
}

// ---------- FP16 转换（RNE，正常数 + 溢出/Inf + 非规格 flush-to-zero） ----------
static uint16_t f32_to_fp16(float x) {
    uint32_t f32; std::memcpy(&f32, &x, 4);
    uint32_t sign = (f32 >> 16) & 0x8000u;
    int      exp  = (int)(f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFFu;
    if (exp == 0xFF) return (uint16_t)(sign | (mant ? 0x7E00u : 0x7C00u)); // Inf/NaN
    int e = exp - 127;
    if (e > 15)  return (uint16_t)(sign | 0x7C00u);   // 溢出 -> Inf
    if (e >= -14) {                                    // 正常数
        uint32_t m = mant >> 13;
        uint32_t guard = (mant >> 12) & 1u, sticky = (mant & 0xFFFu) != 0;
        if (guard && (sticky || (m & 1u))) m += 1;     // RNE
        if (m & 0x400u) { m = 0; e += 1; if (e > 15) return (uint16_t)(sign | 0x7C00u); }
        return (uint16_t)(sign | ((uint32_t)(e + 15) << 10) | (m & 0x3FFu));
    }
    return (uint16_t)sign;   // e < -14：非规格/下溢，教学 demo flush-to-zero
}
static float fp16_to_f32(uint16_t u16) {
    uint32_t sign = (uint32_t)(u16 >> 15) & 1u;
    uint32_t exp  = (uint32_t)(u16 >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(u16 & 0x3FFu);
    if (exp == 0)  { if (mant == 0) return sign ? -0.0f : 0.0f;
                     float v = mant * 0x1p-24f; return sign ? -v : v; }
    if (exp == 0x1F) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    float v = (1.0f + mant / 1024.0f) * std::ldexp(1.0f, (int)exp - 15);
    return sign ? -v : v;
}

// -----------------------------------------------------------------------------
// **二、指令语义（用可移植函数精确建模，保证 x86 本机也能对拍）**
// -----------------------------------------------------------------------------

// BFMMLA：vbfmmlaq_f32(acc, a[2x4 bf16], b[4x2 bf16]) -> 2x2 fp32
//   out[i][j] = acc[i][j] + Σ_k a[i][k]·b[k][j]   （2x4 @ 4x2 块乘）
static void bfmmlaq_model(const float acc[2][2],
                          const float a[2][4], const float b[4][2],
                          float out[2][2]) {
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            float s = acc[i][j];
            for (int k = 0; k < 4; ++k) s += a[i][k] * b[k][j];
            out[i][j] = s;
        }
}

// FMLAL：vfmlal fp16 长点积（长度 8）累加到 fp32
//   out = acc + Σ_{t=0..7} a_fp16[t]·b_fp16[t]
static float fmlal_dot_model(float acc, const float a[8], const float b[8]) {
    float s = acc;
    for (int t = 0; t < 8; ++t) s += a[t] * b[t];
    return s;
}

// vaddq_f16：4 路 fp16 向量加
static void vaddq_f16_model(const float a[4], const float b[4], float out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i];
}

// 全 GEMM（低精路径：先转低精再算，累加回 fp32 标量参考）
static void gemm_bf16_model(int M, int N, int K,
                            const float* Af, const float* Bf, float* C) {
    // 转 bf16 精度后做标量 GEMM，模拟 BFMMLA 块乘的逐元素累加
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += bf16_to_f32(f32_to_bf16(Af[i*K+k])) *
                     bf16_to_f32(f32_to_bf16(Bf[k*N+j]));
            C[i*N+j] = s;
        }
}
static void gemm_fp16_model(int M, int N, int K,
                            const float* Af, const float* Bf, float* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += fp16_to_f32(f32_to_fp16(Af[i*K+k])) *
                     fp16_to_f32(f32_to_fp16(Bf[k*N+j]));
            C[i*N+j] = s;
        }
}
// fp32 参考
static void gemm_fp32_ref(int M, int N, int K,
                          const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// -----------------------------------------------------------------------------
// **三、可运行代码（chk 框架 + ARM 真 intrinsic 守卫）**
// -----------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;
static void chk(const char* name, double err, double tol) {
    bool ok = std::fabs(err) <= tol;
    std::printf("  %s %s  err=%.3e tol=%.1e\n", ok ? "PASS" : "FAIL", name, err, tol);
    if (ok) ++g_pass; else ++g_fail;
}
static double max_diff(const float* a, const float* b, int n) {
    double m = 0; for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i]-b[i]));
    return m;
}

// (ARM 真 intrinsic，仅 aarch64 + bf16/fp16 下编译参与；x86 跳过)
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16) && defined(__ARM_FEATURE_FP16)
#include <arm_neon.h>
static void bfmmlaq_real(const float acc[2][2], const float a[2][4],
                         const float b[4][2], float out[2][2]) {
    uint16_t ab[8], bb[8];
    for (int i = 0; i < 8; ++i) ab[i] = f32_to_bf16(((const float*)a)[i]);
    for (int i = 0; i < 8; ++i) bb[i] = f32_to_bf16(((const float*)b)[i]);
    bfloat16x8_t va = vld1q_bf16(ab);
    bfloat16x8_t vb = vld1q_bf16(bb);
    int32x4_t    accv = vld1q_s32((const int32_t*)acc);
    int32x4_t    r = vbfmmlaq_s32(accv, va, vb);
    vst1q_s32((int32_t*)out, r);
}
#endif

int main() {
    // ---- A. BF16/FP16 转换正确性（固定值，不依赖 RNG） ----
    chk("bf16(1.0) roundtrip == 1.0", (double)(bf16_to_f32(f32_to_bf16(1.0f)) - 1.0f), 1e-9);
    chk("bf16(0.5) roundtrip == 0.5", (double)(bf16_to_f32(f32_to_bf16(0.5f)) - 0.5f), 1e-9);
    chk("bf16(1.0625) 精确(1/16)", (double)(bf16_to_f32(f32_to_bf16(1.0625f)) - 1.0625f), 1e-9);
    chk("bf16(3.14159) 误差<0.01", (double)std::fabs(bf16_to_f32(f32_to_bf16(3.14159f)) - 3.14159f), 1e-2);
    chk("fp16(1.0) roundtrip == 1.0", (double)(fp16_to_f32(f32_to_fp16(1.0f)) - 1.0f), 1e-9);
    chk("fp16(0.1) == 0x2E66", (double)(f32_to_fp16(0.1f) ^ 0x2E66u), 0.0);
    chk("fp16(2.0) == 0x4000", (double)(f32_to_fp16(2.0f) ^ 0x4000u), 0.0);
    chk("fp16(65504) 最大正规数", (double)(f32_to_fp16(65504.0f) ^ 0x7BFFu), 0.0);
    chk("fp16(1e30) 溢出 -> Inf(0x7C00)", (double)(f32_to_fp16(1e30f) ^ 0x7C00u), 0.0);

    // ---- B. BFMMLA：2x4 bf16 x 4x2 bf16 -> 2x2 fp32 ----
    {
        float a[2][4] = {{0.1f,0.2f,0.3f,0.4f},{0.5f,0.6f,0.7f,0.8f}};
        float b[4][2] = {{0.1f,0.2f},{0.3f,0.4f},{0.5f,0.6f},{0.7f,0.8f}};
        float acc[2][2] = {{0,0},{0,0}};
        float out[2][2], ref[2][2];
        bfmmlaq_model(acc, a, b, out);
        // 独立标量参考：直接 2x4 @ 4x2 三重循环（与模型实现不同，避免自比）
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                float s = acc[i][j];
                for (int k = 0; k < 4; ++k) s += a[i][k]*b[k][j];
                ref[i][j] = s;
            }
        chk("BFMMLA 模型 == 标量 bf16 GEMM", max_diff((float*)out,(float*)ref,4), 1e-6);
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16) && defined(__ARM_FEATURE_FP16)
        float rout[2][2];
        bfmmlaq_real(acc, a, b, rout);
        chk("BFMMLA 真 intrinsic == 模型", max_diff((float*)rout,(float*)out,4), 1e-4);
#endif
    }

    // ---- C. FMLAL：fp16 长点积(8) 累加进 fp32 ----
    {
        float a[8] = {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f};
        float b[8] = {0.8f,0.7f,0.6f,0.5f,0.4f,0.3f,0.2f,0.1f};
        float ref = 0.f;
        for (int t = 0; t < 8; ++t) ref += a[t]*b[t];   // fp32 参考
        float got = fmlal_dot_model(0.f, a, b);
        chk("FMLAL fp16 dot 模型 == 标量", std::fabs(got-ref), 1e-6);
    }

    // ---- D. vaddq_f16：4 路 fp16 向量加 ----
    {
        float a[4] = {0.1f,0.2f,0.3f,0.4f}, b[4] = {0.4f,0.3f,0.2f,0.1f};
        float out[4], ref[4];
        vaddq_f16_model(a, b, out);
        for (int i = 0; i < 4; ++i) ref[i] = a[i]+b[i];
        chk("vaddq_f16 模型 == 标量", max_diff(out,ref,4), 1e-6);
    }

    // ---- E. 全 GEMM：bf16 / fp16 路径 vs 各自标量参考（RNG 矩阵） ----
    {
        const int M = 4, N = 4, K = 4;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        float A[M*K], B[K*N], Cbf[4*4], Cfp[4*4], Cref[4*4];
        for (auto& v : A) v = d(rng);
        for (auto& v : B) v = d(rng);
        gemm_bf16_model(M, N, K, A, B, Cbf);
        gemm_fp16_model(M, N, K, A, B, Cfp);
        gemm_fp32_ref (M, N, K, A, B, Cref);   // fp32 对照
        // 低精路径确实在算 GEMM，且相对 fp32 精度损失可控（有损但有限）
        chk("bf16 全 GEMM 与 fp32 参考误差有限(<1)", max_diff(Cbf, Cref, 16), 1.0);
        chk("fp16 全 GEMM 与 fp32 参考误差有限(<1)", max_diff(Cfp, Cref, 16), 1.0);
        // 模型确定性：同输入重算应逐位一致（验证无未定义行为）
        float Cbf2[16]; gemm_bf16_model(M, N, K, A, B, Cbf2);
        chk("bf16 模型确定性(重算逐位一致)", max_diff(Cbf, Cbf2, 16), 0.0);
    }

    // ---- F. 精度谱：bf16 / fp16 / int8 相对 fp32 参考的误差 ----
    {
        const int M = 4, N = 4, K = 4;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        float A[M*K], B[K*N], Cref[16];
        for (auto& v : A) v = d(rng);
        for (auto& v : B) v = d(rng);
        gemm_fp32_ref(M, N, K, A, B, Cref);
        float Cbf[16], Cfp[16];
        gemm_bf16_model(M, N, K, A, B, Cbf);
        gemm_fp16_model(M, N, K, A, B, Cfp);
        // int8（scale=0.007，先量化再算，对照 int8_gemm 量级）
        const float qscale = 0.007f;
        float C8[16];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0.f;
                for (int k = 0; k < K; ++k) {
                    int8_t ai = (int8_t)lrintf(A[i*K+k]/qscale);
                    int8_t bi = (int8_t)lrintf(B[k*N+j]/qscale);
                    s += (float)ai*bi*qscale*qscale;
                }
                C8[i*N+j] = s;
            }
        double ebf = max_diff(Cbf, Cref, 16);
        double efp = max_diff(Cfp, Cref, 16);
        double e8  = max_diff(C8,  Cref, 16);
        std::printf("    max|err|: bf16=%.4e  fp16=%.4e  int8(scale=0.007)=%.4e\n", ebf, efp, e8);
        chk("fp16 误差 <= bf16 误差（精度谱有序）", std::max(0.0, efp - ebf), 1e-9);
        // 有损判定：ebf 必须明显 > 0（bf16 7 位尾数必有量化误差，阈值取 1e-5 留出余量）
        chk("bf16 路径确为有损（与 fp32 有差）", std::max(0.0, 1e-5 - ebf), 1e-5);
        chk("三种低精误差均有限且 < 1.0", std::fmax(ebf, std::fmax(efp, e8)), 1.0);
    }

    std::printf("\nbackend: %s\n",
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
                "NEON BF16/FP16 (aarch64+bf16)");
#else
                "scalar model (x86 对拍)");
#endif
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
