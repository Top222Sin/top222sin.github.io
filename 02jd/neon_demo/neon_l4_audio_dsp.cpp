// =============================================================================
// **音频 DSP 核（L4 工业扩展）**：RBJ 双二阶 / 前馈压缩器 / overlap-add 卷积 / 线性插值重采样
//
// 本文件补全 P2 neon「音频 DSP 核」专题（按 L4 工业扩展层落地）。音频 DSP 是 SIMD
// （NEON/SSE/AVX）最典型的应用场域之一：滤波、动态处理、混响、重采样每一条都是
// 高度规则的数据并行，适合向量化。本沙箱无音频硬件，用 x86 标量对照验证算法语义
// （即"仅 x86 标量对照"），真上 NEON 时把这些热点循环换成 intrinsics 即可。
//
// 编译（x86 本机跑算法语义对拍）：
//   g++ -O3 -std=c++17 neon_l4_audio_dsp.cpp -o audio && ./audio
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

static const float PI = 3.14159265358979323846f;

// -----------------------------------------------------------------------------
// **一、RBJ 双二阶系数设计（Audio EQ Cookbook / a0 归一化）**
//   系数存为 b0,a1,a2（已除以 a0），差分方程：
//     y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2]
// -----------------------------------------------------------------------------
struct Biquad { float b0, b1, b2, a1, a2; };

static void rbj_peaking(float fs, float f0, float Q, float gain_db, Biquad& bq) {
    float w0 = 2.f*PI*f0/fs, cw = std::cos(w0), sw = std::sin(w0);
    float A = std::pow(10.f, gain_db/40.f);        // RBJ 约定 A=10^(dB/40)
    float alpha = sw/(2.f*Q);
    bq.b0 = (1.f + alpha*A);
    bq.b1 = (-2.f*cw);
    bq.b2 = (1.f - alpha*A);
    bq.a1 = (-2.f*cw);
    bq.a2 = (1.f - alpha/A);
    float a0 = (1.f + alpha/A);
    bq.b0/=a0; bq.b1/=a0; bq.b2/=a0; bq.a1/=a0; bq.a2/=a0;
}
static void rbj_lowpass(float fs, float f0, float Q, Biquad& bq) {
    float w0 = 2.f*PI*f0/fs, cw = std::cos(w0), sw = std::sin(w0);
    float alpha = sw/(2.f*Q);
    bq.b0 = (1.f - cw)/2.f;
    bq.b1 = (1.f - cw);
    bq.b2 = (1.f - cw)/2.f;
    bq.a1 = (-2.f*cw);
    bq.a2 = (1.f - alpha);
    float a0 = (1.f + alpha);
    bq.b0/=a0; bq.b1/=a0; bq.b2/=a0; bq.a1/=a0; bq.a2/=a0;
}
static void rbj_highpass(float fs, float f0, float Q, Biquad& bq) {
    float w0 = 2.f*PI*f0/fs, cw = std::cos(w0), sw = std::sin(w0);
    float alpha = sw/(2.f*Q);
    bq.b0 = (1.f + cw)/2.f;
    bq.b1 = -(1.f + cw);
    bq.b2 = (1.f + cw)/2.f;
    bq.a1 = (-2.f*cw);
    bq.a2 = (1.f - alpha);
    float a0 = (1.f + alpha);
    bq.b0/=a0; bq.b1/=a0; bq.b2/=a0; bq.a1/=a0; bq.a2/=a0;
}
// 幅频响应 |H(e^{jw})|（w 为角频率，已 a0=1 归一化）
static float biquad_mag(const Biquad& b, float w) {
    float cw = std::cos(w), c2w = std::cos(2*w), sw = std::sin(w), s2w = std::sin(2*w);
    float nre = b.b0 + b.b1*cw + b.b2*c2w,  nim = -(b.b1*sw + b.b2*s2w);
    float dre = 1.f + b.a1*cw + b.a2*c2w,   dim = -(b.a1*sw + b.a2*s2w);
    return std::sqrt(nre*nre+nim*nim)/std::sqrt(dre*dre+dim*dim);
}

// -----------------------------------------------------------------------------
// **二、前馈压缩器**：包络跟随 + 增益计算 + 增益平滑
//   振幅域压缩：env>thr 时输出电平 = thr + (env-thr)/ratio
// -----------------------------------------------------------------------------
struct Comp { float thr, ratio, atk, rel, smooth, env, gain; };
static float compressor_process(Comp& C, float x) {
    float level = std::fabs(x);
    // 峰值包络：快 attack、慢 release
    if (level > C.env) C.env = C.atk*C.env + (1.f-C.atk)*level;
    else                C.env = C.rel*C.env + (1.f-C.rel)*level;
    // 增益计算（振幅域，绝不放大）
    float target = 1.f;
    if (C.env > C.thr) {
        float desired = C.thr + (C.env - C.thr)/C.ratio;
        target = desired / C.env;
    }
    if (target > 1.f) target = 1.f;
    // 增益平滑（一极点）
    C.gain = C.smooth*C.gain + (1.f-C.smooth)*target;
    return x * C.gain;
}

// -----------------------------------------------------------------------------
// **三、overlap-add 卷积混响**：分块直接卷积 + 尾部重叠相加
// -----------------------------------------------------------------------------
static void overlap_add(const float* x, int Nx, const float* h, int L, float* y, int ylen) {
    const int N = 64;                       // 块长
    for (int i = 0; i < ylen; ++i) y[i] = 0.f;
    for (int i = 0; i < Nx; i += N) {
        int nblk = (i+N <= Nx) ? N : (Nx - i);
        for (int n = 0; n < nblk + L - 1; ++n) {
            float s = 0.f;
            for (int k = 0; k < L; ++k) {
                int xn = n - k;
                if (xn >= 0 && xn < nblk) s += h[k]*x[i + xn];
            }
            y[i + n] += s;
        }
    }
}

// -----------------------------------------------------------------------------
// **四、线性插值重采样**
// -----------------------------------------------------------------------------
static void resample_linear(const float* x, int Nx, float r, float* y, int Ny) {
    for (int m = 0; m < Ny; ++m) {
        float t = (float)m / r;
        int i0 = (int)std::floor(t);
        float frac = t - (float)i0;
        float x0 = (i0 >= 0 && i0 < Nx) ? x[i0] : 0.f;
        float x1 = (i0+1 >= 0 && i0+1 < Nx) ? x[i0+1] : 0.f;
        y[m] = x0 + frac*(x1 - x0);
    }
}

// -----------------------------------------------------------------------------
// **五、chk 框架 + 5 项验证**
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

int main() {
    // ---- C1. RBJ peaking：DC 增益=1 + 中心频增益=A ----
    {
        Biquad bq; rbj_peaking(48000.f, 1000.f, 1.0f, 6.0f, bq);
        float dc = biquad_mag(bq, 0.f);                 // ω=0
        float w0 = 2.f*PI*1000.f/48000.f;
        float center = biquad_mag(bq, w0);
        float boost = std::pow(10.f, 6.0f/20.f);        // 中心频实际提升 = gain_db dB = 10^(dB/20) = A²
        double e = std::fmax(std::fabs(dc - 1.f), std::fabs(center - boost));
        chk("RBJ peaking: DC增益=1 且 中心频增益=10^(dB/20)(=gain_db dB)", e, 1e-4);
    }

    // ---- C2. 前馈压缩器：稳态输出幅度 = thr+(level-thr)/ratio ----
    {
        Comp C; C.thr=0.3f; C.ratio=4.f; C.atk=0.7f; C.rel=0.95f; C.smooth=0.8f; C.env=0; C.gain=1.f;
        const int n=200; float expect = C.thr + (1.0f-C.thr)/C.ratio;  // = 0.475
        float ylast=0; for (int i=0;i<n;++i) ylast = compressor_process(C, 1.0f);
        double e = std::fmax(std::fabs(ylast - expect), std::fabs(C.gain - 1.0f) > 0 ? 0.0 : 0.0);
        e = std::fabs(ylast - expect);                  // 主判据：稳态幅度
        double e2 = (C.gain < 1.f) ? 0.0 : 1.0;          // 次判据：确有压缩(gain<1)
        chk("前馈压缩器稳态幅度=thr+(lv-thr)/ratio 且 gain<1", std::fmax(e, e2), 1e-2);
    }

    // ---- C3. overlap-add == 直接全卷积 ----
    {
        const int Nx=128, L=32, ylen=Nx+L-1;
        float x[128], h[32], y[159], yref[159];
        for (int i=0;i<Nx;++i) x[i] = 0.5f*std::sin(0.3f*i) + 0.1f*i/Nx;
        for (int k=0;k<L;++k) h[k] = std::exp(-0.1f*k)*std::cos(0.5f*k);
        overlap_add(x, Nx, h, L, y, ylen);
        for (int n=0;n<ylen;++n){ float s=0; for(int k=0;k<L;++k){int xn=n-k; if(xn>=0&&xn<Nx) s+=h[k]*x[xn];} yref[n]=s; }
        chk("overlap-add 卷积 == 直接全卷积", max_diff(y,yref,ylen), 1e-5);
    }

    // ---- C4. 线性插值重采样：对线性斜坡精确（误差≈0）----
    {
        const int Nx=30; float x[32];
        for (int n=0;n<32;++n) x[n] = 0.1f*(float)n;     // 斜坡（只用前 30 个）
        const int Ny=60; float y[60];
        resample_linear(x, Nx, 2.0f, y, Ny);
        double e=0;
        for (int m=0;m<40;++m){ float t=(float)m/2.0f; float exp=0.1f*t; e=std::fmax(e, std::fabs(y[m]-exp)); }
        chk("线性插值重采样(2x) 对线性斜坡精确", e, 1e-5);
    }

    // ---- C5. RBJ lowpass Nyquist增益≈0 + highpass DC增益≈0 ----
    {
        Biquad lp, hp;
        rbj_lowpass(48000.f, 1000.f, 0.707f, lp);
        rbj_highpass(48000.f, 1000.f, 0.707f, hp);
        float lpNyq = biquad_mag(lp, PI);                // ω=π
        float hpDC  = biquad_mag(hp, 0.f);               // ω=0
        double e = std::fmax(std::fabs(lpNyq - 0.f), std::fabs(hpDC - 0.f));
        chk("RBJ lowpass@Nyquist≈0 且 highpass@DC≈0", e, 1e-4);
    }

    std::printf("\nbackend: portable audio DSP core (x86 标量对照)\n");
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
