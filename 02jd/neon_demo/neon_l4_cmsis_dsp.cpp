// =============================================================================
// **CMSIS-DSP（L4 工业扩展）：库 API 语义可移植复刻**
//
// 本文件补全 P2 neon「CMSIS-DSP」专题（按 L4 工业扩展层落地）。CMSIS-DSP 是 ARM
// Cortex-M 的官方 DSP 库；本沙箱无 ARM、也无该库，因此**可移植复刻其 API 语义**
// （mat_mult / FIR / biquad cascade DF1 / RFFT / PID），对每个函数对拍一个独立
// 参考实现（直接卷积 / 差分方程 / DFT / 手算 PI）。
//
// 编译（x86 本机跑「语义复刻 + 标量参考」对拍，即"仅 x86 标量对照"）：
//   g++ -O3 -std=c++17 neon_l4_cmsis_dsp.cpp -o cmsis && ./cmsis
// Cortex-M 真跑：链入 CMSIS-DSP 库（arm_mat_mult_f32 / arm_fir_f32 / ...）替换本文件实现。
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>

static const float PI = 3.14159265358979323846f;

// -----------------------------------------------------------------------------
// **一、arm_mat_mult_f32 语义**：C = A·B（行主序，后乘）
// -----------------------------------------------------------------------------
static void arm_mat_mult_f32(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = s;
        }
}

// -----------------------------------------------------------------------------
// **二、arm_fir_f32 语义**：流式 FIR（延迟线实现，零初史）
//   状态 state[] 长度 = numTaps，shift 寄存器：state[0]=x[n], state[1]=x[n-1]...
// -----------------------------------------------------------------------------
static void arm_fir_f32(const float* coeffs, int numTaps, float* state,
                        const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = numTaps - 1; k > 0; --k) state[k] = state[k - 1];
        state[0] = in[i];
        float y = 0.f;
        for (int k = 0; k < numTaps; ++k) y += coeffs[k] * state[k];
        out[i] = y;
    }
}

// -----------------------------------------------------------------------------
// **三、arm_biquad_cascade_df1_f32 语义**：S 级二阶节级联（Direct Form I）
//   每节系数 [b0,b1,b2,a1,a2]（a0=1）；每节状态 [x1,x2,y1,y2]。
// -----------------------------------------------------------------------------
static void biquad_section(const float c[5], float s[4], float x, float& y) {
    float xn1 = s[0], xn2 = s[1], yn1 = s[2], yn2 = s[3];
    y = c[0]*x + c[1]*xn1 + c[2]*xn2 - c[3]*yn1 - c[4]*yn2;
    s[0] = x; s[1] = xn1; s[2] = y; s[3] = yn1;
}
static void arm_biquad_cascade_df1_f32(const float* coeffs, int S, float* state,
                                       const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        for (int sec = 0; sec < S; ++sec) {
            float y; biquad_section(&coeffs[5*sec], &state[4*sec], x, y); x = y;
        }
        out[i] = x;
    }
}

// -----------------------------------------------------------------------------
// **四、arm_rfft_fast_f32 语义**：N 点实 FFT（N=8，基-2 DIT + CMSIS 打包）
//   输出 8 个 float：out=[Re0, Re(N/2), Re1,Im1, Re2,Im2, Re3,Im3]
// -----------------------------------------------------------------------------
static void cfft8(float* re, float* im) {
    int rev[8] = {0,4,2,6,1,5,3,7};                 // 位反转（N=8）
    float tr[8], ti[8];
    for (int i = 0; i < 8; ++i) { tr[i] = re[rev[i]]; ti[i] = im[rev[i]]; }
    for (int i = 0; i < 8; ++i) { re[i] = tr[i]; im[i] = ti[i]; }
    for (int s = 1; s <= 3; ++s) {                  // 3 级蝶形
        int m = 1 << s, mh = m >> 1;
        for (int k = 0; k < 8; k += m)
            for (int j = 0; j < mh; ++j) {
                float ang = -2.f*PI*(float)j/(float)m;
                float wr = std::cos(ang), wi = std::sin(ang);
                int t = k + j, b = k + j + mh;
                float xr = re[t] + wr*re[b] - wi*im[b];
                float xi = im[t] + wr*im[b] + wi*re[b];
                float yr = re[t] - wr*re[b] + wi*im[b];
                float yi = im[t] - wr*im[b] - wi*re[b];
                re[t] = xr; im[t] = xi; re[b] = yr; im[b] = yi;
            }
    }
}
static void arm_rfft_fast_f32(const float* in, float* out) {
    float re[8], im[8];
    for (int i = 0; i < 8; ++i) { re[i] = in[i]; im[i] = 0.f; }
    cfft8(re, im);
    out[0] = re[0]; out[1] = re[4];                 // DC, Nyquist
    out[2] = re[1]; out[3] = im[1];
    out[4] = re[2]; out[5] = im[2];
    out[6] = re[3]; out[7] = im[3];
}

// -----------------------------------------------------------------------------
// **五、arm_pid_f32 语义**：A0/A1/A2 形式，Kd=0 退化为 PI
//   A0=Kp+Ki+Kd, A1=-(Kp+2Kd), A2=Kd； y=A0·e + A1·e1 + A2·e2（e1/e2 为前两次误差）
// -----------------------------------------------------------------------------
struct Pid { float Kp, Ki, Kd, A0, A1, A2, y1, e1, e2; };  // y1=y[n-1], e1=e[n-1], e2=e[n-2]
static float arm_pid_f32(Pid& S, float e) {
    // CMSIS 真义：y[n] = A0·e[n] + A1·e[n-1] + A2·e[n-2] + y[n-1]
    //   （含前次输出，构成积分递推；Kd=0 时退化为增量式 PI）
    float y = S.A0*e + S.A1*S.e1 + S.A2*S.e2 + S.y1;
    S.e2 = S.e1; S.e1 = e; S.y1 = y;
    return y;
}

// -----------------------------------------------------------------------------
// **六、chk 框架 + 验证**
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
    // ---- C1. mat_mult vs 朴素 ----
    {
        const int M=3,N=3,K=3;
        float A[9]={1,2,3, 4,5,6, 7,8,9};
        float B[9]={9,8,7, 6,5,4, 3,2,1};
        float C[9], R[9];
        arm_mat_mult_f32(A,B,C,M,N,K);
        for(int i=0;i<M;i++)for(int j=0;j<N;j++){float s=0;for(int k=0;k<K;k++)s+=A[i*K+k]*B[k*N+j];R[i*N+j]=s;}
        chk("arm_mat_mult_f32 == 朴素 A·B", max_diff(C,R,9), 1e-5);
    }

    // ---- C2. FIR vs 直接卷积 ----
    {
        const int NT=5;                       // 抽头数
        const int n=8;
        float h[5]={0.2f,0.5f,0.3f,-0.1f,0.05f};
        float x[8]={1,2,3,4,5,6,7,8};
        float y[8], ref[8]; float state[5]={0,0,0,0,0};
        arm_fir_f32(h, NT, state, x, y, n);
        for(int i=0;i<n;i++){ float s=0; for(int k=0;k<NT;k++){int idx=i-k; if(idx>=0) s+=h[k]*x[idx];} ref[i]=s; }
        chk("arm_fir_f32 == 直接卷积", max_diff(y,ref,8), 1e-6);
    }

    // ---- C3. biquad cascade DF1 vs 直接差分方程 ----
    {
        // 单节低通原型系数（b0,b1,b2,a1,a2），a0=1
        float c[5]={0.1f,0.2f,0.1f, -0.5f,-0.2f};
        float st[4]={0,0,0,0};
        const int n=10;
        float x[10]={1,0.5f,-0.3f,0.8f,-0.2f,0.4f,0.1f,-0.6f,0.3f,0.2f};
        float y[10], ref[10];
        arm_biquad_cascade_df1_f32(c,1,st,x,y,n);
        float x1=0,x2=0,y1=0,y2=0;
        for(int i=0;i<n;i++){
            float yy=c[0]*x[i]+c[1]*x1+c[2]*x2 - c[3]*y1 - c[4]*y2;
            x2=x1;x1=x[i];y2=y1;y1=yy; ref[i]=yy;
        }
        chk("arm_biquad_cascade_df1_f32 == 直接 DF1 差分方程", max_diff(y,ref,10), 1e-6);
    }

    // ---- C4. RFFT vs DFT（N=8）----
    {
        float x[8]={0.5f,1.0f,-0.5f,0.3f,0.8f,-0.2f,0.4f,0.6f};
        float out[8]; arm_rfft_fast_f32(x, out);
        // 朴素 DFT（double 参考）
        double Re[8],Im[8];
        for(int k=0;k<8;k++){double re=0,im=0; for(int nn=0;nn<8;nn++){double a=2*PI*k*nn/8; re+=x[nn]*std::cos(a); im-=x[nn]*std::sin(a);} Re[k]=re;Im[k]=im;}
        double e=0;
        e=std::fmax(e, std::fabs(out[0]-Re[0]));          // DC
        e=std::fmax(e, std::fabs(out[1]-Re[4]));          // Nyquist
        e=std::fmax(e, std::fabs(out[2]-Re[1]));          // bin1 real
        e=std::fmax(e, std::fabs(out[3]-Im[1]));          // bin1 imag
        e=std::fmax(e, std::fabs(out[4]-Re[2]));          // bin2 real
        chk("arm_rfft_fast_f32 == 朴素 DFT（DC/Nyq/bin1/bin2）", e, 1e-4);
    }

    // ---- C5. PID（Kd=0，PI）vs 手算 PI 累加 ----
    {
        Pid S; S.Kp=1.0f; S.Ki=0.2f; S.Kd=0.0f;
        S.A0=S.Kp+S.Ki+S.Kd; S.A1=-(S.Kp+2*S.Kd); S.A2=S.Kd; S.y1=0; S.e1=0; S.e2=0;
        const int n=6;
        float e[6]={1,1,1,2,2,2};
        float y[6], ref[6];
        float integ=0;
        for(int i=0;i<n;i++){
            integ+=e[i];
            ref[i]=S.Kp*e[i]+S.Ki*integ;     // 手算 PI
            y[i]=arm_pid_f32(S, e[i]);
        }
        chk("arm_pid_f32(Kd=0) == 手算 PI 累加", max_diff(y,ref,6), 1e-6);
    }

    std::printf("\nbackend: %s\n",
#if defined(ARM_CMSIS_DSP_LINKED)
                "CMSIS-DSP library (Cortex-M)");
#else
                "portable semantic re-impl (x86 标量对照)");
#endif
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
