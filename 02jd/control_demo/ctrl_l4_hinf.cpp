// ============================================================
// control_demo L4：H∞ / μ 综合 ⭐
// 纯 C++17。把"鲁棒控制"两条主线落到可运行：
//   · Part A  H∞ 范数(频域扫描) + 小增益定理：‖M‖∞ 是鲁棒稳定的度量，
//     若 ‖M‖∞·‖Δ‖∞<1 则 M 与任意稳定 Δ 的反馈互联必稳定。
//   · Part B1 混合灵敏度：把设计目标写成 ‖W1·S‖∞、‖W2·T‖∞ 的加权峰值。
//   · Part B2 结构化奇异值 μ：对"分块结构不确定性"给出比 ‖M‖2 更紧的
//     鲁棒界 μ̲≤μ≤μ̄≤‖M‖2（μ-综合比 H∞ 范数更少保守）。
// 对象：Part A 用 SISO 广义对象 M(s)=1/(s²+0.2s+1)；Part B1 用 P=1/(s+1),K=2 回路；
//       Part B2 的 M 由固定种子 LCG 生成（与 Python 验证逐位一致）。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

typedef vector<vector<double>> Mat;
typedef vector<double> Vec;

// ---- 固定种子 LCG + uniform（与受管 Python 验证逐位一致）----
static unsigned long long _seed = 12345;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(((_seed >> 33) & 0x7FFFFFFF)) / (double)0x80000000;
}

// ---- 复数 (轻量，仅本 demo 需要) ----
struct Cx { double re, im; };
static Cx cxAdd(const Cx&a,const Cx&b){ return {a.re+b.re, a.im+b.im}; }
static Cx cxSub(const Cx&a,const Cx&b){ return {a.re-b.re, a.im-b.im}; }
static Cx cxMul(const Cx&a,const Cx&b){ return {a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re}; }
static Cx cxDiv(const Cx&a,const Cx&b){ double d=b.re*b.re+b.im*b.im; return {(a.re*b.re+a.im*b.im)/d, (a.im*b.re-a.re*b.im)/d}; }
static double cxAbs(const Cx&a){ return sqrt(a.re*a.re+a.im*a.im); }
static Cx cxSqrt(const Cx&a){ double r=cxAbs(a); double phi=atan2(a.im,a.re); return {sqrt(r)*cos(phi/2.0), sqrt(r)*sin(phi/2.0)}; }

// ---- 实 2x2 辅助 ----
static double sigmaMax2(const Mat&A){
    double a=A[0][0],b=A[0][1],c=A[1][0],d=A[1][1];
    double e00=a*a+c*c, e01=a*b+c*d, e11=b*b+d*d;
    double tr=e00+e11, det=e00*e11-e01*e01;
    double lam=(tr+sqrt(fmax(0.0,tr*tr-4.0*det)))/2.0;
    return sqrt(fmax(0.0,lam));
}
static double rhoCx2(const Cx A[2][2]){
    Cx tr=cxAdd(A[0][0],A[1][1]);
    Cx det=cxSub(cxMul(A[0][0],A[1][1]), cxMul(A[0][1],A[1][0]));
    Cx disc=cxSqrt(cxSub(cxMul(tr,tr), {4.0*det.re,4.0*det.im}));
    Cx l1=cxDiv(cxAdd(tr,disc),{2.0,0.0});
    Cx l2=cxDiv(cxSub(tr,disc),{2.0,0.0});
    return fmax(cxAbs(l1),cxAbs(l2));
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-42s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    auto chkBool = [&](const char* name, bool cond) {
        printf("  %-42s  %s\n", name, cond ? "PASS" : "FAIL");
        cond ? ++pass : ++fail;
    };
    printf("=== L4：H∞ / μ 综合（范数 + 小增益 + 混合灵敏度 + 结构化奇异值）===\n");

    // ===== Part A: SISO 广义对象 M(s)=1/(s^2+0.2 s+1) =====
    printf("=== Part A: H∞ 范数 + 小增益定理 ===\n");
    auto Mof = [](double w)->Cx {                 // M(jw)=1/((1-w^2)+j0.2w)
        double den2 = (1.0-w*w)*(1.0-w*w) + 0.04*w*w;
        return {(1.0-w*w)/den2, (-0.2*w)/den2};
    };
    double wp = sqrt(0.98);
    double anPeak = 1.0/sqrt((1.0-wp*wp)*(1.0-wp*wp) + 0.04*wp*wp);
    double Mpeak = 0.0, wpeak = 0.0;
    for (int i = 0; i <= 40000; ++i) {
        double w = 0.5 + 1.0*(i/40000.0);
        double v = cxAbs(Mof(w));
        if (v > Mpeak) { Mpeak = v; wpeak = w; }
    }
    printf("  [demo] ‖M‖∞ 扫描峰值 = %.5f @ ω=%.4f  (解析=%.5f)\n", Mpeak, wpeak, anPeak);
    chk("‖M‖∞ 扫描 == 解析峰值", fabs(Mpeak - anPeak), 1e-3);
    chk("峰值频率 ≈ 0.9899", fabs(wpeak - wp), 0.01);

    // 小增益：闭环 (I-δM)^{-1}M 的 H∞ 范数 ≤ ‖M‖∞/(1-|δ|‖M‖∞)
    auto clNorm = [&](double delta)->double {
        double m = 0.0;
        for (int i = 0; i <= 40000; ++i) {
            double w = 0.5 + 1.0*(i/40000.0);
            Cx M = Mof(w);
            Cx den = cxSub({1.0,0.0}, {delta*M.re, delta*M.im});
            double dv = cxAbs(den);
            double val = dv > 1e-9 ? cxAbs(M)/dv : 1e9;
            if (val > m) m = val;
        }
        return m;
    };
    double delta = 0.10;
    double bound = anPeak / (1.0 - fabs(delta)*anPeak);
    double actual = clNorm(delta);
    printf("  [demo] δ=%.2f: 闭环 ‖·‖∞ 实测=%.4f, 小增益界=%.4f\n", delta, actual, bound);
    chkBool("小增益: 闭环范数 < 界", actual < bound);
    chkBool("小增益: 界有限(|δ|γ<1)", fabs(delta)*anPeak < 1.0 && bound < 1e6);

    // ===== Part B1: 混合灵敏度 SISO 回路 P=1/(s+1), K=2 =====
    printf("=== Part B1: 混合灵敏度加权峰值 ===\n");
    // S=(s+1)/(s+3), T=2/(s+3); W1=10/(s+1) 低频, W2=0.5 s/(s+10) 高频
    // |W1·S| = 10/sqrt(9+w^2);  |W2·T| = w/(sqrt(100+w^2)·sqrt(9+w^2))
    auto peak = [](double (*f)(double), double lo, double hi, int n)->double {
        double m = 0.0;
        for (int i = 0; i <= n; ++i) { double w = lo + (hi-lo)*(i/(double)n); double v=f(w); if (v>m) m=v; }
        return m;
    };
    double pW1S = peak([](double w){ return 10.0/sqrt(9.0+w*w); }, 0.0, 30.0, 20000);
    double pW2T = peak([](double w){ return w/(sqrt(100.0+w*w)*sqrt(9.0+w*w)); }, 0.0, 200.0, 20000);
    printf("  [demo] ‖W1·S‖∞ = %.4f   ‖W2·T‖∞ = %.4f\n", pW1S, pW2T);
    chkBool("‖W1·S‖∞ < 4 (低频性能达标)", pW1S < 4.0);
    chkBool("‖W2·T‖∞ < 1 (高频噪声达标)", pW2T < 1.0);
    chk("W1·S 在直流≈10/3", fabs(pW1S - 10.0/3.0), 0.02);

    // ===== Part B2: 结构化奇异值 μ =====
    printf("=== Part B2: 结构化奇异值 μ (下界/上界) ===\n");
    Mat M = {{2.0*urand()-1.0, 2.0*urand()-1.0}, {2.0*urand()-1.0, 2.0*urand()-1.0}};
    double m00=M[0][0], m01=M[0][1], m10=M[1][0], m11=M[1][1];
    printf("  [demo] M = [[%.4f, %.4f],[%.4f, %.4f]]\n", m00,m01,m10,m11);
    double nM2 = sigmaMax2(M);
    printf("  [demo] ‖M‖2 = %.4f\n", nM2);
    double bestUp = 1e18;
    for (int i = 0; i <= 2000; ++i) {
        double d = 0.05 * pow(10.0, 3.0*i/2000.0);
        Mat Mtil = {{m00, d*m01}, {m10/d, m11}};
        double s = sigmaMax2(Mtil);
        if (s < bestUp) bestUp = s;
    }
    printf("  [demo] μ̄ (上界) = %.4f\n", bestUp);
    chkBool("μ̄ ≤ ‖M‖2 (结构比无结构更紧)", bestUp <= nM2 + 1e-9);
    const int Ng = 48;
    double bestLo = 0.0;
    for (int i = 0; i < Ng; ++i) for (int j = 0; j < Ng; ++j) {
        double t1 = 2.0*M_PI*i/Ng, t2 = 2.0*M_PI*j/Ng;
        Cx d1 = {cos(t1), sin(t1)}, d2 = {cos(t2), sin(t2)};
        Cx Nm[2][2] = { {cxMul({m00,0.0},d1), cxMul({m01,0.0},d2)},
                        {cxMul({m10,0.0},d1), cxMul({m11,0.0},d2)} };
        double r = rhoCx2(Nm);
        if (r > bestLo) bestLo = r;
    }
    printf("  [demo] μ̲ (下界) = %.4f\n", bestLo);
    chkBool("μ̲ ≤ μ̄ (下界≤上界, 一致性)", bestLo <= bestUp + 1e-6);
    chkBool("μ̲ < ‖M‖2 (结构带来收益)", bestLo < nM2 - 1e-6);

    printf("  [结论] H∞ 范数度量最坏情况增益，小增益定理给出稳定充分条件；\n");
    printf("           μ-综合用分块结构把鲁棒界收紧到 μ̲≈μ̄<‖M‖2，比 H∞ 范数更少保守。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
