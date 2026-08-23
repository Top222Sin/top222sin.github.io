// ============================================================
// LS_demo L2.4 递推最小二乘 RLS(= q→0 的 KF!)
// 纯 C++17。θₖ = θₖ₋₁ + K(y − hᵀθ),  K = P⁻h/(hᵀP⁻h+R)
// 验证: ① RLS 与批量 LS 逐位一致(7e-9);
//      ② 遗忘因子 λ=0.9 在参数突变后跟踪误差 0.4 vs λ=1 的 1.4
// 衔接 KF_demo L1.1: RLS+过程噪声(Q>0) 就是 KF —— 一家人
// ⚠ 坑(开发踩过): R 必须与真实 σ² 一致且回归量要有界(回路增益<1)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 0;
static void set_seed(unsigned long long s){ _seed = s; }
static double urand(){ _seed = _seed*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

static double track(double lam) {
    set_seed(909);
    int n = 60;
    const double Rv = 0.01;      // 与真实 σ² 一致!
    double p00 = 1e6, p01 = 0, p11 = 1e6;
    double t0 = 0, t1 = 0;
    double err_sum = 0; int cnt = 0;
    for (int i = 0; i < n; ++i) {
        double x = (i % 10) * 0.1;               // 有界重复激励
        double a_t = (i < 30) ? 1.0 : 3.0;        // 30 步后突变
        double y = a_t*x + 0.5 + gauss()*0.1;
        // P⁻ = P/λ
        double m00 = p00/lam, m01 = p01/lam, m11 = p11/lam;
        double ph0 = m00*x + m01, ph1 = m01*x + m11;
        double S = x*ph0 + ph1 + Rv;
        double k0 = ph0/S, k1 = ph1/S;
        double e = y - (x*t0 + t1);
        t0 += k0*e; t1 += k1*e;
        // Joseph 更新
        double i00 = 1 - k0*x, i01 = -k0, i11 = 1 - k1;
        double q00 = i00*m00 + i01*m01, q01 = i00*m01 + i01*m11;
        double q10 = i01*m00 + i11*m01, q11 = i01*m01 + i11*m11;
        p00 = q00*i00 + q01*i01 + k0*Rv*k0;
        p01 = q00*i01 + q01*i11 + k0*Rv*k1;
        p11 = q10*i01 + q11*i11 + k1*Rv*k1;
        if (i >= 40) { err_sum += fabs(t0 - a_t); ++cnt; }
    }
    return err_sum/cnt;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L2.4 递推最小二乘 RLS ===\n");

    // ① RLS = 批量 LS
    set_seed(9);
    int n = 60;
    double a_true = 1.5, b_true = 0.5;
    vector<double> X(n), Y(n);
    double Sx=0,Sy=0,Sxx=0,Sxy=0;
    for (int i=0;i<n;++i) {
        X[i]=i*0.1; Y[i]=a_true*X[i]+b_true+gauss()*0.2;
        Sx+=X[i]; Sy+=Y[i]; Sxx+=X[i]*X[i]; Sxy+=X[i]*Y[i];
    }
    double det = n*Sxx-Sx*Sx;
    double a_b = (n*Sxy-Sx*Sy)/det, b_b = (Sxx*Sy-Sx*Sxy)/det;
    // RLS
    double p00=1e6,p01=0,p11=1e6,t0=0,t1=0;
    const double Rv = 0.04;   // σ=0.2
    for (int i=0;i<n;++i) {
        double x=X[i], y=Y[i];
        double ph0=p00*x+p01, ph1=p01*x+p11;
        double S = x*ph0+ph1+Rv;
        double k0=ph0/S, k1=ph1/S;
        double e = y-(x*t0+t1);
        t0+=k0*e; t1+=k1*e;
        double i00=1-k0*x, i01=-k0, m10=-k1*x, i11=1-k1;
        double q00=i00*p00+i01*p01, q01=i00*p01+i01*p11;
        double q10=m10*p00+i11*p01, q11=m10*p01+i11*p11;
        p00=q00*i00+q01*i01+k0*Rv*k0;
        p01=q00*m10+q01*i11+k0*Rv*k1;
        p11=q10*m10+q11*i11+k1*Rv*k1;
    }
    chk("RLS = 批量 LS(逐位)", fmax(fabs(t0-a_b), fabs(t1-b_b)), 1e-8);
    printf("  [demo] RLS θ=[%.5f, %.5f] = 批量 [%.5f, %.5f]\n", t0, t1, a_b, b_b);
    // ② 遗忘因子
    double e09 = track(0.9), e10 = track(1.0);
    printf("  [demo] 参数突变后跟踪误差: λ=0.9 → %.3f vs λ=1.0 → %.3f\n", e09, e10);
    chk("遗忘因子 RLS 跟踪突变", e09 < 0.75*e10 ? 0.0 : 1.0, 0.5);
    printf("  [结论] RLS 来一个数据算一次,答案=攒齐了一起算(KF_demo L1.1 同款结论);\n");
    printf("        遗忘因子=指数降权旧数据,突变后 ~1/(1−λ) 步跟上。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
