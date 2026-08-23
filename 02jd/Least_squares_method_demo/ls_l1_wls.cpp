// ============================================================
// LS_demo L1.2 加权最小二乘 WLS 与高斯-马尔可夫定理
// 纯 C++17。W = R⁻¹: min ‖W^{1/2}(Ax−b)‖²
// 高斯-马尔可夫: 噪声零均值等方差不相关时 OLS 是 BLUE;
// 异方差时 WLS 才是最优(方差小的数据权重大)。
// 验证: 异方差数据 50 次蒙特卡洛,WLS 平均误差 < OLS
// 协方差公式: Cov(x̂) = (AᵀWA)⁻¹σ² —— 一并打印
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static unsigned long long _seed = 0;
static void set_seed(unsigned long long s){ _seed = s; }
static double urand(){ _seed = _seed*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L1.2 加权最小二乘与高斯-马尔可夫 ===\n");

    int n = 60;
    double a_true = 1.5, b_true = 0.5;
    // WLS 公式(2x2 手写): Sw,Swx,Swxx,Swxy
    double Sw=0,Swx=0,Swxx=0;
    for (int i=0;i<n;++i){ double x=i*0.1, sig=0.1+0.02*x, w=1.0/(sig*sig);
        Sw+=w; Swx+=w*x; Swxx+=w*x*x; }
    double det = Sw*Swxx - Swx*Swx;
    // 蒙特卡洛对比
    double e_ols=0, e_wls=0; int M = 50;
    for (int m=0;m<M;++m) {
        set_seed(200+m);
        double Swy=0,Swxy=0, Sx=0,Sy=0,Sxx=0,Sxy=0;
        for (int i=0;i<n;++i) {
            double x=i*0.1, sig=0.1+0.02*x;
            double y=a_true*x+b_true+gauss()*sig;
            double w=1.0/(sig*sig);
            Swy+=w*y; Swxy+=w*x*y;
            Sx+=x; Sy+=y; Sxx+=x*x; Sxy+=x*y;
        }
        double a_w=(Sw*Swxy-Swx*Swy)/det;
        double det2=n*Sxx-Sx*Sx;
        double a_o=(n*Sxy-Sx*Sy)/det2;
        e_wls += fabs(a_w-a_true); e_ols += fabs(a_o-a_true);
    }
    e_ols/=M; e_wls/=M;
    printf("  [demo] 平均|Δa|: OLS=%.4f → WLS=%.4f (异方差 σ(x)=0.1+0.02x)\n", e_ols, e_wls);
    chk("WLS 比 OLS 更准(异方差)", e_wls < e_ols ? 0.0 : 1.0, 0.5);
    // 协方差: Var(a) = Swxx/det (σ 单位权)
    printf("  [demo] 理论 std(a_wls) = %.4f (Cov=(AᵀWA)⁻¹)\n", sqrt(Swxx/det));
    printf("  [结论] 噪声不等方差时按 1/σ² 加权=让准的数据多说话;\n");
    printf("        KF 的 R⁻¹、BA 的信息矩阵都是这一族。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
