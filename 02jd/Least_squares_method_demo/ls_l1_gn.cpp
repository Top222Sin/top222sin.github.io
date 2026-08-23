// ============================================================
// LS_demo L1.5 非线性最小二乘:高斯-牛顿 (GN)
// 纯 C++17。min ½‖r(p)‖² → 线性化 r(p+Δ)≈r+JΔ → JΔ=−r
//   Δ = −(JᵀJ)⁻¹Jᵀr    ←  注意负号!(写成 + 就是在爬坡)
// 验证: ① 解析雅可比 vs 有限差分;② 指数拟合 GN 8 步收敛
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 5;
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
    printf("=== L1.5 非线性最小二乘:高斯-牛顿 ===\n");

    int n = 30;
    double a_true = 2.5, b_true = -0.8;
    vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        xs[i] = i*0.1;
        ys[i] = a_true*exp(b_true*xs[i]) + gauss()*0.05;
    }
    auto resid = [&](double p0, double p1) {
        vector<double> r(n);
        for (int i = 0; i < n; ++i) r[i] = ys[i] - p0*exp(p1*xs[i]);
        return r; };
    // ① 解析 vs FD 雅可比
    double p0 = 2.0, p1 = -0.5;
    double jerr = 0;
    double h = 1e-6;
    for (int j = 0; j < 2; ++j) {
        double pp0 = p0 + (j==0)*h, pp1 = p1 + (j==1)*h;
        double pm0 = p0 - (j==0)*h, pm1 = p1 - (j==1)*h;
        auto rp = resid(pp0, pp1), rm = resid(pm0, pm1);
        for (int i = 0; i < n; ++i) {
            double Jfd = (rp[i]-rm[i])/(2*h);
            double e = exp(p1*xs[i]);
            double Jan = (j==0) ? -e : -p0*xs[i]*e;
            jerr = fmax(jerr, fabs(Jfd-Jan));
        }
    }
    chk("指数模型雅可比 vs FD", jerr, 1e-6);
    // ② GN (2x2 法方程手写)
    double a = 1.5, b = -0.5;
    int it = 0;
    for (it = 0; it < 50; ++it) {
        // JᵀJ 与 Jᵀr
        double H00=0,H01=0,H11=0,g0=0,g1=0;
        for (int i = 0; i < n; ++i) {
            double e = exp(b*xs[i]);
            double j0 = -e, j1 = -a*xs[i]*e;
            double r = ys[i] - a*e;
            H00 += j0*j0; H01 += j0*j1; H11 += j1*j1;
            g0 += j0*r; g1 += j1*r;
        }
        double det = H00*H11 - H01*H01;
        double d0 = -( H11*g0 - H01*g1)/det;   // Δ=−(JᵀJ)⁻¹Jᵀr
        double d1 = -(-H01*g0 + H00*g1)/det;
        a += d0; b += d1;
        if (fmax(fabs(d0), fabs(d1)) < 1e-12) break;
    }
    chk("GN 收敛到真值", fmax(fabs(a-a_true), fabs(b-b_true)), 0.05);
    printf("  [demo] GN %d 步: a=%.4f b=%.4f (真值 2.5, -0.8)\n", it, a, b);
    printf("  [结论] GN=把非线性问题在当前点'假装线性',解一个线性 LS,\n");
    printf("        重复直到收敛——这就是 Ceres/g2o 的内核。LM/稳健版见 L2.2/L2.3。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
