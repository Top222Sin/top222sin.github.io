// ============================================================
// LS_demo L2.3 鲁棒估计:M-估计与 Huber IRLS
// 纯 C++17。野值下平方损失爆炸(误差平方放大) → 换 Huber:
//   ρ(e)= e²/2 (|e|≤δ) ;  δ|e|−δ²/2 (|e|>δ)
// IRLS: 权重 w=ρ'(e)/e → 加权 LS 迭代(收敛到 M-估计解)
// 验证: 20% 野值下 OLS 误差 0.72 → Huber 0.19
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 8;
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
    printf("=== L2.3 鲁棒核:Huber IRLS ===\n");

    int n = 50;
    double a_true = 2.0, b_true = 1.0;
    vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        xs[i] = i*0.1;
        double y = a_true*xs[i] + b_true + gauss()*0.2;
        if (urand() < 0.2) y += 5.0;   // 20% 野值
        ys[i] = y;
    }
    // OLS
    double Sx=0,Sy=0,Sxx=0,Sxy=0;
    for (int i=0;i<n;++i){ Sx+=xs[i]; Sy+=ys[i]; Sxx+=xs[i]*xs[i]; Sxy+=xs[i]*ys[i]; }
    double det = n*Sxx - Sx*Sx;
    double a_o = (n*Sxy-Sx*Sy)/det, b_o = (Sxx*Sy-Sx*Sxy)/det;
    // Huber IRLS
    double a = a_o, b = b_o, delta = 1.0;
    for (int it = 0; it < 30; ++it) {
        double Sw=0,Swx=0,Swxx=0,Swy=0,Swxy=0;
        for (int i = 0; i < n; ++i) {
            double r = ys[i] - (a*xs[i]+b);
            double w = (fabs(r) <= delta) ? 1.0 : delta/fmax(fabs(r),1e-12);
            Sw+=w; Swx+=w*xs[i]; Swxx+=w*xs[i]*xs[i]; Swy+=w*ys[i]; Swxy+=w*xs[i]*ys[i];
        }
        double dt = Sw*Swxx - Swx*Swx;
        double an = (Sw*Swxy-Swx*Swy)/dt, bn = (Swxx*Swy-Swx*Swxy)/dt;
        if (fmax(fabs(an-a),fabs(bn-b)) < 1e-12) { a=an; b=bn; break; }
        a=an; b=bn;
    }
    double e_ols = fmax(fabs(a_o-a_true), fabs(b_o-b_true));
    double e_hub = fmax(fabs(a-a_true), fabs(b-b_true));
    printf("  [demo] 20%% 野值(+5σ): OLS 误差=%.3f → Huber=%.3f\n", e_ols, e_hub);
    chk("Huber 显著抗野值", e_hub < 0.5*e_ols ? 0.0 : 1.0, 0.5);
    printf("  [结论] 平方损失=民主投票,野值话语权太大;Huber 给大残差'封顶'。\n");
    printf("        工程标配: Ceres LossFunction、BA 的 Cauchy 核、RANSAC 的硬剔除。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
