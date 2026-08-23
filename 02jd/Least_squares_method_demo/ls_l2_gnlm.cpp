// ============================================================
// LS_demo L2.2 最速下降 vs 高斯-牛顿 vs LM
// 纯 C++17。三兄弟:
//   GD:  p ← p − t·Jᵀr         (稳但慢,锯齿)
//   GN:  p ← p − (JᵀJ)⁻¹Jᵀr    (快但远点发散)
//   LM:  p ← p − (JᵀJ+λI)⁻¹Jᵀr (两者插值,λ 自适应)
// 验证: 远起点下 LM 收敛、不劣于 GN(靠 λ 在 GD/GN 间切换)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 7;
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
    printf("=== L2.2 GD vs GN vs LM ===\n");

    int n = 30;
    double a_true = 3.0, b_true = -1.2;
    vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        xs[i] = i*0.1;
        ys[i] = a_true*exp(b_true*xs[i]) + gauss()*0.05;
    }
    auto costF = [&](double a, double b) {
        double s = 0;
        for (int i = 0; i < n; ++i) { double r = ys[i]-a*exp(b*xs[i]); s += 0.5*r*r; }
        return s; };
    auto jacob = [&](double a, double b, double& H00, double& H01, double& H11,
                     double& g0, double& g1) {
        H00=H01=H11=g0=g1=0;
        for (int i = 0; i < n; ++i) {
            double e = exp(b*xs[i]);
            double j0 = -e, j1 = -a*xs[i]*e;
            double r = ys[i] - a*e;
            H00 += j0*j0; H01 += j0*j1; H11 += j1*j1;
            g0 += j0*r; g1 += j1*r;
        } };
    double aF = 0.3, bF = 0.5;   // 远起点

    // ① 最速下降(回溯步长)
    double a = aF, b = bF;
    for (int it = 0; it < 3000; ++it) {
        double H00,H01,H11,g0,g1;
        jacob(a,b,H00,H01,H11,g0,g1);
        double gn2 = sqrt(g0*g0+g1*g1);
        if (gn2 < 1e-10) break;
        double t = 0.1;
        while (costF(a-t*g0, b-t*g1) > costF(a,b) && t > 1e-8) t *= 0.5;
        a -= t*g0; b -= t*g1;
    }
    double e_gd = fmax(fabs(a-a_true), fabs(b-b_true));
    // ② GN(代价不降即停)
    a = aF; b = bF;
    for (int it = 0; it < 100; ++it) {
        double H00,H01,H11,g0,g1;
        jacob(a,b,H00,H01,H11,g0,g1);
        double det = H00*H11-H01*H01;
        double d0 = -( H11*g0-H01*g1)/det, d1 = -(-H01*g0+H00*g1)/det;
        if (costF(a+d0, b+d1) > costF(a,b)) break;
        a += d0; b += d1;
        if (fmax(fabs(d0),fabs(d1)) < 1e-12) break;
    }
    double e_gn = fmax(fabs(a-a_true), fabs(b-b_true));
    // ③ LM
    a = aF; b = bF;
    double lam = 1e-3;
    for (int it = 0; it < 200; ++it) {
        double H00,H01,H11,g0,g1;
        jacob(a,b,H00,H01,H11,g0,g1);
        bool ok = false;
        double d0=0, d1=0;
        for (int tr = 0; tr < 20; ++tr) {
            double h00 = H00+lam, h11 = H11+lam;
            double det = h00*h11-H01*H01;
            d0 = -( h11*g0-H01*g1)/det;
            d1 = -(-H01*g0+h00*g1)/det;
            if (costF(a+d0, b+d1) < costF(a,b)) { lam = fmax(lam*0.3, 1e-12); ok = true; break; }
            lam *= 10;
        }
        if (!ok) break;
        a += d0; b += d1;
        if (fmax(fabs(d0),fabs(d1)) < 1e-12) break;
    }
    double e_lm = fmax(fabs(a-a_true), fabs(b-b_true));
    printf("  [demo] 远起点(0.3, 0.5): GD err=%.2e | GN err=%.2e | LM err=%.2e\n",
           e_gd, e_gn, e_lm);
    chk("LM 远起点收敛", e_lm, 0.05);
    chk("LM 不劣于 GN(远起点)", e_lm <= e_gn + 1e-9 ? 0.0 : 1.0, 0.5);
    printf("  [结论] λ大→像GD(稳) λ小→像GN(快) 每步按代价是否下降调λ;\n");
    printf("        LM=Ceres 默认,MPMC 迭代里做对比的基线。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
