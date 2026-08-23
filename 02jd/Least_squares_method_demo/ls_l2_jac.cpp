// ============================================================
// LS_demo L2.1 残差雅可比:梯度与海塞的 GN 视角
// 纯 C++17。F(p)=½‖r‖²:  ∇F=Jᵀr   H=JᵀJ+Σrᵢ∇²rᵢ≈JᵀJ(GN 忽略第二项)
// 验证: ① ∇½‖r‖² = Jᵀr vs FD;② 近收敛点 H≈JᵀJ vs FD 海塞
// (残差小 → 忽略项小 → 近收敛点近似成立)
// 衔接 Jacobian_demo:J 就是雅可比,本篇是它在优化里的角色
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 6;
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
    printf("=== L2.1 雅可比:∇=Jᵀr 与 H≈JᵀJ ===\n");

    int n = 20;
    vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        xs[i] = i*0.15;
        ys[i] = 1.2*sin(0.8*xs[i]) + 0.3 + gauss()*0.05;
    }
    // 模型 y = a·sin(b·x) + c
    double a0 = 1.2, b0 = 0.8, c0 = 0.3;   // 直接从近真值出发(近收敛点)
    auto costS = [&](double a, double b, double c) {
        double s = 0;
        for (int i = 0; i < n; ++i) { double r = ys[i]-(a*sin(b*xs[i])+c); s += r*r; }
        return s; };
    // ① 梯度 Jᵀr vs FD
    double g[3] = {0,0,0};
    for (int i = 0; i < n; ++i) {
        double r = ys[i] - (a0*sin(b0*xs[i])+c0);
        g[0] += -sin(b0*xs[i])*r;
        g[1] += -a0*xs[i]*cos(b0*xs[i])*r;
        g[2] += -1.0*r;
    }
    double h = 1e-6;
    double gerr = 0;
    double p[3] = {a0, b0, c0};
    for (int j = 0; j < 3; ++j) {
        double pp[3] = {a0,b0,c0}, pm[3] = {a0,b0,c0};
        pp[j] += h; pm[j] -= h;
        double fd = 0.5*(costS(pp[0],pp[1],pp[2])-costS(pm[0],pm[1],pm[2]))/(2*h);
        gerr = fmax(gerr, fabs(0.5*2*g[j] - fd));
    }
    chk("∇½‖r‖² = Jᵀr vs FD", gerr, 1e-6);
    // ② H≈JᵀJ vs FD 海塞(H[0][0])
    double H00 = 0;
    for (int i = 0; i < n; ++i) H00 += sin(b0*xs[i])*sin(b0*xs[i]);
    double h2 = 1e-5;
    double S0 = costS(a0,b0,c0);
    double Sp = costS(a0+h2,b0,c0), Sm = costS(a0-h2,b0,c0);
    double Hfd = 0.5*(Sp - 2*S0 + Sm)/(h2*h2);
    chk("H≈JᵀJ (GN 近似海塞)", fabs(H00-Hfd), 0.15*fabs(Hfd));
    printf("  [demo] H00: JᵀJ=%.2f vs FD=%.2f (近收敛点近似成立)\n", H00, Hfd);
    printf("  [结论] GN 海塞免二阶导:代价=线性收敛变差、远点不准。\n");
    printf("        Jacobian_demo 是 J 的百科全书,本篇只取'优化视角'。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
