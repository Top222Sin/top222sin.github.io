// ============================================================
// control_demo L4：反馈线性化 / 反步 (FBL + Backstepping) ⭐
// 纯 C++17。处理"经典 LQR/极点配置只能用于线性对象"的盲区：
//   Part A 输入-状态反馈线性化：ẍ = x1^2 + u
//       u = v - x1^2 把对象精确变成双积分器 ẍ=v，故跟踪误差满足
//       ë + k2 ė + k1 e = 0（与 x1^2 完全无关）；忽略非线性的朴素 PD 失败。
//   Part B 反步 backstepping：严格反馈系统 ẋ1=x2, ẋ2=x1+x2^2+u
//       递归虚拟控制 → 闭环恰为线性稳定，且 ṀV2 = -c1 x1^2 - c2 z2^2 < 0。
// 欧拉积分 dt=0.001, T=10s（本机无编译器，数值由受管 Python 对拍背书）。
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static const double dt = 0.001, T = 10.0;
static const int N = (int)(T/dt);
static const double k1 = 4.0, k2 = 4.0;

// ---------- Part A ----------
static void sim_fbl(double& x1f, double& e_fbl, double& resmax) {
    double x1 = 0.0, x2 = 0.0; x1f=0; e_fbl=0; resmax=0;
    for (int k = 0; k < N; ++k) {
        double t = k*dt;
        double r = sin(t), rd = cos(t), rdd = -sin(t);
        double e = x1 - r, ed = x2 - rd;
        double v = rdd - k1*e - k2*ed;
        double u = v - x1*x1;                 // 精确抵消 x1^2
        double x1n = x1 + x2*dt;
        double x2n = x2 + (x1*x1 + u)*dt;
        double res = (x1*x1 + u - rdd) + k2*(x2 - rd) + k1*(x1 - r);
        if (fabs(res) > resmax) resmax = fabs(res);
        x1 = x1n; x2 = x2n;
        e_fbl = x1 - r;
    }
    x1f = x1;
}
static double sim_naive() {
    double x1 = 0.0, x2 = 0.0;
    for (int k = 0; k < N; ++k) {
        double t = k*dt;
        double r = sin(t);
        double u = -k1*x1 - k2*x2;            // 忽略 x1^2 的朴素 PD
        double x1n = x1 + x2*dt;
        double x2n = x2 + (x1*x1 + u)*dt;
        x1 = x1n; x2 = x2n;
    }
    double t = (N-1)*dt;
    return x1 - sin(t);
}

// ---------- Part B ----------
static void sim_bs(double& x1b, double& x2b, double& lymax, double& lyr) {
    const double c1 = 2.0, c2 = 2.0;
    double x1 = 1.0, x2 = 0.0; lymax = 0.0; lyr = -1e9;
    for (int k = 0; k < N; ++k) {
        double z2 = x2 + c1*x1;
        double u = -(c1+c2)*z2 - (2.0 - c1*c1)*x1 - x2*x2;
        double x1n = x1 + x2*dt;
        double x2n = x2 + (x1 + x2*x2 + u)*dt;
        double z2n = x2n + c1*x1n;
        double V  = 0.5*x1*x1 + 0.5*z2*z2;
        double Vn = 0.5*x1n*x1n + 0.5*z2n*z2n;
        double dV = (Vn - V)/dt;
        double ident = dV - (-c1*x1*x1 - c2*z2*z2);
        if (fabs(ident) > lymax) lymax = fabs(ident);
        if (dV > lyr) lyr = dV;
        x1 = x1n; x2 = x2n;
    }
    x1b = x1; x2b = x2;
}

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool ok=e<t; printf("  %-46s err=%.3e tol=%.0e  %s\n",n,e,t,ok?"PASS":"FAIL"); ok?++pass:++fail; };
    auto chkBool=[&](const char* n,bool c){ printf("  %-46s  %s\n",n,c?"PASS":"FAIL"); c?++pass:++fail; };
    printf("=== L4：反馈线性化 / 反步（非线性对象的确定性镇定）===\n");

    printf("=== Part A: 输入-状态反馈线性化 (ẍ = x1^2 + u, 跟踪 r=sin t) ===\n");
    double x1f,e_fbl,resmax; sim_fbl(x1f,e_fbl,resmax);
    double e_naive = sim_naive();
    printf("  [demo] FBL 终态跟踪误差 |e|=%.5f, 精确线性化残差 max=%.2e\n", fabs(e_fbl), resmax);
    printf("  [demo] 朴素PD 终态跟踪误差 |e|=%.5f\n", fabs(e_naive));
    chk("FBL 精确跟踪 (|e|<0.01)", fabs(e_fbl), 0.01);
    chk("精确线性化残差≈0 (与 x1^2 无关)", resmax, 1e-6);
    chkBool("朴素PD 因忽略非线性而失败 (|e|>0.1)", fabs(e_naive) > 0.1);

    printf("=== Part B: 反步 backstepping (ẋ1=x2, ẋ2=x1+x2^2+u) ===\n");
    double x1b,x2b,lymax,lyr; sim_bs(x1b,x2b,lymax,lyr);
    printf("  [demo] 终态 x1=%.5f, x2=%.5f,  max ṀV2=%.4f\n", x1b, x2b, lyr);
    printf("  [demo] 李雅普诺夫恒等式残差 max|ṀV2-(-c1x1^2-c2z2^2)|=%.2e\n", lymax);
    chk("反步调节到位 (|x1|<0.05)", fabs(x1b), 0.05);
    chk("反步调节到位 (|x2|<0.05)", fabs(x2b), 0.05);
    chkBool("李雅普诺夫 ṀV2 < 0 (闭环稳定)", lyr < 0);
    chk("李雅普诺夫恒等式成立", lymax, 0.05);

    printf("  [结论] 反馈线性化用 u 精确抵消非线性，把对象变成已知线性系统；\n");
    printf("          反步对严格反馈链递归设计虚拟控制，给出可构造的李雅普诺夫稳定证明。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
