// ============================================================
// control_demo L3.2 工业用例②：过程控制——双容水箱液位
// 纯 C++17。非线性对象（√流量）:
//   h1' = (u − c12√(h1−h2))/A1
//   h2' = (c12√(h1−h2) − c2√h2)/A2
// 验证：① PI 液位控制稳态误差 <2%；② 纯 P 控制留稳态偏差
//       （非线性对象的积分价值）；③ 后段波动 RMS
// 过程控制行业背景：化工/水处理的串级回路+时滞补偿(Smith)是标配。
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.2 过程控制：双容水箱液位 ===\n");

    const double A1 = 1.0, A2 = 1.0, c12 = 0.5, c2 = 0.5;
    const double href = 1.0;
    const double Kp = 2.0, Ki = 0.5;
    const double dt = 0.002;
    const int N = (int)(60.0 / dt);

    // ① PI
    double h1 = 0.5, h2 = 0.3, z = 0;
    double e2 = 0; int cnt = 0;
    for (int k = 0; k < N; ++k) {
        double e = href - h2;
        z += e * dt;
        double u = fmax(0.0, fmin(2.0, Kp * e + Ki * z));
        double d12 = c12 * sqrt(fmax(h1 - h2, 0.0));
        double d2 = c2 * sqrt(fmax(h2, 0.0));
        h1 += (u - d12) / A1 * dt;
        h2 += (d12 - d2) / A2 * dt;
        if (k > (int)(40.0 / dt)) { e2 += (h2 - href) * (h2 - href); ++cnt; }
    }
    chk("PI 液位稳态误差 < 2%", fabs(h2 - href) / href, 0.02);
    printf("  [demo] PI: h2=%.4f (目标 1.0)\n", h2);
    double rms = sqrt(e2 / cnt);
    chk("后段波动 RMS < 0.01", rms, 0.01);

    // ② 纯 P 对照（非线性对象 → 稳态误差）
    h1 = 0.5; h2 = 0.3;
    for (int k = 0; k < N; ++k) {
        double e = href - h2;
        double u = fmax(0.0, fmin(2.0, Kp * e));
        double d12 = c12 * sqrt(fmax(h1 - h2, 0.0));
        double d2 = c2 * sqrt(fmax(h2, 0.0));
        h1 += (u - d12) / A1 * dt;
        h2 += (d12 - d2) / A2 * dt;
    }
    double ess_p = fabs(h2 - href) / href;
    chk("纯 P 留稳态偏差 > 2%", ess_p > 0.02 ? 0.0 : 1.0, 0.5);
    printf("  [demo] 纯 P: 稳态偏差=%.2f%%（积分消除稳态误差）\n", ess_p * 100);
    printf("  [结论] √流量非线性 + 双容耦合 = 典型过程对象；大时滞时 PID 之外\n");
    printf("         还要 Smith 预估器 / MPC（Jacobian_demo L3.2 演示过 MPC）。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
