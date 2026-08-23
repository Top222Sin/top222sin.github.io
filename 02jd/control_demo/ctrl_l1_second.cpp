// ============================================================
// control_demo L1.1 二阶系统时域分析：ζ、ωn 与超调/调节时间
// 纯 C++17。闭环 G = ωn²/(s²+2ζωn·s+ωn²)。
// 验证：超调量 Mp = exp(−ζπ/√(1−ζ²)) 与阶跃仿真逐点对拍
// ============================================================
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace std;

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.1 二阶系统时域分析 ===\n");

    const double wn = 3.0;
    const double dt = 1e-4;
    for (double zeta : {0.1, 0.3, 0.5, 0.707}) {
        // 半隐式欧拉阶跃仿真
        double y = 0, dy = 0, peak = 0;
        for (int i = 0; i < (int)(6.0 / dt); ++i) {
            double ddy = wn * wn * (1.0 - y) - 2 * zeta * wn * dy;
            dy += ddy * dt;
            y += dy * dt;
            if (y > peak) peak = y;
        }
        double Mp = exp(-zeta * 3.14159265358979323846 / sqrt(1 - zeta * zeta));
        char nm[64];
        snprintf(nm, 64, "超调 ζ=%.3f (理论 vs 仿真)", zeta);
        chk(nm, fabs((peak - 1.0) - Mp), 0.02);
        if (zeta == 0.707)
            printf("  [demo] ζ=0.707: 峰值=%.4f 理论超调=%.4f\n", peak, Mp);
    }
    printf("  [公式] Mp=e^{-ζπ/√(1-ζ²)};  ts(2%%)≈4/(ζωn);  tp=π/(ωn√(1-ζ²))\n");
    printf("  [demo] ζ=0.707, ωn=3 → tp=%.3f s, ts≈%.2f s\n",
           3.14159265358979323846 / (wn * sqrt(1 - 0.5)), 4.0 / (0.707 * wn));
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
