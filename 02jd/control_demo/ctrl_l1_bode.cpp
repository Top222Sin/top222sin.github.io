// ============================================================
// control_demo L1.4 频域分析：Bode、稳定裕度与 Nyquist 思想
// 纯 C++17。G = 10/((s+1)(0.1s+1)(0.01s+1))。
// 验证：① 增益交越 wc 与相位裕度 PM；② 幅值裕度 GM>1（闭环稳定）；
//       ③ PM ↔ ζ 近似（ζ≈PM/100）与闭环阶跃仿真互证
// ⚠ 工程坑：相位扫描必须解缠绕（atan2 输出在 ±180° 跳变）
// ============================================================
#include <cstdio>
#include <cmath>
#include <complex>
using namespace std;
typedef complex<double> C;

static C G(double w, double K = 10.0) {
    C s(0, w);
    return K / ((s + 1.0) * (0.1 * s + 1.0) * (0.01 * s + 1.0));
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.4 Bode 与稳定裕度 ===\n");

    // 增益交越: |G|=1（对数扫描+二分细化）
    double lo = 1e-3, hi = 1e-3;
    while (abs(G(hi)) > 1) hi *= 1.01;   // 从低频向上扫到首个 |G|<1 的点
    for (int i = 0; i < 80; ++i) {
        double mid = sqrt(lo * hi);
        if (abs(G(mid)) >= 1) lo = mid; else hi = mid;
    }
    double wc = sqrt(lo * hi);
    double pm = 180.0 + arg(G(wc)) * 180.0 / 3.14159265358979323846;
    chk("相位裕度 ∈ (45°,55°)", (45 < pm && pm < 55) ? 0.0 : 1.0, 0.5);

    // 相位交越(解缠绕): GM = 1/|G(wp)|
    double w = 1e-3;
    double prev_raw = arg(G(w)) * 180.0 / 3.14159265358979323846;
    double phu = prev_raw;
    double gm = -1;
    while (w < 1e3) {
        double w2 = w * 1.001;
        double ph = arg(G(w2)) * 180.0 / 3.14159265358979323846;
        double d = fmod(ph - prev_raw + 180.0, 360.0) - 180.0;
        double phu2 = phu + d;
        if (phu > -180.0 && phu2 <= -180.0) {
            double l = w, h = w2;
            for (int i = 0; i < 80; ++i) {
                double mid = sqrt(l * h);
                double p = arg(G(mid)) * 180.0 / 3.14159265358979323846;
                if (p > 0) p -= 360.0;   // 交越段在第三象限
                if (p > -180.0) l = mid; else h = mid;
            }
            gm = 1.0 / abs(G(sqrt(l * h)));
            break;
        }
        prev_raw = ph; phu = phu2; w = w2;
    }
    chk("幅值裕度 > 1.2 (闭环稳定)", (gm > 1.2) ? 0.0 : 1.0, 0.5);
    printf("  [demo] wc=%.3f rad/s, PM=%.1f°, GM=%.2f (%.1f dB)\n",
           wc, pm, gm, 20 * log10(gm));

    // PM ↔ ζ 近似与闭环阶跃互证
    double zeta = pm / 100.0;
    printf("  [demo] ζ≈PM/100=%.3f → 预测超调≈%.0f%%\n",
           zeta, 100 * exp(-zeta * 3.14159265358979323846 / sqrt(1 - zeta * zeta)));

    // 闭环阶跃仿真: Y/U = 10000/(s³+111s²+1110s+11000) 状态空间
    double a2 = 111.0, a1 = 1110.0, a0 = 11000.0, b = 10000.0;
    double x1 = 0, x2 = 0, x3 = 0, peak = 0;
    double dt = 2e-4;
    for (int i = 0; i < (int)(20.0 / dt); ++i) {
        double dx1 = x2, dx2 = x3, dx3 = b * 1.0 - a2 * x3 - a1 * x2 - a0 * x1;
        x1 += dx1 * dt; x2 += dx2 * dt; x3 += dx3 * dt;
        peak = fmax(peak, x1);
    }
    chk("闭环阶跃峰值与 PM 预测一致(<1.35)", peak, 1.35);
    printf("  [demo] 闭环阶跃峰值=%.3f\n", peak);
    printf("  [结论] PM 大→阻尼大→超调小；GM 保证不远失稳。Nyquist 判据是背后的理论。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
