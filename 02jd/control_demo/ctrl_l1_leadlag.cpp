// ============================================================
// control_demo L1.5 频域设计：超前补偿器（Lead Compensator）
// 纯 C++17。对象 G = 20/(s(s+2))，指标：Kv=10 且 PM≥45°。
// 超前补偿 C(s) = (1+aTs)/(1+Ts), a>1, 最大超前角在 ωm=1/(√a·T)：
//   sin φm = (a−1)/(a+1)
// 设计流程（教科书三步）：① 未补偿 wc 与 PM；② 求需要的 a；
// ③ ωm 对准新交越点 → T；验证补偿后 PM。
// ⚠ 坑：交越点右移带来相位滞后，需预留 10~15° 安全裕量。
// ============================================================
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
using namespace std;
typedef complex<double> C;

static C G0(double w) { C s(0, w); return 20.0 / (s * (s + 2.0)); }
static double PI_ = 3.14159265358979323846;

template<class F>
static double find_wc(F Gf) {
    double lo = 1e-3, hi = 1e-3;
    while (abs(Gf(hi)) >= 1 && hi < 1e4) hi *= 1.001;
    lo = hi / 1.001;
    for (int i = 0; i < 80; ++i) {
        double mid = sqrt(lo * hi);
        if (abs(Gf(mid)) >= 1) lo = mid; else hi = mid;
    }
    return sqrt(lo * hi);
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.5 超前补偿器设计 ===\n");

    // ① 未补偿裕度
    double wc0 = find_wc(G0);
    double pm0 = 180.0 + arg(G0(wc0)) * 180.0 / PI_;
    printf("  [demo] 未补偿: wc=%.2f, PM=%.1f° (不达标)\n", wc0, pm0);

    // ② 需要的相位超前 → a
    double need = 45.0 - pm0 + 12.0;   // +12° 安全裕量
    double a = (1 + sin(need * PI_ / 180)) / (1 - sin(need * PI_ / 180));
    // ③ ωm ≈ wc0 → T = 1/(√a·ωm)
    double T = 1.0 / (sqrt(a) * wc0);
    auto Gc = [&](double w) { C s(0, w); return (1.0 + a * T * s) / (1.0 + T * s); };
    auto GL = [&](double w) { return Gc(w) * G0(w); };

    double wc1 = find_wc(GL);
    double pm1 = 180.0 + arg(GL(wc1)) * 180.0 / PI_;
    printf("  [demo] 补偿器: a=%.2f, T=%.4f (φm=%.1f° @ %.2f rad/s)\n",
           a, T, need, 1.0 / (sqrt(a) * T));
    printf("  [demo] 补偿后: wc=%.2f, PM=%.1f°\n", wc1, pm1);
    chk("超前补偿后 PM ≥ 45°", (pm1 >= 45.0) ? 0.0 : 1.0, 0.5);

    // 闭环极点验证: 特征方程 T s³+(1+2T)s²+(2+20aT)s+20 = 0
    double c3 = T, c2 = 1 + 2 * T, c1 = 2 + 20 * a * T, c0 = 20;
    auto f = [&](double s) { return ((c3 * s + c2) * s + c1) * s + c0; };
    auto df = [&](double s) { return (3 * c3 * s + 2 * c2) * s + c1; };
    vector<double> re_roots;
    for (int g = -80; g <= 40; ++g) {
        double s = g * 0.5;
        for (int it = 0; it < 80; ++it) {
            double d = df(s);
            if (fabs(d) < 1e-12) break;
            double sn = s - f(s) / d;
            if (fabs(sn - s) < 1e-13) { s = sn; break; }
            s = sn;
        }
        if (fabs(f(s)) < 1e-7) {
            bool dup = false;
            for (double r : re_roots) if (fabs(r - s) < 1e-5) dup = true;
            if (!dup) re_roots.push_back(s);
        }
    }
    // 实根之外的复根对实部 = -(c2+c3*r0)/(2c3)
    double re_min = 1e9;
    for (double r : re_roots) re_min = fmin(re_min, r);
    if (re_roots.size() == 1) {
        double q1 = c2 + c3 * re_roots[0];
        re_min = fmin(re_min, -q1 / (2 * c3));
        printf("  [demo] 闭环极点: %.2f, %.2f±%.2fj\n",
               re_roots[0], -q1 / (2 * c3),
               sqrt(fabs(q1 * q1 / (4 * c3 * c3) - (c1 + q1 * re_roots[0]) / c3)));
    }
    chk("闭环所有根在左半平面", re_min < 0 ? 0.0 : 1.0, 0.5);
    printf("  [结论] 超前=加相位提阻尼(带宽↑)；滞后=提增益消稳差(带宽↓)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
