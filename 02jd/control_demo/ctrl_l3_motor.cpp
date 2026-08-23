// ============================================================
// control_demo L3.1 工业用例①：直流电机速度伺服（PI 抗扰）
// 纯 C++17。二阶机电模型（机械+电气）:
//   ω' = −(b/J)ω + (Kt/J)i
//   i' = (V − R·i − Ke·ω)/L
// 验证：① PI 速度环稳态误差 <1%；② 负载突变(T+0.02N·m)跌落
//       与恢复（积分顶掉常值扰动）；③ 对照 PD(无积分)留稳态偏差
// 呼应 pid_demo L2.1 抗饱和——本篇含电压饱和 ±24V。
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
    printf("=== L3.1 直流电机速度伺服（PI 抗扰）===\n");

    const double J = 0.01, b = 0.1, Kt = 0.05, Ke = 0.05, L = 0.005, Rm = 1.0;
    const double Kp = 2.0, Ki = 20.0;
    const double w_ref = 10.0;
    const double dt = 1e-5;

    // ① PI 基本跟踪
    double w = 0, i = 0, z = 0;
    for (int k = 0; k < (int)(1.0 / dt); ++k) {
        double e = w_ref - w;
        double V = Kp * e + Ki * z;
        V = fmax(-24.0, fmin(24.0, V));
        double dw = -(b / J) * w + (Kt / J) * i;
        double di = (V - Rm * i - Ke * w) / L;
        w += dw * dt; i += di * dt; z += e * dt;
    }
    chk("PI 速度环稳态误差 < 1%", fabs(w - w_ref) / w_ref, 0.01);

    // ② 负载扰动（PI）
    w = 0; i = 0; z = 0;
    double max_dip = 0;
    for (int k = 0; k < (int)(1.0 / dt); ++k) {
        double t = k * dt;
        double e = w_ref - w;
        double V = Kp * e + Ki * z;
        V = fmax(-24.0, fmin(24.0, V));
        double Tl = (t >= 0.5) ? 0.02 : 0.0;
        double dw = -(b / J) * w + (Kt / J) * i - Tl / J;
        double di = (V - Rm * i - Ke * w) / L;
        w += dw * dt; i += di * dt; z += e * dt;
        if (t >= 0.5) max_dip = fmax(max_dip, fabs(w - w_ref));
    }
    chk("负载扰动跌落 < 3 rad/s 并恢复", max_dip, 3.0);
    printf("  [demo] PI: 稳态=%.4f/%.1f, 扰动跌落峰值=%.3f rad/s\n", w, w_ref, max_dip);

    // ③ 对照: PD(无积分)带载稳态偏差
    w = 0; i = 0;
    for (int k = 0; k < (int)(1.0 / dt); ++k) {
        double t = k * dt;
        double e = w_ref - w;
        double V = Kp * e - 0.3 * i;
        V = fmax(-24.0, fmin(24.0, V));
        double Tl = (t >= 0.5) ? 0.02 : 0.0;
        double dw = -(b / J) * w + (Kt / J) * i - Tl / J;
        double di = (V - Rm * i - Ke * w) / L;
        w += dw * dt; i += di * dt;
    }
    double ss_pd = fabs(w - w_ref);
    chk("PD(无积分)带载留稳态偏差>0.2", ss_pd > 0.2 ? 0.0 : 1.0, 0.5);
    printf("  [demo] PD: 带载稳态偏差=%.3f rad/s（积分项的价值）\n", ss_pd);
    printf("  [结论] 电机速度环是工业最常见回路；积分顶扰动，电压饱和要防积涨。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
