// ============================================================
// control_demo L3.4 capstone：位置伺服系统设计全流程 ⭐⭐
// 纯 C++17。工业设计四步法（把 L1 全部串起来）：
//   ① 系统辨识：阶跃速度响应 → 两点法拟合 K/(s+a)
//   ② 指标→频域：Kv=10 定增益；测 PM 发现欠阻尼(PM<45°)
//   ③ 设计：测速反馈(Td) 补阻尼 → 闭环 s²+(3+K·Td)s+K
//   ④ 验收：闭环阶跃仿真 vs 二阶理论(ζ, Mp)
// 对象：G(s)=3/(s(s+3))（电机+减速器的位置伺服）
// ============================================================
#include <cstdio>
#include <cmath>
#include <complex>
using namespace std;
typedef complex<double> C;

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.4 capstone：位置伺服设计全流程 ===\n");

    // ---- ① 系统辨识（ω' = −a·ω + K·V, V=1 阶跃） ----
    const double dt = 1e-4;
    double w = 0, w01 = 0, w10 = 0;
    for (int i = 0; i < (int)(3.0 / dt); ++i) {
        double t = i * dt;
        w += (-3.0 * w + 3.0 * 1.0) * dt;
        if (fabs(t - 0.1) < dt / 2) w01 = w;
        if (fabs(t - 1.0) < dt / 2) w10 = w;
    }
    // 两点法: r = ω(0.1)/ω(1.0), 解 (1−e^{-0.1a}) = r(1−e^{-a}) → 二分
    double r = w01 / w10;
    double lo = 0.1, hi = 10.0;
    for (int i = 0; i < 80; ++i) {
        double mid = 0.5 * (lo + hi);
        if ((1 - exp(-mid * 0.1)) < r * (1 - exp(-mid * 1.0))) lo = mid;
        else hi = mid;
    }
    double a_est = 0.5 * (lo + hi);
    double K_est = a_est * w10 / (1 - exp(-a_est * 1.0));
    chk("辨识 a ≈ 3.0", fabs(a_est - 3.0), 0.1);
    chk("辨识 K ≈ 3.0", fabs(K_est - 3.0), 0.1);
    printf("  [demo] ① 辨识: K=%.3f, a=%.3f → G=K/(s(s+a))\n", K_est, a_est);

    // ---- ② 指标与未补偿裕度 ----
    const double Kv = 10.0;
    double Kc = Kv * a_est;               // Kv = K/a → K=30
    auto G0 = [&](double om) { C s(0, om); return Kc / (s * (s + a_est)); };
    double l = 0.01, h = 0.01;
    while (abs(G0(h)) >= 1) h *= 1.001;
    l = h / 1.001;
    for (int i = 0; i < 60; ++i) {
        double mid = sqrt(l * h);
        if (abs(G0(mid)) >= 1) l = mid; else h = mid;
    }
    double wc = sqrt(l * h);
    double pm = 180.0 + arg(G0(wc)) * 180.0 / 3.14159265358979323846;
    chk("② 未补偿 PM < 45°（欠阻尼）", pm < 45.0 ? 0.0 : 1.0, 0.5);
    printf("  [demo] ② Kv=%.0f → K=%.1f; wc=%.2f, PM=%.1f°（需补阻尼）\n",
           Kv, Kc, wc, pm);

    // ---- ③ 测速反馈设计: 闭环 s²+(a+K·Td)s+K → ζ 目标 0.55 ----
    double zeta_target = 0.55;
    double wn = sqrt(Kc);
    double Td = (2 * zeta_target * wn - a_est) / Kc;
    double zeta = (a_est + Kc * Td) / (2 * wn);
    printf("  [demo] ③ 测速反馈 Td=%.3f → ζ=%.3f\n", Td, zeta);

    // ---- ④ 闭环阶跃仿真 vs 二阶理论 ----
    double th = 0, om2 = 0, peak = 0;
    for (int i = 0; i < (int)(4.0 / dt); ++i) {
        double V = Kc * (1.0 - th) - Kc * Td * om2;   // PD 位置环(含测速反馈)
        V = fmax(-15.0, fmin(15.0, V));
        double dom = -a_est * om2 + V;                 // 电机一阶等效
        om2 += dom * dt;
        th += om2 * dt;
        peak = fmax(peak, th);
    }
    double Mp = exp(-zeta * 3.14159265358979323846 / sqrt(1 - zeta * zeta));
    chk("④ 超调与二阶理论一致", fabs((peak - 1.0) - Mp), 0.03);
    chk("④ 阶跃跟踪收敛", fabs(th - 1.0), 0.02);
    printf("  [demo] ④ 闭环峰值=%.3f, 理论超调=%.3f, 稳态=%.4f\n", peak, Mp, th);
    printf("  [结论] 建模→指标→设计→验收 四步闭环；每个数字都有出处——这就是\n");
    printf("         L1~L3 全部知识的工业用法。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
