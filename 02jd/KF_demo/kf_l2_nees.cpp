// ============================================================
// KF_demo L2.4 一致性检验：NEES / NIS 与卡方分布 ⭐
// 纯 C++17。滤波器的"诚实度体检"：
//   NEES = eᵀP⁻¹e ~ χ²(n)   （需要真值；E[NEES]=n）
//   NIS  = νᵀS⁻¹ν  ~ χ²(m)  （只需观测；E[NIS]=m）
// 验证（400 次蒙特卡洛）：
//  ① 模型匹配时 NEES≈1、NIS≈1（1 维）
//  ② Q 故意调小 200×（过信）→ NEES 爆表 ≫1 → 体检红灯
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static unsigned long long _seed = 0;
static void set_seed(unsigned long long s) { _seed = s; }
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// 跑 M 次蒙特卡洛，返回 (平均 NEES, 平均 NIS)
static void run_mc(double q_model, int M, double* nees_avg, double* nis_avg) {
    double q_true = 0.1, r = 0.25;
    int N = 40;
    double acc_n = 0, acc_i = 0;
    for (int m = 0; m < M; ++m) {
        double xt = 0, x = 0, P = 1.0;
        for (int k = 0; k < N; ++k) {
            xt += gauss() * sqrt(q_true);
            double z = xt + gauss() * sqrt(r);
            double Pm = P + q_model;
            // NIS（更新前）：ν=z-x, S=Pm+r
            double nis = (z - x) * (z - x) / (Pm + r);
            acc_i += nis / N;
            double K = Pm / (Pm + r);
            x = x + K * (z - x);
            P = (1.0 - K) * Pm;
            acc_n += (x - xt) * (x - xt) / P / N;
        }
    }
    *nees_avg = acc_n / M;
    *nis_avg = acc_i / M;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.4 一致性检验 NEES/NIS（400 次蒙特卡洛）===\n");
    printf("  1 维：NEES~χ²(1), E=1；95%% 区间均值约 [0.94, 1.06]×批量波动\n");

    int M = 400;
    double nees_ok, nis_ok;
    set_seed(100);
    run_mc(0.1, M, &nees_ok, &nis_ok);          // 匹配
    double nees_bad, nis_bad;
    set_seed(100);
    run_mc(0.1 * 0.005, M, &nees_bad, &nis_bad); // Q 调小 200× → 过信

    printf("  [demo] 匹配:   NEES=%.3f  NIS=%.3f   → 滤波器诚实\n", nees_ok, nis_ok);
    printf("  [demo] Q小200x: NEES=%.2f  NIS=%.3f   → NEES 爆表(过信)\n", nees_bad, nis_bad);

    chk("匹配时 NEES≈1", fabs(nees_ok - 1.0), 0.25);
    chk("匹配时 NIS≈1",  fabs(nis_ok - 1.0), 0.25);
    chk("过信时 NEES≫1 (红灯)", nees_bad > 3.0 ? 0.0 : 1.0, 0.5);

    printf("  [结论] 调参好坏不再靠感觉：NEES 持续 > n ⇒ Q 偏小(过信)/R 偏大；\n");
    printf("         持续 < n ⇒ 欠信。NIS 无需真值即可在线监控。\n");

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
