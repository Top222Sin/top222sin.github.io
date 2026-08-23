// ============================================================
// KF_demo L1.5 Q/R 调参：过信 / 欠信 与 RMSE 网格
// 纯 C++17。同一真值轨迹（多随机种子取平均，消除单次波动）下：
//   Q 过大 → 滤波器"太信观测" → 跟噪声（RMSE↑）
//   R 过大 → 滤波器"太信模型" → 滞后（RMSE↑）
//   Q/R 与真实噪声匹配 → RMSE 最小
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
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

// 2D CV-KF（与 L1.4 相同结构，简化协方差更新）
static double run(double q_model, double r_model, unsigned long long seed) {
    set_seed(seed);
    double dt = 0.1, q_true = 0.05, r_true = 0.5;
    int N = 150;
    double q11 = q_model * dt * dt * dt * dt / 4.0, q12 = q_model * dt * dt * dt / 2.0;
    double q22 = q_model * dt;
    double xt0 = 0.0, xt1 = 1.0;         // 真值 [p, v]
    double x0 = 0.0, x1 = 0.5;           // 滤波初值
    double P00 = 10.0, P01 = 0.0, P11 = 10.0;
    double e2 = 0;
    for (int k = 0; k < N; ++k) {
        double a = gauss() * sqrt(q_true);
        xt0 += dt * xt1 + 0.5 * dt * dt * a;
        xt1 += dt * a;
        double z = xt0 + gauss() * sqrt(r_true);
        // 预测
        double x0m = x0 + dt * x1;
        double P00m = P00 + 2 * dt * P01 + dt * dt * P11 + q11;
        double P01m = P01 + dt * P11 + q12;
        double P11m = P11 + q22;
        // 更新
        double S = P00m + r_model;
        double K0 = P00m / S, K1 = P01m / S;
        x0 = x0m + K0 * (z - x0m);
        x1 = x1 + K1 * (z - x0m);
        P00 = (1 - K0) * P00m;
        P01 = (1 - K0) * P01m;
        P11 = P11m - K1 * P01m;
        e2 += (x0 - xt0) * (x0 - xt0);
    }
    return sqrt(e2 / N);
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.5 Q/R 调参：过信与欠信 ===\n");

    unsigned long long seeds[] = {99, 1234, 777, 555, 31337, 2024, 8, 42};
    int ns = 8;
    double q_true = 0.05, r_true = 0.5;
    double good = 0, overQ = 0, overR = 0;
    for (int i = 0; i < ns; ++i) {
        good  += run(q_true,        r_true,        seeds[i]);
        overQ += run(q_true * 400,  r_true,        seeds[i]);
        overR += run(q_true,        r_true * 400,  seeds[i]);
    }
    good /= ns; overQ /= ns; overR /= ns;
    printf("  [demo] 平均RMSE: 匹配=%.4f  Q过大400x=%.4f  R过大400x=%.4f\n", good, overQ, overR);
    chk("匹配参数 RMSE 最小(平均)", (good <= overQ && good <= overR) ? 0.0 : 1.0, 0.5);

    // Q/R 扫描表（直观）
    printf("  [demo] Q\\R 扫描表（RMSE，越小越好）:\n");
    printf("           R=0.05   R=0.5    R=5.0    R=50\n");
    double qs[] = {0.0005, 0.005, 0.05, 0.5, 5.0};
    double rs[] = {0.05, 0.5, 5.0, 50.0};
    for (double qq : qs) {
        printf("        Q=%-7.4g", qq);
        for (double rr : rs) {
            double s = 0;
            for (int i = 0; i < ns; ++i) s += run(qq, rr, seeds[i]);
            printf(" %-8.4f", s / ns);
        }
        printf("\n");
    }
    printf("  (真值 q=%.3f r=%.2f → 对角线附近最优)\n", q_true, r_true);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
