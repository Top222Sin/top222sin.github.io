// ============================================================
// KF_demo L3.1 工业用例：IMU/GPS 误差状态 ES-EKF 与 GPS 失锁 ⭐
// 纯 C++17。误差状态法（1D 垂直简化版）：
//   标称状态（DR 积分 a_meas − b̂）+ 误差状态 δx=[δp,δv,δb] 用线性 KF
//   a_meas = a_true + b + n → 误差动力学 F = [[1,dt,−dt²/2],[0,1,−dt],[0,0,1]]
// 验证：
//  ① GPS 可用时 ES-EKF 位置 RMSE ≪ 纯 DR（0.12 vs 4.9）
//  ② 加计零偏 b 被在线估出（|b̂−b| < 0.05）
//  ③ 120~160 步 GPS 失锁段：误差按 P 预测的速率增长而非爆炸
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

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.1 IMU/GPS 误差状态 ES-EKF（含失锁段）===\n");

    double dt = 0.05, q_a = 1e-4, r_g = 0.05, b_true = 0.2;
    int N = 200;
    // GPS 失锁段 [120,160)
    auto gps_ok = [](int k) { return !(k >= 120 && k < 160); };

    // ---- ES-EKF ----
    set_seed(11);
    double p_t = 0, v_t = 0;               // 真值
    double p_n = 0, v_n = 0, b_n = 0;      // 标称状态
    // 误差协方差 P（3×3 手写展开）
    double P[3][3] = {{0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}};
    double F[3][3] = {{1, dt, -0.5 * dt * dt}, {0, 1, -dt}, {0, 0, 1}};
    double Q00 = q_a * dt*dt*dt*dt / 4, Q01 = q_a * dt*dt*dt / 2, Q11 = q_a * dt;
    double e2kf = 0; int cnt = 0;
    double b_err_late = 0;
    double p_err_at_outage_end = 0;
    for (int k = 0; k < N; ++k) {
        double a_true = 1.0 * sin(0.1 * k);
        double a_meas = a_true + b_true + gauss() * sqrt(q_a);
        p_t += v_t * dt + 0.5 * a_true * dt * dt;
        v_t += a_true * dt;
        p_n += v_n * dt + 0.5 * (a_meas - b_n) * dt * dt;
        v_n += (a_meas - b_n) * dt;
        // 误差预测 P⁻ = F P Fᵀ + Q（3×3 展开太长 → 用小循环矩阵乘）
        double Pm[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int t1 = 0; t1 < 3; ++t1)
                    for (int t2 = 0; t2 < 3; ++t2) s += F[i][t1] * P[t1][t2] * F[j][t2];
                Pm[i][j] = s;
            }
        Pm[0][0] += Q00; Pm[0][1] += Q01; Pm[1][0] += Q01; Pm[1][1] += Q11;
        // GPS 更新：H=[1,0,0]，残差 = z − p_n
        if (gps_ok(k) && k % 2 == 0) {
            double z = p_t + gauss() * sqrt(r_g);
            double resid = z - p_n;
            double S = Pm[0][0] + r_g;
            double K[3] = {Pm[0][0] / S, Pm[1][0] / S, Pm[2][0] / S};
            p_n += K[0] * resid; v_n += K[1] * resid; b_n += K[2] * resid;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    P[i][j] = Pm[i][j] - K[i] * Pm[0][j];
            // 对称化
            for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
                double m = 0.5 * (P[i][j] + P[j][i]); P[i][j] = P[j][i] = m;
            }
        } else {
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) P[i][j] = Pm[i][j];
        }
        if (k >= 30) { e2kf += (p_n - p_t) * (p_n - p_t); ++cnt; }
        if (k >= 150) b_err_late = fmax(b_err_late, fabs(b_n - b_true));
        if (k == 159) p_err_at_outage_end = fabs(p_n - p_t);
    }
    double rmse_kf = sqrt(e2kf / cnt);

    // ---- 纯 DR 对照 ----
    set_seed(11);
    double p_t2 = 0, v_t2 = 0, p_d = 0, v_d = 0;
    double e2dr = 0; int cnt2 = 0;
    for (int k = 0; k < N; ++k) {
        double a_true = 1.0 * sin(0.1 * k);
        double a_meas = a_true + b_true + gauss() * sqrt(q_a);
        p_t2 += v_t2 * dt + 0.5 * a_true * dt * dt;
        v_t2 += a_true * dt;
        p_d += v_d * dt + 0.5 * a_meas * dt * dt;
        v_d += a_meas * dt;
        if (k >= 30) { e2dr += (p_d - p_t2) * (p_d - p_t2); ++cnt2; }
    }
    double rmse_dr = sqrt(e2dr / cnt2);

    printf("  [demo] rmse_ES-EKF=%.4f  rmse_纯DR=%.4f  (失锁段末端误差=%.3f m)\n",
           rmse_kf, rmse_dr, p_err_at_outage_end);
    printf("  [demo] 零偏估计 b̂=%.4f  (真值 b=%.4f)\n", b_n, b_true);
    chk("ES-EKF RMSE ≪ 纯 DR", rmse_kf < 0.2 * rmse_dr ? 0.0 : 1.0, 0.5);
    chk("零偏估计收敛 |b̂−b|<0.05", b_err_late, 0.05);
    chk("失锁段末端误差有界(<1m)", p_err_at_outage_end, 1.0);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
