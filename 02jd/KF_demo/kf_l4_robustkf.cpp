// ============================================================
// KF_demo L4.3 鲁棒 / 自适应卡尔曼滤波（M-估计 + Sage-Husa）⭐
// 纯 C++17（无第三方库）。承接 L3.4 融合 capstone 的 NEES/NIS 闸门
// （那里只 gate 不重加权），把"野值处理 / 噪声自适应"在 KF 里落实：
//   Part A  鲁棒代价（M-估计）在 KF 更新里做 IRLS 重加权：
//           标准 KF 更新 = 最小化高斯 NLL；换成 Huber / Student-t 后，
//           每步用标准化残差 r = y/√S 算权重 w(r)=psi(r)/r，增益 K -> w*K。
//           => 野值被大幅降权，估计不被带偏。
//   Part B  Sage-Husa 自适应 R 估计（在线估计量测噪声协方差）：
//           R_hat <- (1-b)R_hat + b*( y^2 - Pp )，期望恰为真实 R。
//           过程噪声 Q 主导预测协方差时收敛稳定，能跟踪时变噪声。
// 验证（受管 Python 3.13.12 先行对拍，本文件为确定性重现）：
//   7 项检查全 PASS，末行打印 "=== N passed, M failed ==="
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

// ---- 固定种子 LCG + Box-Muller（与验证端逐位一致，seed=20260820）----
static unsigned long long _seed = 20260820ULL;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    _seed &= 0xFFFFFFFFFFFFFFFFULL;
    return (double)((_seed >> 33) & 0x7FFFFFFFULL) / (double)0x80000000ULL;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// ---- 鲁棒权重（M-估计 psi 的 w(r)=psi(r)/r）----
static double huber_w(double r, double c) {
    double a = fabs(r);
    return (a <= c) ? 1.0 : c / a;
}
static double studentt_w(double r, double nu) {
    return (nu + 1.0) / (nu + r * r);
}

// ============================================================
// Part A：鲁棒 KF（1D 随机游走 + 注入野值）
// mode: 0=标准高斯, 1=Huber(c=1.345), 2=Student-t(nu=4)
// ============================================================
struct RobustOut { Vec est; Vec W; };
static RobustOut run_kf_robust(double x0, double P0, double Q, double R,
                               const Vec& z, int mode) {
    double x = x0, P = P0;
    RobustOut o; o.est.resize(z.size()); o.W.resize(z.size());
    for (size_t k = 0; k < z.size(); ++k) {
        double Pp = P + Q;
        double y = z[k] - x;
        double S = Pp + R;
        double K = Pp / S;
        double w = 1.0;
        if (mode == 1) { double r = y / sqrt(S); w = huber_w(r, 1.345); }
        else if (mode == 2) { double r = y / sqrt(S); w = studentt_w(r, 4.0); }
        double Kr = w * K;
        x = x + Kr * y;
        P = (1.0 - Kr) * Pp;
        o.est[k] = x; o.W[k] = w;
    }
    return o;
}

static double rmse(const Vec& a, const Vec& b) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; s += d * d; }
    return sqrt(s / a.size());
}

static void partA(int& pass, int& fail) {
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-46s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== A. 鲁棒代价（Huber / Student-t）IRLS 重加权 KF ===\n");
    const int N = 120;
    const double Q = 0.1, R = 1.0, x0 = 0.0, P0 = 1.0;

    Vec xt(N + 1); xt[0] = x0;
    for (int k = 0; k < N; ++k) xt[k+1] = xt[k] + gauss() * sqrt(Q);
    Vec z(N);
    for (int k = 0; k < N; ++k) z[k] = xt[k+1] + gauss() * sqrt(R);

    // 注入野值：确定步集上叠加大幅尖峰（±18，远超 σ=1）
    bool outlier[120] = {false};
    int osteps[8] = {8, 23, 41, 57, 74, 92, 108, 117};
    for (int i = 0; i < 8; ++i) outlier[osteps[i]] = true;
    for (int k = 0; k < N; ++k) if (outlier[k]) z[k] += 18.0 * (k % 2 == 0 ? 1.0 : -1.0);

    RobustOut std = run_kf_robust(x0, P0, Q, R, z, 0);
    RobustOut hub = run_kf_robust(x0, P0, Q, R, z, 1);
    RobustOut st  = run_kf_robust(x0, P0, Q, R, z, 2);

    Vec true_traj(N);
    for (int k = 0; k < N; ++k) true_traj[k] = xt[k+1];

    double rmse_std = rmse(std.est, true_traj);
    double rmse_hub = rmse(hub.est, true_traj);
    double rmse_st  = rmse(st.est,  true_traj);

    double min_hub_w = 1e9;
    for (int i = 0; i < 8; ++i) min_hub_w = min(min_hub_w, hub.W[osteps[i]]);

    printf("  [info] 标准KF RMSE=%.4f, Huber RMSE=%.4f, Student-t RMSE=%.4f\n", rmse_std, rmse_hub, rmse_st);
    printf("  [info] 野值步上 Huber 最小权重 = %.4f (应 <<1，被降权)\n", min_hub_w);
    chk("Huber 鲁棒 KF RMSE 显著低于标准 KF(<0.6x)", rmse_hub, 0.6 * rmse_std);
    chk("Student-t 鲁棒 KF RMSE 显著低于标准 KF(<0.6x)", rmse_st, 0.6 * rmse_std);
    chk("野值步 Huber 权重被降权(<0.5)", max(0.0, min_hub_w - 0.5), 1e-9);
    chk("Huber 与 Student-t 改进幅度相近(<0.5xStd)", fabs(rmse_hub - rmse_st), 0.5 * rmse_std);
}

// ============================================================
// Part B：Sage-Husa 自适应 R 估计（时变噪声）
// ============================================================
static void partB(int& pass, int& fail) {
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-46s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("\n=== B. Sage-Husa 自适应 R 估计（噪声时变）===\n");
    const int N = 300;
    const double Q = 4.0, x0 = 0.0, P0 = 1.0;
    const double R1 = 2.0, R2 = 6.0;
    const int sw = 100;
    const double b = 0.015;

    Vec xt(N + 1); xt[0] = x0;
    for (int k = 0; k < N; ++k) xt[k+1] = xt[k] + gauss() * sqrt(Q);
    Vec z(N);
    for (int k = 0; k < N; ++k) {
        double Rt = (k >= sw) ? R2 : R1;
        z[k] = xt[k+1] + gauss() * sqrt(Rt);
    }

    double Rhat = R1;
    double x = x0, P = P0;
    Vec Rtraj(N), NIS(N);
    for (int k = 0; k < N; ++k) {
        double Pp = P + Q;
        double y = z[k] - x;
        double S = Pp + Rhat;
        double K = Pp / S;
        x = x + K * y;
        P = (1.0 - K) * Pp;
        Rhat = (1.0 - b) * Rhat + b * (y * y - Pp);
        if (Rhat < 1e-6) Rhat = 1e-6;
        Rtraj[k] = Rhat;
        NIS[k] = y * y / (Pp + Rhat);
    }

    double end1 = 0; for (int k = sw - 10; k < sw; ++k) end1 += Rtraj[k]; end1 /= 10.0;
    double end2 = 0; for (int k = N - 20; k < N; ++k) end2 += Rtraj[k]; end2 /= 20.0;
    double nis_late = 0; for (int k = sw + 30; k < N; ++k) nis_late += NIS[k]; nis_late /= (N - (sw + 30));

    printf("  [info] R_hat 第1段末≈%.4f (真值 %.2f)\n", end1, R1);
    printf("  [info] R_hat 第2段末≈%.4f (真值 %.2f)\n", end2, R2);
    printf("  [info] 第2段后期 NIS 均值=%.4f (自适应后新息尺度应≈1)\n", nis_late);

    chk("Sage-Husa 跟踪第1段 R(≈2.0)", fabs(end1 - R1), 1.5);
    chk("Sage-Husa 适应突变(第2段 R 显著大于第1段)",
        max(0.0, (R1 + 0.5 * (R2 - R1)) - end2), 1e-9);
    chk("自适应后 NIS 均值≈1(新息尺度一致)", fabs(nis_late - 1.0), 0.6);
}

int main() {
    int pass = 0, fail = 0;
    partA(pass, fail);
    partB(pass, fail);
    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
