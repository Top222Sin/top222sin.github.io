// ============================================================
// control_demo L2.6 LQG = LQR + KF（分离定理实战）⭐⭐
// 纯 C++17。只有位置可测、带过程/观测噪声的系统：
//   LQR 设计 u = −Kx̂（假定状态全知）
//   KF 从 y 估 x̂（看不见控制环）
//   分离定理：两个问题独立最优 → 组合全局最优（线性高斯下）
// 衔接：KF_demo 全部（本 demo 是控制侧入口）
// ⚠ 坑：对象传播必须含控制项 Bd·u（漏掉=开环，随机游走发散）
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static unsigned long long _seed = 26;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}
// expm([[0,1],[0,0]], dt) = [[1,dt],[0,1]]（解析已知,免通用 expm）
static const double PI_ = 3.14159265358979323846;

static void lqr_gains(double q1, double q2, double r, double& K0, double& K1) {
    // Kleinman（同 L2.5,内联简版）
    K0 = 1; K1 = 2;
    for (int it = 0; it < 200; ++it) {
        double a11 = 0, a12 = 1, a21 = -K0, a22 = -K1;
        double e11 = q1 + r * K0 * K0, e12 = r * K0 * K1, e22 = q2 + r * K1 * K1;
        double M[3][3] = {{2 * a11, 2 * a21, 0}, {a12, a11 + a22, a21}, {0, 2 * a12, 2 * a22}};
        double b[3] = {-e11, -e12, -e22};
        double D = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                 - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                 + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        double D0 = b[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                  - M[0][1] * (b[1] * M[2][2] - M[1][2] * b[2])
                  + M[0][2] * (b[1] * M[2][1] - M[1][1] * b[2]);
        double D1 = M[0][0] * (b[1] * M[2][2] - M[1][2] * b[2])
                  - b[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                  + M[0][2] * (M[1][0] * b[2] - b[1] * M[2][0]);
        double D2 = M[0][0] * (M[1][1] * b[2] - b[1] * M[2][1])
                  - M[0][1] * (M[1][0] * b[2] - b[1] * M[2][0])
                  + b[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        double P12 = D1 / D, P22 = D2 / D;
        double Kn0 = P12 / r, Kn1 = P22 / r;
        if (fabs(Kn0 - K0) < 1e-13 && fabs(Kn1 - K1) < 1e-13) { K0 = Kn0; K1 = Kn1; return; }
        K0 = Kn0; K1 = Kn1;
    }
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.6 LQG = LQR + KF ===\n");

    // LQR（Q=I, R=0.5）
    double K0, K1;
    lqr_gains(1, 1, 0.5, K0, K1);
    // KF 稳态增益（离散 DARE 迭代, 复用 KF_demo L2.1 思路）
    double dt = 0.02, q_proc = 0.01, r_obs = 0.25;
    // Ad=[[1,dt],[0,1]], Bd=[dt²/2; dt], Qd = 与下方仿真注入噪声协方差一致
    //   （w0=½dt·wv, w1=wv, var(wv)=q·dt → cov=[[q·dt³/4, q·dt²/2],[q·dt²/2, q·dt]]）
    double Ad[2][2] = {{1, dt}, {0, 1}};
    double Bd0 = 0.5 * dt * dt, Bd1 = dt;
    double Q00 = q_proc * dt * dt * dt / 4, Q01 = q_proc * dt * dt / 2;
    double Q11 = q_proc * dt;
    double p00 = 1, p01 = 0, p11 = 1;
    for (int it = 0; it < 2000; ++it) {
        double m00 = p00 + 2 * dt * p01 + dt * dt * p11 + Q00;
        double m01 = p01 + dt * p11 + Q01;
        double m11 = p11 + Q11;
        double S = m00 + r_obs;
        double k0 = m00 / S, k1 = m01 / S;
        p00 = (1 - k0) * m00;
        p01 = (1 - k0) * m01;
        p11 = m11 - k1 * m01;
    }
    double S = p00 + r_obs;
    double kf0 = p00 / S, kf1 = p01 / S;
    printf("  [demo] K_lqr=[%.3f, %.3f], K_kf=[%.3f, %.3f]\n", K0, K1, kf0, kf1);

    // LQG 闭环
    _seed = 26;
    double x0 = 1, x1 = 0, h0 = 0, h1 = 0;
    double e2 = 0; int cnt = 0;
    for (int k = 0; k < 1000; ++k) {
        double u = -(K0 * h0 + K1 * h1);
        double wv = gauss() * sqrt(q_proc * dt);
        double w0 = 0.5 * dt * wv, w1 = wv;
        double nx0 = Ad[0][0] * x0 + Ad[0][1] * x1 + Bd0 * u + w0;
        double nx1 = Ad[1][0] * x0 + Ad[1][1] * x1 + Bd1 * u + w1;
        x0 = nx0; x1 = nx1;
        double y = x0 + gauss() * sqrt(r_obs);
        double nh0 = Ad[0][0] * h0 + Ad[0][1] * h1 + Bd0 * u;
        double nh1 = Ad[1][0] * h0 + Ad[1][1] * h1 + Bd1 * u;
        h0 = nh0 + kf0 * (y - nh0);
        h1 = nh1 + kf1 * (y - nh0);
        if (k >= 200) { e2 += x0 * x0; ++cnt; }
    }
    double rms = sqrt(e2 / cnt);
    printf("  [demo] 位置状态 RMS = %.4f（噪声驱动下的最优调节）\n", rms);
    chk("LQG 输出反馈稳定 (RMS<0.5)", rms, 0.5);
    printf("  [结论] 分离定理：LQR 管控制、KF 管估计，各调各的最优即全局最优。\n");
    printf("         这就是 KF_demo + 本 L2.5 的组合应用。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
