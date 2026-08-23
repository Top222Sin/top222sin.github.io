// ============================================================
// control_demo L3.3 工业用例③：无人机姿态 LQR（含积分增强）⭐
// 纯 C++17。滚转通道: θ̈ = (l/Jx)·u + 常值扰动
//   线性化(此处天然线性) → LQR 增益（Kleinman, 同 L2.5）
//   + 积分增广顶掉常值风扰
// 验证：① LQR 增益收敛、姿态收敛；② 常值扰动被积分项顶掉
// 真实对照：PX4 mc_att_control 的姿态环正是"LQR 思想 + 增益调度"。
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static void lqr_gains(double b, double q1, double q2, double r, double& K0, double& K1) {
    // A=[[0,1],[0,0]], B=[0;b] → Kleinman
    // 初始稳定 K（u=-[k1,k2]θ,ω → 闭环 s²+k2·b·s+k1·b）
    K0 = 4.0 / b; K1 = 4.0 / b;
    for (int it = 0; it < 300; ++it) {
        double a11 = 0, a12 = 1, a21 = -b * K0, a22 = -b * K1;
        double e11 = q1 + r * K0 * K0, e12 = r * K0 * K1, e22 = q2 + r * K1 * K1;
        double M[3][3] = {{2 * a11, 2 * a21, 0}, {a12, a11 + a22, a21}, {0, 2 * a12, 2 * a22}};
        double bvec[3] = {-e11, -e12, -e22};
        double D = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                 - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                 + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        double D0 = bvec[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                  - M[0][1] * (bvec[1] * M[2][2] - M[1][2] * bvec[2])
                  + M[0][2] * (bvec[1] * M[2][1] - M[1][1] * bvec[2]);
        double D1 = M[0][0] * (bvec[1] * M[2][2] - M[1][2] * bvec[2])
                  - bvec[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                  + M[0][2] * (M[1][0] * bvec[2] - bvec[1] * M[2][0]);
        double D2 = M[0][0] * (M[1][1] * bvec[2] - bvec[1] * M[2][1])
                  - M[0][1] * (M[1][0] * bvec[2] - bvec[1] * M[2][0])
                  + bvec[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        double P12 = D1 / D, P22 = D2 / D;
        // B=[0;b] → K = R⁻¹BᵀP = [b·P12/r, b·P22/r]
        double Kn0 = b * P12 / r, Kn1 = b * P22 / r;
        if (fabs(Kn0 - K0) < 1e-12 && fabs(Kn1 - K1) < 1e-12) { K0 = Kn0; K1 = Kn1; return; }
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
    printf("=== L3.3 无人机姿态 LQR + 积分增强 ===\n");

    const double l = 0.25, Jx = 0.02;
    const double Bin = l / Jx;             // 输入映射 12.5
    double K0, K1;
    lqr_gains(Bin, 20.0, 1.0, 0.1, K0, K1);
    const double Ki_int = 2.0;
    printf("  [demo] K_lqr=[%.3f, %.3f] + Ki=%.1f（输入映射 l/Jx=%.1f）\n", K0, K1, Ki_int, Bin);

    // 带常值扰动仿真: u = −Kθ·θ −Kω·ω − Ki·∫θ
    double th = 0.3, om = 0.0, z = 0.0;
    const double dist = 0.5;                // 常值扰动角加速度 (rad/s²)
    const double dt = 1e-4;
    double max_u = 0;
    for (int i = 0; i < (int)(8.0 / dt); ++i) {
        double u = -(K0 * th + K1 * om + Ki_int * z);
        u = fmax(-3.0, fmin(3.0, u));
        max_u = fmax(max_u, fabs(u));
        om += (Bin * u + dist) * dt;
        th += om * dt;
        z += th * dt;
    }
    chk("姿态收敛 |θ| < 0.01", fabs(th), 0.01);
    chk("常值扰动被积分顶掉 |ω|<0.01", fabs(om), 0.01);
    printf("  [demo] 8s 后 θ=%.5f, ω=%.5f, u_max=%.2f\n", th, om, max_u);
    printf("  [结论] LQR 无积分时常值扰动会留稳态角误差——加 ∫θ 增广后顶掉；\n");
    printf("         PX4 姿态环同思想（含前馈 + 增益调度）。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
