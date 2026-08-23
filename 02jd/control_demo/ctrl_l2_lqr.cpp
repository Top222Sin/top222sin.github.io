// ============================================================
// control_demo L2.5 LQR 最优控制（Riccati 方程）⭐
// 纯 C++17。min J=∫(xᵀQx + uᵀRu)dt → K = R⁻¹BᵀP
//   P 满足代数 Riccati: AᵀP + PA − PBR⁻¹BᵀP + Q = 0
// 求解：Kleinman 迭代（交替解 Lyapunov 方程与更新 K，从稳定 K₀ 起）
// 验证：① ARE 残差 ~1e-15；② K=R⁻¹BᵀP；③ 仿真代价 = x₀ᵀPx₀（最优性证书）
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static void solve_lyapunov(double a11, double a12, double a21, double a22,
                           double q11, double q12, double q22,
                           double& p11, double& p12, double& p22) {
    // (AᵀP+PA=−Q) 的 3 个线性方程（对称未知数）
    double M[3][3] = {
        {2 * a11, 2 * a21, 0.0},
        {a12, a11 + a22, a21},
        {0.0, 2 * a12, 2 * a22}
    };
    double b[3] = {-q11, -q12, -q22};
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
    p11 = D0 / D; p12 = D1 / D; p22 = D2 / D;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.5 LQR（Kleinman 解 ARE）===\n");

    // 双积分器 A=[[0,1],[0,0]], B=[0;1], Q=diag(10,1), R=1
    double q11w = 10, q22w = 1, r = 1;
    double K0 = 1, K1 = 2;              // 初始稳定增益
    double P11 = 0, P12 = 0, P22 = 0;
    int iters = 0;
    for (int it = 0; it < 200; ++it) {
        // Acl = A − BK = [[0,1],[−K0,−K1]]
        double a11 = 0, a12 = 1, a21 = -K0, a22 = -K1;
        // Qeq = Q + KᵀRK
        double e11 = q11w + r * K0 * K0, e12 = r * K0 * K1, e22 = q22w + r * K1 * K1;
        solve_lyapunov(a11, a12, a21, a22, e11, e12, e22, P11, P12, P22);
        double Kn0 = P12 / r, Kn1 = P22 / r;    // B=[0;1] → BᵀP = [P12, P22]
        iters = it + 1;
        if (fabs(Kn0 - K0) < 1e-13 && fabs(Kn1 - K1) < 1e-13) { K0 = Kn0; K1 = Kn1; break; }
        K0 = Kn0; K1 = Kn1;
    }
    // ① ARE 残差: AᵀP+PA − PBR⁻¹BᵀP + Q
    // A 对称部分: AᵀP+PA 手算（A=[[0,1],[0,0]]）:
    double AP11 = 0, AP12 = 0, AP22 = 0;   // AᵀP+PA = [[0, P11],[P11, 2P12]]
    double M11 = 2 * 0 * P11;              // = 0
    double M12 = P11;                      // (AᵀP+PA)[0][1]
    double M22 = 2 * P12;
    // PBR⁻¹BᵀP = P[:,1]·P[1,:]ᵀ
    double G11 = P12 * P12 / r, G12 = P12 * P22 / r, G22 = P22 * P22 / r;
    double res = fmax(fabs(M11 - G11 + q11w), fabs(M12 - G12));
    res = fmax(res, fabs(M22 - G22 + q22w));
    chk("ARE 残差 = 0", res, 1e-9);
    chk("K = R⁻¹BᵀP", fmax(fabs(K0 - P12 / r), fabs(K1 - P22 / r)), 1e-12);
    printf("  [demo] K=[%.4f, %.4f]（Kleinman %d 步收敛）, P=[[%.3f,%.3f],[%.3f,%.3f]]\n",
           K0, K1, iters, P11, P12, P12, P22);

    // ③ 最优性证书: 仿真代价 = x₀ᵀPx₀
    double x0 = 1, x1 = 0, dt = 1e-4, cost = 0;
    for (int i = 0; i < (int)(10.0 / dt); ++i) {
        double u = -(K0 * x0 + K1 * x1);
        cost += (q11w * x0 * x0 + q22w * x1 * x1 + u * u) * dt;
        x0 += x1 * dt; x1 += u * dt;
    }
    double J_opt = P11;   // x₀=[1,0]
    printf("  [demo] J_sim=%.4f  J_opt=x₀ᵀPx₀=%.4f（最优性证书）\n", cost, J_opt);
    chk("仿真代价 = x₀ᵀPx₀（相对差）", fabs(cost - J_opt) / J_opt, 0.02);
    chk("闭环收敛", fmax(fabs(x0), fabs(x1)), 0.01);
    printf("  [结论] Q 大→跟得狠(状态惩罚)；R 大→省着用(能量惩罚)。LQR 保证\n");
    printf("         相位裕度 ≥60°、增益裕度 ∞——这就是工程师爱它的原因。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
