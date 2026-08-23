// ============================================================
// control_demo L2.4 观测器设计与分离原理
// 纯 C++17。Luenberger 观测器: x̂' = Ax̂ + Bu + L(y−Cx̂)
//   误差动力学 e' = (A−LC)e → 极点可任意配置（能观）
// 验证：① 观测器闭环极点 s²+16s+128（比控制环快 ~4 倍）；
//       ② 分离原理：观测器 + 状态反馈 = 输出反馈，可独立设计
// ⚠ 坑：观测器模型必须包含控制输入 u（漏掉会发散）
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
    printf("=== L2.4 观测器与分离原理 ===\n");

    // A=[[0,1],[0,0]], C=[1,0] → A−LC = [[−l1,1],[−l2,0]]
    // 期望观测器极点 −8±8j: s²+16s+128 → l1=16, l2=128
    double l1 = 16.0, l2 = 128.0;
    double tr = -l1 + 0.0;
    double det = (-l1) * 0.0 - 1.0 * (-l2);
    chk("观测器极点 s²+16s+128（迹）", fabs(tr + 16.0), 1e-12);
    chk("观测器极点 s²+16s+128（行列式）", fabs(det - 128.0), 1e-12);

    // 分离原理仿真: u = −Kx̂ (K=[8,4] 来自 L2.3 思路), 全系统只有 y 可测
    double K0 = 8.0, K1 = 4.0;
    double x0 = 1, x1 = 0, h0 = 0, h1 = 0;
    double dt = 1e-4;
    for (int i = 0; i < (int)(8.0 / dt); ++i) {
        double y = x0;
        double u = -(K0 * h0 + K1 * h1);
        // 真实
        x0 += x1 * dt;
        x1 += u * dt;
        // 观测器（含控制输入 u！）
        double dy0 = h1 + l1 * (y - h0);
        double dy1 = u + l2 * (y - h0);
        h0 += dy0 * dt;
        h1 += dy1 * dt;
    }
    chk("分离原理: 输出反馈收敛", fmax(fabs(x0), fabs(x1)), 0.01);
    chk("估计误差收敛", fmax(fabs(h0 - x0), fabs(h1 - x1)), 0.01);
    printf("  [demo] 8s 后 x=[%.6f, %.6f], x̂=[%.6f, %.6f]\n", x0, x1, h0, h1);
    printf("  [结论] 控制增益 K 与观测器增益 L 可独立设计再拼装（分离原理）；\n");
    printf("         观测器极点取控制环 3~5 倍快：估计别拖控制后腿。\n");
    printf("         KF(L2.6) 是 L 在噪声下的最优版本。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
