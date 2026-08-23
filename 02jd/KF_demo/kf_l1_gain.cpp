// ============================================================
// KF_demo L1.3 增益 K 的物理意义与边界行为
// 纯 C++17。验证：
//  ① 一维 Riccati 迭代收敛到稳态 (p∞, K∞)，K_k 单调收敛
//  ② 边界：r→0 ⇒ K→1（全信观测）；r→∞ ⇒ K→0（全信模型）
// ============================================================
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace std;

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.3 增益 K：边界行为与稳态 ===\n");

    double q = 0.1, r = 1.0;
    // 稳态：迭代 1D Riccati p ← (1-K)(p+q), K=(p+q)/(p+q+r)
    double p = 1.0, Kinf = 0;
    for (int i = 0; i < 2000; ++i) {
        double pm = p + q;
        double K = pm / (pm + r);
        p = (1.0 - K) * pm;
    }
    Kinf = (p + q) / (p + q + r);
    printf("  [demo] p_ss=%.6f  K_inf=%.6f  (q=%.2f r=%.2f)\n", p, Kinf, q, r);

    // K_k 单调收敛到 Kinf（本例 p0 较大 → K 递减）
    {
        double pk = 1.0, prev = -1;
        bool mono = true;
        for (int i = 0; i < 200; ++i) {
            double pm = pk + q;
            double Kk = pm / (pm + r);
            pk = (1.0 - Kk) * pm;
            if (prev > 0 && Kk > prev + 1e-15) mono = false;
            if (prev < 0) prev = Kk; else prev = prev;  // 记录首个
            prev = Kk;
        }
        chk("K_k 收敛到 K∞", fabs(prev - Kinf), 1e-12);
        chk("K 单调(本例递减)", mono ? 0.0 : 1.0, 0.5);
    }

    // 边界行为
    {
        double K_r0 = (p + q) / (p + q + 1e-9);
        double K_rinf = (p + q) / (p + q + 1e9);
        chk("r→0  ⇒ K→1 (全信观测)", fabs(K_r0 - 1.0), 1e-6);
        chk("r→∞  ⇒ K→0 (全信模型)", K_rinf, 1e-6);
    }

    // 直观表格：不同 r 下的 K∞
    printf("  [demo] r      K∞      直觉\n");
    for (double rr : {0.01, 0.1, 1.0, 10.0, 100.0}) {
        double pk = 1.0;
        for (int i = 0; i < 2000; ++i) {
            double pm = pk + q;
            double K = pm / (pm + rr);
            pk = (1.0 - K) * pm;
        }
        double Kk = (pk + q) / (pk + q + rr);
        printf("        %-6.2f %.4f  %s\n", rr, Kk,
               Kk > 0.7 ? "紧贴观测" : (Kk < 0.1 ? "紧贴模型" : "折中"));
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
