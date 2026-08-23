// ============================================================
// KF_demo L1.2 贝叶斯视角：高斯乘积(更新)与卷积(预测)
// 纯 C++17。验证：
//  ① 两高斯乘积公式 == KF 更新代数（均值/方差逐位一致）
//  ② 预测 = 卷积：Var(a+b)=Var(a)+Var(b)（20 万样本蒙特卡洛）
//  ③ 信息形式 Ω=1/σ²，更新 Ω=Ω⁻+1/r
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static unsigned long long _seed = 42;
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
    printf("=== L1.2 贝叶斯视角：高斯的积与卷积 ===\n");

    // ---- ① 高斯乘积 == KF 更新 ----
    double m1 = 3.0, v1 = 2.0;   // 先验 N(3, 2)（v 为方差）
    double m2 = 1.0, v2 = 0.5;   // 观测 N(1, 0.5)
    // 乘积公式（解析）
    double mu_prod = (m1 * v2 + m2 * v1) / (v1 + v2);
    double var_prod = v1 * v2 / (v1 + v2);
    // KF 更新代数: K=p/(p+r)
    double K = v1 / (v1 + v2);
    double mu_kf = m1 + K * (m2 - m1);
    double var_kf = (1.0 - K) * v1;
    chk("高斯乘积=KF更新 均值", fabs(mu_prod - mu_kf), 1e-12);
    chk("高斯乘积=KF更新 方差", fabs(var_prod - var_kf), 1e-12);

    // ---- ② 卷积 = 预测（蒙特卡洛验证方差相加） ----
    {
        int n = 20000;
        double add_noise_var = 0.3;
        double s = 0, s2 = 0;
        for (int i = 0; i < n; ++i) {
            double b = gauss() * sqrt(v1) + m1;   // 先验样本
            double c = b + gauss() * sqrt(add_noise_var);  // 加过程噪声
            s += c; s2 += c * c;
        }
        double m_emp = s / n;
        double v_emp = s2 / n - m_emp * m_emp;
        chk("卷积均值=预测均值(均值不变)", fabs(m_emp - m1), 0.05);
        chk("卷积方差=方差相加(预测)", fabs(v_emp - (v1 + add_noise_var)), 0.15);
    }

    // ---- ③ 信息形式 ----
    {
        double Om = 1.0 / v1 + 1.0 / v2;   // 更新: Ω = Ω⁻ + 1/r
        chk("信息形式 Ω=Σ⁻¹", fabs(Om - 1.0 / var_prod), 1e-12);
        printf("  [demo] Ω_prior=%.4f Ω_obs=%.4f Ω_post=%.4f\n",
               1.0 / v1, 1.0 / v2, Om);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
