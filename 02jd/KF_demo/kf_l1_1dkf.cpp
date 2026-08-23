// ============================================================
// KF_demo L1.1 一维卡尔曼滤波：五公式与最小例子
// 纯 C++17 标准库。合成数据 + 真值对拍 + PASS/FAIL。
// 验证：① q=0 时 KF 递推 == 批量贝叶斯(WLS) 精确等价
//       ② q>0 随机游走真值下 滤波 RMSE < 原始观测 RMSE
// 数学已用受管 Python 3.13.12 对拍验证（详见 README）。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static const double PI = 3.14159265358979323846;
static unsigned long long _seed = 42;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.1 一维卡尔曼滤波：五公式 ===\n");
    printf("  预测: x-=x, p-=p+q\n");
    printf("  更新: K=p-/(p-+r), x=x-+K(z-x-), p=(1-K)p-\n");

    // ---- ① q=0：与批量贝叶斯精确等价 ----
    {
        double mu0 = 2.0, p0 = 4.0, r = 0.25, xtrue = 1.0;
        int N = 50;
        vector<double> zs(N);
        double sz = 0;
        for (int i = 0; i < N; ++i) { zs[i] = xtrue + gauss() * sqrt(r); sz += zs[i]; }
        // 批量：后验精度 = 1/p0 + N/r；均值 = (mu0/p0 + sum(z)/r) / precision
        double prec = 1.0 / p0 + N / r;
        double mu = (mu0 / p0 + sz / r) / prec;
        double var = 1.0 / prec;
        // KF 递推（q=0 → 预测步不变）
        double x = mu0, P = p0;
        for (int i = 0; i < N; ++i) {
            double K = P / (P + r);
            x = x + K * (zs[i] - x);
            P = (1.0 - K) * P;
        }
        chk("1D-KF vs 批量贝叶斯 均值", fabs(x - mu), 1e-10);
        chk("1D-KF vs 批量贝叶斯 方差", fabs(P - var), 1e-10);
    }

    // ---- ② q>0 随机游走：滤波 RMSE < 观测 RMSE ----
    {
        double q = 0.05, r = 0.25;
        double xt = 1.0, x = 1.0, P = 1.0;
        double e2f = 0, e2z = 0;
        int M = 200;
        for (int k = 0; k < M; ++k) {
            xt += gauss() * sqrt(q);
            double z = xt + gauss() * sqrt(r);
            double Pm = P + q;
            double K = Pm / (Pm + r);
            x = x + K * (z - x);
            P = (1.0 - K) * Pm;
            e2f += (x - xt) * (x - xt);
            e2z += (z - xt) * (z - xt);
        }
        double rmse_f = sqrt(e2f / M), rmse_z = sqrt(e2z / M);
        printf("  [demo] rmse_filter=%.4f  rmse_meas=%.4f\n", rmse_f, rmse_z);
        chk("滤波 RMSE < 观测 RMSE", rmse_f < rmse_z ? 0.0 : 1.0, 0.5);
    }

    // ---- ③ K 的收敛轨迹（直观感受） ----
    {
        double q = 0.05, r = 0.25, P = 1.0;
        printf("  [demo] K 的轨迹: ");
        for (int k = 0; k < 12; ++k) {
            double Pm = P + q;
            double K = Pm / (Pm + r);
            if (k < 6) printf("K%d=%.3f ", k, K);
            P = (1.0 - K) * Pm;
        }
        double Pm = P + q, Kinf = Pm / (Pm + r);
        printf("... K_inf=%.4f\n", Kinf);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
