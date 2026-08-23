// ============================================================
// KF_demo L3.3 粒子滤波 PF：与 KF 的分界线 ⭐
// 纯 C++17。SIR（Sampling Importance Resampling）+ 系统重采样。
// 验证（同一条噪声序列，公平对拍）：
//  ① 单峰线性高斯：PF(N=2000) ≈ KF（RMSE 比 < 1.3）——KF 的主场
//  ② 30% 野值(+3) 造成双峰似然：PF 明显优于 KF（KF 被拉偏）
// 结论：高斯+线性/轻度非线性 → KF/EKF/UKF；多峰/强非线性/野值 → PF
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

static const int NP = 2000;

static double run_kf(const vector<double>& zs, const vector<double>& xt,
                     double q, double r) {
    double x = 0.0, P = 1.0, e2 = 0;
    for (size_t i = 0; i < zs.size(); ++i) {
        double Pm = P + q;
        double K = Pm / (Pm + r);
        x = x + K * (zs[i] - x);
        P = (1.0 - K) * Pm;
        e2 += (x - xt[i]) * (x - xt[i]);
    }
    return sqrt(e2 / zs.size());
}

// 似然函数：mode=0 单峰 N(z; p, r)；mode=1 双峰 0.7·N(z;p,r)+0.3·N(z;p+3,r)
static double lik(double z, double p, double r, int mode) {
    double l1 = exp(-0.5 * (z - p) * (z - p) / r);
    if (mode == 0) return l1;
    double l2 = exp(-0.5 * (z - p - 3.0) * (z - p - 3.0) / r);
    return 0.7 * l1 + 0.3 * l2;
}

static double run_pf(const vector<double>& zs, const vector<double>& xt,
                     double q, double r, int mode, unsigned long long seed) {
    set_seed(seed);
    vector<double> parts(NP);
    for (int i = 0; i < NP; ++i) parts[i] = gauss();
    vector<double> ws(NP), cum(NP);
    double e2 = 0;
    for (size_t t = 0; t < zs.size(); ++t) {
        for (int i = 0; i < NP; ++i) parts[i] += gauss() * sqrt(q);
        for (int i = 0; i < NP; ++i) ws[i] = lik(zs[t], parts[i], r, mode);
        double sw = 0; for (int i = 0; i < NP; ++i) sw += ws[i];
        if (sw < 1e-300) sw = 1e-300;
        for (int i = 0; i < NP; ++i) ws[i] /= sw;
        double xm = 0; for (int i = 0; i < NP; ++i) xm += parts[i] * ws[i];
        e2 += (xm - xt[t]) * (xm - xt[t]);
        double neff = 0; for (int i = 0; i < NP; ++i) neff += ws[i] * ws[i];
        neff = 1.0 / neff;
        if (neff < NP / 2.0) {
            // 系统重采样
            double c = 0; cum.clear();
            for (int i = 0; i < NP; ++i) { c += ws[i]; cum.push_back(c); }
            vector<double> newp(NP);
            double u0 = urand() / NP;
            int idx = 0;
            for (int j = 0; j < NP; ++j) {
                double u = u0 + (double)j / NP;
                while (idx < NP - 1 && cum[idx] < u) ++idx;
                newp[j] = parts[idx];
            }
            parts.swap(newp);
        }
    }
    return sqrt(e2 / zs.size());
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.3 粒子滤波 PF vs KF ===\n");

    double q = 0.1, r = 0.25;
    int N = 40;

    // ---- ① 单峰：生成共享噪声序列 ----
    vector<double> zs(N), xt(N);
    set_seed(2024);
    {
        double x = 0.0;
        for (int k = 0; k < N; ++k) {
            x += gauss() * sqrt(q);
            xt[k] = x;
            zs[k] = x + gauss() * sqrt(r);
        }
    }
    double rmse_kf1 = run_kf(zs, xt, q, r);
    double rmse_pf1 = run_pf(zs, xt, q, r, 0, 31415);
    printf("  [demo] 单峰:  PF=%.4f  KF=%.4f  (PF 略逊但同量级)\n", rmse_pf1, rmse_kf1);
    chk("单峰: PF ≈ KF (比<1.3)", rmse_pf1 < 1.3 * rmse_kf1 ? 0.0 : 1.0, 0.5);

    // ---- ② 双峰（30% 野值 +3）----
    vector<double> zs2(N), xt2(N);
    set_seed(5566);
    {
        double x = 0.0;
        for (int k = 0; k < N; ++k) {
            x += gauss() * sqrt(q);
            double z = x + gauss() * sqrt(r);
            if (urand() < 0.3) z += 3.0;   // 野值 → 双峰似然
            xt2[k] = x; zs2[k] = z;
        }
    }
    double rmse_kf2 = run_kf(zs2, xt2, q, r);
    double rmse_pf2 = run_pf(zs2, xt2, q, r, 1, 2718);
    printf("  [demo] 双峰:  PF=%.4f  KF=%.4f  (KF 被野值持续拉偏)\n", rmse_pf2, rmse_kf2);
    chk("多峰: PF < KF", rmse_pf2 < rmse_kf2 ? 0.0 : 1.0, 0.5);

    printf("  [结论] KF 假设单峰高斯 → 野值/多峰下被拉偏；PF 用粒子表达任意分布，\n");
    printf("         代价是 O(N) 计算与维度灾难（状态维数 >8 维后粒子数爆炸）。\n");

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
