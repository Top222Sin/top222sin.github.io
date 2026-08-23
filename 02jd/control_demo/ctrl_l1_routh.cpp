// ============================================================
// control_demo L1.2 劳斯判据：不解特征方程判稳定
// 纯 C++17。劳斯表第一列符号变化次数 = 右半平面根数。
// 验证：三个三次多项式，劳斯判据 vs 牛顿法+综合除法实根数
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static int routh(const vector<double>& c) {
    int n = c.size() - 1;
    vector<double> r1, r2;
    for (size_t i = 0; i < c.size(); i += 2) r1.push_back(c[i]);
    for (size_t i = 1; i < c.size(); i += 2) r2.push_back(c[i]);
    if (r2.size() < r1.size()) r2.push_back(0.0);
    vector<vector<double>> T = {r1, r2};
    for (int i = 2; i <= n; ++i) {
        vector<double> prev = T[i - 1], prev2 = T[i - 2];
        int m = prev2.size();
        while ((int)prev.size() < m) prev.push_back(0.0);
        vector<double> row;
        for (int j = 0; j < m - 1; ++j) {
            double a = prev2[j] * prev[j + 1] - prev2[j + 1] * prev[j];
            row.push_back(fabs(prev[0]) > 1e-12 ? -a / prev[0] : 0.0);
        }
        if (row.empty()) break;
        T.push_back(row);
    }
    int signs = 0;
    for (size_t i = 1; i < T.size(); ++i)
        if (T[i - 1][0] * T[i][0] < 0) ++signs;
    return signs;
}

// 三次方程实根个数(正实部) —— 牛顿 + 综合除法
static int rhp_roots3(double a3, double a2, double a1, double a0) {
    auto f = [&](double s) { return ((a3 * s + a2) * s + a1) * s + a0; };
    auto df = [&](double s) { return (3 * a3 * s + 2 * a2) * s + a1; };
    vector<double> found;
    for (int g = -40; g <= 40; ++g) {
        double s = g * 0.5;
        for (int it = 0; it < 80; ++it) {
            double d = df(s);
            if (fabs(d) < 1e-12) break;
            double sn = s - f(s) / d;
            if (fabs(sn - s) < 1e-13) { s = sn; break; }
            s = sn;
        }
        if (fabs(f(s)) < 1e-7) {
            bool dup = false;
            for (double r : found) if (fabs(r - s) < 1e-5) dup = true;
            if (!dup) found.push_back(s);
        }
    }
    int cnt = 0;
    for (double r : found) if (r > 0) ++cnt;
    // 复根对: 若实根数为 1, 剩下一对共轭复根, 实部 = -(a2+a3*r0)/a3 /2
    if (found.size() == 1) {
        double r0 = found[0];
        double q2 = a3, q1 = a2 + q2 * r0;
        // s² + (q1/q2)s + (q0/q2); q0 = a1 + q1*r0
        double q0 = a1 + q1 * r0;
        double re = -q1 / (2 * q2);
        if (re > 0) cnt += 2;
    }
    return cnt;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.2 劳斯判据 ===\n");

    struct Case { double c[4]; const char* label; };
    Case cases[] = {
        {{1, 2, 3, 4}, "s³+2s²+3s+4 (稳定, 0 右根)"},
        {{1, 1, 2, 8}, "s³+s²+2s+8 (2 个右根)"},
        {{1, 6, 11, 6}, "s³+6s²+11s+6 (稳定)"},
    };
    for (auto& cs : cases) {
        int routh_cnt = routh({cs.c[0], cs.c[1], cs.c[2], cs.c[3]});
        int true_cnt = rhp_roots3(cs.c[0], cs.c[1], cs.c[2], cs.c[3]);
        printf("  [demo] %s: 劳斯=%d 数值=%d\n", cs.label, routh_cnt, true_cnt);
        chk(cs.label, abs(routh_cnt - true_cnt), 0.5);
    }
    printf("  [结论] 第一列符号变化次数 = 右半平面极点数, 无需求解特征方程。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
