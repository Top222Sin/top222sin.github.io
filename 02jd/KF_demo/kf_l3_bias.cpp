// ============================================================
// KF_demo L3.2 状态增广：传感器零偏估计与可观测性
// 纯 C++17。增广状态 x=[p, v, b]（加计零偏）：
//   系统可观测 ⇔ rank(O) = n，O = [H; HF; HF²; ...]
// 验证：
//  ① 只测位置时 [p,v,b] 可观测（rank=3）：b 通过 v 的残差泄漏被估出
//  ② 反例：只测 vx（4D 状态 [px,py,vx,vy]）rank=1 < 4：
//     位置/另一轴速度完全不可观测
// 数值 rank 用带主元的行消元（阈值 1e-8）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}

// 数值秩（高斯消元 + 主元阈值）
static int rank_of(Mat A, double tol = 1e-8) {
    int m = A.size(), n = A[0].size(), r = 0;
    for (int c = 0; c < n; ++c) {
        int piv = -1;
        for (int i = r; i < m; ++i) if (fabs(A[i][c]) > tol) { piv = i; break; }
        if (piv < 0) continue;
        swap(A[r], A[piv]);
        for (int i = 0; i < m; ++i) if (i != r && A[i][c] != 0.0) {
            double f = A[i][c] / A[r][c];
            for (int j = 0; j < n; ++j) A[i][j] -= f * A[r][j];
        }
        ++r;
    }
    return r;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.2 状态增广与可观测性 ===\n");

    // ---- ① [p, v, b] 增广，只测位置 ----
    double dt = 0.1;
    Mat F = {{1.0, dt, -0.5 * dt * dt},
             {0.0, 1.0, -dt},
             {0.0, 0.0, 1.0}};
    Mat H = {{1.0, 0.0, 0.0}};
    Mat O;
    Mat Hk = H;
    for (int i = 0; i < 3; ++i) { O.push_back(Hk[0]); Hk = matmul(Hk, F); }
    printf("  [demo] 可观测性矩阵 O (H; HF; HF²):\n");
    for (auto& row : O)
        printf("        [ %8.4f %8.4f %8.4f ]\n", row[0], row[1], row[2]);
    int rk = rank_of(O);
    printf("  [demo] rank(O) = %d / 3\n", rk);
    chk("位置观测 ⇒ [p,v,b] 可观测", rk == 3 ? 0.0 : 1.0, 0.5);

    // ---- ② 反例：4D [px,py,vx,vy]，只测 vx ----
    Mat F2 = {{1, 0, dt, 0}, {0, 1, 0, dt}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Mat H2 = {{0, 0, 1, 0}};
    Mat O2;
    Mat Hk2 = H2;
    for (int i = 0; i < 4; ++i) { O2.push_back(Hk2[0]); Hk2 = matmul(Hk2, F2); }
    int rk2 = rank_of(O2);
    printf("  [demo] 反例 rank = %d / 4（只测 vx：py/vy 不可观测）\n", rk2);
    chk("反例: 只测vx ⇒ rank<4", rk2 < 4 ? 0.0 : 1.0, 0.5);

    printf("  [结论] 不可观测的方向上 P 不会收敛——增广零偏前先检查 rank(O)；\n");
    printf("         若不可观测，需增加激励（如加计转向机动）或删掉该状态。\n");

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
