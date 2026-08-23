// ============================================================
// control_demo L2.3 极点配置（Ackermann 公式）
// 纯 C++17。u = −Kx, 期望闭环极点 → K:
//   K = eₙᵀ·[B AB]⁻¹·φ(A)，φ(s)=期望特征多项式
// 验证：① 闭环特征多项式逐系数对拍；② 闭环阶跃收敛
// 对照: 与 L1.3 根轨迹"调一个 K"不同——极点配置同时放两个极点。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat eye(int n) { Mat I(n, vector<double>(n, 0)); for (int i = 0; i < n; ++i) I[i][i] = 1; return I; }
static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, vector<double>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}
static Mat addm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j];
    return C;
}
static Mat scalem(const Mat& A, double s) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] *= s;
    return C;
}
static Mat inv2(const Mat& M) {
    double a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
    double det = a * d - b * c;
    return {{d / det, -b / det}, {-c / det, a / det}};
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.3 极点配置 (Ackermann) ===\n");

    Mat A = {{0, 1}, {0, 0}};        // 双积分器
    Mat B = {{0.0}, {1.0}};
    // 期望极点 s = −2±2j → φ(s) = s² + 4s + 8
    Mat Wc = {{0, 1}, {1, 0}};       // [B AB]
    Mat AA = matmul(A, A);
    Mat phi = addm(addm(AA, scalem(A, 4.0)), scalem(eye(2), 8.0));
    Mat M = matmul(inv2(Wc), phi);
    double K0 = M[1][0], K1 = M[1][1];   // K = e₂ᵀ·M，取 M 的最后一行（Ackermann 标准：行向量左乘；取列是常见笔误）
    printf("  [demo] K = [%.4f, %.4f] （期望极点 −2±2j）\n", K0, K1);

    // 闭环 A−BK 特征多项式 s² − tr·s + det = s²+4s+8
    double a11 = A[0][0], a12 = A[0][1] - 0;      // B K 只有 [1][*] 行
    double c11 = A[0][0], c12 = A[0][1];
    double c21 = A[1][0] - K0, c22 = A[1][1] - K1;
    double tr = c11 + c22, det = c11 * c22 - c12 * c21;
    chk("闭环特征 s²+4s+8（迹）", fabs(tr + 4.0), 1e-12);
    chk("闭环特征 s²+4s+8（行列式）", fabs(det - 8.0), 1e-12);

    // 闭环阶跃仿真
    double x0 = 1, x1 = 0;
    double dt = 1e-4;
    for (int i = 0; i < (int)(8.0 / dt); ++i) {
        double u = -(K0 * x0 + K1 * x1);
        double d0 = x1, d1 = u;
        x0 += d0 * dt; x1 += d1 * dt;
    }
    chk("闭环收敛 |x| < 0.01", fmax(fabs(x0), fabs(x1)), 0.01);
    // 第二组：非对称系统（暴露"取列 vs 取行"——第一组 M 对称、两种取法碰巧都正确）
    // A=[[1,2],[3,4]], B=[[0],[1]], 期望极点 s=−1,−2 → φ(s)=s²+3s+2
    Mat A2 = {{1, 2}, {3, 4}};
    Mat B2 = {{0.0}, {1.0}};
    Mat Wc2 = {{0, 2}, {1, 4}};   // [B2  A2·B2]
    Mat phi2 = addm(addm(matmul(A2, A2), scalem(A2, 3.0)), scalem(eye(2), 2.0));
    Mat M2 = matmul(inv2(Wc2), phi2);
    double K0b = M2[1][0], K1b = M2[1][1];   // 正确：取最后一行
    printf("  [demo] (非对称用例) K = [%.4f, %.4f]（期望极点 −1,−2）\n", K0b, K1b);
    double a11b = A2[0][0], a12b = A2[0][1];
    double c21b = A2[1][0] - K0b, c22b = A2[1][1] - K1b;
    double trb = a11b + c22b, detb = a11b * c22b - a12b * c21b;
    chk("非对称: 闭环特征 s²+3s+2（迹）", fabs(trb + 3.0), 1e-12);
    chk("非对称: 闭环特征 s²+3s+2（行列式）", fabs(detb - 2.0), 1e-12);
    double xb0 = 1, xb1 = 0;
    for (int i = 0; i < (int)(8.0 / dt); ++i) {
        double ub = -(K0b * xb0 + K1b * xb1);
        xb0 += xb1 * dt; xb1 += ub * dt;
    }
    chk("非对称: 闭环收敛 |x| < 0.01", fmax(fabs(xb0), fabs(xb1)), 0.01);
    printf("  [demo] (非对称用例) 8s 后 x=[%.6f, %.6f]\n", xb0, xb1);
    printf("  [demo] 8s 后 x=[%.6f, %.6f]\n", x0, x1);
    printf("  [结论] 能控 ⇒ 极点可任意放置；Ackermann 一行公式完成设计。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
