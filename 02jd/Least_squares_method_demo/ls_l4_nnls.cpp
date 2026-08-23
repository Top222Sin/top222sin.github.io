// ls_l4_nnls.cpp —— L4.8 非负最小二乘（NNLS）
// 纯 C++17 标准库实现；与受管 Python 验证同算法、同确定性 PRNG（LCG seed=20260820）
//
// 核心：NNLS = min ||A x - b||^2  s.t. x >= 0。
//   - 这是「带约束的 LS」：把无约束 LS 投影到 x>=0 锥上。
//   - 本项目用【投影梯度下降（PGD）】：x <- max(0, x - alpha * grad)，
//     其中 grad = 2 Aᵀ(Ax - b)，步长 alpha = 1/(2*lambda_max(AᵀA))。
//   - PGD 在凸二次问题上必然收敛到真约束最优；收敛点天然满足 KKT 互补松弛。
//   - 验证用 KKT 条件（不依赖外部求解器）：活动分量梯度<=0、被动分量梯度~0。
//
// 四件套：合成数据(真值 x_true>=0) + NNLS 求解 + 真值/约束对拍 + PASS/FAIL。

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

using VD = std::vector<double>;
using MD = std::vector<std::vector<double>>;

// ---------- 小矩阵/线性代数 ----------
static MD matmul(const MD& A, const MD& B) {
    int n = (int)A.size(), m = (int)A[0].size(), p = (int)B[0].size();
    MD C(n, VD(p, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) {
            double a = A[i][k]; if (a == 0.0) continue;
            for (int j = 0; j < p; ++j) C[i][j] += a * B[k][j];
        }
    return C;
}
static VD matvec(const MD& A, const VD& x) {
    int n = (int)A.size(), m = (int)A[0].size();
    VD y(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) y[i] += A[i][j] * x[j];
    return y;
}
static MD transpose(const MD& A) {
    int n = (int)A.size(), m = (int)A[0].size();
    MD T(m, VD(n, 0.0));
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) T[j][i] = A[i][j];
    return T;
}
// 高斯消元 + 部分主元，解方阵 A x = b（仅用于无约束 LS 基线）
static VD solve(const MD& A, const VD& b) {
    int n = (int)A.size();
    MD M(n, VD(n + 1, 0.0));
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) M[i][j] = A[i][j]; M[i][n] = b[i]; }
    for (int c = 0; c < n; ++c) {
        int piv = c; double best = std::fabs(M[c][c]);
        for (int r = c + 1; r < n; ++r) { double v = std::fabs(M[r][c]); if (v > best) { best = v; piv = r; } }
        std::swap(M[c], M[piv]);
        double d = M[c][c];
        for (int j = c; j <= n; ++j) M[c][j] /= d;
        for (int r = 0; r < n; ++r) if (r != c && M[r][c] != 0.0) {
            double f = M[r][c];
            for (int j = c; j <= n; ++j) M[r][j] -= f * M[c][j];
        }
    }
    VD x(n); for (int i = 0; i < n; ++i) x[i] = M[i][n];
    return x;
}

// ---------- 确定性 PRNG（LCG，与 Python 一致） ----------
static uint32_t g_lcg = 20260820u;
static double rnd() {
    g_lcg = g_lcg * 1103515245u + 12345u;
    return (double)(g_lcg & 0x7fffffff) / 2147483647.0;
}
static MD make_matrix(int m, int n) {
    MD A(m, VD(n, 0.0));
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) A[i][j] = 2.0 * rnd() - 1.0;
    return A;
}

// ---------- lambda_max(AᵀA) 幂迭代（确定性初值） ----------
static double lambda_max_ATA(const MD& A) {
    int n = (int)A[0].size();
    MD AtA = matmul(transpose(A), A);
    VD v(n, 1.0 / std::sqrt((double)n));
    for (int it = 0; it < 80; ++it) {
        VD w = matvec(AtA, v);
        double nv = 0.0; for (double x : w) nv += x * x;
        nv = std::sqrt(nv); if (nv == 0.0) break;
        for (int i = 0; i < n; ++i) v[i] = w[i] / nv;
    }
    VD w = matvec(AtA, v);
    double s = 0.0; for (int i = 0; i < n; ++i) s += v[i] * w[i];
    return s;
}

// ---------- NNLS：投影梯度下降 ----------
static VD nnls_pgd(const MD& A, const VD& b, int iters = 20000) {
    int m = (int)A.size(), n = (int)A[0].size();
    double Lsmooth = 2.0 * lambda_max_ATA(A);
    double alpha = 1.0 / Lsmooth;
    VD x(n, 0.0);
    for (int it = 0; it < iters; ++it) {
        VD Ax = matvec(A, x);
        VD g(m);
        for (int i = 0; i < m; ++i) g[i] = 2.0 * (Ax[i] - b[i]);   // ∇f = 2Aᵀ(Ax-b)
        VD grad = matvec(transpose(A), g);
        for (int j = 0; j < n; ++j) x[j] = std::max(0.0, x[j] - alpha * grad[j]);
    }
    return x;
}
static VD lstsq_unconstrained(const MD& A, const VD& b) {
    return solve(matmul(transpose(A), A), matvec(transpose(A), b));
}
static double obj(const MD& A, const VD& b, const VD& x) {
    double s = 0.0;
    for (int i = 0; i < (int)A.size(); ++i) {
        double r = b[i]; for (int k = 0; k < (int)A[0].size(); ++k) r -= A[i][k] * x[k];
        s += r * r;
    }
    return s;
}

// ---------- 检查器 ----------
static int g_pass = 0, g_fail = 0;
static void chk(const char* name, double got, double thr, bool ge, const char* extra) {
    bool ok = ge ? (got >= thr - 1e-12) : (std::fabs(got) <= thr + 1e-12);
    printf("  [%s] %s: %.3e (thr=%+.1e) %s\n", ok ? "PASS" : "FAIL", name, ge ? got : std::fabs(got), thr, extra);
    if (ok) ++g_pass; else ++g_fail;
}

int main() {
    int m = 12, n = 7;
    MD A = make_matrix(m, n);
    VD x_true(n, 0.0);
    int posc[] = { 0, 1, 3, 4, 6 };
    for (int q = 0; q < 5; ++q) x_true[posc[q]] = 0.5 + rnd();
    VD b = matvec(A, x_true);
    for (int i = 0; i < m; ++i) b[i] += 0.001 * (2.0 * rnd() - 1.0);

    VD x_est = nnls_pgd(A, b);
    VD x_ls = lstsq_unconstrained(A, b);
    VD x_clamped(n); for (int j = 0; j < n; ++j) x_clamped[j] = std::max(0.0, x_ls[j]);

    // s = Aᵀ(b - A x_est)
    VD r = b; VD Ax = matvec(A, x_est);
    for (int i = 0; i < m; ++i) r[i] -= Ax[i];
    VD s = matvec(transpose(A), r);

    printf("== L4.8 NNLS 验证 ==\n");
    // ① 可行性
    double xmin = x_est[0]; for (double v : x_est) xmin = std::min(xmin, v);
    char ex1[64]; snprintf(ex1, sizeof(ex1), "min=%.3e", xmin);
    chk("可行性 x>=0", xmin, 1e-9, false, ex1);

    // ② 恢复误差（相对）
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) { num += (x_est[j] - x_true[j]) * (x_est[j] - x_true[j]); den += x_true[j] * x_true[j]; }
    double rec = std::sqrt(num) / (std::sqrt(den) + 1.0);
    char ex2[64]; snprintf(ex2, sizeof(ex2), "rel=%.3e", rec);
    chk("恢复误差(相对)", rec, 1e-2, false, ex2);

    // ③ KKT 互补松弛
    double worst = 0.0;
    for (int j = 0; j < n; ++j) {
        if (x_est[j] <= 1e-6) worst = std::max(worst, s[j]);        // 活动：需 s_j <= 0
        else worst = std::max(worst, std::fabs(s[j]));              // 被动：需 s_j ~ 0
    }
    char ex3[64]; snprintf(ex3, sizeof(ex3), "maxviol=%.3e", worst);
    chk("KKT互补松弛", worst, 1e-5, false, ex3);

    // ④ 残差不劣于无约束 LS：f_nnls >= f_ls - 1e-8
    double d4 = obj(A, b, x_est) - obj(A, b, x_ls);
    char ex4[64]; snprintf(ex4, sizeof(ex4), "d=%+.3e", d4);
    chk("残差>=LS残差", d4, -1e-8, true, ex4);

    // ⑤ 优于朴素截断 LS：f_nnls <= f_clamped + 1e-7（单侧上界）
    double d5 = obj(A, b, x_est) - obj(A, b, x_clamped);
    char ex5[64]; snprintf(ex5, sizeof(ex5), "d=%+.3e", d5);
    chk("优于截断LS(单侧上界)", std::max(0.0, d5 - 1e-7), 1e-12, false, ex5);

    printf("PASS=%d  FAIL=%d  (total=%d)\n", g_pass, g_fail, g_pass + g_fail);
    printf("RESULT: %s\n", g_fail == 0 ? "ALL PASS" : "HAS FAIL");
    return 0;
}
