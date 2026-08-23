// ============================================================
// KF_demo L3.6 扩展：MSCKF 多状态约束滤波（视觉惯性里程计核心）
// 纯 C++17。MSCKF 思想（miniature，2D bearing-only）：
//   ① 滑窗克隆 K 个位姿进状态（EKF-SLAM 把路标放状态，MSCKF 不放！）
//   ② 特征点只用粗略估计（三角化），其误差 δf 通过【左零空间投影】消掉
//   ③ A·Hf = 0 → A·r 是纯位姿约束，对特征点估计一阶不敏感
// 验证：
//  ① A·Hf = 0（零空间正交性 ~1e-16）
//  ② A·r 对 δf 扰动一阶不变（~1e-12）
//  ③ 窗口位姿经 MSCKF 约束修正后误差下降
// 这就是 AR 眼镜/无人机 VIO 的核心技巧：计算量与特征数解耦。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

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

static Mat eye(int n) { Mat I(n, Vec(n, 0.0)); for (int i = 0; i < n; ++i) I[i][i] = 1.0; return I; }
static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, Vec(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}
static Mat transpose(const Mat& A) {
    int n = A.size(), m = A[0].size(); Mat T(m, Vec(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) T[j][i] = A[i][j];
    return T;
}
static Mat addm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j];
    return C;
}
static Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}
static Mat inv(const Mat& A) {
    int n = A.size();
    Mat M(n, Vec(2 * n, 0.0));
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) M[i][j] = A[i][j]; M[i][n + i] = 1.0; }
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int i = c; i < n; ++i) if (fabs(M[i][c]) > fabs(M[p][c])) p = i;
        swap(M[c], M[p]);
        double piv = M[c][c];
        for (int j = 0; j < 2 * n; ++j) M[c][j] /= piv;
        for (int i = 0; i < n; ++i) if (i != c && M[i][c] != 0.0) {
            double f = M[i][c];
            for (int j = 0; j < 2 * n; ++j) M[i][j] -= f * M[c][j];
        }
    }
    Mat R(n, Vec(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) R[i][j] = M[i][n + j];
    return R;
}
static double wrapPi(double a) { return atan2(sin(a), cos(a)); }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.6 MSCKF：零空间投影的多状态约束 ===\n");

    const int K = 5;                       // 窗口位姿数
    double fx = 5.0, fy = 3.0;             // 特征点真值
    double r_brg = 0.005;                  // 方位噪声方差
    double sig = 0.08;                     // 里程计噪声

    // 真值位姿（弧线）与带噪里程计估计
    vector<Vec> poses_true, poses_est;
    for (int i = 0; i < K; ++i)
        poses_true.push_back({i * 1.0, 0.5 * i * 0.3, 0.2 * i * 0.1});
    poses_est.push_back(poses_true[0]);
    for (int i = 1; i < K; ++i)
        poses_est.push_back({poses_est[i-1][0] + (poses_true[i][0] - poses_true[i-1][0]) + gauss() * sig,
                             poses_est[i-1][1] + (poses_true[i][1] - poses_true[i-1][1]) + gauss() * sig,
                             poses_est[i-1][2] + (poses_true[i][2] - poses_true[i-1][2]) + gauss() * sig});
    // 观测
    auto h_bear = [](const Vec& p, double fx2, double fy2) {
        return atan2(fy2 - p[1], fx2 - p[0]) - p[2];
    };
    Vec zs(K);
    for (int i = 0; i < K; ++i)
        zs[i] = h_bear(poses_true[i], fx, fy) + gauss() * sqrt(r_brg);
    // 特征点粗估（模拟三角化，带误差）
    double fe_x = fx + 0.15, fe_y = fy - 0.10;

    // 残差与雅可比
    auto build = [&](double fex, double fey, Mat& Hx, Mat& Hf, Vec& r) {
        Hx = Mat(K, Vec(3 * K, 0.0));
        Hf = Mat(K, Vec(2, 0.0));
        r.assign(K, 0.0);
        for (int i = 0; i < K; ++i) {
            const Vec& p = poses_est[i];
            double dx = fex - p[0], dy = fey - p[1];
            double rr = hypot(dx, dy), r2 = rr * rr;
            Hf[i][0] = -dy / r2;
            Hf[i][1] = dx / r2;
            Hx[i][3 * i + 0] = dy / r2;
            Hx[i][3 * i + 1] = -dx / r2;
            Hx[i][3 * i + 2] = -1.0;
            r[i] = wrapPi(zs[i] - h_bear(p, fex, fey));
        }
    };
    Mat Hx, Hf; Vec r;
    build(fe_x, fe_y, Hx, Hf, r);

    // ---- 左零空间 A：Gram-Schmidt 求 Hf 列空间的正交补 ----
    vector<Vec> cols = {{Hf[0][0], Hf[1][0], Hf[2][0], Hf[3][0], Hf[4][0]},
                        {Hf[0][1], Hf[1][1], Hf[2][1], Hf[3][1], Hf[4][1]}};
    vector<Vec> basis;
    for (auto& c : cols) {
        Vec v = c;
        for (auto& b : basis) {
            double d = 0; for (int i = 0; i < K; ++i) d += v[i] * b[i];
            for (int i = 0; i < K; ++i) v[i] -= d * b[i];
        }
        double nrm = 0; for (int i = 0; i < K; ++i) nrm += v[i] * v[i];
        nrm = sqrt(nrm);
        if (nrm > 1e-10) { for (int i = 0; i < K; ++i) v[i] /= nrm; basis.push_back(v); }
    }
    vector<Vec> full = basis;
    for (int e = 0; e < K; ++e) {
        Vec v(K, 0.0); v[e] = 1.0;
        for (auto& b : full) {
            double d = 0; for (int i = 0; i < K; ++i) d += v[i] * b[i];
            for (int i = 0; i < K; ++i) v[i] -= d * b[i];
        }
        double nrm = 0; for (int i = 0; i < K; ++i) nrm += v[i] * v[i];
        nrm = sqrt(nrm);
        if (nrm > 1e-8) { for (int i = 0; i < K; ++i) v[i] /= nrm; full.push_back(v); }
    }
    Mat A(full.begin() + basis.size(), full.end());   // (K-2) x K
    // ① A·Hf = 0
    Mat AHf = matmul(A, Hf);
    double e1 = 0;
    for (auto& row : AHf) for (double v : row) e1 = fmax(e1, fabs(v));
    chk("A·Hf = 0 (零空间)", e1, 1e-10);
    // ② A·r 对 δf 一阶不变
    Mat Hx2, Hf2; Vec r2_;
    build(fe_x + 1e-5, fe_y + 1e-5, Hx2, Hf2, r2_);
    Vec Ar1 = matvec(A, r), Ar2 = matvec(A, r2_);
    double dAr = 0, dr = 0;
    for (int i = 0; i < (int)Ar1.size(); ++i) dAr = fmax(dAr, fabs(Ar1[i] - Ar2[i]));
    for (int i = 0; i < K; ++i) dr = fmax(dr, fabs(r2_[i] - r[i]));
    chk("A·r 对 δf 一阶不变", dAr, fmax(dr * 1e-2, 1e-9));
    // ③ MSCKF 更新
    Mat P = eye(3 * K);
    for (int i = 0; i < 3 * K; ++i) P[i][i] = sig * sig * (1 + i / 3);
    Mat Rm = eye(K);
    for (int i = 0; i < K; ++i) Rm[i][i] = r_brg;
    Vec Ar = matvec(A, r);
    Mat AHx = matmul(A, Hx);
    Mat S = addm(matmul(matmul(AHx, P), transpose(AHx)), matmul(matmul(A, Rm), transpose(A)));
    Mat Km = matmul(matmul(P, transpose(AHx)), inv(S));
    Vec dxv = matvec(Km, Ar);
    vector<Vec> poses_upd(K);
    for (int i = 0; i < K; ++i)
        poses_upd[i] = {poses_est[i][0] + dxv[3 * i], poses_est[i][1] + dxv[3 * i + 1], poses_est[i][2] + dxv[3 * i + 2]};
    double e_before = 0, e_after = 0;
    for (int i = 0; i < K; ++i) {
        e_before = fmax(e_before, hypot(poses_est[i][0] - poses_true[i][0], poses_est[i][1] - poses_true[i][1]));
        e_after  = fmax(e_after,  hypot(poses_upd[i][0]  - poses_true[i][0], poses_upd[i][1]  - poses_true[i][1]));
    }
    printf("  [demo] 窗口位姿误差: 里程计=%.4f → MSCKF=%.4f  (A 是 %zux%d)\n",
           e_before, e_after, A.size(), K);
    chk("MSCKF 修正后误差减小", e_after < e_before ? 0.0 : 1.0, 0.5);

    printf("  [结论] 特征点不进状态（对比 EKF-SLAM）：状态只随窗口大小增长，\n");
    printf("         计算量与场景特征数解耦——这是 VIO 能实时跑的根本原因。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
