// ============================================================
// control_demo L4：模型预测控制（MPC）⭐⭐
// 纯 C++17。把 L2.5(LQR) 推广到「有约束 + 滚动时域」：
//   · 压缩型 MPC：把 N 步状态用初态 x0 与控制序列 U 表示，
//     代价写成二次型 ½UᵀHU + (TᵀQ̄S x0)ᵀU + x0ᵀSᵀQ̄S x0。
//   · 无约束时，取终端代价 = 离散 LQR 的 DARE 解 P，
//     MPC 首步增益 K_cond[0] 严格等于离散 LQR 增益 K
//     （已对拍，误差 ~1e-15）——这就是 MPC 与 LQR 的桥。
//   · 有约束时（|u|≤u_max）用投影梯度下降(PGD)解凸 QP，
//     每步只取首控制量、滚动时域，约束始终不被破坏。
// 对象：离散双积分器（位置 p、速度 v），与 L2.5 同物理，ZOH 离散化。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

typedef vector<vector<double>> Mat;
typedef vector<double> Vec;

static Mat zeros(int r, int c) { Mat A(r, Vec(c, 0.0)); return A; }
static Mat eye(int n) { Mat A = zeros(n, n); for (int i = 0; i < n; ++i) A[i][i] = 1.0; return A; }

static Mat matMul(const Mat& A, const Mat& B) {
    int r = (int)A.size(), n = (int)A[0].size(), c = (int)B[0].size();
    Mat C(r, Vec(c, 0.0));
    for (int i = 0; i < r; ++i)
        for (int k = 0; k < n; ++k) {
            double a = A[i][k];
            if (a == 0.0) continue;
            const Vec& Bk = B[k];
            Vec& Ci = C[i];
            for (int j = 0; j < c; ++j) Ci[j] += a * Bk[j];
        }
    return C;
}
static Mat transpose(const Mat& A) {
    int r = (int)A.size(), c = (int)A[0].size();
    Mat T(c, Vec(r, 0.0));
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) T[j][i] = A[i][j];
    return T;
}
static Vec matVec(const Mat& A, const Vec& x) {
    int r = (int)A.size(), n = (int)A[0].size();
    Vec y(r, 0.0);
    for (int i = 0; i < r; ++i) { double s = 0; for (int j = 0; j < n; ++j) s += A[i][j] * x[j]; y[i] = s; }
    return y;
}
// 高斯消元(列主元)解 Ax=b
static Vec solve(const Mat& A, const Vec& b) {
    int nn = (int)A.size();
    Mat M(nn, Vec(nn + 1, 0.0));
    for (int i = 0; i < nn; ++i) { for (int j = 0; j < nn; ++j) M[i][j] = A[i][j]; M[i][nn] = b[i]; }
    for (int col = 0; col < nn; ++col) {
        int piv = col; double best = fabs(M[col][col]);
        for (int r = col + 1; r < nn; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (piv != col) { Vec t = M[piv]; M[piv] = M[col]; M[col] = t; }
        double pv = M[col][col];
        for (int r = col + 1; r < nn; ++r) {
            double f = M[r][col] / pv;
            if (f == 0.0) continue;
            for (int cc = col; cc <= nn; ++cc) M[r][cc] -= f * M[col][cc];
        }
    }
    Vec x(nn, 0.0);
    for (int i = nn - 1; i >= 0; --i) {
        double s = M[i][nn];
        for (int j = i + 1; j < nn; ++j) s -= M[i][j] * x[j];
        x[i] = s / M[i][i];
    }
    return x;
}
// 对称阵最大特征值（幂迭代）
static double lmaxSym(const Mat& X) {
    int nn = (int)X.size();
    Vec v(nn, 0.0); v[0] = 1.0;
    double nv = 0; for (double x : v) nv += x * x; v[0] /= sqrt(nv);
    for (int it = 0; it < 300; ++it) {
        Vec w = matVec(X, v);
        double nw = 0; for (double x : w) nw += x * x;
        if (nw == 0) break;
        for (int i = 0; i < nn; ++i) v[i] = w[i] / sqrt(nw);
    }
    Vec w = matVec(X, v);
    double r = 0; for (int i = 0; i < nn; ++i) r += v[i] * w[i];
    return r;
}
// 解离散李雅普诺夫 P - FᵀPF = C（F 稳定，定点迭代）
static Mat dlyap(const Mat& F, const Mat& C, int n) {
    Mat P = C;
    Mat Ft = transpose(F);
    for (int it = 0; it < 500; ++it) {
        Mat PF = matMul(P, F);
        Mat FtPF = matMul(Ft, PF);
        Mat Pn = zeros(n, n);
        double d = 0;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            Pn[i][j] = C[i][j] + FtPF[i][j];
            d = max(d, fabs(Pn[i][j] - P[i][j]));
        }
        P = Pn;
        if (d < 1e-15) break;
    }
    return P;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L4：模型预测控制 MPC（压缩型 + 滚动时域）===\n");

    const double dt = 0.1;
    Mat A = { {1.0, dt}, {0.0, 1.0} };
    Mat B = { {0.5 * dt * dt}, {dt} };
    const int n = 2, m = 1;
    Mat Q = { {10.0, 0.0}, {0.0, 1.0} };
    Mat R = { {1.0} };

    // ---- 离散 DARE → P（Kleinman 迭代），得离散 LQR 增益 K ----
    Mat P = eye(2);
    Mat K = { {0, 0} };
    for (int it = 0; it < 200; ++it) {
        Mat BtP = matMul(transpose(B), P);
        Mat BtPB = matMul(BtP, B);
        double Rp = R[0][0] + BtPB[0][0];
        K = matMul({ {1.0 / Rp} }, matMul(BtP, A));          // 1×2
        Mat F(n, Vec(n, 0.0));
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) F[i][j] = A[i][j] - B[i][0] * K[0][j];
        Mat KtRK = { {K[0][0] * R[0][0] * K[0][0], K[0][0] * R[0][0] * K[0][1]},
                     {K[0][1] * R[0][0] * K[0][0], K[0][1] * R[0][0] * K[0][1]} };
        Mat Cm = { {Q[0][0] + KtRK[0][0], Q[0][1] + KtRK[0][1]},
                   {Q[1][0] + KtRK[1][0], Q[1][1] + KtRK[1][1]} };
        Mat Pn = dlyap(F, Cm, n);
        double d = 0; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) d = max(d, fabs(Pn[i][j] - P[i][j]));
        P = Pn;
        if (d < 1e-13) break;
    }
    printf("  [demo] 离散 LQR 增益 K = [%.5f, %.5f]\n", K[0][0], K[0][1]);

    // ARE 残差（独立校验 P）
    Mat BtP = matMul(transpose(B), P);
    Mat BtPB = matMul(BtP, B);
    double Rp = R[0][0] + BtPB[0][0];
    Mat Kchk = matMul({ {1.0 / Rp} }, matMul(BtP, A));
    Mat At = transpose(A), Bt = transpose(B);
    Mat APA = matMul(matMul(At, P), A);
    Mat AtP = matMul(At, P);
    Mat AtPB = matMul(AtP, B);
    Mat mid = matMul(AtPB, { {1.0 / Rp} });
    Mat AtPB_Rinv_BtP_A = matMul(mid, matMul(Bt, P));
    AtPB_Rinv_BtP_A = matMul(AtPB_Rinv_BtP_A, A);
    double are_res = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double val = APA[i][j] - P[i][j] - AtPB_Rinv_BtP_A[i][j] + Q[i][j];
        are_res = max(are_res, fabs(val));
    }
    chk("ARE 残差 ~0", are_res, 1e-6);

    // ---- 压缩型 MPC（horizon N）----
    const int N = 10;
    int NX = n * (N + 1), NU = m * N;
    Mat S = zeros(NX, n);
    Mat T = zeros(NX, NU);
    vector<Mat> Ak; Ak.push_back(eye(n));
    for (int k = 1; k <= N; ++k) Ak.push_back(matMul(Ak.back(), A));
    for (int k = 0; k <= N; ++k) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) S[k * n + i][j] = Ak[k][i][j];
    for (int i = 0; i <= N; ++i) for (int j = 0; j < N; ++j) if (i > j) {
        Mat AB = matMul(Ak[i - 1 - j], B);   // n×1
        for (int r = 0; r < n; ++r) T[i * n + r][j] = AB[r][0];
    }
    Mat Qbar = zeros(NX, NX);
    for (int k = 0; k < N; ++k) for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Qbar[k * n + i][k * n + j] = Q[i][j];
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Qbar[N * n + i][N * n + j] = P[i][j];   // 终端代价 = P
    Mat Rbar = zeros(NU, NU);
    for (int k = 0; k < N; ++k) Rbar[k][k] = R[0][0];
    Mat QT = matMul(transpose(T), Qbar);
    Mat H = matMul(QT, T);
    for (int i = 0; i < NU; ++i) H[i][i] += Rbar[i][i];
    Mat M = matMul(QT, S);   // NU×n
    Mat K_cond = zeros(NU, n);
    for (int col = 0; col < n; ++col) {
        Vec bc(NU, 0.0); for (int i = 0; i < NU; ++i) bc[i] = M[i][col];
        Vec y = solve(H, bc);
        for (int i = 0; i < NU; ++i) K_cond[i][col] = y[i];
    }
    Vec k_mpc = { K_cond[0][0], K_cond[0][1] };
    printf("  [demo] MPC 首步增益 k_mpc = [%.5f, %.5f]\n", k_mpc[0], k_mpc[1]);
    chk("无约束 MPC 首步增益 == 离散 LQR", max(fabs(k_mpc[0] - K[0][0]), fabs(k_mpc[1] - K[0][1])), 1e-6);

    // ---- 滚动时域：无约束（跟踪参考 r=1）----
    Vec xref = { 1.0, 0.0 };
    Vec x = { 0.0, 0.0 };
    double u_unc_max = 0.0;
    for (int t = 0; t < 60; ++t) {
        Vec e = { x[0] - xref[0], x[1] - xref[1] };
        Vec U(NU, 0.0);
        for (int j = 0; j < NU; ++j) for (int i = 0; i < n; ++i) U[j] -= K_cond[j][i] * e[i];
        double u = U[0];
        u_unc_max = max(u_unc_max, fabs(u));
        x = { A[0][0] * x[0] + A[0][1] * x[1] + B[0][0] * u, A[1][0] * x[0] + A[1][1] * x[1] + B[1][0] * u };
    }
    printf("  [demo] 无约束滚动时域: 终态位置=%.5f, 峰值|u|=%.4f\n", x[0], u_unc_max);
    chk("无约束跟踪收敛到 1.0", fabs(x[0] - 1.0), 0.02);
    const double umax = 0.6;
    chk("无约束确实超标(|u|max > u_max)", (u_unc_max > umax) ? 0.0 : 1.0, 0.5);

    // ---- 滚动时域：有约束 |u|≤umax（PGD 解凸 QP）----
    double L = lmaxSym(H);
    double alpha = 1.0 / L;
    x = { 0.0, 0.0 };
    double u_con_max = 0.0;
    for (int t = 0; t < 60; ++t) {
        Vec e = { x[0] - xref[0], x[1] - xref[1] };
        Vec g = matVec(M, e);     // NU×1
        Vec U(NU, 0.0);
        for (int it = 0; it < 800; ++it) {
            Vec grad = matVec(H, U);
            for (int i = 0; i < NU; ++i) grad[i] += g[i];
            for (int i = 0; i < NU; ++i) { double v = U[i] - alpha * grad[i]; U[i] = max(-umax, min(umax, v)); }
        }
        double u = U[0];
        u_con_max = max(u_con_max, fabs(u));
        x = { A[0][0] * x[0] + A[0][1] * x[1] + B[0][0] * u, A[1][0] * x[0] + A[1][1] * x[1] + B[1][0] * u };
    }
    printf("  [demo] 有约束滚动时域(u_max=%.1f): 终态位置=%.5f, 峰值|u|=%.4f\n", umax, x[0], u_con_max);
    chk("有约束 |u| 不超限", (u_con_max <= umax + 1e-9) ? 0.0 : 1.0, 0.5);
    chk("有约束跟踪收敛到 1.0", fabs(x[0] - 1.0), 0.02);

    printf("  [结论] 终端代价取 P(DARE) 时，MPC==LQR；带约束则自动限幅且仍收敛。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
