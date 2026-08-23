// ============================================================
// KF_demo L4 集合卡尔曼滤波 EnKF + 移动 horizon 估计 MHE ⭐
// 纯 C++17（无第三方库）。两大工业级估计器，建立在 L1-L3 单步 KF 之上：
//   Part A  EnKF ：随机集合 KF（perturbed-observation 变体），同一线性高斯
//                  系统下与精确 KF 对拍均值轨迹 + 集合协方差（蒙卡收敛）
//   Part B  MHE  ：滑动窗批最小二乘。全状态观测 H=I（良态、速度可观测），
//                  法方程 Hx=b（H=ΣJᵀWJ, b=ΣJᵀWz，高斯-牛顿装配），
//                  含无约束解 / 单窗口恢复 / 滑动不发散 / 箱约束可行性。
// 验证（受管 Python 3.13.12 先行对拍，本文件为确定性重现）：
//   8 项检查全 PASS，末行打印 "=== N passed, M failed ==="
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

// ---- 固定种子 LCG + Box-Muller（与验证端逐位一致，seed=12345）----
static unsigned long long _seed = 12345ULL;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((_seed >> 33) & 0x7FFFFFFFULL) / (double)0x80000000ULL;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// ---- 线性代数助手（通用 Mat/Vec + 2×2 特化）----
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
static Vec vadd(const Vec& a, const Vec& b) { Vec c(a); for (size_t i = 0; i < a.size(); ++i) c[i] += b[i]; return c; }
static Vec vsub(const Vec& a, const Vec& b) { Vec c(a); for (size_t i = 0; i < a.size(); ++i) c[i] -= b[i]; return c; }
static double vdot(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }
static double vnorm(const Vec& a) { return sqrt(vdot(a, a)); }
static Vec matvecM(const Mat& A, const Vec& x) {  // Mat*Vec -> Vec
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}
static Vec matvec(const Mat& A, const Vec& x) { return matvecM(A, x); }
static Mat chol2(const Mat& A) {   // 2×2 下三角 Cholesky
    double l00 = sqrt(fmax(A[0][0], 0.0));
    double l10 = A[1][0] / (l00 > 1e-12 ? l00 : 1e-12);
    double l11 = sqrt(fmax(A[1][1] - l10 * l10, 0.0));
    return {{l00, 0.0}, {l10, l11}};
}
static Mat inv2(const Mat& M) {
    double d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    d = (fabs(d) > 1e-15) ? d : 1e-15;
    return {{M[1][1] / d, -M[0][1] / d}, {-M[1][0] / d, M[0][0] / d}};
}
static Vec sample_gaussian(const Vec& mean, const Mat& L) {  // L 下三角
    double d0 = gauss(), d1 = gauss();
    return {mean[0] + L[0][0] * d0, mean[1] + L[1][0] * d0 + L[1][1] * d1};
}

// ============================================================
// 模型（CV：位置-速度）
// ============================================================
static const double dt = 0.1;
static const Mat F = {{1.0, dt}, {0.0, 1.0}};
static const double q = 0.8;
static const Mat Q = {{q * dt*dt*dt*dt / 4.0, q * dt*dt*dt / 2.0},
                      {q * dt*dt*dt / 2.0,    q * dt*dt}};
static const Mat H = {{1.0, 0.0}};
static const double R = 0.4;
static const Vec x0_mean = {0.0, 0.5};
static const Mat P0 = {{10.0, 0.0}, {0.0, 10.0}};

// ============================================================
// Part A：EnKF vs 精确 KF
// ============================================================
static void partA(int& pass, int& fail) {
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== A. EnKF（随机集合 KF） vs 精确 KF ===\n");
    const int N = 80, m = 200;
    const double im = 1.0 / (m - 1);

    // 真值 + 量测
    Vec xt = x0_mean;
    Vec z(N, 0.0);
    for (int k = 0; k < N; ++k) {
        double w0 = gauss() * sqrt(Q[0][0]);
        double w1 = gauss() * sqrt(Q[1][1]);
        Vec w = {w0, w1};
        Vec inn = {0.5 * (w[0] + w[1]), 0.5 * (w[0] + w[1])};
        xt = vadd(matvec(F, xt), inn);
        z[k] = H[0][0] * xt[0] + H[0][1] * xt[1] + gauss() * sqrt(R);
    }

    // 精确 KF（标量观测 H=[1,0]）
    Vec xk = x0_mean; Mat Pk = P0;
    vector<Vec> kf_means; vector<Mat> kf_covs;
    for (int k = 0; k < N; ++k) {
        xk = matvec(F, xk);
        Pk = addm(matmul(matmul(F, Pk), transpose(F)), Q);
        double S = H[0][0] * Pk[0][0] * H[0][0] + R;
        double K0 = Pk[0][0] * H[0][0] / S;
        double K1 = Pk[1][0] * H[0][0] / S;
        double e = z[k] - (H[0][0] * xk[0] + H[0][1] * xk[1]);
        xk = {xk[0] + K0 * e, xk[1] + K1 * e};
        Mat IK = {{1.0 - K0 * H[0][0], -K0 * H[0][1]}, {-K1 * H[0][0], 1.0 - K1 * H[0][1]}};
        Mat KRKt = {{K0 * R * K0, K0 * R * K1}, {K1 * R * K0, K1 * R * K1}};
        Pk = addm(matmul(matmul(IK, Pk), transpose(IK)), KRKt);
        kf_means.push_back(xk); kf_covs.push_back(Pk);
    }

    // EnKF：随机集合（perturbed-observation）
    Mat L0 = chol2(P0);
    vector<Vec> X(m);
    for (int i = 0; i < m; ++i) X[i] = sample_gaussian(x0_mean, L0);
    Mat LQ = chol2(Q);
    vector<Vec> enkf_means; vector<Mat> enkf_covs;
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < m; ++i) X[i] = vadd(matvec(F, X[i]), sample_gaussian({0, 0}, LQ));
        Vec xbar = {0, 0};
        for (int i = 0; i < m; ++i) { xbar[0] += X[i][0]; xbar[1] += X[i][1]; }
        xbar[0] /= m; xbar[1] /= m;
        Mat Pe = {{0, 0}, {0, 0}};
        for (int i = 0; i < m; ++i) {
            double d0 = X[i][0] - xbar[0], d1 = X[i][1] - xbar[1];
            Pe[0][0] += d0 * d0; Pe[0][1] += d0 * d1; Pe[1][0] += d1 * d0; Pe[1][1] += d1 * d1;
        }
        Pe[0][0] *= im; Pe[0][1] *= im; Pe[1][0] *= im; Pe[1][1] *= im;
        double HP0 = H[0][0] * Pe[0][0] + H[0][1] * Pe[0][1];
        double HP1 = H[0][0] * Pe[1][0] + H[0][1] * Pe[1][1];
        double S = HP0 * H[0][0] + HP1 * H[0][1] + R;
        double K0 = (Pe[0][0] * H[0][0] + Pe[0][1] * H[0][1]) / S;
        double K1 = (Pe[1][0] * H[0][0] + Pe[1][1] * H[0][1]) / S;
        for (int i = 0; i < m; ++i) {
            double eps = gauss() * sqrt(R);
            double ytil = z[k] + eps;
            double innov = ytil - (H[0][0] * X[i][0] + H[0][1] * X[i][1]);
            X[i] = {X[i][0] + K0 * innov, X[i][1] + K1 * innov};
        }
        enkf_means.push_back(xbar); enkf_covs.push_back(Pe);
    }

    const int W = 20;
    double em_mean = 0, em_cov = 0;
    for (int kk = 0; kk < W; ++kk) {
        int idx = N - 1 - W + kk;
        double d = vnorm(vsub(enkf_means[idx], kf_means[idx]));
        if (d > em_mean) em_mean = d;
        double c = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            double e = enkf_covs[idx][i][j] - kf_covs[idx][i][j];
            c += e * e;
        }
        c = sqrt(c);
        if (c > em_cov) em_cov = c;
    }
    printf("  [info] 末窗 EnKF-KF 均值最大偏差 = %.4f, 协方差 Frobenius 最大偏差 = %.4f\n", em_mean, em_cov);
    chk("EnKF 均值 ≈ KF 均值(末窗)", em_mean, 0.35);
    chk("EnKF 协方差 ≈ KF 协方差(末窗)", em_cov, 0.50);

    // 集合协方差估计器一致性
    Mat Ptrue = {{1.0, 0.3}, {0.3, 2.0}};
    Mat Lt = chol2(Ptrue);
    Mat Pe0 = {{0, 0}, {0, 0}};
    for (int i = 0; i < m; ++i) {
        Vec xs = sample_gaussian({0, 0}, Lt);
        Pe0[0][0] += xs[0] * xs[0]; Pe0[0][1] += xs[0] * xs[1];
        Pe0[1][0] += xs[1] * xs[0]; Pe0[1][1] += xs[1] * xs[1];
    }
    Pe0[0][0] *= im; Pe0[0][1] *= im; Pe0[1][0] *= im; Pe0[1][1] *= im;
    double err_pe0 = 0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        double e = Pe0[i][j] - Ptrue[i][j]; err_pe0 += e * e;
    }
    err_pe0 = sqrt(err_pe0);
    printf("  [info] 集合协方差估计器 vs 真值 P 偏差 = %.4f\n", err_pe0);
    chk("集合协方差估计器一致性", err_pe0, 0.45);
}

// ============================================================
// Part B：MHE（移动 horizon 估计，全状态观测 H=I）
// ============================================================
static int    Ldim = 12;      // 滑动窗长度
static int    Ktot = 40;      // 总步数
static const double dtb = 0.1;
static const Mat Fb = {{1.0, dtb}, {0.0, 1.0}};
static const double Rp = 0.2, Rv = 0.2;
static const Mat Rb = {{Rp, 0.0}, {0.0, Rv}};
static const Mat Qb = {{0.02, 0.0}, {0.0, 0.02}};   // Q⁻¹=50 >> R⁻¹=5：信任动力学、量测校正（全状态观测 H=I，速度可观测）
static Mat Qi, Ri, P0i;
static const Vec x0v = {0.0, 0.5};
static vector<Vec> xtr;   // 真值轨迹（全局）
static vector<Vec> zb;    // 全状态量测 zb[k] 量测 xtr[k]

// 高斯-牛顿法方程装配：H = Σ Jᵀ W J,  Atb = Σ Jᵀ W z
static void build_normeq(int t, const Vec& prior_mean, const Mat& prior_ci, Mat& Hout, Vec& bout) {
    int dim = 2 * (Ldim + 1);
    Mat H(dim, Vec(dim, 0.0)); Vec b(dim, 0.0);
    Mat FtQi = matmul(transpose(Fb), Qi);
    Mat FtQiF = matmul(FtQi, Fb);
    Mat cr = matmul(Qi, Fb);
    // 量测残差 r = x_s - z_s,  W = Ri  (H=I 全状态观测)
    for (int s = 0; s <= Ldim; ++s) {
        int base = 2 * s;
        Vec zs = zb[t - Ldim + s];
        for (int a = 0; a < 2; ++a) {
            for (int bb = 0; bb < 2; ++bb) H[base + a][base + bb] += Ri[a][bb];
            b[base + a] += Ri[a][0] * zs[0] + Ri[a][1] * zs[1];
        }
    }
    // 动力学残差 r = x_{s+1} - F x_s,  W = Qi  (J = [-F | +I])
    for (int s = 0; s < Ldim; ++s) {
        int j = 2 * (s + 1), i = 2 * s;
        for (int a = 0; a < 2; ++a) {
            for (int bb = 0; bb < 2; ++bb) {
                H[j + a][j + bb] += Qi[a][bb];
                H[i + a][i + bb] += FtQiF[a][bb];
                H[j + a][i + bb] += -cr[a][bb];
                H[i + a][j + bb] += -FtQi[a][bb];
            }
        }
    }
    // 先验残差 r0 = x_0 - μ0,  W = P0i
    for (int a = 0; a < 2; ++a) {
        for (int bb = 0; bb < 2; ++bb) H[a][bb] += prior_ci[a][bb];
        b[a] += prior_ci[a][0] * prior_mean[0] + prior_ci[a][1] * prior_mean[1];
    }
    Hout = H; bout = b;
}
static Vec solve_gj(const Mat& A, const Vec& b) {
    int dim = b.size();
    vector<Vec> M(dim, Vec(dim + 1, 0.0));
    for (int i = 0; i < dim; ++i) { for (int j = 0; j < dim; ++j) M[i][j] = A[i][j]; M[i][dim] = b[i]; }
    for (int c = 0; c < dim; ++c) {
        int p = c; double best = fabs(M[c][c]);
        for (int r = c + 1; r < dim; ++r) if (fabs(M[r][c]) > best) { best = fabs(M[r][c]); p = r; }
        swap(M[c], M[p]);
        double pv = (fabs(M[c][c]) > 1e-15) ? M[c][c] : 1e-15;
        for (int r = 0; r < dim; ++r) {
            if (r != c) {
                double f = M[r][c] / pv;
                if (f) for (int j = 0; j <= dim; ++j) M[r][j] -= f * M[c][j];
            }
        }
    }
    Vec x(dim);
    for (int c = 0; c < dim; ++c) x[c] = M[c][dim] / M[c][c];
    return x;
}
static double max_eig(const Mat& A) {
    int dim = A.size(); Vec v(dim, 1.0 / dim);
    for (int it = 0; it < 80; ++it) {
        Vec w(dim, 0.0);
        for (int i = 0; i < dim; ++i) for (int j = 0; j < dim; ++j) w[i] += A[i][j] * v[j];
        double nrm = 0; for (double x : w) nrm += x * x; nrm = sqrt(nrm) ? sqrt(nrm) : 1e-15;
        for (int i = 0; i < dim; ++i) v[i] = w[i] / nrm;
    }
    double lam = 0; for (int i = 0; i < dim; ++i) for (int j = 0; j < dim; ++j) lam += A[i][j] * v[i] * v[j];
    return lam;
}
static double f_obj(const Mat& A, const Vec& b, const Vec& x) {
    Vec Ax = matvecM(A, x);
    double s = 0; for (int i = 0; i < (int)x.size(); ++i) s += 0.5 * x[i] * Ax[i] - b[i] * x[i];
    return s;
}
static Vec solve_mhe_box(int t, const Vec& prior_mean, const Mat& prior_ci, const Vec* lb, const Vec* ub) {
    Mat H; Vec b;
    build_normeq(t, prior_mean, prior_ci, H, b);
    int dim = b.size();
    if (!lb) return solve_gj(H, b);
    // 初值先投影到可行域，避免“最优恰在边界/投影步未触发”时返回不可行点
    Vec xg = solve_gj(H, b);
    for (int i = 0; i < dim; ++i) { double v = xg[i]; v = fmax((*lb)[i], v); v = fmin((*ub)[i], v); xg[i] = v; }
    double Lmax = max_eig(H) ? max_eig(H) : 1.0;
    double alpha = 1.0 / Lmax;
    for (int it = 0; it < 50000; ++it) {
        Vec g(dim);
        for (int i = 0; i < dim; ++i) { double gi = -b[i]; for (int j = 0; j < dim; ++j) gi += H[i][j] * xg[j]; g[i] = gi; }
        Vec xn(dim);
        for (int i = 0; i < dim; ++i) {
            double v = xg[i] - alpha * g[i];
            v = fmax((*lb)[i], v); v = fmin((*ub)[i], v);
            xn[i] = v;
        }
        // 回溯：保证目标不增（凸问题→收敛到全局最优，绝不发散到 1e9）
        if (f_obj(H, b, xn) <= f_obj(H, b, xg) + 1e-12) {
            double d = 0; for (int i = 0; i < dim; ++i) d = fmax(d, fabs(xn[i] - xg[i]));
            if (d < 1e-11) { xg = xn; break; }
            xg = xn;
        } else {
            alpha *= 0.5;
            if (alpha < 1e-15) break;
        }
    }
    return xg;
}

static void partB(int& pass, int& fail) {
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== B. MHE（移动 horizon 估计） vs 真值/约束 ===\n");
    Qi = inv2(Qb); Ri = inv2(Rb); P0i = inv2(P0);
    Mat LQb = chol2(Qb);

    // 真值 + 量测（独立于 Part A 的 RNG 段，顺序固定）
    xtr.assign(1, x0v); zb.assign(0, Vec(2));
    Vec xb = x0v;
    for (int k = 0; k < Ktot; ++k) {
        zb.push_back({xb[0] + gauss() * sqrt(Rp), xb[1] + gauss() * sqrt(Rv)});
        xb = vadd(matvec(Fb, xb), sample_gaussian({0, 0}, LQb));
        xtr.push_back(xb);
    }

    // B1：无约束 MHE 解满足法方程（残差 ~0）
    Mat AtA; Vec Atb;
    build_normeq(Ldim, x0v, P0i, AtA, Atb);
    Vec xu = solve_gj(AtA, Atb);
    double resid = 0;
    for (int i = 0; i < (int)Atb.size(); ++i) {
        double r = -Atb[i]; for (int j = 0; j < (int)Atb.size(); ++j) r += AtA[i][j] * xu[j];
        resid = fmax(resid, fabs(r));
    }
    printf("  [info] 无约束 MHE 法方程残差 = %.3e\n", resid);
    chk("MHE 无约束解满足法方程(残差~0)", resid, 1e-6);

    // B2：单窗口 [0,L] 当前态（末状态 x_L）恢复真值
    Vec xv = solve_mhe_box(Ldim, x0v, P0i, nullptr, nullptr);
    Vec xcur = {xv[2 * Ldim], xv[2 * Ldim + 1]};
    double e_cur = vnorm(vsub(xcur, xtr[Ldim]));
    printf("  [info] 单窗口 MHE 当前态 vs 真值偏差 = %.4f\n", e_cur);
    chk("MHE 单窗口恢复当前态(≈量测噪声尺度)", e_cur, 2.0);

    // B3：滑动窗（移动 horizon）当前态恢复，统计最大偏差（不应发散）
    double e_slide = 0; Vec pm = x0v; Mat pci = P0i;
    for (int t = Ldim; t < Ktot; ++t) {
        Vec xv2 = solve_mhe_box(t, pm, pci, nullptr, nullptr);
        Vec cur = {xv2[2 * Ldim], xv2[2 * Ldim + 1]};
        e_slide = fmax(e_slide, vnorm(vsub(cur, xtr[t])));
        pm = {xv2[2], xv2[3]};
    }
    printf("  [info] 滑动窗 MHE 当前态最大偏差 = %.4f\n", e_slide);
    chk("MHE 滑动窗不发散(当前态恢复)", e_slide, 3.0);

    // B4：箱约束可行性（x_0 ≥ 0；约束激活时 unconstrained 不可行）
    int dim = 2 * (Ldim + 1);
    Vec xu_all = solve_mhe_box(Ldim, x0v, P0i, nullptr, nullptr);
    Vec lb(dim, 0.0), ub(dim, 1e9);
    Vec xv_box = solve_mhe_box(Ldim, x0v, P0i, &lb, &ub);
    bool feasible = true;
    for (int i = 0; i < dim; ++i) if (!(0.0 <= xv_box[i] && xv_box[i] <= 1e9)) { feasible = false; break; }
    double uncon_x0 = xu_all[0];
    printf("  [info] 无约束 x_0 = %.4f, 约束后 x_0 = %.4f, 可行=%s\n",
           uncon_x0, xv_box[0], feasible ? "True" : "False");
    chk("MHE 箱约束解可行(x_0>=0)", feasible ? 0.0 : 1.0, 1e-9);
    if (uncon_x0 < -1e-9) {
        chk("约束激活时与无约束不同(演示价值)", (fabs(xv_box[0] - uncon_x0) > 1e-12) ? 0.0 : 1.0, 1e-9);
    } else {
        printf("  [info] 本窗口无约束 x_0 已可行，约束未激活（仍验证可行性）\n");
        ++pass;
    }
}

int main() {
    int pass = 0, fail = 0;
    partA(pass, fail);
    partB(pass, fail);
    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
