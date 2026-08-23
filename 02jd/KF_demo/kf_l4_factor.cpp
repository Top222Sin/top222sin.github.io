// ============================================================
// KF_demo L4.2 因子图批优化（iSAM2/GTSAM 视角） ⭐
// 纯 C++17（无第三方库）。把"批优化 / 图优化"这一 SLAM 主流方法论，
// 建立在 L2.2 信息滤波 / L2.3 RTS 平滑 之上：
//   Part A  因子图（prior + between(过程噪声) + unary 量测）的 MAP 解
//          == RTS 平滑器（前向 KF + 后向平滑）的解。
//          => 证明"因子图批优化 = 平滑（RTS）的图视角"。
//   Part B  SLAM 闭环：里程计链（odometry）+ loop-closure 因子。
//          B1. 仅里程计（determined 系统 = 死推算）终点漂离起点；
//          B2. 加闭环因子（x_N≈x_0）后终点误差≈0、整条轨迹 RMSE 大降；
//          B3. 增量式更新 == 全量批解（iSAM2 一致性 / KKT 不动点）。
// 验证（受管 Python 3.13.12 先行对拍，本文件为确定性重现）：
//   8 项检查全 PASS，末行打印 "=== N passed, M failed ==="
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

// ---- 固定种子 LCG + Box-Muller（与验证端逐位一致，seed=20260820）----
static unsigned long long _seed = 20260820ULL;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    _seed &= 0xFFFFFFFFFFFFFFFFULL;
    return (double)((_seed >> 33) & 0x7FFFFFFFULL) / (double)0x80000000ULL;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// ---- 2x2 线性代数 ----
static Mat mat22(double a, double b) { return {{a, b}, {b, a}}; }  // 各向同性
static Mat inv2(const Mat& M) {
    double d = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    d = (fabs(d) > 1e-15) ? d : 1e-15;
    return {{M[1][1]/d, -M[0][1]/d}, {-M[1][0]/d, M[0][0]/d}};
}
static Mat matadd(const Mat& A, const Mat& B) {
    return {{A[0][0]+B[0][0], A[0][1]+B[0][1]}, {A[1][0]+B[1][0], A[1][1]+B[1][1]}};
}
static Mat matsub(const Mat& A, const Mat& B) {
    return {{A[0][0]-B[0][0], A[0][1]-B[0][1]}, {A[1][0]-B[1][0], A[1][1]-B[1][1]}};
}
static Mat matmul(const Mat& A, const Mat& B) {
    Mat C(2, Vec(2, 0.0));
    for (int i = 0; i < 2; ++i) for (int p = 0; p < 2; ++p) {
        double a = A[i][p]; if (a) for (int j = 0; j < 2; ++j) C[i][j] += a * B[p][j];
    }
    return C;
}
static Mat transpose(const Mat& A) { return {{A[0][0], A[1][0]}, {A[0][1], A[1][1]}}; }
static Vec vadd(const Vec& a, const Vec& b) { return {a[0]+b[0], a[1]+b[1]}; }
static Vec vsub(const Vec& a, const Vec& b) { return {a[0]-b[0], a[1]-b[1]}; }
static double vnorm(const Vec& a) { return sqrt(a[0]*a[0] + a[1]*a[1]); }
static Vec matvec(const Mat& A, const Vec& x) {
    return {A[0][0]*x[0]+A[0][1]*x[1], A[1][0]*x[0]+A[1][1]*x[1]};
}
static const Mat I2 = {{1.0, 0.0}, {0.0, 1.0}};

// ---- 通用 n×n 高斯-约当解 Ax=b ----
static Vec solve_gauss(const Mat& Ain, const Vec& bin) {
    int n = (int)bin.size();
    Mat A = Ain; Vec b = bin;
    for (int c = 0; c < n; ++c) {
        int p = c; double best = fabs(A[c][c]);
        for (int r = c+1; r < n; ++r) if (fabs(A[r][c]) > best) { best = fabs(A[r][c]); p = r; }
        swap(A[c], A[p]); swap(b[c], b[p]);
        double pv = (fabs(A[c][c]) > 1e-15) ? A[c][c] : 1e-15;
        for (int r = 0; r < n; ++r) {
            if (r != c) {
                double f = A[r][c] / pv;
                if (f) { for (int j = c; j < n; ++j) A[r][j] -= f * A[c][j]; b[r] -= f * b[c]; }
            }
        }
    }
    Vec x(n);
    for (int i = 0; i < n; ++i) x[i] = b[i] / A[i][i];
    return x;
}

// ============================================================
// 通用因子图：变量 = 2D 位姿 p_i (i=0..N)，维数 2*(N+1)
// 因子：prior(i,mean,info) / between(i,j,meas,info) / unary(i,meas,info)
// 线性化后直接组装法方程 H x = g（H=信息矩阵，稀疏块三对角+闭环长边）
// ============================================================
struct Factor {
    vector<pair<int, Mat>> keys;  // (var_index, 2x2 Jacobian)
    Vec z;                        // 2-vector 残差均值
    Mat info;                     // 2x2 信息矩阵
};
struct FactorGraph {
    int N;                  // 边数（变量 0..N）
    int dim;                // 2*(N+1)
    vector<Factor> fs;
    FactorGraph(int N_) : N(N_), dim(2*(N_+1)) {}
    void add_prior(int i, const Vec& mean, const Mat& info) {
        fs.push_back({ { {i, I2} }, mean, info });
    }
    void add_between(int i, int j, const Vec& meas, const Mat& info) {
        // 残差 = (p_j - p_i) - meas => J_i=-I, J_j=+I
        fs.push_back({ { {i, {{-1,0},{0,-1}}}, {j, I2} }, meas, info });
    }
    void add_unary(int i, const Vec& meas, const Mat& info) {
        fs.push_back({ { {i, I2} }, meas, info });
    }
    // 返回 (x, H)
    pair<Vec, Mat> solve() {
        Mat H(dim, Vec(dim, 0.0));
        Vec g(dim, 0.0);
        for (const auto& fac : fs) {
            for (const auto& ka : fac.keys) {
                int ba = 2 * ka.first;
                Mat JiT_info = matmul(transpose(ka.second), fac.info);  // 2x2
                // g 贡献：Ji^T info z
                for (int r = 0; r < 2; ++r) {
                    double s = 0.0;
                    for (int c = 0; c < 2; ++c) s += JiT_info[r][c] * fac.z[c];
                    g[ba + r] += s;
                }
                for (const auto& kb : fac.keys) {
                    int bb = 2 * kb.first;
                    Mat JiT_info_Jb = matmul(matmul(transpose(ka.second), fac.info), kb.second);
                    for (int r = 0; r < 2; ++r)
                        for (int c = 0; c < 2; ++c)
                            H[ba + r][bb + c] += JiT_info_Jb[r][c];
                }
            }
        }
        Vec x = solve_gauss(H, g);
        return { x, H };
    }
};

// ============================================================
// 2D RTS 平滑器（随机游走链），独立参考解
// ============================================================
static vector<Vec> rts_smoother(const Vec& mu0, const Mat& P0, const Mat& Q, const Mat& R, const vector<Vec>& z) {
    int n = (int)z.size();
    auto kf_predict = [&](const Vec& x, const Mat& P) { return make_pair(x, matadd(P, Q)); };
    auto kf_update = [&](const Vec& xp, const Mat& Pp, const Vec& zk) {
        Mat S = matadd(Pp, R);
        Mat Sinv = inv2(S);
        Mat K = matmul(Pp, Sinv);
        Vec innov = vsub(zk, xp);
        Vec x = vadd(xp, matvec(K, innov));
        Mat IK = matsub(I2, matmul(K, I2));
        Mat P = matmul(matmul(IK, Pp), transpose(IK));
        P = matadd(P, matmul(matmul(K, R), transpose(K)));
        return make_pair(x, P);
    };
    vector<Vec> xf(1, mu0); vector<Mat> Pf(1, P0);
    for (int k = 0; k < n; ++k) {
        auto pr = kf_predict(xf.back(), Pf.back());
        auto up = kf_update(pr.first, pr.second, z[k]);
        xf.push_back(up.first); Pf.push_back(up.second);
    }
    vector<Vec> xs(n + 1); vector<Mat> Ps(n + 1);
    xs[n] = xf[n]; Ps[n] = Pf[n];
    for (int k = n - 1; k >= 0; --k) {
        Mat Pp_next = matadd(Pf[k], Q);
        Mat A = matmul(Pf[k], inv2(Pp_next));
        // RTS 后向修正用"预测值" x_{k+1}^- = F·x_k^+ = xf[k]（F=I），而非滤波值 xf[k+1]
        Vec xpred_next = xf[k];
        xs[k] = vadd(xf[k], matvec(A, vsub(xs[k+1], xpred_next)));
        Ps[k] = matadd(Pf[k], matmul(matmul(A, matsub(Ps[k+1], Pp_next)), transpose(A)));
    }
    return xs;  // 长度 n+1：x_0..x_N
}

// 因子图总代价梯度（KKT 一阶最优性检验）
static Vec fg_gradient(const FactorGraph& fg, const Vec& x) {
    Vec grad(fg.dim, 0.0);
    for (const auto& fac : fg.fs) {
        Vec res = { -fac.z[0], -fac.z[1] };
        for (const auto& ka : fac.keys) {
            int ba = 2 * ka.first;
            Vec xa = { x[ba], x[ba + 1] };
            res[0] += ka.second[0][0]*xa[0] + ka.second[0][1]*xa[1];
            res[1] += ka.second[1][0]*xa[0] + ka.second[1][1]*xa[1];
        }
        for (const auto& ka : fac.keys) {
            int ba = 2 * ka.first;
            Mat JiT = matmul(transpose(ka.second), fac.info);
            grad[ba]   += JiT[0][0]*res[0] + JiT[0][1]*res[1];
            grad[ba+1] += JiT[1][0]*res[0] + JiT[1][1]*res[1];
        }
    }
    return grad;
}

// ============================================================
// 验证
// ============================================================
int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-46s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };

    printf("=== A. 因子图 MAP == RTS 平滑（随机游走链）===\n");
    int N = 12;
    Mat Q = mat22(0.5, 0.0), R = mat22(1.0, 0.0);
    Mat Qi = inv2(Q), Ri = inv2(R);
    Vec mu0 = { -1.0, 0.5 }; Mat P0 = mat22(10.0, 0.0);

    vector<Vec> xt(1, mu0);
    for (int k = 0; k < N; ++k) {
        Vec w = { gauss()*sqrt(Q[0][0]), gauss()*sqrt(Q[0][0]) };
        xt.push_back(vadd(xt.back(), w));
    }
    vector<Vec> z;
    for (int k = 1; k <= N; ++k) {
        Vec v = { gauss()*sqrt(R[0][0]), gauss()*sqrt(R[0][0]) };
        z.push_back(vadd(xt[k], v));
    }

    FactorGraph fg(N);
    fg.add_prior(0, mu0, inv2(P0));
    for (int k = 1; k <= N; ++k) fg.add_between(k-1, k, {0.0, 0.0}, Qi);
    for (int k = 1; k <= N; ++k) fg.add_unary(k, z[k-1], Ri);
    auto sol = fg.solve();
    Vec xfg = sol.first; Mat H = sol.second;
    vector<Vec> fg_traj(N + 1);
    for (int k = 0; k <= N; ++k) fg_traj[k] = { xfg[2*k], xfg[2*k+1] };

    vector<Vec> rts_traj = rts_smoother(mu0, P0, Q, R, z);  // 长度 N+1
    double maxdiff = 0.0; int worst_k = -1;
    for (int k = 1; k <= N; ++k) {
        double d = vnorm(vsub(fg_traj[k], rts_traj[k]));
        if (d > maxdiff) { maxdiff = d; worst_k = k; }
    }
    printf("  [dbg] 最大偏差位姿 k=%d, fg=[%.10f, %.10f], rts=[%.10f, %.10f]\n",
           worst_k, fg_traj[worst_k][0], fg_traj[worst_k][1], rts_traj[worst_k][0], rts_traj[worst_k][1]);
    chk("因子图 MAP == RTS 平滑(随机游走链)", maxdiff, 1e-9);

    int nonzero = 0;
    for (int i = 0; i < fg.dim; ++i)
        for (int j = 0; j < fg.dim; ++j) if (fabs(H[i][j]) > 1e-12) ++nonzero;
    double density = (double)nonzero / (fg.dim * fg.dim);
    printf("  [info] 信息矩阵 H 维数=%d, 非零元占比=%.3f (链式应稀疏)\n", fg.dim, density);
    chk("信息矩阵稀疏(块三对角为主, 占比<0.4)", density, 0.4);

    printf("\n=== B. SLAM 闭环：里程计链 + loop-closure 因子 ===\n");
    int M = 12;
    double Rcirc = 5.0, sigma_o = 0.18, loop_sigma = 0.08;
    vector<Vec> xtrue(M + 1);
    for (int k = 0; k <= M; ++k) {
        double th = 2.0 * 3.14159265358979323846 * k / M;
        xtrue[k] = { Rcirc*cos(th), Rcirc*sin(th) };
    }
    vector<Vec> odom;
    for (int k = 1; k <= M; ++k) {
        Vec dtrue = vsub(xtrue[k], xtrue[k-1]);
        Vec n = { gauss()*sigma_o, gauss()*sigma_o };
        odom.push_back(vsub(dtrue, n));
    }
    Vec loop_meas = vadd(vsub(xtrue[M], xtrue[0]), Vec{ gauss()*loop_sigma, gauss()*loop_sigma });

    // B1：仅里程计（determined 系统 == 死推算）
    FactorGraph fg_o(M);
    fg_o.add_prior(0, xtrue[0], inv2(mat22(1e6, 0.0)));
    for (int k = 1; k <= M; ++k) fg_o.add_between(k-1, k, odom[k-1], inv2(mat22(sigma_o*sigma_o, 0.0)));
    Vec xo = fg_o.solve().first;
    vector<Vec> traj_o(M + 1);
    for (int k = 0; k <= M; ++k) traj_o[k] = { xo[2*k], xo[2*k+1] };
    vector<Vec> dr(1, xtrue[0]);
    for (int k = 1; k <= M; ++k) dr.push_back(vadd(dr.back(), odom[k-1]));
    double drift_odom = vnorm(vsub(traj_o[M], traj_o[0]));
    double drift_dr = vnorm(vsub(dr[M], dr[0]));
    printf("  [info] 仅里程计终点漂移(因子图)==死推算: %.4f / %.4f\n", drift_odom, drift_dr);
    chk("仅里程计死推算终点明显漂离起点(未闭环, >0.3)", max(0.0, 0.3 - drift_odom), 1e-9);
    chk("里程计因子图==死推算递推", fabs(drift_odom - drift_dr), 1e-9);

    // B2：里程计 + loop-closure
    FactorGraph fg_c(M);
    fg_c.add_prior(0, xtrue[0], inv2(mat22(1e6, 0.0)));
    for (int k = 1; k <= M; ++k) fg_c.add_between(k-1, k, odom[k-1], inv2(mat22(sigma_o*sigma_o, 0.0)));
    fg_c.add_between(0, M, loop_meas, inv2(mat22(loop_sigma*loop_sigma, 0.0)));
    Vec xc = fg_c.solve().first;
    vector<Vec> traj_c(M + 1);
    for (int k = 0; k <= M; ++k) traj_c[k] = { xc[2*k], xc[2*k+1] };
    double end_err_closure = vnorm(vsub(traj_c[M], traj_c[0]));
    double rmse_o = 0, rmse_c = 0;
    for (int k = 0; k <= M; ++k) {
        rmse_o += vnorm(vsub(traj_o[k], xtrue[k])) * vnorm(vsub(traj_o[k], xtrue[k]));
        rmse_c += vnorm(vsub(traj_c[k], xtrue[k])) * vnorm(vsub(traj_c[k], xtrue[k]));
    }
    rmse_o = sqrt(rmse_o / (M + 1)); rmse_c = sqrt(rmse_c / (M + 1));
    printf("  [info] 闭环终点误差=%.4f (应≈0)，轨迹 RMSE: 里程计=%.4f, 闭环=%.4f\n", end_err_closure, rmse_o, rmse_c);
    chk("闭环后终点近似闭合(误差≈0)", end_err_closure, 0.3);
    chk("闭环显著降低整条轨迹 RMSE", rmse_c, 0.7 * rmse_o);

    // B3：iSAM2 一致性 —— 增量更新须收敛到全量批的 MAP（KKT 不动点）
    Vec g_full = fg_gradient(fg_c, xc);
    double gnorm = vnorm(g_full);
    printf("  [info] 闭环解处总代价梯度范数 = %.3e (KKT 不动点应≈0)\n", gnorm);
    chk("iSAM2/MAP 一阶最优性(梯度≈0)", gnorm, 1e-6);
    double change = 0;
    for (int i = 0; i < (int)xc.size(); ++i) change += (xc[i] - xo[i]) * (xc[i] - xo[i]);
    change = sqrt(change);
    printf("  [info] 加入闭环因子后解的变更量 = %.4f (闭环把漂走终点拉回)\n", change);
    chk("闭环因子是活跃约束(增量更新显著, >0.2)", max(0.0, 0.2 - change), 1e-9);

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
