// ============================================================
// KF_demo L3.5 扩展：EKF-SLAM（路标增广与回环收敛）
// 纯 C++17。状态 x=[px,py,θ, m1x,m1y, ...]（机器人 3 + 路标 2M 维）
//   预测：里程计模型 + G/V 雅可比（过程噪声 V·Qc·Vᵀ 只进机器人块）
//   更新：路标 range+bearing 观测，H 在机器人块(1/r,1/r²,−1)与路标块非零
//   首次观测的路标用当前位姿反解初始化（真实系统标准做法）
// 验证：
//  ① H 与有限差分一致（含方位角缠绕差分）
//  ② 绕圈回环后 3 个路标误差 < 0.3 m（相关性让"回到旧路标"修正全局漂移）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 427;
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
static Mat subm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] -= B[i][j];
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

// 路标观测雅可比（slot = 路标槽位）
static Mat slam_H(const Vec& x, int slot) {
    int n2 = x.size();
    double dx = x[3 + 2 * slot] - x[0], dy = x[4 + 2 * slot] - x[1];
    double r_ = hypot(dx, dy), r2 = r_ * r_;
    Mat H(2, Vec(n2, 0.0));
    H[0][0] = -dx / r_; H[0][1] = -dy / r_;
    H[1][0] = dy / r2;  H[1][1] = -dx / r2; H[1][2] = -1.0;
    H[0][3 + 2 * slot] = dx / r_;  H[0][4 + 2 * slot] = dy / r_;
    H[1][3 + 2 * slot] = -dy / r2; H[1][4 + 2 * slot] = dx / r2;
    return H;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.5 EKF-SLAM：路标增广与回环 ===\n");

    const int M = 3;
    double lm_true[M][2] = {{4.0, 4.0}, {8.0, 0.0}, {4.0, -4.0}};
    double sig_d = 0.015, sig_t = 0.01, r_rng = 0.04, r_brg = 0.006;

    // 真值轨迹：连续左转一圈（半径 R=edge/(π/2)，回到起点）
    vector<Vec> traj;
    {
        Vec p = {0.0, 0.0, 0.0};
        traj.push_back(p);
        double edge = 8.0; int Ns = 40;
        for (int side = 0; side < 4; ++side)
            for (int i = 0; i < Ns; ++i) {
                double ds = edge / Ns, dth = (3.14159265358979323846 / 2) / Ns;
                double mid = p[2] + dth / 2;
                p = {p[0] + ds * cos(mid), p[1] + ds * sin(mid), p[2] + dth};
                traj.push_back(p);
            }
    }
    double loop_gap = hypot(traj.back()[0] - traj[0][0], traj.back()[1] - traj[0][1]);
    chk("轨迹回环(首尾<0.2)", loop_gap, 0.2);

    // ---- ① H vs FD ----
    {
        Vec xt9 = {1.0, -0.5, 0.3, 4.1, 3.9, 8.2, -0.1, 3.8, -4.2};
        Mat H0 = slam_H(xt9, 0);
        double h = 1e-6;
        Mat Hfd(2, Vec(9, 0.0));
        for (int j = 0; j < 9; ++j) {
            Vec xp = xt9, xm = xt9;
            xp[j] += h; xm[j] -= h;
            double dxp = xp[3] - xp[0], dyp = xp[4] - xp[1];
            double dxm = xm[3] - xm[0], dym = xm[4] - xm[1];
            double rp0 = hypot(dxp, dyp), rm0 = hypot(dxm, dym);
            Hfd[0][j] = (rp0 - rm0) / (2 * h);
            Hfd[1][j] = wrapPi((atan2(dyp, dxp) - xp[2]) - (atan2(dym, dxm) - xm[2])) / (2 * h);
        }
        double err = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 9; ++j) err = fmax(err, fabs(H0[i][j] - Hfd[i][j]));
        chk("SLAM H vs 有限差分", err, 1e-6);
    }

    // ---- ② 完整 EKF-SLAM ----
    Vec x = traj[0];
    Mat P = Mat(3, Vec(3, 0.0));        // 初始位姿精确
    int lm_slot[M] = {-1, -1, -1};
    int obs_cnt = 0;
    for (size_t k = 1; k < traj.size(); ++k) {
        Vec pt = traj[k], prev = traj[k - 1];
        double dth = pt[2] - prev[2];
        double ds = hypot(pt[0] - prev[0], pt[1] - prev[1]);
        double ds_n = ds + gauss() * sig_d;
        double dth_n = dth + gauss() * sig_t;
        double mid = x[2] + dth_n / 2;
        x[0] += ds_n * cos(mid); x[1] += ds_n * sin(mid); x[2] += dth_n;
        double c = cos(mid), s = sin(mid);
        Mat G = eye(3); G[0][2] = -ds_n * s; G[1][2] = ds_n * c;
        Mat V = {{c, -ds_n * s / 2}, {s, ds_n * c / 2}, {0.0, 1.0}};
        Mat Qc = {{sig_d * sig_d, 0}, {0, sig_t * sig_t}};
        Mat Qr = matmul(matmul(V, Qc), transpose(V));
        int n = x.size();
        Mat F = eye(n);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) F[i][j] = G[i][j];
        Mat Qn = Mat(n, Vec(n, 0.0));
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Qn[i][j] = Qr[i][j];
        P = addm(matmul(matmul(F, P), transpose(F)), Qn);
        // 观测
        for (int j = 0; j < M; ++j) {
            double dx = lm_true[j][0] - pt[0], dy = lm_true[j][1] - pt[1];
            double r_ = hypot(dx, dy);
            if (r_ > 7.0) continue;
            double zr = r_ + gauss() * sqrt(r_rng);
            double zb = atan2(dy, dx) - pt[2] + gauss() * sqrt(r_brg);
            ++obs_cnt;
            if (lm_slot[j] < 0) {
                // 首次：反解初始化 + 状态/协方差增广
                x.push_back(x[0] + zr * cos(zb + x[2]));
                x.push_back(x[1] + zr * sin(zb + x[2]));
                int n1 = x.size();
                Mat Pn(n1, Vec(n1, 0.0));
                for (int i = 0; i < n1 - 2; ++i) for (int jj = 0; jj < n1 - 2; ++jj) Pn[i][jj] = P[i][jj];
                Pn[n1 - 2][n1 - 2] = Pn[n1 - 1][n1 - 1] = 1.0;
                P = Pn;
                lm_slot[j] = (n1 - 5) / 2;
            } else {
                int sl = lm_slot[j];
                Mat H = slam_H(x, sl);
                Mat R = {{r_rng, 0}, {0, r_brg}};
                double zp0 = hypot(x[3 + 2 * sl] - x[0], x[4 + 2 * sl] - x[1]);
                double zp1 = atan2(x[4 + 2 * sl] - x[1], x[3 + 2 * sl] - x[0]) - x[2];
                double inno = wrapPi(zb - zp1);
                Mat S = addm(matmul(matmul(H, P), transpose(H)), R);
                Mat K = matmul(matmul(P, transpose(H)), inv(S));
                Vec dxv = matvec(K, {zr - zp0, inno});
                for (size_t i = 0; i < x.size(); ++i) x[i] += dxv[i];
                Mat IK = subm(eye(x.size()), matmul(K, H));
                P = addm(matmul(matmul(IK, P), transpose(IK)), matmul(K, matmul(R, transpose(K))));
            }
        }
    }
    double lm_max = 0; int seen = 0;
    printf("  [demo] 路标误差:");
    for (int j = 0; j < M; ++j) {
        if (lm_slot[j] >= 0) {
            int sl = lm_slot[j];
            double e = hypot(x[3 + 2 * sl] - lm_true[j][0], x[4 + 2 * sl] - lm_true[j][1]);
            printf(" lm%d=%.3f", j, e);
            lm_max = fmax(lm_max, e); ++seen;
        } else printf(" lm%d=unseen", j);
    }
    printf("\n  [demo] 回环间隙=%.3f m, 观测数=%d, 状态维数=%d\n", loop_gap, obs_cnt, (int)x.size());
    chk("路标误差 < 0.3 m", lm_max, 0.3);
    chk("三路标全部观测到", seen == M ? 0.0 : 1.0, 0.5);

    printf("  [结论] 路标与机器人在 P 中强相关：回环重访旧路标时，全局漂移被\n");
    printf("         相关性'拉'回来——这就是 EKF-SLAM 的精髓（也是 O(n²) 的代价来源）。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
