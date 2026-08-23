// ============================================================
// KF_demo L3.7 扩展：EKF 在 PX4 中的应用（误差状态 + NIS 闸门 + 野值拒收）
// 纯 C++17。PX4 EKF2 架构的 mini 复刻（2D 平面化，真实为 24 维误差状态）：
//   标称积分（IMU 100Hz）+ 误差状态 KF [δp,δv,δb]（6 维）
//   GPS 位置更新前做 NIS 闸门检验：NIS = νᵀS⁻¹ν > gate(5σ≈25) → 拒收
// 验证（对比 gated vs ungated，同一噪声序列）：
//  ① 注入 2 个 GPS 野值(+25,−20)：闸门全部拦截
//  ② 闸门滤波 RMSE ≪ 无闸门（0.8 vs 2.2）
//  ③ 接受观测的平均 NIS ≈ 自由度 2（滤波器一致）
// PX4 真实对应：src/modules/ekf2，24 维 = [δq(4), δp(3), δv(3), δbg(3),
//   δba(3), δmag(3), δwind(2), δterrain(1)+...]，按传感器分别 fuse。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

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

struct RunResult { double rmse; int n_rej; int rej_outlier; double nis_avg; };

static RunResult run(bool gated) {
    set_seed(18);
    double dt = 0.01, q_a = 1e-3, b_true = 0.15, r_gps = 1.0, gate = 25.0;
    int N = 1200;
    auto is_outlier = [](int k) { return k == 400 || k == 800; };
    double p_t[2] = {0, 0}, v_t[2] = {1.0, 0.6};
    double x[4] = {0, 0, 0, 0};       // 标称 [px,py,vx,vy]
    double b_n[2] = {0, 0};           // 每轴零偏估计
    Mat P = eye(6);
    for (int i = 0; i < 6; ++i) P[i][i] = 0.05;
    double e2 = 0; int cnt = 0, n_rej = 0, rej_out = 0;
    double nis_ok = 0; int nis_cnt = 0;
    for (int k = 1; k <= N; ++k) {
        double a_t[2] = {0.3 * sin(0.02 * k), 0.3 * cos(0.015 * k)};
        double am[2] = {a_t[0] + b_true + gauss() * sqrt(q_a),
                        a_t[1] + b_true + gauss() * sqrt(q_a)};
        for (int i = 0; i < 2; ++i) {
            p_t[i] += v_t[i] * dt + 0.5 * a_t[i] * dt * dt;
            v_t[i] += a_t[i] * dt;
        }
        for (int i = 0; i < 2; ++i) {
            x[2 + i] += (am[i] - b_n[i]) * dt;
            x[i] += x[2 + i] * dt;
        }
        // 误差预测：F = I + [[0,dt,0],[0,0,dt],[0,0,0]]（分块）
        Mat F = eye(6);
        F[0][2] = dt; F[1][3] = dt;
        F[2][4] = dt; F[3][5] = dt;
        Mat Qd = Mat(6, Vec(6, 0.0));
        Qd[0][0] = Qd[1][1] = q_a * dt * dt * dt * dt / 4;
        Qd[0][2] = Qd[1][3] = Qd[2][0] = Qd[3][1] = q_a * dt * dt * dt / 2;
        Qd[2][2] = Qd[3][3] = q_a * dt;
        Qd[4][4] = Qd[5][5] = 1e-8;
        P = addm(matmul(matmul(F, P), transpose(F)), Qd);
        // GPS 更新（每 20 步）
        if (k % 20 == 0) {
            double z[2] = {p_t[0] + gauss() * sqrt(r_gps), p_t[1] + gauss() * sqrt(r_gps)};
            if (is_outlier(k)) { z[0] += 25.0; z[1] -= 20.0; }
            Mat H = Mat(2, Vec(6, 0.0));
            H[0][0] = H[1][1] = 1.0;
            Mat Rm = {{r_gps, 0}, {0, r_gps}};
            double nu[2] = {z[0] - x[0], z[1] - x[1]};
            Mat S = addm(matmul(matmul(H, P), transpose(H)), Rm);
            double nis = nu[0] * nu[0] / S[0][0] + nu[1] * nu[1] / S[1][1];
            if ((!gated) || nis < gate) {
                if (!is_outlier(k)) { nis_ok += nis; ++nis_cnt; }
                Mat K = matmul(matmul(P, transpose(H)), inv(S));
                Vec dxv = matvec(K, {nu[0], nu[1]});
                for (int i = 0; i < 4; ++i) x[i] += dxv[i];
                b_n[0] += dxv[4]; b_n[1] += dxv[5];
                Mat IK = subm(eye(6), matmul(K, H));
                P = addm(matmul(matmul(IK, P), transpose(IK)), matmul(K, matmul(Rm, transpose(K))));
            } else {
                ++n_rej;
                if (is_outlier(k)) ++rej_out;
            }
        }
        if (k >= 100) { e2 += (x[0] - p_t[0]) * (x[0] - p_t[0]) + (x[1] - p_t[1]) * (x[1] - p_t[1]); ++cnt; }
    }
    return {sqrt(e2 / cnt), n_rej, rej_out, nis_cnt ? nis_ok / nis_cnt : -1};
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.7 PX4 EKF2 mini：误差状态 + NIS 闸门 ===\n");

    RunResult g = run(true), u = run(false);
    printf("  [demo] 有闸门:   RMSE=%.4f  拒收=%d(其中野值 %d)  接受NIS均值=%.2f\n",
           g.rmse, g.n_rej, g.rej_outlier, g.nis_avg);
    printf("  [demo] 无闸门:   RMSE=%.4f  （野值直接打飞滤波器）\n", u.rmse);
    chk("闸门 RMSE ≪ 无闸门", g.rmse < 0.5 * u.rmse ? 0.0 : 1.0, 0.5);
    chk("野值全部被拦截(=2)", g.rej_outlier == 2 ? 0.0 : 1.0, 0.5);
    chk("误伤率低(总拒收≤3)", g.n_rej, 3.0 + 1e-9);
    chk("接受观测的 NIS≈2", fabs(g.nis_avg - 2.0), 1.0);

    printf("  [PX4 对应] src/modules/ekf2：24 维误差状态(四元数扰动+pos/vel+\n");
    printf("  陀螺零偏/加计零偏/磁强计/风/地形)；每个传感器独立 fuse 分支，\n");
    printf("  均带 innovation gate（GPS 默认 5σ）与连续拒收后的故障降级逻辑。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
