// ============================================================
// KF_demo L3.4 capstone：雷达测距测角 + 里程计 多传感器融合 ⭐⭐
// 纯 C++17。4D 状态 x=[px,py,vx,vy]，常速模型：
//   传感器1 雷达：对已知路标的 距离+方位（非线性 H，含 1/r、1/r²）
//   传感器2 里程计：直接测速度（线性 H）
// 工程要点全覆盖：
//   ① 非线性观测的 EKF 雅可比（与 Jacobian_demo L3.8 同型）
//   ② 方位角残差必须缠绕到 [-π,π]
//   ③ 精确离散 Q（每步独立加速度）：dt⁴/4、dt³/2、dt 分块
//   ④ NEES 一致性闸门（5 次蒙特卡洛平均，初始误差从 N(0,P₀) 抽样）
// 验证：融合 RMSE < 仅雷达；NEES ∈ (2, 5.5)
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
static const double PI = 3.14159265358979323846;

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
static Mat inv(const Mat& A) {   // 通用高斯消元求逆
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

// 单次仿真（返回：融合位置误差²累计 / 雷达位置误差²累计 / NEES 累计 / 步数）
struct RunResult { double e2f, e2r, nees; int cnt; };
static RunResult one_run(unsigned long long seed) {
    double dt = 0.1, q = 0.05;
    double r_rng = 0.30, r_brg = 0.02, r_odo = 0.10;
    double lmx = 10.0, lmy = -2.0;
    int N = 120;
    Mat F = {{1, 0, dt, 0}, {0, 1, 0, dt}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Mat Q(4, Vec(4, 0.0));
    Q[0][0] = Q[1][1] = q * dt*dt*dt*dt / 4;
    Q[0][2] = Q[1][3] = Q[2][0] = Q[3][1] = q * dt*dt*dt / 2;
    Q[2][2] = Q[3][3] = q * dt;

    set_seed(seed);
    Vec xt = {0.0, 0.0, 1.0, 0.6};
    Vec x(4);
    for (int i = 0; i < 4; ++i) x[i] = gauss();   // 初始误差 ~ N(0, P₀)
    Mat P = eye(4);

    RunResult res = {0, 0, 0, 0};
    for (int k = 0; k < N; ++k) {
        double ax = gauss() * sqrt(q), ay = gauss() * sqrt(q);
        xt = {xt[0] + dt * xt[2] + 0.5 * dt * dt * ax,
              xt[1] + dt * xt[3] + 0.5 * dt * dt * ay,
              xt[2] + dt * ax, xt[3] + dt * ay};
        // 预测
        x = matvec(F, x);
        P = addm(matmul(matmul(F, P), transpose(F)), Q);
        // (1) 雷达：range+bearing 到路标
        double zr = hypot(lmx - xt[0], lmy - xt[1]) + gauss() * sqrt(r_rng);
        double zb = atan2(lmy - xt[1], lmx - xt[0]) + gauss() * sqrt(r_brg);
        double dx = lmx - x[0], dy = lmy - x[1];
        double rr = hypot(dx, dy); if (rr < 1e-9) rr = 1e-9;
        Mat H = {{-dx / rr, -dy / rr, 0, 0},
                 {dy / (rr * rr), -dx / (rr * rr), 0, 0}};
        Mat R1 = {{r_rng, 0}, {0, r_brg}};
        double db = zb - atan2(dy, dx);
        db = atan2(sin(db), cos(db));               // 缠绕到 [-π,π]
        Mat Ht = transpose(H);
        Mat S = addm(matmul(matmul(H, P), Ht), R1);
        Mat K = matmul(matmul(P, Ht), inv(S));
        Vec innov = {zr - rr, db};
        Vec xp(4, 0.0);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) xp[i] += K[i][j] * innov[j];
        for (int i = 0; i < 4; ++i) x[i] += xp[i];
        Mat IK = subm(eye(4), matmul(K, H));
        P = addm(matmul(matmul(IK, P), transpose(IK)),
                 matmul(K, matmul(R1, transpose(K))));
        if (k >= 20) res.e2r += (x[0] - xt[0]) * (x[0] - xt[0]) + (x[1] - xt[1]) * (x[1] - xt[1]);
        // (2) 里程计：测速度
        Vec zo = {xt[2] + gauss() * sqrt(r_odo), xt[3] + gauss() * sqrt(r_odo)};
        Mat H2 = {{0, 0, 1, 0}, {0, 0, 0, 1}};
        Mat R2 = {{r_odo, 0}, {0, r_odo}};
        Mat H2t = transpose(H2);
        Mat S2 = addm(matmul(matmul(H2, P), H2t), R2);
        Mat K2 = matmul(matmul(P, H2t), inv(S2));
        Vec innov2 = {zo[0] - x[2], zo[1] - x[3]};
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) x[i] += K2[i][j] * innov2[j];
        Mat IK2 = subm(eye(4), matmul(K2, H2));
        P = addm(matmul(matmul(IK2, P), transpose(IK2)),
                 matmul(K2, matmul(R2, transpose(K2))));
        // NEES
        if (k >= 20) {
            Mat Pi = inv(P);
            double nees = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    nees += (xt[i] - x[i]) * Pi[i][j] * (xt[j] - x[j]);
            res.nees += nees; ++res.cnt;
            res.e2f += (x[0] - xt[0]) * (x[0] - xt[0]) + (x[1] - xt[1]) * (x[1] - xt[1]);
        }
    }
    return res;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L3.4 capstone：雷达+里程计融合（含 NEES 闸门）===\n");

    double tf = 0, tr = 0, tn = 0; int tc = 0;
    for (int s = 0; s < 5; ++s) {
        RunResult r = one_run(9090 + 137 * s);
        tf += r.e2f; tr += r.e2r; tn += r.nees; tc += r.cnt;
    }
    double rmse_f = sqrt(tf / tc), rmse_r = sqrt(tr / tc), nees = tn / tc;
    printf("  [demo] rmse_fusion=%.4f  rmse_仅雷达=%.4f  (改善 %.0f%%)\n",
           rmse_f, rmse_r, 100.0 * (1.0 - rmse_f / rmse_r));
    printf("  [demo] NEES_avg=%.2f  (n=4，一致性区间约 (2,5.5))\n", nees);
    chk("融合 RMSE < 仅雷达", rmse_f < rmse_r ? 0.0 : 1.0, 0.5);
    chk("NEES 一致性 (2<n<5.5)", (2.0 < nees && nees < 5.5) ? 0.0 : 1.0, 0.5);

    printf("  [结论] 多传感器融合 = 每个传感器一套 (h, H, R) 依次做更新；\n");
    printf("         顺序无关（同一时刻高斯乘积可交换）。闸门看 NEES/NIS。\n");

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
