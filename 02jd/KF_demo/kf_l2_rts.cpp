// ============================================================
// KF_demo L2.3 RTS 平滑器（Rauch-Tung-Striebel）
// 纯 C++17。前向 KF 存档 + 后向递推：
//   C_k = P_k Fᵀ (P⁻_{k+1})⁻¹
//   xˢ_k = x_k + C_k (xˢ_{k+1} - x⁻_{k+1})
//   Pˢ_k = P_k + C_k (Pˢ_{k+1} - P⁻_{k+1}) C_kᵀ
// 验证：① 平滑 RMSE < 滤波 RMSE（离线全信息）
//       ② 末端时刻 平滑 == 滤波（没有未来信息可用）
//       ③ 中段 平滑 P ≤ 滤波 P
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 33;
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
static Mat inv2(const Mat& M) {
    double a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
    double det = a * d - b * c;
    return {{d / det, -b / det}, {-c / det, a / det}};
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.3 RTS 平滑器 ===\n");

    double dt = 0.1, q = 0.05, r = 0.5;
    int N = 80;
    Mat F = {{1.0, dt}, {0.0, 1.0}};
    Mat Q = {{q * dt*dt*dt*dt / 4.0, q * dt*dt*dt / 2.0},
             {q * dt*dt*dt / 2.0,    q * dt}};
    Mat H = {{1.0, 0.0}};

    Vec xt = {0.0, 1.0};
    Vec x = {0.0, 0.5};
    Mat P = eye(2); P[0][0] = P[1][1] = 10.0;

    vector<Vec> xs_f(N), xs_p(N), truths(N);
    vector<Mat> Ps_f(N), Ps_p(N);

    // ---- 前向 KF（存档）----
    for (int k = 0; k < N; ++k) {
        double a = gauss() * sqrt(q);
        xt = {xt[0] + dt * xt[1] + 0.5 * dt * dt * a, xt[1] + dt * a};
        double z = xt[0] + gauss() * sqrt(r);
        Vec xm = matvec(F, x);
        Mat Pm = addm(matmul(matmul(F, P), transpose(F)), Q);
        double S = Pm[0][0] + r;
        double k0 = Pm[0][0] / S, k1 = Pm[1][0] / S;
        double e = z - xm[0];
        x = {xm[0] + k0 * e, xm[1] + k1 * e};
        Mat IK = subm(eye(2), {{k0, 0.0}, {k1, 0.0}});
        Mat KRKt = {{k0 * r * k0, k0 * r * k1}, {k1 * r * k0, k1 * r * k1}};
        P = addm(matmul(matmul(IK, Pm), transpose(IK)), KRKt);
        xs_f[k] = x; Ps_f[k] = P;
        xs_p[k] = xm; Ps_p[k] = Pm;
        truths[k] = xt;
    }

    // ---- 后向 RTS ----
    vector<Vec> xs_s(N);
    vector<Mat> Ps_s(N);
    xs_s[N - 1] = xs_f[N - 1];
    Ps_s[N - 1] = Ps_f[N - 1];
    for (int k = N - 2; k >= 0; --k) {
        Mat C = matmul(matmul(Ps_f[k], transpose(F)), inv2(Ps_p[k + 1]));
        Vec d(N > 0 ? 2 : 0);
        Vec dxv = {xs_s[k + 1][0] - xs_p[k + 1][0], xs_s[k + 1][1] - xs_p[k + 1][1]};
        xs_s[k] = {xs_f[k][0] + C[0][0] * dxv[0] + C[0][1] * dxv[1],
                   xs_f[k][1] + C[1][0] * dxv[0] + C[1][1] * dxv[1]};
        Mat dP = subm(Ps_s[k + 1], Ps_p[k + 1]);
        Ps_s[k] = addm(Ps_f[k], matmul(matmul(C, dP), transpose(C)));
    }

    double rmf = 0, rms = 0;
    for (int k = 0; k < N; ++k) {
        rmf += (xs_f[k][0] - truths[k][0]) * (xs_f[k][0] - truths[k][0]);
        rms += (xs_s[k][0] - truths[k][0]) * (xs_s[k][0] - truths[k][0]);
    }
    rmf = sqrt(rmf / N); rms = sqrt(rms / N);
    printf("  [demo] rmse_filter=%.4f  rmse_smooth=%.4f  (改善 %.0f%%)\n",
           rmf, rms, 100.0 * (1.0 - rms / rmf));
    chk("平滑 RMSE < 滤波 RMSE", rms < rmf ? 0.0 : 1.0, 0.5);
    chk("末端: 平滑 == 滤波", fmax(fabs(xs_s[N-1][0] - xs_f[N-1][0]),
                                 fabs(xs_s[N-1][1] - xs_f[N-1][1])), 1e-12);
    chk("中段: 平滑P <= 滤波P", Ps_s[N/2][0][0] <= Ps_f[N/2][0][0] ? 0.0 : 1.0, 0.5);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
