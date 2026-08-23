// ============================================================
// KF_demo L2.2 信息滤波器（Information Filter）
// 纯 C++17。信息形式：Ω=P⁻¹（信息矩阵），ξ=P⁻¹x（信息向量）。
//   预测: Ω⁻ = (F Ω⁻¹ Fᵀ + Q)⁻¹,  ξ⁻ = Ω⁻ F Ω⁻¹ ξ
//   更新: Ω = Ω⁻ + Hᵀ R⁻¹ H,      ξ = ξ⁻ + Hᵀ R⁻¹ z   ← 更新是"简单相加"！
// 验证：与标准 KF 数值等价（逐位一致 ~1e-9）。
// 工程意义：无测量时 Ω 只会衰减、更新只需加法，适合稀疏 SLAM。
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 21;
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
// 通用高斯消元求逆
static Mat inv(const Mat& A) {
    int n = A.size();
    Mat M(n, Vec(2 * n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) M[i][j] = A[i][j];
        M[i][n + i] = 1.0;
    }
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

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.2 信息滤波器 vs 标准 KF ===\n");

    double dt = 0.1, q = 0.05, r = 0.5;
    int N = 60;
    Mat F = {{1.0, dt}, {0.0, 1.0}};
    Mat Q = {{q * dt*dt*dt*dt / 4.0, q * dt*dt*dt / 2.0},
             {q * dt*dt*dt / 2.0,    q * dt}};
    Mat H = {{1.0, 0.0}};
    Mat R = {{r}};
    Mat Ht = transpose(H);
    Mat Rinv = inv(R);
    Mat HRH = matmul(Ht, matmul(Rinv, H));   // HᵀR⁻¹H（2×2，只有 [0][0]=1/r）

    Vec xt = {0.0, 1.0};
    Vec xk = {0.0, 0.5};                       // KF 状态
    Mat P = eye(2); P[0][0] = P[1][1] = 10.0;
    Mat Om = inv(P);                           // IF 信息矩阵
    Vec xi = matvec(Om, xk);                   // IF 信息向量

    for (int k = 0; k < N; ++k) {
        double a = gauss() * sqrt(q);
        xt = {xt[0] + dt * xt[1] + 0.5 * dt * dt * a, xt[1] + dt * a};
        double z = xt[0] + gauss() * sqrt(r);
        // --- KF ---
        xk = matvec(F, xk);
        P = addm(matmul(matmul(F, P), transpose(F)), Q);
        double S = P[0][0] + r;
        double k0 = P[0][0] / S, k1 = P[1][0] / S;
        double e = z - xk[0];
        xk = {xk[0] + k0 * e, xk[1] + k1 * e};
        Mat IK = subm(eye(2), {{k0, 0.0}, {k1, 0.0}});
        Mat KRKt = {{k0 * r * k0, k0 * r * k1}, {k1 * r * k0, k1 * r * k1}};
        P = addm(matmul(matmul(IK, P), transpose(IK)), KRKt);
        // --- IF ---
        Mat Pprev = inv(Om);
        Vec xprev = matvec(Pprev, xi);
        Vec xm = matvec(F, xprev);
        Mat Pm = addm(matmul(matmul(F, Pprev), transpose(F)), Q);
        Mat Omm = inv(Pm);
        Vec xim = matvec(Omm, xm);
        Om = addm(Omm, HRH);                       // Ω = Ω⁻ + HᵀR⁻¹H（纯加法！）
        Vec hRz = {matmul(Ht, Rinv)[0][0] * z, 0.0}; // HᵀR⁻¹z
        xi = {xim[0] + hRz[0], xim[1] + hRz[1]};     // ξ = ξ⁻ + HᵀR⁻¹z
    }
    Vec x_if = matvec(inv(Om), xi);
    Mat P_if = inv(Om);
    double dx = fmax(fabs(x_if[0] - xk[0]), fabs(x_if[1] - xk[1]));
    double dP = 0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) dP = fmax(dP, fabs(P_if[i][j] - P[i][j]));
    printf("  [demo] KF x=(%.5f, %.5f)  IF x=(%.5f, %.5f)\n", xk[0], xk[1], x_if[0], x_if[1]);
    chk("IF vs KF 状态等价", dx, 1e-9);
    chk("IF vs KF 协方差等价", dP, 1e-9);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
