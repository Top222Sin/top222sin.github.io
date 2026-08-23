// ============================================================
// KF_demo L2.1 稳态 KF 与 Riccati / DARE 迭代
// 纯 C++17。验证：
//  ① 迭代离散 Riccati 方程收敛到稳态 P∞ 与 K∞
//  ② 用固定 K∞ 的"稳态滤波器"与逐拍全 KF 后期一致
//     （稳态 KF ≡ Wiener 滤波器，工程上可省每拍重算）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 55;
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

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.1 稳态 KF 与 Riccati / DARE ===\n");

    double dt = 0.1, q = 0.05, r = 0.5;
    Mat F = {{1.0, dt}, {0.0, 1.0}};
    Mat Q = {{q * dt*dt*dt*dt / 4.0, q * dt*dt*dt / 2.0},
             {q * dt*dt*dt / 2.0,    q * dt}};
    Mat H = {{1.0, 0.0}};
    Mat R = {{r}};

    // ---- ① 迭代 DARE：P⁻=FPFᵀ+Q；K=P⁻HᵀS⁻¹；P=(I-KH)P⁻(I-KH)ᵀ+KRKᵀ ----
    Mat P = eye(2);
    double Kinf0 = 0, Kinf1 = 0;
    int iters = 0;
    for (int it = 0; it < 10000; ++it) {
        Mat Pm = addm(matmul(matmul(F, P), transpose(F)), Q);
        double S = Pm[0][0] + r;                     // H P⁻ Hᵀ + R（1×1）
        double k0 = Pm[0][0] / S, k1 = Pm[1][0] / S; // K = P⁻Hᵀ/S
        Mat IK = subm(eye(2), {{k0, 0.0}, {k1, 0.0}});
        Mat KRKt = {{k0 * r * k0, k0 * r * k1}, {k1 * r * k0, k1 * r * k1}};
        Mat Pnew = addm(matmul(matmul(IK, Pm), transpose(IK)), KRKt);
        double derr = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) derr = fmax(derr, fabs(Pnew[i][j] - P[i][j]));
        P = Pnew; ++iters;
        if (derr < 1e-14) break;
    }
    // 稳态增益（用收敛 P 算下一拍）
    {
        Mat Pm = addm(matmul(matmul(F, P), transpose(F)), Q);
        double S = Pm[0][0] + r;
        Kinf0 = Pm[0][0] / S; Kinf1 = Pm[1][0] / S;
    }
    printf("  [demo] DARE 收敛于 %d 步: P_ss=[[%.5f, %.5f],[%.5f, %.5f]]\n",
           iters, P[0][0], P[0][1], P[1][0], P[1][1]);
    printf("  [demo] K_inf = [%.5f, %.5f]ᵀ\n", Kinf0, Kinf1);

    // ---- ② 稳态（固定 K）vs 全 KF，后期应一致 ----
    {
        int N = 120;
        Vec xt = {0.0, 1.0};
        Vec x1 = {0.0, 0.5}; Mat P1 = eye(2); P1[0][0] = P1[1][1] = 10.0;
        Vec x2 = {0.0, 0.5};
        for (int k = 0; k < N; ++k) {
            double a = gauss() * sqrt(q);
            xt = {xt[0] + dt * xt[1] + 0.5 * dt * dt * a, xt[1] + dt * a};
            double z = xt[0] + gauss() * sqrt(r);
            // 全 KF
            x1 = matvec(F, x1);
            P1 = addm(matmul(matmul(F, P1), transpose(F)), Q);
            double S = P1[0][0] + r;
            double k0 = P1[0][0] / S, k1 = P1[1][0] / S;
            double e = z - x1[0];
            x1 = {x1[0] + k0 * e, x1[1] + k1 * e};
            Mat IK = subm(eye(2), {{k0, 0.0}, {k1, 0.0}});
            Mat KRKt = {{k0 * r * k0, k0 * r * k1}, {k1 * r * k0, k1 * r * k1}};
            P1 = addm(matmul(matmul(IK, P1), transpose(IK)), KRKt);
            // 稳态 KF：固定 K∞
            x2 = matvec(F, x2);
            double e2 = z - x2[0];
            x2 = {x2[0] + Kinf0 * e2, x2[1] + Kinf1 * e2};
        }
        double diff = fmax(fabs(x1[0] - x2[0]), fabs(x1[1] - x2[1]));
        printf("  [demo] 末态: 全KF=(%.4f,%.4f) 稳态KF=(%.4f,%.4f) 差=%.2e\n",
               x1[0], x1[1], x2[0], x2[1], diff);
        chk("稳态KF = 全KF (后期一致)", diff, 2e-2);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
