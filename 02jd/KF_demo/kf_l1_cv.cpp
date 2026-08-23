// ============================================================
// KF_demo L1.4 二维常速模型 CV-KF：完整例子 + Joseph 形式
// 纯 C++17。状态 x=[p,v]，F=[[1,dt],[0,1]]，H=[[1,0]]（只测位置）。
// 验证：
//  ① 位置滤波 RMSE < 观测 RMSE；速度从位置序列被"恢复"
//  ② 精确离散 Q（每步独立加速度）：q·[[dt⁴/4, dt³/2],[dt³/2, dt]]
//  ③ Joseph 形式更新保持 P 对称正定
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 7;
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
    Mat C(A);
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j];
    return C;
}
static Mat subm(const Mat& A, const Mat& B) {
    Mat C(A);
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] -= B[i][j];
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
    printf("=== L1.4 常速模型 CV-KF（含 Joseph 形式）===\n");

    double dt = 0.1, q = 0.05, r = 0.5;
    int N = 400;
    Mat F = {{1.0, dt}, {0.0, 1.0}};
    // 精确离散 Q（每步独立采加速度 a~N(0,q)，作用于 (0.5dt²a, dt·a)）
    Mat Q = {{q * dt * dt * dt * dt / 4.0, q * dt * dt * dt / 2.0},
             {q * dt * dt * dt / 2.0,      q * dt}};
    Mat H = {{1.0, 0.0}};
    Mat R = {{r}};

    Vec xt = {0.0, 1.0};           // 真值初值
    Vec x = {0.0, 0.5};            // 滤波初值（故意偏）
    Mat P = eye(2); P[0][0] = P[1][1] = 10.0;

    double e2p = 0, e2v = 0, e2z = 0; int cnt = 0;
    for (int k = 0; k < N; ++k) {
        double a = gauss() * sqrt(q);
        xt = {xt[0] + dt * xt[1] + 0.5 * dt * dt * a, xt[1] + dt * a};
        double z = xt[0] + gauss() * sqrt(r);

        // 预测
        x = matvec(F, x);
        P = addm(matmul(matmul(F, P), transpose(F)), Q);
        // 更新（Joseph 形式）
        Mat Ht = transpose(H);
        double S = (matmul(matmul(H, P), Ht))[0][0] + R[0][0];   // 1×1
        Vec Kh = {P[0][0] / S, P[1][0] / S};                      // K = P Hᵀ / S（2×1）
        double resid = z - x[0];
        x = {x[0] + Kh[0] * resid, x[1] + Kh[1] * resid};
        Mat IK = subm(eye(2), {{Kh[0], 0.0}, {Kh[1], 0.0}});      // I - K·H
        Mat KRKt = {{Kh[0] * r * Kh[0], Kh[0] * r * Kh[1]},
                    {Kh[1] * r * Kh[0], Kh[1] * r * Kh[1]}};
        P = addm(matmul(matmul(IK, P), transpose(IK)), KRKt);

        if (k >= 50) {   // 跳过暂态
            e2p += (x[0] - xt[0]) * (x[0] - xt[0]);
            e2v += (x[1] - xt[1]) * (x[1] - xt[1]);
            e2z += (z - xt[0]) * (z - xt[0]);
            ++cnt;
        }
    }
    double rmse_p = sqrt(e2p / cnt), rmse_v = sqrt(e2v / cnt), rmse_z = sqrt(e2z / cnt);
    printf("  [demo] rmse_pos=%.4f rmse_vel=%.4f rmse_meas=%.4f\n", rmse_p, rmse_v, rmse_z);
    chk("位置滤波RMSE < 观测RMSE", rmse_p < rmse_z ? 0.0 : 1.0, 0.5);
    chk("速度RMSE ≈ √P_vv (<0.3)", rmse_v, 0.3);
    chk("Joseph P 对称", fabs(P[0][1] - P[1][0]), 1e-12);
    double det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
    chk("Joseph P 正定", det > 0 ? 0.0 : 1.0, 0.5);
    printf("  [demo] 末态 P=[[%.5f, %.5f],[%.5f, %.5f]] (√P_vv=%.4f)\n",
           P[0][0], P[0][1], P[1][0], P[1][1], sqrt(P[1][1]));

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
