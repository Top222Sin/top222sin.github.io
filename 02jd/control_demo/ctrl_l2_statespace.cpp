// ============================================================
// control_demo L2.1 状态空间与矩阵指数 e^{At}
// 纯 C++17。缩方+泰勒实现 expm。
// 验证：① 旋转矩阵 e^{At}（解析 cos/sin 对拍）；
//       ② 双积分器 e^{At}=[[1,t],[0,1]]；
//       ③ ZOH 精确离散化（Ad=e^{AΔ}, Bd=ΣA^kΔ^{k+1}/((k+1)k!)）
//         与 RK4 连续仿真逐位对拍
// ⚠ 坑：Bd 级数分母是 (k+1)·k! 而非 (k+1)（漏 k! 会让 k=2 项差 2 倍）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat eye(int n) { Mat I(n, vector<double>(n, 0)); for (int i = 0; i < n; ++i) I[i][i] = 1; return I; }
static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, vector<double>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}
static Mat addm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j];
    return C;
}
static Mat scalem(const Mat& A, double s) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] *= s;
    return C;
}
static Mat expm(const Mat& A, double t) {
    int n = A.size();
    Mat At = scalem(A, t);
    double e = 0;
    for (auto& r : At) for (double v : r) e = fmax(e, fabs(v));
    int k = 0;
    while (e / (1 << k) > 0.25) ++k;
    Mat As = scalem(At, 1.0 / (1 << k));
    Mat S = eye(n), term = eye(n);
    for (int i = 1; i < 20; ++i) {
        term = matmul(term, As);
        term = scalem(term, 1.0 / i);
        S = addm(S, term);
    }
    for (int i = 0; i < k; ++i) S = matmul(S, S);
    return S;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.1 状态空间与 e^{At} ===\n");

    {   // ① 旋转矩阵
        double th = 1.2;
        Mat A = {{0, -th}, {th, 0}};
        Mat E = expm(A, 1.0);
        double err = fmax(fabs(E[0][0] - cos(th)), fabs(E[0][1] + sin(th)));
        err = fmax(err, fabs(E[1][0] - sin(th)));
        chk("e^At 旋转矩阵 (cos/sin)", err, 1e-12);
    }
    {   // ② 双积分器
        Mat A = {{0, 1}, {0, 0}};
        Mat E = expm(A, 0.7);
        chk("双积分器 e^At = [[1,t],[0,1]]", fabs(E[0][1] - 0.7), 1e-12);
    }
    {   // ③ ZOH 离散化 vs RK4
        Mat A = {{0, 1}, {-4.0, -0.8}};
        double Bv[2] = {0, 1};
        double dt = 0.01;
        Mat Ad = expm(A, dt);
        double Bd[2] = {0, 0};
        Mat term = eye(2);
        double fact = 1.0;
        for (int k = 0; k < 12; ++k) {
            for (int j = 0; j < 2; ++j)
                for (int c = 0; c < 2; ++c) Bd[j] += term[j][c] * Bv[c] * pow(dt, k + 1) / ((k + 1) * fact);
            fact *= (k + 1);
            term = matmul(term, A);
        }
        double xd[2] = {1, 0}, xc[2] = {1, 0};
        for (int i = 0; i < 300; ++i) {
            // 离散
            double nx[2];
            for (int j = 0; j < 2; ++j) nx[j] = Ad[j][0] * xd[0] + Ad[j][1] * xd[1] + Bd[j];
            xd[0] = nx[0]; xd[1] = nx[1];
            // RK4 ×10 子步
            auto f = [](const double x[2], double dx[2]) { dx[0] = x[1]; dx[1] = -4 * x[0] - 0.8 * x[1] + 1; };
            double hs = dt / 10;
            for (int s = 0; s < 10; ++s) {
                double k1[2], k2[2], k3[2], k4[2], t2[2];
                f(xc, k1);
                t2[0] = xc[0] + hs / 2 * k1[0]; t2[1] = xc[1] + hs / 2 * k1[1]; f(t2, k2);
                t2[0] = xc[0] + hs / 2 * k2[0]; t2[1] = xc[1] + hs / 2 * k2[1]; f(t2, k3);
                t2[0] = xc[0] + hs * k3[0]; t2[1] = xc[1] + hs * k3[1]; f(t2, k4);
                for (int j = 0; j < 2; ++j) xc[j] += hs / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
            }
        }
        chk("ZOH 离散化 = RK4 连续", fmax(fabs(xd[0] - xc[0]), fabs(xd[1] - xc[1])), 1e-6);
        printf("  [demo] 300 步后 x=[%.5f, %.5f]\n", xd[0], xd[1]);
    }
    printf("  [结论] e^{At} 是连续→离散的桥梁；ZOH 假设 = 数字保持器采样。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
