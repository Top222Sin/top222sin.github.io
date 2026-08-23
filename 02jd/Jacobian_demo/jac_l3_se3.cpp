// jac_l3_se3.cpp  —  SE(3) 完整位姿雅可比
//
// 雅可比矩阵在 SE(3)（三维刚体运动群）上的系统用法。
// 与 L3.1（SE(2) 位姿图）对应，这里把“位姿”升级为完整的 6 自由度：
//   旋转(3) + 平移(3) = 李代数 se(3) 中的一个“旋量/twist” ξ = (ω, v)。
//
// 演示 4 件事（全部与有限差分/恒等式对拍，打印 PASS/FAIL）：
//   1) exp_se3 / log_se3 互逆（刚体运动 = 旋量的指数映射）
//   2) 伴随矩阵 Ad(T) 满足  Ad(T1·T2) = Ad(T1)·Ad(T2)
//   3) 点作用雅可比  J(ξ)=∂(exp(ξ̂)x)/∂ξ = [−R[x]^, R]·J_r(ξ)   （J_r 为 SE(3) 右雅可比）
//   4) 旋量变换（链式法则）：d/dt Log(T2·exp(tξ)·T2⁻¹)|₀ = Ad(T2)·ξ
//
// 仅用 C++17 标准库。无 Eigen / OpenCV。

#include <cstdio>
#include <cmath>
#include <vector>

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

static Mat zeros(int r, int c) { return Mat(r, Vec(c, 0.0)); }
static Mat eye(int n) {
    Mat M = zeros(n, n);
    for (int i = 0; i < n; ++i) M[i][i] = 1.0;
    return M;
}
static Mat matmul(const Mat& A, const Mat& B) {
    int n = (int)A.size(), m = (int)B[0].size(), k = (int)B.size();
    Mat C = zeros(n, m);
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) {
            double a = A[i][p];
            if (a == 0) continue;
            for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j];
        }
    return C;
}
static Mat transpose(const Mat& A) {
    int n = (int)A.size(), m = (int)A[0].size();
    Mat T = zeros(m, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) T[j][i] = A[i][j];
    return T;
}
// 3x3 反对称矩阵
static Mat skew(const Vec& v) {
    Mat S = zeros(3, 3);
    S[0][1] = -v[2]; S[0][2] = v[1];
    S[1][0] =  v[2]; S[1][2] = -v[0];
    S[2][0] = -v[1]; S[2][1] =  v[0];
    return S;
}
static Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

// ---- SE(3) 指数映射：T = exp(ξ̂) ----
static Mat exp_se3(const Vec& xi) {
    const Vec& w = {xi[0], xi[1], xi[2]};
    const Vec& v = {xi[3], xi[4], xi[5]};
    double th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    Mat R, V;
    if (th < 1e-12) {
        R = eye(3); V = eye(3);
    } else {
        Vec wn = {w[0]/th, w[1]/th, w[2]/th};
        Mat K = skew(wn), K2 = matmul(K, K);
        double c = std::cos(th), s = std::sin(th);
        // R = I + s·K + (1-c)·K²     （K 为单位反对称，故无需 /θ）
        R = eye(3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R[i][j] += s*K[i][j] + (1-c)*K2[i][j];
        // V = I + ((1-c)/θ)·K + ((θ-s)/θ)·K²
        V = eye(3);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                V[i][j] += ((1-c)/th)*K[i][j] + ((th-s)/th)*K2[i][j];
    }
    Vec p(3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) p[i] += V[i][j] * v[j];
    Mat T = zeros(4, 4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T[i][j] = R[i][j];
        T[i][3] = p[i];
    }
    T[3][3] = 1.0;
    return T;
}

// 3x3 高斯消元求逆
static Mat inv3(const Mat& M) {
    Mat A(3, Vec(6, 0.0));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) A[i][j] = M[i][j];
        A[i][3+i] = 1.0;
    }
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int r = c+1; r < 3; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        double d = A[c][c];
        for (int j = 0; j < 6; ++j) A[c][j] /= d;
        for (int r = 0; r < 3; ++r) if (r != c) {
            double f = A[r][c];
            for (int j = 0; j < 6; ++j) A[r][j] -= f * A[c][j];
        }
    }
    Mat M2 = zeros(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M2[i][j] = A[i][3+j];
    return M2;
}

// ---- SE(3) 对数映射：ξ = log(T) ----
static Vec log_se3(const Mat& T) {
    Mat R(3, Vec(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = T[i][j];
    Vec p = {T[0][3], T[1][3], T[2][3]};
    double tr = R[0][0] + R[1][1] + R[2][2];
    double cos_th = (tr - 1) * 0.5;
    cos_th = std::fmax(-1.0, std::fmin(1.0, cos_th));
    double th = std::acos(cos_th);
    Vec w(3, 0.0);
    if (th > 1e-9) {
        double s = std::sin(th);
        if (std::fabs(s) < 1e-9) {            // θ ≈ π
            int i = 0;
            for (int k = 0; k < 3; ++k) if (R[k][k] > R[i][i]) i = k;
            w[i] = std::sqrt(R[i][i] + 1.0);
            int j = (i+1)%3, k = (i+2)%3;
            if ((R[i][j] - R[j][i]) < 0) w[i] = -w[i];
            w[j] = (R[i][j] + R[j][i]) / (2.0 * w[i]);
            w[k] = (R[i][k] + R[k][i]) / (2.0 * w[i]);
            for (int t = 0; t < 3; ++t) w[t] *= th;
        } else {                               // 一般情形
            Mat K(3, Vec(3));
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    K[i][j] = (R[i][j] - R[j][i]) * th / (2.0*s);
            w = {K[2][1], K[0][2], K[1][0]};
        }
    } else {                                  // θ ≈ 0
        w = {(R[2][1]-R[1][2])*0.5, (R[0][2]-R[2][0])*0.5, (R[1][0]-R[0][1])*0.5};
    }
    double tc = std::fmax(th, 1e-12);
    Mat wh = skew(w), wh2 = matmul(wh, wh);
    double c = std::cos(tc), s = std::sin(tc);
    // V = I + ((1-c)/θ²)·wh + ((θ-s)/θ³)·wh²   （wh 为完整反对称）
    Mat V = eye(3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            V[i][j] += ((1-c)/(tc*tc))*wh[i][j] + ((tc-s)/(tc*tc*tc))*wh2[i][j];
    Mat Vi = inv3(V);
    Vec vpt(3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) vpt[i] += Vi[i][j] * p[j];
    return {w[0], w[1], w[2], vpt[0], vpt[1], vpt[2]};
}

// 伴随矩阵 Ad(T)
static Mat Ad(const Mat& T) {
    Mat R(3, Vec(3)), M = zeros(6, 6);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = T[i][j];
    Vec p = {T[0][3], T[1][3], T[2][3]};
    Mat pR = matmul(skew(p), R);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            M[i][j] = R[i][j];
            M[3+i][3+j] = R[i][j];
            M[3+i][j] = pR[i][j];
        }
    return M;
}
static Mat Ad_inv(const Mat& T) {
    Mat R(3, Vec(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = T[i][j];
    Vec p = {T[0][3], T[1][3], T[2][3]};
    Mat RT = transpose(R);
    Vec pp = matvec(RT, p);
    Mat pR = matmul(skew({-pp[0], -pp[1], -pp[2]}), RT);
    Mat M = zeros(6, 6);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            M[i][j] = RT[i][j];
            M[3+i][3+j] = RT[i][j];
            M[3+i][j] = pR[i][j];
        }
    return M;
}
static Mat inv_se3(const Mat& T) {
    Mat R(3, Vec(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = T[i][j];
    Mat RT = transpose(R);
    Vec p = {T[0][3], T[1][3], T[2][3]};
    Vec pinv(3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) pinv[i] -= RT[i][j] * p[j];
    Mat M = zeros(4, 4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = RT[i][j];
        M[i][3] = pinv[i];
    }
    M[3][3] = 1.0;
    return M;
}

// SE(3) 右雅可比 J_r(ξ)：由定义（log-of-product 的有限差分）稳健求出
static Mat Jr_se3(const Vec& xi) {
    Mat T = exp_se3(xi), Tinv = inv_se3(T);
    Mat Jr = zeros(6, 6);
    double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec d(6, 0.0); d[k] = h;
        Vec xik(6);
        for (int i = 0; i < 6; ++i) xik[i] = xi[i] + d[i];
        Mat Td = exp_se3(xik);
        Vec g = log_se3(matmul(Tinv, Td));
        for (int i = 0; i < 6; ++i) Jr[i][k] = g[i] / h;
    }
    return Jr;
}

static Vec action(const Mat& T, const Vec& x) {
    Vec y(3, 0.0);
    for (int i = 0; i < 3; ++i)
        y[i] = T[i][3] + T[i][0]*x[0] + T[i][1]*x[1] + T[i][2]*x[2];
    return y;
}

// 有限差分的“点作用”雅可比
static Mat fd_action_J(const Mat& T, const Vec& x) {
    Vec xi0 = log_se3(T);
    Vec y0 = action(T, x);
    Mat J = zeros(3, 6);
    double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec d(6, 0.0); d[k] = h;
        Vec xik(6); for (int i = 0; i < 6; ++i) xik[i] = xi0[i] + d[i];
        Vec yk = action(exp_se3(xik), x);
        for (int i = 0; i < 3; ++i) J[i][k] = (yk[i] - y0[i]) / h;
    }
    return J;
}

static double max_err(const Mat& A, const Mat& B) {
    double m = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j)
            m = std::fmax(m, std::fabs(A[i][j] - B[i][j]));
    return m;
}

int main() {
    int pass = 0, fail = 0;
    auto check = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-28s err=%.2e  %s\n", name, err, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== SE(3) 完整位姿雅可比 ===\n");

    // 1) exp/log 互逆
    double rt = 0.0;
    for (int t = 0; t < 30; ++t) {
        Vec xi(6);
        for (int i = 0; i < 6; ++i) xi[i] = ((t*37 + i*13) % 100)/50.0 - 1.0; // 确定性散布
        Vec x2 = log_se3(exp_se3(xi));
        for (int i = 0; i < 6; ++i) rt = std::fmax(rt, std::fabs(xi[i]-x2[i]));
    }
    check("exp/log roundtrip", rt, 1e-9);

    // 2) 伴随：Ad(T1·T2) = Ad(T1)·Ad(T2)
    double ad = 0.0;
    for (int t = 0; t < 10; ++t) {
        Vec a(6), b(6);
        for (int i = 0; i < 6; ++i) { a[i] = ((t*17+i*5)%100)/50.0-1.0; b[i] = ((t*29+i*7)%100)/50.0-1.0; }
        Mat T1 = exp_se3(a), T2 = exp_se3(b);
        ad = std::fmax(ad, max_err(matmul(Ad(T1), Ad(T2)), Ad(matmul(T1, T2))));
    }
    check("Ad(T1 T2)=Ad(T1)Ad(T2)", ad, 1e-9);

    // 3) 点作用雅可比 = [−R[x]^, R]·J_r(ξ)
    double aj = 0.0;
    for (int t = 0; t < 10; ++t) {
        Vec xi(6), x(3);
        for (int i = 0; i < 6; ++i) xi[i] = (((t*11+i*3)%40)-20)/50.0;
        for (int i = 0; i < 3; ++i) x[i] = (((t*5+i)%40)-20)/20.0;
        Mat T = exp_se3(xi);
        Mat R(3, Vec(3));
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) R[i][j]=T[i][j];
        Mat Rsx = matmul(R, skew(x));
        Mat M = zeros(3, 6);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) M[i][j] = -Rsx[i][j];          // −R[x]^
            for (int j = 0; j < 3; ++j) M[i][3+j] = R[i][j];          //  R
        }
        Mat Jana = matmul(M, Jr_se3(xi));
        Mat Jfd = fd_action_J(T, x);
        aj = std::fmax(aj, max_err(Jana, Jfd));
    }
    check("action J=[-Rx^,R]Jr", aj, 1e-5);

    // 4) 旋量变换（链式法则）：d/dt Log(T2 exp(tξ) T2⁻¹)|₀ = Ad(T2)·ξ
    double cg = 0.0; double h = 1e-6;
    for (int t = 0; t < 10; ++t) {
        Vec x2(6), xi(6);
        for (int i=0;i<6;++i){ x2[i]=(((t*13+i)%40)-20)/50.0; xi[i]=(((t*19+i*2)%30)-15)/50.0; }
        Mat T2 = exp_se3(x2), T2inv = inv_se3(T2);
        Vec hxi(6); for (int i=0;i<6;++i) hxi[i]=h*xi[i];
        Mat Sh = matmul(matmul(T2, exp_se3(hxi)), T2inv);
        Vec logSh = log_se3(Sh);
        Vec pred = matvec(Ad(T2), xi);
        for (int i=0;i<6;++i) cg = std::fmax(cg, std::fabs(logSh[i]-h*pred[i]));
    }
    check("conjugation Ad twist", cg, 1e-5);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
