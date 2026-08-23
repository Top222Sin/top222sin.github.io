// jac_l3_cnn.cpp  —  深度学习：一维 CNN 的雅可比
//
// 把“CNN 的雅可比 ∂输出/∂输入”这个看似高深的对象，拆成你早已熟悉的零件：
//   一维 CNN = 卷积(Toeplitz 稀疏线性映射) → ReLU(对角) → 卷积 → ReLU → 全连接(稠密矩阵)
// 整张雅可比 = 这些零件雅可比的链式法则乘积。
//
// 关键技巧：反向传播时把“输出梯度”设为 one-hot e_j，反向传到输入就得到 J 的第 j 列。
// 即  ∂y/∂x = backprop( identity ) 。本 demo 用该法算 J(2×10)，并与有限差分对拍。
//
// 网络：in(1ch, L=10) → Conv1(3ch, k=3) → ReLU → Conv2(2ch, k=3) → ReLU → FC(12→2)
// 仅用 C++17 标准库。

#include <cstdio>
#include <cmath>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;                       // 行优先二维
using Ten = std::vector<std::vector<Vec>>;          // [Cout][Cin][K] 卷积核

// 确定性 RNG（线性同余），保证每次运行权重一致
static unsigned long long rng_state = 2024ULL;
static double urand(double lo, double hi) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(rng_state >> 11) / (double)(1ULL << 53);
    return lo + u * (hi - lo);
}

static Vec relu(const Vec& v) {
    Vec r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = v[i] > 0 ? v[i] : 0.0;
    return r;
}
static Vec d_relu(const Vec& v) {
    Vec r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = v[i] > 0 ? 1.0 : 0.0;
    return r;
}

// 1D valid 卷积：in_ch[Cin][Lin] -> out[Cout][Lout], Lout=Lin-K+1
static Mat conv_valid(const Mat& in_ch, const Ten& W, const Vec& B, int K) {
    int Cin = (int)in_ch.size(), Lin = (int)in_ch[0].size();
    int Lout = Lin - K + 1, Cout = (int)W.size();
    Mat out(Cout, Vec(Lout, 0.0));
    for (int co = 0; co < Cout; ++co)
        for (int n = 0; n < Lout; ++n) {
            double s = B[co];
            for (int ci = 0; ci < Cin; ++ci)
                for (int k = 0; k < K; ++k)
                    s += W[co][ci][k] * in_ch[ci][n + k];
            out[co][n] = s;
        }
    return out;
}

static const int L0 = 10, C1 = 3, K1 = 3, L1 = L0 - K1 + 1;
static const int C2 = 2,  K2 = 3, L2 = L1 - K2 + 1;
static const int DOUT = 2;

static Ten W1, W2; Vec b1, b2; Mat Wfc; Vec bfc;

static void init_net() {
    W1.assign(C1, Mat(1, Vec(K1)));
    for (int c = 0; c < C1; ++c)
        for (int k = 0; k < K1; ++k) W1[c][0][k] = urand(-0.8, 0.8);
    W2.assign(C2, Mat(C1, Vec(K2)));
    for (int c = 0; c < C2; ++c)
        for (int ci = 0; ci < C1; ++ci)
            for (int k = 0; k < K2; ++k) W2[c][ci][k] = urand(-0.8, 0.8);
    b1.assign(C1, 0.0); b2.assign(C2, 0.0);
    for (int i = 0; i < C1; ++i) b1[i] = urand(-0.3, 0.3);
    for (int i = 0; i < C2; ++i) b2[i] = urand(-0.3, 0.3);
    Wfc.assign(DOUT, Vec(C2 * L2));
    for (int o = 0; o < DOUT; ++o)
        for (int i = 0; i < C2 * L2; ++i) Wfc[o][i] = urand(-0.8, 0.8);
    bfc.assign(DOUT, 0.0);
    for (int o = 0; o < DOUT; ++o) bfc[o] = urand(-0.3, 0.3);
}

static Vec forward(const Vec& x) {
    Mat xin(1, x);
    Mat cv1 = conv_valid(xin, W1, b1, K1);
    Vec h1;
    for (int c = 0; c < C1; ++c) for (int n = 0; n < L1; ++n) h1.push_back(cv1[c][n]);
    h1 = relu(h1);
    Mat h1ch(C1, Vec(L1));
    for (int c = 0; c < C1; ++c) for (int n = 0; n < L1; ++n) h1ch[c][n] = h1[c * L1 + n];
    Mat cv2 = conv_valid(h1ch, W2, b2, K2);
    Vec h2;
    for (int c = 0; c < C2; ++c) for (int n = 0; n < L2; ++n) h2.push_back(cv2[c][n]);
    h2 = relu(h2);
    Vec y(DOUT, 0.0);
    for (int o = 0; o < DOUT; ++o) {
        double s = bfc[o];
        for (int i = 0; i < C2 * L2; ++i) s += Wfc[o][i] * h2[i];
        y[o] = s;
    }
    return y;
}

// 雅可比：backprop-identity。返回 J (DOUT × L0)
static Mat jacobian_backprop(const Vec& x) {
    Mat J(DOUT, Vec(L0, 0.0));
    // 前向缓存（用于 ReLU 掩码）
    Mat xin(1, x);
    Mat cv1 = conv_valid(xin, W1, b1, K1);
    Vec h1; for (int c = 0; c < C1; ++c) for (int n = 0; n < L1; ++n) h1.push_back(cv1[c][n]);
    Vec h1r = relu(h1);
    Mat h1ch(C1, Vec(L1));
    for (int c = 0; c < C1; ++c) for (int n = 0; n < L1; ++n) h1ch[c][n] = h1r[c * L1 + n];
    Mat cv2 = conv_valid(h1ch, W2, b2, K2);
    Vec h2; for (int c = 0; c < C2; ++c) for (int n = 0; n < L2; ++n) h2.push_back(cv2[c][n]);
    Vec h2r = relu(h2);

    Vec m1 = d_relu(h1r), m2 = d_relu(h2r);

    for (int o = 0; o < DOUT; ++o) {
        // 输出 one-hot 梯度
        Vec g_y(DOUT, 0.0); g_y[o] = 1.0;
        // FC 反传 -> g_h2
        Vec g_h2(C2 * L2, 0.0);
        for (int i = 0; i < C2 * L2; ++i)
            for (int oo = 0; oo < DOUT; ++oo) g_h2[i] += Wfc[oo][i] * g_y[oo];
        // 重塑为 C2 × L2 并乘 ReLU 掩码
        Mat g_cv2(C2, Vec(L2));
        for (int c = 0; c < C2; ++c) for (int n = 0; n < L2; ++n) g_cv2[c][n] = g_h2[c * L2 + n] * m2[c * L2 + n];
        // Conv2 反传 -> g_h1ch (C1 × L1)
        Mat g_h1ch(C1, Vec(L1, 0.0));
        for (int co = 0; co < C2; ++co)
            for (int ci = 0; ci < C1; ++ci)
                for (int n = 0; n < L2; ++n)
                    for (int k = 0; k < K2; ++k)
                        g_h1ch[ci][n + k] += W2[co][ci][k] * g_cv2[co][n];
        // ReLU 反传 -> g_h1flat
        Vec g_h1flat(C1 * L1, 0.0);
        for (int c = 0; c < C1; ++c) for (int n = 0; n < L1; ++n)
            g_h1flat[c * L1 + n] = g_h1ch[c][n] * m1[c * L1 + n];
        // Conv1 反传 -> g_xin (1 × L0)
        Vec g_xin(L0, 0.0);
        for (int c = 0; c < C1; ++c)
            for (int n = 0; n < L1; ++n)
                for (int k = 0; k < K1; ++k)
                    g_xin[n + k] += W1[c][0][k] * g_h1flat[c * L1 + n];
        for (int j = 0; j < L0; ++j) J[o][j] = g_xin[j];
    }
    return J;
}

static Mat jacobian_fd(const Vec& x) {
    Vec y0 = forward(x);
    Mat J(DOUT, Vec(L0, 0.0));
    double h = 1e-6;
    for (int j = 0; j < L0; ++j) {
        Vec xp = x; xp[j] += h;
        Vec yp = forward(xp);
        for (int o = 0; o < DOUT; ++o) J[o][j] = (yp[o] - y0[o]) / h;
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
    init_net();
    Vec x(L0);
    for (int i = 0; i < L0; ++i) x[i] = urand(-1.0, 1.0);

    Mat Jbp = jacobian_backprop(x);
    Mat Jfd = jacobian_fd(x);
    double err = max_err(Jbp, Jfd);

    printf("=== 一维 CNN 雅可比 (backprop-identity 对拍有限差分) ===\n");
    printf("  Jacobian 维度: %d x %d  (输出通道 x 输入长度)\n", DOUT, L0);
    printf("  backprop J  vs  有限差分 J : err=%.2e  %s\n", err, err < 1e-5 ? "PASS" : "FAIL");
    Vec y = forward(x);
    printf("  样例输出 y = [%.4f, %.4f]\n", y[0], y[1]);
    printf("  J 第一行(∂y0/∂x): ");
    for (int j = 0; j < L0; ++j) printf("%.3f ", Jbp[0][j]);
    printf("\n");
    return err < 1e-5 ? 0 : 1;
}
