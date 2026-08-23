// L4.11 分位数回归（作为 LS）
// 纯 C++17 标准库实现；算法经受管 Python 3.13.12 验证（5/5 PASS）+ C++ 控制流镜像复核。
// 角度：OLS 最小化 Σ(y−Xβ)²（对称 L2）；分位数回归把平方损失换成非对称 check/pinball
//       损失 ρ_τ(r)=r(τ−1_{r<0})（正残差权重 τ、负残差权重 τ−1），用平滑 check loss 的
//       梯度下降（或其 LS 式 IRLS 重加权）求解——与 LS 同骨架。
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

static Mat matmul(const Mat& A, const Mat& B) {
    int n = (int)A.size(), m = (int)A[0].size(), p = (int)B[0].size();
    Mat C(n, Vec(p, 0.0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < m; k++) {
            double a = A[i][k];
            if (a == 0.0) continue;
            const Vec& Bk = B[k];
            for (int j = 0; j < p; j++) C[i][j] += a * Bk[j];
        }
    return C;
}
static Vec matvec(const Mat& A, const Vec& x) {
    int n = (int)A.size(), m = (int)A[0].size(); Vec y(n, 0.0);
    for (int i = 0; i < n; i++) {
        double s = 0.0; for (int j = 0; j < m; j++) s += A[i][j]*x[j];
        y[i] = s;
    }
    return y;
}
static Mat transpose(const Mat& A) {
    int n = (int)A.size(), m = (int)A[0].size(); Mat T(m, Vec(n, 0.0));
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
}
static Vec solve(const Mat& A_, const Vec& b) {
    int n = (int)A_.size();
    Mat M(n, Vec(n+1, 0.0));
    for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) M[i][j] = A_[i][j]; M[i][n] = b[i]; }
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col; r < n; r++) if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        std::swap(M[col], M[piv]);
        double d = M[col][col];
        for (int j = col; j <= n; j++) M[col][j] /= d;
        for (int r = 0; r < n; r++) if (r != col && M[r][col] != 0.0) {
            double f = M[r][col]; for (int j = col; j <= n; j++) M[r][j] -= f*M[col][j];
        }
    }
    Vec x(n); for (int i = 0; i < n; i++) x[i] = M[i][n];
    return x;
}

struct Rng { long long state; double operator()() {
    state = (state*1103515245LL + 12345) & 0x7fffffffLL;
    return (double)state / 2147483647.0;
} };

static const int N = 200;

static void gen_data(Rng& rng, Mat& X, Vec& y) {
    X.assign(N, Vec(2, 0.0)); y.assign(N, 0.0);
    for (int i = 0; i < N; i++) {
        double xi = 2.0*rng() - 1.0;
        double u1 = rng(); double u2 = rng();
        if (u1 <= 1e-12) u1 = 1e-12;
        double z = std::sqrt(-2.0*std::log(u1)) * std::cos(2.0*M_PI*u2);
        double sigma = 0.4 + 0.3*std::fabs(xi);
        double yi = 1.0 + 2.0*xi + sigma*z;
        X[i][0] = 1.0; X[i][1] = xi; y[i] = yi;
    }
}
static Vec ols(const Mat& X, const Vec& y) {
    Mat Xt = transpose(X); Mat XtX = matmul(Xt, X); Vec Xty = matvec(Xt, y);
    return solve(XtX, Xty);
}
static double smooth_loss(const Mat& X, const Vec& y, const Vec& beta, double tau, double delta) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double r = y[i] - (X[i][0]*beta[0] + X[i][1]*beta[1]);
        s += (tau-0.5)*r + 0.5*(std::sqrt(r*r + delta*delta) - delta);
    }
    return s;
}
static Vec smooth_grad(const Mat& X, const Vec& y, const Vec& beta, double tau, double delta) {
    Vec g(2, 0.0);
    for (int i = 0; i < N; i++) {
        double r = y[i] - (X[i][0]*beta[0] + X[i][1]*beta[1]);
        double gi = (tau-0.5) + 0.5*r/std::sqrt(r*r + delta*delta);
        g[0] += -X[i][0]*gi; g[1] += -X[i][1]*gi;
    }
    return g;
}
static Vec fit_quantile(const Mat& X, const Vec& y, double tau, double delta, int n_iter=4000) {
    Vec beta = ols(X, y);
    double lr = 1.0;
    for (int it = 0; it < n_iter; it++) {
        Vec g = smooth_grad(X, y, beta, tau, delta);
        double L0 = smooth_loss(X, y, beta, tau, delta);
        double dot = g[0]*g[0] + g[1]*g[1];
        double t = lr;
        while (t > 1e-12) {
            Vec nb = {beta[0]-t*g[0], beta[1]-t*g[1]};
            if (smooth_loss(X, y, nb, tau, delta) <= L0 - 1e-4*t*dot) break;
            t *= 0.5;
        }
        beta = {beta[0]-t*g[0], beta[1]-t*g[1]};
    }
    return beta;
}
static double check_loss(const Mat& X, const Vec& y, const Vec& beta, double tau) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double r = y[i] - (X[i][0]*beta[0] + X[i][1]*beta[1]);
        if (r >= 0) s += tau*r; else s += (tau-1.0)*r;
    }
    return s;
}
static double frac_neg(const Mat& X, const Vec& y, const Vec& beta) {
    int c = 0;
    for (int i = 0; i < N; i++) {
        double r = y[i] - (X[i][0]*beta[0] + X[i][1]*beta[1]);
        if (r < 0) c++;
    }
    return (double)c / N;
}
static double pred_at(const Vec& beta, double xv) { return beta[0] + beta[1]*xv; }

static bool chk(const std::string& name, double got, double thr, bool ge, const std::string& extra) {
    bool ok; double val;
    if (ge) { ok = got >= thr - 1e-12; val = got; }
    else   { ok = std::fabs(got) <= thr + 1e-12; val = std::fabs(got); }
    printf("  [%s] %s: %.3e (thr=%+.1e) %s\n", ok?"PASS":"FAIL", name.c_str(), val, thr, extra.c_str());
    return ok;
}

int main() {
    Rng rng; rng.state = 20260820;
    Mat X; Vec y; gen_data(rng, X, y);
    double taus[5] = {0.1, 0.3, 0.5, 0.7, 0.9};
    double delta = 0.02;
    Vec beta_tau[5]; double gradn[5];
    for (int k = 0; k < 5; k++) {
        beta_tau[k] = fit_quantile(X, y, taus[k], delta);
        Vec g = smooth_grad(X, y, beta_tau[k], taus[k], delta);
        gradn[k] = std::sqrt(g[0]*g[0] + g[1]*g[1]);
        printf("  tau=%.1f: b0=%.4f b1=%.4f |grad|=%.2e frac(r<0)=%.3f\n",
               taus[k], beta_tau[k][0], beta_tau[k][1], gradn[k], frac_neg(X,y,beta_tau[k]));
    }
    Vec b_ols = ols(X, y);

    int passed = 0, total = 0;
    printf("== L4.11 分位数回归 验证 ==\n");
    total++;
    double worst_cov = 0.0;
    for (int k = 0; k < 5; k++) worst_cov = std::max(worst_cov, std::fabs(frac_neg(X,y,beta_tau[k]) - taus[k]));
    if (chk("分位覆盖 frac(r<0)≈τ", worst_cov, 0.085, false, "max|Δ|="+std::to_string(worst_cov))) passed++;
    total++;
    bool ok_mono = true;
    for (int k = 0; k < 4; k++)
        if (pred_at(beta_tau[k+1],0.5) <= pred_at(beta_tau[k],0.5) + 1e-4) ok_mono = false;
    if (chk("单调性 pred(τ)↑", ok_mono?0.0:1.0, 1e-9, false, "mono=ok")) passed++;
    total++;
    double worst_g = 0.0; for (int k = 0; k < 5; k++) worst_g = std::max(worst_g, gradn[k]);
    if (chk("平滑损失梯度≈0(收敛)", worst_g, 1e-2, false, "max|grad|="+std::to_string(worst_g))) passed++;
    total++;
    double worst4 = 0.0;
    for (int k = 0; k < 5; k++)
        worst4 = std::max(worst4, check_loss(X,y,beta_tau[k],taus[k]) - check_loss(X,y,b_ols,taus[k]));
    if (chk("check loss≤OLS(单侧上界)", std::max(0.0,worst4), 1e-2, false, "max(CL̂−CL_ols)="+std::to_string(worst4))) passed++;
    total++;
    double spread = std::fabs(pred_at(beta_tau[4],0.5) - pred_at(beta_tau[0],0.5));
    if (chk("τ0.9≠τ0.1(捕捉分位)", std::max(0.0,0.1-spread), 1e-9, false, "|pred0.9−pred0.1|="+std::to_string(spread))) passed++;

    printf("PASS=%d  FAIL=%d  (total=%d)\n", passed, total-passed, total);
    printf("RESULT: %s\n", passed==total ? "ALL PASS" : "HAS FAIL");
    return passed==total ? 0 : 1;
}
