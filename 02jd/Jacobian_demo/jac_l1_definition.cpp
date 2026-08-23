// jac_l1_definition.cpp
// ============================================================================
// L1.1 雅可比矩阵的定义与局部线性化
// ----------------------------------------------------------------------------
// 取映射 f: R^2 -> R^2,  f(x,y) = ( x^2 + sin(y),  e^x * y )
// 雅可比 J(p) = df/dp 是"一阶偏导数矩阵"，也是 f 在 p 处的"最佳线性近似"。
// 本 demo 验证两条最核心的性质：
//   (1) 局部线性化： f(p+dp) ≈ f(p) + J(p)·dp
//   (2) 链式法则：   D(g∘f)(p) = Dg(f(p)) · Df(p)
// 全程合成数据，运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l1_definition.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;

// 目标映射 f
static V f(const V& p){
    double x = p[0], y = p[1];
    return { x*x + sin(y), exp(x)*y };
}
// f 的解析雅可比
static M Jf(const V& p){
    double x = p[0], y = p[1];
    return { {2*x, cos(y)}, {exp(x)*y, exp(x)} };
}
// 第二层映射 g: R^2 -> R^2
static V g(const V& p){
    double u = p[0], v = p[1];
    return { sin(u) + v, u*v };
}
// g 的解析雅可比
static M Jg(const V& p){
    double u = p[0], v = p[1];
    return { {cos(u), 1.0}, {v, u} };
}

static V matVec(const M& A, const V& x){
    int m = A.size(), n = A[0].size();
    V y(m, 0.0);
    for (int i = 0; i < m; i++){
        double s = 0.0;
        for (int j = 0; j < n; j++) s += A[i][j]*x[j];
        y[i] = s;
    }
    return y;
}
static M matMul(const M& A, const M& B){
    int m = A.size(), n = B[0].size(), k = A[0].size();
    M C(m, V(n, 0.0));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][p]*B[p][j];
    return C;
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);
    const double h = 1e-6;

    // (1) 局部线性化验证
    V p   = {0.3, 0.7};
    V dp  = {0.01, -0.005};
    V fp  = f(p);
    V fpp = f({p[0]+dp[0], p[1]+dp[1]});
    V lin = matVec(Jf(p), dp);
    double err = 0.0;
    for (int i = 0; i < 2; i++)
        err = max(err, fabs((fpp[i]-fp[i]) - lin[i]));
    cout << "[L1.1] 局部线性化误差 = " << err << "   (应 < 1e-4)\n";
    if (!(err < 1e-4)){ pass = false; cout << "  [失败] 线性化误差过大\n"; }

    // (2) 链式法则验证：D(g∘f) = Dg(f) · Df
    V fp2 = f(p);
    M Dg  = Jg(fp2);
    M Df  = Jf(p);
    M Dgf = matMul(Dg, Df);                 // 解析链式法则结果 (2x2)
    V h0  = g(f(p));
    M Dgf_fd(2, V(2, 0.0));                 // 有限差分结果
    for (int c = 0; c < 2; c++){
        V pp = p; pp[c] += h;
        V hh = g(f(pp));
        for (int r = 0; r < 2; r++) Dgf_fd[r][c] = (hh[r]-h0[r])/h;
    }
    double cerr = 0.0;
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            cerr = max(cerr, fabs(Dgf[r][c] - Dgf_fd[r][c]));
    cout << "[L1.1] 链式法则误差 = " << cerr << "   (应 < 1e-4)\n";
    if (!(cerr < 1e-4)){ pass = false; cout << "  [失败] 链式法则不符\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 局部线性化 + 链式法则 均成立" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
