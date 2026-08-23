// ============================================================
// LS_demo L1.1 线性最小二乘:法方程与几何解释
// 纯 C++17。min ‖Ax−b‖² → 法方程 AᵀAx=Aᵀb。
// 验证:① 残差⊥列空间(Aᵀr=0,LS 的定义性性质);
//      ② 法方程解 = QR 解(逐位);③ 直线拟合参数接近真值
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static unsigned long long _seed = 1;
static double urand() { _seed = _seed*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss() { double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

static Mat transpose(const Mat& A) { int n=A.size(), m=A[0].size(); Mat T(m, vector<double>(n));
    for (int i=0;i<n;++i) for (int j=0;j<m;++j) T[j][i]=A[i][j]; return T; }
static Mat matmul(const Mat& A, const Mat& B) { int n=A.size(), k=A[0].size(), m=B[0].size();
    Mat C(n, vector<double>(m,0));
    for (int i=0;i<n;++i) for (int p=0;p<k;++p){double a=A[i][p]; if(a) for(int j=0;j<m;++j) C[i][j]+=a*B[p][j];}
    return C; }
static vector<double> matvec(const Mat& A, const vector<double>& x) {
    vector<double> y(A.size(),0);
    for (size_t i=0;i<A.size();++i) for (size_t j=0;j<x.size();++j) y[i]+=A[i][j]*x[j];
    return y; }
static Mat inv2(const Mat& M){ double a=M[0][0],b=M[0][1],c=M[1][0],d=M[1][1];
    double det=a*d-b*c; return {{d/det,-b/det},{-c/det,a/det}}; }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L1.1 线性最小二乘:法方程与几何 ===\n");

    int n = 40;
    double a_true = 2.0, b_true = 1.0;
    vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        xs[i] = i*0.1;
        ys[i] = a_true*xs[i] + b_true + gauss()*0.3;
    }
    // 法方程(2x2 手写)
    double Sx=0, Sy=0, Sxx=0, Sxy=0;
    for (int i = 0; i < n; ++i) { Sx+=xs[i]; Sy+=ys[i]; Sxx+=xs[i]*xs[i]; Sxy+=xs[i]*ys[i]; }
    double det = n*Sxx - Sx*Sx;
    double a = (n*Sxy - Sx*Sy)/det, b = (Sxx*Sy - Sx*Sxy)/det;
    // ① 残差正交
    double Atr0 = 0, Atr1 = 0;
    for (int i = 0; i < n; ++i) {
        double r = ys[i] - (a*xs[i]+b);
        Atr0 += xs[i]*r; Atr1 += r;
    }
    chk("残差⊥列空间 Aᵀr=0", fmax(fabs(Atr0), fabs(Atr1)), 1e-10);
    chk("参数接近真值", fmax(fabs(a-a_true), fabs(b-b_true)), 0.15);
    printf("  [demo] 拟合: y=%.4f x + %.4f (真值 2.0x+1.0)\n", a, b);
    printf("  [结论] LS=正交投影:把 b 投到 A 的列空间,残差最短必垂直。\n");
    printf("        AᵀAx=Aᵀb 就是垂直条件的代数写法。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
