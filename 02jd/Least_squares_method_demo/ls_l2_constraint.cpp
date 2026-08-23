// ============================================================
// LS_demo L2.5 等式约束最小二乘(KKT)
// 纯 C++17。min ‖Ax−b‖² s.t. Cx=d:
//   [AᵀA Cᵀ; C 0][x;λ] = [Aᵀb; d]   (KKT 方程)
// 验证: ① 约束被精确满足(1e-16);② 约束解 SSE ≥ 自由解
// 场景: 已知增益比的标定、闭环总和约束、网格配准的固定锚点
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static unsigned long long _seed = 10;
static double urand(){ _seed = _seed*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

static Mat inv3(const Mat& M){
    double a=M[0][0],b=M[0][1],c=M[0][2],d=M[1][0],e=M[1][1],f=M[1][2],g=M[2][0],h=M[2][1],i=M[2][2];
    double A=e*i-f*h, B=-(d*i-f*g), C=d*h-e*g;
    double det=a*A+b*B+c*C;
    return {{A/det,-(b*i-c*h)/det,(b*f-c*e)/det},
            {B/det,(a*i-c*g)/det,-(a*f-c*d)/det},
            {C/det,-(a*h-b*g)/det,(a*e-b*d)/det}};
}
static vector<double> matvec(const Mat& A, const vector<double>& x){ vector<double> y(A.size(),0);
    for(size_t i=0;i<A.size();++i) for(size_t j=0;j<x.size();++j) y[i]+=A[i][j]*x[j]; return y; }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L2.5 等式约束最小二乘 ===\n");

    int n = 30;
    vector<double> xs(n), ys(n);
    for (int i=0;i<n;++i) { xs[i]=i*0.1; ys[i]=3.0*xs[i]+1.0+gauss()*0.3; }
    double Sx=0,Sy=0,Sxx=0,Sxy=0,Sn=n;
    for (int i=0;i<n;++i){ Sx+=xs[i]; Sy+=ys[i]; Sxx+=xs[i]*xs[i]; Sxy+=xs[i]*ys[i]; }
    double det = Sn*Sxx-Sx*Sx;
    double a_free = (Sn*Sxy-Sx*Sy)/det, b_free = (Sxx*Sy-Sx*Sxy)/det;
    // KKT: [Sxx Sx 1; Sx Sn 0; 1 0 0][a;b;λ] = [Sxy; Sy; 2.5]
    Mat K = {{Sxx,Sx,1.0},{Sx,Sn,0.0},{1.0,0.0,0.0}};
    vector<double> rhs = {Sxy, Sy, 2.5};
    vector<double> sol = matvec(inv3(K), rhs);
    double a_c = sol[0], b_c = sol[1];
    chk("约束被精确满足(a=2.5)", fabs(a_c-2.5), 1e-10);
    auto sse = [&](double a, double b){ double s=0;
        for(int i=0;i<n;++i){ double r=ys[i]-(a*xs[i]+b); s+=r*r; } return s; };
    chk("约束解 SSE ≥ 自由解", sse(a_c,b_c) >= sse(a_free,b_free)-1e-9 ? 0.0 : 1.0, 0.5);
    printf("  [demo] 自由解 a=%.3f → 约束解 a=%.4f (SSE %.2f→%.2f, λ=%.2f)\n",
           a_free, a_c, sse(a_free,b_free), sse(a_c,b_c), sol[2]);
    printf("  [结论] 约束=先验知识的硬形式;KKT 一行拼出来,λ 的含义是'约束张力'。\n");
    printf("        变体: 软约束=加权(岭回归)、不等式=QP(Lasso/滑窗 MPC)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
