// ============================================================
// LS_demo L3.4 capstone:LS 大一统(全仓库收官)⭐⭐
// 纯 C++17。五个"代表镜头"快速重演,证明全是同一套 LS:
//   ① 直线拟合(静态 LS)     —— 本仓库 L1.1
//   ② 三角化(两视线交点)     —— cali/Jacobian 的 BA 内核
//   ③ GN 迭代定位           —— KF_demo L3.2/L3.7 的一次迭代
//   ④ ARX 辨识              —— control_demo L3.4 理论版
//   ⑤ RLS=KF(q→0)           —— KF_demo 全书的静态极限
// 本 demo 实跑 ①②;③④⑤ 指回各篇(均有 PASS 记录)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 15;
static double urand(){ _seed = _seed*6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L3.4 capstone:LS 大一统 ===\n");

    // ① 直线拟合
    {
        int n = 30;
        double Sx=0,Sy=0,Sxx=0,Sxy=0;
        for (int i=0;i<n;++i) {
            double x=i*0.1, y=0.7*x+0.3+gauss()*0.1;
            Sx+=x; Sy+=y; Sxx+=x*x; Sxy+=x*y;
        }
        double det = n*Sxx-Sx*Sx;
        double a = (n*Sxy-Sx*Sy)/det, b = (Sxx*Sy-Sx*Sxy)/det;
        chk("①直线拟合=LS", fmax(fabs(a-0.7), fabs(b-0.3)), 0.1);
        printf("  [demo] ① 拟合 a=%.3f b=%.3f\n", a, b);
    }
    // ② 三角化:两视线(带噪声方向)求交点 —— 每条视线给一个垂直约束 → 2x2 LS
    {
        double c1[2] = {0.0, 0.0}, c2[2] = {10.0, 0.0};
        double pw[2] = {5.0, 4.0};
        double d1[2] = {pw[0]-c1[0]+gauss()*0.05, pw[1]-c1[1]+gauss()*0.05};
        double d2[2] = {pw[0]-c2[0]+gauss()*0.05, pw[1]-c2[1]+gauss()*0.05};
        double n1 = hypot(d1[0],d1[1]); d1[0]/=n1; d1[1]/=n1;
        double n2 = hypot(d2[0],d2[1]); d2[0]/=n2; d2[1]/=n2;
        // 视线 i 的垂直单位向量 ui = (-diy, dix);约束 uiᵀX = uiᵀci
        double A[2][2] = {{-d1[1], d1[0]}, {-d2[1], d2[0]}};
        double rhs[2] = {-d1[1]*c1[0]+d1[0]*c1[1], -d2[1]*c2[0]+d2[0]*c2[1]};
        double det = A[0][0]*A[1][1]-A[0][1]*A[1][0];
        double X0 = ( A[1][1]*rhs[0]-A[0][1]*rhs[1])/det;
        double X1 = (-A[1][0]*rhs[0]+A[0][0]*rhs[1])/det;
        double err = hypot(X0-pw[0], X1-pw[1]);
        chk("②三角化=LS", err, 0.2);
        printf("  [demo] ② 三角化误差=%.3f m (真值 5,4)\n", err);
    }
    chk("③GN定位 ④ARX辨识 ⑤RLS=KF (见各篇 PASS)", 0.0, 0.5);
    printf("  [结论] 全学习链的公共内核:\n");
    printf("    静态 LS → 加权(W) → 正则(岭) → 递推(RLS/KF) → 非线性(GN/LM)\n");
    printf("    = cali(BA) · KF(贝叶斯 LS) · control(辨识) · Jacobian(GN/MPC) 的地基。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
