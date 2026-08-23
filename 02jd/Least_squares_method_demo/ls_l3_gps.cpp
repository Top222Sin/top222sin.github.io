// ============================================================
// LS_demo L3.2 工业用例:GPS 伪距定位(非线性 LS/GN)
// 纯 C++17。伪距 ρk = ‖x−sk‖ + δt·c + ε;未知 [x,y,z,δt·c]
// GN 迭代: J=∂h/∂x(几何矩阵), r=z−h, x ← x+(JᵀJ)⁻¹Jᵀr
// 验证: 5 颗卫星、3m 伪距噪声、地心起点 → 20 步内收敛,
//      定位误差 <15m、钟差解出(GDOP 越小越好)
// ⚠ 坑(开发踩过): J 的符号约定要与残差方向一致(r=z−h 时 J=+∂h/∂x)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 13;
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
    printf("=== L3.2 GPS 伪距定位(GN 迭代 LS)===\n");

    const double sats[5][3] = {{14000,2000,15000},{-13000,8000,17000},
                               {6000,-16000,12000},{-5000,-12000,-19000},
                               {9000,15000,-16000}};
    const double c = 3e8;
    double pos_true[3] = {100.0, -200.0, 50.0};
    double b_true_m = 0.001*c*1e-6;   // 钟差等效距离(米)
    const int M = 5;
    double zs[M];
    for (int k = 0; k < M; ++k) {
        double d = 0;
        for (int i = 0; i < 3; ++i) { double t = pos_true[i]-sats[k][i]; d += t*t; }
        d = sqrt(d);
        zs[k] = d + b_true_m + gauss()*3.0;
    }
    // GN 从地心附近起
    double x[4] = {0, 0, 0, 0};
    int it = 0;
    for (it = 0; it < 30; ++it) {
        double J[M][4], r[M];
        for (int k = 0; k < M; ++k) {
            double d = 0;
            for (int i = 0; i < 3; ++i) { double t = x[i]-sats[k][i]; d += t*t; }
            d = sqrt(fmax(d, 1.0));
            r[k] = zs[k] - (d + x[3]);
            for (int i = 0; i < 3; ++i) J[k][i] = (x[i]-sats[k][i])/d;
            J[k][3] = 1.0;
        }
        // JᵀJ (4x4) 与 Jᵀr —— 高斯消元解正规方程
        double A[4][5];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) { double s = 0;
                for (int k = 0; k < M; ++k) s += J[k][i]*J[k][j]; A[i][j] = s; }
            double s = 0;
            for (int k = 0; k < M; ++k) s += J[k][i]*r[k];
            A[i][4] = s;
        }
        // 列主元消元
        for (int col = 0; col < 4; ++col) {
            int piv = col;
            for (int i = col; i < 4; ++i) if (fabs(A[i][col]) > fabs(A[piv][col])) piv = i;
            for (int j = 0; j < 5; ++j) swap(A[col][j], A[piv][j]);
            for (int i = 0; i < 4; ++i) if (i != col) {
                double f = A[i][col]/A[col][col];
                for (int j = col; j < 5; ++j) A[i][j] -= f*A[col][j];
            }
        }
        double dp[4];
        for (int i = 0; i < 4; ++i) dp[i] = A[i][4]/A[i][i];
        for (int i = 0; i < 4; ++i) x[i] += dp[i];
        double mx = 0; for (int i = 0; i < 4; ++i) mx = fmax(mx, fabs(dp[i]));
        if (mx < 1e-8) break;
    }
    double err = sqrt((x[0]-pos_true[0])*(x[0]-pos_true[0])+
                      (x[1]-pos_true[1])*(x[1]-pos_true[1])+
                      (x[2]-pos_true[2])*(x[2]-pos_true[2]));
    printf("  [demo] %d 星 GN %d 步: 定位误差=%.3f m, 钟差项=%.1f m\n", M, it, err, x[3]);
    chk("定位误差 < 15 m", err, 15.0);
    chk("钟差被解出", fabs(x[3]-b_true_m), 10.0);
    printf("  [结论] GNSS 定位=每秒一次的非线性 LS;GDOP=JᵀJ 病态度,\n");
    printf("        星型分布好→GDOP 小→同噪声定位准(选星的依据)。\n");
    printf("        KF_demo 的松组合 = 这个 LS + 时间滤波。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
