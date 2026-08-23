// ============================================================
// LS_demo L3.3 工业用例:ARX 系统辨识
// 纯 C++17。对象 y(k) = a·y(k−1) + b·u(k−1) + e
//   φ = [y(k−1), u(k−1)] → θ=[a,b]ᵀ = (ΦᵀΦ)⁻¹ΦᵀY
// 验证: ① 参数辨识误差 <0.02;② 残差与噪声同级
// 呼应 control_demo L3.4(阶跃两点法辨识——时域特例,这里更普适)
// 关键概念: 持续激励(方波输入)——输入不丰富则 ΦᵀΦ 病态
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 14;
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
    printf("=== L3.3 ARX 系统辨识(最小二乘)===\n");

    const double a1 = -0.8, b0 = 0.5;
    int N = 200;
    vector<double> phi_y, phi_u, Y;
    double y_prev = 0.0, u_prev = 0.0;
    for (int k = 0; k < N; ++k) {
        double u = ((k/20) % 2 == 0) ? 1.0 : -1.0;   // 方波:持续激励
        double y = a1*y_prev + b0*u_prev + gauss()*0.02;
        if (k > 1) { phi_y.push_back(y_prev); phi_u.push_back(u_prev); Y.push_back(y); }
        y_prev = y; u_prev = u;
    }
    int n = Y.size();
    double Syy=0,Syu=0,Suu=0,SyY=0,SuY=0;
    for (int i = 0; i < n; ++i) {
        Syy += phi_y[i]*phi_y[i]; Syu += phi_y[i]*phi_u[i]; Suu += phi_u[i]*phi_u[i];
        SyY += phi_y[i]*Y[i];     SuY += phi_u[i]*Y[i];
    }
    double det = Syy*Suu - Syu*Syu;
    double a_hat = ( Suu*SyY - Syu*SuY)/det;
    double b_hat = (-Syu*SyY + Syy*SuY)/det;
    double resmax = 0;
    for (int i = 0; i < n; ++i)
        resmax = fmax(resmax, fabs(Y[i] - (a_hat*phi_y[i] + b_hat*phi_u[i])));
    printf("  [demo] ARX 辨识: a=%.4f b=%.4f (真值 -0.8, 0.5), 最大残差=%.4f\n",
           a_hat, b_hat, resmax);
    chk("ARX 参数辨识", fmax(fabs(a_hat-a1), fabs(b_hat-b0)), 0.02);
    chk("残差与噪声同级", resmax, 0.08);
    printf("  [结论] 只要输入'持续激励',辨识=一次线性 LS;模型阶数靠 AIC/残差白性。\n");
    printf("        control_demo L3.4 的阶跃两点法 = 本篇的时域特例。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
