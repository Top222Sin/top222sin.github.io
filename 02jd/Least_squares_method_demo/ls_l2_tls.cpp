// ============================================================
// LS_demo L2.6 总体最小二乘 TLS(数据双方都有噪声)
// 纯 C++17。OLS 假设 x 精确、y 有噪;x 也有噪(EIV 模型)时
// OLS 斜率被系统性衰减;TLS = 数据协方差阵 λ_max 特征向量。
// 验证: 20 次蒙特卡洛平均——OLS 偏置显著、TLS 无偏
// ⚠ 坑(开发踩过): λ_min 特征向量是"法向",λ_max 才是直线方向!
// ============================================================
#include <cstdio>
#include <cmath>
using namespace std;

static unsigned long long _seed = 11;
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
    printf("=== L2.6 总体最小二乘 TLS ===\n");

    int n = 100, MT = 20;
    double a_true = 2.0;
    double sum_ols = 0, sum_tls = 0;
    for (int m = 0; m < MT; ++m) {
        double Sx=0,Sy=0,Sxx=0,Sxy=0;
        double mx=0,my=0;
        static double cxx=0, cyy=0, cxy=0;   // 每轮重算
        double X[100], Y[100];
        for (int i=0;i<n;++i) {
            double t = i*0.1;
            double x = t + gauss()*0.3;   // x 也带噪!
            double y = a_true*t + gauss()*0.3;
            X[i]=x; Y[i]=y;
            Sx+=x; Sy+=y; Sxx+=x*x; Sxy+=x*y;
        }
        double a_ols = (n*Sxy-Sx*Sy)/(n*Sxx-Sx*Sx);
        mx=Sx/n; my=Sy/n;
        cxx=0;cyy=0;cxy=0;
        for (int i=0;i<n;++i) { cxx+=(X[i]-mx)*(X[i]-mx); cyy+=(Y[i]-my)*(Y[i]-my);
                                cxy+=(X[i]-mx)*(Y[i]-my); }
        cxx/=n; cyy/=n; cxy/=n;
        double tr = cxx+cyy, det = cxx*cyy-cxy*cxy;
        double disc = sqrt(fmax(tr*tr/4-det, 0));
        double lam_max = tr/2 + disc;
        double vx = -cxy, vy = cxx - lam_max;   // λ_max 特征向量=直线方向
        double a_tls = vy/vx;
        sum_ols += a_ols; sum_tls += a_tls;
    }
    double bias_ols = fabs(sum_ols/MT - a_true);
    double bias_tls = fabs(sum_tls/MT - a_true);
    printf("  [demo] 双侧噪声(σx=σy=0.3) 20 次平均:\n");
    printf("        OLS 斜率=%.4f (偏置 %.4f, 系统性衰减)\n", sum_ols/MT, bias_ols);
    printf("        TLS 斜率=%.4f (偏置 %.4f)\n", sum_tls/MT, bias_tls);
    chk("TLS 无偏 vs OLS 衰减偏置", bias_tls < bias_ols ? 0.0 : 1.0, 0.5);
    chk("TLS 偏置 < OLS 一半", bias_tls < 0.5*bias_ols ? 0.0 : 1.0, 0.5);
    printf("  [结论] x 有噪不补偿 → 回归稀释(斜率往 0 缩);\n");
    printf("        TLS=对数据点做正交距离最小化(SVD 求解),标定对板时常用。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
