// ============================================================
// LS_demo L3.1 工业用例:传感器线性标定
// 纯 C++17。ADC = a·T + b + ε;7 点标定 → LS 反演 T=(ADC−b)/a
// 验证: ① 16 次平均标定参数偏差 <8e-3;② 反演 RMSE ~1°C vs naive 25°C
// 呼应 cali_demo:那边用 OpenCV 做相机标定(非线性 LS),
// 这边是最小、最常见的线性版——一切标定的"Hello World"
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 0;
static void set_seed(unsigned long long s){ _seed = s; }
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
    printf("=== L3.1 传感器线性标定 ===\n");

    const double a_true = 0.0185, b_true = -0.42;
    const double Ts[7] = {0, 10, 25, 40, 60, 80, 100};
    // 16 次标定取平均
    double pa = 0, pb = 0; int M = 16;
    for (int m = 0; m < M; ++m) {
        set_seed(1200+m);
        double Sx=0,Sy=0,Sxx=0,Sxy=0; int n=7;
        for (int i=0;i<n;++i) {
            double y = a_true*Ts[i] + b_true + gauss()*0.02;
            Sx+=Ts[i]; Sy+=y; Sxx+=Ts[i]*Ts[i]; Sxy+=Ts[i]*y;
        }
        double det = n*Sxx-Sx*Sx;
        pa += (n*Sxy-Sx*Sy)/det;
        pb += (Sxx*Sy-Sx*Sxy)/det;
    }
    pa/=M; pb/=M;
    printf("  [demo] 标定 16 次平均: a=%.5f b=%.4f (真值 0.0185, -0.42)\n", pa, pb);
    chk("标定参数接近真值(16次平均)", fmax(fabs(pa-a_true), fabs(pb-b_true)), 8e-3);
    // 反演精度
    set_seed(777);
    double e2c = 0, e2r = 0; int cnt = 0;
    for (int i = 0; i < 30; ++i) {
        double T = urand()*100;
        double adc = a_true*T + b_true + gauss()*0.02;
        double T_cal = (adc - pb)/pa;
        double T_naive = adc/0.02;           // 未标定:拍脑袋增益
        e2c += (T_cal-T)*(T_cal-T);
        e2r += (T_naive-T)*(T_naive-T);
        ++cnt;
    }
    printf("  [demo] 温度反演 RMSE: 标定=%.2f °C vs 未标定=%.2f °C\n",
           sqrt(e2c/cnt), sqrt(e2r/cnt));
    chk("标定后反演精度提升 10×+", sqrt(e2c/cnt) < 0.1*sqrt(e2r/cnt) ? 0.0 : 1.0, 0.5);
    printf("  [结论] 两点决定一条直线,多点+LS 抗噪;ADC 增益/零飘、压力计、\n");
    printf("        电子秤——出厂标定全是这一篇。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
