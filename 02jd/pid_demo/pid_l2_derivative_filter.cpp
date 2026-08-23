// pid_l2_derivative_filter.cpp
// ============================================================================
// L2.2  微分噪声与 D 项滤波
// ----------------------------------------------------------------------------
// D 项 = Kd·de/dt，对测量噪声极敏感（噪声被 1/dt 放大）。本 demo：
//   ① 给反馈加测量噪声，对比「D 作用误差(无滤波)」与「D 作用测量量+低通滤波」
//   ② 顺便展示 derivative-on-measurement 消除设定值突变(setpoint kick)冲击
// 全程合成数据（plant + 噪声），运行即出 PASS/FAIL。参照 PX4 对 D/角加速度做 LPF。
//
// 编译: g++ -O3 -std=c++17 pid_l2_derivative_filter.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

using namespace std;

static double clamp(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

struct R { vector<double> y, u; };

// useFilter: 对微分低通滤波; derivOnMeas: 微分作用在被测量(而非误差)
static R run(bool useFilter, bool derivOnMeas, double Kp,double Ki,double Kd,
             double dt,int N,double noise, mt19937& g){
    R r; r.y.assign(N+1,0); r.u.assign(N+1,0);
    double x=0, integ=0, pe=0, pxm=0, dlp=0;
    uniform_real_distribution<double> un(-noise,noise);
    for(int k=0;k<N;k++){
        double xm = x + un(g);                 // 带噪声测量
        double e = 1.0 - xm;
        double der = derivOnMeas ? (-(xm - pxm)/dt) : ((e - pe)/dt);
        double der_f = der;
        if(useFilter){ double a=dt/(0.05+dt); dlp = dlp + a*(der - dlp); der_f = dlp; }
        double u = clamp(Kp*e + integ + Kd*der_f, -10.0, 10.0);
        integ += Ki*(1.0 - x)*dt;             // 积分用"真值误差"，避免噪声 windup
        x = x + dt*(-x + u);
        pe=e; pxm=xm; r.y[k+1]=x; r.u[k+1]=u;
    }
    return r;
}

static double steadyErr(const vector<double>& y){ return 1.0 - y.back(); }
static double variance(const vector<double>& u, int from){
    double m=0; for(size_t i=from;i<u.size();i++) m+=u[i]; m/=(u.size()-from);
    double v=0; for(size_t i=from;i<u.size();i++) v+=(u[i]-m)*(u[i]-m); v/=(u.size()-from);
    return v;
}
static bool finite(const vector<double>& y){ for(double v:y) if(!isfinite(v)) return false; return true; }

int main(){
    bool pass=true; mt19937 g(12345);
    const double dt=0.02, N=600, noise=0.08;
    cout<<fixed<<setprecision(4);

    auto A = run(false, false, 2.0, 1.0, 3.0, dt, N, noise, g);  // 误差微分, 无滤波
    auto B = run(true,  true , 2.0, 1.0, 3.0, dt, N, noise, g);  // 测量微分 + 低通

    double varA=variance(A.u,400), varB=variance(B.u,400);
    double eA=steadyErr(A.y), eB=steadyErr(B.y);
    double kickA=abs(A.u[1]), kickB=abs(B.u[1]);
    cout<<"[L2.2 D 项滤波] 反馈噪声=±"<<noise<<"\n";
    cout<<"  误差微分·无滤波 : u方差="<<varA<<"  初始kick="<<kickA<<"  稳态误差="<<eA<<"\n";
    cout<<"  测量微分·低通   : u方差="<<varB<<"  初始kick="<<kickB<<"  稳态误差="<<eB<<"\n";

    if(!finite(A.y)||!finite(B.y)){ pass=false; cout<<"  [失败] 响应发散\n"; }
    if(eA>0.1||eB>0.1){ pass=false; cout<<"  [失败] 滤波损害跟踪\n"; }
    if(!(varB < 0.3*varA)){ pass=false; cout<<"  [失败] 滤波未显著抑制噪声\n"; }
    if(!(kickA > kickB + 1.0)){ pass=false; cout<<"  [失败] 未消除设定值冲击\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 低通+测量微分 显著降噪且不损跟踪":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
