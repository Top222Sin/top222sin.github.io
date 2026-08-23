// pid_l4_adrc.cpp
// ============================================================================
// L3  ADRC 自抗扰控制 (Active Disturbance Rejection Control)
// ----------------------------------------------------------------------------
// 传统 PID 把"模型误差+外部扰动"全部塞给积分项慢慢消, 遇到大扰动/强非线性很被动。
// ADRC 的核心思想: 用一只 扩张状态观测器(ESO) 在线估计"总扰动" f
//   (f = 一切未知动力学 + 外部扰动, 对二阶对象即含 -a1*y'-a0*y + d(t)),
// 再用控制律把 f 直接"抵消"掉, 闭环退化成想要的二阶理想系统。
// 最妙的是: 控制器**只用一个 b0(控制增益)**, 完全不需要知道对象精确模型!
//
// 编译: g++ -O3 -std=c++17 pid_l4_adrc.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <functional>

using namespace std;

// 二阶对象: y'' = f + b0*u,  其中 f = -a1*y' - a0*y + d(t)  (总扰动, 未知)
// 控制器只用 b0, 不知道 a1/a0/d —— 这正是 ADRC 的"模型无关"威力
struct Sim { vector<double> y, z3, f; };

static Sim simADRC(double r, const function<double(double)>& dist,
                   double T, double dt, double b0, double w_o, double w_c){
    Sim s; int N=int(T/dt);
    double y=0,yd=0,z1=0,z2=0,z3=0,ut=0;
    double b1=3*w_o, b2=3*w_o*w_o, b3=w_o*w_o*w_o;
    double Kp=w_c*w_c, Kd=2*w_c;
    for(int k=0;k<N;k++){
        double t=k*dt;
        double d = dist(t);                // 真实扰动(控制器看不到!)
        double e0 = z1 - y;
        z1 += (z2 - b1*e0)*dt;
        z2 += (z3 - b2*e0 + b0*ut)*dt;
        z3 += (-b3*e0)*dt;                // ESO 估计总扰动 f
        double u0 = Kp*(r - z1) + Kd*(0 - z2);
        double u  = (u0 - z3)/b0;         // 抵消估计出的扰动
        ut = u;
        double f = -2.0*yd - 1.0*y + d;   // 真实总扰动(仅用于校验)
        yd += (f + b0*u)*dt;
        y  += yd*dt;
        s.y.push_back(y); s.z3.push_back(z3); s.f.push_back(f);
    }
    return s;
}

static vector<double> simPD(double r, const function<double(double)>& dist,
                            double T, double dt, double Kp, double Kd){
    int N=int(T/dt); vector<double> y(N,0);
    double yy=0,yd=0;
    for(int k=0;k<N;k++){
        double t=k*dt; double d=dist(t);
        double u = Kp*(r-yy) + Kd*(0-yd);   // 无积分, 无法消除常值/斜坡扰动
        double f = -2.0*yd - 1.0*yy + d;
        yd += (f+u)*dt; yy += yd*dt; y[k]=yy;
    }
    return y;
}

static double maxabs_last(const vector<double>& e, double frac=0.5){
    int n=(int)e.size(); int m=(int)(n*(1-frac));
    double mx=0; for(int i=m;i<n;i++){ double a=abs(e[i]); if(a>mx)mx=a; }
    return mx;
}

int main(){
    bool pass=true;
    const double dt=0.001, T=10.0;
    cout<<fixed<<setprecision(4);
    cout<<"[L3 ADRC 自抗扰控制] 对象 y''=f+b0*u, 仅用 b0=1, ESO 估计总扰动 f\n\n";

    auto dStep = [](double t){ return t>=1.0? 0.5 : 0.0; };   // 常值阶跃扰动
    auto dRamp = [](double t){ return t; };                   // 斜坡扰动 d(t)=t

    // A/B: 常值阶跃扰动 d=0.5 (t>=1)
    auto s = simADRC(1.0, dStep, T, dt, 1.0, 10.0, 5.0);
    double eA = abs(s.z3.back() - s.f.back());
    double eB = abs(1.0 - s.y.back());
    double eB2 = maxabs_last(s.y, 0.5);  // 末半程 max|y|; 见下用 1-y 修正
    // 末半程 max|1-y|
    eB2=0; for(size_t i=(size_t)(s.y.size()*0.5); i<s.y.size(); i++){ double a=abs(1.0-s.y[i]); if(a>eB2)eB2=a; }

    cout<<"  A. ESO 估计总扰动 f (稳态)  : |z3-f| = "<<eA<<"  (应<0.05)\n";
    cout<<"  B. 跟踪 r=1 稳态误差        : |1-y|  = "<<eB<<"  (应<0.02)\n";
    cout<<"  B2.末半程最大跟踪误差       : max|e| = "<<eB2<<"  (应<0.02)\n";
    if(!(eA<0.05)){ pass=false; cout<<"  [失败] ESO 未收敛到 f\n"; }
    if(!(eB<0.02)){ pass=false; cout<<"  [失败] 跟踪稳态误差过大\n"; }
    if(!(eB2<0.02)){ pass=false; cout<<"  [失败] 末半程仍在漂\n"; }

    // C: 斜坡扰动 d(t)=t, ADRC vs 无积分 PD
    auto sR = simADRC(1.0, dRamp, T, dt, 1.0, 10.0, 5.0);
    auto pR = simPD(1.0, dRamp, T, dt, 25.0, 10.0);
    double adrcE=0, pdE=0;
    for(size_t i=(size_t)(sR.y.size()*0.5); i<sR.y.size(); i++){ double a=abs(1.0-sR.y[i]); if(a>adrcE)adrcE=a; }
    for(size_t i=(size_t)(pR.size()*0.5); i<pR.size(); i++){ double a=abs(1.0-pR[i]); if(a>pdE)pdE=a; }
    cout<<"  C. 斜坡扰动 d(t)=t 末半程 max|e|: ADRC="<<adrcE<<"  PD(无积分)="<<pdE<<"\n";
    cout<<"     ADRC 直接抵消总扰动 -> 跟踪几乎不受斜坡影响; 无积分 PD 被斜坡拖离设定值\n";
    if(!(adrcE<0.05)){ pass=false; cout<<"  [失败] ADRC 未抑制斜坡扰动\n"; }
    if(!(pdE>10.0*adrcE)){ pass=false; cout<<"  [失败] PD 对照未明显更差\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] ADRC 模型无关 + 总扰动抵消 + 强抗扰":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
