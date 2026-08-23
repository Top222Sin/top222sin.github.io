// pid_l2_cascade.cpp
// ============================================================================
// L2.1  串级 PID 与多回路控制
// ----------------------------------------------------------------------------
// 单环搞不定时，套环：外环输出 = 内环设定值。本 demo 用"位置环 → 速度环"双环：
//   外环(位置P): v_sp = Kp_p·(x_sp − x)，并对速度限幅（防超调）
//   内环(速度PID): u = Kp_v·e_v + Ki_v·∫e_v + Kd_v·de_v
// 与"单环位置 PID 直接出推力"对比，说明串级如何分离时间尺度、约束速度。
// 全程合成数据，运行即出 PASS/FAIL。参照 PX4 PositionControl::update 串级结构。
//
// 编译: g++ -O3 -std=c++17 pid_l2_cascade.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

static double clamp(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

// 串级：位置环(外) → 速度环(内)。plant: v'=(u−v)/tau, x'=v
static void simCascade(double Kp_p,double Kp_v,double Ki_v,double Kd_v,double vmax,double tau,double dt,int N,
                       vector<double>& xo, vector<double>& vo){
    xo.assign(N+1,0); vo.assign(N+1,0);
    double xk=0,vk=0,integ=0,pe=0;
    for(int k=0;k<N;k++){
        double perr=1.0-xk;
        double vsp=clamp(Kp_p*perr,-vmax,vmax);
        double verr=vsp-vk;
        integ+=Ki_v*verr*dt;
        double der=(verr-pe)/dt;
        double u=Kp_v*verr+integ+Kd_v*der;
        vk=vk+dt*((u-vk)/tau);
        xk=xk+dt*vk;
        pe=verr; xo[k+1]=xk; vo[k+1]=vk;
    }
}

// 单环：位置 PID 直接出推力（D 作用在被测量速度上）
static void simSingle(double Kp,double Ki,double Kd,double tau,double dt,int N,
                      vector<double>& xo, vector<double>& vo){
    xo.assign(N+1,0); vo.assign(N+1,0);
    double xk=0,vk=0,integ=0,pe=0;
    for(int k=0;k<N;k++){
        double perr=1.0-xk;
        integ+=Ki*perr*dt;
        double u=Kp*perr+integ+Kd*(-vk);   // D on measurement(velocity)
        vk=vk+dt*((u-vk)/tau);
        xk=xk+dt*vk;
        pe=perr; xo[k+1]=xk; vo[k+1]=vk;
    }
}

static double steadyErr(const vector<double>& y){ return 1.0-y.back(); }
static double overshoot(const vector<double>& y){
    double peak=*max_element(y.begin(),y.end()); return peak<=1e-6?0:max(0.0,(peak-1.0));
}
static double maxAbs(const vector<double>& y){ double m=0; for(double v:y) m=max(m,abs(v)); return m; }
static bool finite(const vector<double>& y){ for(double v:y) if(!isfinite(v)) return false; return true; }

int main(){
    bool pass=true;
    const double dt=0.01, N=1000, tau=0.3;
    vector<double> xC,vC,xS,vS;
    simCascade(4.0, 10.0, 5.0, 1.0, 2.0, tau, dt, N, xC, vC);
    simSingle  (8.0, 2.0, 1.5,        tau, dt, N, xS, vS);

    double oC=overshoot(xC), oS=overshoot(xS);
    double eC=steadyErr(xC), eS=steadyErr(xS);
    double vmaxC=maxAbs(vC);
    cout<<fixed<<setprecision(4);
    cout<<"[L2.1 串级 PID] plant: v'=(u−v)/"<<tau<<", x'=v\n";
    cout<<"  串级 : 超调="<<oC*100<<"%  稳态误差="<<eC<<"  最大速度="<<vmaxC<<"\n";
    cout<<"  单环 : 超调="<<oS*100<<"%  稳态误差="<<eS<<"\n";

    if(!finite(xC)||!finite(xS)){ pass=false; cout<<"  [失败] 响应发散\n"; }
    if(eC>0.05||eS>0.05){ pass=false; cout<<"  [失败] 未达设定值\n"; }
    if(!(oC < 0.15)){ pass=false; cout<<"  [失败] 串级超调过大\n"; }
    if(!(oC <= oS + 0.05)){ pass=false; cout<<"  [失败] 串级未优于单环\n"; }
    if(vmaxC > 2.0*1.2){ pass=false; cout<<"  [失败] 速度未受内环约束\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 串级分离时间尺度、约束速度、抑制超调":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
