// pid_l2_gain_sched.cpp
// ============================================================================
// L2.5  增益调度与自适应（Gain Scheduling）⭐
// ----------------------------------------------------------------------------
// 线性 PID 在"增益随工况变化"的非线性系统上，固定增益只能顾一头。解法：按工况
// (operating point) 调度增益。本 demo 用增益随工况 op 变化的二阶 plant：
//   固定增益：在 op 两端表现差异巨大（一端迟钝、一端超调）
//   调度增益：Kp(op)=Kp0·K0/K(op) 补偿 plant 增益变化 → 全工况表现一致
// 全程合成数据，运行即出 PASS/FAIL。参照 PX4 按推力/空速调度姿态·速率增益。
//
// 编译: g++ -O3 -std=c++17 pid_l2_gain_sched.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

static double Kof(double op){ return 0.3 + 3.0*op; }   // plant 增益随工况变化

// 二阶 plant: v'=-2ζωv-ω²x + K(op)·u ; D 作用在被测量(消 setpoint kick)
static vector<double> simGS(double op, double Kp,double Ki,double Kd,double dt,int N){
    vector<double> y(N+1,0.0);
    double x=0,v=0,integ=0,px=0;
    double K=Kof(op), w=2.0, zeta=0.3;
    for(int k=0;k<N;k++){
        double e=1.0-x;
        integ+=Ki*e*dt;
        double der=-(x-px)/dt;                 // D on measurement
        double u=Kp*e+integ+Kd*der;
        v=v+dt*(-2*zeta*w*v - w*w*x + K*u);
        x=x+dt*v; px=x;
        y[k+1]=x;
    }
    return y;
}
static double steadyErr(const vector<double>& y){ return 1.0-y.back(); }
static double overshoot(const vector<double>& y){ double pk=*max_element(y.begin(),y.end()); return pk<=1?0:pk-1; }
static double settling(const vector<double>& y,double dt){
    int N=(int)y.size()-1;
    for(int i=N;i>=0;i--) if(abs(y[i]-1.0)>0.02) return (i+1)*dt;
    return 0;
}
static double variance(const vector<double>& v){
    double m=0; for(double x:v) m+=x; m/=v.size();
    double s=0; for(double x:v) s+=(x-m)*(x-m); return s/v.size();
}

int main(){
    bool pass=true;
    const double dt=0.01, N=800;
    vector<double> ops{0.1,0.3,0.5,0.7,0.9};
    cout<<fixed<<setprecision(4);

    // 固定增益（按名义 op=0.5 整定）
    double Kp0=10.0, Ki=5.0, Kd=1.5, K0=Kof(0.5);
    vector<double> fix_set, fix_ov;
    cout<<"[L2.5 增益调度] op 范围: K="<<Kof(0.1)<<".."<<Kof(0.9)<<"\n";
    for(double op:ops){
        auto y=simGS(op,Kp0,Ki,Kd,dt,N);
        fix_set.push_back(settling(y,dt)); fix_ov.push_back(overshoot(y));
        cout<<"  固定 Kp="<<Kp0<<" op="<<op<<" 超调="<<overshoot(y)*100<<"% 调节="<<settling(y,dt)<<"s 稳态误差="<<steadyErr(y)<<"\n";
        if(steadyErr(y)>0.05) pass=false;
    }
    // 调度增益：Kp(op)=Kp0·K0/K(op)
    vector<double> sch_set, sch_ov;
    for(double op:ops){
        double Kp=Kp0*K0/Kof(op);
        auto y=simGS(op,Kp,Ki,Kd,dt,N);
        sch_set.push_back(settling(y,dt)); sch_ov.push_back(overshoot(y));
        cout<<"  调度 Kp="<<Kp<<" op="<<op<<" 超调="<<overshoot(y)*100<<"% 调节="<<settling(y,dt)<<"s 稳态误差="<<steadyErr(y)<<"\n";
        if(steadyErr(y)>0.05) pass=false;
    }

    double vF=variance(fix_set), vS=variance(sch_set);
    double maxOvF=*max_element(fix_ov.begin(),fix_ov.end());
    double maxOvS=*max_element(sch_ov.begin(),sch_ov.end());
    cout<<"  调节时间方差: 固定="<<vF<<"  调度="<<vS<<"\n";
    cout<<"  最大超调: 固定="<<maxOvF*100<<"%  调度="<<maxOvS*100<<"%\n";

    if(!(vS < 0.5*vF)){ pass=false; cout<<"  [失败] 调度未显著统一各工况表现\n"; }
    if(!(maxOvS <= maxOvF + 1e-6)){ pass=false; cout<<"  [失败] 调度未抑制超调\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 增益调度补偿 plant 增益变化，全工况表现一致":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
