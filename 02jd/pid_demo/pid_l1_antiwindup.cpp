// pid_l1_antiwindup.cpp
// ============================================================================
// L1.5  抗积分饱和（Integral Windup）基础
// ----------------------------------------------------------------------------
// 执行器有物理限幅（u∈[−uMax,uMax]）。当误差大、PID 输出长期饱和时，积分器仍
// 在"偷偷"累积 → 一旦误差反向，巨大的积分项把系统甩出大超调，这就是 windup。
// 本 demo 对比：① 无抗饱和（积分照常累积）② 条件积分(clamping)抗饱和。
// 全程合成数据（一阶 plant + 限幅），运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 pid_l1_antiwindup.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Cfg { double Kp=3, Ki=1.5, Kd=0, uMax=2.0, dt=0.02; bool aw=false; };

// 一阶 plant G(s)=1/(s+1)，带执行器限幅；aw=true 时启用条件积分抗饱和
static vector<double> sim(const Cfg& c, int N, double sp){
    vector<double> y(N+1, 0.0);
    double x=0, integ=0, pe=0;
    for(int k=0;k<N;k++){
        double e = sp - x;
        double der = (e - pe)/c.dt;
        double u_unsat = c.Kp*e + integ + c.Kd*der;
        double u;
        if(u_unsat > c.uMax){
            u = c.uMax;
            if(!c.aw) integ += c.Ki*e*c.dt;   // 无抗饱和：饱和时仍积分 → windup
        } else if(u_unsat < -c.uMax){
            u = -c.uMax;
            if(!c.aw) integ += c.Ki*e*c.dt;
        } else {
            u = u_unsat;
            integ += c.Ki*e*c.dt;             // 未饱和才积分
        }
        x = x + c.dt*(-x + u);                // plant 积分
        pe = e; y[k+1] = x;
    }
    return y;
}

static double steadyErr(const vector<double>& y, double sp){ return sp - y.back(); }
static double overshoot(const vector<double>& y, double sp){
    double peak=*max_element(y.begin(),y.end()); double f=y.back();
    return f<=1e-6?0:max(0.0,(peak-f)/f);
}
static bool finite(const vector<double>& y){ for(double v:y) if(!isfinite(v)) return false; return true; }

int main(){
    bool pass=true;
    const double dt=0.02, N=1500; Cfg base; base.dt=dt;
    cout<<fixed<<setprecision(4);

    Cfg cNo = base; cNo.aw=false;
    Cfg cAW = base; cAW.aw=true;
    auto yNo = sim(cNo, N, 1.0);
    auto yAW = sim(cAW, N, 1.0);

    double oNo=overshoot(yNo,1), oAW=overshoot(yAW,1);
    double eNo=steadyErr(yNo,1), eAW=steadyErr(yAW,1);
    cout<<"[L1.5 抗积分饱和] plant G(s)=1/(s+1), uMax="<<base.uMax<<"\n";
    cout<<"  无抗饱和 : 超调="<<oNo*100<<"%  稳态误差="<<eNo<<"\n";
    cout<<"  条件积分 : 超调="<<oAW*100<<"%  稳态误差="<<eAW<<"\n";

    if(!finite(yNo)||!finite(yAW)){ pass=false; cout<<"  [失败] 响应发散\n"; }
    if(!(oAW < 0.4)){ pass=false; cout<<"  [失败] 抗饱和后超调仍过大\n"; }
    if(eAW > 0.05){ pass=false; cout<<"  [失败] 抗饱和后未达设定值\n"; }
    if(!(oNo > oAW + 0.1)){ pass=false; cout<<"  [失败] 抗饱和未显著改善超调\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 条件积分有效抑制 windup 超调":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
