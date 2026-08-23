// pid_l2_px4_arch.cpp
// ============================================================================
// L2.3  多旋翼 PID 架构解析（PX4 代码走读）⭐
// ----------------------------------------------------------------------------
// 把 PX4 真实飞控控制律搬到纯 C++ 里跑，验证三件事：
//   ① 形式等价：PX4 并行式 u = K·(P·e + I·∫e + D·d) 与经典 PID 等效增益
//      (Kp=K·P, Ki=K·I, Kd=K·D) 逐拍一致 —— 对应 rate_control.hpp 的 setPidGains
//   ② 前馈 FF：u += FF·rate_sp 能立刻补上设定值，减小跟踪滞后
//   ③ 抗饱和：来自 control allocator 的饱和反馈(setSaturationStatus) → 条件积分，
//      与裸 PID 相比显著抑制 windup 超调
// 全程合成数据（一阶角速率 plant），运行即出 PASS/FAIL。
// 参照: lib/rate_control/rate_control.hpp, mc_rate_control/MulticopterRateControl.cpp
//
// 编译: g++ -O3 -std=c++17 pid_l2_px4_arch.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Px4Cfg { double K=1,P=0,I=0,D=0,FF=0,dt=0.005,intLim=1e9,uMax=1e9; bool filterD=false; };

static void simPx4(const Px4Cfg& c, double tau, int N, double sp, bool aw,
                   vector<double>& y, vector<double>& u){
    y.assign(N+1,0); u.assign(N+1,0);
    double x=0, integ=0, dlp=0, pe=0;
    for(int k=0;k<N;k++){
        double e=sp-x;
        double deriv=(e-pe)/c.dt;
        double dterm=deriv;
        if(c.filterD){ double a=c.dt/(0.02+c.dt); dlp=dlp+a*(deriv-dlp); dterm=dlp; }
        double uu = c.K*(c.P*e + integ + c.D*dterm) + c.FF*sp;   // 并行式 + 前馈
        bool sat=false;
        if(uu>c.uMax){ uu=c.uMax; sat=true; }
        else if(uu<-c.uMax){ uu=-c.uMax; sat=true; }
        if(!(aw && sat)) integ += c.K*c.I*e*c.dt;                // 条件积分(抗饱和)
        if(integ>c.intLim) integ=c.intLim; if(integ<-c.intLim) integ=-c.intLim;
        x = x + c.dt*((-x + uu)/tau);
        pe=e; y[k+1]=x; u[k+1]=uu;
    }
}
static void simClassic(double Kp,double Ki,double Kd,double dt,double tau,int N,double sp,
                       vector<double>& y, vector<double>& u){
    y.assign(N+1,0); u.assign(N+1,0);
    double x=0, integ=0, pe=0;
    for(int k=0;k<N;k++){
        double e=sp-x; double deriv=(e-pe)/dt;
        double uu=Kp*e+integ+Kd*deriv; integ+=Ki*e*dt;
        x=x+dt*((-x+uu)/tau); pe=e; y[k+1]=x; u[k+1]=uu;
    }
}

static double maxDiff(const vector<double>& a,const vector<double>& b){ double m=0; for(size_t i=0;i<a.size();i++) m=max(m,abs(a[i]-b[i])); return m; }
static double rmsErr(const vector<double>& y,double sp,int to){ double s=0; for(int i=0;i<to;i++) s+=(y[i]-sp)*(y[i]-sp); return sqrt(s/to); }
static double overshoot(const vector<double>& y,double sp){ double pk=*max_element(y.begin(),y.end()); return pk<=sp?0:pk-sp; }
static double steadyErr(const vector<double>& y,double sp){ return sp-y.back(); }
static bool finite(const vector<double>& y){ for(double v:y) if(!isfinite(v)) return false; return true; }

int main(){
    bool pass=true;
    cout<<fixed<<setprecision(5);

    // ① 形式等价：K=1, P=0.15,I=0.2,D=0.003 → 等效 Kp=0.15,Ki=0.2,Kd=0.003
    Px4Cfg c1{1,0.15,0.2,0.003,0,0.005,1e9,1e9,false};
    vector<double> yP,uP,yC,uC;
    simPx4(c1,0.05,1000,1.0,false,yP,uP);
    simClassic(0.15,0.2,0.003,0.005,0.05,1000,1.0,yC,uC);
    double md=maxDiff(yP,yC);
    cout<<"[L2.3 PX4 架构] ① 并行式 vs 经典 PID 最大偏差="<<md<<"\n";
    if(md>1e-9){ pass=false; cout<<"  [失败] 两种形式应逐拍一致\n"; }

    // ② 前馈 FF 改善跟踪
    Px4Cfg c0=c1; c0.FF=0.0; c0.filterD=true;
    Px4Cfg cF=c1; cF.FF=0.1; cF.filterD=true;
    vector<double> y0,u0,yF,uF;
    simPx4(c0,0.05,400,1.0,false,y0,u0);
    simPx4(cF,0.05,400,1.0,false,yF,uF);
    double r0=rmsErr(y0,1.0,150), rF=rmsErr(yF,1.0,150);
    cout<<"  ② 无FF RMS="<<r0<<"  有FF(0.1) RMS="<<rF<<"\n";
    if(!(rF < r0)){ pass=false; cout<<"  [失败] 前馈未改善跟踪\n"; }

    // ③ 抗饱和(条件积分) 抑制 windup
    Px4Cfg cW; cW.K=1; cW.P=3; cW.I=2.0; cW.D=0; cW.dt=0.02; cW.uMax=0.5; cW.intLim=1e9;
    vector<double> yN,uN,yA,uA;
    simPx4(cW,0.2,1500,1.0,false,yN,uN);   // 无抗饱和
    simPx4(cW,0.2,1500,1.0,true ,yA,uA);   // 条件积分抗饱和
    double oN=overshoot(yN,1), oA=overshoot(yA,1);
    cout<<"  ③ 无抗饱和超调="<<oN*100<<"%  条件积分超调="<<oA*100<<"%  稳态误差="<<steadyErr(yA,1)<<"\n";
    if(!finite(yN)||!finite(yA)){ pass=false; cout<<"  [失败] 发散\n"; }
    if(oA>0.4){ pass=false; cout<<"  [失败] 抗饱和后仍超调过大\n"; }
    if(!(oN > oA + 0.1)){ pass=false; cout<<"  [失败] 抗饱和未显著改善\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] PX4 并行式等价经典PID + FF有效 + 条件积分抗饱和":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
