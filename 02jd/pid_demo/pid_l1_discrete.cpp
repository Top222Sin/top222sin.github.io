// pid_l1_discrete.cpp
// ============================================================================
// L1.3  PID 离散化形式（位置式 vs 增量式；采样周期 dt 的影响）
// ----------------------------------------------------------------------------
// ① 位置式 PID 与增量式(velocity) PID 是同一连续 PID 的两种写法：用 PD(Ki=0)
//    证明二者逐拍完全一致（浮点级吻合）。
// ② 同一 PID 用不同采样周期 dt 实现：dt 越大，离散化误差越大、响应越差
//    （这里把 plant 用细子步积分、只让控制器按 dt 采样，隔离"控制器离散化"效应）。
// 全程合成数据，运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 pid_l1_discrete.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// plant 用细子步欧拉积分，控制器按 dt 采样（零阶保持），隔离控制器离散化效应
static vector<double> simPos(double Kp,double Ki,double Kd,double dt,int N,double sp,double plantT){
    vector<double> y(N+1,0.0); double x=0, integ=0, pe=0;
    double fine=dt/20.0; int M=20;
    for(int k=0;k<N;k++){
        double e=sp-x;
        integ+=Ki*e*dt;
        double der=(e-pe)/dt;
        double u=Kp*e+integ+Kd*der;
        double xx=x; for(int s=0;s<M;s++) xx=xx+fine*(-xx+u)/plantT;
        x=xx; pe=e; y[k+1]=x;
    }
    return y;
}
static vector<double> simInc(double Kp,double Ki,double Kd,double dt,int N,double sp,double plantT){
    vector<double> y(N+1,0.0); double x=0, u=0, e1=0, e2=0;
    double fine=dt/20.0; int M=20;
    for(int k=0;k<N;k++){
        double e=sp-x;
        double du=Kp*(e-e1)+Ki*e*dt+Kd*(e-2*e1+e2)/dt;
        u=u+du;
        double xx=x; for(int s=0;s<M;s++) xx=xx+fine*(-xx+u)/plantT;
        x=xx; e2=e1; e1=e; y[k+1]=x;
    }
    return y;
}

static double steadyErr(const vector<double>& y,double sp){ return sp-y.back(); }
static double overshoot(const vector<double>& y,double sp){
    double peak=*max_element(y.begin(),y.end()); double f=y.back();
    return f<=1e-6?0:max(0.0,(peak-f)/f);
}

int main(){
    bool pass=true;
    const double plantT=0.5, sp=1.0;
    cout<<fixed<<setprecision(5);

    // ① 位置式 vs 增量式（PD, Ki=0 → 应逐拍完全一致）
    auto yP=simPos(3,0,0.5,0.02,500,sp,plantT);
    auto yI=simInc(3,0,0.5,0.02,500,sp,plantT);
    double maxdiff=0; for(size_t i=0;i<yP.size();i++) maxdiff=max(maxdiff,abs(yP[i]-yI[i]));
    cout<<"[L1.3 离散化形式]\n  位置式 vs 增量式(PD) 最大偏差="<<maxdiff<<"\n";
    if(maxdiff>1e-9){ pass=false; cout<<"  [失败] 两种形式在 PD 下应逐拍一致\n"; }

    // ② 采样周期 dt 的影响（隔离控制器离散化）
    auto y1=simPos(3,2,0.5,0.01,800,sp,plantT);
    auto y2=simPos(3,2,0.5,0.10,800,sp,plantT);
    auto y3=simPos(3,2,0.5,0.30,800,sp,plantT);
    double o1=overshoot(y1,sp), o2=overshoot(y2,sp), o3=overshoot(y3,sp);
    double e1=steadyErr(y1,sp), e3=steadyErr(y3,sp);
    cout<<"  dt=0.01s: 超调="<<o1*100<<"%  稳态误差="<<e1<<"\n";
    cout<<"  dt=0.10s: 超调="<<o2*100<<"%  稳态误差="<<steadyErr(y2,sp)<<"\n";
    cout<<"  dt=0.30s: 超调="<<o3*100<<"%  稳态误差="<<e3<<"\n";

    if(e1>0.05){ pass=false; cout<<"  [失败] 小 dt 下应良好跟踪\n"; }
    if(!(o3 >= o1 - 1e-3)){ pass=false; cout<<"  [失败] dt 增大应使响应不改善\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 两形式等价(PD) 且 dt 越大离散化越差":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
