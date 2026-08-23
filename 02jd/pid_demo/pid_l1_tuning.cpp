// pid_l1_tuning.cpp
// ============================================================================
// L1.4  整定直觉（Ziegler-Nichols 阶跃响应法 + 试凑对比）
// ----------------------------------------------------------------------------
// 经典 Z-N 有两类：① 闭环临界振荡法（需 plant 有相位穿越）② 开环阶跃响应法
// （对带纯延迟的一阶/二阶 plant 最实用）。本 demo 用②：
//   ① 对"已知"plant 做开环阶跃，拟合 FOPDT(K,T,L)（含少量噪声，测拟合鲁棒性）
//   ② 套 Z-N 公式得 Kp/Ti/Td，跑闭环 PID
//   ③ 对比"纯 P 试凑"——证明 Z-N 自动整定消除稳态误差且不过度振荡
// 全程合成数据（plant 真值已知），运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 pid_l1_tuning.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

using namespace std;

struct Plant { double K=1.0, T=2.0, L=0.5; };

// 开环阶跃响应（含纯延迟 L），加可控噪声
static vector<double> openStep(const Plant& p, double dt, int N, mt19937& g, double noise){
    vector<double> y(N+1,0.0); uniform_real_distribution<double> un(-noise,noise);
    for(int k=0;k<=N;k++){
        double t=k*dt;
        double yy = (t>=p.L)? p.K*(1.0-exp(-(t-p.L)/p.T)) : 0.0;
        y[k]=yy+un(g);
    }
    return y;
}

// FOPDT 图解法拟合：K=final, L=首次越过阈值, T=(达 63.2% 时刻)-L
static void fitFOPDT(const vector<double>& y, double dt, double& K, double& T, double& L){
    int N=(int)y.size()-1;
    K=0; for(int i=N-19;i<=N;i++) K+=y[i]; K/=20.0;
    // 找 L：首个 y>0.02*K? 用绝对阈值 0.02
    L=0; bool found=false;
    for(int k=0;k<=N;k++){ if(y[k]>0.02){ L=k*dt; found=true; break; } }
    if(!found) L=0;
    // 找 63.2% 时刻
    double target=0.632*K; int k632=N;
    for(int k=0;k<=N;k++){ if(y[k]>=target){ k632=k; break; } }
    T = max(0.05, k632*dt - L);
}

// 闭环 PID（plant 带纯延迟，零阶保持）
static vector<double> closedLoop(const Plant& p, double Kp,double Ki,double Kd,double dt,int N){
    vector<double> y(N+1,0.0); double x=0, integ=0, pe=0;
    int dN=max(1,(int)round(p.L/dt)); vector<double> buf(dN,0); int bi=0;
    for(int k=0;k<N;k++){
        double u_in=buf[bi];                 // 延迟后的控制量
        x = x + dt*(-x + p.K*u_in)/p.T;      // 一阶 plant
        double e=1.0 - x;
        integ += Ki*e*dt;
        double der=(e-pe)/dt;
        double u=Kp*e + integ + Kd*der;
        buf[bi]=u; bi=(bi+1)%dN; pe=e; y[k+1]=x;
    }
    return y;
}

static double steadyErr(const vector<double>& y){ return 1.0 - y.back(); }
static double overshoot(const vector<double>& y){
    double peak=*max_element(y.begin(),y.end()); double f=y.back();
    return f<=1e-6?0:max(0.0,(peak-f)/f);
}
static bool finite(const vector<double>& y){
    for(double v:y) if(!isfinite(v)) return false; return true;
}

int main(){
    bool pass=true; mt19937 g(20260817);
    const double dt=0.05, N=400; Plant p;
    cout<<fixed<<setprecision(4);

    auto ys=openStep(p,dt,N,g,0.003);
    double K,T,L; fitFOPDT(ys,dt,K,T,L);
    cout<<"[L1.4 整定] plant 真值 K="<<p.K<<" T="<<p.T<<" L="<<p.L<<"\n";
    cout<<"  拟合 FOPDT: K="<<K<<" T="<<T<<" L="<<L<<"\n";

    // Z-N 阶跃响应公式
    double Kp=1.2*T/(K*L), Ti=2.0*L, Td=0.5*L;
    double Ki=Kp/Ti, Kd=Kp*Td;
    cout<<"  Z-N 增益: Kp="<<Kp<<" Ki="<<Ki<<" Kd="<<Kd<<"\n";

    auto yP = closedLoop(p, 1.0, 0.0, 0.0, dt, N);   // 试凑：纯 P
    auto yZN= closedLoop(p, Kp, Ki, Kd, dt, N);       // Z-N PID

    double eP=steadyErr(yP), eZN=steadyErr(yZN);
    double oZN=overshoot(yZN), oP=overshoot(yP);
    cout<<"  纯P(Kp=1): 稳态误差="<<eP<<" 超调="<<oP*100<<"%\n";
    cout<<"  Z-N PID : 稳态误差="<<eZN<<" 超调="<<oZN*100<<"%\n";

    if(abs(K-p.K)/p.K>0.1){ pass=false; cout<<"  [失败] K 拟合偏差过大\n"; }
    if(abs(T-p.T)/p.T>0.25){ pass=false; cout<<"  [失败] T 拟合偏差过大\n"; }
    if(abs(L-p.L)/p.L>0.4){ pass=false; cout<<"  [失败] L 拟合偏差过大\n"; }
    if(!finite(yZN)){ pass=false; cout<<"  [失败] Z-N 闭环发散\n"; }
    if(!(eP > eZN + 0.1)){ pass=false; cout<<"  [失败] Z-N 未显著改善稳态误差\n"; }
    if(eZN>0.1){ pass=false; cout<<"  [失败] Z-N 稳态误差过大\n"; }
    if(oZN>0.7){ pass=false; cout<<"  [失败] Z-N 超调过大\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] FOPDT 拟合 + Z-N 整定有效消除稳态误差":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
