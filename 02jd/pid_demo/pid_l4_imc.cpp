// pid_l4_imc.cpp
// ============================================================================
// L2  IMC / λ 整定 (Internal Model Control) —— 一阶对象的"一个旋钮"整定法
// ----------------------------------------------------------------------------
// 一阶对象 Gp(s)=K/(Ts+1): IMC 控制器 C(s)=(Ts+1)/(K(λs+1))
// 等价 PI 增益(闭式):  Kc = T/(Kλ),   τI = T,   Ki = Kc/τI = 1/(Kλ)
// 完美模型下闭环 Y/R = 1/(λs+1)  -> 时间常数 TC = λ
//   λ 是唯一旋钮: λ 小→快但激进; λ 大→慢但鲁棒。比 Z-N 更可解释、可直接设带宽。
//
// 编译: g++ -O3 -std=c++17 pid_l4_imc.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <utility>

using namespace std;

static vector<double> simPI(double r,double Kc,double Ki,double K,double Tp,double dt,int N){
    vector<double> ys(N,0); double y=0, integ=0;
    for(int k=1;k<N;k++){
        double e=r-y;
        integ += Ki*e*dt;
        double u = Kc*e + integ;
        y = y + dt*(-y + K*u)/Tp;
        ys[k]=y;
    }
    return ys;
}
static double settle(const vector<double>& y,double r,double dt,double band=0.02){
    double lo=r*(1-band), hi=r*(1+band); int entered=-1;
    for(int k=0;k<(int)y.size();k++){ if(lo<=y[k]&&y[k]<=hi){ if(entered<0)entered=k; } else entered=-1; }
    return entered<0? y.size()*dt : entered*dt;
}

int main(){
    bool pass=true;
    const double dt=0.02, T=25.0; int N=int(T/dt);
    const double K=1.0, Tp=2.0, r=1.0;
    auto imc=[&](double lam){ double Kc=Tp/(K*lam); double Ki=1.0/(K*lam); return make_pair(Kc,Ki); };
    cout<<fixed<<setprecision(4);
    cout<<"[L2 IMC / λ 整定] 对象 Gp(s)="<<K<<"/("<<Tp<<"s+1)   IMC 增益 Kc=T/(Kλ), Ki=1/(Kλ)\n\n";

    // 1) 增益公式
    auto g=imc(2.0);
    bool f1 = abs(g.first-Tp/(K*2.0))<1e-9 && abs(g.second-1.0/(K*2.0))<1e-9;
    cout<<"  1. 增益公式 Kc="<<g.first<<" Ki="<<g.second<<"  (Kc=T/(Kλ), Ki=1/(Kλ))\n";
    if(!f1){ pass=false; cout<<"  [失败] 增益公式错\n"; }

    // 2) 闭环 TC ≈ λ
    for(double lam: {1.0,2.0,4.0}){
        auto gg=imc(lam);
        auto ys=simPI(r, gg.first, gg.second, K, Tp, dt, N);
        double st=settle(ys,r,dt); double exp=3.912*lam;
        cout<<"  λ="<<lam<<": 调节时间="<<st<<"s (≈3.912λ="<<exp<<")  末值="<<ys.back()<<"\n";
        if(!(abs(st-exp)<0.6)){ pass=false; cout<<"  [失败] λ="<<lam<<" TC 不符\n"; }
        if(!(abs(ys.back()-r)<0.01)){ pass=false; cout<<"  [失败] λ="<<lam<<" 稳态误差\n"; }
    }

    // 3) λ 单调可调
    auto a=imc(1.0), b=imc(4.0);
    double st1=settle(simPI(r,a.first,a.second,K,Tp,dt,N),r,dt);
    double st4=settle(simPI(r,b.first,b.second,K,Tp,dt,N),r,dt);
    cout<<"  λ: 1→调节"<<st1<<"s, 4→调节"<<st4<<"s  (λ↑ 更慢更保守)\n";
    if(!(st4>st1+1.0)){ pass=false; cout<<"  [失败] λ 不单调\n"; }

    // 4) 鲁棒性: 对象增益失配 K_actual=1.5, 模型 K=1 -> 有效 TC=λ/K_actual, 零稳态误差
    double lam=2.0; auto gg=imc(lam);
    auto ys=simPI(r, gg.first, gg.second, 1.5, Tp, dt, N);
    double st=settle(ys,r,dt); double exp=3.912*(lam/1.5);
    cout<<"  增益失配 K=1.5(模型K=1): 调节="<<st<<"s (有效TC=λ/K= "<<exp<<") 末值="<<ys.back()<<"\n";
    if(!(abs(ys.back()-r)<0.02 && abs(st-exp)<0.6)){ pass=false; cout<<"  [失败] 失配鲁棒性不符\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] IMC: λ 是唯一旋钮, TC=λ 可预测可设带宽":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
