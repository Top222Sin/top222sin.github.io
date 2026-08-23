// pid_l2_stability.cpp
// ============================================================================
// L2.4  离散稳定性与频率域
// ----------------------------------------------------------------------------
// 离散闭环稳不稳，看特征根(极点)是否在单位圆内。本 demo 用一阶 plant x'=-x+u：
//   • 前向欧拉离散：极点 p_FE = 1 − dt·(1+Kp)，稳定需 |p_FE|<1 ⟹ Kp < 2/dt − 1
//   • Tustin(双线性)离散：A-稳定，极点恒在单位圆内 ⟹ 任意 Kp>0 都稳
// 用仿真验证：同一 Kp，前向欧拉越界就发散，Tustin 永远收敛。并展示 dt 越大稳定域越窄。
// 全程合成数据，运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 pid_l2_stability.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// 前向欧拉闭环：x_{k+1}=x_k + dt·(−x_k + Kp·(R−x_k))
static double simFE(double dt,double Kp,int N,double R=1.0){
    double x=0; for(int k=0;k<N;k++){ x = x + dt*(-x + Kp*(R - x)); }
    return x;
}
// Tustin 闭环：x_{k+1} = (2Kp − (b+Kp)·x_k)/(a+Kp)，a=1+2/dt, b=1−2/dt
static double simTustin(double dt,double Kp,int N,double R=1.0){
    double a=1+2/dt, b=1-2/dt; double x=0;
    for(int k=0;k<N;k++){ x = (2*Kp*R - (b+Kp)*x)/(a+Kp); }
    return x;
}

int main(){
    bool pass=true;
    const int N=600;
    cout<<fixed<<setprecision(4);

    // ① 前向欧拉：小 Kp 稳定，大 Kp 发散
    double dt=0.1;
    double pFE_small = 1 - dt*(1+5);     // Kp=5
    double pFE_big   = 1 - dt*(1+25);    // Kp=25
    double xFE_s = simFE(dt,5,N);
    double xFE_b = simFE(dt,25,N);
    cout<<"[L2.4 离散稳定性] dt="<<dt<<"\n";
    cout<<"  前向欧拉 Kp=5 : 极点|p|="<<abs(pFE_small)<<"  末值x="<<xFE_s
        <<"  ("<<(abs(pFE_small)<1?"稳定":"发散")<<")\n";
    cout<<"  前向欧拉 Kp=25: 极点|p|="<<abs(pFE_big)<<"  末值x="<<xFE_b
        <<"  ("<<(abs(pFE_big)<1?"稳定":"发散")<<")\n";

    if(!(abs(pFE_small)<1)){ pass=false; cout<<"  [失败] Kp=5 应稳定\n"; }
    if(!(abs(xFE_s)<5)){ pass=false; cout<<"  [失败] Kp=5 应收敛\n"; }
    if(!(abs(pFE_big)>1)){ pass=false; cout<<"  [失败] Kp=25 应越界\n"; }
    if(!(abs(xFE_b)>1e3)){ pass=false; cout<<"  [失败] Kp=25 应发散\n"; }

    // ② Tustin：同样 Kp=25 仍稳定（A-稳定）
    double xT = simTustin(dt,25,N);
    cout<<"  Tustin   Kp=25: 末值x="<<xT<<"  ("<<(abs(xT)<5?"稳定":"发散")<<")\n";
    if(!(abs(xT)<5)){ pass=false; cout<<"  [失败] Tustin 应始终稳定\n"; }

    // ③ dt 越大稳定域越窄：dt=0.05 时 Kp=50 前向欧拉也应发散
    double dt2=0.05;
    double pFE_d2 = 1 - dt2*(1+50);
    double xFE_d2 = simFE(dt2,50,N);
    cout<<"  dt="<<dt2<<" 前向欧拉 Kp=50: 极点|p|="<<abs(pFE_d2)<<" 末值x="<<xFE_d2<<"\n";
    if(!(abs(pFE_d2)>1)){ pass=false; cout<<"  [失败] 大 dt 下 Kp=50 应越界\n"; }
    if(!(abs(xFE_d2)>1e3)){ pass=false; cout<<"  [失败] 大 dt 下 Kp=50 应发散\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] 极点|p|<1↔稳定；前向欧拉有稳定界，Tustin A-稳定":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
