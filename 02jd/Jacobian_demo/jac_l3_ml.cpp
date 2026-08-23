// jac_l3_ml.cpp
// ============================================================================
// L3.3 机器学习：反向传播 = 雅可比链式法则
// ----------------------------------------------------------------------------
// 一个最小的全连接网络：  x(2) → [W1,b1] → z1 → tanh → h(3) → [W2,b2] → y(1)
// 标量损失  L = ½ (y − y*)²。
// 反向传播的数学本质，就是把"损失对各参数的梯度"写成雅可比的链式乘积：
//   δ2 = ∂L/∂z2 = (y − y*)
//   δ1 = (W2ᵀ δ2) ⊙ tanh'(z1)
//   ∂L/∂W2 = δ2 · hᵀ,   ∂L/∂b2 = δ2
//   ∂L/∂W1 = δ1 · xᵀ,   ∂L/∂b1 = δ1
// 本 demo：
//   (1) 反向传播得到的梯度，与逐参数有限差分数值梯度一致；
//   (2) 输出对输入的雅可比 ∂y/∂x = W2·diag(tanh')·W1，与 FD 一致。
// 一句话：反向传播就是雅可比链式法则的工程化实现。
//
// 编译: g++ -O3 -std=c++17 jac_l3_ml.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

static const int DIN=2, DH=3, DOUT=1;
// 固定（确定性）参数，便于复现
static double W1[DH][DIN]={{0.5,-0.3},{0.2,0.8},{-0.6,0.1}};
static double b1[DH]={0.1,-0.2,0.05};
static double W2[DOUT][DH]={{0.4,0.9,-0.5}};
static double b2[DOUT]={0.2};
static const double X[DIN]={1.0,-0.5};
static const double YT=0.7;

static double tanh_(double v){ return tanh(v); }

static double forward(const double x[DIN], double z1[DH], double h[DH]){
    for(int i=0;i<DH;i++){
        z1[i]=W1[i][0]*x[0]+W1[i][1]*x[1]+b1[i];
        h[i]=tanh_(z1[i]);
    }
    double y=0; for(int i=0;i<DH;i++) y+=W2[0][i]*h[i]; y+=b2[0];
    return y;
}
static double loss(const double x[DIN]){
    double z1[DH],h[DH]; double y=forward(x,z1,h);
    return 0.5*(y-YT)*(y-YT);
}

int main(){
    bool pass=true;
    cout<<fixed<<setprecision(9);
    double z1[DH],h[DH]; double y=forward(X,z1,h);

    // ---- 反向传播 ----
    double d2=y-YT;                                  // ∂L/∂z2 (DOUT=1)
    double gW2[DOUT][DH]; for(int k=0;k<DH;k++) gW2[0][k]=d2*h[k];
    double gb2[DOUT]={d2};
    double d1[DH];
    for(int i=0;i<DH;i++) d1[i]=W2[0][i]*d2*(1-h[i]*h[i]);   // (W2ᵀδ2)⊙tanh'
    double gW1[DH][DIN];
    for(int i=0;i<DH;i++) for(int k=0;k<DIN;k++) gW1[i][k]=d1[i]*X[k];
    double gb1[DH]; for(int i=0;i<DH;i++) gb1[i]=d1[i];

    // ---- 数值梯度（中心差分） ----
    double emax=0; const double hh=1e-6;
    // W2
    for(int k=0;k<DH;k++){
        double o=W2[0][k]; W2[0][k]=o+hh; double lp=loss(X); W2[0][k]=o-hh; double lm=loss(X); W2[0][k]=o;
        double ng=(lp-lm)/(2*hh); emax=max(emax,fabs(gW2[0][k]-ng));
    }
    // b1
    for(int i=0;i<DH;i++){
        double o=b1[i]; b1[i]=o+hh; double lp=loss(X); b1[i]=o-hh; double lm=loss(X); b1[i]=o;
        double ng=(lp-lm)/(2*hh); emax=max(emax,fabs(gb1[i]-ng));
    }
    // W1
    for(int i=0;i<DH;i++) for(int k=0;k<DIN;k++){
        double o=W1[i][k]; W1[i][k]=o+hh; double lp=loss(X); W1[i][k]=o-hh; double lm=loss(X); W1[i][k]=o;
        double ng=(lp-lm)/(2*hh); emax=max(emax,fabs(gW1[i][k]-ng));
    }
    cout<<"[L3.3] 反向传播梯度 vs 数值梯度 最大误差 = "<<emax<<"\n";
    if(!(emax<1e-6)){ pass=false; cout<<"  [失败] 反向传播梯度与数值梯度不符\n"; }

    // ---- 输出对输入雅可比 ∂y/∂x = W2·diag(tanh')·W1 (1x2) ----
    double Jyx[DOUT][DIN]={0};
    for(int k=0;k<DIN;k++){
        double s=0; for(int i=0;i<DH;i++) s+=W2[0][i]*(1-h[i]*h[i])*W1[i][k];
        Jyx[0][k]=s;
    }
    double ey=0;
    for(int k=0;k<DIN;k++){
        double xp[DIN]={X[0],X[1]}; xp[k]+=hh; double yp=forward(xp,z1,h);
        double xm[DIN]={X[0],X[1]}; xm[k]-=hh; double ym=forward(xm,z1,h);
        double ng=(yp-ym)/(2*hh); ey=max(ey,fabs(Jyx[0][k]-ng));
    }
    cout<<"[L3.3] ∂y/∂x 解析=("<<Jyx[0][0]<<", "<<Jyx[0][1]<<")  与FD最大误差="<<ey<<"\n";
    if(!(ey<1e-6)){ pass=false; cout<<"  [失败] ∂y/∂x 与FD不符\n"; }

    cout<<"\n========================================\n";
    cout<<(pass? " [PASS] 反向传播 = 雅可比链式法则；∂y/∂x 正确" : " [FAIL] 见上")<<"\n";
    cout<<"========================================\n";
    return pass?0:1;
}
