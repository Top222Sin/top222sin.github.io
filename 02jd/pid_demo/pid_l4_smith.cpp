// pid_l4_smith.cpp
// ============================================================================
// L2  Smith 预估器 (Smith Predictor) —— 大纯滞后对象的标配补偿
// ----------------------------------------------------------------------------
// FOPDT 对象: dy/dt = (-y + K*u(t-L))/T   (一阶 + 死区时间 L)
// 死区时间一旦进入反馈环, 会严重限制可用增益(否则振荡/发散)。
// Smith 预估器在反馈环内"减去滞后、加回无滞后模型输出", 把 L 移出反馈环:
//   反馈给控制器的量 yf = y - ym_delayed + ym   (ym=无滞后模型输出)
// 于是同样激进的增益, 普通 PI 直接作用于带滞后对象会失稳, Smith 下却稳又快。
//
// 编译: g++ -O3 -std=c++17 pid_l4_smith.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

static vector<double> piPlain(double r,double Kp,double Ki,double K,double Tp,double L,double dt,int N){
    vector<double> ys(N,0), u(N,0);
    int d=int(round(L/dt));
    double y=0, integ=0;
    for(int k=1;k<N;k++){
        double uk = u[max(0,k-d)];          // 滞后 L 的执行量
        y = y + dt*(-y + K*uk)/Tp;
        double e = r - y;
        integ += Ki*e*dt;
        u[k] = Kp*e + integ;
        ys[k]=y;
    }
    return ys;
}

static vector<double> piSmith(double r,double Kp,double Ki,double K,double Tp,double L,double Lm,double dt,int N){
    vector<double> ys(N,0), u(N,0);
    int d=int(round(L/dt)), dm=int(round(Lm/dt));
    double y=0, ym=0, integ=0;
    vector<double> ymh(dm+1, 0.0);
    for(int k=1;k<N;k++){
        double uk = u[k-1];
        y  = y  + dt*(-y  + K*uk)/Tp;       // 真实对象(含滞后)
        ym = ym + dt*(-ym + K*uk)/Tp;       // 模型(无滞后)
        ymh.push_back(ym);
        double ymd = ymh[ymh.size()-1-dm];  // 模型滞后输出
        double yf  = y - ymd + ym;          // 预测(无滞后)反馈量
        double e = r - yf;
        integ += Ki*e*dt;
        u[k] = Kp*e + integ;
        ys[k]=y;
    }
    return ys;
}

static double settle(const vector<double>& y,double r,double dt,double band=0.02){
    double lo=r*(1-band), hi=r*(1+band); int entered=-1;
    for(int k=0;k<(int)y.size();k++){
        if(lo<=y[k]&&y[k]<=hi){ if(entered<0)entered=k; } else entered=-1;
    }
    return entered<0? y.size()*dt : entered*dt;
}
static double overshoot(const vector<double>& y,double r){
    double m=0; for(double v:y) if(v>m)m=v;
    return r>0? max(0.0,(m-r)/r) : 0.0;
}

int main(){
    bool pass=true;
    const double dt=0.05, T=60.0; int N=int(T/dt);
    const double K=1.0, Tp=2.0, L=4.0, r=1.0;
    const double Kp=1.6, Ki=0.8;   // 对带滞后对象而言"激进"的增益
    cout<<fixed<<setprecision(4);
    cout<<"[L2 Smith 预估器] FOPDT  K="<<K<<" T="<<Tp<<" L="<<L<<"  增益 Kp="<<Kp<<" Ki="<<Ki<<"\n\n";

    auto yp = piPlain(r, Kp, Ki, K, Tp, L, dt, N);
    auto ys = piSmith(r, Kp, Ki, K, Tp, L, L, dt, N);

    double sp=settle(yp,r,dt), ss=settle(ys,r,dt);
    double op=overshoot(yp,r), os=overshoot(ys,r);
    cout<<"  普通PI(同激进增益) : 调节时间="<<sp<<"s 超调="<<op*100<<"%  末值="<<yp.back()<<"\n";
    cout<<"  Smith(同激进增益)   : 调节时间="<<ss<<"s 超调="<<os*100<<"%  末值="<<ys.back()<<"\n";

    if(!(abs(ys.back()-r)<0.05)){ pass=false; cout<<"  [失败] Smith 未收敛\n"; }
    if(!(op>0.5)){ pass=false; cout<<"  [失败] 同增益普通PI 未失稳\n"; }
    if(!(ss<0.8*sp)){ pass=false; cout<<"  [失败] Smith 不比普通PI快\n"; }
    if(!(os<0.2*op)){ pass=false; cout<<"  [失败] Smith 超调未显著更小\n"; }

    // 模型失配: 模型 Lm=3.5 略小于真实 4.0, 仍应稳且快
    auto ys2 = piSmith(r, Kp, Ki, K, Tp, L, 3.5, dt, N);
    double ss2=settle(ys2,r,dt);
    cout<<"  模型略失配(Lm=3.5)  : 调节时间="<<ss2<<"s 末值="<<ys2.back()<<"\n";
    if(!(abs(ys2.back()-r)<0.05 && ss2<0.8*sp)){ pass=false; cout<<"  [失败] 失配下 Smith 退化\n"; }

    cout<<"\n=================================================\n";
    cout<<(pass? " [PASS] Smith 把死区移出反馈环: 同增益稳而快":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
