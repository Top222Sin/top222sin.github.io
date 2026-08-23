// pid_l1_terms.cpp
// ============================================================================
// L1.2  PID 三项（P / I / D）的物理意义
// ----------------------------------------------------------------------------
// 用同一个一阶 plant G(s)=1/(s+1)，分别加 P / PI / PID 控制器，看：
//   P  ：响应快，但**有稳态误差**（type-0 系统必有 offset）
//   PI ：积分消除了稳态误差，但引入超调/振荡
//   PID：微分提供阻尼，抑制超调、加快稳定
// 全程合成数据（plant 已知），运行即出 PASS/FAIL + 三项对比指标。
//
// 编译: g++ -O3 -std=c++17 pid_l1_terms.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// 一阶 plant 受控仿真：离散 PID（位置式）控制 u，plant x' = -x + u
static vector<double> sim(double Kp, double Ki, double Kd, double dt, int N, double sp = 1.0){
    vector<double> y(N + 1, 0.0);
    double x = 0.0, integ = 0.0, prev_err = 0.0;
    const double UMAX = 10.0; // 限幅（本 demo 不触发，仅防发散）
    for (int k = 0; k < N; k++){
        double err = sp - x;
        integ += Ki * err * dt;
        double deriv = (err - prev_err) / dt;
        double u = Kp * err + integ + Kd * deriv;
        if (u >  UMAX) u =  UMAX;
        if (u < -UMAX) u = -UMAX;
        x = x + dt * (-x + u);          // plant 积分
        prev_err = err;
        y[k + 1] = x;
    }
    return y;
}

static double steadyErr(const vector<double>& y, double sp){ return sp - y.back(); }
static double overshoot(const vector<double>& y, double sp){
    double peak = *max_element(y.begin(), y.end());
    double final = y.back();
    if (final <= 1e-6) return 0;
    return max(0.0, (peak - final) / final);
}
static double settling(const vector<double>& y, double dt, double sp, double tol = 0.02){
    double lo = sp * (1 - tol), hi = sp * (1 + tol);
    int i = (int)y.size() - 1;
    for (; i >= 0; i--) if (y[i] < lo || y[i] > hi) break;
    return (i + 1) * dt;
}

int main(){
    bool pass = true;
    const double dt = 0.02, N = 500; // 10s
    cout << fixed << setprecision(4);

    auto yP  = sim(2.0, 0.0, 0.0, dt, N);   // 仅 P
    auto yPI = sim(2.0, 1.0, 0.0, dt, N);   // P + I
    auto yPID= sim(2.0, 1.0, 1.0, dt, N);   // P + I + D

    double eP = steadyErr(yP,1), ePI = steadyErr(yPI,1), ePID = steadyErr(yPID,1);
    double oP = overshoot(yP,1), oPI = overshoot(yPI,1), oPID = overshoot(yPID,1);
    double tP = settling(yP,dt,1), tPI = settling(yPI,dt,1), tPID = settling(yPID,dt,1);

    cout << "[L1.2 PID 三项]  plant G(s)=1/(s+1)\n";
    cout << "        P : 稳态误差=" << eP  << "  超调=" << oP*100 << "%  调节时间=" << tP  << "s\n";
    cout << "        PI: 稳态误差=" << ePI << "  超调=" << oPI*100 << "%  调节时间=" << tPI << "s\n";
    cout << "        PID:稳态误差=" << ePID << "  超调=" << oPID*100 << "%  调节时间=" << tPID << "s\n";

    // 预期验证
    if (!(eP > 0.15)){ pass = false; cout << "  [失败] P 项应有明显稳态误差\n"; }
    if (!(ePI < 0.03)){ pass = false; cout << "  [失败] I 项应消除稳态误差\n"; }
    if (!(ePID < 0.03)){ pass = false; cout << "  [失败] PID 也应消除稳态误差\n"; }
    if (!(oPID <= oPI + 0.02)){ pass = false; cout << "  [失败] D 项应不增大超调\n"; }
    if (tPID > tPI * 1.3 + 1e-6){ pass = false; cout << "  [失败] D 项不应显著拖慢调节\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] P留误差 / I消误差 / D抑超调 均符合预期" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
