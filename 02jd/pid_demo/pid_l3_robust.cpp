// pid_l3_robust.cpp
// ============================================================================
// L3.2  鲁棒与抗扰（Robustness & Disturbance Rejection）⭐
// ----------------------------------------------------------------------------
// 增益再好，未知扰动照样把系统拖偏。本 demo 比较三种抗扰策略（同一 plant，阶跃
// 扰动 d 在 t1 注入）：
//   ① PID 仅（只靠积分慢慢磨回）→ 大瞬态偏差、恢复慢
//   ② PID + 扰动前馈(完美测量 d) → 扰动被抵消，几乎无偏差
//   ③ PID + 扰动观测器 DOB（低通估计 d 并抵消）→ 偏差明显减小
// 真值对拍：扰动真值 d0 已知；DOB 估计 d_hat 应收敛到 d0（容差 10%）。
// 对应工程：前馈/DOB 是鲁棒控制直觉，PX4 用加速度前馈/扰动估计增强抗风扰。
//
// 编译: g++ -O3 -std=c++17 pid_l3_robust.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Cfg { double Kp, Ki, Kd; bool useFF; bool useDOB; double tau; };

// plant: x' = -x + u + d ；返回序列，并回写 d_hat 终值
static vector<double> sim(const Cfg& c, double dt, int N, double sp,
                          double d0, double t1, double Umax, double& dHatFinal){
    vector<double> y(N + 1, 0.0);
    double x = 0, integ = 0, prevErr = 0, xprev = 0, dHat = 0;
    for (int k = 0; k < N; k++){
        double tn = k * dt;
        double dnow = (tn >= t1) ? d0 : 0.0;
        double e = sp - x;
        integ += c.Ki * e * dt;
        double deriv = (e - prevErr) / dt;
        double uPid = c.Kp * e + integ + c.Kd * deriv;
        double uff = 0;
        if      (c.useFF)   uff = -dnow;          // 完美测量前馈
        else if (c.useDOB)  uff = -dHat;          // 观测器前馈
        double u = uPid + uff;
        if (u >  Umax) { u =  Umax; integ -= c.Ki * e * dt; }
        if (u < -Umax) { u = -Umax; integ += c.Ki * e * dt; }
        xprev = x;
        x = x + dt * (-x + u + dnow);
        // DOB：d_hat = LPF( x' + x - u_total )
        if (c.useDOB){
            double xp = (x - xprev) / dt;
            double proxy = xp + x - u;
            double a = dt / (c.tau + dt);
            dHat = dHat + a * (proxy - dHat);
        }
        prevErr = e;
        y[k + 1] = x;
    }
    dHatFinal = dHat;
    return y;
}

// t1 之后的峰值偏差
static double peakDev(const vector<double>& y, double dt, double sp, double t1){
    int k0 = (int)(t1 / dt);
    double pk = 0;
    for (int k = k0; k < (int)y.size(); k++) pk = max(pk, abs(y[k] - sp));
    return pk;
}
static double steadyErr(const vector<double>& y, double sp){ return sp - y.back(); }

int main(){
    bool pass = true;
    const double dt = 0.02, N = 1000;     // 20s
    const double sp = 1.0, d0 = 0.5, t1 = 5.0, Umax = 10.0;
    cout << fixed << setprecision(4);

    Cfg base{ 2.0, 1.0, 0.5, false, false, 0.3 };

    double dHatFF = 0, dHatDOB = 0;
    auto yPID = sim(base,                          dt, N, sp, d0, t1, Umax, dHatFF);
    Cfg ff  = base; ff.useFF   = true;
    auto yFF  = sim(ff,                         dt, N, sp, d0, t1, Umax, dHatFF);
    Cfg dob = base; dob.useDOB = true;
    auto yDOB = sim(dob,                        dt, N, sp, d0, t1, Umax, dHatDOB);

    double pkPID = peakDev(yPID, dt, sp, t1);
    double pkFF  = peakDev(yFF,  dt, sp, t1);
    double pkDOB = peakDev(yDOB, dt, sp, t1);
    cout << "[L3.2 鲁棒抗扰] plant x'=-x+u+d, 扰动 d0=" << d0 << " @t=" << t1 << "s\n";
    cout << "  峰值偏差: PID仅=" << pkPID << "  PID+FF=" << pkFF << "  PID+DOB=" << pkDOB << "\n";
    cout << "  DOB 估计 d_hat=" << dHatDOB << " (真值 d0=" << d0 << ")\n";
    cout << "  稳态误差: PID=" << steadyErr(yPID,sp) << " FF=" << steadyErr(yFF,sp)
         << " DOB=" << steadyErr(yDOB,sp) << "\n";

    // 对拍判据
    if (!(pkFF  < 0.2 * pkPID)){ pass = false; cout << "  [失败] 完美前馈未显著减小偏差\n"; }
    if (!(pkDOB < 0.6 * pkPID)){ pass = false; cout << "  [失败] DOB 未显著减小偏差\n"; }
    if (!(abs(steadyErr(yPID, sp)) < 0.02)){ pass = false; cout << "  [失败] PID 终值未消除偏差\n"; }
    if (!(abs(steadyErr(yFF,  sp)) < 0.02)){ pass = false; cout << "  [失败] FF 终值偏差\n"; }
    if (!(abs(steadyErr(yDOB, sp)) < 0.02)){ pass = false; cout << "  [失败] DOB 终值偏差\n"; }
    if (!(abs(dHatDOB - d0) < 0.1 * d0)){ pass = false; cout << "  [失败] DOB 未收敛到真值扰动\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 扰动前馈/DOB 显著抑制瞬态偏差，DOB 估计收敛到真值扰动"
                  : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
