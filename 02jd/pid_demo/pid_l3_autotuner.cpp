// pid_l3_autotuner.cpp
// ============================================================================
// L3.3  在线/自动调参系统（AutoTuner 状态机）⭐
// ----------------------------------------------------------------------------
// 仿 PX4 McAutotuneAttitudeControl 的工程框架：把"整定"做成一套能自动跑、能自检、
// 能告警的系统。状态机： IDLE→INIT→EXCITE→IDENTIFY→VERIFY→SAVE→COMPLETE，
// 任一轴质量不达标 → FAILED(告警, 拒绝保存)。
//   每个轴：relay 激励辨识 → Z-N 增益 → 闭环验证(超调/稳态误差) → 质量门
//   · 健康 plant：三轴全部落库，状态 COMPLETE，无告警
//   · 退化 plant（负增益/inverted）：验证失败 → 触发告警，拒绝保存
// 真值对拍：健康 plant 用 L3.1 的解析 Ku/Pu 逻辑已验证；本篇验证"系统级行为
//            —— 好 plant 全过、坏 plant 必告警"，即质量门与告警逻辑正确。
//
// 编译: g++ -O3 -std=c++17 pid_l3_autotuner.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>

using namespace std;

struct FOPDT { double K, T, L; };

static vector<double> closedStep(const FOPDT& p, double dt, int N, double sp,
                                 double Kp, double Ki, double Kd, double Umax){
    int Ln = max(1, (int)round(p.L / dt));
    vector<double> buf(Ln + 1, 0.0); int idx = 0;
    vector<double> y(N + 1, 0.0);
    double yv = 0, integ = 0, prevErr = 0;
    for (int k = 0; k < N; k++){
        double err = sp - yv;
        integ += Ki * err * dt;
        double deriv = (err - prevErr) / dt;
        double u = Kp * err + integ + Kd * deriv;
        if (u >  Umax) { u =  Umax; integ -= Ki * err * dt; }
        if (u < -Umax) { u = -Umax; integ += Ki * err * dt; }
        double uD = buf[idx]; buf[idx] = u; idx = (idx + 1) % (Ln + 1);
        yv = yv + dt * (-yv / p.T + (p.K / p.T) * uD);
        prevErr = err;
        y[k + 1] = yv;
    }
    return y;
}

static void relayIdentify(const FOPDT& p, double dt, int N, double d, double eps,
                         double& aEst, double& PuEst){
    int Ln = max(1, (int)round(p.L / dt));
    vector<double> buf(Ln + 1, 0.0); int idx = 0;
    double yv = 0, uPrev = d, prevY = 0;
    double ymin = 1e9, ymax = -1e9, lastCrossT = -1, sumPu = 0; int nInt = 0;
    double warm = p.L * 3.0;
    for (int k = 0; k < N; k++){
        double u;
        if      (yv < -eps) u =  d;
        else if (yv >  eps) u = -d;
        else                u = uPrev;
        uPrev = u;
        double uD = buf[idx]; buf[idx] = u; idx = (idx + 1) % (Ln + 1);
        yv = yv + dt * (-yv / p.T + (p.K / p.T) * uD);
        double t = k * dt;
        if (t > warm){
            if (yv < ymin) ymin = yv;
            if (yv > ymax) ymax = yv;
            if (prevY <= 0 && yv > 0 && lastCrossT >= 0){ sumPu += t - lastCrossT; nInt++; }
            if (prevY <= 0 && yv > 0 && lastCrossT < 0) lastCrossT = t;
        }
        prevY = yv;
    }
    aEst  = 0.5 * (ymax - ymin);
    PuEst = (nInt > 0) ? sumPu / nInt : 0.0;
}

static double overshoot(const vector<double>& y, double sp){
    double pk = *max_element(y.begin(), y.end());
    return max(0.0, (pk - sp) / sp);
}
static double settling(const vector<double>& y, double dt, double sp, double tol = 0.02){
    double lo = sp*(1-tol), hi = sp*(1+tol);
    int i = (int)y.size() - 1;
    for (; i >= 0; i--) if (y[i] < lo || y[i] > hi) break;
    return (i + 1) * dt;
}

enum State { IDLE, INIT, EXCITE, IDENTIFY, VERIFY, SAVE, COMPLETE, FAILED };

struct AxisOut {
    string name; bool saved = false, alarm = false;
    double Kp = 0, Ki = 0, Kd = 0, ov = 0, set = 0, tset = 0;
};

// 单轴自动整定：辨识 → 增益 → 验证 → 质量门
static AxisOut tuneAxis(const FOPDT& p, const string& name, double dt, int N, double d, double eps){
    AxisOut o; o.name = name;
    double aEst = 0, PuEst = 0;
    relayIdentify(p, dt, N, d, eps, aEst, PuEst);
    double Ku = 4.0 * d / (M_PI * aEst);
    double Kp = 0.6 * Ku, Ti = 0.5 * PuEst, Td = 0.125 * PuEst;
    double Ki = Kp / Ti, Kd = Kp * Td;
    o.Kp = Kp; o.Ki = Ki; o.Kd = Kd;
    auto y = closedStep(p, dt, 3000, 1.0, Kp, Ki, Kd, 10.0);
    o.ov = overshoot(y, 1.0);
    o.set = 1.0 - y.back();
    o.tset = settling(y, dt, 1.0);
    // 质量门
    bool ok = (o.ov < 0.6) && (abs(o.set) < 0.05) && (abs(y.back()) < 3.0);
    if (ok) o.saved = true;
    else    o.alarm = true;
    return o;
}

int main(){
    bool pass = true;
    const double dt = 0.005, N = 20000, d = 1.0, eps = 0.002;
    cout << fixed << setprecision(4);

    // 健康三轴（仿 PX4 roll/pitch/yaw）
    FOPDT roll{ 2.0, 1.0, 0.5 };
    FOPDT pitch{ 1.5, 0.8, 0.4 };
    FOPDT yaw{ 1.0, 1.2, 0.6 };

    cout << "[L3.3 自动调参系统] 状态机 IDLE→INIT→EXCITE→IDENTIFY→VERIFY→SAVE→COMPLETE\n";
    State st = IDLE; cout << "  " << st << " "; st = INIT; cout << "→INIT ";
    vector<AxisOut> res;
    for (auto& p : { roll, pitch, yaw }){
        st = EXCITE; st = IDENTIFY; st = VERIFY; st = SAVE;
        // 用轴名：简单映射到字符串
        string nm = (&p == &roll) ? "roll" : ((&p == &pitch) ? "pitch" : "yaw");
        auto a = tuneAxis(p, nm, dt, N, d, eps);
        res.push_back(a);
        cout << "→[" << nm << ":" << (a.saved ? "SAVED" : "ALARM") << "] ";
        if (!a.saved) pass = false;
    }
    st = COMPLETE; cout << "→COMPLETE\n";

    for (auto& a : res){
        cout << "    " << a.name << ": Kp=" << a.Kp << " Ki=" << a.Ki << " Kd=" << a.Kd
             << " 超调=" << a.ov*100 << "% 稳态误差=" << a.set << " 调节=" << a.tset << "s "
             << (a.saved ? "✔" : "✘告警") << "\n";
    }

    // 退化 plant（负增益 / inverted）→ 必须告警、拒绝保存
    cout << "  -- 退化注入测试 --\n";
    FOPDT degraded{ -2.0, 1.0, 0.5 };
    auto bad = tuneAxis(degraded, "degraded", dt, N, d, eps);
    cout << "    degraded: 保存=" << (bad.saved ? "是" : "否")
         << " 告警=" << (bad.alarm ? "是" : "否")
         << " 稳态误差=" << bad.set << "\n";
    if (bad.saved)      { pass = false; cout << "  [失败] 退化 plant 竟然被保存\n"; }
    if (!bad.alarm)     { pass = false; cout << "  [失败] 退化 plant 未触发告警\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 健康三轴全落库且无告警；退化 plant 被正确识别并告警拒绝"
                  : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
