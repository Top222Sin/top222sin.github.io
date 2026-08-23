// pid_l3_autotune.cpp
// ============================================================================
// L3.1  自动整定（系统辨识 + relay / Åström-Hägglund）⭐
// ----------------------------------------------------------------------------
// 不靠手调——让控制器自己"测"出好增益。方法：把 plant 包在一个中继(relay)
// 反馈环里，它自己会起振；量出振荡幅值 a 与周期 Pu，由 Åström-Hägglund 公式
//    Ku = 4·d / (π·a)
// 反推临界增益，再套 Ziegler-Nichols 关断反应式 PID：
//    Kp = 0.6·Ku ,  Ti = 0.5·Pu ,  Td = 0.125·Pu
// 真值对拍：对 FOPDT  plant K·e^{-Ls}/(Ts+1)，精确相位条件 ωu·L+atan(ωu·T)=π
//    (描述函数法 G(jωu) 相位=-π)。数值求根得 ωu，进而 Ku_true=√(1+(ωu·T)²)/K、
//    Pu_true=2π/ωu。注: 工程口诀 ωu≈π/(2L)、Pu≈4L 是 T≫L 极限近似，对本 demo
//    plant (T/L=2) 偏差 12-17%，故用精确求根作真值。
// 对应 PX4 mc_autotune：阶跃激励做 _sys_id 系统辨识 → pid_design::computePidGmvc。
//
// 编译: g++ -O3 -std=c++17 pid_l3_autotune.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct FOPDT { double K, T, L; };

// 闭环步进仿真（带纯滞后 L）：位置式并行 PID + 输出限幅
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
        if (u >  Umax) { u =  Umax; integ -= Ki * err * dt; }   // 限幅回撤积分(抗windup)
        if (u < -Umax) { u = -Umax; integ += Ki * err * dt; }
        double uD = buf[idx]; buf[idx] = u; idx = (idx + 1) % (Ln + 1);
        yv = yv + dt * (-yv / p.T + (p.K / p.T) * uD);
        prevErr = err;
        y[k + 1] = yv;
    }
    return y;
}

// 中继自激振荡辨识：返回输出幅值 a 与周期 Pu
static void relayIdentify(const FOPDT& p, double dt, int N, double d, double eps,
                         double& aEst, double& PuEst){
    int Ln = max(1, (int)round(p.L / dt));
    vector<double> buf(Ln + 1, 0.0); int idx = 0;
    double yv = 0, uPrev = d, prevY = 0;
    double ymin = 1e9, ymax = -1e9;
    double lastCrossT = -1, sumPu = 0; int nInt = 0;
    double warm = p.L * 3.0;
    for (int k = 0; k < N; k++){
        // 理想中继(带小滞环)关于 r=0：e=-y
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
            if (prevY <= 0 && yv > 0){            // 向上过零
                if (lastCrossT >= 0){ sumPu += t - lastCrossT; nInt++; }
                lastCrossT = t;
            }
        }
        prevY = yv;
    }
    aEst  = 0.5 * (ymax - ymin);
    PuEst = (nInt > 0) ? sumPu / nInt : 0.0;
}

static double steadyErr(const vector<double>& y, double sp){ return sp - y.back(); }
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

int main(){
    bool pass = true;
    const double dt = 0.005, N = 20000;
    FOPDT p{ 2.0, 1.0, 0.5 };          // K=2, T=1s, L=0.5s
    const double d = 1.0, eps = 0.002; // 中继幅值 / 滞环
    cout << fixed << setprecision(4);

    // ---- relay 辨识 ----
    double aEst = 0, PuEst = 0;
    relayIdentify(p, dt, N, d, eps, aEst, PuEst);
    double KuEst = 4.0 * d / (M_PI * aEst);

    // ---- 精确真值（FOPDT 临界点，数值求根 ωu·L+atan(ωu·T)=π）----
    // 描述函数法: G(jωu) 相位=-π → ωu·L+atan(ωu·T)=π。单调递增收敛。
    double wu;
    {
        double lo = 1e-3, hi = 1e3;
        for (int i = 0; i < 100; i++) {
            double mid = 0.5 * (lo + hi);
            if (mid * p.L + atan(mid * p.T) < M_PI) lo = mid;
            else hi = mid;
        }
        wu = 0.5 * (lo + hi);
    }
    double KuTr = sqrt(1.0 + pow(wu * p.T, 2)) / p.K;
    double PuTr = 2.0 * M_PI / wu;

    cout << "[L3.1 自动整定] plant K=" << p.K << " T=" << p.T << " L=" << p.L << "\n";
    cout << "  relay: a=" << aEst << " Pu=" << PuEst << "s  =>  Ku=" << KuEst << "\n";
    cout << "  精确真值: Pu=" << PuTr << "s  Ku=" << KuTr << "\n";

    // ---- Z-N 关断反应式 PID ----
    double Kp = 0.6 * KuEst, Ti = 0.5 * PuEst, Td = 0.125 * PuEst;
    double Ki = Kp / Ti, Kd = Kp * Td;
    cout << "  Z-N PID: Kp=" << Kp << " Ki=" << Ki << " Kd=" << Kd << "\n";

    auto y = closedStep(p, dt, 3000, 1.0, Kp, Ki, Kd, 10.0);  // 30s 阶跃
    double e = steadyErr(y, 1.0), o = overshoot(y, 1.0), tset = settling(y, dt, 1.0);
    cout << "  闭环阶跃: 稳态误差=" << e << " 超调=" << o*100 << "% 调节=" << tset << "s\n";

    // ---- 对拍判据 ----
    double relKu = abs(KuEst - KuTr) / KuTr;
    double relPu = abs(PuEst - PuTr) / PuTr;
    if (!(relKu < 0.15)){ pass = false; cout << "  [失败] relay 估 Ku 偏离真值 >15%\n"; }
    if (!(relPu < 0.15)){ pass = false; cout << "  [失败] relay 估 Pu 偏离真值 >15%\n"; }
    if (!(e < 0.05)){ pass = false; cout << "  [失败] 整定后稳态误差未消除\n"; }
    if (!(abs(y.back()) < 3.0)){ pass = false; cout << "  [失败] 闭环发散\n"; }
    if (!(o < 0.6)){ pass = false; cout << "  [失败] 超调过大(>60%)\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] relay 自激振荡辨识 Ku/Pu 与精确真值吻合，整定后闭环稳定且无稳态误差"
                  : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
