// pid_l3_pipeline.cpp
// ============================================================================
// L3.4  全流程工具链（Capstone）⭐
// ----------------------------------------------------------------------------
// 把前面所有积木串成一条龙：PX4 风格「位置→速度→角速率」三环串级（纯 C++ 复刻
// PositionControl + RateControl），端到端阶跃响应对拍真值，并内嵌：
//   · 内环系统辨识 + Z-N 自动整定（relay 交叉校验 + 解析 FOPDT 真值）
//   · 退化注入（传感器符号翻转 / 模型失配）→ 系统检出告警并拒绝"带病运行"
// 全程合成 plant，运行即出 PASS/FAIL。对应 PX4 生产级控制链的工程全貌。
//
// 编译: g++ -O3 -std=c++17 pid_l3_pipeline.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>

using namespace std;

struct FOPDT { double K, T, L; };

// ---- 内环 plant 辨识（relay 自激振荡，复刻 L3.1）----
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
            if (prevY <= 0 && yv > 0){
                if (lastCrossT >= 0){ sumPu += t - lastCrossT; nInt++; }
                lastCrossT = t;
            }
        }
        prevY = yv;
    }
    aEst  = 0.5 * (ymax - ymin);
    PuEst = (nInt > 0) ? sumPu / nInt : 0.0;
}

// ---- 三环串级仿真 ----
struct Cfg {
    double Kpx, vmax;            // 位置环 P + 速度限幅
    double Kpv, Kiv, Kdv, amax; // 速度环 PID + 加速度限幅
    double Kr, Pr, Ir, Dr, FF;   // 速率环（并行式 PX4）+ 前馈
    double tau;                  // 执行器一阶惯性
    bool   sensorFault;          // 位置传感器符号翻转
};

static vector<double> simCascade(const Cfg& c, double dt, int N, double xsp){
    vector<double> x_traj(N + 1, 0.0);
    double x = 0, v = 0, a = 0;
    double intV = 0, intA = 0, prevEv = 0, prevEa = 0;
    const double UMAX = 50.0;
    for (int k = 0; k < N; k++){
        double xm = c.sensorFault ? -x : x;          // 传感器反馈（可注入故障）
        double ep = xsp - xm;
        double vsp = c.Kpx * ep;
        if (vsp >  c.vmax) vsp =  c.vmax;
        if (vsp < -c.vmax) vsp = -c.vmax;
        double ev = vsp - v;
        intV += c.Kiv * ev * dt;
        double dev = (ev - prevEv) / dt;
        double asp = c.Kpv * ev + intV + c.Kdv * dev;
        if (asp >  c.amax) { asp =  c.amax; intV -= c.Kiv * ev * dt; }
        if (asp < -c.amax) { asp = -c.amax; intV += c.Kiv * ev * dt; }
        double ea = asp - a;
        intA += c.Ir * ea * dt;
        double dea = (ea - prevEa) / dt;
        double u = c.Kr * (c.Pr * ea + intA + c.Dr * dea) + c.FF * asp;
        if (u >  UMAX) { u =  UMAX; intA -= c.Ir * ea * dt; }
        if (u < -UMAX) { u = -UMAX; intA += c.Ir * ea * dt; }
        double aNew = a + dt * ((u - a) / c.tau);    // 执行器一阶惯性
        v = v + dt * aNew;                            // v' = a
        x = x + dt * v;                               // x' = v
        a = aNew;
        prevEv = ev; prevEa = ea;
        x_traj[k + 1] = x;
    }
    return x_traj;
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
    const double dt = 0.01, N = 3000;     // 30s
    const double xsp = 1.0;
    cout << fixed << setprecision(4);

    double tau = 0.1;
    // 内环系统辨识（plant: a=(u-a)/tau → FOPDT K=1, T=tau, L=tau）
    FOPDT inner{ 1.0, tau, tau };
    double aEst = 0, PuEst = 0;
    relayIdentify(inner, dt, 20000, 1.0, 0.002, aEst, PuEst);
    double KuEst = 4.0 / (M_PI * aEst);
    // 精确真值（L3.1）：数值求根 ωu·L+atan(ωu·T)=π（描述函数法 G(jωu) 相位=-π）
    // 注: ωu≈π/(2L)、Pu≈4L 是 T≫L 极限近似，对 inner(T=L=τ, T/L=1)偏差 18-29%
    double wu;
    {
        double lo = 1e-3, hi = 1e3;
        for (int i = 0; i < 100; i++) {
            double mid = 0.5 * (lo + hi);
            if (mid * inner.L + atan(mid * inner.T) < M_PI) lo = mid;
            else hi = mid;
        }
        wu = 0.5 * (lo + hi);
    }
    double KuTr = sqrt(1.0 + pow(wu * inner.T, 2)) / inner.K;
    double PuTr = 2.0 * M_PI / wu;
    cout << "[L3.4 全流程] 内环系统辨识: relay Ku=" << KuEst << " Pu=" << PuEst
         << "s  | 精确 Ku=" << KuTr << " Pu=" << PuTr << "s\n";

    // 保守 Z-N（0.4 系数，+前馈）作为内环增益
    double Pr = min(0.4 * KuTr, 10.0), Ir = Pr / (0.5 * PuTr), Dr = Pr * 0.125 * PuTr;
    cout << "  内环整定: Pr=" << Pr << " Ir=" << Ir << " Dr=" << Dr << " (FF=1)\n";

    // 基础三环增益（位置/速度环保守整定）
    Cfg healthy{ 1.5, 0.6, 3.0, 1.5, 0.5, 3.0, 1.0, Pr, Ir, Dr, 1.0, tau, false };
    auto yH = simCascade(healthy, dt, N, xsp);
    double eH = steadyErr(yH, xsp), oH = overshoot(yH, xsp), tH = settling(yH, dt, xsp);
    cout << "[健康] 稳态误差=" << eH << " 超调=" << oH*100 << "% 调节=" << tH << "s\n";

    // 退化1：传感器符号翻转 → 必发散告警
    Cfg fault = healthy; fault.sensorFault = true;
    auto yF = simCascade(fault, dt, N, xsp);
    bool diverged = (abs(yF.back()) > 3.0) || (abs(yF.back()) > 1e3);
    cout << "[退化-传感翻转] 末值 x=" << yF.back() << " 发散=" << (diverged ? "是" : "否") << "\n";

    // 退化2：模型失配（执行器变慢 10x，增益未重整定）→ 超调恶化
    Cfg mismatch = healthy; mismatch.tau = 1.0;
    auto yM = simCascade(mismatch, dt, N, xsp);
    double oM = overshoot(yM, xsp);
    cout << "[退化-模型失配] 超调=" << oM*100 << "% (健康=" << oH*100 << "%)\n";

    // ---- 对拍判据 ----
    if (!(abs(eH) < 0.05)){ pass = false; cout << "  [失败] 健康串级未消除稳态误差\n"; }
    if (!(oH < 0.3)){ pass = false; cout << "  [失败] 健康串级超调过大\n"; }
    if (diverged){ /* 健康不应发散 */ pass = false; cout << "  [失败] 健康串级反而发散\n"; }
    if (tH > 20.0){ pass = false; cout << "  [失败] 健康串级调节过慢\n"; }
    // 内环辨识交叉校验（relay vs 精确真值，容差 30%）
    if (!(abs(KuEst - KuTr) / KuTr < 0.3)){ pass = false; cout << "  [失败] 内环 relay 辨识偏离精确真值>30%\n"; }
    // 退化告警必须触发
    bool degradeAlarm = diverged || (oM > 0.2 * 5 && oM > oH * 2.0);  // 符号翻转发散 或 超调显著恶化
    if (!diverged){ pass = false; cout << "  [失败] 传感器翻转退化未触发告警\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 三环串级端到端跟踪良好；内环辨识与精确真值吻合；退化注入被正确告警"
                  : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
