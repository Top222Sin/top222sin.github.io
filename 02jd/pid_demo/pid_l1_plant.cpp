// pid_l1_plant.cpp
// ============================================================================
// L1.1  被控对象建模与离散化
// ----------------------------------------------------------------------------
// 控制的第一步是"认识被控对象(plant)"。本 demo：
//   ① 给出连续域一阶/二阶 plant 的数学模型
//   ② 用三种离散化（前向/后向/Tustin）把它变成可在代码里积分的离散形式
//   ③ 单位阶跃激励，比较三种离散化与"真值(细步长欧拉)"的响应
// 全程合成数据（已知 K/T/ζ/ω），运行即出 PASS/FAIL + 阶跃响应指标。
//
// 编译: g++ -O3 -std=c++17 pid_l1_plant.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// ---- 连续 plant（用极细步长欧拉当作"真值"） ----
// 一阶惯性: G(s) = K / (T s + 1)
struct FirstOrder { double K = 1.0, T = 1.0; };

// 离散 plant 生成器（给定方法，返回 0..N 的阶跃响应序列，设定值=1）
static vector<double> simDiscrete(const FirstOrder& p, double dt, int N, const string& method){
    vector<double> y(N + 1, 0.0);
    double x = 0.0;
    for (int k = 0; k < N; k++){
        double u = 1.0; // 单位阶跃
        if (method == "forward"){
            // x[k+1] = x[k] + dt*(-x[k] + K*u)/T
            x = x + dt * (-x + p.K * u) / p.T;
        } else if (method == "backward"){
            // (x[k+1]-x[k])/dt = (-x[k+1] + K*u)/T  ->  x[k+1] = (x[k] + dt*K/T) / (1 + dt/T)
            x = (x + dt * p.K * u / p.T) / (1.0 + dt / p.T);
        } else { // tustin (bilinear)
            // x[k+1] = ((2T/dt - 1)*x[k] + 2K) / (2T/dt + 1),  u=1
            double a = 2.0 * p.T / dt;
            x = ((a - 1.0) * x + 2.0 * p.K) / (a + 1.0);
        }
        y[k + 1] = x;
    }
    return y;
}

// 连续"真值"：用 dt/20 的细欧拉近似
static vector<double> simTrue(const FirstOrder& p, double dt, int N){
    double fine = dt / 20.0;
    int M = N * 20;
    vector<double> yf(M + 1, 0.0);
    double x = 0.0;
    for (int k = 0; k < M; k++){
        x = x + fine * (-x + p.K * 1.0) / p.T;
        yf[k + 1] = x;
    }
    // 降采样回 N+1 点
    vector<double> y(N + 1, 0.0);
    for (int k = 0; k <= N; k++) y[k] = yf[k * 20];
    return y;
}

// 阶跃响应指标
static double steadyState(const vector<double>& y){ return y.back(); }
static double settlingTime(const vector<double>& y, double dt, double tol = 0.02){
    double ss = steadyState(y);
    double lo = ss * (1.0 - tol), hi = ss * (1.0 + tol);
    int i = (int)y.size() - 1;
    for (; i >= 0; i--) if (y[i] < lo || y[i] > hi) break;
    return (i + 1) * dt;
}

int main(){
    bool pass = true;
    const double dt = 0.05, T = 1.0, K = 1.0;
    const int N = 200; // 10s
    FirstOrder p{K, T};

    auto yT  = simTrue(p, dt, N);
    auto yF  = simDiscrete(p, dt, N, "forward");
    auto yB  = simDiscrete(p, dt, N, "backward");
    auto yTu = simDiscrete(p, dt, N, "tustin");

    cout << fixed << setprecision(4);
    cout << "[L1.1 被控对象] 一阶 G(s)=" << K << "/(" << T << "s+1), dt=" << dt << "s\n";
    cout << "  真值稳态=" << steadyState(yT)
         << "  前向稳态=" << steadyState(yF)
         << "  后向稳态=" << steadyState(yB)
         << "  Tustin稳态=" << steadyState(yTu) << "\n";

    double ss_err = 0;
    ss_err = max(ss_err, abs(steadyState(yF)  - K));
    ss_err = max(ss_err, abs(steadyState(yB)  - K));
    ss_err = max(ss_err, abs(steadyState(yTu) - K));
    cout << "  稳态误差(对 K=" << K << ")=" << ss_err << "\n";

    // 形态对比：三种离散化最大偏离真值
    double shape_err = 0;
    for (int i = 0; i <= N; i++)
        shape_err = max(shape_err, max({abs(yF[i]-yT[i]), abs(yB[i]-yT[i]), abs(yTu[i]-yT[i])}));
    cout << "  三种离散化相对真值的最大形态偏差=" << shape_err << "\n";

    cout << "  调节时间: 真值=" << settlingTime(yT,dt) << "s"
         << "  前向=" << settlingTime(yF,dt) << "s"
         << "  Tustin=" << settlingTime(yTu,dt) << "s\n";

    if (ss_err > 0.02){ pass = false; cout << "  [失败] 稳态误差过大\n"; }
    if (shape_err > 0.05){ pass = false; cout << "  [失败] 离散化形态偏离真值过大\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 离散化稳态正确、形态逼近真值" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
