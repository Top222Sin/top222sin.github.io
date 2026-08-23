// ============================================================
// KF_demo L2.5 非线性(一)：扩展卡尔曼滤波 EKF
// 纯 C++17。F=∂f/∂x、H=∂h/∂x 在当前估计点线性化（衔接 Jacobian_demo）。
// 验证：
//  ① 极坐标观测 h=[r,θ]=[√(x²+y²), atan2(y,x)] 的 H 与有限差分一致
//  ② 单摆 f=[θ+dt·ω, ω−dt·sinθ] 的 F 与有限差分一致
//  ③ 单摆 EKF 滤波 RMSE < 观测 RMSE
// 陷阱演示：线性化点离真值远时 EKF 可能发散（见 .md 讨论）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 66;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.5 EKF：线性化 + 解析雅可比 ===\n");

    // ---- ① 极坐标观测 H vs 有限差分 ----
    {
        double px = 1.5, py = -0.7;
        double r_ = hypot(px, py), r2 = r_ * r_;
        Mat H = {{px / r_,  py / r_},
                 {-py / r2, px / r2}};
        double h = 1e-6;
        Mat Hfd(2, Vec(2));
        double zp[2], zm[2];
        for (int j = 0; j < 2; ++j) {
            double d[2] = {0, 0}; d[j] = h;
            double xp = px + d[0], yp = py + d[1];
            double xm = px - d[0], ym = py - d[1];
            zp[0] = hypot(xp, yp); zp[1] = atan2(yp, xp);
            zm[0] = hypot(xm, ym); zm[1] = atan2(ym, xm);
            for (int i = 0; i < 2; ++i) Hfd[i][j] = (zp[i] - zm[i]) / (2 * h);
        }
        double err = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) err = fmax(err, fabs(H[i][j] - Hfd[i][j]));
        chk("极坐标 H vs 有限差分", err, 1e-6);
    }

    // ---- ② 单摆 F vs 有限差分 ----
    {
        double dt = 0.02;
        double th = 0.5, om = 0.3;
        Mat F = {{1.0, dt}, {-dt * cos(th), 1.0}};
        double h = 1e-6;
        Mat Ffd(2, Vec(2));
        for (int j = 0; j < 2; ++j) {
            double sp[2] = {th, om}, sm[2] = {th, om};
            sp[j] += h; sm[j] -= h;
            double fp[2] = {sp[0] + dt * sp[1], sp[1] - dt * sin(sp[0])};
            double fm[2] = {sm[0] + dt * sm[1], sm[1] - dt * sin(sm[0])};
            for (int i = 0; i < 2; ++i) Ffd[i][j] = (fp[i] - fm[i]) / (2 * h);
        }
        double err = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) err = fmax(err, fabs(F[i][j] - Ffd[i][j]));
        chk("单摆 F vs 有限差分", err, 1e-7);
    }

    // ---- ③ 单摆 EKF：滤波 RMSE < 观测 RMSE ----
    {
        double dt = 0.02, r_ = 0.02, qsc = 1e-4;
        int N = 250;
        double th_t = 0.8, om_t = 0.0;          // 真值
        double th = 0.0, om = 0.0;              // 滤波初值
        double P00 = 1.0, P01 = 0.0, P11 = 1.0;
        double q11 = qsc * dt*dt*dt*dt / 4, q12 = qsc * dt*dt*dt / 2, q22 = qsc * dt;
        double e2f = 0, e2z = 0;
        for (int k = 0; k < N; ++k) {
            // 真值（半隐式欧拉）
            th_t += dt * om_t;
            om_t -= dt * sin(th_t);
            double z = th_t + gauss() * sqrt(r_);
            // EKF 预测（在当前估计点线性化）：F=[[1,dt],[f10,1]]
            double thm = th + dt * om;
            double omm = om - dt * sin(th);
            double f10 = -dt * cos(th);   // ∂(om-dt·sinθ)/∂θ
            // P⁻ = F P Fᵀ + Q（精确展开）
            double P00m = P00 + 2 * dt * P01 + dt * dt * P11 + q11;
            double P01m = f10 * P00 + (1 + dt * f10) * P01 + dt * P11 + q12;
            double P11m = f10 * f10 * P00 + 2 * f10 * P01 + P11 + q22;
            // 更新：H=[[1,0]], R=r_
            double S = P00m + r_;
            double K0 = P00m / S, K1 = P01m / S;
            th = thm + K0 * (z - thm);
            om = omm + K1 * (z - thm);
            P00 = (1 - K0) * P00m;
            P01 = (1 - K0) * P01m;
            P11 = P11m - K1 * P01m;
            e2f += (th - th_t) * (th - th_t);
            e2z += (z - th_t) * (z - th_t);
        }
        double rmse_f = sqrt(e2f / N), rmse_z = sqrt(e2z / N);
        printf("  [demo] 单摆: rmse_ekf=%.4f  rmse_meas=%.4f\n", rmse_f, rmse_z);
        chk("单摆 EKF 滤波 < 观测", rmse_f < rmse_z ? 0.0 : 1.0, 0.5);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
