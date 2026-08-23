// ============================================================
// KF_demo L2.6 非线性(二)：无迹卡尔曼滤波 UKF ⭐
// 纯 C++17。sigma 点确定性采样逼近均值/协方差（无需雅可比）。
//   λ = α²(n+κ)−n；X₀=x, Xᵢ=x±√(n+λ)·colᵢ(√P)
//   权：Wm0=λ/(n+λ), Wc0=Wm0+(1−α²+β), Wi=1/(2(n+λ))
//   更新阶段从 N(x⁻,P⁻) 重新采点（保证线性时 UKF≡KF，见 .md 讨论）
// 验证：
//  ① 线性模型下 UKF == KF（解析等价，~1e-11）
//  ② 经典案例：极坐标→笛卡尔矩转换，UKF 比 EKF 线性化
//     更接近 40 万样本蒙特卡洛真值（均值+协方差）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Vec = vector<double>;
using Mat = vector<Vec>;

static unsigned long long _seed = 88;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static Mat eye(int n) { Mat I(n, Vec(n, 0.0)); for (int i = 0; i < n; ++i) I[i][i] = 1.0; return I; }
static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, Vec(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}
static Mat transpose(const Mat& A) {
    int n = A.size(), m = A[0].size(); Mat T(m, Vec(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) T[j][i] = A[i][j];
    return T;
}
static Mat addm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] += B[i][j];
    return C;
}
static Mat subm(const Mat& A, const Mat& B) {
    Mat C(A); for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < A[0].size(); ++j) C[i][j] -= B[i][j];
    return C;
}
static Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}
static Mat chol2(const Mat& A) {   // 2×2 下三角 Cholesky
    double l00 = sqrt(fmax(A[0][0], 0.0));
    double l10 = A[1][0] / (l00 > 1e-12 ? l00 : 1e-12);
    double l11 = sqrt(fmax(A[1][1] - l10 * l10, 0.0));
    return {{l00, 0.0}, {l10, l11}};
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-34s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.6 UKF：sigma 点 vs 线性化 ===\n");

    const int n = 2;
    const double alpha = 1.0, beta = 2.0, kappa = 0.0;
    const double lam = alpha * alpha * (n + kappa) - n;
    const double Wm0 = lam / (n + lam);
    const double Wc0 = Wm0 + (1 - alpha * alpha + beta);
    const double Wi = 1.0 / (2 * (n + lam));

    auto sigma_pts = [](const Vec& m, const Mat& Pv) {
        Mat S = chol2(Pv);
        vector<Vec> pts;
        pts.push_back(m);
        for (int j = 0; j < 2; ++j) {
            Vec c = {S[0][j], S[1][j]};
            pts.push_back({m[0] + sqrt(2.0) * c[0], m[1] + sqrt(2.0) * c[1]});
            pts.push_back({m[0] - sqrt(2.0) * c[0], m[1] - sqrt(2.0) * c[1]});
        }
        return pts;
    };

    // ---- ① 线性模型：UKF == KF ----
    {
        double dt = 0.1, q = 0.05, r = 0.5;
        int N = 30;
        Mat F = {{1.0, dt}, {0.0, 1.0}};
        Mat Q = {{q * dt*dt*dt*dt / 4.0, q * dt*dt*dt / 2.0},
                 {q * dt*dt*dt / 2.0,    q * dt}};
        Vec xt = {0.0, 1.0};
        Vec xk = {0.0, 0.5}; Mat P = eye(2); P[0][0] = P[1][1] = 10.0;
        Vec xu = xk; Mat Pu = P;
        double ex = 0, eP = 0;
        for (int k = 0; k < N; ++k) {
            double a = gauss() * sqrt(q);
            xt = {xt[0] + dt * xt[1] + 0.5 * dt * dt * a, xt[1] + dt * a};
            double z = xt[0] + gauss() * sqrt(r);
            // KF
            xk = matvec(F, xk);
            P = addm(matmul(matmul(F, P), transpose(F)), Q);
            double S1 = P[0][0] + r;
            double k0 = P[0][0] / S1, k1 = P[1][0] / S1;
            double e = z - xk[0];
            xk = {xk[0] + k0 * e, xk[1] + k1 * e};
            Mat IK = subm(eye(2), {{k0, 0.0}, {k1, 0.0}});
            Mat KRKt = {{k0 * r * k0, k0 * r * k1}, {k1 * r * k0, k1 * r * k1}};
            P = addm(matmul(matmul(IK, P), transpose(IK)), KRKt);
            // UKF（预测 + 更新各采一组 sigma 点）
            auto pts = sigma_pts(xu, Pu);
            Vec xm = {0, 0};
            for (int i = 0; i < 5; ++i) {
                double w = (i == 0) ? Wm0 : Wi;
                Vec pr = matvec(F, pts[i]);
                xm[0] += w * pr[0]; xm[1] += w * pr[1];
                pts[i] = pr;   // 原地保存传播后
            }
            Mat Pm(2, Vec(2, 0.0));
            for (int i = 0; i < 5; ++i) {
                double w = (i == 0) ? Wc0 : Wi;
                double d0 = pts[i][0] - xm[0], d1 = pts[i][1] - xm[1];
                Pm[0][0] += w * d0 * d0; Pm[0][1] += w * d0 * d1;
                Pm[1][0] += w * d1 * d0; Pm[1][1] += w * d1 * d1;
            }
            Pm = addm(Pm, Q);
            // 更新：从 N(xm, Pm) 重新采点，h(p)=[p0]
            auto pts2 = sigma_pts(xm, Pm);
            double zm = 0;
            for (int i = 0; i < 5; ++i) { double w = (i == 0) ? Wm0 : Wi; zm += w * pts2[i][0]; }
            double Pzz = 0, Px0 = 0, Px1 = 0;
            for (int i = 0; i < 5; ++i) {
                double w = (i == 0) ? Wc0 : Wi;
                double dz = pts2[i][0] - zm;
                Pzz += w * dz * dz;
                Px0 += w * (pts2[i][0] - xm[0]) * dz;
                Px1 += w * (pts2[i][1] - xm[1]) * dz;
            }
            Pzz += r;
            double Ku0 = Px0 / Pzz, Ku1 = Px1 / Pzz;
            xu = {xm[0] + Ku0 * (z - zm), xm[1] + Ku1 * (z - zm)};
            Pu = {{Pm[0][0] - Ku0 * Pzz * Ku0, Pm[0][1] - Ku0 * Pzz * Ku1},
                  {Pm[1][0] - Ku1 * Pzz * Ku0, Pm[1][1] - Ku1 * Pzz * Ku1}};
            ex = fmax(ex, fmax(fabs(xu[0] - xk[0]), fabs(xu[1] - xk[1])));
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) eP = fmax(eP, fabs(Pu[i][j] - P[i][j]));
        }
        chk("线性模型 UKF=KF 状态", ex, 1e-8);
        chk("线性模型 UKF=KF 协方差", eP, 1e-8);
    }

    // ---- ② 经典案例：极坐标→笛卡尔矩转换 ----
    {
        // (r,θ) ~ N([2, π/2], diag(0.05², 0.3²)) → x=r cosθ, y=r sinθ
        Vec m_pol = {2.0, 3.14159265358979323846 / 2};
        Mat P_pol = {{0.0025, 0.0}, {0.0, 0.09}};
        // 40 万样本蒙特卡洛真值
        _seed = 777;
        int NM = 400000;
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (int i = 0; i < NM; ++i) {
            double rp = m_pol[0] + gauss() * 0.05;
            double tp = m_pol[1] + gauss() * 0.3;
            double xv = rp * cos(tp), yv = rp * sin(tp);
            sx += xv; sy += yv; sxx += xv * xv; sxy += xv * yv; syy += yv * yv;
        }
        double mx = sx / NM, my = sy / NM;
        double C00 = sxx / NM - mx * mx, C01 = sxy / NM - mx * my, C11 = syy / NM - my * my;
        // EKF：一阶线性化
        double r0 = m_pol[0], t0 = m_pol[1];
        double J[2][2] = {{cos(t0), -r0 * sin(t0)}, {sin(t0), r0 * cos(t0)}};
        double mu_ekf[2] = {r0 * cos(t0), r0 * sin(t0)};
        double C_ekf[2][2];
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            C_ekf[i][j] = J[i][0] * P_pol[0][0] * J[j][0]
                        + J[i][0] * P_pol[0][1] * J[j][1]
                        + J[i][1] * P_pol[1][0] * J[j][0]
                        + J[i][1] * P_pol[1][1] * J[j][1];
        }
        // UKF：无迹变换
        auto pts = sigma_pts(m_pol, P_pol);
        double wts[5] = {Wm0, Wi, Wi, Wi, Wi};
        double mu_ukf[2] = {0, 0};
        for (int i = 0; i < 5; ++i) {
            double xv = pts[i][0] * cos(pts[i][1]), yv = pts[i][0] * sin(pts[i][1]);
            mu_ukf[0] += wts[i] * xv; mu_ukf[1] += wts[i] * yv;
            pts[i] = {xv, yv};
        }
        double C_ukf[2][2] = {{0, 0}, {0, 0}};
        for (int i = 0; i < 5; ++i) {
            double w = (i == 0) ? Wc0 : Wi;
            double d0 = pts[i][0] - mu_ukf[0], d1 = pts[i][1] - mu_ukf[1];
            C_ukf[0][0] += w * d0 * d0; C_ukf[0][1] += w * d0 * d1;
            C_ukf[1][0] += w * d1 * d0; C_ukf[1][1] += w * d1 * d1;
        }
        double e_m_ekf = fmax(fabs(mu_ekf[0] - mx), fabs(mu_ekf[1] - my));
        double e_m_ukf = fmax(fabs(mu_ukf[0] - mx), fabs(mu_ukf[1] - my));
        double e_c_ekf = 0, e_c_ukf = 0;
        double Cmc[2][2] = {{C00, C01}, {C01, C11}};
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            e_c_ekf = fmax(e_c_ekf, fabs(C_ekf[i][j] - Cmc[i][j]));
            e_c_ukf = fmax(e_c_ukf, fabs(C_ukf[i][j] - Cmc[i][j]));
        }
        printf("  [demo] 均值误差:  UKF=%.4f < EKF=%.4f\n", e_m_ukf, e_m_ekf);
        printf("  [demo] 协方差误差: UKF=%.4f < EKF=%.4f (EKF 常低估不确定性)\n", e_c_ukf, e_c_ekf);
        chk("极坐标→直角: UKF 均值更准", e_m_ukf < e_m_ekf ? 0.0 : 1.0, 0.5);
        chk("极坐标→直角: UKF 协方差更准", e_c_ukf < e_c_ekf ? 0.0 : 1.0, 0.5);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
