// ============================================================
// KF_demo L4.4 多目标跟踪（JPDA / GM-PHD）+ OOSM 时间偏移 ⭐
// 纯 C++17（无第三方库）。把"多目标数据关联 + 乱序量测"三大工业级知识点，
// 建立在 L2.x（KF/信息滤波/RTS）+ L3.x（数据关联/融合）之上：
//   Part A  JPDA 软关联：门控(χ² 99% 门限) + 联合关联事件枚举 + 边缘关联概率 β_ij
//          => 组合新息是候选新息的凸组合（不强制硬判决，避免歧义量测错配）。
//   Part B  OOSM 乱序量测：迟到量测经"从 m-1 重滤波"等价于全量测正序重滤波
//          （OOSM 定理 RHS）；永久丢弃则会损失信息（后验不确定度变大）。
//   Part C  GM-PHD 基数估计：加权高斯分量 Σw 直接给出目标个数
//          （出生→存活→死亡，无需显式数据关联）。
// 验证（受管 Python 3.13.12 先行对拍，本文件为确定性重现）：
//   15 项检查全 PASS，末行打印 "=== N passed, M failed ==="
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
using namespace std;

using Vec  = vector<double>;
using Mat  = vector<Vec>;
static const double PI = 3.14159265358979323846;

// ---- 固定种子 LCG + Box-Muller（与验证端逐位一致，seed=20260820）----
static unsigned long long _seed = 20260820ULL;
static double urand() {
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    _seed &= 0xFFFFFFFFFFFFFFFFULL;
    return (double)((_seed >> 33) & 0x7FFFFFFFULL) / (double)0x80000000ULL;
}
static double gauss() {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

// ---- 2x2 线性代数 ----
static Mat inv2(const Mat& M) {
    double d = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    d = (fabs(d) > 1e-15) ? d : (d >= 0 ? 1e-15 : -1e-15);
    return {{M[1][1]/d, -M[0][1]/d}, {-M[1][0]/d, M[0][0]/d}};
}
static Mat matadd(const Mat& A, const Mat& B) {
    return {{A[0][0]+B[0][0], A[0][1]+B[0][1]}, {A[1][0]+B[1][0], A[1][1]+B[1][1]}};
}
static Vec vsub(const Vec& a, const Vec& b) { return {a[0]-b[0], a[1]-b[1]}; }
static Vec vadd(const Vec& a, const Vec& b) { return {a[0]+b[0], a[1]+b[1]}; }
static double vnorm(const Vec& a) { return sqrt(a[0]*a[0] + a[1]*a[1]); }
static Vec matvec(const Mat& A, const Vec& x) {
    return {A[0][0]*x[0]+A[0][1]*x[1], A[1][0]*x[0]+A[1][1]*x[1]};
}
static double gauss2(const Vec& z, const Vec& m, const Mat& S) {
    double det = S[0][0]*S[1][1] - S[0][1]*S[1][0];
    Mat Sinv = inv2(S);
    Vec nu = vsub(z, m);
    double ex = -0.5 * (nu[0]*(Sinv[0][0]*nu[0]+Sinv[0][1]*nu[1]) + nu[1]*(Sinv[1][0]*nu[0]+Sinv[1][1]*nu[1]));
    return exp(ex) / (2.0 * PI * sqrt(fabs(det)));
}
static double mahal2(const Vec& z, const Vec& m, const Mat& Sinv) {
    Vec nu = vsub(z, m);
    return nu[0]*(Sinv[0][0]*nu[0]+Sinv[0][1]*nu[1]) + nu[1]*(Sinv[1][0]*nu[0]+Sinv[1][1]*nu[1]);
}

// ---- 全局验证计数 ----
static int g_pass = 0, g_fail = 0;
static void chk(const char* name, double err, double tol) {
    bool ok = err < tol;
    printf("  %-56s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
    ok ? ++g_pass : ++g_fail;
}

// ============================================================
// Part A：JPDA 软关联（2D 双目标 + 杂波 + 门控）
// ============================================================
static void partA() {
    printf("=== A. JPDA 软关联（门控 + 联合关联事件枚举）===\n");
    Vec x1 = {2.0, 0.0}, x2 = {-2.0, 0.0};
    Mat P  = {{0.5, 0.0}, {0.0, 0.5}};
    Mat R  = {{1.0, 0.0}, {0.0, 1.0}};
    Mat S  = matadd(P, R);
    Mat Sinv = inv2(S);
    double gate_gamma = 9.21;   // 2D 99% χ² 门限

    // 量测：m1 靠近 t1，m2 靠近 t2，m3 远处杂波，m4 两目标中点（歧义）
    map<string, Vec> meas = {
        {"m1", {2.3, 0.1}}, {"m2", {-2.3, -0.1}},
        {"m3", {10.0, 10.0}}, {"m4", {0.0, 0.0}}
    };
    vector<string> keys = {"m1", "m2", "m3", "m4"};

    auto in_gate = [&](const string& zk, const Vec& x) { return mahal2(meas[zk], x, Sinv) <= gate_gamma; };
    vector<string> val1, val2;
    for (auto& k : keys) { if (in_gate(k, x1)) val1.push_back(k); }
    for (auto& k : keys) { if (in_gate(k, x2)) val2.push_back(k); }
    if (find(val1.begin(), val1.end(), "m3") != val1.end() ||
        find(val2.begin(), val2.end(), "m3") != val2.end())
        printf("  [assert] m3 不应落在任何门内\n");

    double PD = 0.9, PG = 1.0, lambda_c = 0.1;

    // 联合关联事件枚举（每量测至多分给一个目标）
    using Event = map<int, string>;   // target -> meas key (""=漏检)
    map<int, vector<string>> valsets = {{1, val1}, {2, val2}};
    vector<int> targets = {1, 2};

    vector<Event> events;
    auto enumerate = [&](auto& self, size_t idx, set<string>& taken, Event& cur) -> void {
        if (idx == targets.size()) { events.push_back(cur); return; }
        int t = targets[idx];
        cur[t] = "";                                   // 漏检分支
        self(self, idx + 1, taken, cur);
        for (auto& m : valsets[t]) {                   // 关联分支
            if (taken.count(m)) continue;
            cur[t] = m; taken.insert(m);
            self(self, idx + 1, taken, cur);
            taken.erase(m);
        }
        cur.erase(t);
    };
    set<string> taken; Event cur;
    enumerate(enumerate, 0, taken, cur);

    auto event_L = [&](const Event& ev) -> double {
        double L = 1.0;
        set<string> assigned;
        for (auto& kv : ev) if (kv.second != "") assigned.insert(kv.second);
        for (auto& kv : ev) {
            int t = kv.first; string m = kv.second;
            Vec x = (t == 1) ? x1 : x2;
            if (m == "") L *= (1.0 - PD * PG);
            else          L *= PD * PG * gauss2(meas[m], x, S);
        }
        set<string> unassigned;
        for (auto& k : keys) if (!assigned.count(k)) unassigned.insert(k);
        for (size_t i = 0; i < unassigned.size(); ++i) L *= lambda_c / (1.0 - PD * PG);
        return L;
    };

    double Ltot = 0.0;
    for (auto& e : events) Ltot += event_L(e);

    map<int, map<string, double>> beta;  // β_ij
    map<int, double> beta0;              // β_i0 (漏检)
    for (auto& e : events) {
        double p = event_L(e) / Ltot;
        for (auto& kv : e) {
            int t = kv.first; string m = kv.second;
            if (m == "") beta0[t] += p;
            else          beta[t][m] += p;
        }
    }

    // A1：边缘关联概率每行归一化
    double s1 = 0.0; for (auto& kv : beta[1]) s1 += kv.second; s1 += beta0[1];
    double s2 = 0.0; for (auto& kv : beta[2]) s2 += kv.second; s2 += beta0[2];
    chk("JPDA 边缘关联概率归一化(目标1 和=1)", fabs(s1 - 1.0), 1e-9);
    chk("JPDA 边缘关联概率归一化(目标2 和=1)", fabs(s2 - 1.0), 1e-9);

    // A2：远处杂波 m3 不被任何目标关联
    double b13 = beta[1].count("m3") ? beta[1]["m3"] : 0.0;
    double b23 = beta[2].count("m3") ? beta[2]["m3"] : 0.0;
    chk("门控剔除远处杂波(m3 关联概率=0)", max(b13, b23), 1e-12);

    // A3：歧义量测 m4（两目标中点）被软分配，对称
    double b14 = beta[1].count("m4") ? beta[1]["m4"] : 0.0;
    double b24 = beta[2].count("m4") ? beta[2]["m4"] : 0.0;
    chk("歧义量测软分配(β_1m4≈β_2m4, 对称)", fabs(b14 - b24), 1e-9);
    chk("歧义量测确被两目标共享(β_1m4>0 且 β_2m4>0)", max(0.0, 1e-3 - min(b14, b24)), 1e-9);

    // A4：组合新息是候选新息的凸组合（模长 ≤ 最大单量测新息）
    auto combined_innov = [&](const Vec& x, const map<string, double>& betas) -> Vec {
        Vec nu = {0.0, 0.0};
        for (auto& kv : betas) {
            Vec d = vsub(meas[kv.first], x);
            nu = {nu[0] + kv.second * d[0], nu[1] + kv.second * d[1]};
        }
        return nu;
    };
    Vec nu1 = combined_innov(x1, beta[1]);
    Vec nu2 = combined_innov(x2, beta[2]);
    double maxn1 = 0.0, maxn2 = 0.0;
    for (auto& kv : beta[1]) maxn1 = max(maxn1, vnorm(vsub(meas[kv.first], x1)));
    for (auto& kv : beta[2]) maxn2 = max(maxn2, vnorm(vsub(meas[kv.first], x2)));
    chk("JPDA 组合新息为凸组合(模长≤最大单量测新息, 目标1)", max(0.0, vnorm(nu1) - maxn1), 1e-9);
    chk("JPDA 组合新息为凸组合(模长≤最大单量测新息, 目标2)", max(0.0, vnorm(nu2) - maxn2), 1e-9);
    printf("  [info] β_1=%.4f β_2=%.4f（m4 对称软分配）；组合新息模长 目标1=%.4f 目标2=%.4f\n",
           b14, b24, vnorm(nu1), vnorm(nu2));
}

// ============================================================
// Part B：OOSM 乱序量测（1D，与重滤波对拍）
// ============================================================
static void refilter(const vector<double>& z, int skip, double Q, double R,
                     vector<double>& xs, vector<double>& Ps) {
    // 标量 KF（F=1, H=1, Q, R）；skip=k 时该步量测不参与（模拟"迟到/丢弃"）
    int n = (int)z.size();
    xs.assign(n, 0.0); Ps.assign(n, 0.0);
    double x = 0.0, P = 10.0;
    for (int k = 0; k < n; ++k) {
        double xpred = x, Ppred = P + Q;
        if (k != skip) {
            double S = Ppred + R, K = Ppred / S;
            x = xpred + K * (z[k] - xpred); P = (1.0 - K) * Ppred;
        } else {
            x = xpred; P = Ppred;
        }
        xs[k] = x; Ps[k] = P;
    }
}

static pair<double, double> oosm(const vector<double>& z, int m,
                                 const vector<double>& xf, const vector<double>& Pf,
                                 double Q, double R) {
    // 精确 OOSM（等价于"从延迟量测时刻 m-1 起正序重滤波"）：
    // 取前向丢弃运行在 m-1 时刻的后验 xf[m-1]（已正确处理 z[0..m-1]，与正序重滤波一致），
    // 再正序滤波 z[m], z[m+1], ..., z[N]。该结果与"全部量测正序"重滤波严格相等。
    int n = (int)z.size();
    double x = xf[m-1], P = Pf[m-1];
    for (int k = m; k < n; ++k) {
        double xpred = x, Ppred = P + Q;
        double S = Ppred + R, K = Ppred / S;
        x = xpred + K * (z[k] - xpred); P = (1.0 - K) * Ppred;
    }
    return {x, P};
}

static double gauss1d(double z, double m, double S) {
    return 1.0 / sqrt(2.0 * PI * S) * exp(-0.5 * (z - m) * (z - m) / S);
}

static void partB() {
    printf("\n=== B. OOSM 乱序量测处理（1D，与重滤波对拍）===\n");
    int N = 30; double Q = 0.5, R = 1.0;
    // 真值随机游走 + 量测（受管 RNG，固定调用序）
    vector<double> xt(N + 1, 0.0);
    for (int k = 0; k < N; ++k) xt[k+1] = xt[k] + gauss() * sqrt(Q);
    vector<double> z(N + 1);
    for (int k = 0; k <= N; ++k) z[k] = xt[k] + gauss() * sqrt(R);

    vector<double> xs_ref, Ps_ref, xs_d1, Ps_d1, xs_d2, Ps_d2;
    refilter(z, -1,     Q, R, xs_ref, Ps_ref);   // 参考：全量测正序（OOSM 定理 RHS）
    refilter(z, N-1,    Q, R, xs_d1, Ps_d1);      // 对照：z[N-1] 迟到后永久丢弃（1 步滞后）
    refilter(z, N-2,    Q, R, xs_d2, Ps_d2);      // 对照：z[N-2] 迟到后永久丢弃（2 步滞后）

    auto [x_oosm1, P_oosm1] = oosm(z, N-1, xs_d1, Ps_d1, Q, R);
    auto [x_oosm2, P_oosm2] = oosm(z, N-2, xs_d2, Ps_d2, Q, R);

    // B1/B2：OOSM 结果 ≡ 重滤波（全部量测正序）—— OOSM 定理数值恒等判据（对噪声实现鲁棒）
    chk("OOSM(1步) ≡ 重滤波(全部量测正序)", fabs(x_oosm1 - xs_ref[N]), 1e-9);
    chk("OOSM(2步) ≡ 重滤波(全部量测正序)", fabs(x_oosm2 - xs_ref[N]), 1e-9);
    // B3：OOSM 校正确实并入迟到量测（≠ 永久丢弃方案）
    chk("OOSM(1步) 与丢弃方案存在显著差异(已并入迟到量测)", max(0.0, 1e-6 - fabs(x_oosm1 - xs_d1[N])), 1e-9);

    // B4/B5：常量目标（F=1, Q=0）大滞后场景——并入迟到量测严格降低后验不确定度
    int Nc = 20; double Rc = 1.0, x_true = 5.0;
    vector<double> zc(Nc + 1);
    for (int k = 0; k <= Nc; ++k) zc[k] = x_true + gauss() * sqrt(Rc);
    vector<double> xs_ref_c, Ps_ref_c, xs_dc_c, Ps_dc_c;
    refilter(zc, -1,   0.0, Rc, xs_ref_c, Ps_ref_c);
    refilter(zc, Nc-5, 0.0, Rc, xs_dc_c, Ps_dc_c);
    auto [x_oosm_c, P_oosm_c] = oosm(zc, Nc-5, xs_dc_c, Ps_dc_c, 0.0, Rc);
    chk("常量目标: OOSM ≡ 重滤波(全部量测正序)", fabs(x_oosm_c - xs_ref_c[Nc]), 1e-9);
    chk("常量目标大滞后: OOSM 后验不确定度 < 丢弃后验不确定度",
        max(0.0, P_oosm_c - Ps_dc_c[Nc]), 1e-9);
    printf("  [info] 随机游走: oosm1=%.5f disc1=%.5f ref=%.5f | 常量: oosm=%.4f disc=%.4f ref=%.4f P_oosm=%.4f P_disc=%.4f\n",
           x_oosm1, xs_d1[N], xs_ref[N], x_oosm_c, xs_dc_c[Nc], xs_ref_c[Nc], P_oosm_c, Ps_dc_c[Nc]);
}

// ============================================================
// Part C：GM-PHD 基数估计（出生 / 存活 / 死亡）
// ============================================================
struct Comp { double m, P, w; };   // 1D 高斯分量：均值 / 协方差 / 权重

static void partC() {
    printf("\n=== C. GM-PHD 基数估计（出生→存活→死亡）===\n");
    double PD = 0.99, PS = 0.95, lambda_c = 0.01, R = 0.5, Q = 0.1;
    double w_min = 1e-3;
    int Kmax = 60, birth_k = 10, death_k = 40;
    double x_birth = 3.0;

    vector<Comp> comps;
    vector<double> card_seq;
    for (int k = 0; k <= Kmax; ++k) {
        // 1) 预测：存活已有分量 + 加 birth
        vector<Comp> new_comps;
        for (auto& c : comps) new_comps.push_back({c.m, c.P + Q, c.w * PS});
        if (k == birth_k) new_comps.push_back({x_birth, 1.0, 0.8});   // 出生分量
        comps = new_comps;
        // 2) 生成量测：目标存活期内生成 1 个（带噪）
        vector<double> meas_list;
        if (birth_k <= k && k <= death_k) meas_list.push_back(x_birth + gauss() * sqrt(R));
        // 3) 更新：漏检分量 + 每个量测的更新分量
        vector<Comp> upd;
        for (auto& c : comps) {
            upd.push_back({c.m, c.P, (1.0 - PD) * c.w});   // 漏检
            if (!meas_list.empty()) {
                double denom = lambda_c;
                for (auto& cl : comps) {
                    double Sl = cl.P + R;
                    denom += PD * cl.w * gauss1d(meas_list[0], cl.m, Sl);
                }
                for (double zj : meas_list) {
                    double S = c.P + R;
                    double q = gauss1d(zj, c.m, S);
                    double wu = (PD * c.w / denom) * q;
                    double K = c.P / S;
                    double mu = c.m + K * (zj - c.m);
                    double Pu = (1.0 - K) * c.P;
                    upd.push_back({mu, Pu, wu});
                }
            }
        }
        // 4) 剪枝
        vector<Comp> pruned;
        for (auto& c : upd) if (c.w > w_min) pruned.push_back(c);
        comps = pruned;
        // 5) 基数估计 = Σ w
        double Nhat = 0.0; for (auto& c : comps) Nhat += c.w;
        card_seq.push_back(Nhat);
    }

    double Nhat_pre  = card_seq[5];    // 出生前
    double Nhat_alv  = card_seq[35];   // 存活期
    double Nhat_post = card_seq[55];   // 死亡后
    chk("出生前基数≈0(无目标)", max(0.0, Nhat_pre - 0.05), 1e-9);
    chk("存活期基数≈1(单目标)", fabs(Nhat_alv - 1.0), 0.15);
    chk("死亡后基数≈0(目标离开)", max(0.0, Nhat_post - 0.05), 1e-9);
    printf("  [info] N̂: 出生前(k=5)=%.4f, 存活(k=35)=%.4f, 死亡后(k=55)=%.4f\n",
           Nhat_pre, Nhat_alv, Nhat_post);
}

// ============================================================
// 主验证
// ============================================================
int main() {
    partA();
    partB();
    partC();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
