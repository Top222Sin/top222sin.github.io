// ls_l4_pca_lowrank.cpp —— L4.9 PCA / 低秩（作为 LS）
// 纯 C++17 标准库；与受管 Python 验证同算法、同确定性 PRNG（LCG seed=20260820）
//
// 核心：PCA = 最佳低秩逼近（Eckart-Young），等价于在协方差矩阵 C=XcᵀXc/(n-1)
//       上做特征分解，取 top-k 特征向量张成的子空间。这正是「把数据投影到
//       使重构误差最小的 k 维子空间」——一个序列最小二乘（幂迭代 / Rayleigh 商）。
//   特征分解用 Jacobi 旋转（对称阵，纯 C++ 可写、确定、稳定）。
//
// 四件套：合成低秩数据(秩2)+噪声 → 中心化 → 协方差 Jacobi 特征分解 →
//         子空间/方差解释/重建误差(Eckart-Young) 对拍 → PASS/FAIL。

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

using VD = std::vector<double>;
using MD = std::vector<std::vector<double>>;

static MD matmul(const MD& A, const MD& B) {
    int n = (int)A.size(), m = (int)A[0].size(), p = (int)B[0].size();
    MD C(n, VD(p, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) { double a = A[i][k]; if (a == 0.0) continue;
            for (int j = 0; j < p; ++j) C[i][j] += a * B[k][j]; }
    return C;
}
static VD matvec(const MD& A, const VD& x) {
    int n = (int)A.size(), m = (int)A[0].size(); VD y(n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) y[i] += A[i][j] * x[j];
    return y;
}
static MD transpose(const MD& A) {
    int n = (int)A.size(), m = (int)A[0].size(); MD T(m, VD(n, 0.0));
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) T[j][i] = A[i][j];
    return T;
}
static double dot(const VD& a, const VD& b) { double s = 0.0; for (size_t i = 0; i < a.size(); ++i) s += a[i]*b[i]; return s; }
static double normv(const VD& v) { double s = 0.0; for (double x : v) s += x*x; return std::sqrt(s); }

static uint32_t g_lcg = 20260820u;
static double rnd() { g_lcg = g_lcg * 1103515245u + 12345u; return (double)(g_lcg & 0x7fffffff) / 2147483647.0; }

// Gram-Schmidt 正交化（列）
static MD ortho_columns(const MD& M) {
    int n = (int)M.size(), k = (int)M[0].size();
    MD Q(n, VD(k, 0.0));
    for (int j = 0; j < k; ++j) {
        VD v(n); for (int i = 0; i < n; ++i) v[i] = M[i][j];
        for (int q = 0; q < j; ++q) {
            VD qj(n); for (int i = 0; i < n; ++i) qj[i] = Q[i][q];
            double proj = dot(v, qj);
            for (int i = 0; i < n; ++i) v[i] -= proj * qj[i];
        }
        double nv = normv(v);
        for (int i = 0; i < n; ++i) Q[i][j] = v[i] / nv;
    }
    return Q;
}

// Jacobi 特征分解（对称阵），返回降序特征值 + 对应特征向量（列）
static void jacobi_eig(const MD& Min, VD& eig, MD& V) {
    int n = (int)Min.size();
    MD A = Min;
    V = MD(n, VD(n, 0.0)); for (int i = 0; i < n; ++i) V[i][i] = 1.0;
    for (int sweep = 0; sweep < 200; ++sweep) {
        int p = 0, q = 1; double amax = 0.0;
        for (int i = 0; i < n; ++i) for (int j = i+1; j < n; ++j)
            if (std::fabs(A[i][j]) > amax) { amax = std::fabs(A[i][j]); p = i; q = j; }
        if (amax < 1e-12) break;
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        if (apq == 0.0) continue;
        double theta = (aqq - app) / (2.0 * apq);
        double t = std::copysign(1.0, theta) / (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
        double c = 1.0 / std::sqrt(t*t + 1.0), s = t * c;
        A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;
        A[p][q] = 0.0; A[q][p] = 0.0;
        for (int r = 0; r < n; ++r) if (r != p && r != q) {
            double arp = A[r][p], arq = A[r][q];
            A[r][p] = c*arp - s*arq; A[p][r] = A[r][p];
            A[r][q] = s*arp + c*arq; A[q][r] = A[r][q];
        }
        for (int r = 0; r < n; ++r) {
            double vrp = V[r][p], vrq = V[r][q];
            V[r][p] = c*vrp - s*vrq; V[r][q] = s*vrp + c*vrq;
        }
    }
    eig.resize(n);
    for (int i = 0; i < n; ++i) eig[i] = A[i][i];
    std::vector<int> order(n); for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){ return eig[a] > eig[b]; });
    VD e2 = eig; for (int i = 0; i < n; ++i) eig[i] = e2[order[i]];
    MD Vnew(n, VD(n, 0.0));
    for (int r = 0; r < n; ++r) for (int kk = 0; kk < n; ++kk) Vnew[r][kk] = V[r][order[kk]];
    V = Vnew;
}

static int g_pass = 0, g_fail = 0;
static void chk(const char* name, double got, double thr, bool ge, const char* extra) {
    bool ok = ge ? (got >= thr - 1e-12) : (std::fabs(got) <= thr + 1e-12);
    printf("  [%s] %s: %.3e (thr=%+.1e) %s\n", ok ? "PASS" : "FAIL", name, ge ? got : std::fabs(got), thr, extra);
    if (ok) ++g_pass; else ++g_fail;
}

int main() {
    int n = 40, d = 5, r = 2;
    MD V2raw(d, VD(r)); for (int i = 0; i < d; ++i) for (int j = 0; j < r; ++j) V2raw[i][j] = 2.0*rnd() - 1.0;
    V2raw = ortho_columns(V2raw);
    MD U2raw(n, VD(r)); for (int i = 0; i < n; ++i) for (int j = 0; j < r; ++j) U2raw[i][j] = 2.0*rnd() - 1.0;
    U2raw = ortho_columns(U2raw);
    double S2[2] = { 3.0, 1.5 };
    MD SV(n, VD(r)); for (int i = 0; i < n; ++i) for (int a = 0; a < r; ++a) SV[i][a] = U2raw[i][a]*S2[a];
    MD Xsig = matmul(SV, transpose(V2raw));
    MD X(n, VD(d));
    for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) X[i][j] = Xsig[i][j] + 1e-3*(2.0*rnd()-1.0);
    VD mean(d, 0.0); for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) mean[j] += X[i][j];
    for (int j = 0; j < d; ++j) mean[j] /= n;
    MD Xc(n, VD(d)); for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) Xc[i][j] = X[i][j] - mean[j];

    MD C(d, VD(d));
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) {
        double s = 0.0; for (int i = 0; i < n; ++i) s += Xc[i][a]*Xc[i][b]/(n-1); C[a][b] = s;
    }
    VD eig; MD V;
    jacobi_eig(C, eig, V);

    printf("== L4.9 PCA / 低秩验证 ==\n");
    // ① 特征向量正交归一 VᵀV=I
    double worst = 0.0;
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) {
        VD va(d), vb(d); for (int rr = 0; rr < d; ++rr) { va[rr] = V[rr][a]; vb[rr] = V[rr][b]; }
        double val = dot(va, vb) - (a == b ? 1.0 : 0.0);
        worst = std::max(worst, std::fabs(val));
    }
    { char e[64]; snprintf(e,sizeof(e),"maxdev=%.3e",worst); chk("特征向量正交归一", worst, 1e-6, false, e); }

    // ② top-r 方差解释比
    double sumall = 0.0, sumtop = 0.0; for (int i = 0; i < d; ++i) { sumall += eig[i]; if (i < r) sumtop += eig[i]; }
    double ratio = sumtop/sumall;
    { char e[64]; snprintf(e,sizeof(e),"ratio=%.4f",ratio); chk("top-2 方差解释比", 1.0-ratio, 0.1, false, e); }

    // ③ 真信号子空间被恢复：P = Vtop2 Vtop2ᵀ 投影到 top-r 子空间
    MD Vtop2(r, VD(d)); for (int i = 0; i < r; ++i) for (int rr = 0; rr < d; ++rr) Vtop2[i][rr] = V[rr][i];
    MD P(d, VD(d, 0.0));
    for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) {
        double s = 0.0; for (int i = 0; i < r; ++i) s += Vtop2[i][a]*Vtop2[i][b]; P[a][b] = s;
    }
    double min_proj = 1.0;
    for (int j = 0; j < r; ++j) {
        VD v2j(d); for (int i = 0; i < d; ++i) v2j[i] = V2raw[i][j];
        VD pv(d, 0.0); for (int a = 0; a < d; ++a) for (int b = 0; b < d; ++b) pv[a] += P[a][b]*v2j[b];
        min_proj = std::min(min_proj, normv(pv));
    }
    { char e[64]; snprintf(e,sizeof(e),"min||proj||=%.4f",min_proj); chk("真信号子空间被恢复", 1.0-min_proj, 1e-2, false, e); }

    // ④ 重建误差 ≈ 尾部和：‖X−X_k‖²_F = (n-1) Σ_{i>r} λ_i
    MD Xk = matmul(Xc, P);
    double rec_err = 0.0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) { double er = Xc[i][j]-Xk[i][j]; rec_err += er*er; }
    double tailsum = 0.0; for (int i = r; i < d; ++i) tailsum += eig[i];
    double tail = (n-1)*tailsum;
    double rel = std::fabs(rec_err - tail)/(tail + 1e-12);
    { char e[64]; snprintf(e,sizeof(e),"rec=%.4f tail=%.4f",rec_err,tail); chk("重建误差≈尾部和(Eckart-Young)", rel, 1e-3, false, e); }

    // ⑤ 第一主方向 = 最大方差方向（Rayleigh 商）；幂迭代收敛到 v1
    VD v(d, 0.0); for (int i = 0; i < d; ++i) v[i] = 1.0/std::sqrt((double)d);
    for (int it = 0; it < 100; ++it) { VD w = matvec(C, v); double nv = normv(w); for (int i = 0; i < d; ++i) v[i] = w[i]/nv; }
    VD v1(d); for (int rr = 0; rr < d; ++rr) v1[rr] = V[rr][0];
    double cos1 = std::fabs(dot(v, v1));
    VD Cv1 = matvec(C, v1); double var1 = dot(v1, Cv1);
    double worst5 = std::max(1.0 - cos1, std::fabs(var1 - eig[0])/(eig[0] + 1e-12));
    { char e[64]; snprintf(e,sizeof(e),"cos=%.4f var1=%.4f λ1=%.4f",cos1,var1,eig[0]); chk("幂迭代收敛到v1 & 方差=λ1", worst5, 1e-3, false, e); }

    printf("PASS=%d  FAIL=%d  (total=%d)\n", g_pass, g_fail, g_pass + g_fail);
    printf("RESULT: %s\n", g_fail == 0 ? "ALL PASS" : "HAS FAIL");
    return 0;
}
