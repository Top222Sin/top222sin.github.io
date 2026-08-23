// L4.10 Schur 补 BA（矩阵块累加同构）
// 纯 C++17 标准库实现；算法经受管 Python 3.13.12 验证（5/5 PASS）+ C++ 控制流镜像复核。
// 关键点：
//   (1) 固定相机 0（锚定坐标系）消除 BA 的 7 自由度相似歧义（gauge），法方程 H 满秩可逆；
//   (2) BA 用 Levenberg-Marquardt（λI 阻尼）应对近 gauge 病态；
//   (3) 法方程块结构 [Hcc Hcp; Hpc Hpp]，Schur 补 S=Hcc−Hcp·Hpp⁻¹·Hpc 边缘化 3D 点 → 约化相机系统；
//   (4) 逐项 JᵢᵀJᵢ 累加 ≡ JᵀJ（与 SME ZA 外积、KF 因子图信息矩阵、LS JᵀJ 同构）。
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

// ---------- 小矩阵/线性代数 ----------
static Mat matmul(const Mat& A, const Mat& B) {
    int n = (int)A.size(), m = (int)A[0].size(), p = (int)B[0].size();
    Mat C(n, Vec(p, 0.0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < m; k++) {
            double a = A[i][k];
            if (a == 0.0) continue;
            const Vec& Bk = B[k];
            for (int j = 0; j < p; j++) C[i][j] += a * Bk[j];
        }
    return C;
}
static Vec matvec(const Mat& A, const Vec& x) {
    int n = (int)A.size(), m = (int)A[0].size();
    Vec y(n, 0.0);
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < m; j++) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}
static Mat transpose(const Mat& A) {
    int n = (int)A.size(), m = (int)A[0].size();
    Mat T(m, Vec(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
}
static Vec solve(const Mat& A_, const Vec& b) {
    int n = (int)A_.size();
    Mat M(n, Vec(n + 1, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) M[i][j] = A_[i][j];
        M[i][n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col; r < n; r++)
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        std::swap(M[col], M[piv]);
        double d = M[col][col];
        for (int j = col; j <= n; j++) M[col][j] /= d;
        for (int r = 0; r < n; r++) {
            if (r != col && M[r][col] != 0.0) {
                double f = M[r][col];
                for (int j = col; j <= n; j++) M[r][j] -= f * M[col][j];
            }
        }
    }
    Vec x(n);
    for (int i = 0; i < n; i++) x[i] = M[i][n];
    return x;
}
static Mat inv(const Mat& A) {
    int n = (int)A.size();
    Mat cols;
    for (int j = 0; j < n; j++) {
        Vec b(n, 0.0); b[j] = 1.0;
        cols.push_back(solve(A, b));
    }
    return transpose(cols);
}
static Mat cholesky(const Mat& A) {
    int n = (int)A.size();
    Mat L(n, Vec(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 1e-15) return Mat();
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return L;
}

// ---------- SO(3) 工具 ----------
static Mat skew(const Vec& v) {
    double x = v[0], y = v[1], z = v[2];
    return Mat{{0.0,-z,y},{z,0.0,-x},{-y,x,0.0}};
}
static Mat rodrigues(const Vec& w) {
    double th = std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    if (th < 1e-12) return Mat{{1,0,0},{0,1,0},{0,0,1}};
    Vec uw = {w[0]/th, w[1]/th, w[2]/th};
    Mat K = skew(uw);
    Mat K2 = matmul(K, K);
    double s = std::sin(th), c = std::cos(th);
    Mat R(3, Vec(3, 0.0));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = (i==j?1.0:0.0) + s*K[i][j] + (1.0-c)*K2[i][j];
    return R;
}
static Mat jr(const Vec& w) {
    double th = std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    if (th < 1e-12) return Mat{{1,0,0},{0,1,0},{0,0,1}};
    Vec uw = {w[0]/th, w[1]/th, w[2]/th};
    Mat K = skew(uw);
    Mat K2 = matmul(K, K);
    double a = (1.0-std::cos(th))/(th*th);
    double b = (th-std::sin(th))/(th*th*th);
    Mat J(3, Vec(3, 0.0));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            J[i][j] = (i==j?1.0:0.0) - a*K[i][j] + b*K2[i][j];
    return J;
}
static Vec matvec3(const Mat& A, const Vec& v) {
    return Vec{
        A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
        A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
        A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]};
}

// ---------- 确定性 PRNG（LCG，与 Python 一致） ----------
struct Rng {
    long long state;
    double operator()() {
        state = (state * 1103515245LL + 12345) & 0x7fffffffLL;
        return (double)state / 2147483647.0;
    }
};

// ---------- 问题参数 ----------
const int NC = 3;
const int CAM_VAR[2] = {1, 2};
const int NVAR_CAM = 2;
const int NP = 8;
const double F = 500.0, CX = 320.0, CY = 320.0;
const int NCAM = 6 * NVAR_CAM;   // 12
const int NPAR = NCAM + 3 * NP;  // 36

static std::pair<double,double> project(const Vec& w, const Vec& t, const Vec& X) {
    Mat R = rodrigues(w);
    Vec Pc = matvec3(R, X);
    for (int i = 0; i < 3; i++) Pc[i] += t[i];
    double Xc = Pc[0], Yc = Pc[1], Zc = Pc[2];
    if (Zc <= 1e-6) return {1e9, 1e9};
    return {F*Xc/Zc + CX, F*Yc/Zc + CY};
}

// 相机/点拆分
struct Cam { Vec w, t; };
struct Split { std::vector<Cam> cams; std::vector<Vec> X; };
static Split split(const Vec& p, const Vec& fw0, const Vec& ft0) {
    Split s;
    s.cams.resize(NC);
    s.cams[0] = {fw0, ft0};
    for (int vc = 0; vc < NVAR_CAM; vc++) {
        int c = CAM_VAR[vc];
        Vec w(p.begin()+6*vc,   p.begin()+6*vc+3);
        Vec t(p.begin()+6*vc+3, p.begin()+6*vc+6);
        s.cams[c] = {w, t};
    }
    s.X.resize(NP);
    for (int j = 0; j < NP; j++)
        s.X[j] = Vec(p.begin()+NCAM+3*j, p.begin()+NCAM+3*j+3);
    return s;
}

static double cost_of(const Vec& p, const double mu[3][8], const double mv[3][8],
                      const Vec& fw0, const Vec& ft0) {
    Split s = split(p, fw0, ft0);
    double cost = 0.0;
    for (int c = 0; c < NC; c++)
        for (int j = 0; j < NP; j++) {
            auto [u, v] = project(s.cams[c].w, s.cams[c].t, s.X[j]);
            double du = u - mu[c][j], dv = v - mv[c][j];
            cost += du*du + dv*dv;
        }
    return cost;
}

static Vec residual_vec(const Vec& p, const double mu[3][8], const double mv[3][8],
                        const Vec& fw0, const Vec& ft0) {
    Split s = split(p, fw0, ft0);
    Vec r(NC*NP*2, 0.0);
    int ri = 0;
    for (int c = 0; c < NC; c++)
        for (int j = 0; j < NP; j++) {
            auto [u, v] = project(s.cams[c].w, s.cams[c].t, s.X[j]);
            r[ri++] = u - mu[c][j];
            r[ri++] = v - mv[c][j];
        }
    return r;
}

static Mat jacobian(const Vec& p, const Vec& fw0, const Vec& ft0) {
    Split s = split(p, fw0, ft0);
    Mat J(NC*NP*2, Vec(NPAR, 0.0));
    int ri = 0;
    for (int c = 0; c < NC; c++) {
        int var_idx = -1;
        for (int vi = 0; vi < NVAR_CAM; vi++) if (CAM_VAR[vi] == c) var_idx = vi;
        for (int j = 0; j < NP; j++) {
            Mat R = rodrigues(s.cams[c].w);
            Vec Pc = matvec3(R, s.X[j]);
            for (int i = 0; i < 3; i++) Pc[i] += s.cams[c].t[i];
            double Xc = Pc[0], Yc = Pc[1], Zc = Pc[2];
            Vec gu = {F/Zc, 0.0, -F*Xc/(Zc*Zc)};
            Vec gv = {0.0, F/Zc, -F*Yc/(Zc*Zc)};
            Mat dX(2, Vec(3, 0.0));
            for (int a = 0; a < 2; a++) {
                Vec grada = (a==0)? gu : gv;
                for (int k = 0; k < 3; k++) {
                    double sm = 0.0;
                    for (int bb = 0; bb < 3; bb++) sm += grada[bb]*R[bb][k];
                    dX[a][k] = sm;
                }
            }
            Mat dT(2, Vec(3, 0.0));
            dT[0] = gu; dT[1] = gv;
            Vec RX = matvec3(R, s.X[j]);
            Mat RXx = skew(RX);
            Mat JR = jr(s.cams[c].w);
            Mat tmp = matmul(RXx, JR);
            Mat dW(2, Vec(3, 0.0));
            for (int a = 0; a < 2; a++)
                for (int k = 0; k < 3; k++) dW[a][k] = -tmp[a][k];
            for (int a = 0; a < 2; a++) {
                if (var_idx >= 0) {
                    for (int k = 0; k < 3; k++) J[ri+a][6*var_idx+k]     = dW[a][k];
                    for (int k = 0; k < 3; k++) J[ri+a][6*var_idx+3+k]   = dT[a][k];
                }
                for (int k = 0; k < 3; k++) J[ri+a][NCAM+3*j+k] = dX[a][k];
            }
            ri += 2;
        }
    }
    return J;
}

static Mat block(const Mat& H, int r0, int r1, int c0, int c1) {
    Mat B(r1-r0, Vec(c1-c0, 0.0));
    for (int i = r0; i < r1; i++)
        for (int j = c0; j < c1; j++) B[i-r0][j-c0] = H[i][j];
    return B;
}

// 阻尼系统 (H+λI) 的 Schur 边缘化
static void schur_solve(const Mat& Hcc, const Mat& Hcp, const Mat& Hpc, const Mat& Hpp,
                        const Vec& bc, const Vec& bp, double lam,
                        Vec& dc, Vec& dp, Mat& S) {
    int n_c = (int)Hcc.size(), n_p = (int)Hpp.size();
    Mat Hcc_l(n_c, Vec(n_c, 0.0)), Hpp_l(n_p, Vec(n_p, 0.0));
    for (int i = 0; i < n_c; i++)
        for (int j = 0; j < n_c; j++) Hcc_l[i][j] = Hcc[i][j] + (i==j?lam:0.0);
    for (int i = 0; i < n_p; i++)
        for (int j = 0; j < n_p; j++) Hpp_l[i][j] = Hpp[i][j] + (i==j?lam:0.0);
    Mat Hpp_i = inv(Hpp_l);
    Mat mid = matmul(Hcp, Hpp_i);     // n_c × n_p
    Mat mid2 = matmul(mid, Hpc);      // n_c × n_c
    S = Mat(n_c, Vec(n_c, 0.0));
    for (int a = 0; a < n_c; a++)
        for (int b = 0; b < n_c; b++) S[a][b] = Hcc_l[a][b] - mid2[a][b];
    Vec bp_via = matvec(mid, bp);
    Vec rhs_c(n_c);
    for (int a = 0; a < n_c; a++) rhs_c[a] = bc[a] - bp_via[a];
    dc = solve(S, rhs_c);
    Vec Hpc_dc = matvec(Hpc, dc);
    Vec z(n_p);
    for (int k = 0; k < n_p; k++) z[k] = bp[k] - Hpc_dc[k];
    dp = matvec(Hpp_i, z);
}

// ---------- 检查工具 ----------
static bool chk(const std::string& name, double got, double thr, bool ge, const std::string& extra) {
    bool ok; double val;
    if (ge) { ok = got >= thr - 1e-12; val = got; }
    else   { ok = std::fabs(got) <= thr + 1e-12; val = std::fabs(got); }
    printf("  [%s] %s: %.3e (thr=%+.1e) %s\n", ok?"PASS":"FAIL", name.c_str(), val, thr, extra.c_str());
    return ok;
}
static double max_abs_diff(const Mat& A, const Mat& B) {
    double m = 0.0;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            m = std::max(m, std::fabs(A[i][j]-B[i][j]));
    return m;
}
static double max_abs_vec(const Vec& a, const Vec& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i]-b[i]));
    return m;
}

int main() {
    Rng rng; rng.state = 20260820;

    // ---- 生成问题（与 Python 同序）----
    Mat true_w(NC, Vec(3,0.0)), true_t(NC, Vec(3,0.0)), true_X(NP, Vec(3,0.0));
    for (int c = 0; c < NC; c++)
        for (int k = 0; k < 3; k++) true_w[c][k] = 2.0*(rng()-0.5)*0.05;
    for (int c = 0; c < NC; c++) {
        Vec base = {0.0,0.0,5.0};
        if (c==1) base = {0.6,0.0,5.0};
        else if (c==2) base = {-0.6,0.0,5.0};
        for (int k = 0; k < 3; k++) true_t[c][k] = base[k] + 2.0*(rng()-0.5)*0.1;
    }
    for (int j = 0; j < NP; j++)
        for (int k = 0; k < 3; k++) true_X[j][k] = 2.0*(rng()-0.5)*1.5;
    double mu[3][8], mv[3][8];
    for (int c = 0; c < NC; c++)
        for (int j = 0; j < NP; j++) {
            auto [u, v] = project(true_w[c], true_t[c], true_X[j]);
            u += 2.0*(rng()-0.5)*0.5;
            v += 2.0*(rng()-0.5)*0.5;
            mu[c][j] = u; mv[c][j] = v;
        }
    Mat init_w(NC, Vec(3,0.0)), init_t(NC, Vec(3,0.0));
    for (int c = 0; c < NC; c++) {
        if (c==1 || c==2) {
            for (int k = 0; k < 3; k++) init_w[c][k] = true_w[c][k] + 2.0*(rng()-0.5)*0.02;
            for (int k = 0; k < 3; k++) init_t[c][k] = true_t[c][k] + 2.0*(rng()-0.5)*0.05;
        } else {
            init_w[c] = true_w[c]; init_t[c] = true_t[c];
        }
    }
    Mat init_X(NP, Vec(3,0.0));
    for (int j = 0; j < NP; j++)
        for (int k = 0; k < 3; k++) init_X[j][k] = true_X[j][k] + 2.0*(rng()-0.5)*0.03;

    Vec fw0 = true_w[0], ft0 = true_t[0];
    Vec p0;
    for (int vc = 0; vc < NVAR_CAM; vc++) {
        int c = CAM_VAR[vc];
        for (int k = 0; k < 3; k++) p0.push_back(init_w[c][k]);
        for (int k = 0; k < 3; k++) p0.push_back(init_t[c][k]);
    }
    for (int j = 0; j < NP; j++)
        for (int k = 0; k < 3; k++) p0.push_back(init_X[j][k]);

    const double LAM = 1e-2;

    Mat J = jacobian(p0, fw0, ft0);
    Vec r = residual_vec(p0, mu, mv, fw0, ft0);
    Mat H_full = matmul(transpose(J), J);
    Vec g = matvec(transpose(J), r);
    Vec bc(NCAM), bp(NPAR-NCAM);
    for (int k = 0; k < NCAM; k++) bc[k] = -g[k];
    for (int k = 0; k < NPAR-NCAM; k++) bp[k] = -g[NCAM+k];

    Mat H_acc(NPAR, Vec(NPAR, 0.0));
    int nres = (int)J.size();
    for (int i = 0; i < nres; i++) {
        Mat Ji(1, Vec(NPAR, 0.0));
        for (int j = 0; j < NPAR; j++) Ji[0][j] = J[i][j];
        Mat JiT = transpose(Ji);
        Mat JiTJi = matmul(JiT, Ji);
        for (int a = 0; a < NPAR; a++)
            for (int b = 0; b < NPAR; b++) H_acc[a][b] += JiTJi[a][b];
    }

    Mat Hcc = block(H_acc, 0, NCAM, 0, NCAM);
    Mat Hcp = block(H_acc, 0, NCAM, NCAM, NPAR);
    Mat Hpc = block(H_acc, NCAM, NPAR, 0, NCAM);
    Mat Hpp = block(H_acc, NCAM, NPAR, NCAM, NPAR);

    Vec dc, dp; Mat S;
    schur_solve(Hcc, Hcp, Hpc, Hpp, bc, bp, LAM, dc, dp, S);

    Mat Hbig(NPAR, Vec(NPAR, 0.0));
    for (int i = 0; i < NPAR; i++)
        for (int j = 0; j < NPAR; j++) Hbig[i][j] = H_acc[i][j] + (i==j?LAM:0.0);
    Vec rhs_big(NPAR);
    for (int k = 0; k < NPAR; k++) rhs_big[k] = -g[k];
    Vec dbig = solve(Hbig, rhs_big);
    Vec dc_full(dbig.begin(), dbig.begin()+NCAM);
    Vec dp_full(dbig.begin()+NCAM, dbig.begin()+NPAR);

    // LM 迭代（Schur 实现，带 λ 回溯保证下降）
    auto lm_step = [&](const Vec& p, double lam0) -> std::pair<Vec,double> {
        Mat Jl = jacobian(p, fw0, ft0);
        Vec rl = residual_vec(p, mu, mv, fw0, ft0);
        Mat Hl = matmul(transpose(Jl), Jl);
        Vec gl = matvec(transpose(Jl), rl);
        Mat Hccl = block(Hl,0,NCAM,0,NCAM), Hcpl = block(Hl,0,NCAM,NCAM,NPAR);
        Mat Hpcl = block(Hl,NCAM,NPAR,0,NCAM), HPPl = block(Hl,NCAM,NPAR,NCAM,NPAR);
        Vec bcl(NCAM), bpl(NPAR-NCAM);
        for (int k = 0; k < NCAM; k++) bcl[k] = -gl[k];
        for (int k = 0; k < NPAR-NCAM; k++) bpl[k] = -gl[NCAM+k];
        double c0 = cost_of(p, mu, mv, fw0, ft0);
        double lam = lam0;
        Vec dcl, dpl; Mat Sl;
        Vec delta(NPAR, 0.0);
        for (int it = 0; it < 50; it++) {
            schur_solve(Hccl, Hcpl, Hpcl, HPPl, bcl, bpl, lam, dcl, dpl, Sl);
            for (int k = 0; k < NCAM; k++) delta[k] = dcl[k];
            for (int k = 0; k < NPAR-NCAM; k++) delta[NCAM+k] = dpl[k];
            Vec pnew(NPAR);
            for (int k = 0; k < NPAR; k++) pnew[k] = p[k] + delta[k];
            if (cost_of(pnew, mu, mv, fw0, ft0) < c0) return {delta, lam};
            lam *= 3.0;
        }
        return {delta, lam};
    };

    double cost0 = cost_of(p0, mu, mv, fw0, ft0);
    Vec p = p0;
    std::vector<double> costs = {cost0};
    double lam = LAM;
    for (int it = 0; it < 25; it++) {
        auto [delta, nlam] = lm_step(p, lam);
        lam = nlam;
        for (int k = 0; k < NPAR; k++) p[k] += delta[k];
        costs.push_back(cost_of(p, mu, mv, fw0, ft0));
    }
    double cost_final = costs.back();
    double rms0 = std::sqrt(cost0/(NC*NP*2));
    double rms_final = std::sqrt(cost_final/(NC*NP*2));

    int passed = 0, total = 0;
    printf("== L4.10 Schur 补 BA 验证 ==\n");
    total++; if (chk("块累加≡JᵀJ(同构)", max_abs_diff(H_acc, H_full), 1e-9, false,
                     "max|Δ|="+std::to_string(max_abs_diff(H_acc,H_full)))) passed++;
    total++; {
        double e2 = std::max(max_abs_vec(dc, dc_full), max_abs_vec(dp, dp_full));
        if (chk("Schur约化δ≡全解δ(阻尼)", e2, 1e-7, false, "max|Δδ|="+std::to_string(e2))) passed++;
    }
    total++; {
        Mat Lm = cholesky(S);
        bool spd_ok = !Lm.empty();
        double minpiv = spd_ok ? Lm[0][0] : 0.0;
        for (int i = 0; spd_ok && i < (int)Lm.size(); i++) minpiv = std::min(minpiv, Lm[i][i]);
        if (chk("Sλ对称正定(Cholesky)", spd_ok?0.0:1.0, 1e-9, false,
                "minpiv="+(spd_ok?std::to_string(minpiv):std::string("None")))) passed++;
    }
    total++; {
        double e4 = std::max(0.0, costs[1]-cost0);
        if (chk("LM单步代价下降", e4, 1e-9, false, "Δcost="+std::to_string(costs[1]-cost0))) passed++;
    }
    total++; {
        double e5 = std::max(0.0, rms_final - 0.5*rms0);
        if (chk("LM收敛(final<0.5·init RMS)", e5, 1e-3, false,
                "rms0="+std::to_string(rms0)+"→rms_final="+std::to_string(rms_final)+"px")) passed++;
    }

    printf("PASS=%d  FAIL=%d  (total=%d)\n", passed, total-passed, total);
    printf("RESULT: %s\n", passed==total ? "ALL PASS" : "HAS FAIL");
    return passed==total ? 0 : 1;
}
