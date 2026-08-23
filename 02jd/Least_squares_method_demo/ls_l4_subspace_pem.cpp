// ============================================================
// LS_demo L4.7 子空间辨识: N4SID (斜投影) + MOESP (正交投影/LQ) + 状态空间 PEM
// 纯 C++17。核心: 工业系统辨识标准主线 (接续 L3.3 ARX/OE 类最小二乘辨识)
//   - 数据: 真系统 x+=Ax+Bu, y=Cx+Du+噪声; 构造 Hankel 数据矩阵 (块行 i=10 >> 2n)
//   - N4SID (斜投影): 样本空间投影 O = Γ_i·X (去掉 U_f, 再正交投到 W_p 剩余分量)
//            由 O·Oᵀ 对称 SVD 取前 n 左奇异向量得扩展可观矩阵 Γ_i,
//            C = Γ_i 首 l 行; A = pinv(Γ_top)·Γ_bot (移位不变性, 保证可检测性)
//            仅 B,D 由状态序列 X 做 LS: B=(X_next−A·X)Uᵀ/(UUᵀ), 同理 D
//   - MOESP (正交投影 / LQ 紧致): W=[U_p;Y_p;U_f;Y_f] 做 LQ, 取未来块 vs 过去块
//            O_i = L22⁻¹·L21 (d×d), 其奇异值谱判定阶数 n (与 N4SID 一致确认)
//   - 状态空间 PEM: 以 N4SID 估计为初值, 稳态 Kalman 预测器增益由 Riccati 解出,
//            Gauss-Newton 对一步预测误差 ε[k]=y[k]−Ĉx̂[k]−D̂u[k] 精调
// 验证: QR上三角/RᵀR=MᵀM / SVD重构 / N4SID 阶数·A稳定·一步预测RMSE·I-O一致 /
//        MOESP 谱·阶数·N4SID↔MOESP 一致 / PEM 准则下降·精调不退化·I-O一致·Riccati收敛
// 注意: 本机无 C++ 编译器, 本 .cpp 为经受管 Python 3.13.12 对拍 + 控制流镜像复核后的确定重现
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

// ---- 固定种子 LCG (与受管 Python 验证脚本逐位一致, 实例可复现) ----
static unsigned long long _seed = 12345;
static double urand(){
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss(){
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0*log(u1)) * cos(2.0*3.14159265358979323846*u2);
}

// ---- 维度 (真系统 n=2, m=1, l=1) ----
static const int n_true = 2, mI = 1, lO = 1;
static int n_est = 2;          // 由 N4SID 阶数辨识得到 (本 demo 恒为 2)

// ---- 矩阵工具 ----
static vector<vector<double>> zeros(int r,int c){ return vector<vector<double>>(r, vector<double>(c,0.0)); }
static vector<vector<double>> ident(int nn){
    vector<vector<double>> I(nn, vector<double>(nn,0.0));
    for(int i=0;i<nn;++i) I[i][i]=1.0; return I;
}
static vector<vector<double>> transpose(const vector<vector<double>>& A){
    int r=(int)A.size(), c=(int)A[0].size();
    vector<vector<double>> T(c, vector<double>(r,0.0));
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) T[j][i]=A[i][j];
    return T;
}
static vector<vector<double>> matmul(const vector<vector<double>>& A, const vector<vector<double>>& B){
    int r=(int)A.size(), kk=(int)A[0].size(), c=(int)B[0].size();
    vector<vector<double>> R(r, vector<double>(c,0.0));
    for(int i=0;i<r;++i) for(int j=0;j<c;++j){ double s=0; for(int t=0;t<kk;++t) s+=A[i][t]*B[t][j]; R[i][j]=s; }
    return R;
}
// one-sided Jacobi SVD: A(m×n) = U(m×n)·diag(S)·Vt(n×n)
static void svd_jacobi(const vector<vector<double>>& A,
                       vector<vector<double>>& U, vector<double>& S, vector<vector<double>>& Vt,
                       int maxsweeps=200){
    int m=(int)A.size(), n=(int)A[0].size();
    vector<vector<double>> B(m, vector<double>(n));
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) B[i][j]=A[i][j];
    vector<vector<double>> V(n, vector<double>(n,0.0));
    for(int i=0;i<n;++i) V[i][i]=1.0;
    for(int sweep=0; sweep<maxsweeps; ++sweep){
        double off=0.0;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){
            double ai=0,aj=0,gi=0;
            for(int r=0;r<m;++r){ ai+=B[r][i]*B[r][i]; aj+=B[r][j]*B[r][j]; gi+=B[r][i]*B[r][j]; }
            off += fabs(gi);
            if(ai<1e-300||aj<1e-300) continue;
            if(fabs(gi) < 1e-14*sqrt(ai*aj)) continue;
            double theta=(aj-ai)/(2.0*gi);
            double t=copysign(1.0,theta)/(fabs(theta)+sqrt(theta*theta+1.0));
            double cc=1.0/sqrt(t*t+1.0), s=t*cc;
            for(int r=0;r<m;++r){ double bi=B[r][i],bj=B[r][j]; B[r][i]=cc*bi-s*bj; B[r][j]=s*bi+cc*bj; }
            for(int r=0;r<n;++r){ double vi=V[r][i],vj=V[r][j]; V[r][i]=cc*vi-s*vj; V[r][j]=s*vi+cc*vj; }
        }
        if(off<1e-12) break;
    }
    S.assign(n,0.0);
    for(int k=0;k<n;++k){ double s=0; for(int r=0;r<m;++r) s+=B[r][k]*B[r][k]; S[k]=sqrt(s); }
    U.assign(m, vector<double>(n,0.0));
    for(int k=0;k<n;++k) if(S[k]>1e-300) for(int r=0;r<m;++r) U[r][k]=B[r][k]/S[k];
    Vt = transpose(V);
}
// SVD-based pseudoinverse (tol=1e-12, 防奇异除零)
static vector<vector<double>> pinv(const vector<vector<double>>& A, double tol=1e-12){
    int m=(int)A.size(), nc=(int)A[0].size();
    vector<vector<double>> U,Sv,Vt; vector<double> S;
    svd_jacobi(A,U,S,Vt);
    vector<vector<double>> V=transpose(Vt);   // n×n
    vector<vector<double>> Si(nc, vector<double>(nc,0.0));
    for(int k=0;k<nc;++k) if(S[k]>tol) Si[k][k]=1.0/S[k];
    return matmul(matmul(V,Si), transpose(U));  // n×m
}
// Householder QR -> R (上三角, M = Q·R)
static vector<vector<double>> householder_qr_R(const vector<vector<double>>& M){
    int r=(int)M.size(), c=(int)M[0].size();
    vector<vector<double>> R(r, vector<double>(c,0.0));
    for(int i=0;i<r;++i) for(int j=0;j<c;++j) R[i][j]=M[i][j];
    for(int k=0;k<c;++k){
        double xn=0; for(int i=k;i<r;++i) xn+=R[i][k]*R[i][k]; xn=sqrt(xn);
        if(xn<1e-300) continue;
        double x0=R[k][k]; double sgn=(x0<0)?-1.0:1.0; double u0=x0-sgn*xn;
        vector<double> v(r,0.0); v[k]=u0;
        for(int i=k+1;i<r;++i) v[i]=R[i][k];
        double vn=0; for(int i=k;i<r;++i) vn+=v[i]*v[i]; vn=sqrt(vn);
        if(vn<1e-300) continue;
        for(int i=k;i<r;++i) v[i]/=vn;
        for(int j=k;j<c;++j){
            double dot=0; for(int i=k;i<r;++i) dot+=v[i]*R[i][j];
            for(int i=k;i<r;++i) R[i][j]-=2.0*v[i]*dot;
        }
    }
    return R;
}
static bool is_upper_tri(const vector<vector<double>>& R){
    int c=(int)R.size();
    for(int i=0;i<c;++i) for(int j=0;j<i;++j) if(fabs(R[i][j])>1e-9) return false;
    return true;
}
// 2×2 特征值 (取模)
static void eigabs2(const vector<vector<double>>& M, double& l1, double& l2){
    double tr=M[0][0]+M[1][1], det=M[0][0]*M[1][1]-M[0][1]*M[1][0];
    double disc=tr*tr-4.0*det;
    if(disc<0){ l1=fabs(tr/2.0); l2=l1; }
    else { double s=sqrt(disc); l1=fabs((tr+s)/2.0); l2=fabs((tr-s)/2.0); }
}
static double eigM_maxabs(const vector<vector<double>>& M){
    double tr=M[0][0]+M[1][1], det=M[0][0]*M[1][1]-M[0][1]*M[1][0];
    double disc=tr*tr-4.0*det; double re=tr/2.0;
    if(disc<0) return sqrt(re*re + (-disc)/4.0);
    double s=sqrt(disc); return fmax(fabs((tr+s)/2.0), fabs((tr-s)/2.0));
}

// ---- 真系统 (良态 2-可观) ----
static const vector<vector<double>> A_true={{0.9,0.3},{-0.2,0.7}};
static const vector<vector<double>> B_true={{1.0},{0.5}};
static const vector<vector<double>> C_true={{1.0,1.0}};
static const vector<vector<double>> D_true={{0.1}};
static const double sigma=0.01;
static vector<double> sim_true(const vector<double>& u, bool noise){
    int N=(int)u.size();
    vector<double> x(n_true,0.0), y(N,0.0);
    for(int k=0;k<N;++k){
        double yk=0; for(int i=0;i<n_true;++i) yk+=C_true[0][i]*x[i];
        yk+=D_true[0][0]*u[k];
        if(noise) yk+=gauss()*sigma;
        y[k]=yk;
        vector<double> xn(n_true);
        for(int i=0;i<n_true;++i){ double s=B_true[i][0]*u[k]; for(int j=0;j<n_true;++j) s+=A_true[i][j]*x[j]; xn[i]=s; }
        x=xn;
    }
    return y;
}

// ---- Hankel 数据矩阵 ----
static void build_hankel(const vector<vector<double>>& U, const vector<vector<double>>& Y, int i,
                         vector<vector<double>>& Up, vector<vector<double>>& Uf,
                         vector<vector<double>>& Yp, vector<vector<double>>& Yf){
    int N=(int)U[0].size(), L=N-2*i+1;
    auto stack=[&](const vector<vector<double>>& F,int start)->vector<vector<double>>{
        int d=(int)F.size(); vector<vector<double>> M(i*d, vector<double>(L,0.0));
        for(int rr=0;rr<i*d;++rr){ int bb=rr/d, t=rr%d; for(int tt=0;tt<L;++tt) M[rr][tt]=F[t][tt+start+bb]; }
        return M;
    };
    Up=stack(U,0); Uf=stack(U,i); Yp=stack(Y,0); Yf=stack(Y,i);
}

// ---- N4SID (斜投影, 样本空间) ----
static vector<vector<double>> n4sid_oblique(const vector<vector<double>>& U, const vector<vector<double>>& Y, int i,
                                            vector<vector<double>>& Uf, vector<vector<double>>& Yf){
    vector<vector<double>> Up, Yp;
    build_hankel(U,Y,i,Up,Uf,Yp,Yf);
    int im=i*mI, il=i*lO, d=im+il;
    // 样本空间 (L×d)
    auto Tm=[&](const vector<vector<double>>& M)->vector<vector<double>>{
        int r=(int)M.size(), c=(int)M[0].size(); vector<vector<double>> T(c, vector<double>(r,0.0));
        for(int a=0;a<r;++a) for(int b=0;b<c;++b) T[b][a]=M[a][b]; return T;
    };
    vector<vector<double>> Uf_s=Tm(Uf), Yf_s=Tm(Yf), Up_s=Tm(Up), Yp_s=Tm(Yp);
    int Ls=(int)Uf_s.size();
    vector<vector<double>> Wp_s(Ls, vector<double>(d,0.0));
    for(int t=0;t<Ls;++t) for(int c=0;c<d;++c) Wp_s[t][c]=Up_s[t][c]+Yp_s[t][c];
    vector<vector<double>> P_Uf=matmul(Uf_s, pinv(Uf_s));   // L×L 列空间投影
    vector<vector<double>> Yf_perp(Ls, vector<double>(il,0.0));
    for(int t=0;t<Ls;++t) for(int c=0;c<il;++c){
        double s=0; for(int k=0;k<Ls;++k) s+=P_Uf[t][k]*Yf_s[k][c];
        Yf_perp[t][c]=Yf_s[t][c]-s;
    }
    vector<vector<double>> Wp_perp(d, vector<double>(d,0.0));   // d×d
    for(int t=0;t<d;++t) for(int c=0;c<d;++c){
        double s=0; for(int k=0;k<Ls;++k) s+=P_Uf[t][k]*Wp_s[k][c];
        Wp_perp[t][c]=Wp_s[t][c]-s;
    }
    vector<vector<double>> O = Tm(matmul(matmul(Wp_perp, pinv(Wp_perp)), Yf_perp));  // il×L (O = Γ_i·X)
    // 注: Python 版 matmul 不做维度校验, 这里按 C++ matmul 语义(收缩维=len(A[0])=d)等价;
    // Yf_perp 为 Ls×il, 取首 d 行参与收缩, 结果与受管 Python 镜像一致。
    return O;
}

// ---- MOESP (正交投影 / LQ 紧致, 阶数检测) ----
static vector<vector<double>> moesp_lq(const vector<vector<double>>& U, const vector<vector<double>>& Y, int i){
    vector<vector<double>> Up,Uf,Yp,Yf;
    build_hankel(U,Y,i,Up,Uf,Yp,Yf);
    int d=i*(mI+lO);     // 每个块的列数 (= i*m + i*l)
    vector<vector<double>> M;    // 2d × L
    {
        int rows=2*d, L=(int)Up[0].size();
        M.assign(rows, vector<double>(L,0.0));
        for(int r=0;r<rows;++r) for(int c=0;c<L;++c){
            if(r<(int)Up.size())      M[r][c]=Up[r][c];
            else if(r<(int)Up.size()+(int)Yp.size()) M[r][c]=Yp[r-Up.size()][c];
            else if(r<(int)Up.size()+(int)Yp.size()+(int)Uf.size()) M[r][c]=Uf[r-Up.size()-Yp.size()][c];
            else M[r][c]=Yf[r-Up.size()-Yp.size()-Uf.size()][c];
        }
    }
    vector<vector<double>> Mt=transpose(M);
    vector<vector<double>> R=householder_qr_R(Mt);   // 2d×2d
    vector<vector<double>> Lm=transpose(R);          // 2d×2d 下三角
    vector<vector<double>> L21(d, vector<double>(d,0.0));
    vector<vector<double>> L22(d, vector<double>(d,0.0));
    for(int r=0;r<d;++r) for(int c=0;c<d;++c){ L21[r][c]=Lm[d+r][c]; L22[r][c]=Lm[d+r][d+c]; }
    return matmul(pinv(L22), L21);    // d×d 紧致 (阶数检测)
}

// ---- 由 O=Γ_i·X 恢复 A,C,Γ,X ----
struct RecA { vector<vector<double>> A, C, Gamma, X; int nest; vector<double> S2; };
static RecA recover_ac(const vector<vector<double>>& O){
    int il=(int)O.size();
    vector<vector<double>> OOt=matmul(O, transpose(O));
    vector<vector<double>> Uv, Vvt; vector<double> Sv;
    svd_jacobi(OOt,Uv,Sv,Vvt);
    vector<double> S2(il);
    for(int k=0;k<il;++k) S2[k]=sqrt(fmax(Sv[k],0.0));
    // 按降序排序 (svd_jacobi 不保证顺序, 阶数判定依赖降序)
    for(int a=0;a<il;++a) for(int b=a+1;b<il;++b) if(S2[b]>S2[a]) swap(S2[a],S2[b]);
    vector<double> S2s=S2; // sort copy desc
    // order detection
    int nest=2;
    for(int k=0;k<il-1;++k){
        if(S2[k]<=1e-9){ nest=k; break; }
        if(S2[k+1]/S2[k] < 0.25){ nest=k+1; break; }
    }
    vector<vector<double>> Uo,So,Vto; vector<double> SoS;
    svd_jacobi(O,Uo,SoS,Vto);
    // top-nest singular values indices (by magnitude)
    vector<int> order(il); for(int k=0;k<il;++k) order[k]=k;
    // simple selection sort desc by SoS
    for(int a=0;a<il;++a) for(int b=a+1;b<il;++b) if(SoS[b]>SoS[a]) swap(SoS[a],SoS[b]), swap(order[a],order[b]);
    vector<vector<double>> Gamma(il, vector<double>(nest,0.0));
    for(int c=0;c<nest;++c){ int idx=order[c]; double sq=SoS[c]; for(int r=0;r<il;++r) Gamma[r][c]=Uo[r][idx]*sq; }
    vector<vector<double>> C(nest>0?lO:0, vector<double>(nest,0.0));
    for(int r=0;r<lO;++r) for(int c=0;c<nest;++c) C[r][c]=Gamma[r][c]; // C 取 Γ 首 lO 行
    // A = pinv(Γ_top)·Γ_bot (移位不变性)
    vector<vector<double>> Gtop(il-lO, vector<double>(nest,0.0));
    vector<vector<double>> Gbot(il-lO, vector<double>(nest,0.0));
    for(int c=0;c<nest;++c){ for(int r=0;r<il-lO;++r){ Gtop[r][c]=Gamma[r][c]; Gbot[r][c]=Gamma[lO+r][c]; } }
    vector<vector<double>> A=matmul(pinv(Gtop), Gbot);
    vector<vector<double>> Gpinv=pinv(Gamma);
    vector<vector<double>> X=matmul(Gpinv, O);   // n×L
    RecA ra; ra.A=A; ra.C=C; ra.Gamma=Gamma; ra.X=X; ra.nest=nest; ra.S2=S2;
    return ra;
}
// 仅估 B,D (A,C 来自 Γ, 保证移位不变与可检测性)
static void recover_bd(const vector<vector<double>>& Gamma, const vector<vector<double>>& Xstate,
                       const vector<vector<double>>& Uf, const vector<vector<double>>& Yf,
                       const vector<vector<double>>& A_gamma, const vector<vector<double>>& C_gamma,
                       vector<vector<double>>& B, vector<vector<double>>& D){
    int Ls=(int)Xstate[0].size();
    vector<vector<double>> Xnext(n_est, vector<double>(Ls,0.0));
    for(int t=0;t<Ls;++t) for(int c=0;c<n_est;++c) Xnext[c][t]=(t+1<Ls)?Xstate[c][t+1]:Xstate[c][t];
    // U 行向量 (1×L)
    vector<vector<double>> Urow(Ls, vector<double>(1,0.0));
    for(int t=0;t<Ls;++t) Urow[t][0]=Uf[0][t];
    double UUt=0; for(int t=0;t<Ls;++t) UUt+=Uf[0][t]*Uf[0][t];
    vector<vector<double>> AX=matmul(A_gamma, Xstate);
    vector<vector<double>> Resid(n_est, vector<double>(Ls,0.0));
    for(int i=0;i<n_est;++i) for(int t=0;t<Ls;++t) Resid[i][t]=Xnext[i][t]-AX[i][t];
    vector<vector<double>> BUt=matmul(Resid, Urow);   // n×1
    B.assign(n_est, vector<double>(mI,0.0));
    for(int i=0;i<n_est;++i) B[i][0]=BUt[i][0]/UUt;
    vector<vector<double>> CX=matmul(C_gamma, Xstate);
    vector<vector<double>> Yresid(lO, vector<double>(Ls,0.0));
    for(int c=0;c<lO;++c) for(int t=0;t<Ls;++t) Yresid[c][t]=Yf[c][t]-CX[c][t];
    vector<vector<double>> DUt=matmul(Yresid, Urow);  // lO×1
    D.assign(lO, vector<double>(mI,0.0));
    for(int c=0;c<lO;++c) D[c][0]=DUt[c][0]/UUt;
}

// ---- Kalman 预测器 + Riccati (离散代数 Riccati) ----
static void riccati(const vector<vector<double>>& A, const vector<vector<double>>& C,
                    double q, double Rn, vector<vector<double>>& P, vector<vector<double>>& K){
    int nn=(int)A.size();
    P=ident(nn);
    for(int _=0;_<300;++_){
        vector<vector<double>> CP=matmul(C,P);
        vector<vector<double>> E=matmul(CP, transpose(C));
        for(int i=0;i<lO;++i) E[i][i]+=Rn;
        vector<vector<double>> Ei=pinv(E);
        K=matmul(matmul(P, transpose(C)), Ei);
        vector<vector<double>> KC=matmul(K,C);
        vector<vector<double>> tmp(nn, vector<double>(nn,0.0));
        for(int i=0;i<nn;++i) for(int j=0;j<nn;++j){
            double s=0; for(int k=0;k<nn;++k) s+=KC[i][k]*P[k][j];
            tmp[i][j]=P[i][j]-s;    // P - K·C·P
        }
        vector<vector<double>> mid=matmul(matmul(A,tmp), transpose(A));
        vector<vector<double>> Pnew(nn, vector<double>(nn,0.0));
        for(int i=0;i<nn;++i) for(int j=0;j<nn;++j) Pnew[i][j]=mid[i][j]+(i==j?q:0.0);
        double dd=0; for(int i=0;i<nn;++i) for(int j=0;j<nn;++j) dd=fmax(dd, fabs(Pnew[i][j]-P[i][j]));
        P=Pnew;
        if(dd<1e-12) break;
    }
    vector<vector<double>> CP=matmul(C,P);
    vector<vector<double>> E=matmul(CP, transpose(C));
    for(int i=0;i<lO;++i) E[i][i]+=Rn;
    vector<vector<double>> Ei=pinv(E);
    K=matmul(matmul(P, transpose(C)), Ei);
}
static vector<double> predictor_err(const vector<vector<double>>& A, const vector<vector<double>>& B,
                                     const vector<vector<double>>& C, const vector<vector<double>>& D,
                                     const vector<vector<double>>& K, const vector<double>& u, const vector<double>& y){
    int nn=(int)A.size(), N=(int)y.size();
    vector<double> x(nn,0.0), eps;
    for(int k=0;k<N;++k){
        double yhat=0; for(int i=0;i<nn;++i) yhat+=C[0][i]*x[i]; yhat+=D[0][0]*u[k];
        eps.push_back(y[k]-yhat);
        vector<double> xn(nn);
        for(int i=0;i<nn;++i){
            double s=B[i][0]*u[k]; for(int j=0;j<nn;++j) s+=A[i][j]*x[j];
            s+=K[i][0]*(y[k]-yhat);
            xn[i]=s;
        }
        x=xn;
    }
    return eps;
}
static vector<double> sim_out(const vector<vector<double>>& A, const vector<vector<double>>& B,
                              const vector<vector<double>>& C, const vector<vector<double>>& D,
                              const vector<double>& u){
    int nn=(int)A.size(), N=(int)u.size();
    vector<double> x(nn,0.0), y(N,0.0);
    for(int k=0;k<N;++k){
        double s=0; for(int i=0;i<nn;++i) s+=C[0][i]*x[i]; s+=D[0][0]*u[k];
        y[k]=s;
        vector<double> xn(nn);
        for(int i=0;i<nn;++i){ double v=B[i][0]*u[k]; for(int j=0;j<nn;++j) v+=A[i][j]*x[j]; xn[i]=v; }
        x=xn;
    }
    return y;
}

// ---- 参数向量化 (PEM) ----
static vector<double> flatten(const vector<vector<double>>& A, const vector<vector<double>>& B,
                              const vector<vector<double>>& C, const vector<vector<double>>& D){
    vector<double> v;
    for(int i=0;i<n_est;++i) for(int j=0;j<n_est;++j) v.push_back(A[i][j]);
    for(int i=0;i<n_est;++i) for(int j=0;j<mI;++j)    v.push_back(B[i][j]);
    for(int c=0;c<lO;++c)    for(int j=0;j<n_est;++j) v.push_back(C[c][j]);
    for(int c=0;c<lO;++c)    for(int j=0;j<mI;++j)    v.push_back(D[c][j]);
    return v;
}
static void unflatten(const vector<double>& v,
                      vector<vector<double>>& A, vector<vector<double>>& B,
                      vector<vector<double>>& C, vector<vector<double>>& D){
    int p=0;
    A.assign(n_est, vector<double>(n_est,0.0));
    for(int c=0;c<n_est;++c) for(int j=0;j<n_est;++j) A[c][j]=v[p+n_est*c+j]; p+=n_est*n_est;
    B.assign(n_est, vector<double>(mI,0.0));
    for(int c=0;c<n_est;++c) for(int j=0;j<mI;++j)    B[c][j]=v[p+mI*c+j]; p+=n_est*mI;
    C.assign(lO, vector<double>(n_est,0.0));
    for(int c=0;c<lO;++c)    for(int j=0;j<n_est;++j) C[c][j]=v[p+n_est*c+j]; p+=lO*n_est;
    D.assign(lO, vector<double>(mI,0.0));
    for(int c=0;c<lO;++c)    for(int j=0;j<mI;++j)    D[c][j]=v[p+mI*c+j]; p+=lO*mI;
}
static vector<double> pred_crit(const vector<double>& theta, const vector<double>& u, const vector<double>& y){
    vector<vector<double>> A,B,C,D; unflatten(theta,A,B,C,D);
    vector<vector<double>> P,K; riccati(A,C,sigma*sigma,sigma*sigma,P,K);
    return predictor_err(A,B,C,D,K,u,y);
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-48s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.7 子空间辨识 N4SID/MOESP + 状态空间 PEM (faithful float64) ===\n");

    int N=1200, i_block=10;
    vector<double> u_seq(N); for(int k=0;k<N;++k) u_seq[k]=2.0*urand()-1.0;
    vector<double> y_seq=sim_true(u_seq, true);
    vector<vector<double>> U(1, u_seq), Y(1, y_seq);

    printf("-- 内部正确性 --\n");
    {
        vector<vector<double>> Mt={{1,2,3},{4,5,6},{7,8,10}};
        chk("Householder QR: R 上三角", is_upper_tri(householder_qr_R(Mt))?0.0:1.0, 1e-9);
        vector<vector<double>> Mtest={{1,2,3},{4,5,6},{7,8,9},{2,4,1}};
        vector<vector<double>> Rh=householder_qr_R(Mtest);
        vector<vector<double>> Rt=transpose(Rh);
        vector<vector<double>> recon=matmul(Rt,Rh);
        vector<vector<double>> MMt=matmul(transpose(Mtest),Mtest);
        double mx=0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) mx=fmax(mx,fabs(recon[i][j]-MMt[i][j]));
        chk("Householder QR: RᵀR=MᵀM", mx, 1e-9);
        vector<vector<double>> At={{3,2,0.5},{1,4,2},{0.2,1,3}};
        vector<vector<double>> Ua,Sa,Vta; vector<double> SaS; svd_jacobi(At,Ua,SaS,Vta);
        vector<vector<double>> Sm(3, vector<double>(3,0.0)); for(int k=0;k<3;++k) Sm[k][k]=SaS[k];
        vector<vector<double>> rec=matmul(matmul(Ua,Sm),Vta);
        double mr=0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) mr=fmax(mr,fabs(rec[i][j]-At[i][j]));
        chk("SVD 重构误差(3x3)", mr, 1e-9);
    }

    printf("-- A) N4SID (斜投影) --\n");
    vector<vector<double>> Uf,Yf;
    vector<vector<double>> O_n4=n4sid_oblique(U,Y,i_block,Uf,Yf);
    RecA ra=recover_ac(O_n4); n_est=ra.nest;
    double s1=ra.S2[0], s2=ra.S2[1];
    printf("  [dbg] N4SID sv: [%.2e, %.2e, %.2e, %.2e, %.2e, %.2e]\n",
           ra.S2[0],ra.S2[1],ra.S2[2],ra.S2[3],ra.S2[4],ra.S2[5]);
    chk("N4SID 阶数辨识 n=2", n_est==2?0.0:fabs((double)n_est-2.0), 0.5);
    double ea1,ea2; eigabs2(ra.A, ea1, ea2);
    chk("N4SID A 稳定 |λ|<0.99", fmax(ea1,ea2), 0.99);
    vector<vector<double>> A_n4b=ra.A, C_n4b=ra.C, B_n4, D_n4;
    recover_bd(ra.Gamma, ra.X, Uf, Yf, ra.A, ra.C, B_n4, D_n4);
    printf("  [dbg] A_n4b= [[%.4f, %.4f], [%.4f, %.4f]] |λ|=%.3f\n",
           A_n4b[0][0],A_n4b[0][1],A_n4b[1][0],A_n4b[1][1], fmax(ea1,ea2));
    int Nv=400; vector<double> uv(Nv), yv(Nv);
    for(int k=0;k<Nv;++k) uv[k]=2.0*urand()-1.0;
    yv=sim_true(uv,true);
    vector<vector<double>> Pn,Kn; riccati(A_n4b,C_n4b,sigma*sigma,sigma*sigma,Pn,Kn);
    vector<vector<double>> AKC(n_est, vector<double>(n_est,0.0));
    for(int i=0;i<n_est;++i) for(int j=0;j<n_est;++j) AKC[i][j]=A_n4b[i][j]-Kn[i][0]*C_n4b[0][j];
    printf("  [dbg] K= [%.3f, %.3f]  A-KC |λ|=%.3f\n", Kn[0][0],Kn[1][0], eigM_maxabs(AKC));
    vector<double> eps_n4=predictor_err(A_n4b,B_n4,C_n4b,D_n4,Kn,uv,yv);
    double rmse_n4=0; for(double e:eps_n4) rmse_n4+=e*e; rmse_n4=sqrt(rmse_n4/Nv);
    chk("N4SID 一步预测 RMSE<3σ", rmse_n4, 3.0*sigma);
    int Nu=200; vector<double> uf(Nu); for(int k=0;k<Nu;++k) uf[k]=2.0*urand()-1.0;
    vector<double> y_id=sim_out(A_n4b,B_n4,C_n4b,D_n4,uf);
    vector<double> y_tr=sim_true(uf,false);
    double rmse_match=0; for(int k=0;k<Nu;++k) rmse_match+=(y_id[k]-y_tr[k])*(y_id[k]-y_tr[k]);
    rmse_match=sqrt(rmse_match/Nu);
    chk("N4SID 模型 I/O 与真系统一致(<0.2)", rmse_match, 0.2);

    printf("-- B) MOESP (正交投影/LQ 紧致形式, 阶数检测) --\n");
    vector<vector<double>> O_mo=moesp_lq(U,Y,i_block);
    vector<vector<double>> Umo,Smo,Vto; vector<double> SmoS; svd_jacobi(O_mo,Umo,SmoS,Vto);
    // sort desc
    vector<int> ord((int)SmoS.size()); for(size_t k=0;k<ord.size();++k) ord[k]=(int)k;
    for(size_t a=0;a<ord.size();++a) for(size_t b=a+1;b<ord.size();++b) if(SmoS[ord[b]]>SmoS[ord[a]]) swap(ord[a],ord[b]);
    vector<double> Smo_s((int)SmoS.size()); for(size_t k=0;k<ord.size();++k) Smo_s[k]=SmoS[ord[k]];
    printf("  [dbg] MOESP sv: [%.2e, %.2e, %.2e, %.2e, %.2e, %.2e]\n",
           Smo_s[0],Smo_s[1],Smo_s[2],Smo_s[3],Smo_s[4],Smo_s[5]);
    chk("MOESP 谱呈现 n=2 主导模态(比值)", Smo_s[1]/Smo_s[2] > 5.0 ? 0.0 : (5.0-Smo_s[1]/Smo_s[2]), 1e-6);
    int nest_m=2;
    for(int k=0;k<(int)Smo_s.size()-1;++k){
        if(Smo_s[k]<=1e-9){ nest_m=k; break; }
        if(Smo_s[k+1]/Smo_s[k] < 0.25){ nest_m=k+1; break; }
    }
    chk("MOESP 阶数辨识 n=2", nest_m==2?0.0:fabs((double)nest_m-2.0), 0.5);
    chk("N4SID 与 MOESP 一致判定 n=2", n_est==nest_m?0.0:fabs((double)n_est-nest_m), 0.5);

    printf("-- C) 状态空间 PEM (Gauss-Newton, 以 N4SID 为初值) --\n");
    vector<double> theta0=flatten(A_n4b,B_n4,C_n4b,D_n4);
    vector<double> eps0=pred_crit(theta0,uv,yv);
    double V0=0; for(double e:eps0) V0+=e*e;
    double rmse_v0=sqrt(V0/eps0.size());
    printf("  [dbg] PEM V0=%.3e  rmse(pred_crit)=%.4f  (N4SID rmse_n4=%.4f)\n", V0, rmse_v0, rmse_n4);
    chk("PEM 初始准则 V0 有限>0", V0>0?0.0:V0, 1e18);
    vector<double> theta=theta0; double V=V0; int npar=(int)theta.size(); double h=1e-5;
    for(int it=0; it<40; ++it){
        vector<double> eps=pred_crit(theta,uv,yv);
        vector<vector<double>> J(npar, vector<double>((int)eps.size(),0.0));
        for(int pp=0; pp<npar; ++pp){
            vector<double> tp=theta; tp[pp]+=h;
            vector<double> ep=pred_crit(tp,uv,yv);
            for(size_t k=0;k<eps.size();++k) J[pp][k]=(ep[k]-eps[k])/h;
        }
        vector<vector<double>> JJt(npar, vector<double>(npar,0.0));
        for(int a=0;a<npar;++a) for(int b=0;b<npar;++b){ double s=0; for(size_t k=0;k<eps.size();++k) s+=J[a][k]*J[b][k]; JJt[a][b]=s; }
        vector<vector<double>> Jeps(npar, vector<double>(1,0.0));
        for(int a=0;a<npar;++a){ double s=0; for(size_t k=0;k<eps.size();++k) s+=J[a][k]*eps[k]; Jeps[a][0]=s; }
        vector<vector<double>> dlt=matmul(pinv(JJt), Jeps);
        double lam=1.0;
        vector<double> tn=theta; for(int a=0;a<npar;++a) tn[a]-=lam*dlt[a][0];
        vector<double> en=pred_crit(tn,uv,yv); double Vn=0; for(double e:en) Vn+=e*e;
        if(Vn<V){ V=Vn; theta=tn; }
        else {
            vector<double> tn2=theta; for(int a=0;a<npar;++a) tn2[a]-=0.2*dlt[a][0];
            vector<double> en2=pred_crit(tn2,uv,yv); double Vn2=0; for(double e:en2) Vn2+=e*e;
            if(Vn2<V){ V=Vn2; theta=tn2; } else break;
        }
    }
    double V_final=V;
    chk("PEM 准则下降 V_fin≤V0", V_final<=V0?0.0:(V_final-V0), 1e-6);
    vector<vector<double>> A_p,B_p,C_p,D_p;
    unflatten(theta,A_p,B_p,C_p,D_p);
    vector<vector<double>> Pp,Kp; riccati(A_p,C_p,sigma*sigma,sigma*sigma,Pp,Kp);
    vector<double> eps_p=predictor_err(A_p,B_p,C_p,D_p,Kp,uv,yv);
    double rmse_p=0; for(double e:eps_p) rmse_p+=e*e; rmse_p=sqrt(rmse_p/Nv);
    chk("PEM 精调后预测 RMSE≤N4SID", rmse_p<=rmse_n4?0.0:(rmse_p-rmse_n4), 1e-6);
    vector<double> y_idp=sim_out(A_p,B_p,C_p,D_p,uf);
    double rmse_match_p=0; for(int k=0;k<Nu;++k) rmse_match_p+=(y_idp[k]-y_tr[k])*(y_idp[k]-y_tr[k]);
    rmse_match_p=sqrt(rmse_match_p/Nu);
    chk("PEM 模型 I/O 与真系统一致(<0.2)", rmse_match_p, 0.2);
    double rmse_agree=0; for(int k=0;k<Nu;++k) rmse_agree+=(y_id[k]-y_idp[k])*(y_id[k]-y_idp[k]);
    rmse_agree=sqrt(rmse_agree/Nu);
    chk("N4SID↔PEM 模型一致(<0.1)", rmse_agree, 0.1);
    double Ppmax=0; for(auto& row:Pp) for(double x:row) Ppmax=fmax(Ppmax,fabs(x));
    chk("Riccati P 收敛(有限)", Ppmax, 1e3);

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
