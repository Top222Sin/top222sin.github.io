// ============================================================
// pid_demo L4：分数阶 PID (FOPID) ⭐（暖通/家电/高精度）
// 纯 C++17。整数 PID 5 参: Kp,Ki,Kd; 分数阶 PID 再加两阶: λ(积分阶), μ(微分阶)。
//   连续型:  u(t) = Kp·e(t) + Ki·D^{-λ}e(t) + Kd·D^{μ}e(t)
// 离散化用 Grünwald-Letnikov (GL): D^{α}e(k) ≈ h^{-α}·Σ_{j=0}^{k}(-1)^j C(α,j)·e(k-j)
//   积分(α=-λ): D^{-λ}e = h^{+λ}·Σ wI[j]·e(k-j)   (lam=1 退化为真正积分器, 全历史不截断)
//   微分(α=+μ): D^{ μ}e = h^{-μ}·Σ wD[j]·e(k-j)   (权重快速衰减, 截断窗口即可)
// 演示: 同一 Kp/Ki/Kd 下, 分数阶微分 μ=1.2 比整数 μ=1 相位超前更多 -> 超调更低 -> IAE 更小。
// 编译: g++ -O3 -std=c++17 pid_l4_fopid.cpp -o t && ./t
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

typedef vector<vector<double>> Mat;
typedef vector<double> Vec;

static Mat eyeM(int n){ Mat E(n,Vec(n,0.0)); for(int i=0;i<n;++i) E[i][i]=1.0; return E; }
static Mat matmul(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(), p=(int)B[0].size();
    Mat C(n,Vec(p,0.0));
    for(int i=0;i<n;++i) for(int k=0;k<m;++k){ double a=A[i][k]; if(a==0.0)continue;
        for(int j=0;j<p;++j) C[i][j]+=a*B[k][j]; }
    return C;
}
static Mat transpose(const Mat&A){ int n=(int)A.size(), m=(int)A[0].size(); Mat T(m,Vec(n,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j]; return T; }
static Mat matsub(const Mat&A,const Mat&B){ int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]-B[i][j]; return C; }
static Mat matadd(const Mat&A,const Mat&B){ int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]+B[i][j]; return C; }
static Mat scalemat(double s,const Mat&A){ int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=s*A[i][j]; return C; }
static Mat inv(const Mat&A){
    int n=(int)A.size(); Mat M(n,Vec(2*n,0.0));
    for(int i=0;i<n;++i){ for(int j=0;j<n;++j) M[i][j]=A[i][j]; M[i][n+i]=1.0; }
    for(int col=0;col<n;++col){
        int piv=col; double best=0.0;
        for(int r=col;r<n;++r){ double v=fabs(M[r][col]); if(v>best){best=v;piv=r;} }
        if(piv!=col) for(int j=0;j<2*n;++j) swap(M[col][j],M[piv][j]);
        double pv=M[col][col]; for(int j=0;j<2*n;++j) M[col][j]/=pv;
        for(int r=0;r<n;++r) if(r!=col && fabs(M[r][col])>0.0){
            double f=M[r][col]; for(int j=0;j<2*n;++j) M[r][j]-=f*M[col][j];
        }
    }
    Mat R(n,Vec(n,0.0)); for(int i=0;i<n;++i) for(int j=0;j<n;++j) R[i][j]=M[i][n+j];
    return R;
}

// GL 系数: w_j = (-1)^j C(α,j), j=0..H-1
static Vec gl_weights(double alpha, int H){
    Vec w(H,0.0); w[0]=1.0;
    for(int j=1;j<H;++j) w[j]=w[j-1]*(j-1-alpha)/j;
    return w;
}

// Tustin 离散化(连续 ss -> 离散 ss)
static void tustin_ss(const Mat&Ac,const Mat&Bc,const Vec&Cc,double dt,
                      Mat&Ad,Mat&Bd,Vec&Cc2){
    int n=(int)Ac.size();
    Mat I=eyeM(n);
    Mat M=matsub(I, scalemat(dt/2.0, Ac));
    Mat Minv=inv(M);
    Ad=matmul(Minv, matadd(I, scalemat(dt/2.0, Ac)));
    Bd=matmul(Minv, scalemat(dt, Bc));
    Cc2=Cc;
}

// FOPID 闭环仿真; 返回 y, e 轨迹
static void sim_fopid(const Mat&Ad,const Mat&Bd,const Vec&Cc,double dt,double T,
                      double Kp,double Ki,double Kd,double lam,double mu,double r,
                      Vec&y,Vec&e){
    int N=(int)(T/dt); int n=(int)Ad.size();
    Vec x(n,0.0); y.clear(); e.clear(); Vec hist;
    Vec wI=gl_weights(-lam, N);            // 积分系数(全历史)
    int Hd=400; Vec wD=gl_weights(mu, Hd< N? Hd : N);  // 微分系数(截断窗口)
    for(int k=0;k<N;++k){
        double yk=0.0; for(int i=0;i<n;++i) yk+=Cc[i]*x[i];
        double ek=r-yk; e.push_back(ek); hist.push_back(ek);
        double sI=0.0; for(int j=0;j<=k;++j) sI+=wI[j]*hist[k-j];   // 全历史积分
        int jmax=(k<Hd-1)? k : Hd-1;
        double sD=0.0; for(int j=0;j<=jmax;++j) sD+=wD[j]*hist[k-j]; // 截断微分
        double u=Kp*ek + Ki*(pow(dt,lam))*sI + Kd*(1.0/pow(dt,mu))*sD;
        Vec xn(n,0.0);
        for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<n;++j) s+=Ad[i][j]*x[j]; xn[i]=s; }
        for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<(int)Bd[0].size();++j) s+=Bd[i][j]*u; xn[i]+=s; }
        x=xn; y.push_back(yk);
    }
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* nm,double err,double tol){
        bool ok=err<tol; printf("  %-40s err=%.3e tol=%.0e  %s\n",nm,err,tol,ok?"PASS":"FAIL");
        ok?++pass:++fail;
    };
    printf("=== L4：分数阶 PID / FOPID（λ,μ 两阶自由度）===\n");

    double dt=0.02, T=25.0, r=1.0;
    // 二阶欠阻尼对象: G(s)=1/(s^2+0.5s+1)  (wn=1, zeta=0.25)
    Mat Ac={{0.0,1.0},{-1.0,-0.5}}; Mat Bc={{0.0},{1.0}}; Vec Cc={1.0,0.0};
    Mat Ad,Bd; Vec Cc2;
    tustin_ss(Ac,Bc,Cc,dt,Ad,Bd,Cc2);

    double Kp=1.6,Ki=0.9,Kd=0.9;
    // 整数 PID (lam=1, mu=1)
    Vec yI,eI; sim_fopid(Ad,Bd,Cc2,dt,T,Kp,Ki,Kd,1.0,1.0,r,yI,eI);
    double iaeI=0.0; for(double v:eI) iaeI+=fabs(v)*dt;
    double osI=0.0; for(double v:yI) if(v-r>osI) osI=v-r;
    // 分数阶 PID (lam=1, mu=1.2)
    Vec yF,eF; sim_fopid(Ad,Bd,Cc2,dt,T,Kp,Ki,Kd,1.0,1.20,r,yF,eF);
    double iaeF=0.0; for(double v:eF) iaeF+=fabs(v)*dt;
    double osF=0.0; for(double v:yF) if(v-r>osF) osF=v-r;

    printf("  整数PID(lam=1,mu=1)   IAE=%.4f 超调=%.4f\n", iaeI, osI);
    printf("  FOPID(lam=1,mu=1.2)    IAE=%.4f 超调=%.4f\n", iaeF, osF);
    chk("FOPID IAE < 整数PID IAE", max(0.0, iaeF-iaeI), 1e-3);
    chk("FOPID 超调 < 整数PID 超调", max(0.0, osF-osI), 1e-3);
    chk("FOPID 稳态误差<0.02", fabs(eF.back()), 0.02);

    // GL 退化自检: μ=1 时系数应为 (1,-1,0,0,0)
    Vec wD1=gl_weights(1.0,5);
    double gerr=max(max(fabs(wD1[0]-1.0), fabs(wD1[1]+1.0)), fabs(wD1[2]));
    chk("GL mu=1 系数(1,-1,0,0,0)", gerr, 1e-12);

    printf("  [结论] 同 Kp/Ki/Kd 下, 分数阶微分 μ=1.2 多给相位超前, 超调 %.4f->%.4f, IAE %.4f->%.4f。\n", osI,osF,iaeI,iaeF);
    printf("           λ,μ 是 PID 之外的两个额外自由度(暖通/家电整定更柔顺)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
