// ============================================================
// LS_demo L4.2 通信信号处理:信道估计 + LMS + 波束成形
// 纯 C++17。① OFDM 导频 LS: H_p=Y_p/X_p + 插值 → 降误码
// ② LMS: 随机梯度版 LS,w→Wiener 解 R⁻¹p
// ③ MVDR 波束成形: w=R⁻¹a/(aᴴR⁻¹a), 期望向增益 1、干扰零点
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>
using namespace std;
typedef complex<double> C;

static unsigned long long _seed = 42;
static double urand(){ _seed=_seed*6364136223846793005ULL+1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }
static const double PI_ = 3.14159265358979323846;

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-38s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.2 信道估计 + LMS + 波束成形 ===\n");

    // ---- ① OFDM LS 信道估计 ----
    const int N=64, Np=8;
    vector<C> h={1.0, 0.5*exp(C(0,0.3)), 0.2*exp(C(0,-1.1))};
    auto H=[&](int k){ C s=0; for(int l=0;l<3;++l) s+=h[l]*exp(C(0,-2*PI_*k*l/N)); return s; };
    vector<int> X(N);
    for(int k=0;k<N;++k) X[k]= (urand()<0.5)?1:-1;
    double sigma=sqrt(0.5/20.0);
    vector<C> Y(N);
    for(int k=0;k<N;++k) Y[k]=H(k)*double(X[k])+C(gauss(),gauss())*sigma;
    int pilot[Np];
    for(int i=0;i<Np;++i) pilot[i]=i*(N/Np);
    vector<C> Hp(Np);
    for(int i=0;i<Np;++i) Hp[i]=Y[pilot[i]]/(double)X[pilot[i]];
    vector<C> Hest(N, C(1,0));
    for(int seg=0;seg<Np-1;++seg){
        int k0=pilot[seg], k1=pilot[seg+1];
        for(int k=k0;k<k1;++k){ double t=(double)(k-k0)/(k1-k0);
            Hest[k]=Hp[seg]*(1-t)+Hp[seg+1]*t; } }
    for(int k=pilot[Np-1];k<N;++k) Hest[k]=Hp[Np-1];
    int err_ls=0, err0=0;
    for(int k=0;k<N;++k){
        C xh=Y[k]/Hest[k];
        if((xh.real()>0?1:-1)!=X[k]) ++err_ls;
        if((Y[k].real()>0?1:-1)!=X[k]) ++err0;
    }
    chk("LS 信道估计降低误码", err_ls<=err0?0.0:1.0, 0.5);
    printf("  [demo] 64 子载波 8 导频: 误码 %d → %d\n", err0, err_ls);

    // ---- ② LMS ----
    double wt[3]={0.8,-0.4,0.2}, mu=0.05, w[3]={0,0,0};
    for(int n=0;n<2000;++n){
        double u[3]={gauss(),gauss(),gauss()};
        double d=wt[0]*u[0]+wt[1]*u[1]+wt[2]*u[2]+gauss()*0.01;
        double y=w[0]*u[0]+w[1]*u[1]+w[2]*u[2];
        double e=d-y;
        for(int i=0;i<3;++i) w[i]+=mu*e*u[i];
    }
    double werr=fmax(fabs(w[0]-wt[0]),fmax(fabs(w[1]-wt[1]),fabs(w[2]-wt[2])));
    chk("LMS 收敛到 Wiener 解", werr, 0.08);
    printf("  [demo] LMS 2000 步: w=[%.3f,%.3f,%.3f](真值 0.8,-0.4,0.2)\n",w[0],w[1],w[2]);

    // ---- ③ MVDR 波束成形 ----
    const int M=4; double dd=0.5;
    auto steer=[&](double th){ vector<C> a(M);
        for(int i=0;i<M;++i) a[i]=exp(C(0,-2*PI_*dd*sin(th)*i)); return a; };
    auto a0=steer(30*PI_/180), a1=steer(-20*PI_/180);
    vector<vector<C>> R(M, vector<C>(M, 0));
    for(int n=0;n<2000;++n){
        C s0=C(gauss(),gauss()), s1=C(gauss(),gauss())*10.0;
        vector<C> x(M);
        for(int i=0;i<M;++i) x[i]=a0[i]*s0+a1[i]*s1+C(gauss(),gauss())*0.5;
        for(int i=0;i<M;++i) for(int j=0;j<M;++j) R[i][j]+=x[i]*conj(x[j]);
        }
    for(int i=0;i<M;++i) for(int j=0;j<M;++j) R[i][j]/=2000.0;
    // 复矩阵高斯消元解 R·Ra = a0
    vector<vector<C>> Aug(M, vector<C>(M+1));
    for(int i=0;i<M;++i){ for(int j=0;j<M;++j) Aug[i][j]=R[i][j]; Aug[i][M]=a0[i]; }
    for(int c=0;c<M;++c){ int p=c;
        for(int i=c;i<M;++i) if(abs(Aug[i][c])>abs(Aug[p][c])) p=i;
        swap(Aug[c],Aug[p]);
        for(int i=0;i<M;++i) if(i!=c){ C f=Aug[i][c]/Aug[c][c];
            for(int j=c;j<=M;++j) Aug[i][j]-=f*Aug[c][j]; } }
    vector<C> Ra(M);
    for(int i=0;i<M;++i) Ra[i]=Aug[i][M]/Aug[i][i];
    C den=0; for(int i=0;i<M;++i) den+=conj(a0[i])*Ra[i];
    vector<C> wv(M);
    for(int i=0;i<M;++i) wv[i]=Ra[i]/den;
    double g0=0, g1=0;
    for(int i=0;i<M;++i){ g0+= (conj(wv[i])*a0[i]).real(); g1+= (conj(wv[i])*a1[i]).real(); }
    g0=abs(C(g0,0)); g1=abs(C(g1,0));
    chk("波束成形: 期望方向增益=1", fabs(g0-1), 1e-3);
    chk("波束成形: 干扰方向零点", g1, 0.15);
    printf("  [demo] 增益 g(30°)=%.3f, g(-20°)=%.4f(强干扰零深)\n", g0, g1);
    printf("  [结论] 三者同源: 导频 LS=频域线性方程;LMS=LS 的随机梯度在线版;\n");
    printf("        MVDR=带方向约束的加权 LS(等价 Wiener 滤波)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
