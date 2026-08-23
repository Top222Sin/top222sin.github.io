// ============================================================
// control_demo L4：事件触发控制 ETC ⭐（网络控制/带宽受限）
// 纯 C++17。网络控制里"每步都传"浪费带宽。ETC 只在状态偏离上次采样
//   足够大时才重算并发送控制：触发 ‖x(k)-x(t_k)‖ > σ·‖x(t_k)‖。
//   本 demo 故意把闭环设计成严格收缩映射 A-BK=D（对角 0.9/0.85），
//   保证保持控制期间仍稳定；验证 ETC 终态稳定且传输次数仅为周期的 ~15%。
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
static Mat matsub(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]-B[i][j]; return C;
}
static Vec matvec(const Mat&A,const Vec&x){
    int n=(int)A.size(), m=(int)A[0].size(); Vec y(n,0.0);
    for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; } return y;
}
static double matnormF(const Mat&A){ double s=0.0; for(auto&r:A) for(double v:r) s+=v*v; return sqrt(s); }
static double vecnorm(const Vec&x){ double s=0.0; for(double v:x) s+=v*v; return sqrt(s); }
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

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool ok=e<t; printf("  %-46s err=%.3e tol=%.0e  %s\n",n,e,t,ok?"PASS":"FAIL"); ok?++pass:++fail; };
    printf("=== L4：事件触发控制 ETC（网络带宽受限）===\n");

    Mat A={{0.7,0.0},{0.0,0.6}};
    Mat B={{1.0,0.3},{0.2,1.0}};
    Mat D={{0.9,0.0},{0.0,0.85}};
    Mat K=matmul(inv(B), matsub(A, D));             // K=B^{-1}(A-D) -> A-BK=D(严格收缩)
    Mat Acl=matsub(A, matmul(B,K));
    double ec=matnormF(matsub(Acl, D));
    printf("  [demo] A-BK 与对角收缩 D 残差=%.2e (应≈0)\n", ec);
    chk("闭环=严格收缩映射", ec, 1e-12);

    const int N=400; Vec x0={1.0,1.0};
    auto sim=[&](double sigma)->pair<Vec,int>{
        Vec x=x0, xs=x0; int nev=1;
        Vec u={ -K[0][0]*xs[0]-K[0][1]*xs[1], -K[1][0]*xs[0]-K[1][1]*xs[1] };
        for(int k=0;k<N;++k){
            Vec xn={ A[0][0]*x[0]+A[0][1]*x[1]+B[0][0]*u[0]+B[0][1]*u[1],
                     A[1][0]*x[0]+A[1][1]*x[1]+B[1][0]*u[0]+B[1][1]*u[1] };
            x=xn;
            double ex=x[0]-xs[0], ey=x[1]-xs[1];
            double err=sqrt(ex*ex+ey*ey);
            double thr=sigma*sqrt(xs[0]*xs[0]+xs[1]*xs[1]);
            if(err>thr || k==0){
                xs=x;
                u={ -K[0][0]*xs[0]-K[0][1]*xs[1], -K[1][0]*xs[0]-K[1][1]*xs[1] };
                ++nev;
            }
        }
        return {x,nev};
    };
    auto per=sim(0.0);
    double sigma=0.3;
    auto res=sim(sigma);
    Vec x_etc=res.first; int nev=res.second;
    int nper=N+1;
    printf("  [demo] 周期终态‖x‖=%.3e, ETC(σ=%.1f)终态‖x‖=%.3e, 事件=%d/%d (节省%.0f%%)\n",
           vecnorm(per.first), sigma, vecnorm(x_etc), nev, nper, 100.0*(1.0-nev/(double)nper));
    chk("ETC 稳定(终态小)", vecnorm(x_etc), 0.05);
    chk("ETC 通信/计算次数 < 周期", nev-0.8*nper, 1e-9);
    chk("ETC 节省 >70% 传输(事件稀疏)", nev-0.3*nper, 1e-9);

    printf("  [结论] ETC 把控制更新/传输从\"每步\"降为\"按需\"；稳定性靠触发阈值\n");
    printf("          与收缩闭环共同保证。σ 越大越省带宽但收敛越慢，需在二者间折中。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
