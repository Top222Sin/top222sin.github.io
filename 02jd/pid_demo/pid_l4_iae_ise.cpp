// ============================================================
// pid_demo L4：IAE / ISE / ITAE / ITSE 性能指标 ⭐（整定质量量化）
// 纯 C++17。整定好坏不能只看"稳不稳", 要用量化指标:
//   IAE=∫|e|dt  ISE=∫e²dt  ITAE=∫t|e|dt  ITSE=∫t·e²dt
// 离散:  Σ |e|·h / Σ e²·h / Σ k·h·|e|·h / Σ k·h·e²·h
// 演示: 优调 PID 的 IAE/ISE 明显低于激进 ZN 高 Kp 整定; 定常误差时积分有闭式自洽值。
// 编译: g++ -O3 -std=c++17 pid_l4_iae_ise.cpp -o t && ./t
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

static double iae(const Vec&e,double dt){ double s=0.0; for(double v:e) s+=fabs(v)*dt; return s; }
static double ise(const Vec&e,double dt){ double s=0.0; for(double v:e) s+=v*v*dt; return s; }
static double itae(const Vec&e,double dt){ double s=0.0; int N=(int)e.size(); for(int k=0;k<N;++k) s+=(k*dt)*fabs(e[k])*dt; return s; }
static double itse(const Vec&e,double dt){ double s=0.0; int N=(int)e.size(); for(int k=0;k<N;++k) s+=(k*dt)*e[k]*e[k]*dt; return s; }

static void tustin_ss(const Mat&Ac,const Mat&Bc,const Vec&Cc,double dt,
                      Mat&Ad,Mat&Bd,Vec&Cc2){
    int n=(int)Ac.size(); Mat I=eyeM(n);
    Mat M=matsub(I, scalemat(dt/2.0, Ac)); Mat Minv=inv(M);
    Ad=matmul(Minv, matadd(I, scalemat(dt/2.0, Ac)));
    Bd=matmul(Minv, scalemat(dt, Bc)); Cc2=Cc;
}

// 二阶对象上的 PID 闭环仿真(返回 y,e)
static void sim_pid_2ss(const Mat&Ad,const Mat&Bd,const Vec&Cc,double dt,double T,
                        double Kp,double Ki,double Kd,double r,Vec&y,Vec&e){
    int N=(int)(T/dt); int n=(int)Ad.size();
    Vec x(n,0.0); y.clear(); e.clear();
    double Iacc=0.0, ep=0.0;
    for(int k=0;k<N;++k){
        double yk=0.0; for(int i=0;i<n;++i) yk+=Cc[i]*x[i];
        double ek=r-yk; e.push_back(ek); Iacc+=ek*dt; double der=(ek-ep)/dt;
        double u=Kp*ek+Ki*Iacc+Kd*der;
        if(u>1e9)u=1e9; if(u<-1e9)u=-1e9;
        Vec xn(n,0.0);
        for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<n;++j) s+=Ad[i][j]*x[j]; xn[i]=s; }
        for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<(int)Bd[0].size();++j) s+=Bd[i][j]*u; xn[i]+=s; }
        x=xn; ep=ek; y.push_back(yk);
    }
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* nm,double err,double tol){
        bool ok=err<tol; printf("  %-34s err=%.3e tol=%.0e  %s\n",nm,err,tol,ok?"PASS":"FAIL");
        ok?++pass:++fail;
    };
    printf("=== L4：IAE/ISE/ITAE/ITSE 性能指标 ===\n");

    double dt=0.02, T=20.0, r=1.0;
    Mat Ac={{0.0,1.0},{-1.0,-0.8}}; Mat Bc={{0.0},{1.0}}; Vec Cc={1.0,0.0};
    Mat Ad,Bd; Vec Cc2; tustin_ss(Ac,Bc,Cc,dt,Ad,Bd,Cc2);

    // 自洽: 定常误差 e=0.5 全时程 -> IAE=0.5*T, ISE=0.25*T
    int N=(int)(T/dt); Vec e_const(N,0.5);
    double iae_c=iae(e_const,dt), ise_c=ise(e_const,dt);
    printf("  定常误差 e=0.5: IAE=%.4f (应=%.4f)  ISE=%.4f (应=%.4f)\n", iae_c,0.5*T,ise_c,0.25*T);
    chk("定常误差 IAE=0.5*T 自洽", fabs(iae_c-0.5*T), 1e-6);
    chk("定常误差 ISE=0.25*T 自洽", fabs(ise_c-0.25*T), 1e-6);

    // 两种整定: 激进(ZN高Kp,大超调) vs 优调(缓和)
    Vec yA,eA,yB,eB;
    sim_pid_2ss(Ad,Bd,Cc2,dt,T,3.0,2.0,0.2,r,yA,eA);   // 激进
    sim_pid_2ss(Ad,Bd,Cc2,dt,T,1.4,0.8,0.5,r,yB,eB);   // 优调
    double iaeA=iae(eA,dt), iaeB=iae(eB,dt);
    double iseA=ise(eA,dt),  iseB=ise(eB,dt);
    double itaeA=itae(eA,dt), itaeB=itae(eB,dt);
    double osA=0.0; for(double v:yA) if(v-r>osA) osA=v-r;
    double osB=0.0; for(double v:yB) if(v-r>osB) osB=v-r;
    printf("  激进PID(高Kp)  IAE=%.4f ISE=%.4f ITAE=%.2f 超调=%.4f\n", iaeA,iseA,itaeA,osA);
    printf("  优调PID        IAE=%.4f ISE=%.4f ITAE=%.2f 超调=%.4f\n", iaeB,iseB,itaeB,osB);
    chk("优调 IAE < 激进 IAE", max(0.0, iaeB-iaeA), 1e-3);
    chk("优调 ISE < 激进 ISE", max(0.0, iseB-iseA), 1e-3);
    chk("超调与IAE同向(激进更大)", (osA>osB && iaeA>iaeB)?0.0:1.0, 1e-6);
    chk("优调稳态误差<0.02", fabs(eB.back()), 0.02);

    printf("  [结论] IAE/ISE 把\"整定质量\"变成一个数: 优调 IAE %.4f<激进 %.4f, ISE 同理;\n", iaeB,iaeA);
    printf("           ITAE/ITSE 更惩罚尾差(优调 ITAE %.2f<<激进 %.2f)。整定时用指标打分即可自动化寻优。\n", itaeB,itaeA);
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
