// ============================================================
// LS_demo L4.3 软测量与预测性维护
// 纯 C++17。① 软测量: 温度/压力/流量 → 产品质量多元回归(归一化特征)
// ② 预测性维护: 振动退化趋势二次拟合(归一化时间) → 外推 RUL
// ⚠ 坑(开发踩过): 特征必须归一化(裸温度 350~400 会让 AᵀA 病态);
//   阈值必须设在观测区间之外(否则 RUL 为负)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 43;
static double urand(){ _seed=_seed*6364136223846793005ULL+1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

static void solve4(vector<vector<double>> A, vector<double> b, vector<double>& x) {
    int n=4;
    for(int c=0;c<n;++c){ int p=c;
        for(int i=c;i<n;++i) if(fabs(A[i][c])>fabs(A[p][c])) p=i;
        swap(A[c],A[p]); swap(b[c],b[p]);
        for(int i=0;i<n;++i) if(i!=c){ double f=A[i][c]/A[c][c];
            for(int j=c;j<n;++j) A[i][j]-=f*A[c][j]; b[i]-=f*b[c]; } }
    x.assign(n,0);
    for(int i=0;i<n;++i) x[i]=b[i]/A[i][i];
}

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-38s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.3 软测量 + 预测性维护 ===\n");

    // ---- ① 软测量 ----
    double bt[4]={0.5,2.0,-1.5,0.8};
    int n=80;
    vector<vector<double>> X; vector<double> y;
    for(int i=0;i<n;++i){
        double t=350+urand()*50, p=1+urand()*2, f=10+urand()*20;
        X.push_back({1.0,(t-350)/50,(p-1)/2,(f-10)/20});   // 归一化特征!
        y.push_back(bt[0]+bt[1]*(t-350)/50+bt[2]*(p-1)/2+bt[3]*(f-10)/20+gauss()*0.05);
    }
    vector<vector<double>> XtA(4, vector<double>(4,0));
    vector<double> Xty(4,0);
    for(int i=0;i<n;++i) for(int a=0;a<4;++a){ Xty[a]+=X[i][a]*y[i];
        for(int b=0;b<4;++b) XtA[a][b]+=X[i][a]*X[i][b]; }
    vector<double> beta;
    solve4(XtA, Xty, beta);
    double berr=0;
    for(int i=0;i<4;++i) berr=fmax(berr,fabs(beta[i]-bt[i]));
    chk("软测量回归系数", berr, 0.05);
    _seed=431;
    double e2=0;
    for(int i=0;i<30;++i){
        double t=350+urand()*50,p=1+urand()*2,f=10+urand()*20;
        double yv=bt[0]+bt[1]*(t-350)/50+bt[2]*(p-1)/2+bt[3]*(f-10)/20+gauss()*0.05;
        double yp=beta[0]+beta[1]*(t-350)/50+beta[2]*(p-1)/2+beta[3]*(f-10)/20;
        e2+=(yp-yv)*(yp-yv);
    }
    double rmse=sqrt(e2/30);
    chk("软测量预测 RMSE < 0.1", rmse, 0.1);
    printf("  [demo] 软测量: β_err=%.3f, 预测 RMSE=%.3f(辛烷值)\n", berr, rmse);

    // ---- ② 预测性维护 ----
    _seed=432;
    const double th=15.0;
    vector<double> ts, vs;
    for(int k=0;k<50;++k){ double t=k*10.0; ts.push_back(t);
        vs.push_back(0.02*t+1.0+1e-5*t*t+gauss()*0.02); }
    vector<vector<double>> A3(3, vector<double>(3,0));
    vector<double> b3(3,0);
    for(int i=0;i<50;++i){ double u=(ts[i]-245)/245;
        double r[3]={1,u,u*u};
        for(int a=0;a<3;++a){ b3[a]+=r[a]*vs[i];
            for(int b=0;b<3;++b) A3[a][b]+=r[a]*r[b]; } }
    for(int c=0;c<3;++c){ int p=c;
        for(int i=c;i<3;++i) if(fabs(A3[i][c])>fabs(A3[p][c])) p=i;
        swap(A3[c],A3[p]); swap(b3[c],b3[p]);
        for(int i=0;i<3;++i) if(i!=c){ double f=A3[i][c]/A3[c][c];
            for(int j=c;j<3;++j) A3[i][j]-=f*A3[c][j]; b3[i]-=f*b3[c]; } }
    double c0=b3[0]/A3[0][0], c1=b3[1]/A3[1][1], c2=b3[2]/A3[2][2];
    double t=490.0;
    while(c0+c1*(t-245)/245+c2*((t-245)/245)*((t-245)/245)<th && t<2000) t+=0.5;
    double rul_hat=t-490.0;
    double disc=sqrt(0.02*0.02+4*1e-5*14);
    double rul_true=(-0.02+disc)/(2e-5)-490.0;
    chk("RUL 预测误差 < 50h", fabs(rul_hat-rul_true), 50.0);
    printf("  [demo] RUL: 预测=%.0fh vs 真值=%.0fh(阈值 %.0f)\n", rul_hat, rul_true, th);
    printf("  [结论] 软测量=多元线性回归+特征归一化;预测性维护=趋势拟合+外推;\n");
    printf("        过程建模(PID 整定/优化)见 control_demo 与 pid_demo。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
