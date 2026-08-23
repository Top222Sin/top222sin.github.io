// ============================================================
// pid_demo L4：MIMO PID（2x2 耦合对象 + 静态解耦 + 各路 PID）⭐（多变量回路）
// 纯 C++17。单回路 PID 直接套多变量对象会"按下葫芦浮起瓢": 调好一路把另一路带跑。
//   先用 DC 增益 K=(I-A)^{-1}B 的逆 D=K^{-1} 作预补偿器, 使 D·K=I(有效对象对角化),
//   再在每条对角回路上跑普通 PID -> 各路独立整定, 动态交叉耦合被压到 ~0。
// 编译: g++ -O3 -std=c++17 pid_l4_mimo.cpp -o t && ./t
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
static Vec matvec(const Mat&A,const Vec&x){
    int n=(int)A.size(), m=(int)A[0].size(); Vec y(n,0.0);
    for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; }
    return y;
}
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
static double matnormF(const Mat&A){ double s=0.0; for(auto&row:A) for(double v:row) s+=v*v; return sqrt(s); }

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* nm,double err,double tol){
        bool ok=err<tol; printf("  %-34s err=%.3e tol=%.0e  %s\n",nm,err,tol,ok?"PASS":"FAIL");
        ok?++pass:++fail;
    };
    printf("=== L4：MIMO PID（2x2 解耦 + 各路 PID）===\n");

    double dt=0.05, T=15.0;
    double tau1=1.0, tau2=2.0;
    Mat K={{1.0,0.3},{0.5,1.0}};          // 稳态增益矩阵(含交叉耦合)
    double a11=exp(-dt/tau1), a22=exp(-dt/tau2);
    double b11=K[0][0]*(1-a11), b12=K[0][1]*(1-a11);
    double b21=K[1][0]*(1-a22), b22=K[1][1]*(1-a22);
    Mat A={{a11,0.0},{0.0,a22}}, B={{b11,b12},{b21,b22}};
    Mat Kss=matmul(inv(matsub(eyeM(2),A)), B);   // 应 == K
    Mat D=inv(Kss);                              // 解耦器

    chk("DC增益 Kss≈K(残差小)", matnormF(matsub(Kss,K)), 1e-9);
    Mat DK=matmul(D,Kss);
    chk("D*Kss==I(解耦)", matnormF(matsub(DK,eyeM(2))), 1e-12);

    double Kp=1.0,Ki=0.5,Kd=0.2;
    auto sim_mimo=[&](bool decouple,const Vec&r,Vec&y2_traj,double&y1ss,double&y2ss){
        int N=(int)(T/dt); Vec y={0.0,0.0}; Vec e_prev={0.0,0.0}, Iacc={0.0,0.0};
        y2_traj.clear();
        for(int k=0;k<N;++k){
            Vec e={r[0]-y[0], r[1]-y[1]};
            Iacc={Iacc[0]+e[0]*dt, Iacc[1]+e[1]*dt};
            Vec der={(e[0]-e_prev[0])/dt, (e[1]-e_prev[1])/dt};
            Vec u_loop={Kp*e[0]+Ki*Iacc[0]+Kd*der[0], Kp*e[1]+Ki*Iacc[1]+Kd*der[1]};
            Vec u = decouple? matvec(D,u_loop) : u_loop;
            Vec yn={
                A[0][0]*y[0]+B[0][0]*u[0]+B[0][1]*u[1],
                A[1][0]*y[0]+A[1][1]*y[1]+B[1][0]*u[0]+B[1][1]*u[1]};
            y=yn; e_prev=e; y2_traj.push_back(y[1]);
        }
        y1ss=y[0]; y2ss=y[1];
    };

    Vec r={1.0,0.0};
    Vec y2_nd; double y1_nd,y2s_nd; sim_mimo(false,r,y2_nd,y1_nd,y2s_nd);
    Vec y2_d;  double y1_d, y2s_d;  sim_mimo(true, r,y2_d, y1_d, y2s_d);
    double peak_nd=0.0; for(double v:y2_nd) if(fabs(v)>peak_nd) peak_nd=fabs(v);
    double peak_d =0.0; for(double v:y2_d ) if(fabs(v)>peak_d ) peak_d =fabs(v);
    printf("  无解耦 r=[1,0] -> y1_ss=%.4f y2_ss=%.4f  瞬态y2峰值=%.4f\n", y1_nd,y2s_nd,peak_nd);
    printf("  解耦   r=[1,0] -> y1_ss=%.4f y2_ss=%.4f  瞬态y2峰值=%.4f\n", y1_d, y2s_d, peak_d);
    chk("无解耦 y1_ss≈1", fabs(y1_nd-1.0), 0.02);
    chk("无解耦 瞬态交叉耦合可见(峰值>0.03)", max(0.0, 0.03-peak_nd), 1e-6);
    chk("解耦后 y1_ss≈1", fabs(y1_d-1.0), 0.05);
    chk("解耦后 瞬态y2峰值 << 无解耦", max(0.0, peak_d-0.5*peak_nd), 1e-6);
    chk("解耦后 y2_ss≈0(交叉消除)", fabs(y2s_d), 0.02);

    printf("  [结论] 解耦器 D=K^{-1} 把 2x2 对象在直流严格对角化, 各路按 SISO 整定;\n");
    printf("           瞬态交叉耦合峰值 %.4f -> %.4f(≈0), 与 control_demo L4.5 同构。\n", peak_nd, peak_d);
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
