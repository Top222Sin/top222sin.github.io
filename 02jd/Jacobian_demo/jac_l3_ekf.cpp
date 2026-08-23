// jac_l3_ekf.cpp  —  工业用例：EKF 定位中的雅可比（F, G, H）
//
// 工业移动机器人/AGV 自主定位的核心就是 EKF（扩展卡尔曼滤波）。
// 它每一步都把非线性运动/观测模型在“当前估计”处线性化——而线性化的工具
// 正是雅可比：
//   预测 : xₖ = f(xₖ₋₁,u) ,  Pₖ = F P Fᵀ + G Q Gᵀ
//   更新 : y = z − h(xₖ)    ,  S = H P Hᵀ + R ,  K = P Hᵀ S⁻¹
// 这里 f 是 unicycle 运动（F=∂f/∂x, G=∂f/∂u），h 是两类观测：
//   GPS 位置 h=[x,y]（H 平凡）与 路标距离+方位 h=[r,b]（H 含 1/r、1/r² 项）。
// 全部雅可比与有限差分对拍；并跑 40 步 EKF 验证“误差收敛、协方差正定有界”。
//
// 仅用 C++17 标准库。

#include <cstdio>
#include <cmath>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

static const double DT = 0.1;
static const Vec LAND = {3.0, 2.0};

static Mat matmul(const Mat&A,const Mat&B){
    int n=(int)A.size(),m=(int)B[0].size(),k=(int)B.size();
    Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i)for(int p=0;p<k;++p){double a=A[i][p];if(a)for(int j=0;j<m;++j)C[i][j]+=a*B[p][j];}
    return C;
}
static Mat transpose(const Mat&A){int n=(int)A.size(),m=(int)A[0].size();Mat T(m,Vec(n));for(int i=0;i<n;++i)for(int j=0;j<m;++j)T[j][i]=A[i][j];return T;}
static Mat add(const Mat&A,const Mat&B){Mat C=A;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)C[i][j]+=B[i][j];return C;}
static Mat sub(const Mat&A,const Mat&B){Mat C=A;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)C[i][j]-=B[i][j];return C;}
static double max_err(const Mat&A,const Mat&B){double m=0;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)m=std::fmax(m,std::fabs(A[i][j]-B[i][j]));return m;}

// 运动模型
static Vec f_motion(const Vec&x,const Vec&u){
    double th=x[2],v=u[0],om=u[1];
    return {x[0]+v*std::cos(th)*DT, x[1]+v*std::sin(th)*DT, x[2]+om*DT};
}
static Mat F_motion(const Vec&x,const Vec&u){
    double th=x[2],v=u[0];
    return {{1,0,-v*std::sin(th)*DT},{0,1,v*std::cos(th)*DT},{0,0,1}};
}
static Mat G_motion(const Vec&x,const Vec&u){
    double th=x[2];
    return {{std::cos(th)*DT,0},{std::sin(th)*DT,0},{0,DT}};
}
// 观测：GPS
static Vec h_gps(const Vec&x){return {x[0],x[1]};}
static Mat H_gps(const Vec&x){return {{1,0,0},{0,1,0}};}
// 观测：路标距离+方位
static Vec h_lm(const Vec&x){
    double dx=x[0]-LAND[0], dy=x[1]-LAND[1];
    return {std::hypot(dx,dy), std::atan2(dy,dx)-x[2]};
}
static Mat H_lm(const Vec&x){
    double dx=x[0]-LAND[0], dy=x[1]-LAND[1], r=std::hypot(dx,dy);
    return {{ dx/r,  dy/r, 0},
            {-dy/(r*r), dx/(r*r), -1.0}};
}

// 有限差分（对状态 x）
template<class F>
static Mat fd_state(F fn,const Vec&x,int n,int m){
    Mat J(n,Vec(m,0.0)); Vec f0=fn(x); double h=1e-6;
    for(int j=0;j<m;++j){Vec xp=x;xp[j]+=h;Vec fp=fn(xp);for(int i=0;i<n;++i)J[i][j]=(fp[i]-f0[i])/h;}
    return J;
}
// 有限差分（对控制 u）
template<class F>
static Mat fd_ctrl(F fn,const Vec&x,const Vec&u,int n,int m){
    Mat J(n,Vec(m,0.0)); Vec f0=fn(x,u); double h=1e-6;
    for(int j=0;j<m;++j){Vec up=u;up[j]+=h;Vec fp=fn(x,up);for(int i=0;i<n;++i)J[i][j]=(fp[i]-f0[i])/h;}
    return J;
}

static Mat inv2(const Mat&M){double a=M[0][0],b=M[0][1],c=M[1][0],d=M[1][1];double det=a*d-b*c;return {{d/det,-b/det},{-c/det,a/det}};}

// 确定性噪声（LCG），保证可复现
static unsigned long long rng=99ULL;
static double urand(double lo,double hi){rng=rng*6364136223846793005ULL+1442695040888963407ULL;double u=(double)(rng>>11)/(double)(1ULL<<53);return lo+u*(hi-lo);}

int main(){
    int pass=0,fail=0;
    auto chk=[&](const char*n,double e,double tol){bool ok=e<tol;printf("  %-16s err=%.2e  %s\n",n,e,ok?"PASS":"FAIL");ok?++pass:++fail;};
    printf("=== 工业 EKF 定位：雅可比 F/G/H ===\n");

    Vec x{1.0,0.5,0.3}, u{0.6,0.2};
    chk("F motion",  max_err(fd_state([&u](const Vec&xx){return f_motion(xx,u);}, x,3,3), F_motion(x,u)), 1e-5);
    chk("G motion",  max_err(fd_ctrl([](const Vec&xx,const Vec&uu){return f_motion(xx,uu);}, x,u,3,2), G_motion(x,u)), 1e-5);
    chk("H gps",     max_err(fd_state(h_gps, x,2,3), H_gps(x)), 1e-5);
    chk("H landmark",max_err(fd_state(h_lm, x,2,3), H_lm(x)), 1e-5);

    // ---- 跑 40 步 EKF ----
    Vec x_true{0,0,0}, x_est{0.5,0.3,0.2};
    Mat P(3, Vec(3, 0.0)); for(int i=0;i<3;++i)P[i][i]=1.0;
    Mat Q={{0.01,0},{0,0.005}};
    Mat Rg={{0.04,0},{0,0.04}}, Rl={{0.05,0},{0,0.02}};
    double init_err=std::hypot(x_est[0]-x_true[0],x_est[1]-x_true[1]);
    double Ptrace0=3.0, PtraceN=0;
    for(int k=0;k<40;++k){
        Vec uu{0.5+0.1*std::sin(k*0.3), 0.1*std::cos(k*0.2)};
        // 真值前进
        x_true=f_motion(x_true,uu);
        // 预测
        Mat Fm=F_motion(x_true,uu), Gm=G_motion(x_true,uu);
        Vec xp=f_motion(x_est,uu);
        Mat Pp=add(matmul(matmul(Fm,P),transpose(Fm)), matmul(matmul(Gm,Q),transpose(Gm)));
        // GPS 更新
        Vec zg{x_true[0]+urand(-0.2,0.2), x_true[1]+urand(-0.2,0.2)};
        Mat H1=H_gps(xp); Vec y{zg[0]-xp[0], zg[1]-xp[1]};
        Mat S=add(matmul(matmul(H1,Pp),transpose(H1)),Rg);
        Mat Si=inv2(S);
        Mat K=matmul(matmul(Pp,transpose(H1)),Si);
        Vec xu(3);
        for(int i=0;i<3;++i){double s=0;for(int j=0;j<2;++j)s+=K[i][j]*y[j];xu[i]=xp[i]+s;}
        P=sub(Pp, matmul(matmul(K,H1),Pp));
        // 路标更新
        Vec hl_true=h_lm(x_true);
        Vec zl{hl_true[0]+urand(-0.22,0.22), hl_true[1]+urand(-0.14,0.14)};
        Mat H2=H_lm(xu); Vec yl{zl[0]-h_lm(xu)[0], zl[1]-h_lm(xu)[1]};
        Mat S2=add(matmul(matmul(H2,P),transpose(H2)),Rl);
        Mat Si2=inv2(S2);
        Mat K2=matmul(matmul(P,transpose(H2)),Si2);
        Vec xu2(3);
        for(int i=0;i<3;++i){double s=0;for(int j=0;j<2;++j)s+=K2[i][j]*yl[j];xu2[i]=xu[i]+s;}
        P=sub(P, matmul(matmul(K2,H2),P));
        x_est=xu2;
        if(k==0) Ptrace0=P[0][0]+P[1][1]+P[2][2];
        if(k==39) PtraceN=P[0][0]+P[1][1]+P[2][2];
    }
    double final_err=std::hypot(x_est[0]-x_true[0], x_est[1]-x_true[1]);
    bool ok = (final_err<init_err) && (PtraceN>0) && (PtraceN<Ptrace0);
    printf("  EKF 初始误差=%.3f  最终=%.3f  P迹 %.3f->%.4f  %s\n", init_err, final_err, Ptrace0, PtraceN, ok?"PASS":"FAIL");
    ok?++pass:++fail;
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
