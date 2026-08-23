// jac_l3_dynamics.cpp  —  动力学更多模型：解析雅可比 A=∂f/∂x, B=∂f/∂u
//
// 运控（L3.2）只用了“独轮车”模型。这里把动力学模型扩展到三种，
// 对每一种都解析地写出 A、B 雅可比，并与有限差分对拍：
//   1) 双积分器       x=[p,v],   u=[a]
//   2) 自行车/车辆    x=[x,y,θ], u=[v,φ]      (A 含 -v sinθ 等，B 含 sec²φ)
//   3) 车摆 cart-pole x=[x,vx,θ,θ̇], u=[F]     (非线性最强，A/B 解析求导)
// 最后用自行车模型跑一步 MPC 最优控制 + 灵敏度 S=∂u*/∂x（与 L3.2 同框架）。
//
// 仅用 C++17 标准库。

#include <cstdio>
#include <cmath>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

static Mat matmul(const Mat& A, const Mat& B) {
    int n=(int)A.size(), m=(int)B[0].size(), k=(int)B.size();
    Mat C(n, Vec(m,0.0));
    for (int i=0;i<n;++i) for(int p=0;p<k;++p){double a=A[i][p]; if(a) for(int j=0;j<m;++j) C[i][j]+=a*B[p][j];}
    return C;
}
static Mat transpose(const Mat& A){int n=(int)A.size(),m=(int)A[0].size();Mat T(m,Vec(n));for(int i=0;i<n;++i)for(int j=0;j<m;++j)T[j][i]=A[i][j];return T;}
static Mat add(const Mat&A,const Mat&B){Mat C=A;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)C[i][j]+=B[i][j];return C;}
static Mat sub(const Mat&A,const Mat&B){Mat C=A;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)C[i][j]-=B[i][j];return C;}
static double max_err(const Mat&A,const Mat&B){double m=0;for(size_t i=0;i<A.size();++i)for(size_t j=0;j<A[0].size();++j)m=std::fmax(m,std::fabs(A[i][j]-B[i][j]));return m;}

// ---------------- 模型 1：双积分器 ----------------
static Vec f_di(const Vec&x,const Vec&u){return {x[1], u[0]};}
static void AB_di(const Vec&x,const Vec&u,Mat&A,Mat&B){
    A={{0,1},{0,0}}; B={{0},{1}};
}
// ---------------- 模型 2：自行车 ----------------
static const double Lwb=1.0;
static Vec f_bike(const Vec&x,const Vec&u){
    double th=x[2],v=u[0],phi=u[1];
    return {v*std::cos(th), v*std::sin(th), (v/Lwb)*std::tan(phi)};
}
static void AB_bike(const Vec&x,const Vec&u,Mat&A,Mat&B){
    double th=x[2],v=u[0],phi=u[1];
    A={{0,0,-v*std::sin(th)},{0,0,v*std::cos(th)},{0,0,0}};
    B={{std::cos(th),0},{std::sin(th),0},{(1.0/Lwb)*std::tan(phi),(v/Lwb)*(1.0/(std::cos(phi)*std::cos(phi)))}};
}
// ---------------- 模型 3：车摆 cart-pole ----------------
static const double Mm=1.0, mm=0.2, ll=0.5, gg=9.81;
static Vec f_cart(const Vec&x,const Vec&u){
    double th=x[2],thd=x[3],F=u[0];
    double den=Mm+mm*std::sin(th)*std::sin(th);
    double atheta=((Mm+mm)*gg*std::sin(th)-F*std::cos(th)-mm*ll*thd*thd*std::sin(th)*std::cos(th))/(ll*den);
    double ax=(F-mm*ll*std::cos(th)*atheta+mm*ll*thd*thd*std::sin(th))/(Mm+mm);
    return {x[1], ax, thd, atheta};
}
static void AB_cart(const Vec&x,const Vec&u,Mat&A,Mat&B){
    double th=x[2],thd=x[3],F=u[0];
    double den=Mm+mm*std::sin(th)*std::sin(th);
    double denp=2*mm*std::sin(th)*std::cos(th);
    double num=(Mm+mm)*gg*std::sin(th)-F*std::cos(th)-mm*ll*thd*thd*std::sin(th)*std::cos(th);
    double dnum=(Mm+mm)*gg*std::cos(th)+F*std::sin(th)-mm*ll*thd*thd*(std::cos(th)*std::cos(th)-std::sin(th)*std::sin(th));
    double da_dth=(dnum*den-num*denp)/(ll*den*den);
    double da_dthd=(-2*mm*thd*std::sin(th)*std::cos(th))/den;
    double da_dF=(-std::cos(th))/(ll*den);
    double atheta=num/(ll*den);
    double dax_dth=(mm*ll*std::sin(th)*atheta-mm*ll*std::cos(th)*da_dth+mm*ll*thd*thd*std::cos(th))/(Mm+mm);
    double dax_dthd=(-mm*ll*std::cos(th)*da_dthd+2*mm*ll*thd*std::sin(th))/(Mm+mm);
    double dax_dF=(1-mm*ll*std::cos(th)*da_dF)/(Mm+mm);
    A={{0,1,0,0},
       {0,0,dax_dth,dax_dthd},
       {0,0,0,1},
       {0,0,da_dth,da_dthd}};
    B={{0},{dax_dF},{0},{da_dF}};
}

// 有限差分求 A,B（central difference）
static void fd_AB(Vec(*f)(const Vec&,const Vec&),const Vec&x,const Vec&u,int n,int m,Mat&A,Mat&B){
    A=Mat(n,Vec(n,0.0)); B=Mat(n,Vec(m,0.0));
    Vec f0=f(x,u); double h=1e-6;
    for(int i=0;i<n;++i){
        Vec xp=x; xp[i]+=h; Vec xm=x; xm[i]-=h;
        Vec fp=f(xp,u), fm=f(xm,u);
        for(int a=0;a<n;++a) A[a][i]=(fp[a]-fm[a])/(2*h);
    }
    for(int j=0;j<m;++j){
        Vec up=u; up[j]+=h; Vec um=u; um[j]-=h;
        Vec fp=f(x,up), fm=f(x,um);
        for(int a=0;a<n;++a) B[a][j]=(fp[a]-fm[a])/(2*h);
    }
}

// 自行车 MPC（一步最优控制）+ 灵敏度
static Mat inv2(const Mat&M){
    double a=M[0][0],b=M[0][1],c=M[1][0],d=M[1][1];
    double det=a*d-b*c; return {{d/det,-b/det},{-c/det,a/det}};
}
static Vec mpc_u(const Mat&A,const Mat&B,const Vec&x,const Vec&xg,const Mat&Q,double rho){
    Mat Bt=transpose(B);
    Mat M_=add(matmul(Bt,matmul(Q,B)),{{rho,0},{0,rho}});
    Mat Minv=inv2(M_);
    Mat xc{{x[0]},{x[1]},{x[2]}}, xgc{{xg[0]},{xg[1]},{xg[2]}};
    Mat rhs=matmul(Bt,matmul(Q,sub(matmul(A,xc),xgc)));
    return {-Minv[0][0]*rhs[0][0]-Minv[0][1]*rhs[1][0], -Minv[1][0]*rhs[0][0]-Minv[1][1]*rhs[1][0]};
}
static Mat mpc_S(const Mat&A,const Mat&B,const Mat&Q,double rho){
    Mat Bt=transpose(B);
    Mat M_=add(matmul(Bt,matmul(Q,B)),{{rho,0},{0,rho}});
    Mat Minv=inv2(M_);
    Mat ret=matmul(matmul(Minv,Bt),matmul(Q,A));
    for(int a=0;a<2;++a) for(int k=0;k<3;++k) ret[a][k]=-ret[a][k];  // S=-M⁻¹BᵀQA
    return ret;
}

int main(){
    int pass=0,fail=0;
    auto chk=[&](const char*name,double e,double tol){bool ok=e<tol;printf("  %-22s err=%.2e  %s\n",name,e,ok?"PASS":"FAIL");ok?++pass:++fail;};
    printf("=== 多动力学模型：解析 A/B 雅可比 ===\n");

    // 逐一验证三种模型的解析 A/B
    {
        Vec x{0.3,-0.2},u{0.5}; Mat A,B,Af,Bf; AB_di(x,u,A,B); fd_AB(f_di,x,u,2,1,Af,Bf);
        chk("double-integrator A,B", std::fmax(max_err(A,Af),max_err(B,Bf)), 1e-5);
    }
    {
        Vec x{0.2,0.1,0.15},u{0.6,0.1}; Mat A,B,Af,Bf; AB_bike(x,u,A,B); fd_AB(f_bike,x,u,3,2,Af,Bf);
        chk("bicycle A,B", std::fmax(max_err(A,Af),max_err(B,Bf)), 1e-5);
    }
    {
        Vec x{0.1,-0.2,0.15,0.3},u{0.5}; Mat A,B,Af,Bf; AB_cart(x,u,A,B); fd_AB(f_cart,x,u,4,1,Af,Bf);
        chk("cart-pole A,B", std::fmax(max_err(A,Af),max_err(B,Bf)), 1e-5);
    }

    // 自行车 MPC + 灵敏度
    {
        Vec x0{0,0,0}, u0{0.6,0.1}, xg{2.0,1.0,0.5};
        Mat A,B; AB_bike(x0,u0,A,B);
        Mat Q{{1,0,0},{0,1,0},{0,0,1}};
        double rho=0.1;
        Vec ustar=mpc_u(A,B,x0,xg,Q,rho);
        Mat S=mpc_S(A,B,Q,rho);
        // S 的有限差分(固定 A,B, 与解析 S=-M⁻¹BᵀQA 同口径)
        Mat Sfd(2,Vec(3,0.0)); double h=1e-6;
        for(int k=0;k<3;++k){
            Vec xp=x0; xp[k]+=h;
            Vec uk=mpc_u(A,B,xp,xg,Q,rho);
            for(int a=0;a<2;++a) Sfd[a][k]=(uk[a]-ustar[a])/h;
        }
        chk("bicycle MPC sens S", max_err(S,Sfd), 1e-5);
        printf("    MPC u* = [%.4f, %.4f]\n", ustar[0], ustar[1]);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
