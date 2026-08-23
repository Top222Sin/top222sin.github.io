// jac_l3_motion.cpp
// ============================================================================
// L3.2 运控：离散动力学雅可比 A/B 与 MPC 灵敏度
// ----------------------------------------------------------------------------
// 运动控制(MPC/ LQR)的核心是把非线性系统在其标称轨迹上"线性化"：
//     x_{k+1} = f(x_k, u_k)  ≈  A_k x_k + B_k u_k + c
// 其中 A = ∂f/∂x, B = ∂f/∂u 就是雅可比矩阵。模型预测控制(MPC)在每一步
// 求解   u* = argmin_u ‖A x + B u − x_goal‖²_Q + ρ‖u‖²
//   解析解  u* = −(BᵀQB+ρI)⁻¹ BᵀQ (A x − x_goal)
// 其灵敏度（最优控制对状态的雅可比）为
//     S = ∂u*/∂x = −(BᵀQB+ρI)⁻¹ BᵀQ A   （2×3）
// 本 demo：
//   (1) 解析 A,B 与有限差分一致；
//   (2) 解析灵敏度 S 与有限差分一致。
// 模型：平面运动学小车(unicycle)  x=(px,py,θ), u=(v,ω)。
//
// 编译: g++ -O3 -std=c++17 jac_l3_motion.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;

static M transpose(const M& A){ int m=A.size(),n=A[0].size(); M T(n,V(m,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)T[j][i]=A[i][j]; return T; }
static V matVec(const M& A, const V& x){ int m=A.size(),n=A[0].size(); V y(m,0); for(int i=0;i<m;i++){double s=0;for(int j=0;j<n;j++)s+=A[i][j]*x[j];y[i]=s;} return y; }
static M matMul(const M& A, const M& B){ int m=A.size(),n=B[0].size(),k=A[0].size(); M C(m,V(n,0)); for(int i=0;i<m;i++)for(int p=0;p<k;p++)for(int j=0;j<n;j++)C[i][j]+=A[i][p]*B[p][j]; return C; }

// 离散动力学 f
static V f(const V& x, const V& u, double dt){
    return { x[0]+u[0]*cos(x[2])*dt, x[1]+u[0]*sin(x[2])*dt, x[2]+u[1]*dt };
}
// 解析 A = ∂f/∂x
static M Amat(const V& x, const V& u, double dt){
    double th=x[2], v=u[0];
    return { {1,0,-v*sin(th)*dt},
             {0,1, v*cos(th)*dt},
             {0,0, 1} };
}
// 解析 B = ∂f/∂u
static M Bmat(const V& x, const V& u, double dt){
    double th=x[2];
    return { {cos(th)*dt, 0},
             {sin(th)*dt, 0},
             {0, dt} };
}
static M fdA(const V& x, const V& u, double dt, double h=1e-6){
    V x0=f(x,u,dt); M A(3,V(3,0));
    for(int k=0;k<3;k++){ V xp=x; xp[k]+=h; V x1=f(xp,u,dt); for(int r=0;r<3;r++) A[r][k]=(x1[r]-x0[r])/h; }
    return A;
}
static M fdB(const V& x, const V& u, double dt, double h=1e-6){
    V x0=f(x,u,dt); M B(3,V(2,0));
    for(int k=0;k<2;k++){ V up=u; up[k]+=h; V u1=f(x,up,dt); for(int r=0;r<3;r++) B[r][k]=(u1[r]-x0[r])/h; }
    return B;
}

// 一步 MPC 最优控制 u*
static V mpc_u(const V& x, const V& xg, const M& A, const M& B, const V& Q, double rho){
    int n=x.size(); // 3
    M Qm(n,V(n,0)); for(int i=0;i<n;i++) Qm[i][i]=Q[i];
    M Bt=transpose(B);
    M Mm=matMul(matMul(Bt,Qm),B);
    for(int i=0;i<(int)B[0].size();i++) Mm[i][i]+=rho;
    M rhs=matMul(Bt,Qm);
    V ax=matVec(A,x);
    V d(n,0); for(int i=0;i<n;i++) d[i]=ax[i]-xg[i];
    // r = rhs·d
    V rr((int)B[0].size(),0);
    for(int i=0;i<(int)B[0].size();i++){ double s=0; for(int k=0;k<n;k++) s+=rhs[i][k]*d[k]; rr[i]=s; }
    // 解 2x2 (BᵀQB+ρI) u = −r
    double a=Mm[0][0],b=Mm[0][1],c=Mm[1][0],dd=Mm[1][1];
    double det=a*dd-b*c;
    return { (b*rr[1]-dd*rr[0])/det, (c*rr[0]-a*rr[1])/det };
}
// 解析灵敏度 S = - (BᵀQB+ρI)⁻¹ BᵀQ A  (2x3)
static M sensS(const M& A, const M& B, const V& Q, double rho){
    int n=A[0].size();
    M Qm(n,V(n,0)); for(int i=0;i<n;i++) Qm[i][i]=Q[i];
    M Bt=transpose(B);
    M Mm=matMul(matMul(Bt,Qm),B);
    for(int i=0;i<(int)B[0].size();i++) Mm[i][i]+=rho;
    double a=Mm[0][0],b=Mm[0][1],c=Mm[1][0],dd=Mm[1][1];
    double det=a*dd-b*c;
    M Minv={{dd/det,-b/det},{-c/det,a/det}};
    M rhs=matMul(Bt,Qm);
    M Sb=matMul(Minv,rhs);          // 2x3
    M S(2,V(3,0));
    for(int i=0;i<2;i++) for(int k=0;k<3;k++) S[i][k]=-Sb[i][k];
    return S;
}

int main(){
    bool pass=true;
    cout<<fixed<<setprecision(9);
    V x={1.0,0.5,0.3}, u={0.8,0.2}; double dt=0.1;

    // (1) A,B 与 FD 对比
    double ea=0; M Aa=Amat(x,u,dt), Af=fdA(x,u,dt);
    for(int r=0;r<3;r++) for(int k=0;k<3;k++) ea=max(ea,fabs(Aa[r][k]-Af[r][k]));
    double eb=0; M Ba=Bmat(x,u,dt), Bf=fdB(x,u,dt);
    for(int r=0;r<3;r++) for(int k=0;k<2;k++) eb=max(eb,fabs(Ba[r][k]-Bf[r][k]));
    cout<<"[L3.2] max|A−A_fd|="<<ea<<"   max|B−B_fd|="<<eb<<"\n";
    if(!(ea<1e-6 && eb<1e-6)){ pass=false; cout<<"  [失败] 动力学雅可比与FD不符\n"; }

    // (2) MPC 灵敏度 S 与 FD 对比
    V xg={2.0,1.0,0.0}, Q={1,1,0.5}; double rho=0.1;
    M A=Amat(x,u,dt), B=Bmat(x,u,dt);
    M S=sensS(A,B,Q,rho);
    double es=0;
    for(int c=0;c<3;c++){
        V xp=x; xp[c]+=1e-6;
        V u0=mpc_u(x,xg,A,B,Q,rho), u1=mpc_u(xp,xg,A,B,Q,rho);
        for(int i=0;i<2;i++) es=max(es,fabs(S[i][c]-(u1[i]-u0[i])/1e-6));
    }
    cout<<"[L3.2] max|S_analytic−S_fd|="<<es<<"\n";
    if(!(es<1e-6)){ pass=false; cout<<"  [失败] MPC灵敏度与FD不符\n"; }

    cout<<"\n========================================\n";
    cout<<(pass? " [PASS] 动力学雅可比 A/B 与 MPC 灵敏度均正确" : " [FAIL] 见上")<<"\n";
    cout<<"========================================\n";
    return pass?0:1;
}
