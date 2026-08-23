// jac_l3_slam.cpp
// ============================================================================
// L3.1 SLAM 位姿图(Pose Graph)的 SE(3) 李代数雅可比
// ----------------------------------------------------------------------------
// SLAM 后端本质是"在流形上做非线性最小二乘"。一条相对位姿边约束：
//   残差 e = log( Z^{-1} · T_i^{-1} · T_j )^∨   (SE(2) 简化版，SE(3) 同理)
// 其中 T_i,T_j 是待优化顶点，Z 是相对位姿测量。梯度下降/高斯-牛顿需要
// 残差对 T_i、T_j 的雅可比 J_i = ∂e/∂ξ_i, J_j = ∂e/∂ξ_j（ξ 为李代数坐标）。
//
// 本 demo 做两件事：
//   (1) 解析写出 J_i、J_j，并用有限差分(FD)交叉验证二者一致；
//   (2) 用这两个雅可比做一次 1 边/2 顶点的高斯-牛顿，残差 → 0，证明
//       雅可比正确可用。
// (真实系统 Z 是 T_i^{-1}T_j 的真值；这里给了"一致"与"有偏"两组验证。)
//
// 编译: g++ -O3 -std=c++17 jac_l3_slam.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <tuple>
using namespace std;

using V3 = tuple<double,double,double>;   // SE(2) 位姿 (x, y, θ)

static V3 comp(const V3& a, const V3& b){
    double c=cos(get<2>(a)), s=sin(get<2>(a));
    return { get<0>(a)+c*get<0>(b)-s*get<1>(b),
             get<1>(a)+s*get<0>(b)+c*get<1>(b),
             get<2>(a)+get<2>(b) };
}
static V3 inv(const V3& t){
    double c=cos(get<2>(t)), s=sin(get<2>(t));
    return { -(c*get<0>(t)+s*get<1>(t)), s*get<0>(t)-c*get<1>(t), -get<2>(t) };
}
// 残差 e = Z^{-1} · T_i^{-1} · T_j  （取 (x,y,θ) 作为切空间坐标）
static V3 err(const V3& Ti, const V3& Tj, const V3& Z){
    return comp(inv(Z), comp(inv(Ti), Tj));
}
// 解析雅可比：∂e/∂ξ_i  (3x3)
static void Ji(const V3& Ti, const V3& Tj, const V3& Z, double J[3][3]){
    double xi=get<0>(Ti), yi=get<1>(Ti), ti=get<2>(Ti);
    double xj=get<0>(Tj), yj=get<1>(Tj);
    double c=cos(ti), s=sin(ti);
    V3 ZI = inv(Z); double cz=cos(get<2>(ZI)), sz=sin(get<2>(ZI));
    double A = s*xi - c*yi - s*xj + c*yj;   // ∂d.x/∂ti
    double B = c*xi + s*yi - c*xj - s*yj;   // ∂d.y/∂ti
    for(int r=0;r<3;r++) for(int k=0;k<3;k++) J[r][k]=0;
    J[0][0]=-cz*c - sz*s;  J[0][1]=-cz*s + sz*c;  J[0][2]= cz*A - sz*B;
    J[1][0]=-sz*c + cz*s;  J[1][1]=-sz*s - cz*c;  J[1][2]= sz*A + cz*B;
    J[2][2]=-1.0;
}
// 解析雅可比：∂e/∂ξ_j  (3x3)
static void Jj(const V3& Ti, const V3& Tj, const V3& Z, double J[3][3]){
    double ti=get<2>(Ti); double c=cos(ti), s=sin(ti);
    V3 ZI = inv(Z); double cz=cos(get<2>(ZI)), sz=sin(get<2>(ZI));
    for(int r=0;r<3;r++) for(int k=0;k<3;k++) J[r][k]=0;
    J[0][0]= cz*c + sz*s;  J[0][1]= cz*s - sz*c;  J[2][2]= 1.0;
    J[1][0]= sz*c - cz*s;  J[1][1]= sz*s + cz*c;
}
static V3 fdJi(const V3& Ti, const V3& Tj, const V3& Z, int k){
    double h=1e-6; V3 T2=Ti;
    // perturb component k of Ti
    double *p = (k==0)?&get<0>(T2):(k==1)?&get<1>(T2):&get<2>(T2);
    *p += h;
    V3 e0=err(Ti,Tj,Z), e1=err(T2,Tj,Z);
    return {(get<0>(e1)-get<0>(e0))/h,(get<1>(e1)-get<1>(e0))/h,(get<2>(e1)-get<2>(e0))/h};
}
static V3 fdJj(const V3& Ti, const V3& Tj, const V3& Z, int k){
    double h=1e-6; V3 T2=Tj;
    double *p = (k==0)?&get<0>(T2):(k==1)?&get<1>(T2):&get<2>(T2);
    *p += h;
    V3 e0=err(Ti,Tj,Z), e1=err(Ti,T2,Z);
    return {(get<0>(e1)-get<0>(e0))/h,(get<1>(e1)-get<1>(e0))/h,(get<2>(e1)-get<2>(e0))/h};
}
// 解 6x6 线性方程组 (高斯消元) -> 给 GN 用
static void solve6(double A[6][6], const double b[6], double x[6]){
    double M[6][7];
    for(int i=0;i<6;i++){ for(int j=0;j<6;j++) M[i][j]=A[i][j]; M[i][6]=b[i]; }
    for(int c=0;c<6;c++){
        int p=c; for(int r=c+1;r<6;r++) if(fabs(M[r][c])>fabs(M[p][c])) p=r;
        for(int j=0;j<7;j++) swap(M[c][j],M[p][j]);
        double pv=M[c][c];
        for(int r=0;r<6;r++) if(r!=c){ double f=M[r][c]/pv; for(int j=c;j<7;j++) M[r][j]-=f*M[c][j]; }
    }
    for(int i=0;i<6;i++) x[i]=M[i][6]/M[i][i];
}

int main(){
    bool pass=true;
    cout<<fixed<<setprecision(9);

    // (1) 一致位姿：Z = T_i^{-1} T_j → 残差应≈0
    V3 Ti={1.2,0.5,0.3}, Tj={2.4,1.1,0.7}, Z=comp(inv(Ti),Tj);
    V3 e0=err(Ti,Tj,Z);
    cout<<"[L3.1] 一致位姿残差 = ("<<get<0>(e0)<<", "<<get<1>(e0)<<", "<<get<2>(e0)<<")\n";

    double emax=0;
    double Ja_i[3][3], Ja_j[3][3]; Ji(Ti,Tj,Z,Ja_i); Jj(Ti,Tj,Z,Ja_j);
    for(int r=0;r<3;r++) for(int k=0;k<3;k++){
        V3 fdi=fdJi(Ti,Tj,Z,k); emax=max(emax,fabs(Ja_i[r][k]-((r==0)?get<0>(fdi):(r==1)?get<1>(fdi):get<2>(fdi))));
        V3 fdj=fdJj(Ti,Tj,Z,k); emax=max(emax,fabs(Ja_j[r][k]-((r==0)?get<0>(fdj):(r==1)?get<1>(fdj):get<2>(fdj))));
    }
    cout<<"[L3.1] max|J_analytic - J_fd| = "<<emax<<"\n";
    if(!(emax<1e-5)){ pass=false; cout<<"  [失败] 雅可比与有限差分不符\n"; }

    // (2) GN：1 边/2 顶点，把有偏测量的残差优化到 0
    Ti={1.0,0.4,0.2}; Tj={2.0,1.3,0.9}; Z={0.85,0.5,0.6};
    V3 e=err(Ti,Tj,Z);
    cout<<"[L3.1] 初始残差范数 = "<<sqrt(get<0>(e)*get<0>(e)+get<1>(e)*get<1>(e)+get<2>(e)*get<2>(e))<<"\n";
    int it;
    for(it=0; it<30; it++){
        e=err(Ti,Tj,Z);
        double JXi[3][3], JXj[3][3]; Ji(Ti,Tj,Z,JXi); Jj(Ti,Tj,Z,JXj);
        // J(3x6) = [JXi | JXj]
        double J[3][6];
        for(int r=0;r<3;r++) for(int c=0;c<3;c++){ J[r][c]=JXi[r][c]; J[r][c+3]=JXj[r][c]; }
        double H[6][6]={0}, g[6]={0};
        for(int a=0;a<3;a++) for(int b=0;b<6;b++){
            for(int c=0;c<6;c++) H[b][c]+=J[a][b]*J[a][c];
            g[b]+=J[a][b]*((a==0)?get<0>(e):(a==1)?get<1>(e):get<2>(e));
        }
        for(int i=0;i<6;i++) H[i][i]+=1e-6;
        double dx[6]; double gb[6]; for(int i=0;i<6;i++) gb[i]=-g[i];
        solve6(H,gb,dx);
        Ti={get<0>(Ti)+dx[0],get<1>(Ti)+dx[1],get<2>(Ti)+dx[2]};
        Tj={get<0>(Tj)+dx[3],get<1>(Tj)+dx[4],get<2>(Tj)+dx[5]};
        V3 en=err(Ti,Tj,Z);
        if(sqrt(get<0>(en)*get<0>(en)+get<1>(en)*get<1>(en)+get<2>(en)*get<2>(en))<1e-9) break;
    }
    V3 ef=err(Ti,Tj,Z);
    double nf=sqrt(get<0>(ef)*get<0>(ef)+get<1>(ef)*get<1>(ef)+get<2>(ef)*get<2>(ef));
    cout<<"[L3.1] GN 收敛于 "<<(it+1)<<" 步，最终残差范数 = "<<nf<<"\n";
    if(!(nf<1e-7)){ pass=false; cout<<"  [失败] GN 未收敛到 0\n"; }

    cout<<"\n========================================\n";
    cout<<(pass? " [PASS] SLAM 位姿图残差雅可比正确，GN 可收敛" : " [FAIL] 见上")<<"\n";
    cout<<"========================================\n";
    return pass?0:1;
}
