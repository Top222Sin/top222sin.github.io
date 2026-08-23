// jac_l3_ibvs.cpp
// ============================================================================
// L3.4 综合案例：基于图像的视觉伺服 IBVS（融合 视觉 + 机械臂 + DLS）
// ----------------------------------------------------------------------------
// 相机装在 2R 平面机械臂末端。世界中有一个特征点 P。相机帧下：
//     Pc = R(−φ)·(P − 末端位置),   图像特征(归一化) s = Pc.y / Pc.x
// 其"图像雅可比"(交互矩阵) L 把相机空间速度 v_c 映射到特征速度：
//     ṡ = L · v_c,   L = [∂s/∂xe, ∂s/∂ye, ∂s/∂φ]
// 末端速度由机械臂雅可比给出：v_c = J(q)·q̇。
// 因此 ṡ = (L·J)·q̇，IBVS 控制律为：
//     v_c* = −λ L⁺(s − s_des)          (DLS 保证奇异鲁棒)
//     q̇   = J⁺ v_c*                      (DLS 求关节速度)
// 本 demo 验证：(1) L、LJ=∂s/∂q 与有限差分一致；(2) 伺服循环把 s 收敛到 s_des。
// 这是"计算机视觉 × 机器人学 × 数值鲁棒性"的交汇点。
//
// 编译: g++ -O3 -std=c++17 jac_l3_ibvs.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

static const double L1=1.0, L2=1.0;
static const double Px=1.4, Py=1.0;

// 末端位姿 (xe, ye, φ=θ1+θ2)
static void fk(const double q[2], double eef[3]){
    double t1=q[0], t2=q[1];
    eef[0]=L1*cos(t1)+L2*cos(t1+t2);
    eef[1]=L1*sin(t1)+L2*sin(t1+t2);
    eef[2]=t1+t2;
}
// 点 P 在相机帧坐标
static void Pc(const double eef[3], double& X, double& Y){
    double dx=Px-eef[0], dy=Py-eef[1], phi=eef[2];
    X = cos(phi)*dx + sin(phi)*dy;
    Y = -sin(phi)*dx + cos(phi)*dy;
}
static double sval(const double eef[3]){
    double X,Y; Pc(eef,X,Y); return Y/X;
}
// 图像雅可比 L (1x3) = [∂s/∂xe, ∂s/∂ye, ∂s/∂φ]
static void Lmat(const double eef[3], double L[3]){
    double X,Y; Pc(eef,X,Y); double phi=eef[2];
    double dx=Px-eef[0], dy=Py-eef[1];
    L[0]=(sin(phi)*X + Y*cos(phi))/(X*X);
    L[1]=(-cos(phi)*X + Y*sin(phi))/(X*X);
    L[2]=-(1.0 + (Y/X)*(Y/X));
}
// 机械臂雅可比 J (3x2): v_e = J q̇
static void Jmat(const double q[2], double J[3][2]){
    double t1=q[0], t2=q[1];
    double xe1=-L1*sin(t1)-L2*sin(t1+t2), xe2=-L2*sin(t1+t2);
    double ye1= L1*cos(t1)+L2*cos(t1+t2), ye2= L2*cos(t1+t2);
    J[0][0]=xe1; J[0][1]=xe2;
    J[1][0]=ye1; J[1][1]=ye2;
    J[2][0]=1.0; J[2][1]=1.0;
}

int main(){
    bool pass=true;
    cout<<fixed<<setprecision(9);

    double q[2]={2.0,4.2};
    double eef[3]; fk(q,eef);
    double s0=sval(eef);
    cout<<"[L3.4] 起点 q=("<<q[0]<<","<<q[1]<<") 特征 s="<<s0<<"\n";

    // (1) 图像雅可比 L 与 FD 对比
    double L[3], Lfd[3]; Lmat(eef,L);
    double h=1e-7;
    for(int k=0;k<3;k++){
        double e2[3]={eef[0],eef[1],eef[2]}; e2[k]+=h;
        Lfd[k]=(sval(e2)-s0)/h;
    }
    double eL=0; for(int k=0;k<3;k++) eL=max(eL,fabs(L[k]-Lfd[k]));
    cout<<"[L3.4] 图像雅可比 L = ("<<L[0]<<", "<<L[1]<<", "<<L[2]<<")   max|L−FD|="<<eL<<"\n";
    if(!(eL<1e-5)){ pass=false; cout<<"  [失败] 图像雅可比与FD不符\n"; }

    // (2) 特征对关节雅可比 LJ = ∂s/∂q = L·J  与 FD 对比
    double J[3][2]; Jmat(q,J);
    double LJ[2]={0,0}; for(int i=0;i<2;i++) for(int c=0;c<3;c++) LJ[i]+=L[c]*J[c][i];
    double s_q0=sval(eef); double LJfd[2];
    for(int k=0;k<2;k++){
        double q2[2]={q[0],q[1]}; q2[k]+=h; double e2[3]; fk(q2,e2);
        LJfd[k]=(sval(e2)-s_q0)/h;
    }
    double eJ=0; for(int k=0;k<2;k++) eJ=max(eJ,fabs(LJ[k]-LJfd[k]));
    cout<<"[L3.4] ∂s/∂q = ("<<LJ[0]<<", "<<LJ[1]<<")   max|LJ−FD|="<<eJ<<"\n";
    if(!(eJ<1e-5)){ pass=false; cout<<"  [失败] ∂s/∂q 与FD不符\n"; }

    // (3) IBVS 伺服循环
    double qd[2]={1.8,4.5}; double eefd[3]; fk(qd,eefd); double s_des=sval(eefd);
    cout<<"[L3.4] 目标 s_des="<<s_des<<"\n";
    double lam=0.8, lamj=0.02, dt=0.2;
    int it;
    for(it=0; it<500; it++){
        fk(q,eef); double s=sval(eef); double e=s-s_des;
        if(fabs(e)<1e-4) break;
        // 图像雅可比 L，DLS 求相机速度（1 个特征 → 3 维相机速度，取最小范数）
        double Lc[3]; Lmat(eef,Lc);
        double a=Lc[0]*Lc[0]+Lc[1]*Lc[1]+Lc[2]*Lc[2];
        double denom=a+lam*lam;
        double vc[3]; for(int i=0;i<3;i++) vc[i]=-lam*Lc[i]/denom*e;
        // 机械臂雅可比 J，DLS 求关节速度 q̇ = (JᵀJ+λj²I)⁻¹Jᵀ vc
        double Jc[3][2]; Jmat(q,Jc);
        double JTJ[2][2]={{0,0},{0,0}};
        for(int i=0;i<3;i++) for(int r=0;r<2;r++) for(int cc=0;cc<2;cc++) JTJ[r][cc]+=Jc[i][r]*Jc[i][cc];
        JTJ[0][0]+=lamj*lamj; JTJ[1][1]+=lamj*lamj;
        double det=JTJ[0][0]*JTJ[1][1]-JTJ[0][1]*JTJ[1][0];
        double iJTJ[2][2]={{JTJ[1][1]/det,-JTJ[0][1]/det},{-JTJ[1][0]/det,JTJ[0][0]/det}};
        double g[2]={0,0}; for(int i=0;i<2;i++) for(int r=0;r<3;r++) g[i]+=Jc[r][i]*vc[r];
        double qd_vec[2]={0,0}; for(int i=0;i<2;i++) for(int j=0;j<2;j++) qd_vec[i]+=iJTJ[i][j]*g[j];
        q[0]+=qd_vec[0]*dt; q[1]+=qd_vec[1]*dt;
    }
    fk(q,eef); double s_end=sval(eef); double e_end=s_end-s_des;
    cout<<"[L3.4] 伺服 "<<(it+1)<<" 步后 特征误差="<<e_end<<"\n";
    if(!(fabs(e_end)<1e-3)){ pass=false; cout<<"  [失败] IBVS 未收敛\n"; }

    cout<<"\n========================================\n";
    cout<<(pass? " [PASS] 图像雅可比/∂s/∂q 正确，IBVS 伺服收敛" : " [FAIL] 见上")<<"\n";
    cout<<"========================================\n";
    return pass?0:1;
}
