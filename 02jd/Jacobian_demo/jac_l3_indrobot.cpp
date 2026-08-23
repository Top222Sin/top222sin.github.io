// jac_l3_indrobot.cpp
// ============================================================================
// L3.5 工业用例：2-link 机械臂计算力矩控制（Computed-Torque Control）
// ----------------------------------------------------------------------------
// 工业机器人的核心是把"关节空间"和"任务空间（末端笛卡尔位姿）"通过雅可比
// 联系起来，并用完整的动力学方程做前馈补偿：
//
//   关节动力学（拉格朗日二阶系统）:
//       M(q) q̈ + C(q,q̇) q̇ + G(q) = τ
//   其中
//     M(q)         —— 质量矩阵（与动能 Hessian 等价）
//     C(q,q̇)      —— 科氏/离心项（满足 Ṁ−2C 反对称）
//     G(q)         —— 重力项（= 势能梯度）
//   几何雅可比 J(q) = ∂FK/∂q 把关节速度映射到末端速度 ẏ = J q̇
//
//   计算力矩控制（反馈线性化）:
//       τ = M(q)(q̈_des + Kp·e + Kd·ē) + C(q,q̇) q̇ + G(q)
//   闭环得到 ë + Kd ė + Kp e = 0  →  指数收敛到期望轨迹。
//
//   任务空间奇异回避用 DLS 伪逆:  J⁺ = Jᵀ (J Jᵀ + λ²I)⁻¹
//
// 本 demo 用纯 C++17 stdlib，所有公式先做**独立物理/有限差分**校验，再跑
// 闭环跟踪仿真验证整体自洽。
//
// 编译: g++ -O3 -std=c++17 jac_l3_indrobot.cpp -o t && ./t
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
static M matAdd(const M& A, const M& B){ int m=A.size(),n=A[0].size(); M C(m,V(n,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)C[i][j]=A[i][j]+B[i][j]; return C; }
static M matSub(const M& A, const M& B){ int m=A.size(),n=A[0].size(); M C(m,V(n,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)C[i][j]=A[i][j]-B[i][j]; return C; }
static M matScale(const M& A, double s){ int m=A.size(),n=A[0].size(); M C(m,V(n,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)C[i][j]=A[i][j]*s; return C; }
static M inv2(const M& A){
    double det=A[0][0]*A[1][1]-A[0][1]*A[1][0];
    return {{A[1][1]/det, -A[0][1]/det}, {-A[1][0]/det, A[0][0]/det}};
}
static double maxDiffMat(const M& A, const M& B){
    double m=0; for(int i=0;i<A.size();i++)for(int j=0;j<A[0].size();j++) m=max(m,abs(A[i][j]-B[i][j])); return m;
}
static double maxDiffVec(const V& a, const V& b){
    double m=0; for(size_t i=0;i<a.size();i++) m=max(m,abs(a[i]-b[i])); return m;
}
static double maxDiffScal(const V& a, const V& b){ // 列向量比较
    double m=0; for(size_t i=0;i<a.size();i++) m=max(m,abs(a[i]-b[i])); return m;
}

// ---------- 物理参数 ----------
static const double L1=1.0, L2=0.8;
static const double LC1=0.5, LC2=0.4;
static const double M1=1.0, M2=0.8;
static const double I1=0.1, I2=0.08;
static const double GG=9.81;
static const double alpha = M1*LC1*LC1 + M2*(L1*L1+LC2*LC2) + I1 + I2;
static const double betaM  = M2*L1*LC2;
static const double gammaM = M2*LC2*LC2 + I2;
static const double b1 = (M1*LC1 + M2*L1)*GG;
static const double b2 = M2*LC2*GG;

// ---------- 正运动学 & 几何雅可比 ----------
static V fk(const V& q){
    return { L1*cos(q[0]) + L2*cos(q[0]+q[1]),
             L1*sin(q[0]) + L2*sin(q[0]+q[1]) };
}
// 几何雅可比 J(q): q̇ -> [ẋ, ẏ]
static M J_analytic(const V& q){
    double s1=sin(q[0]), c1=cos(q[0]);
    double s12=sin(q[0]+q[1]), c12=cos(q[0]+q[1]);
    return { {-L1*s1 - L2*s12, -L2*s12},
             { L1*c1 + L2*c12,  L2*c12} };
}
static M fd_J(const V& q, double h=1e-6){
    V p0=fk(q); M J(2,V(2,0));
    for(int k=0;k<2;k++){
        V qp=q, qm=q; qp[k]+=h; qm[k]-=h;
        V pp=fk(qp), pm=fk(qm);
        for(int r=0;r<2;r++) J[r][k]=(pp[r]-pm[r])/(2*h);
    }
    return J;
}

// ---------- 质量矩阵 / 科氏 / 重力 ----------
static M Mass(const V& q){
    double c2=cos(q[1]), s2=sin(q[1]);
    return {{alpha + 2*betaM*c2, gammaM + betaM*c2},
            {gammaM + betaM*c2,   gammaM}};
}
static M Cor(const V& q, const V& qd){
    double s2=sin(q[1]);
    return {{ -betaM*s2*qd[1],            -betaM*s2*(qd[0]+qd[1])},
            {  betaM*s2*qd[0],             0.0}};
}
static V Grav(const V& q){
    double c1=cos(q[0]), c12=cos(q[0]+q[1]);
    return { b1*c1 + b2*c12, b2*c12 };
}

// 独立校验：M 应等于动能 Hessian  K = ½ q̇ᵀ M q̇
static double KE(const V& q, const V& qd){
    double v1sq = LC1*LC1*qd[0]*qd[0];
    double KE1 = 0.5*M1*v1sq + 0.5*I1*qd[0]*qd[0];
    double s1=sin(q[0]), c1=cos(q[0]), s12=sin(q[0]+q[1]), c12=cos(q[0]+q[1]);
    double dx = -(L1*s1+LC2*s12)*qd[0] - LC2*s12*qd[1];
    double dy =  (L1*c1+LC2*c12)*qd[0] + LC2*c12*qd[1];
    double v2sq = dx*dx + dy*dy;
    double KE2 = 0.5*M2*v2sq + 0.5*I2*(qd[0]+qd[1])*(qd[0]+qd[1]);
    return KE1 + KE2;
}
static M M_from_KE(const V& q){
    M Mk(2,V(2,0)); double h=1e-4;
    for(int i=0;i<2;i++)for(int j=0;j<2;j++){
        V qdpp={0,0}, qdp={0,0}, qdj={0,0};
        qdpp[i]+=h; qdpp[j]+=h; qdp[i]+=h; qdj[j]+=h;
        Mk[i][j] = (KE(q,qdpp)-KE(q,qdp)-KE(q,qdj)+KE(q,{0,0}))/(h*h);
    }
    return Mk;
}
// 独立校验：G 应等于势能梯度  PE = m1 g y1 + m2 g y2
static double PE(const V& q){
    double y1 = LC1*sin(q[0]);
    double y2 = L1*sin(q[0]) + LC2*sin(q[0]+q[1]);
    return M1*GG*y1 + M2*GG*y2;
}
static V fd_G(const V& q, double h=1e-6){
    V g(2,0);
    for(int k=0;k<2;k++){
        V qp=q, qm=q; qp[k]+=h; qm[k]-=h;
        g[k]=(PE(qp)-PE(qm))/(2*h);
    }
    return g;
}
// 独立校验：C 满足 Ṁ − 2C 反对称（Ṁ 由有限差分）
static M fd_Mdot(const V& q, const V& qd, double h=1e-5){
    V qp=q, qm=q; qp[1]+=h*qd[1]; qm[1]-=h*qd[1];
    M Mp=Mass(qp), Mm=Mass(qm);
    return matScale(matSub(Mp, Mm), 1.0/(2*h));
}

// ---------- DLS 任务空间伪逆 ----------
static M dls_Jpinv(const M& J, double lam=1e-3){
    M JJt = matMul(J, transpose(J));
    JJt[0][0]+=lam*lam; JJt[1][1]+=lam*lam;
    return matMul(transpose(J), inv2(JJt));
}

// ---------- 计算力矩控制律 + 闭环动力学 ----------
static V computed_torque(const V& q, const V& qd, const V& q_des, const V& qd_des,
                         const V& qdd_des, double Kp, double Kd){
    V e={q_des[0]-q[0], q_des[1]-q[1]};
    V ed={qd_des[0]-qd[0], qd_des[1]-qd[1]};
    V qdd_cmd={ qdd_des[0]+Kp*e[0]+Kd*ed[0], qdd_des[1]+Kp*e[1]+Kd*ed[1] };
    V tau = matVec(Mass(q), qdd_cmd);
    V cq = matVec(Cor(q,qd), qd);
    V gq = Grav(q);
    return { tau[0]+cq[0]+gq[0], tau[1]+cq[1]+gq[1] };
}
static V forward_dyn(const V& q, const V& qd, const V& tau){
    V cq = matVec(Cor(q,qd), qd);
    V gq = Grav(q);
    V rhs={ tau[0]-cq[0]-gq[0], tau[1]-cq[1]-gq[1] };
    return matVec(inv2(Mass(q)), rhs);
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const string& name, double err, double tol){
        bool ok = err < tol;
        cout << "  [" << (ok?"PASS":"FAIL") << "] " << name
             << "  maxErr=" << scientific << setprecision(2) << err << endl;
        ok?pass++:fail++;
    };
    cout << "=== L3.9 工业机械臂：计算力矩控制 (2-link planar) ===" << endl;

    // 1) J(q) vs FD
    for(V q : {V{0.3,0.5}, V{1.1,-0.7}, V{-0.9,2.0}}){
        chk("J vs FD", maxDiffMat(J_analytic(q), fd_J(q)), 1e-6);
    }
    // 2) M(q) vs 动能 Hessian
    for(V q : {V{0.3,0.5}, V{1.1,-0.7}, V{-0.9,2.0}}){
        chk("M vs KE-Hessian", maxDiffMat(Mass(q), M_from_KE(q)), 1e-6);
    }
    // 3) G(q) vs 势能梯度
    for(V q : {V{0.3,0.5}, V{1.1,-0.7}, V{-0.9,2.0}}){
        chk("G vs PE-gradient", maxDiffVec(Grav(q), fd_G(q)), 1e-6);
    }
    // 4) C: Ṁ − 2C 反对称
    for(auto [q,qd] : {pair<V,V>{V{0.3,0.5}, V{0.4,0.2}},
                       pair<V,V>{V{1.1,-0.7}, V{-0.3,0.5}},
                       pair<V,V>{V{-0.9,2.0}, V{0.1,-0.6}}}){
        M S = matSub(fd_Mdot(q,qd), matScale(Cor(q,qd),2.0));
        chk("skew(Mdot-2C)", maxDiffMat(S, matScale(transpose(S),-1.0)), 1e-6);
    }
    // 5) DLS: J (J^+ yd) ≈ yd
    for(V q : {V{0.3,0.5}, V{1.1,-0.7}}){
        M Jp = dls_Jpinv(J_analytic(q));
        V yd={0.2,-0.1};
        V qd_sol = matVec(Jp, yd);
        V yhat = matVec(J_analytic(q), qd_sol);
        chk("DLS J J^+ yd ~ yd", maxDiffScal(yhat, yd), 1e-5);
    }
    // 6) 闭环跟踪仿真（收敛性）
    {
        double Kp=100.0, Kd=20.0, dt=0.005;
        V q={-0.6,1.0}, qd={0,0};
        double T=3.0; int steps=(int)(T/dt);
        double e_final=0;
        for(int i=0;i<steps;i++){
            double t=i*dt;
            V qd_d  = {0.5*cos(t),            -0.28*sin(0.7*t)};
            V qdd_d = {-0.5*sin(t), -0.4*0.49*cos(0.7*t)};
            V q_des = {0.5*sin(t), 0.4*cos(0.7*t)};
            V tau = computed_torque(q,qd,q_des,qd_d,qdd_d,Kp,Kd);
            V qdd = forward_dyn(q,qd,tau);
            qd={qd[0]+qdd[0]*dt, qd[1]+qdd[1]*dt};
            q ={q[0]+qd[0]*dt,    q[1]+qd[1]*dt};
            V q_t={0.5*sin(t+dt), 0.4*cos(0.7*(t+dt))};
            if(i>=steps-2) e_final = max(abs(q[0]-q_t[0]), abs(q[1]-q_t[1]));
        }
        cout << fixed << setprecision(5);
        cout << "  [INFO] computed-torque tracking final|e| = " << e_final << endl;
        chk("tracking converges (final|e|<1e-2)", e_final, 1e-2);
    }

    cout << "------------------------------------------" << endl;
    cout << "  PASS=" << pass << "  FAIL=" << fail << "  ("
         << (fail==0 ? "ALL PASS" : "SOME FAIL") << ")" << endl;
    return fail==0 ? 0 : 1;
}
