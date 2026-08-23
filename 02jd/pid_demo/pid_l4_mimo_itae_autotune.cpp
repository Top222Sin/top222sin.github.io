// pid_l4_mimo_itae_autotune.cpp
// L4.10 —— MIMO 解耦 + ITAE 自动整定（"指标即偏好" 亲手演示）
//
// 复用 L4.7/L4.9 的 2x2 耦合对象 K=[[1,0.3],[0.5,1]] (tau1=1, tau2=2) 与
// 静态解耦器 D = Kss^-1 (直流严格对角化)。解耦后每回路独立用指标驱动整定：
//   - 指标 A：IAE  = Σ |e|·dt          （L4.6 / L4.9 用过的目标）
//   - 指标 B：ITAE = Σ (k·dt)·|e|·dt    （时间加权，惩罚中后段误差）
// 对同一对象、同一解耦器、同一顺序整定流程，仅把目标从 IAE 换成 ITAE，
// 观察最优增益如何移动 —— 结论：ITAE 把最优"推"向更温和的增益。
//
// 验证：本机无 C++ 编译器，采用
//   受管 Python 算法镜像 11/11 PASS → 本文件 C++17 逐字转写 → C++ 控制流镜像 11/11 PASS
// 小矩阵运算（eyeM/matmul/matsub/matvec/inv）全部手写，无第三方依赖。
//
// 运行：无编译器环境由 Python 镜像兜底；有 g++ 时直接 g++ -std=c++17 -O2 编译运行。

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using VD = std::vector<double>;

// ---------- 小矩阵手写库（与 L4.7/L4.9 同款）----------
static VD eyeM(int n){
    VD E(n*n,0.0);
    for(int i=0;i<n;i++) E[i*n+i]=1.0;
    return E;
}
// A(n×m) * B(m×p) -> C(n×p)
static VD matmul(const VD& A,int n,int m,const VD& B,int p){
    VD C(n*p,0.0);
    for(int i=0;i<n;i++) for(int k=0;k<m;k++){
        double a=A[i*m+k]; if(a==0.0) continue;
        for(int j=0;j<p;j++) C[i*p+j]+=a*B[k*p+j];
    }
    return C;
}
static VD matsub(const VD& A,const VD& B){
    int n=(int)A.size(); VD C(n);
    for(int i=0;i<n;i++) C[i]=A[i]-B[i];
    return C;
}
static VD matvec(const VD& A,int n,int m,const VD& x){
    VD y(n,0.0);
    for(int i=0;i<n;i++){ double s=0.0; for(int j=0;j<m;j++) s+=A[i*m+j]*x[j]; y[i]=s; }
    return y;
}
// 高斯-约当求逆 (n×n, 以一维行主序传入)
static VD inv(const VD& A){
    int n=(int)std::sqrt((double)A.size());
    VD M(n*(2*n),0.0);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) M[i*(2*n)+j]=A[i*n+j];
    for(int i=0;i<n;i++) M[i*(2*n)+n+i]=1.0;
    for(int col=0;col<n;col++){
        int piv=col; double best=0.0;
        for(int r=col;r<n;r++){ double v=std::fabs(M[r*(2*n)+col]); if(v>best){best=v;piv=r;} }
        if(piv!=col){ for(int j=0;j<2*n;j++) std::swap(M[col*(2*n)+j],M[piv*(2*n)+j]); }
        double pv=M[col*(2*n)+col];
        for(int j=0;j<2*n;j++) M[col*(2*n)+j]/=pv;
        for(int r=0;r<n;r++) if(r!=col && std::fabs(M[r*(2*n)+col])>0.0){
            double f=M[r*(2*n)+col];
            for(int j=0;j<2*n;j++) M[r*(2*n)+j]-=f*M[col*(2*n)+j];
        }
    }
    VD R(n*n,0.0);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) R[i*n+j]=M[i*(2*n)+(n+j)];
    return R;
}

// ---------- 对象离散化 + 解耦器 ----------
static const double dt=0.05, T=15.0, tau1=1.0, tau2=2.0, Kd=0.2;
static VD A2,B2,D2;        // 离散 A(2x2), B(2x2), 解耦器 D(2x2)
static void build_plant(){
    double a11=std::exp(-dt/tau1), a22=std::exp(-dt/tau2);
    double b11=1.0*(1.0-a11), b12=0.3*(1.0-a11);
    double b21=0.5*(1.0-a22), b22=1.0*(1.0-a22);
    A2={a11,0.0, 0.0,a22};
    B2={b11,b12, b21,b22};
    // Kss = (I-A)^-1 B
    VD I=eyeM(2);
    VD Kss=matmul(inv(matsub(I,A2)),2,2,B2,2);
    D2=inv(Kss);   // 静态解耦器：D·Kss = I（直流严格对角化）
}

// 闭环仿真：返回 y1,y2,e1,e2 轨迹
static void sim(bool decouple,const VD& r,const VD& g1,const VD& g2,
                VD& Y1,VD& Y2,VD& E1,VD& E2){
    int N=(int)(T/dt);
    VD y={0.0,0.0}, e_prev={0.0,0.0}, Iacc={0.0,0.0};
    Y1.clear();Y2.clear();E1.clear();E2.clear();
    for(int k=0;k<N;k++){
        VD e={r[0]-y[0], r[1]-y[1]};
        Iacc[0]+=e[0]*dt; Iacc[1]+=e[1]*dt;
        VD der={(e[0]-e_prev[0])/dt,(e[1]-e_prev[1])/dt};
        VD u_loop={ g1[0]*e[0]+g1[1]*Iacc[0]+g1[2]*der[0],
                    g2[0]*e[1]+g2[1]*Iacc[1]+g2[2]*der[1] };
        VD u = decouple ? matvec(D2,2,2,u_loop) : u_loop;
        VD yn={ A2[0]*y[0]+B2[0]*u[0]+B2[1]*u[1],
                A2[2]*y[0]+A2[3]*y[1]+B2[2]*u[0]+B2[3]*u[1] };
        y=yn; e_prev=e;
        Y1.push_back(y[0]);Y2.push_back(y[1]);E1.push_back(e[0]);E2.push_back(e[1]);
    }
}

static double iae(const VD& e){ double s=0; for(double v:e) s+=std::fabs(v)*dt; return s; }
static double itae(const VD& e){ double s=0; for(size_t k=0;k<e.size();k++) s+=(double)k*dt*std::fabs(e[k])*dt; return s; }

// ---------- 顺序整定：对某一指标，逐回路网格搜索 (Kp,Ki)，Kd 固定 ----------
static const double gKp[7]={0.5,0.8,1.1,1.4,1.7,2.0,2.3};
static const double gKi[7]={0.3,0.5,0.7,0.9,1.1,1.3,1.5};
// metric: 0=IAE, 1=ITAE
static VD tune(int metric,const VD& r,const VD& fixed_other){
    int best_i=-1,best_j=-1; double best=1e30;
    for(int a=0;a<7;a++) for(int b=0;b<7;b++){
        VD var={gKp[a],gKi[b],Kd};
        VD Y1,Y2,E1,E2;
        double j;
        if(r[0]!=0.0){            // 整定回路1：变增益进 g1 槽，测 E1
            sim(true,r,var,fixed_other,Y1,Y2,E1,E2);
            j = (metric==0)? iae(E1) : itae(E1);
        }else{                    // 整定回路2：变增益进 g2 槽，测 E2
            sim(true,r,fixed_other,var,Y1,Y2,E1,E2);
            j = (metric==0)? iae(E2) : itae(E2);
        }
        if(j<best){ best=j; best_i=a; best_j=b; }
    }
    return {gKp[best_i],gKi[best_j],Kd};
}

int main(){
    build_plant();
    std::cout<<std::fixed<<std::setprecision(4);
    auto chk=[&](const std::string& name,double err,double tol)->bool{
        bool ok=std::fabs(err)<=tol;
        std::cout<<(ok?"PASS":"FAIL")<<" : "<<name<<"  (err="<<err<<")"<<std::endl;
        return ok;
    };

    VD NOM={1.0,0.5,0.2};

    // === IAE 目标顺序整定 ===
    VD o1i=tune(0,{1,0},NOM);
    VD o2i=tune(0,{0,1},o1i);
    VD Y1i,Y2i,E1i,E2i;
    sim(true,{1,1},o1i,o2i,Y1i,Y2i,E1i,E2i);
    double IAE_full_i=iae(E1i)+iae(E2i);
    double ITAE_full_i=itae(E1i)+itae(E2i);

    // === ITAE 目标顺序整定 ===
    VD o1t=tune(1,{1,0},NOM);
    VD o2t=tune(1,{0,1},o1t);
    VD Y1t,Y2t,E1t,E2t;
    sim(true,{1,1},o1t,o2t,Y1t,Y2t,E1t,E2t);
    double IAE_full_t=iae(E1t)+iae(E2t);
    double ITAE_full_t=itae(E1t)+itae(E2t);

    std::cout<<"========================================================"<<std::endl;
    std::cout<<"L4.10 : IAE vs ITAE 顺序整定（指标即偏好）"<<std::endl;
    std::cout<<"========================================================"<<std::endl;
    std::cout<<"IAE  目标: 回路1=("<<o1i[0]<<","<<o1i[1]<<") 回路2=("<<o2i[0]<<","<<o2i[1]<<")"
             <<" | 全系统 IAE="<<IAE_full_i<<" ITAE="<<ITAE_full_i<<std::endl;
    std::cout<<"ITAE 目标: 回路1=("<<o1t[0]<<","<<o1t[1]<<") 回路2=("<<o2t[0]<<","<<o2t[1]<<")"
             <<" | 全系统 IAE="<<IAE_full_t<<" ITAE="<<ITAE_full_t<<std::endl;

    int passed=0,total=0;
    auto REC=[&](bool ok){ total++; if(ok) passed++; };

    std::cout<<"--- 断言 ---"<<std::endl;
    // 1) IAE 整定复现 L4.9 基线（faithfulness）
    REC(chk("IAE整定复现 L4.9 基线 IAE≈1.4750", std::fabs(IAE_full_i-1.4750), 1e-3));
    // 2) 换成 ITAE 后该指标自身更优：违反=max(0, ITAE_t - ITAE_i)
    REC(chk("ITAE目标→ITAE指标更优 (1.2646<1.5910)", std::max(0.0, ITAE_full_t-ITAE_full_i), 1e-3));
    // 3) 代价：ITAE 目标的 IAE 略升（<5%）
    REC(chk("代价: ITAE目标 IAE 上升<5%", std::max(0.0,(IAE_full_t-IAE_full_i)/IAE_full_i-0.05), 1e-3));
    // 4) loop1 Kp 在 ITAE 下更温和
    REC(chk("回路1 Kp: ITAE(1.4) < IAE(2.3) 更温和", std::max(0.0, o1t[0]-o1i[0]), 1e-9));
    // 5) loop2 Ki 在 ITAE 下更温和
    REC(chk("回路2 Ki: ITAE(1.3) < IAE(1.5) 更温和", std::max(0.0, o2t[1]-o2i[1]), 1e-9));
    // 6) 稳态 y_ss≈1（两目标）
    REC(chk("IAE目标 稳态 y1≈1", std::fabs(Y1i.back()-1.0), 0.02));
    REC(chk("IAE目标 稳态 y2≈1", std::fabs(Y2i.back()-1.0), 0.02));
    REC(chk("ITAE目标 稳态 y1≈1", std::fabs(Y1t.back()-1.0), 0.02));
    REC(chk("ITAE目标 稳态 y2≈1", std::fabs(Y2t.back()-1.0), 0.02));
    // 7) 解耦后交叉耦合：单回路阶跃 r=[1,0]，看回路2 是否仍被扰动
    VD Y1ci,Y2ci,E1ci,E2ci; sim(true,{1,0},o1i,o2i,Y1ci,Y2ci,E1ci,E2ci);
    double peak_cc_i=0; for(double v:Y2ci) peak_cc_i=std::max(peak_cc_i,std::fabs(v));
    VD Y1ct,Y2ct,E1ct,E2ct; sim(true,{1,0},o1t,o2t,Y1ct,Y2ct,E1ct,E2ct);
    double peak_cc_t=0; for(double v:Y2ct) peak_cc_t=std::max(peak_cc_t,std::fabs(v));
    REC(chk("IAE目标 单回路阶跃下交叉耦合峰值<0.05", std::max(0.0,peak_cc_i-0.05), 1e-3));
    REC(chk("ITAE目标 单回路阶跃下交叉耦合峰值<0.05", std::max(0.0,peak_cc_t-0.05), 1e-3));

    std::cout<<"========================================================"<<std::endl;
    std::cout<<"结果: "<<passed<<"/"<<total<<" 通过"<<std::endl;
    std::cout<<"ITAE→IAE 指标改进 = "<<(ITAE_full_i-ITAE_full_t)/ITAE_full_i*100.0<<"%"
             <<" ; IAE 代价 = "<<(IAE_full_t-IAE_full_i)/IAE_full_i*100.0<<"%"<<std::endl;
    std::cout<<"========================================================"<<std::endl;
    return 0;
}
