// ============================================================
// pid_demo L4：MIMO 解耦 + IAE 自动整定（per-loop autotune）⭐（多变量回路自动整定）
// 纯 C++17。L4.7 用静态解耦器 D=K(0)^{-1} 把 2x2 对象在直流严格对角化,
//   再给每个对角回路一个"手工固定"的 PID(Kp=1,Ki=0.5,Kd=0.2)。
// 本 demo 把 L4.8 的"指标驱动寻优"思想套到 MIMO 上: 解耦后, 每个回路独立用
//   L4.6 的 IAE 指标作目标函数, 对 (Kp,Ki) 做网格搜索自动整定:
//     IAE = Σ|e_i(k)|·h  —— 越小整定越好
// 结论: 解耦后各回路可当 SISO 独立整定, 全系统 IAE 比统一手工整定降 ~63%;
//   但 IAE 单目标在稳定对象上无内点最优(总偏好更大增益), 故优调落在搜索窗的
//   激进端 —— 这正是工程整定要加超调/调节时间约束的原因(见 .md 局限一节)。
// 编译: g++ -O3 -std=c++17 pid_l4_mimo_autotune.cpp -o t && ./t
// 注: 本机无 C++ 编译器，.cpp 经"受管 Python 算法验证 + C++ 控制流镜像复核"交付。
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

// g=(Kp,Ki,Kd) 每回路; 返回四条全程轨迹
static void sim_mimo(bool decouple,const Vec&r,
                     double Kp1,double Ki1,double Kd1, double Kp2,double Ki2,double Kd2,
                     const Mat&D,const Mat&A,const Mat&B,double dt,double T,
                     Vec&y1,Vec&y2,Vec&e1,Vec&e2){
    int N=(int)(T/dt); Vec y={0.0,0.0}; Vec e_prev={0.0,0.0}, Iacc={0.0,0.0};
    y1.clear(); y2.clear(); e1.clear(); e2.clear();
    for(int k=0;k<N;++k){
        Vec e={r[0]-y[0], r[1]-y[1]};
        Iacc={Iacc[0]+e[0]*dt, Iacc[1]+e[1]*dt};
        Vec der={(e[0]-e_prev[0])/dt, (e[1]-e_prev[1])/dt};
        Vec u_loop={Kp1*e[0]+Ki1*Iacc[0]+Kd1*der[0], Kp2*e[1]+Ki2*Iacc[1]+Kd2*der[1]};
        Vec u = decouple? matvec(D,u_loop) : u_loop;
        Vec yn={ A[0][0]*y[0]+B[0][0]*u[0]+B[0][1]*u[1],
                 A[1][0]*y[0]+A[1][1]*y[1]+B[1][0]*u[0]+B[1][1]*u[1] };
        y=yn; e_prev=e;
        y1.push_back(y[0]); y2.push_back(y[1]); e1.push_back(e[0]); e2.push_back(e[1]);
    }
}
static double iae(const Vec&e,double dt){ double s=0.0; for(double v:e) s+=fabs(v)*dt; return s; }

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* nm,double err,double tol){
        bool ok=err<tol; printf("  %-44s err=%.3e tol=%.0e  %s\n",nm,err,tol,ok?"PASS":"FAIL");
        ok?++pass:++fail;
    };
    printf("=== L4：MIMO 解耦 + IAE 自动整定（per-loop autotune）===\n");

    double dt=0.05, T=15.0, tau1=1.0, tau2=2.0;
    Mat K={{1.0,0.3},{0.5,1.0}};
    double a11=exp(-dt/tau1), a22=exp(-dt/tau2);
    double b11=K[0][0]*(1-a11), b12=K[0][1]*(1-a11);
    double b21=K[1][0]*(1-a22), b22=K[1][1]*(1-a22);
    Mat A={{a11,0.0},{0.0,a22}}, B={{b11,b12},{b21,b22}};
    Mat Kss=matmul(inv(matsub(eyeM(2),A)), B);
    Mat D=inv(Kss);
    chk("DC增益 Kss≈K(残差小)", matnormF(matsub(Kss,K)), 1e-9);
    chk("D*Kss==I(解耦)", matnormF(matsub(matmul(D,Kss),eyeM(2))), 1e-12);

    double NOM_Kp=1.0, NOM_Ki=0.5, NOM_Kd=0.2;   // L4.7 手工整定
    double Kp_g[]={0.5,0.8,1.1,1.4,1.7,2.0,2.3}, Ki_g[]={0.3,0.5,0.7,0.9,1.1,1.3,1.5};
    int G=7;
    Vec y1d,y2d,e1d,e2d;

    // ---- 回路1 自动整定: r=[1,0], 回路2 固定 NOM ----
    double best1=1e30; double Kp1_opt=0.0,Ki1_opt=0.0;
    for(int a=0;a<G;++a){ double Kp=Kp_g[a];
        for(int b=0;b<G;++b){ double Ki=Ki_g[b];
            sim_mimo(true,{1.0,0.0}, Kp,Ki,NOM_Kd, NOM_Kp,NOM_Ki,NOM_Kd, D,A,B,dt,T, y1d,y2d,e1d,e2d);
            double j=iae(e1d,dt);
            if(j<best1){ best1=j; Kp1_opt=Kp; Ki1_opt=Ki; }
        }
    }
    // ---- 回路2 自动整定: r=[0,1], 回路1 固定 NOM ----
    double best2=1e30; double Kp2_opt=0.0,Ki2_opt=0.0;
    for(int a=0;a<G;++a){ double Kp=Kp_g[a];
        for(int b=0;b<G;++b){ double Ki=Ki_g[b];
            sim_mimo(true,{0.0,1.0}, NOM_Kp,NOM_Ki,NOM_Kd, Kp,Ki,NOM_Kd, D,A,B,dt,T, y1d,y2d,e1d,e2d);
            double j=iae(e2d,dt);
            if(j<best2){ best2=j; Kp2_opt=Kp; Ki2_opt=Ki; }
        }
    }
    // 固定 NOM 基准
    Vec y1n,y2n,e1n,e2n;
    sim_mimo(true,{1.0,0.0}, NOM_Kp,NOM_Ki,NOM_Kd, NOM_Kp,NOM_Ki,NOM_Kd, D,A,B,dt,T, y1n,y2n,e1n,e2n);
    double iae1_nom=iae(e1n,dt);
    sim_mimo(true,{0.0,1.0}, NOM_Kp,NOM_Ki,NOM_Kd, NOM_Kp,NOM_Ki,NOM_Kd, D,A,B,dt,T, y1n,y2n,e1n,e2n);
    double iae2_nom=iae(e2n,dt);

    printf("  回路1 自动 (Kp,Ki)=(%.1f,%.1f) IAE1=%.4f  (NOM=%.4f, 降%.1f%%)\n",
           Kp1_opt,Ki1_opt,best1,iae1_nom,(iae1_nom-best1)/iae1_nom*100.0);
    printf("  回路2 自动 (Kp,Ki)=(%.1f,%.1f) IAE2=%.4f  (NOM=%.4f, 降%.1f%%)\n",
           Kp2_opt,Ki2_opt,best2,iae2_nom,(iae2_nom-best2)/iae2_nom*100.0);

    // ---- 全系统: 用自动整定增益, 各自阶跃 ----
    double Kd_t=NOM_Kd;
    sim_mimo(true,{1.0,0.0}, Kp1_opt,Ki1_opt,Kd_t, Kp2_opt,Ki2_opt,Kd_t, D,A,B,dt,T, y1d,y2d,e1d,e2d);
    double IAE1_t=iae(e1d,dt), y1_ss_a=y1d.back(), y2_ss_a=y2d.back();
    double peak_y2_s1=0.0; for(double v:y2d) if(fabs(v)>peak_y2_s1) peak_y2_s1=fabs(v);
    sim_mimo(true,{0.0,1.0}, Kp1_opt,Ki1_opt,Kd_t, Kp2_opt,Ki2_opt,Kd_t, D,A,B,dt,T, y1d,y2d,e1d,e2d);
    double IAE2_t=iae(e2d,dt), y2_ss_b=y2d.back(), y1_ss_b=y1d.back();
    double peak_y1_s2=0.0; for(double v:y1d) if(fabs(v)>peak_y1_s2) peak_y1_s2=fabs(v);
    double IAE_fixed=iae1_nom+iae2_nom, IAE_tuned=IAE1_t+IAE2_t;
    printf("  全系统 IAE 总和: 固定%.4f -> 自动%.4f (降%.1f%%)\n",
           IAE_fixed,IAE_tuned,(IAE_fixed-IAE_tuned)/IAE_fixed*100.0);

    // ---- 检验 ----
    chk("LOOP1 自动 IAE1 < 固定 NOM IAE1", max(0.0,IAE1_t-iae1_nom), 1e-6);
    chk("LOOP2 自动 IAE2 < 固定 NOM IAE2", max(0.0,IAE2_t-iae2_nom), 1e-6);
    chk("全系统 IAE 总和 自动 < 固定", max(0.0,IAE_tuned-IAE_fixed), 1e-6);
    chk("回路1整定增益已偏离NOM(Kp变了>0.05)", max(0.0,0.05-fabs(Kp1_opt-NOM_Kp)), 1e-9);
    chk("回路1阶跃 稳态 y1_ss≈1", fabs(y1_ss_a-1.0), 0.02);
    chk("回路1阶跃 交叉耦合 y2_ss≈0", fabs(y2_ss_a), 0.02);
    chk("回路1阶跃 瞬态交叉耦合仍被压低(<0.03)", max(0.0,peak_y2_s1-0.03), 1e-6);
    chk("回路2阶跃 稳态 y2_ss≈1", fabs(y2_ss_b-1.0), 0.02);
    chk("回路2阶跃 交叉耦合 y1_ss≈0", fabs(y1_ss_b), 0.02);
    chk("回路2阶跃 瞬态交叉耦合仍被压低(<0.03)", max(0.0,peak_y1_s2-0.03), 1e-6);

    printf("  [结论] 解耦后每回路可当 SISO 用 L4.6 IAE 独立整定, 全系统 IAE 降 %.0f%%;\n",
           (IAE_fixed-IAE_tuned)/IAE_fixed*100.0);
    printf("           但 IAE 单目标在稳定对象上无内点最优(总偏好更大增益), 优调落在搜索窗激进端;\n");
    printf("           工程整定须加超调/调节时间约束 —— 见 .md 局限一节。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
