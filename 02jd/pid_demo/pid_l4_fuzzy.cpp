// ============================================================
// pid_demo L4：模糊 PID (Fuzzy PID) ⭐（暖通/家电/高精度）
// 纯 C++17。固定增益 PID 在强欠阻尼对象上易大超调; 模糊 PID 在线用 e,ec 经
//   模糊规则库调节 Kp/Ki/Kd: 大误差降 Kp(防超调)、增 Ki(消稳态)、按需调 Kd(阻尼)。
// 推理: 三角隶属(e,ec∈[-3,3]) -> 7x7 规则(ΔKp/ΔKi/ΔKd) -> 重心法(CoG)解模糊。
// 编译: g++ -O3 -std=c++17 pid_l4_fuzzy.cpp -o t && ./t
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
static Mat matadd(const Mat&A,const Mat&B){ int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]+B[i][j]; return C; }
static Mat scalemat(double s,const Mat&A){ int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=s*A[i][j]; return C; }
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

// 模糊集合: NB,NM,NS,ZO,PS,PM,PB -> 中心 -3..3
static const double CEN[7]={-3,-2,-1,0,1,2,3};
static double trimf(double x,double c,double w){ double d=fabs(x-c)/w; return d<1.0? 1.0-d : 0.0; }

// 7x7 规则库(存后继语言索引 0..6)
static const int TAB_KP[7][7]={
    {6,6,5,5,4,3,3},{6,6,5,4,4,3,2},{5,5,5,4,3,2,2},
    {5,5,4,3,2,1,1},{4,4,3,2,2,1,1},{4,3,2,1,1,1,0},{3,3,2,1,1,0,0}};
static const int TAB_KI[7][7]={
    {0,0,1,1,2,3,3},{0,0,1,2,2,3,3},{0,1,2,2,3,4,4},
    {1,1,2,3,4,5,5},{1,2,3,4,4,5,6},{3,3,4,4,5,6,6},{3,3,4,5,5,6,6}};
static const int TAB_KD[7][7]={
    {4,2,0,0,0,1,4},{4,2,0,1,1,2,3},{3,2,1,1,2,2,3},
    {3,2,2,2,2,2,3},{3,3,3,3,3,3,3},{6,2,4,4,4,4,6},{6,5,5,5,4,4,6}};

// 重心法解模糊: 返回 ΔKp/ΔKi/ΔKd 的归一化连续值
static double fuzzy_out(double e_norm,double ec_norm,const int tab[7][7]){
    double num=0.0, den=0.0;
    for(int i=0;i<7;++i){
        double me=trimf(e_norm, CEN[i], 1.0); if(me<=0.0) continue;
        for(int j=0;j<7;++j){
            double mc=trimf(ec_norm, CEN[j], 1.0); if(mc<=0.0) continue;
            double s=me<mc? me:mc;
            int cidx=tab[i][j];
            num+=s*CEN[cidx]; den+=s;
        }
    }
    return den>1e-12? num/den : 0.0;
}

static void tustin_ss(const Mat&Ac,const Mat&Bc,const Vec&Cc,double dt,
                      Mat&Ad,Mat&Bd,Vec&Cc2){
    int n=(int)Ac.size(); Mat I=eyeM(n);
    Mat M=matsub(I, scalemat(dt/2.0, Ac)); Mat Minv=inv(M);
    Ad=matmul(Minv, matadd(I, scalemat(dt/2.0, Ac)));
    Bd=matmul(Minv, scalemat(dt, Bc)); Cc2=Cc;
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* nm,double err,double tol){
        bool ok=err<tol; printf("  %-38s err=%.3e tol=%.0e  %s\n",nm,err,tol,ok?"PASS":"FAIL");
        ok?++pass:++fail;
    };
    printf("=== L4：模糊 PID（e,ec 模糊调节 Kp/Ki/Kd）===\n");

    // 推理引擎自检
    double dkp_big=fuzzy_out(3.0,0.0,TAB_KP);   // 大正误差 -> 降 Kp 防超调
    double dkp_neg=fuzzy_out(-3.0,0.0,TAB_KP);
    double dkp_zero=fuzzy_out(0.0,0.0,TAB_KP);
    double dki_pos=fuzzy_out(3.0,0.0,TAB_KI);
    printf("  推理自检: ΔKp(e=3,0)=%.2f  ΔKp(e=-3,0)=%.2f  ΔKp(0,0)=%.2f  ΔKi(e=3,0)=%.2f\n",
           dkp_big,dkp_neg,dkp_zero,dki_pos);
    chk("e=+3,ec=0 -> ΔKp<0(大误差降Kp防超调)", max(0.0, dkp_big+1.5), 1e-6);
    chk("e=-3,ec=0 -> ΔKp>0(对称)",          max(0.0, 1.5-dkp_neg), 1e-6);
    chk("e=0,ec=0 -> ΔKp≈0",                fabs(dkp_zero), 0.05);
    chk("e=+3 -> ΔKi>0(增积分消稳态)",       max(0.0, -dki_pos), 1e-6);

    // 闭环: 强欠阻尼对象 G(s)=1/(s^2+0.4s+1) (zeta=0.2)
    double dt=0.02, T=20.0, r=1.0;
    Mat Ac={{0.0,1.0},{-1.0,-0.4}}; Mat Bc={{0.0},{1.0}}; Vec Cc={1.0,0.0};
    Mat Ad,Bd; Vec Cc2; tustin_ss(Ac,Bc,Cc,dt,Ad,Bd,Cc2);
    double Kp0=2.2,Ki0=1.4,Kd0=0.4;
    double ge=3.0,gec=3.0,kps=0.50,kis=0.15,kds=0.12;

    auto sim_fuzzy=[&](bool use_fuzzy,Vec&y,Vec&e){
        int N=(int)(T/dt); int n=(int)Ad.size();
        Vec x(n,0.0); y.clear(); e.clear();
        double ep=0.0, Iacc=0.0;
        for(int k=0;k<N;++k){
            double yk=0.0; for(int i=0;i<n;++i) yk+=Cc2[i]*x[i];
            double ek=r-yk; e.push_back(ek);
            Iacc+=ek*dt; double der=(ek-ep)/dt;
            double kp=Kp0,ki=Ki0,kd=Kd0;
            if(use_fuzzy){
                double en=ek*ge, ecn=der*gec;
                double dkp=kps*fuzzy_out(en,ecn,TAB_KP);
                double dki=kis*fuzzy_out(en,ecn,TAB_KI);
                double dkd=kds*fuzzy_out(en,ecn,TAB_KD);
                kp=Kp0+dkp; if(kp<0)kp=0; ki=Ki0+dki; if(ki<0)ki=0; kd=Kd0+dkd; if(kd<0)kd=0;
            }
            double u=kp*ek+ki*Iacc+kd*der;
            if(u>5.0)u=5.0; if(u<-5.0)u=-5.0;
            Vec xn(n,0.0);
            for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<n;++j) s+=Ad[i][j]*x[j]; xn[i]=s; }
            for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<(int)Bd[0].size();++j) s+=Bd[i][j]*u; xn[i]+=s; }
            x=xn; ep=ek; y.push_back(yk);
        }
    };

    Vec yS,eS,yF,eF;
    sim_fuzzy(false,yS,eS); sim_fuzzy(true,yF,eF);
    double iaeS=0.0; for(double v:eS) iaeS+=fabs(v)*dt;
    double iaeF=0.0; for(double v:eF) iaeF+=fabs(v)*dt;
    double osS=0.0; for(double v:yS) if(v-r>osS) osS=v-r;
    double osF=0.0; for(double v:yF) if(v-r>osF) osF=v-r;
    printf("  固定PID  IAE=%.4f 超调=%.4f\n", iaeS, osS);
    printf("  模糊PID  IAE=%.4f 超调=%.4f\n", iaeF, osF);
    chk("模糊PID 超调 < 固定PID 超调", max(0.0, osF-osS), 1e-3);
    chk("模糊PID IAE < 固定PID IAE",   max(0.0, iaeF-iaeS), 1e-3);
    chk("模糊PID 稳态误差<0.02",       fabs(eF.back()), 0.02);

    printf("  [结论] 模糊 PID 大误差降 Kp/增 Ki, 超调 %.4f->%.4f, IAE %.4f->%.4f(更柔顺, 适合暖通/家电)。\n", osS,osF,iaeS,iaeF);
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
