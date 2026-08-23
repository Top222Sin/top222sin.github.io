// ============================================================
// LS_demo L4.5 自动驾驶:ICP + 车道线 + 滑窗 LS
// 纯 C++17。① 2D ICP(最近点对应已知,Kabsch/LS 求 R,t 一步收敛);
// ② 车道线三次多项式拟合(归一化 Vandermonde);③ 滑窗 LS 平滑降噪
// ⚠ 坑(开发踩过): 2D Kabsch 角度 = atan2(sxy−syx, sxx+syy)(方向别反)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 45;
static double urand(){ _seed=_seed*6364136223846793005ULL+1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-38s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.5 ICP + 车道线 + 滑窗 LS ===\n");

    // ---- ① ICP ----
    double th=0.3, cx=cos(th), sx=sin(th);
    double R[2][2]={{cx,-sx},{sx,cx}}, t0v=1.5, t1v=-0.8;
    vector<vector<double>> ref, src;
    for(int i=0;i<60;++i){
        double u0=gauss(), u1=gauss();
        ref.push_back({u0,u1});
        src.push_back({R[0][0]*u0+R[0][1]*u1+t0v+gauss()*0.02,
                       R[1][0]*u0+R[1][1]*u1+t1v+gauss()*0.02});
    }
    auto icp_step=[&](vector<vector<double>>& s){
        int n=s.size();
        double cs0=0,cs1=0,cr0=0,cr1=0;
        for(int i=0;i<n;++i){ cs0+=s[i][0];cs1+=s[i][1];cr0+=ref[i][0];cr1+=ref[i][1]; }
        cs0/=n;cs1/=n;cr0/=n;cr1/=n;
        double sxx=0,sxy=0,syx=0,syy=0;
        for(int i=0;i<n;++i){
            double a0=s[i][0]-cs0,a1=s[i][1]-cs1,b0=ref[i][0]-cr0,b1=ref[i][1]-cr1;
            sxx+=a0*b0; sxy+=a0*b1; syx+=a1*b0; syy+=a1*b1;
        }
        double theta=atan2(sxy-syx, sxx+syy);   // Kabsch 2D(src→ref)
        double c2=cos(theta), s2=sin(theta);
        for(int i=0;i<n;++i){
            double x=s[i][0], y=s[i][1];
            s[i][0]=c2*x-s2*y+(cr0-(c2*cs0-s2*cs1));
            s[i][1]=s2*x+c2*y+(cr1-(s2*cs0+c2*cs1));
        }
    };
    icp_step(src); icp_step(src);
    double th_res=0, t_res=0;
    for(int i=0;i<60;++i){
        th_res=fmax(th_res,hypot(src[i][0]-ref[i][0],src[i][1]-ref[i][1]));
    }
    chk("ICP 收敛(点距残差<0.08)", th_res, 0.08);
    printf("  [demo] ICP 两步后最大点距=%.4f(噪声 σ=0.02)\n", th_res);

    // ---- ② 车道线 ----
    _seed=451;
    double ct[4]={0.0,0.0,0.15,-0.008};
    vector<vector<double>> A4(4, vector<double>(4,0));
    vector<double> b4(4,0);
    for(int i=0;i<40;++i){
        double x=i*2.0;
        double y=ct[0]+ct[1]*x+ct[2]*x*x+ct[3]*x*x*x+gauss()*0.3;
        double u=x/40.0;
        double r[4]={1,u,u*u,u*u*u};
        for(int a=0;a<4;++a){ b4[a]+=r[a]*y;
            for(int b=0;b<4;++b) A4[a][b]+=r[a]*r[b]; }
    }
    for(int c=0;c<4;++c){ int p=c;
        for(int i=c;i<4;++i) if(fabs(A4[i][c])>fabs(A4[p][c])) p=i;
        swap(A4[c],A4[p]); swap(b4[c],b4[p]);
        for(int i=0;i<4;++i) if(i!=c){ double f=A4[i][c]/A4[c][c];
            for(int j=c;j<4;++j) A4[i][j]-=f*A4[c][j]; b4[i]-=f*b4[c]; } }
    double aa[4];
    for(int i=0;i<4;++i) aa[i]=b4[i]/A4[i][i];
    double c_est[4]={aa[0], aa[1]/40, aa[2]/1600, aa[3]/64000};
    double cerr=0;
    for(int i=0;i<4;++i) cerr=fmax(cerr,fabs(c_est[i]-ct[i]));
    chk("车道线三次拟合", cerr, 0.15);
    printf("  [demo] 车道线: c=[%.4f,%.4f,%.4f,%.4f](真值 0,0,0.15,-0.008)\n",
           c_est[0],c_est[1],c_est[2],c_est[3]);

    // ---- ③ 滑窗 LS ----
    _seed=452;
    vector<double> sig;
    for(int i=0;i<100;++i) sig.push_back(2.0+0.01*i+gauss()*0.3);
    double e2pt=0, e2sw=0;
    for(int i=4;i<96;++i){
        double truev=2.0+0.01*i;
        e2pt+=(sig[i]-truev)*(sig[i]-truev);
        double mu=0;
        for(int j=i-4;j<=i+4;++j) mu+=sig[j];
        mu/=9;
        e2sw+=(mu-truev)*(mu-truev);
    }
    chk("滑窗 LS 降噪(<0.6×单点)", e2sw<0.6*e2pt?0.0:1.0, 0.5);
    printf("  [demo] 滑窗降噪 %.1f%%(窗口=局部常值 LS)\n", 100*(1-e2sw/e2pt));
    printf("  [结论] ICP='最近点+LS'交替(对应未知时的非线性版);\n");
    printf("        滑动窗口优化=窗口内残差联合 LS(视觉里程计/VIO 核心,\n");
    printf("        MSCKF/图优化版见 KF_demo L3.6)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
