// ============================================================
// LS_demo L4.1 传感器校准专题:IMU 六面校准 + 磁力计椭球拟合
// 纯 C++17。加计: 12 朝向 → 12 参数(M 矩阵 9 + bias 3)线性 LS;
//   模型 a_meas = M·g_dir + b(M 含刻度与非正交)
// 磁力计: 椭球拟合(对角) → 偏心=bias, RMS 半径=刻度(E[u²]=1/3)
// 相机标定/手眼标定见 cali_demo(非线性 LS,重投影/AX=XB)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 41;
static double urand(){ _seed=_seed*6364136223846793005ULL+1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }
static const double PI_ = 3.14159265358979323846;

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-38s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.1 IMU 校准 + 磁力计椭球拟合 ===\n");

    // ---- ① 加计 12 参数 ----
    const double g=9.81;
    double Mt[3][3]={{1.02,0.01,-0.005},{0.008,0.98,0.012},{-0.01,0.015,1.03}};
    double bt[3]={0.12,-0.08,0.05};
    double dirs[12][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {0.5,0.5,0.707},{-0.5,0.5,-0.707},{0.707,-0.5,0.5},{-0.707,-0.5,-0.5},
        {0.5,-0.707,-0.5},{-0.5,0.707,0.5}};
    // 法方程 12x12(高斯消元)
    int NP=12;
    vector<vector<double>> A(NP, vector<double>(NP,0));
    vector<double> rhs(NP,0);
    for (int d=0; d<12; ++d) {
        double nn=sqrt(dirs[d][0]*dirs[d][0]+dirs[d][1]*dirs[d][1]+dirs[d][2]*dirs[d][2]);
        double gd[3]={g*dirs[d][0]/nn, g*dirs[d][1]/nn, g*dirs[d][2]/nn};
        for (int i=0;i<3;++i) {
            double meas = Mt[i][0]*gd[0]+Mt[i][1]*gd[1]+Mt[i][2]*gd[2]+bt[i]+gauss()*0.005;
            double row[12]={0};
            row[3*i]=gd[0]; row[3*i+1]=gd[1]; row[3*i+2]=gd[2]; row[9+i]=1;
            for (int a=0;a<NP;++a){ rhs[a]+=row[a]*meas;
                for (int b=0;b<NP;++b) A[a][b]+=row[a]*row[b]; }
        }
    }
    // 消元
    for (int c=0;c<NP;++c){ int p=c;
        for(int i=c;i<NP;++i) if(fabs(A[i][c])>fabs(A[p][c])) p=i;
        swap(A[c],A[p]); swap(rhs[c],rhs[p]);
        for(int i=0;i<NP;++i) if(i!=c){ double f=A[i][c]/A[c][c];
            for(int j=c;j<NP;++j) A[i][j]-=f*A[c][j]; rhs[i]-=f*rhs[c]; } }
    vector<double> p(NP);
    for(int i=0;i<NP;++i) p[i]=rhs[i]/A[i][i];
    double err=0;
    for(int i=0;i<3;++i){ for(int j=0;j<3;++j) err=fmax(err,fabs(p[3*i+j]-Mt[i][j]));
        err=fmax(err,fabs(p[9+i]-bt[i])); }
    chk("加计 12 参数校准恢复", err, 0.01);
    printf("  [demo] M 误差=%.4f, b=[%.3f,%.3f,%.3f]\n", err, p[9],p[10],p[11]);

    // ---- ② 磁力计椭球 ----
    _seed=412;
    const double Bm=48.0;
    double bm[3]={5.0,-3.0,8.0}, sm[3]={1.1,0.95,1.05};
    vector<vector<double>> pts;
    for (int k=0;k<200;++k){
        double u[3]={gauss(),gauss(),gauss()};
        double nn=sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        vector<double> pt;
        for(int i=0;i<3;++i) pt.push_back(sm[i]*Bm*u[i]/nn+bm[i]+gauss()*0.5);
        pts.push_back(pt);
    }
    double berr=0, serr=0;
    for(int i=0;i<3;++i){
        double mx=-1e9, mn=1e9;
        for(auto&pt:pts){ mx=fmax(mx,pt[i]); mn=fmin(mn,pt[i]); }
        double b_est=0.5*(mx+mn);
        double rms=0;
        for(auto&pt:pts) rms+=(pt[i]-b_est)*(pt[i]-b_est);
        rms=sqrt(rms/pts.size());
        double s_est=rms*sqrt(3.0)/Bm;   // E[u²]=1/3
        berr=fmax(berr,fabs(b_est-bm[i]));
        serr=fmax(serr,fabs(s_est-sm[i]));
    }
    chk("磁力计椭球: 偏心 <1.5uT", berr, 1.5);
    chk("磁力计椭球: 刻度 <0.05", serr, 0.05);
    printf("  [demo] 磁力计 b_err=%.2fuT s_err=%.3f\n", berr, serr);
    printf("  [结论] 六面法=12 参数线性 LS;椭球拟合是它的非线性推广;\n");
    printf("        相机内外参/手眼标定 = 重投影/AX=XB 非线性 LS(cali_demo)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
