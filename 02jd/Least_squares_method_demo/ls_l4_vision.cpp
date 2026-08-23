// ============================================================
// LS_demo L4.4 计算机视觉:仿射 DLT + RANSAC + 光流
// 纯 C++17。① 4+ 点对仿射变换 DLT(线性 LS);
// ② RANSAC 抗 50% 外点;③ 1D Lucas-Kanade 光流(GN 最小化光度误差)
// 三维重建(三角化/BA)见 L3.4 capstone 与 cali/Jacobian_demo
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static unsigned long long _seed = 44;
static double urand(){ _seed=_seed*6364136223846793005ULL+1442695040888963407ULL;
    return (unsigned long long)((_seed>>33)&0x7FFFFFFF)/(double)0x80000000; }
static double gauss(){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*3.14159265358979323846*u2); }

static void solve6(vector<vector<double>> A, vector<double> b, vector<double>& x) {
    int n=6;
    for(int c=0;c<n;++c){ int p=c;
        for(int i=c;i<n;++i) if(fabs(A[i][c])>fabs(A[p][c])) p=i;
        swap(A[c],A[p]); swap(b[c],b[p]);
        for(int i=0;i<n;++i) if(i!=c){ double f=A[i][c]/A[c][c];
            for(int j=c;j<n;++j) A[i][j]-=f*A[c][j]; b[i]-=f*b[c]; } }
    x.assign(n,0);
    for(int i=0;i<n;++i) x[i]=b[i]/A[i][i];
}

int main() {
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-38s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.4 仿射 DLT + RANSAC + 光流 ===\n");

    double At[6]={1.1,0.1,0.2,0.9,5.0,-3.0};

    // ---- ① 仿射 DLT(30 点带噪) ----
    vector<vector<double>> rows; vector<double> ys;
    for(int i=0;i<30;++i){
        double x=urand()*10, y=urand()*10;
        double xp=At[0]*x+At[1]*y+At[4]+gauss()*0.03;
        double yp=At[2]*x+At[3]*y+At[5]+gauss()*0.03;
        rows.push_back({x,y,0,0,1,0}); ys.push_back(xp);
        rows.push_back({0,0,x,y,0,1}); ys.push_back(yp);
    }
    vector<vector<double>> A6(6, vector<double>(6,0)); vector<double> b6(6,0);
    for(size_t i=0;i<rows.size();++i) for(int a=0;a<6;++a){ b6[a]+=rows[i][a]*ys[i];
        for(int b2=0;b2<6;++b2) A6[a][b2]+=rows[i][a]*rows[i][b2]; }
    vector<double> a;
    solve6(A6,b6,a);
    double aerr=0;
    for(int i=0;i<6;++i) aerr=fmax(aerr,fabs(a[i]-At[i]));
    chk("仿射 DLT(10点带噪)", aerr, 0.02);
    printf("  [demo] DLT 误差=%.4f\n", aerr);

    // ---- ② RANSAC ----
    _seed=441;
    vector<vector<double>> data;
    for(int i=0;i<30;++i){
        double x=urand()*10, y=urand()*10;
        double xp=At[0]*x+At[1]*y+At[4], yp=At[2]*x+At[3]*y+At[5];
        if(urand()<0.5){ xp+=urand()*10-5; yp+=urand()*10-5; }
        data.push_back({x,y,xp,yp});
    }
    int best_in=0; vector<double> best_m;
    for(int it=0;it<60;++it){
        int i1=urand()*30, i2=urand()*30, i3=urand()*30;
        if(i1==i2||i2==i3||i1==i3) continue;
        vector<vector<double>> R2; vector<double> yy;
        for(int idx : {i1,i2,i3}){
            auto& d=data[idx];
            R2.push_back({d[0],d[1],0,0,1,0}); yy.push_back(d[2]);
            R2.push_back({0,0,d[0],d[1],0,1}); yy.push_back(d[3]);
        }
        vector<vector<double>> A2(6, vector<double>(6,0)); vector<double> b2(6,0);
        for(int i=0;i<6;++i) for(int aa=0;aa<6;++aa){ b2[aa]+=R2[i][aa]*yy[i];
            for(int bb=0;bb<6;++bb) A2[aa][bb]+=R2[i][aa]*R2[i][bb]; }
        vector<double> m;
        solve6(A2,b2,m);
        int inl=0;
        for(auto& d : data){
            double ex=m[0]*d[0]+m[1]*d[1]+m[4]-d[2];
            double ey=m[2]*d[0]+m[3]*d[1]+m[5]-d[3];
            if(ex*ex+ey*ey<0.01) ++inl;
        }
        if(inl>best_in){ best_in=inl; best_m=m; }
    }
    double rerr=0;
    for(int i=0;i<6;++i) rerr=fmax(rerr,fabs(best_m[i]-At[i]));
    chk("RANSAC 抗 50% 外点", rerr, 0.05);
    chk("RANSAC 内点数 ≥ 11", best_in>=11?0.0:1.0, 0.5);
    printf("  [demo] RANSAC: 模型误差=%.2e, 内点=%d/30\n", rerr, best_in);

    // ---- ③ 1D LK 光流 ----
    const double dx_true=0.7;
    auto I=[&](double x,double dx){ return sin(1.5*(x-dx))+0.3*cos(3.0*(x-dx)); };
    double dx=0.0;
    for(int it=0;it<20;++it){
        double sd=0, sn=0;
        for(int i=0;i<21;++i){
            double x=(i-10)*0.3;
            double r=I(x,dx_true)-I(x,dx);     // 当前帧 vs 扭曲模板
            double Ix=-1.5*cos(1.5*(x-dx))-0.9*sin(3.0*(x-dx));
            sd+=Ix*r; sn+=Ix*Ix;
        }
        dx+=sd/sn;
    }
    chk("光流位移恢复", fabs(dx-dx_true), 0.02);
    printf("  [demo] 光流: dx=%.4f(真值 0.7), GN 迭代最小化光度误差\n", dx);
    printf("  [结论] 图像配准=DLT/单应 LS(+RANSAC 剔误匹配);\n");
    printf("        光流=窗口内灰度误差平方最小的 GN;BA 见 L3.4/cali_demo。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
