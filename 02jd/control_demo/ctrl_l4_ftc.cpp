// ============================================================
// control_demo L4：容错控制 FTC ⭐（执行器故障/网络攻击）
// 纯 C++17。多变量对象执行器会"部分失效"(ρ∈(0,1])或被卡死(ρ=0)。
//   两类典型恢复：
//   · Part A 部分失效：用 Λ^{-1} 预补偿把失效精确抵消（u=Λ^{-1}·v），
//     闭环恢复成健康 A-BK；不补偿则出现稳态误差。
//   · Part B 执行器2完全卡死：靠冗余重分配(虚拟执行器)，把健康指令的
//     输出贡献重新映射到存活执行器上，恢复设定值；不重分配则输出偏离。
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
static Mat matsub(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]-B[i][j]; return C;
}
static Mat matadd(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]+B[i][j]; return C;
}
static Mat scalemat(double s,const Mat&A){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=s*A[i][j]; return C;
}
static Vec matvec(const Mat&A,const Vec&x){
    int n=(int)A.size(), m=(int)A[0].size(); Vec y(n,0.0);
    for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; } return y;
}
static double matnormF(const Mat&A){ double s=0.0; for(auto&r:A) for(double v:r) s+=v*v; return sqrt(s); }
static double vecnorm(const Vec&x){ double s=0.0; for(double v:x) s+=v*v; return sqrt(s); }
// 通用逆（Gauss-Jordan）
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

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool ok=e<t; printf("  %-46s err=%.3e tol=%.0e  %s\n",n,e,t,ok?"PASS":"FAIL"); ok?++pass:++fail; };
    printf("=== L4：容错控制 FTC（执行器失效 + 冗余重分配）===\n");

    printf("=== Part A: 部分失效 ρ2=0.4, Λ^{-1} 预补偿精确恢复 ===\n");
    Mat A={{0.7,0.1},{0.0,0.6}};
    Mat B={{1.0,0.3},{0.2,1.0}};
    Mat A_des={{0.3,0.0},{0.0,0.2}};
    Mat K=matmul(inv(B), matsub(A, A_des));           // K=B^{-1}(A-A_des) -> A-BK=A_des
    Mat Acl_h=matsub(A, matmul(B,K));
    double eh=matnormF(matsub(Acl_h, A_des));
    printf("  [demo] 健康闭环 A-BK 残差=%.2e (应≈0)\n", eh);
    chk("健康闭环极点=期望", eh, 1e-12);
    Vec r={1.0,1.0};
    Vec u_ff=matvec(inv(B), matvec(matsub(eyeM(2), A_des), r));
    double rho2=0.4;
    Mat Lam={{1.0,0.0},{0.0,rho2}};
    Mat Blam=matmul(B, Lam);
    Mat Acl_nom=matsub(A, matmul(Blam, K));
    Vec xnom_star=matvec(inv(matsub(eyeM(2), Acl_nom)), matvec(Blam, u_ff));
    double err_nom=vecnorm(Vec{xnom_star[0]-r[0], xnom_star[1]-r[1]});
    Mat Lam_inv={{1.0,0.0},{0.0,1.0/rho2}};
    auto sim=[&](bool use_ftc, int Ns)->Vec{
        Vec x={1.0,1.0};
        for(int k=0;k<Ns;++k){
            Vec v={ -matvec(K,x)[0]+u_ff[0], -matvec(K,x)[1]+u_ff[1] };
            Vec u = use_ftc ? Vec{ Lam_inv[0][0]*v[0]+Lam_inv[0][1]*v[1],
                                   Lam_inv[1][0]*v[0]+Lam_inv[1][1]*v[1] } : v;
            x={ A[0][0]*x[0]+A[0][1]*x[1]+Blam[0][0]*u[0]+Blam[0][1]*u[1],
                A[1][0]*x[0]+A[1][1]*x[1]+Blam[1][0]*u[0]+Blam[1][1]*u[1] };
        }
        return x;
    };
    Vec x_ftc=sim(true, 300), x_nom=sim(false, 300);
    double err_ftc=vecnorm(Vec{x_ftc[0]-r[0], x_ftc[1]-r[1]});
    double err_nom_dyn=vecnorm(Vec{x_nom[0]-r[0], x_nom[1]-r[1]});
    printf("  [demo] 稳态误差 FTC=%.2e, 名义(无FTC)=%.4f\n", err_ftc, err_nom_dyn);
    chk("FTC 恢复跟踪(误差≈0)", err_ftc, 1e-2);
    chk("名义无FTC 明显降级(误差大)", 0.2-err_nom_dyn, 1e-9);
    chk("FTC 误差 << 名义误差", err_ftc-0.1*err_nom_dyn, 1e-9);

    printf("=== Part B: 执行器2完全卡死 ρ2=0 + 冗余重分配 ===\n");
    Mat A2={{0.8,0.0},{0.0,0.7}};
    Mat B2={{1.0,0.5},{0.5,1.0}};
    Mat C2={{1.0,0.0}};
    double rr=1.0;
    Vec u_h=matvec(inv(B2), matvec(matsub(eyeM(2), A2), Vec{rr,0.0}));
    Mat B_eff={{B2[0][0],0.0},{B2[1][0],0.0}};   // 执行器2 卡死 -> 第2列清零
    Vec x_nomB=matvec(inv(matsub(eyeM(2), A2)), matvec(B_eff, Vec{u_h[0],0.0}));
    double y_nomB=matvec(C2, x_nomB)[0];
    Vec u_ftcB={ (B2[0][0]*u_h[0]+B2[0][1]*u_h[1])/B2[0][0], 0.0 };
    Vec x_ftcB=matvec(inv(matsub(eyeM(2), A2)), matvec(B_eff, u_ftcB));
    double y_ftcB=matvec(C2, x_ftcB)[0];
    printf("  [demo] 输出: 健康目标=%.3f, FTC重分配=%.3f, 名义故障=%.3f\n", rr, y_ftcB, y_nomB);
    chk("FTC 输出恢复=设定值", fabs(y_ftcB-rr), 1e-6);
    chk("名义故障 输出偏离设定值", 0.1-fabs(y_nomB-rr), 1e-9);

    printf("  [结论] 部分失效用 Λ^{-1} 预补偿精确抵消；完全失效靠控制分配把指令\n");
    printf("          重映射到存活执行器。两类都需\"故障诊断/隔离\"先给出 Λ 或失效标志。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
