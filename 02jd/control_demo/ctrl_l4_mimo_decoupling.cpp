// ============================================================
// control_demo L4：MIMO 静态解耦控制 ⭐（多变量产线/半导体）
// 纯 C++17。SISO 控制在多变量对象上直接套会"按下葫芦浮起瓢"：
//   2×2 对象 G(s) 含强交叉耦合，调好一路会把另一路带跑。
//   本 demo 用 DC 增益逆预补偿器 D = G(0)^{-1}，使 D·G(s) 在直流
//   严格对角化（D·G(0)=I），从而两路独立、交叉耦合被压制 >100×。
// 离散化采用 ZOH 等价：y(k+1)=Ay(k)+Bu(k)，DC 增益 K=(I-A)^{-1}B。
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
static Mat transpose(const Mat&A){
    int n=(int)A.size(), m=(int)A[0].size();
    Mat T(m,Vec(n,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j];
    return T;
}
static Mat matsub(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]-B[i][j]; return C;
}
static Vec matvec(const Mat&A,const Vec&x){
    int n=(int)A.size(), m=(int)A[0].size(); Vec y(n,0.0);
    for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; } return y;
}
static Mat scalemat(double s,const Mat&A){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=s*A[i][j]; return C;
}
// 通用逆（Gauss-Jordan, 小矩阵）
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
static double matnormF(const Mat&A){
    double s=0.0; for(auto&row:A) for(double v:row) s+=v*v; return sqrt(s);
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool ok=e<t; printf("  %-46s err=%.3e tol=%.0e  %s\n",n,e,t,ok?"PASS":"FAIL"); ok?++pass:++fail; };
    printf("=== L4：MIMO 静态解耦控制（2×2 多变量对象）===\n");

    // ---- 离散对象：ZOH 等价，DC 增益含强耦合 ----
    double dt=0.1, tau1=1.0, tau2=3.0;
    double a11=exp(-dt/tau1), a22=exp(-dt/tau2);
    Mat kDC={{1.0,0.5},{0.3,1.0/3.0}};
    double b11=kDC[0][0]*(1-a11), b12=kDC[0][1]*(1-a11);
    double b21=kDC[1][0]*(1-a22), b22=kDC[1][1]*(1-a22);
    Mat A={{a11,0.0},{0.0,a22}};
    Mat B={{b11,b12},{b21,b22}};
    Mat K=matmul(inv(matsub(eyeM(2),A)), B);   // DC 增益 K=(I-A)^{-1}B，应==kDC
    printf("  [demo] K = [[%.4f, %.4f],[%.4f, %.4f]]\n", K[0][0],K[0][1],K[1][0],K[1][1]);

    // ---- 解耦预补偿器 D = K^{-1} ----
    Mat D=inv(K);
    Mat DK=matmul(D,K);
    double resDK=matnormF(matsub(DK, eyeM(2)));
    printf("  [demo] D*K 残差 = %.2e\n", resDK);
    chk("解耦预补偿器 D*K==I", resDK, 1e-12);

    // ---- 无解耦：直接施加参考 r=[1,0] ----
    Vec r={1.0,0.0};
    Vec yC=matvec(K, r);                        // 稳态 = K*r
    double y2_coupled=yC[1];
    printf("  [demo] 无解耦 r=[1,0] -> y2_ss=%.4f (交叉耦合)\n", y2_coupled);
    chk("无解耦交叉响应 y2≈0.3", fabs(y2_coupled-0.3), 1e-6);

    // ---- 有解耦：u = D*r ----
    Vec uD=matvec(D, r);
    Vec yD=matvec(K, uD);
    double y2_dec=yD[1], y1_dec=yD[0];
    printf("  [demo] 解耦后 r=[1,0] -> y1_ss=%.4f, y2_ss=%.2e\n", y1_dec, y2_dec);
    chk("解耦后交叉响应 y2≈0", fabs(y2_dec), 1e-3);

    // ---- 数值仿真核对稳态 ----
    Vec y={0.0,0.0}; Vec u=uD;
    for(int k=0;k<2000;++k){
        Vec yn={ A[0][0]*y[0]+B[0][0]*u[0]+B[0][1]*u[1],
                 A[1][0]*y[0]+A[1][1]*y[1]+B[1][0]*u[0]+B[1][1]*u[1] };
        y=yn;
    }
    printf("  [demo] 数值仿真 y(2000)=[%.4f, %.2e]\n", y[0], y[1]);
    chk("数值仿真到达稳态 y1≈1", fabs(y[0]-1.0), 1e-2);
    chk("数值仿真到达稳态 y2≈0", fabs(y[1]), 1e-2);

    // ---- 交叉耦合比 ----
    double ratio_c=y2_coupled/fabs(yC[0]);
    double ratio_d=(fabs(y1_dec)>1e-12)? fabs(y2_dec)/fabs(y1_dec):0.0;
    printf("  [demo] 交叉耦合比 无解耦=%.3f, 解耦后=%.3e\n", ratio_c, ratio_d);
    chk("解耦后交叉耦合下降>100x", ratio_d-0.01*ratio_c, 1e-9);

    printf("  [结论] D=G(0)^{-1} 把多变量对象在直流严格对角化；每路据此独立整定，\n");
    printf("          交叉耦合被压制 >100×。对时变/非线性耦合需配动态解耦或更优逆(NGPC)。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
