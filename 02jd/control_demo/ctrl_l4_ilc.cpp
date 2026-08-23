// ============================================================
// control_demo L4：迭代学习控制 ILC ⭐（重复轨迹/产线批次）
// 纯 C++17。同一轨迹反复执行（机械臂、贴片机、晶圆台）：第 k 次误差
//   e_k = R - y_k 含"系统性"部分，可从 u_{k+1}=u_k+α·Gᵀ·e_k 学掉。
//   把对象写成有限维线性映射 Y=G·U，则 E_{k+1}=(I-αGᵀG)E_k，
//   收敛当且仅当谱半径 ρ=‖I-αGᵀG‖<1（即 0<α<2/σ_max²(GᵀG)）。
//   本 demo 用 Jacobi 求 GᵀG 特征值定 α，验证误差单调下降并 40 次迭代学会。
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
    int n=(int)A.size(), m=(int)A[0].size(); Mat T(m,Vec(n,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j]; return T;
}
static Mat matsub(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=A[i][j]-B[i][j]; return C;
}
static Mat scalemat(double s,const Mat&A){
    int n=(int)A.size(), m=(int)A[0].size(); Mat C(n,Vec(m,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) C[i][j]=s*A[i][j]; return C;
}
static Vec matvec(const Mat&A,const Vec&x){
    int n=(int)A.size(), m=(int)A[0].size(); Vec y(n,0.0);
    for(int i=0;i<n;++i){ double s=0.0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; } return y;
}
static Vec scalvec(double s,const Vec&x){ Vec y(x.size(),0.0); for(size_t i=0;i<x.size();++i) y[i]=s*x[i]; return y; }
static double vecnorm(const Vec&x){ double s=0.0; for(double v:x) s+=v*v; return sqrt(s); }

// 对称矩阵 Jacobi 特征分解 -> 特征值(降序)
static Vec eig_sym_jacobi(const Mat&S0){
    int n=(int)S0.size(); Mat A(n,Vec(n,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) A[i][j]=S0[i][j];
    for(int sweep=0;sweep<100;++sweep){
        double off=0.0;
        for(int p=0;p<n;++p) for(int q=p+1;q<n;++q) off+=A[p][q]*A[p][q];
        if(off<1e-18) break;
        for(int p=0;p<n;++p) for(int q=p+1;q<n;++q){
            if(fabs(A[p][q])<1e-15) continue;
            double theta=(A[q][q]-A[p][p])/(2.0*A[p][q]);
            double t=(fabs(theta)>1e-15)? copysign(1.0,theta)/(fabs(theta)+sqrt(theta*theta+1.0)) : 1.0;
            double c=1.0/sqrt(t*t+1.0), s=t*c;
            double App=c*c*A[p][p]+s*s*A[q][q]-2*s*c*A[p][q];
            double Aqq=s*s*A[p][p]+c*c*A[q][q]+2*s*c*A[p][q];
            for(int k=0;k<n;++k) if(k!=p && k!=q){
                double akp=c*A[k][p]-s*A[k][q], akq=s*A[k][p]+c*A[k][q];
                A[k][p]=akp; A[p][k]=akp; A[k][q]=akq; A[q][k]=akq;
            }
            A[p][p]=App; A[q][q]=Aqq; A[p][q]=0.0; A[q][p]=0.0;
        }
    }
    Vec eig(n,0.0); for(int i=0;i<n;++i) eig[i]=A[i][i];
    for(int a=0;a<n;++a) for(int b=a+1;b<n;++b) if(eig[b]>eig[a]) swap(eig[a],eig[b]);
    return eig;
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool ok=e<t; printf("  %-46s err=%.3e tol=%.0e  %s\n",n,e,t,ok?"PASS":"FAIL"); ok?++pass:++fail; };
    printf("=== L4：迭代学习控制 ILC（重复轨迹批次）===\n");

    const int N=16, M=6;
    double gv[6]={1.0,0.1,0.05,0.02,0.01,0.005};
    Mat G(N,Vec(N,0.0));
    for(int i=0;i<N;++i) for(int j=0;j<N;++j){ int d=i-j; if(d>=0 && d<M) G[i][j]=gv[d]; }
    Mat Gt=transpose(G);
    Mat GtG=matmul(Gt,G);
    Vec eig=eig_sym_jacobi(GtG);
    double s2_max=eig[0], s2_min=eig[(int)eig.size()-1];
    printf("  [demo] GᵀG 特征值范围: [%.4f, %.4f]\n", s2_min, s2_max);
    double alpha=0.98*2.0/(s2_min+s2_max);
    double rho=fmax(fabs(1.0-alpha*s2_min), fabs(1.0-alpha*s2_max));
    printf("  [demo] α=%.4f, 收敛因子 ρ=%.4f\n", alpha, rho);
    chk("收敛因子 ρ<1 (单调收敛)", rho-1.0, 1e-9);

    // 参考信号 R（确定性斜坡+正弦）
    Vec R(N,0.0);
    for(int k=0;k<N;++k) R[k]=0.5*(k/(N-1.0))+0.3*sin(2*M_PI*k/(N-1.0));
    // ILC: E_{j+1}=(I-αGᵀG)E_j, E^0=R
    Mat I=eyeM(N);
    Vec E=R; double norms[41]; norms[0]=vecnorm(E);
    for(int j=0;j<40;++j){ E=matvec(matsub(I, scalemat(alpha,GtG)), E); norms[j+1]=vecnorm(E); }
    printf("  [demo] ‖E⁰‖=%.4f, ‖E⁴⁰‖=%.3e\n", norms[0], norms[40]);
    bool mono=true; for(int j=0;j<40;++j) if(norms[j+1]>norms[j]*(1+1e-9)) mono=false;
    printf("  [demo] %s 迭代误差单调下降\n", mono?"":"不");
    chk("迭代误差单调下降", mono?0.0:1.0, 1e-9);
    chk("‖E⁴⁰‖<1e-3 (学会)", norms[40], 1e-3);
    chk("学会后误差比<0.6", norms[40]/norms[0]-0.6, 1e-9);

    // 学到的输入 U⁴⁰ 施加一次应复现 R
    Vec U(N,0.0); Vec E2=R;
    for(int j=0;j<40;++j){
        Vec grad=scalvec(alpha, matvec(Gt, E2));
        for(int k=0;k<N;++k) U[k]+=grad[k];
        E2=matvec(matsub(I, scalemat(alpha,GtG)), E2);
    }
    Vec Y=matvec(G, U);
    double errY=0.0; for(int k=0;k<N;++k){ double e=R[k]-Y[k]; errY+=e*e; } errY=sqrt(errY);
    printf("  [demo] 学习40次后施加U: ‖Y-R‖=%.3e\n", errY);
    chk("学到的输入复现参考", errY, 1e-2);

    printf("  [结论] ILC 把重复任务的系统性误差逐次学掉；收敛由 ρ=‖I-αGᵀG‖<1 保证。\n");
    printf("          模型越准 α 越接近 1/σ_max²，越少迭代即收敛；含未建模动态时需加 Q 滤波。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
