// ============================================================
// LS_demo L4.6 稀疏正则化: Lasso (L1 惩罚最小二乘)
// 纯 C++17。核心: min_x ½‖Ax−b‖² + λ‖x‖₁
//   - 近端梯度(ISTA/FISTA): 平滑部分梯度 ∇=Aᵀ(Ax−b), 步长 η=1/L(L=λ_max(AᵀA))
//     近端算子 = 软阈值 soft(z, ηλ); 有效阈值 T=ηλ=λ/L
//   - 坐标下降(独立参考解, 不同算法求同一 Lasso 解)
//   - 与 OLS / 岭回归(L1.3) 对比: L1 产生稀疏, L2 只收缩
//   - 工业: 高维特征选择 / 压缩感知(欠定恢复) / 稀疏系统辨识 / 异常检测
// 验证: 软阈值闭式 / 1D 闭式 / 稀疏恢复 / 大λ全零 / 2D 网格搜索 /
//       欠定压缩感知 / Lasso vs OLS 支撑对比
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

// ---- 固定种子 LCG (与受管 Python 验证脚本一致, 保证实例可复现) ----
static unsigned long long _seed = 12345;
static double urand(){
    _seed = _seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned long long)((_seed >> 33) & 0x7FFFFFFF) / (double)0x80000000;
}
static double gauss(){
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0*log(u1)) * cos(2.0*3.14159265358979323846*u2);
}

// ---- 矩阵/向量工具 ----
static vector<double> matvec(const vector<vector<double>>& A, const vector<double>& x){
    int m=(int)A.size(), n=(int)x.size();
    vector<double> r(m,0.0);
    for(int i=0;i<m;++i){ double s=0; for(int j=0;j<n;++j) s+=A[i][j]*x[j]; r[i]=s; }
    return r;
}
static vector<double> matTvec(const vector<vector<double>>& A, const vector<double>& r){
    int m=(int)A.size(), n=(int)A[0].size();
    vector<double> out(n,0.0);
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) out[j]+=A[i][j]*r[i];
    return out;
}
static double norm2(const vector<double>& v){
    double s=0; for(double x:v) s+=x*x; return sqrt(s);
}
static double norm2diff(const vector<double>& a, const vector<double>& b){
    double s=0; int n=(int)a.size();
    for(int j=0;j<n;++j){ double d=a[j]-b[j]; s+=d*d; }
    return sqrt(s);
}
// 幂迭代求 λ_max(AᵀA) -> Lipschitz 常数 L
static double maxEigAtA(const vector<vector<double>>& A, int iters=400){
    int n=(int)A[0].size();
    vector<double> v(n);
    for(int j=0;j<n;++j) v[j]=gauss();
    double nv=norm2(v); if(nv<1e-14) nv=1; for(int j=0;j<n;++j) v[j]/=nv;
    for(int _=0;_<iters;++_){
        vector<double> Av=matvec(A,v);
        vector<double> AtAv=matTvec(A,Av);
        double nv2=norm2(AtAv);
        if(nv2<1e-14) break;
        for(int j=0;j<n;++j) v[j]=AtAv[j]/nv2;
    }
    vector<double> Av=matvec(A,v);
    vector<double> AtAv=matTvec(A,Av);
    double num=0,den=0;
    for(int j=0;j<n;++j){ num+=v[j]*AtAv[j]; den+=v[j]*v[j]; }
    return den>1e-14 ? num/den : 1.0;
}
// 软阈值 (L1 近端算子)
static vector<double> softThresh(const vector<double>& z, double t){
    if(t<0) t=0;
    vector<double> out(z.size());
    for(size_t i=0;i<z.size();++i){
        double zi=z[i];
        if(zi>t) out[i]=zi-t;
        else if(zi<-t) out[i]=zi+t;
        else out[i]=0.0;
    }
    return out;
}
// ISTA: x_{k+1}=soft(x_k − η·Aᵀ(Ax_k−b), ηλ)
static vector<double> ista(const vector<vector<double>>& A, const vector<double>& b,
                           double lam, double L, int maxit, double tol, int* iters){
    int n=(int)A[0].size();
    vector<double> x(n,0.0);
    double eta=1.0/L;
    int it=0;
    for(; it<maxit; ++it){
        vector<double> Ax=matvec(A,x);
        vector<double> r(Ax.size());
        for(size_t i=0;i<Ax.size();++i) r[i]=Ax[i]-b[i];
        vector<double> g=matTvec(A,r);
        vector<double> z(n);
        for(int j=0;j<n;++j) z[j]=x[j]-eta*g[j];
        vector<double> xnew=softThresh(z, lam*eta);
        double d=norm2diff(xnew,x);
        x=xnew;
        if(d<tol) { if(iters)*iters=it+1; return x; }
    }
    if(iters)*iters=maxit;
    return x;
}
// FISTA: 带动量加速
static vector<double> fista(const vector<vector<double>>& A, const vector<double>& b,
                            double lam, double L, int maxit, double tol, int* iters){
    int n=(int)A[0].size();
    vector<double> x(n,0.0), y(n,0.0);
    double eta=1.0/L, tk=1.0;
    int it=0;
    for(; it<maxit; ++it){
        vector<double> Ay=matvec(A,y);
        vector<double> r(Ay.size());
        for(size_t i=0;i<Ay.size();++i) r[i]=Ay[i]-b[i];
        vector<double> g=matTvec(A,r);
        vector<double> z(n);
        for(int j=0;j<n;++j) z[j]=y[j]-eta*g[j];
        vector<double> xnew=softThresh(z, lam*eta);
        double tnext=(1.0+sqrt(1.0+4.0*tk*tk))/2.0;
        for(int j=0;j<n;++j) y[j]=xnew[j]+(tk-1.0)/tnext*(xnew[j]-x[j]);
        tk=tnext;
        double d=norm2diff(xnew,x);
        x=xnew;
        if(d<tol) { if(iters)*iters=it+1; return x; }
    }
    if(iters)*iters=maxit;
    return x;
}
// 坐标下降 Lasso (独立参考解)
static vector<double> lassoCD(const vector<vector<double>>& A, const vector<double>& b,
                              double lam, int maxit=3000, double tol=1e-9){
    int m=(int)A.size(), n=(int)A[0].size();
    vector<double> x(n,0.0);
    vector<double> An2(n,0.0);
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) An2[j]+=A[i][j]*A[i][j];
    for(int _=0;_<maxit;++_){
        double mx=0;
        for(int j=0;j<n;++j){
            if(An2[j]<1e-12) continue;
            double sj=0;
            for(int i=0;i<m;++i){
                double ax=0; for(int k=0;k<n;++k) ax+=A[i][k]*x[k];
                sj+=A[i][j]*(b[i]-ax+A[i][j]*x[j]);
            }
            double xnew=softThresh({sj/An2[j]}, lam/An2[j])[0];
            double d=fabs(xnew-x[j]); x[j]=xnew;
            if(d>mx) mx=d;
        }
        if(mx<tol) break;
    }
    return x;
}
// OLS 法方程 (高斯消元)
static vector<double> ols(const vector<vector<double>>& A, const vector<double>& b){
    int n=(int)A[0].size(), m=(int)A.size();
    vector<vector<double>> AtA(n, vector<double>(n,0.0));
    vector<double> Atb(n,0.0);
    for(int k=0;k<n;++k) for(int j=0;j<n;++j){ double s=0; for(int i=0;i<m;++i) s+=A[i][k]*A[i][j]; AtA[k][j]=s; }
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) Atb[j]+=A[i][j]*b[i];
    for(int c=0;c<n;++c){
        int p=c; for(int i=c;i<n;++i) if(fabs(AtA[i][c])>fabs(AtA[p][c])) p=i;
        swap(AtA[c],AtA[p]); swap(Atb[c],Atb[p]);
        for(int i=0;i<n;++i) if(i!=c){ double f=AtA[i][c]/AtA[c][c];
            for(int j=c;j<n;++j) AtA[i][j]-=f*AtA[c][j]; Atb[i]-=f*Atb[c]; }
    }
    vector<double> x(n);
    for(int i=0;i<n;++i) x[i]=Atb[i]/AtA[i][i];
    return x;
}
static double obj(const vector<vector<double>>& A, const vector<double>& b,
                  const vector<double>& x, double lam){
    vector<double> Ax=matvec(A,x);
    double r=0; for(size_t i=0;i<Ax.size();++i){ double e=Ax[i]-b[i]; r+=0.5*e*e; }
    for(double v:x) r+=lam*fabs(v);
    return r;
}

int main(){
    int pass=0, fail=0;
    auto chk=[&](const char* n,double e,double t){ bool o=e<t;
        printf("  %-42s err=%.3e tol=%.0e  %s\n",n,e,t,o?"PASS":"FAIL"); o?++pass:++fail; };
    printf("=== L4.6 Lasso 稀疏正则化 (ISTA/FISTA/坐标下降) ===\n");

    // ---- ① 软阈值闭式 ----
    printf("  [1] 软阈值闭式\n");
    struct { double z,t; } sc[] = {{0.5,0.3},{-1.2,0.5},{0.1,0.3},{2.0,0.5}};
    for(auto& c: sc){
        double st=softThresh({c.z}, c.t)[0];
        double exp = (fabs(c.z)>c.t)? copysign(1.0,c.z)*(fabs(c.z)-c.t) : 0.0;
        chk("soft 闭式一致", fabs(st-exp), 1e-12);
    }

    // ---- ② 1D Lasso 闭式 ----
    printf("  [2] 1D Lasso 闭式: min 0.5(ax−b)²+λ|x|\n");
    {
        double a=2.0, bb=1.0, lam=0.5, L1=a*a;
        double xstar = copysign(1.0,bb/a)*fmax(fabs(bb/a)-lam/(a*a),0.0);
        int it; vector<double> x1=ista({{a}},{bb},lam,L1,10000,1e-12,&it);
        chk("1D ISTA vs 闭式", fabs(x1[0]-xstar), 1e-6);
    }

    // ---- ③ 稀疏恢复 (n=20,m=40,k=3) ----
    printf("  [3] 稀疏恢复 n=20 m=40 k=3\n");
    {
        int n=20,m=40;
        vector<double> xtrue(n,0.0);
        xtrue[2]=3.0; xtrue[7]=2.0; xtrue[13]=-2.5;
        vector<vector<double>> A(m, vector<double>(n));
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) A[i][j]=gauss();
        vector<double> b(m);
        for(int i=0;i<m;++i){ double s=0; for(int j=0;j<n;++j) s+=A[i][j]*xtrue[j]; b[i]=s+gauss()*0.0008; }
        double L=maxEigAtA(A); double T=0.03; double lam=T*L;
        int iti,itf;
        vector<double> xis=ista(A,b,lam,L,5000,1e-9,&iti);
        vector<double> xfi=fista(A,b,lam,L,5000,1e-9,&itf);
        vector<double> xcd=lassoCD(A,b,lam);
        double err_i=norm2diff(xis,xtrue);
        double err_f=norm2diff(xfi,xtrue);
        double err_cd=norm2diff(xcd,xtrue);
        int zp=0; for(int j=0;j<n;++j) if(fabs(xtrue[j])<1e-9) ++zp;
        int nzc=0; for(int j=0;j<n;++j) if(fabs(xtrue[j])<1e-9 && fabs(xfi[j])<1e-3) ++nzc;
        int nnon=0; for(double v:xfi) if(fabs(v)>1e-3) ++nnon;
        printf("      L=%.2f T=lam/L=%.4f ISTA收敛it=%d FISTA收敛it=%d\n",L,lam/L,iti,itf);
        printf("      恢复误差 ISTA=%.3e FISTA=%.3e CD=%.3e  真零压零=%d/%d 非零=%d(真=3)\n",
               err_i,err_f,err_cd,nzc,zp,nnon);
        chk("FISTA 恢复误差<0.15", err_f, 0.15);
        chk("真零漏压<=1(即>=16/17)", (double)(zp-nzc), 1.5);
        chk("非零系数 3~5", fabs((double)nnon-3.0) > 2.0 ? 1.0 : 0.0, 0.5);
        chk("ISTA≈CD 一致", fabs(err_f-err_cd), 1e-2);
        // 固定 12 迭代目标对比
        vector<double> Ki=ista(A,b,lam,L,12,1e-9,nullptr);
        vector<double> Kf=fista(A,b,lam,L,12,1e-9,nullptr);
        double oi=obj(A,b,Ki,lam), of=obj(A,b,Kf,lam);
        printf("      [固定12迭代] 目标 ISTA=%.5f FISTA=%.5f (FISTA更低=%s)\n",oi,of, of<oi?"yes":"no");
        chk("FISTA 目标严格低于 ISTA", (oi-of) > 1e-6 ? 0.0 : 1.0, 0.5);
    }

    // ---- ④ 大 λ -> 全零 ----
    printf("  [4] 大 λ -> 全零\n");
    {
        int n=20,m=40;
        vector<double> xtrue(n,0.0); xtrue[2]=3.0; xtrue[7]=2.0; xtrue[13]=-2.5;
        vector<vector<double>> A(m, vector<double>(n));
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) A[i][j]=gauss();
        vector<double> b(m);
        for(int i=0;i<m;++i){ double s=0; for(int j=0;j<n;++j) s+=A[i][j]*xtrue[j]; b[i]=s+gauss()*0.0008; }
        double L=maxEigAtA(A); double lam=50.0*L;
        int it; vector<double> xb=ista(A,b,lam,L,5000,1e-9,&it);
        int nn=0; for(double v:xb) if(fabs(v)>1e-9) ++nn;
        printf("      λ=%.1f (T=%.1f) -> 非零=%d (应=0)\n",lam,lam/L,nn);
        chk("大λ非零个数=0", (double)nn, 0.5);
    }

    // ---- ⑤ 2D 网格搜索目标值 ----
    printf("  [5] 2D 网格搜索确认 ISTA 解最小化目标\n");
    {
        vector<vector<double>> A2={{1.0,0.5,0.3},{0.2,1.1,0.4}};
        vector<double> b2={1.3,0.9};
        double L2=maxEigAtA(A2); double lam2=0.4*L2;
        int it; vector<double> xg=ista(A2,b2,lam2,L2,5000,1e-9,&it);
        double best=obj(A2,b2,xg,lam2);
        double gmin=1e18;
        for(int i0=-40;i0<=40;++i0) for(int i1=-40;i1<=40;++i1) for(int i2=-40;i2<=40;++i2){
            vector<double> x={i0*0.05,i1*0.05,i2*0.05};
            double o=obj(A2,b2,x,lam2); if(o<gmin) gmin=o;
        }
        printf("      2D问题 ISTA目标=%.5f 网格最小=%.5f\n",best,gmin);
        chk("ISTA目标≈网格最小", fabs(best-gmin), 1e-2);
    }

    // ---- ⑥ 欠定压缩感知 (m<n) ----
    printf("  [6] 欠定压缩感知 m=18<n=40 k=3\n");
    {
        int n=40,m3=18;
        vector<double> x3(n,0.0); x3[3]=2.0; x3[17]=-1.5; x3[29]=1.0;
        vector<vector<double>> A3(m3, vector<double>(n));
        for(int i=0;i<m3;++i) for(int j=0;j<n;++j) A3[i][j]=gauss();
        vector<double> b3(m3);
        for(int i=0;i<m3;++i){ double s=0; for(int j=0;j<n;++j) s+=A3[i][j]*x3[j]; b3[i]=s+gauss()*0.0005; }
        double L3=maxEigAtA(A3); double lam3=0.018*L3;
        int it; vector<double> xcs=fista(A3,b3,lam3,L3,5000,1e-9,&it);
        int zp=0; for(int j=0;j<n;++j) if(fabs(x3[j])<1e-9) ++zp;
        int nzc=0; for(int j=0;j<n;++j) if(fabs(x3[j])<1e-9 && fabs(xcs[j])<1e-3) ++nzc;
        int nnon=0; for(double v:xcs) if(fabs(v)>1e-3) ++nnon;
        // 真支撑 ⊆ 恢复支撑
        bool sub=true;
        for(int t: {3,17,29}) if(fabs(xcs[t])<=1e-3) sub=false;
        printf("      T=%.4f FISTA收敛it=%d 真零压零=%d/%d 非零=%d(真=3) 真支撑⊆恢复=%s\n",
               lam3/L3,it,nzc,zp,nnon,sub?"yes":"no");
        chk("真零漏压<=1(即>=36/37)", (double)(zp-nzc), 1.5);
        chk("非零<=6", nnon>6 ? 1.0 : 0.0, 0.5);
        chk("真支撑⊆恢复支撑", sub?0.0:1.0, 0.5);
    }

    // ---- ⑦ Lasso vs OLS 稀疏性 (独立较噪数据) ----
    printf("  [7] Lasso vs OLS: 稀疏性/支撑对比 (较噪数据)\n");
    {
        int n=20,m=40;
        vector<double> x7(n,0.0); x7[2]=3.0; x7[7]=2.0; x7[13]=-2.5;
        vector<vector<double>> A7(m, vector<double>(n));
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) A7[i][j]=gauss();
        vector<double> b7(m);
        for(int i=0;i<m;++i){ double s=0; for(int j=0;j<n;++j) s+=A7[i][j]*x7[j]; b7[i]=s+gauss()*0.02; }
        double L7=maxEigAtA(A7); double lam7=0.05*L7;
        vector<double> xl=lassoCD(A7,b7,lam7);
        vector<double> xo=ols(A7,b7);
        int nn_ols=0; for(double v:xo) if(fabs(v)>1e-3) ++nn_ols;
        int nn_l=0;  for(double v:xl) if(fabs(v)>1e-3) ++nn_l;
        bool hit = fabs(xl[2])>1e-3 && fabs(xl[7])>1e-3 && fabs(xl[13])>1e-3;
        for(int j=0;j<n;++j) if(j!=2&&j!=7&&j!=13 && fabs(xl[j])>1e-3) hit=false;
        printf("      OLS 非零=%d(铺开) Lasso 非零=%d(稀疏) 支撑命中=%s\n",nn_ols,nn_l,hit?"yes":"no");
        chk("OLS 更铺开(非零更多)", nn_ols>nn_l ? 0.0 : 1.0, 0.5);
        chk("Lasso 支撑=真值{2,7,13}", hit?0.0:1.0, 0.5);
    }

    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail==0?0:1;
}
