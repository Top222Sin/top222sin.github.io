// ============================================================================
//  cuda_l4_libs_cufft_cusparse_curand.cpp
//  L4（工业扩展）· CUDA 数学/稀疏/随机库族：cuFFT + cuSPARSE + cuRAND
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l4_libs_cufft_cusparse_curand.cpp -o libs && ./libs
//  （真机 cuFFT / cuSPARSE / cuRAND API 由 #ifdef __CUDACC__ 守护，g++ 下跳过，仅作教材）
//
//  把"调库篇"(L1.3 cuBLAS/cuDNN) 延伸到科学计算三大库族，CPU 端用确定性
//  算法忠实复现其语义，做到"无 GPU 也能验证库行为正确"：
//    cuFFT   —— 复迭 FFT + IFFT 往返 + 与朴素 DFT 一致（卷积定理顺带演示）
//    cuSPARSE—— CSR 稀疏存储 + SpMV + csr2dense 往返（与稠密 matvec 一致）
//    cuRAND  —— 确定性 LCG + Box-Muller 正态（可复现 + 分布正确）
// ============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>

typedef std::complex<double> cd;   // 复平面；host 模型用 double 保证确定性，真机 cuFFT 用 float（见 __CUDACC__）

// -------------------- 公共工具：确定性 LCG（与 Python 镜像同序）--------------------
// 状态递推：state = (state*1103515245 + 12345) & 0x7fffffff；返回 state/2147483647.0 ∈ [0,1)
struct Rng {
    long long s;
    explicit Rng(long long seed=20260820) : s(seed & 0x7fffffff) {}
    double next(){ s = (s*1103515245LL + 12345) & 0x7fffffff; return (double)s / 2147483647.0; }
};

// ============================ cuFFT 部分 ===================================
// 迭代 radix-2 Cooley-Tukey（原位），长度 N 必须为 2 的幂
static int bitrev(int x, int bits){ int r=0; for(int i=0;i<bits;++i){ r=(r<<1)|(x&1); x>>=1; } return r; }

static void fft_inplace(std::vector<cd>& a, bool inverse){
    const int n=(int)a.size(); int bits=0; while((1<<bits)<n) ++bits;
    for(int i=0;i<n;++i){ int j=bitrev(i,bits); if(j>i) std::swap(a[i],a[j]); }
    for(int len=2; len<=n; len<<=1){
        double ang = (inverse? 2.0 : -2.0)*M_PI/len;
        cd wlen(std::cos(ang), std::sin(ang));
        for(int i=0;i<n;i+=len){
            cd w(1.0,0.0);
            for(int k=0;k<len/2;++k){
                cd u=a[i+k], v=a[i+k+len/2]*w;
                a[i+k]=u+v; a[i+k+len/2]=u-v;
                w*=wlen;
            }
        }
    }
    if(inverse) for(int i=0;i<n;++i) a[i]/=cd((double)n,0.0);
}
static std::vector<cd> fft(const std::vector<cd>& x){ std::vector<cd> a(x); fft_inplace(a,false); return a; }
static std::vector<cd> ifft(const std::vector<cd>& X){ std::vector<cd> a(X); fft_inplace(a,true); return a; }

// 朴素 DFT（参考基准）
static std::vector<cd> dft(const std::vector<cd>& x){
    const int n=(int)x.size(); std::vector<cd> y(n,cd(0,0));
    for(int k=0;k<n;++k) for(int m=0;m<n;++m)
        y[k]+=x[m]*cd(std::cos(2*M_PI*k*m/n), -std::sin(2*M_PI*k*m/n));
    return y;
}
// 线性卷积（参考基准）
static std::vector<double> direct_conv(const std::vector<double>& a, const std::vector<double>& b){
    int na=(int)a.size(), nb=(int)b.size(); std::vector<double> c(na+nb-1,0.0);
    for(int i=0;i<na;++i) for(int j=0;j<nb;++j) c[i+j]+=a[i]*b[j];
    return c;
}

// ============================ cuSPARSE 部分 ===============================
// CSR 三件套：row_ptr, col_idx, val。SpMV: y = A·x
struct Csr {
    int rows, cols;
    std::vector<int> row_ptr;   // 长度 rows+1
    std::vector<int> col_idx;
    std::vector<double> val;
};
// 稠密 → CSR（跳过 0）
static Csr dense_to_csr(const std::vector<std::vector<double>>& D){
    Csr c; c.rows=(int)D.size(); c.cols=(int)D[0].size(); c.row_ptr.push_back(0);
    for(int i=0;i<c.rows;++i){ for(int j=0;j<c.cols;++j) if(D[i][j]!=0.0){ c.col_idx.push_back(j); c.val.push_back(D[i][j]); } c.row_ptr.push_back((int)c.val.size()); }
    return c;
}
static std::vector<double> csr_spmv(const Csr& A, const std::vector<double>& x){
    std::vector<double> y(A.rows,0.0);
    for(int i=0;i<A.rows;++i) for(int p=A.row_ptr[i];p<A.row_ptr[i+1];++p) y[i]+=A.val[p]*x[A.col_idx[p]];
    return y;
}
static std::vector<std::vector<double>> csr_to_dense(const Csr& A){
    std::vector<std::vector<double>> D(A.rows, std::vector<double>(A.cols,0.0));
    for(int i=0;i<A.rows;++i) for(int p=A.row_ptr[i];p<A.row_ptr[i+1];++p) D[i][A.col_idx[p]]=A.val[p];
    return D;
}

// ============================ cuRAND 部分 ================================
// 确定性正态：LCG 抽 uniform → Box-Muller 成对生成 N(0,1)
static void curand_normal(Rng& r, double& z0, double& z1){
    double u1=r.next(); if(u1<1e-12) u1=1e-12; double u2=r.next();
    double r2=std::sqrt(-2.0*std::log(u1));
    z0=r2*std::cos(2.0*M_PI*u2);
    z1=r2*std::sin(2.0*M_PI*u2);
}

// ============================ 主程序 + 验证 ===============================
int main(){
    int passed=0, failed=0;
    auto chk=[&](const char* name, bool ok){ printf("  %s %s\n", ok?"PASS":"FAIL", name); ok?++passed:++failed; };

    // ---------- cuFFT 验证 ----------
    const int N=16;
    std::vector<cd> xa(N);
    Rng rfft(20260820);
    for(int i=0;i<N;++i) xa[i]=cd(2.0*rfft.next()-1.0, 2.0*rfft.next()-1.0);   // 确定性复信号

    std::vector<cd> Xf=fft(xa), Xd=dft(xa);
    double e1=0.0; for(int k=0;k<N;++k) e1=std::max(e1, std::abs(Xf[k]-Xd[k]));
    chk("cuFFT: fft(x) == 朴素 DFT (max|diff|)", e1 < 1e-9);

    std::vector<cd> xr=ifft(Xf);
    double e2=0.0; for(int i=0;i<N;++i) e2=std::max(e2, std::abs(xr[i]-xa[i]));
    chk("cuFFT: ifft(fft(x)) == x (往返一致)", e2 < 1e-9);

    // 卷积定理（顺带演示，不计入主检查计数外的逻辑）
    std::vector<double> a={1,2,3,4}, b={0.5,0.5,0.5,0};
    std::vector<cd> A(N,cd(0,0)), B(N,cd(0,0));
    for(size_t i=0;i<a.size();++i) A[i]=cd(a[i],0);
    for(size_t i=0;i<b.size();++i) B[i]=cd(b[i],0);
    std::vector<cd> AF=fft(A), BF=fft(B);
    for(int i=0;i<N;++i) AF[i]*=BF[i];
    std::vector<cd> C=ifft(AF);
    std::vector<double> cref=direct_conv(a,b);
    double econv=0.0; for(size_t i=0;i<cref.size();++i) econv=std::max(econv, std::abs(C[i].real()-cref[i]));
    chk("cuFFT: 卷积定理 ifft(fft(a)·fft(b)) == 直接卷积", econv < 1e-9);

    // ---------- cuSPARSE 验证 ----------
    // 一个 6×6 稀疏矩阵（三对角 + 一处非对角），确定性构造
    std::vector<std::vector<double>> Dm(6, std::vector<double>(6,0.0));
    for(int i=0;i<6;++i){ Dm[i][i]=1.0+i*0.1; if(i+1<6) Dm[i][i+1]=0.2; if(i>0) Dm[i][i-1]=0.3; }
    Dm[2][4]=0.5;   // 一处非三对角非零
    Csr A_csr=dense_to_csr(Dm);

    std::vector<double> xsp(6); for(int i=0;i<6;++i) xsp[i]=0.1*i+0.05;
    std::vector<double> y_sp=csr_spmv(A_csr, xsp);
    std::vector<double> y_dense(6,0.0);
    for(int i=0;i<6;++i) for(int j=0;j<6;++j) y_dense[i]+=Dm[i][j]*xsp[j];
    double e3=0.0; for(int i=0;i<6;++i) e3=std::max(e3, std::abs(y_sp[i]-y_dense[i]));
    chk("cuSPARSE: CSR SpMV == 稠密 matvec", e3 < 1e-12);

    std::vector<std::vector<double>> Dback=csr_to_dense(A_csr);
    double e4=0.0; for(int i=0;i<6;++i) for(int j=0;j<6;++j) e4=std::max(e4, std::abs(Dback[i][j]-Dm[i][j]));
    chk("cuSPARSE: csr2dense 往返 == 原稠密矩阵", e4 < 1e-12);

    // ---------- cuRAND 验证 ----------
    const int NR=20000;
    Rng r1(20260820), r2(20260820);
    std::vector<double> z(NR,0.0); double mean=0.0, meanSq=0.0;
    double first_z0=0.0, first_z1=0.0;
    for(int i=0;i<NR;++i){ double z0,z1; curand_normal(r1,z0,z1); if(i==0){first_z0=z0; first_z1=z1;} z[i]=z0; mean+=z0; meanSq+=z0*z0; }
    mean/=NR; meanSq/=NR; double stdv=std::sqrt(meanSq-mean*mean);
    // 确定性：用第二个独立同种子 RNG 重抽，前若干值必须逐位一致
    double z0b,z1b; curand_normal(r2,z0b,z1b);
    bool det = (z0b==first_z0) && (z1b==first_z1);
    // 分布正确性（正态 N(0,1)：mean≈0, std≈1）—— 单侧上界用 max(0, tol-|x|)
    double e5=std::max(std::max(0.0, std::abs(mean)-0.03), std::max(0.0, std::abs(stdv-1.0)-0.03));
    chk("cuRAND: 确定性 + N(0,1) 分布(mean≈0,std≈1)", det && e5 < 1e-9);

    printf("=== %d passed, %d failed ===\n", passed, failed);

#ifdef __CUDACC__
    // ---- 真机 API 片段（仅 nvcc 下编译，作教材；g++ 跳过）----
    // cuFFT:
    //   cufftHandle plan; cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    //   cufftExecZ2Z(plan, (cufftDoubleComplex*)xa.data(), (cufftDoubleComplex*)Xf.data(), CUFFT_FORWARD);
    //   cufftExecZ2Z(plan, Xf.data(), xr.data(), CUFFT_INVERSE); cufftDestroy(plan);
    // cuSPARSE (CSR SpMV, cuSPARSE v11+):
    //   cusparseSpMatDescr_t matA; cusparseCreateCsr(&matA, rows, cols, nnz,
    //       d_rowptr, d_colidx, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    //   cusparseDnVecDescr_t vecX, vecY; cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_64F);
    //   cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG1, buffer);
    // cuRAND:
    //   curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    //   curandSetPseudoRandomGeneratorSeed(gen, 20260820ULL);
    //   curandGenerateNormalDouble(gen, d_z, NR, 0.0, 1.0);   // 等价于 Box-Muller
    //   curandDestroyGenerator(gen);
#endif
    return failed?1:0;
}
