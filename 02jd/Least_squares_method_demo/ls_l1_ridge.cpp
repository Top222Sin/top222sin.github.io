// ============================================================
// LS_demo L1.3 岭回归与正则化
// 纯 C++17。(AᵀA+λI)x = Aᵀb —— 病态救星。
// 验证: Hilbert-7(条件数 ~1e8)无噪声还原:
//   法方程 LS 爆炸(误差 1e+1) vs 岭 λ=1e-7(误差 1e-2)
// 三个视角: 数值(条件数) / 统计(贝叶斯先验) / 机器学习(L2 正则)
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat eye(int n){ Mat I(n, vector<double>(n,0)); for(int i=0;i<n;++i) I[i][i]=1; return I; }
static Mat matmul(const Mat& A, const Mat& B){ int n=A.size(),k=A[0].size(),m=B[0].size();
    Mat C(n, vector<double>(m,0));
    for(int i=0;i<n;++i) for(int p=0;p<k;++p){double a=A[i][p]; if(a) for(int j=0;j<m;++j) C[i][j]+=a*B[p][j];}
    return C; }
static vector<double> matvec(const Mat& A, const vector<double>& x){ vector<double> y(A.size(),0);
    for(size_t i=0;i<A.size();++i) for(size_t j=0;j<x.size();++j) y[i]+=A[i][j]*x[j]; return y; }
static Mat addm(const Mat& A, const Mat& B){ Mat C(A);
    for(size_t i=0;i<A.size();++i) for(size_t j=0;j<A[0].size();++j) C[i][j]+=B[i][j]; return C; }
static Mat scalem(const Mat& A, double s){ Mat C(A);
    for(size_t i=0;i<A.size();++i) for(size_t j=0;j<A[0].size();++j) C[i][j]*=s; return C; }
static Mat transpose(const Mat& A){ int n=A.size(),m=A[0].size(); Mat T(m, vector<double>(n));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j]; return T; }
static Mat inv(const Mat& A){ int n=A.size(); Mat M(n, vector<double>(2*n,0));
    for(int i=0;i<n;++i){ for(int j=0;j<n;++j) M[i][j]=A[i][j]; M[i][n+i]=1; }
    for(int c=0;c<n;++c){ int p=c;
        for(int i=c;i<n;++i) if(fabs(M[i][c])>fabs(M[p][c])) p=i;
        swap(M[c],M[p]); double piv=M[c][c];
        for(int j=0;j<2*n;++j) M[c][j]/=piv;
        for(int i=0;i<n;++i) if(i!=c && M[i][c]!=0){ double f=M[i][c];
            for(int j=0;j<2*n;++j) M[i][j]-=f*M[c][j]; } }
    Mat R(n, vector<double>(n));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) R[i][j]=M[i][n+j];
    return R; }

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L1.3 岭回归与病态 ===\n");

    int n = 7;
    Mat H(n, vector<double>(n));
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) H[i][j] = 1.0/(i+j+1);
    vector<double> xt(n, 1.0);
    vector<double> b = matvec(H, xt);
    Mat Ht = transpose(H);
    Mat HtH = matmul(Ht, H);
    vector<double> Htb = matvec(Ht, b);
    // LS
    vector<double> x_ls = matvec(inv(HtH), Htb);
    double e_ls = 0; for (int i=0;i<n;++i) e_ls = fmax(e_ls, fabs(x_ls[i]-1.0));
    double nrm_ls = 0; for (double v : x_ls) nrm_ls += v*v; nrm_ls = sqrt(nrm_ls);
    // 岭
    double lam = 1e-7;
    vector<double> x_rd = matvec(inv(addm(HtH, scalem(eye(n), lam))), Htb);
    double e_rd = 0; for (int i=0;i<n;++i) e_rd = fmax(e_rd, fabs(x_rd[i]-1.0));
    double nrm_rd = 0; for (double v : x_rd) nrm_rd += v*v; nrm_rd = sqrt(nrm_rd);
    printf("  [demo] Hilbert-%d 无噪声还原: LS 误差=%.2e, 岭(λ=1e-7)=%.2e\n", n, e_ls, e_rd);
    printf("        系数范数: LS=%.1f → 岭=%.1f (压制爆炸)\n", nrm_ls, nrm_rd);
    chk("病态下岭回归远稳于 LS", e_rd < e_ls ? 0.0 : 1.0, 0.5);
    chk("岭压制系数范数", nrm_rd < nrm_ls ? 0.0 : 1.0, 0.5);
    printf("  [三视角] 数值: 条件数救星 | 统计: x~N(0,σ²/λ) 先验\n");
    printf("          机器学习: L2 正则=权重衰减 (λ↑ 欠拟合, λ↓ 过拟合)\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
