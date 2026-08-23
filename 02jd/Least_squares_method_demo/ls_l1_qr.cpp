// ============================================================
// LS_demo L1.4 QR 分解与数值稳定性
// 纯 C++17。Householder QR: A=QR → x=R⁻¹Qᵀb(不解法方程!)
// 法方程条件数 κ(AᵀA)=κ(A)² —— 病态时灾难。
// 验证: ① Q 正交 ~1e-15、R 上三角;② Hilbert-10 无噪声还原:
//      法方程误差 4.8e+01 vs QR 6.1e-04
// ⚠ 坑(开发踩过): 解要用 Qᵀb 不是 Qb!
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat eye(int n){ Mat I(n, vector<double>(n,0)); for(int i=0;i<n;++i) I[i][i]=1; return I; }
static Mat transpose(const Mat& A){ int n=A.size(),m=A[0].size(); Mat T(m, vector<double>(n));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j]; return T; }
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

static void qr_householder(const Mat& A, Mat& R, Mat& Q) {
    int n = A.size(), m = A[0].size();
    R = A; Q = eye(n);
    for (int k = 0; k < min(n-1, m); ++k) {
        double nx = 0;
        for (int i = k; i < n; ++i) nx += R[i][k]*R[i][k];
        nx = sqrt(nx);
        if (nx < 1e-15) continue;
        double alpha = (R[k][k] >= 0) ? -nx : nx;
        vector<double> v(n, 0.0);
        v[k] = R[k][k] - alpha;
        for (int i = k+1; i < n; ++i) v[i] = R[i][k];   // 子数组第 i-k 个
        double nv = 0; for (double t : v) nv += t*t; nv = sqrt(nv);
        if (nv < 1e-15) continue;
        for (double& t : v) t /= nv;
        for (int j = 0; j < m; ++j) {
            double s = 0; for (int i = 0; i < n; ++i) s += v[i]*R[i][j];
            for (int i = 0; i < n; ++i) R[i][j] -= 2*s*v[i];
        }
        for (int i = 0; i < n; ++i) {
            double s = 0; for (int j = 0; j < n; ++j) s += Q[i][j]*v[j];
            for (int j = 0; j < n; ++j) Q[i][j] -= 2*s*v[j];
        }
    }
}
static vector<double> back_sub(const Mat& R, const vector<double>& b) {
    int m = R[0].size();
    vector<double> x(m, 0.0);
    for (int i = m-1; i >= 0; --i) {
        double s = b[i];
        for (int j = i+1; j < m; ++j) s -= R[i][j]*x[j];
        x[i] = s / R[i][i];
    }
    return x;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-40s err=%.3e tol=%.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
        ok ? ++pass : ++fail; };
    printf("=== L1.4 QR 分解与数值稳定性 ===\n");

    int n = 10;
    Mat H(n, vector<double>(n));
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) H[i][j] = 1.0/(i+j+1);
    vector<double> xt(n);
    for (int i=0;i<n;++i) xt[i] = 1.0/(i+1);
    vector<double> b = matvec(H, xt);
    Mat Ht = transpose(H);
    vector<double> x_ne = matvec(inv(matmul(Ht, H)), matvec(Ht, b));
    Mat R, Q;
    qr_householder(H, R, Q);
    vector<double> x_qr = back_sub(R, matvec(transpose(Q), b));   // Qᵀb!
    double e_ne = 0, e_qr = 0;
    for (int i=0;i<n;++i) { e_ne = fmax(e_ne, fabs(x_ne[i]-xt[i])); e_qr = fmax(e_qr, fabs(x_qr[i]-xt[i])); }
    printf("  [demo] Hilbert-%d 无噪声还原: 法方程误差=%.2e, QR 误差=%.2e\n", n, e_ne, e_qr);
    chk("QR 远稳于法方程(病态)", e_qr < e_ne ? 0.0 : 1.0, 0.5);
    // Q 正交 / R 上三角
    Mat QtQ = matmul(transpose(Q), Q);
    double orth = 0;
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) orth = fmax(orth, fabs(QtQ[i][j] - (i==j?1.0:0.0)));
    chk("Q 正交 (~1e-15)", orth, 1e-13);
    double upper = 0;
    for (int i=0;i<n;++i) for (int j=0;j<i;++j) upper = fmax(upper, fabs(R[i][j]));
    chk("R 上三角", upper, 1e-12);
    printf("  [结论] 法方程 κ² ;QR 只 κ。生产代码(Ceres/Eigen)全用 QR/SVD,\n");
    printf("        法方程只在黑板和 2×2 手算里出现。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
