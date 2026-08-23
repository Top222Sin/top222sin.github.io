// jac_l2_manipulability.cpp
// ============================================================================
// L2.2 可操作度（Manipulability）与奇异规避
// ----------------------------------------------------------------------------
// 平面二连杆雅可比 J(q) (2x2)。可操作度 w = sqrt(det(J J^T))：衡量末端在各方向
// 施力/达到速度能力的"各向同性"。w→0 即奇异位形（列向量平行，维数坍缩）。
//   · 验证：好构型 w 较大；完全伸展/折叠(q=(0,0)或(0,π))时 w≈0（奇异）。
//   · 最小奇异值 σ_min = sqrt(λ_min(JJ^T))，奇异时 →0。
//   · 阻尼最小二乘(DLS)：q̇ = J^T(JJ^T+λ²I)^{-1} ẋ 在奇异处仍有限（普通伪逆 λ=0 发散）。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_manipulability.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;
static const double L1=1.0, L2=1.0;

static M Jfk(const V& q){
    double t1=q[0], t2=q[0]+q[1];
    return { { -L1*sin(t1)-L2*sin(t2), -L2*sin(t2) },
             {  L1*cos(t1)+L2*cos(t2),  L2*cos(t2) } };
}
static M transpose(const M& A){ int m=A.size(),n=A[0].size(); M T(n,V(m,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)T[j][i]=A[i][j]; return T; }
static M matMul(const M& A, const M& B){ int m=A.size(),n=B[0].size(),k=A[0].size(); M C(m,V(n,0)); for(int i=0;i<m;i++)for(int p=0;p<k;p++)for(int j=0;j<n;j++)C[i][j]+=A[i][p]*B[p][j]; return C; }
static M ident(int n){ M I(n,V(n,0)); for(int i=0;i<n;i++)I[i][i]=1; return I; }
static M inv(const M& A){
    int n=A.size(); M a=A,b=ident(n);
    for(int col=0;col<n;col++){
        int piv=col; for(int r=col+1;r<n;r++) if(fabs(a[r][col])>fabs(a[piv][col])) piv=r;
        swap(a[col],a[piv]); swap(b[col],b[piv]);
        double d=a[col][col];
        for(int j=0;j<n;j++){ a[col][j]/=d; b[col][j]/=d; }
        for(int r=0;r<n;r++) if(r!=col){ double f=a[r][col]; for(int j=0;j<n;j++){ a[r][j]-=f*a[col][j]; b[r][j]-=f*b[col][j]; } }
    }
    return b;
}
static double norm2(const V& v){ double s=0; for(double x:v) s+=x*x; return sqrt(s); }
static V matVec(const M& A, const V& x){ int m=A.size(),n=A[0].size(); V y(m,0); for(int i=0;i<m;i++){double s=0;for(int j=0;j<n;j++)s+=A[i][j]*x[j];y[i]=s;} return y; }

static double manip(const M& J){
    M JJt = matMul(J, transpose(J));
    double d = JJt[0][0]*JJt[1][1] - JJt[0][1]*JJt[1][0];
    return sqrt(fabs(d));
}
// 最小奇异值 σ_min = sqrt(较小特征值 of JJ^T)
static double sigmaMin(const M& J){
    M M_ = matMul(J, transpose(J));
    double tr = M_[0][0]+M_[1][1];
    double d  = M_[0][0]*M_[1][1]-M_[0][1]*M_[1][0];
    double lam = (tr - sqrt(fabs(tr*tr-4*d)))/2.0;
    return sqrt(fabs(lam));
}
// 阻尼最小二乘：q̇ = J^T (JJ^T + λ²I)^{-1} ẋ   （注意顺序：J^T 在左）
static V dls(const M& J, const V& xd, double lambda){
    M JJt = matMul(J, transpose(J));
    for(int i=0;i<2;i++) JJt[i][i] += lambda*lambda;
    M iJJt = inv(JJt);
    return matVec(matMul(transpose(J), iJJt), xd);
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    // 好构型
    V qGood = {M_PI/4, M_PI/2};
    double wGood = manip(Jfk(qGood));
    cout << "[L2.2] 好构型 w=" << wGood << "  σ_min=" << sigmaMin(Jfk(qGood)) << "\n";
    if(!(wGood > 0.2)){ pass=false; cout<<"  [失败] 好构型可操作度应较大\n"; }

    // 奇异构型：完全伸展 q=(0,0)
    V qSing = {0.0, 0.0};
    double wSing = manip(Jfk(qSing));
    double sSing = sigmaMin(Jfk(qSing));
    cout << "[L2.2] 奇异(伸展) w=" << wSing << "  σ_min=" << sSing << "\n";
    if(!(wSing < 1e-6 && sSing < 1e-6)){ pass=false; cout<<"  [失败] 伸展构型应为奇异(w≈0)\n"; }

    // 好构型下 DLS 几乎精确复现 ẋ
    V xd = {0.1, 0.1};
    V qdG = dls(Jfk(qGood), xd, 1e-3);
    V xrec = matVec(Jfk(qGood), qdG);
    double errGood = norm2({xrec[0]-xd[0], xrec[1]-xd[1]});
    cout << "[L2.2] 好构型 DLS 复现 ẋ 误差=" << errGood << "\n";
    if(!(errGood < 0.01)){ pass=false; cout<<"  [失败] 好构型 DLS 应精确\n"; }

    // 奇异处 DLS 仍给出有限关节速度（普通伪逆 λ=0 会发散→NaN/Inf）
    V qdDLS = dls(Jfk(qSing), xd, 0.1);
    bool dlsFinite = std::isfinite(qdDLS[0]) && std::isfinite(qdDLS[1]) && norm2(qdDLS) < 10.0;
    cout << "[L2.2] 奇异处 DLS q̇=(" << qdDLS[0] << ", " << qdDLS[1] << ")  ‖q̇‖=" << norm2(qdDLS) << " (有限)\n";
    if(!dlsFinite){ pass=false; cout<<"  [失败] DLS 在奇异处未给出有限解\n"; }

    // 对照：普通伪逆(λ=0) 在奇异处得到非有限解
    M JJt0 = matMul(Jfk(qSing), transpose(Jfk(qSing)));
    M ip = inv(JJt0);   // 奇异 → 含 Inf/NaN
    V qdPIN = matVec(matMul(transpose(Jfk(qSing)), ip), xd);
    bool pinFinite = std::isfinite(qdPIN[0]) && std::isfinite(qdPIN[1]);
    cout << "[L2.2] 普通伪逆(λ=0) 奇异处 finite=" << (pinFinite?"是(异常)":"否(发散)") << "\n";
    if(pinFinite){ pass=false; cout<<"  [失败] 普通伪逆在奇异处不应有限\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] w/σ_min 标识奇异；DLS 稳健、伪逆发散" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
