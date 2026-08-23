// jac_l2_ik.cpp
// ============================================================================
// L2.3 逆运动学：伪逆与阻尼最小二乘(DLS)
// ----------------------------------------------------------------------------
// 给定末端目标 x_des，迭代求关节角 q： q̇ = J(q)^T (J J^T + λ²I)^{-1} (x_des - x)
// 即阻尼最小二乘（damped least squares）。λ=0 退化为普通伪逆（奇异处发散），
// λ>0 在奇异附近给出有限且平滑的关节速度。用回溯线搜索保证误差单调下降。
// 验证：从 q0 收敛到可达目标，误差<1e-2；近奇异处 DLS 仍有限而伪逆发散。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_ik.cpp -o t && ./t
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

static V fk(const V& q){
    double t1=q[0], t2=q[0]+q[1];
    return { L1*cos(t1)+L2*cos(t2), L1*sin(t1)+L2*sin(t2) };
}
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
static V matVec(const M& A, const V& x){ int m=A.size(),n=A[0].size(); V y(m,0); for(int i=0;i<m;i++){double s=0;for(int j=0;j<n;j++)s+=A[i][j]*x[j];y[i]=s;} return y; }
static double norm2(const V& v){ double s=0; for(double x:v) s+=x*x; return sqrt(s); }

// 阻尼最小二乘： q̇ = J^T (J J^T + λ²I)^{-1} e
static V dlsStep(const M& J, const V& e, double lambda){
    M JJt = matMul(J, transpose(J));
    for(int i=0;i<2;i++) JJt[i][i] += lambda*lambda;
    M iJJt = inv(JJt);
    return matVec(matMul(transpose(J), iJJt), e);
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    // 用一个可达目标（由某 q* 正解得到），从另一初值收敛
    V qStar = {0.8, 1.2};
    V xDes  = fk(qStar);
    V q = {0.1, 0.1};
    double lambda = 0.05;

    for (int k=0; k<500; k++){
        V e = { xDes[0]-fk(q)[0], xDes[1]-fk(q)[1] };
        if (norm2(e) < 1e-4) break;
        V qd = dlsStep(Jfk(q), e, lambda);
        // 回溯线搜索：保证末端误差单调下降
        double mu = 1.0;
        V best = q; double bestErr = norm2(e);
        while (mu > 1e-4){
            V qn = { q[0]+mu*qd[0], q[1]+mu*qd[1] };
            double en = norm2({ xDes[0]-fk(qn)[0], xDes[1]-fk(qn)[1] });
            if (en < bestErr){ best = qn; bestErr = en; break; }
            mu *= 0.5;
        }
        q = best;
    }
    V xEnd = fk(q);
    double err = norm2({xEnd[0]-xDes[0], xEnd[1]-xDes[1]});
    cout << "[L2.3] 目标 x_des=(" << xDes[0] << "," << xDes[1] << ")\n";
    cout << "[L2.3] 收敛后 x=(" << xEnd[0] << "," << xEnd[1] << ")  误差=" << err << "\n";
    if(!(err < 1e-2)){ pass=false; cout<<"  [失败] DLS 逆解未收敛到目标\n"; }

    // 奇异附近：DLS(λ>0) 有限；普通伪逆(λ=0) 发散
    V qNear = {0.0, 0.0};
    V eN = {0.1, 0.0};
    V qdDLS = dlsStep(Jfk(qNear), eN, 0.1);
    bool dlsFinite = std::isfinite(qdDLS[0]) && std::isfinite(qdDLS[1]) && norm2(qdDLS) < 10.0;
    M JJt0 = matMul(Jfk(qNear), transpose(Jfk(qNear)));
    V qdPIN = matVec(matMul(transpose(Jfk(qNear)), inv(JJt0)), eN);
    bool pinFinite = std::isfinite(qdPIN[0]) && std::isfinite(qdPIN[1]);
    cout << "[L2.3] 近奇异 λ=0.1 DLS ‖q̇‖=" << norm2(qdDLS)
         << "   普通伪逆 finite=" << (pinFinite?"是":"否(发散)") << "\n";
    if(!dlsFinite){ pass=false; cout<<"  [失败] DLS 在近奇异处未给出有限解\n"; }
    if(pinFinite){ pass=false; cout<<"  [失败] 普通伪逆在奇异处不应有限\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] DLS 逆解收敛且对奇异稳健" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
