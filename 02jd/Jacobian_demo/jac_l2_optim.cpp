// jac_l2_optim.cpp
// ============================================================================
// L2.5 数值优化：牛顿 / 高斯-牛顿(GN) / 莱文伯格-马夸尔特(LM)
// ----------------------------------------------------------------------------
// 非线性最小二乘 min_θ Σ r_i(θ)²，核心都用"残差雅可比 J = ∂r/∂θ"：
//     高斯-牛顿:  Δ = (J^T J)^{-1} J^T r          （无阻尼，近解处二次收敛）
//     莱文伯格-  :  Δ = (J^T J + λI)^{-1} J^T r   （阻尼，远离解也稳健）
// 模型 y = a·exp(b·x)，真值 (a,b)=(2,-0.5)。验证 LM 从易/难初值都收敛到真值，
// GN 在易初值也收敛（难初值易发散，仅展示不强制）。运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_optim.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;

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

// 数据
static vector<double> XS; // x 采样
static V YS;              // y 观测 (= 2*exp(-0.5 x) 无噪)
static void buildData(){
    XS.clear(); YS.clear();
    for(double x=0; x<=3.0001; x+=0.3){ XS.push_back(x); YS.push_back(2.0*exp(-0.5*x)); }
}
// 残差 r_i = a*exp(b x_i) - y_i ；雅可比 J_i = [exp(b x_i), a x_i exp(b x_i)]
static V residuals(const V& th){
    V r(XS.size());
    for(size_t i=0;i<XS.size();i++) r[i] = th[0]*exp(th[1]*XS[i]) - YS[i];
    return r;
}
static M jacobian(const V& th){
    M J(XS.size(), V(2,0));
    for(size_t i=0;i<XS.size();i++){
        double e = exp(th[1]*XS[i]);
        J[i][0] = e;
        J[i][1] = th[0]*XS[i]*e;
    }
    return J;
}
static double cost(const V& th){
    V r=residuals(th); double s=0; for(double x:r) s+=x*x; return s;
}
// GN（lambda=0 固定）或 LM（自适应 lambda）。返回是否收敛到有限解。
static bool solve(V& th, double lambda0, bool adapt, int maxit){
    double lambda = lambda0;
    for(int it=0; it<maxit; it++){
        V r = residuals(th);
        M J = jacobian(th);
        M Jt = transpose(J);
        M A = matMul(Jt, J);           // 2x2
        V g = matVec(Jt, r);           // 2
        for(int i=0;i<2;i++) A[i][i] += lambda;
        M iA = inv(A);
        V d = matVec(iA, g);
        V thNew = { th[0]-d[0], th[1]-d[1] };
        if(!std::isfinite(thNew[0]) || !std::isfinite(thNew[1])) return false;
        if(adapt){
            if(cost(thNew) < cost(th)){ th = thNew; lambda = max(lambda*0.7, 1e-8); }
            else { lambda = min(lambda*2.0, 1e6); }
        } else {
            if(norm2(d) < 1e-9) { th = thNew; break; }
            th = thNew;
        }
        if(norm2(d) < 1e-9) break;
    }
    return std::isfinite(th[0]) && std::isfinite(th[1]);
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);
    buildData();

    // 易初值
    V thE = {1.5, -0.2};
    bool okGN = solve(thE, 0.0, false, 100);
    cout << "[L2.5] GN 易初值 (1.5,-0.2) → (a,b)=(" << thE[0] << "," << thE[1] << ") 收敛=" << okGN << "\n";
    if(!(okGN && fabs(thE[0]-2.0)<0.05 && fabs(thE[1]+0.5)<0.05)){ pass=false; cout<<"  [失败] GN 易初值未收敛到真值\n"; }

    // LM 易初值
    V thL = {1.5, -0.2};
    bool okLM = solve(thL, 1e-3, true, 100);
    cout << "[L2.5] LM 易初值 (1.5,-0.2) → (a,b)=(" << thL[0] << "," << thL[1] << ") 收敛=" << okLM << "\n";
    if(!(okLM && fabs(thL[0]-2.0)<0.05 && fabs(thL[1]+0.5)<0.05)){ pass=false; cout<<"  [失败] LM 易初值未收敛\n"; }

    // LM 难初值（错误符号），应当稳健收敛；GN 通常发散（仅展示）
    V thH = {0.3, 0.6};
    bool okLMh = solve(thH, 1.0, true, 200);
    cout << "[L2.5] LM 难初值 (0.3,0.6) → (a,b)=(" << thH[0] << "," << thH[1] << ") 收敛=" << okLMh << "\n";
    if(!(okLMh && fabs(thH[0]-2.0)<0.05 && fabs(thH[1]+0.5)<0.05)){ pass=false; cout<<"  [失败] LM 难初值未收敛\n"; }

    V thGNh = {0.3, 0.6};
    bool okGNd = solve(thGNh, 0.0, false, 100);
    cout << "[L2.5] GN 难初值 (0.3,0.6) → 收敛=" << okGNd
         << " (a,b)=(" << thGNh[0] << "," << thGNh[1] << ")  [展示：易发散]\n";

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] LM 稳健收敛到真值；GN 近解处亦可行" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
