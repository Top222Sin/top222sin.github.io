// jac_l2_vision.cpp
// ============================================================================
// L2.6 计算机视觉：重投影雅可比与 Bundle Adjustment 初探
// ----------------------------------------------------------------------------
// 针孔相机 K=diag(f,f,1)，位姿 (R,t)。3D 点 X 投影到像素：
//     p_cam = R·X + t ;  u = f·x/z + cx,  v = f·y/z + cy
// 像素对 3D 点坐标的雅可比 ∂(u,v)/∂X 由链式法则得到（含 1/z 透视项）。
// 验证：(1) 解析 ∂(u,v)/∂X 与有限差分一致；(2) 两视 BA：用解析雅可比做高斯-牛顿
// 优化 X，使重投影误差→0。运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_vision.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;
static const double F=500.0, CX=320.0, CY=240.0;

// 恒等旋转 + 平移 t 的投影（两视分别用不同 t）
static V project(const V& X, const V& t){
    double x = X[0]+t[0], y = X[1]+t[1], z = X[2]+t[2];
    return { F*x/z + CX, F*y/z + CY };
}
// 解析重投影雅可比 ∂(u,v)/∂X (2x3)  （R=I，故 ∂p_cam/∂X=I）
static M jacProj(const V& X, const V& t){
    double x = X[0]+t[0], y = X[1]+t[1], z = X[2]+t[2];
    return {
        { F/z,        0.0,       -F*x/(z*z) },
        { 0.0,        F/z,       -F*y/(z*z) }
    };
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
static double norm2(std::initializer_list<double> v){ double s=0; for(double d:v) s+=d*d; return std::sqrt(s); }

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    V X = {0.1, 0.2, 2.0};
    V t0 = {0,0,0}, t1 = {0.3, 0, 0};
    double h = 1e-6;

    // (1) 解析雅可比 vs 有限差分
    M Ja0 = jacProj(X, t0);
    M Jfd(2, V(3,0));
    for(int c=0;c<3;c++){
        V Xp=X; Xp[c]+=h;
        V pu=project(Xp,t0), p0=project(X,t0);
        Jfd[0][c]=(pu[0]-p0[0])/h; Jfd[1][c]=(pu[1]-p0[1])/h;
    }
    double jerr=0; for(int i=0;i<2;i++)for(int j=0;j<3;j++) jerr=max(jerr,fabs(Ja0[i][j]-Jfd[i][j]));
    cout << "[L2.6] 重投影雅可比 解析vs差分 误差=" << jerr << "   (应 < 1e-3)\n";
    if(!(jerr<1e-3)){ pass=false; cout<<"  [失败] 重投影雅可比不对\n"; }

    // (2) 两视 BA：优化 X 重建，使两视重投影误差最小
    V obs0 = project(X, t0);   // 真值观测
    V obs1 = project(X, t1);
    V Xopt = {X[0]+0.05, X[1]-0.04, X[2]+0.10};  // 带扰动的初值
    for(int it=0; it<40; it++){
        // 残差（堆叠 4x1）
        V r0={obs0[0]-project(Xopt,t0)[0], obs0[1]-project(Xopt,t0)[1]};
        V r1={obs1[0]-project(Xopt,t1)[0], obs1[1]-project(Xopt,t1)[1]};
        // 雅可比（堆叠 4x3）
        M J0=jacProj(Xopt,t0), J1=jacProj(Xopt,t1);
        M J(4, V(3,0));
        for(int i=0;i<2;i++){ J[i]=J0[i]; J[2+i]=J1[i]; }
        V r={r0[0],r0[1],r1[0],r1[1]};
        M Jt=transpose(J);
        M A=matMul(Jt,J);      // 3x3
        V g=matVec(Jt,r);       // 3
        V d=matVec(inv(A),g);
        Xopt[0]-=d[0]; Xopt[1]-=d[1]; Xopt[2]-=d[2];
    }
    double e0 = norm2({obs0[0]-project(Xopt,t0)[0], obs0[1]-project(Xopt,t0)[1]});
    double e1 = norm2({obs1[0]-project(Xopt,t1)[0], obs1[1]-project(Xopt,t1)[1]});
    double errBA = e0+e1;
    cout << "[L2.6] BA 后 X=(" << Xopt[0] << "," << Xopt[1] << "," << Xopt[2] << ")\n";
    cout << "[L2.6] 重投影误差 e0=" << e0 << " e1=" << e1 << "   (应→0)\n";
    if(!(errBA < 1e-3)){ pass=false; cout<<"  [失败] BA 未收敛\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 重投影雅可比正确 + 两视BA收敛" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
