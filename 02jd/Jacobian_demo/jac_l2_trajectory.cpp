// jac_l2_trajectory.cpp
// ============================================================================
// L2.4 轨迹规划：零空间冗余分解（3 连杆平面臂）
// ----------------------------------------------------------------------------
// 3R 平面臂做 2D 位置任务（欠驱动冗余：3 关节 → 2 任务）。速度级：
//     q̇ = J^+ ẋ_des + (I - J^+ J)·ẋ_null
//   第一项（阻尼伪逆）跟踪末端到目标；第二项在零空间内优化次要目标
//   （让第3关节逼近期望角），零空间方向不影响主任务。验证：末端收敛到目标
//   （任务误差小），且 q3 朝期望角方向移动（零空间优化生效）。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_trajectory.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;
static const double L1=1.0, L2=1.0, L3=1.0;

static V fk(const V& q){
    double a1=q[0], a2=q[0]+q[1], a3=q[0]+q[1]+q[2];
    return { L1*cos(a1)+L2*cos(a2)+L3*cos(a3),
             L1*sin(a1)+L2*sin(a2)+L3*sin(a3) };
}
// 位置雅可比 J (2x3)
static M Jfk(const V& q){
    double a1=q[0], a2=q[0]+q[1], a3=q[0]+q[1]+q[2];
    M J(2, V(3,0));
    J[0][0] = -L1*sin(a1) - L2*sin(a2) - L3*sin(a3);
    J[1][0] =  L1*cos(a1) + L2*cos(a2) + L3*cos(a3);
    J[0][1] = -L2*sin(a2) - L3*sin(a3);
    J[1][1] =  L2*cos(a2) + L3*cos(a3);
    J[0][2] = -L3*sin(a3);
    J[1][2] =  L3*cos(a3);
    return J;
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

// 阻尼伪逆 J^+ = J^T (J J^T + λ²I)^{-1}  (2x3 → 返回 3x2)
static M pinvDLS(const M& J, double lambda){
    M JJt = matMul(J, transpose(J));
    for(int i=0;i<2;i++) JJt[i][i] += lambda*lambda;
    return matMul(transpose(J), inv(JJt));
}
// 阻尼最小二乘速度：q̇ = J^+ ẋ
static V dlsVel(const M& J, const V& xd, double lambda){
    return matVec(pinvDLS(J, lambda), xd);
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    V q0 = {0.3, 0.4, 0.5};
    V A = fk(q0);
    V B = { A[0]+0.1, A[1]+0.1 };        // 小位移目标（保持良好条件数）
    double q3des = 0.35;                 // 次要目标：第3关节期望角
    int N = 500;
    double lam = 0.01, gain = 1.0, step = 0.1, kSec = 0.01;

    V q = q0;
    double maxErr = 0.0;
    for(int s=0; s<N; s++){
        M J = Jfk(q);
        V e = { B[0]-fk(q)[0], B[1]-fk(q)[1] };
        V primary = dlsVel(J, {gain*e[0], gain*e[1]}, lam);   // 主任务
        // 零空间投影 N = I - J^+ J
        M Jp = pinvDLS(J, lam);
        M JpJ = matMul(Jp, J);
        M Nproj = ident(3);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) Nproj[i][j] -= JpJ[i][j];
        // 次要目标梯度：H=0.5(q3-q3des)^2 → dH/dq = (0,0,q3-q3des)
        V nullVel = { 0, 0, -kSec*(q[2]-q3des) };
        V qd(3,0);
        for(int i=0;i<3;i++){
            qd[i] = primary[i];
            for(int j=0;j<3;j++) qd[i] += Nproj[i][j]*nullVel[j];
        }
        q[0]+=qd[0]*step; q[1]+=qd[1]*step; q[2]+=qd[2]*step;
    }
    V xEnd = fk(q);
    double taskErr = norm2({xEnd[0]-B[0], xEnd[1]-B[1]});
    cout << "[L2.4] 目标 B=(" << B[0] << "," << B[1] << ")  末端终点=(" << xEnd[0] << "," << xEnd[1] << ")\n";
    cout << "[L2.4] 任务误差=" << taskErr << "   q3: " << q0[2] << " → " << q[2] << " (期望 " << q3des << ")\n";
    if(!(taskErr < 0.02)){ pass=false; cout<<"  [失败] 主任务(末端轨迹)误差过大\n"; }
    if(!(fabs(q[2]-q3des) < fabs(q0[2]-q3des))){ pass=false; cout<<"  [失败] 零空间未把 q3 推向期望角\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 末端点达目标，且零空间优化了次要关节" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
