// jac_l1_manipulator.cpp
// ============================================================================
// L1.3 机械臂速度级运动学：J(q)·q̇ = ẋ
// ----------------------------------------------------------------------------
// 二连杆平面臂（l1=l2=1）。正运动学：
//   x = l1·cosθ1 + l2·cos(θ1+θ2)
//   y = l1·sinθ1 + l2·sin(θ1+θ2)
// 对时间求导得到机械臂雅可比 J(q)：末端线速度 ẋ = J(q)·q̇。
// 验证：解析 J·q̇  与  有限差分 (FK(q+hq̇)-FK(q))/h  一致。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l1_manipulator.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;

static const double L1 = 1.0, L2 = 1.0;

// 正运动学 FK(q) -> (x,y)
static V fk(const V& q){
    double t1 = q[0], t2 = q[0]+q[1];
    return { L1*cos(t1) + L2*cos(t2), L1*sin(t1) + L2*sin(t2) };
}
// 解析雅可比 J(q) (2x2)
static M Jfk(const V& q){
    double t1 = q[0], t2 = q[0]+q[1];
    double c1 = cos(t1), s1 = sin(t1), c12 = cos(t2), s12 = sin(t2);
    return {
        { -L1*s1 - L2*s12, -L2*s12 },
        {  L1*c1 + L2*c12,  L2*c12 }
    };
}
static V matVec(const M& A, const V& x){
    int m = A.size(), n = A[0].size();
    V y(m, 0.0);
    for (int i = 0; i < m; i++){ double s=0; for(int j=0;j<n;j++) s+=A[i][j]*x[j]; y[i]=s; }
    return y;
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);
    V q  = {0.6, 1.1};   // 关节角
    V qd = {0.4, -0.7};  // 关节角速度
    double h = 1e-6;

    V xdot_analytic = matVec(Jfk(q), qd);
    V qp = {q[0]+h*qd[0], q[1]+h*qd[1]};
    V fk0 = fk(q), fkp = fk(qp);
    V xdot_fd = { (fkp[0]-fk0[0])/h, (fkp[1]-fk0[1])/h };

    double err = max(fabs(xdot_analytic[0]-xdot_fd[0]), fabs(xdot_analytic[1]-xdot_fd[1]));
    cout << "[L1.3] ẋ(解析) = (" << xdot_analytic[0] << ", " << xdot_analytic[1] << ")\n";
    cout << "[L1.3] ẋ(差分) = (" << xdot_fd[0] << ", " << xdot_fd[1] << ")\n";
    cout << "[L1.3] 误差 = " << err << "   (应 < 1e-4)\n";
    if (!(err < 1e-4)){ pass = false; cout << "  [失败] 雅可比与差分不符\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] J(q)·q̇ = ẋ 成立" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
