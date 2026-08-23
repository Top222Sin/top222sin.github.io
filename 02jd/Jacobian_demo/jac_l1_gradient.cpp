// jac_l1_gradient.cpp
// ============================================================================
// L1.4 数值梯度与最速下降：雅可比是"一阶信息"的最小化入口
// ----------------------------------------------------------------------------
// 标量函数 f(x,y) = (x-2)^2 + (y-3)^2 + 1，唯一极小点在 (2,3)，f_min=1。
// 标量函数的雅可比 = 梯度 ∇f（1×2 行向量）。
// 验证：
//   (1) 数值梯度（中心差分）≈ 解析梯度
//   (2) 最速下降 x ← x - α·∇f 能收敛到 (2,3)，f→1
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l1_gradient.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;

static double f(const V& p){ double x=p[0], y=p[1]; return (x-2)*(x-2) + (y-3)*(y-3) + 1; }
// 解析梯度 (标量函数的雅可比, 1×2)
static V grad(const V& p){ return { 2*(p[0]-2), 2*(p[1]-3) }; }

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    // (1) 数值梯度 vs 解析梯度
    V p = {1.3, 4.7};
    double h = 1e-6;
    V numg = {
        (f({p[0]+h,p[1]})-f({p[0]-h,p[1]}))/(2*h),
        (f({p[0],p[1]+h})-f({p[0],p[1]-h}))/(2*h)
    };
    V ag = grad(p);
    double gerr = max(fabs(numg[0]-ag[0]), fabs(numg[1]-ag[1]));
    cout << "[L1.4] 数值梯度 = (" << numg[0] << ", " << numg[1] << ")\n";
    cout << "[L1.4] 解析梯度 = (" << ag[0] << ", " << ag[1] << ")  误差=" << gerr << "\n";
    if (!(gerr < 1e-4)){ pass = false; cout << "  [失败] 数值梯度不对\n"; }

    // (2) 最速下降
    V x = {0.0, 0.0};
    double alpha = 0.1;
    int iters = 200;
    for (int k = 0; k < iters; k++){
        V g = grad(x);
        x[0] -= alpha*g[0];
        x[1] -= alpha*g[1];
    }
    double fmin = f(x);
    cout << "[L1.4] 最速下降后 x = (" << x[0] << ", " << x[1] << ")  f = " << fmin << "\n";
    if (!(fabs(x[0]-2) < 0.05 && fabs(x[1]-3) < 0.05 && fabs(fmin-1) < 0.05)){
        pass = false; cout << "  [失败] 未收敛到极小点\n";
    }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 梯度正确 + 最速下降收敛" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
