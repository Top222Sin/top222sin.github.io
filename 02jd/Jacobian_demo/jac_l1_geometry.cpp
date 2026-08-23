// jac_l1_geometry.cpp
// ============================================================================
// L1.2 雅可比的几何直觉：行列式 = 局部"面积/体积缩放因子"，秩 = 维数
// ----------------------------------------------------------------------------
// 三个合成映射演示：
//   f1 线性  (2x+0.3y, 0.5x+1.5y)        det=2.85  —— 单位正方形面积精确放大 2.85 倍
//   f2 非线性 (x^2, y)                   det=2x     —— 局部面积 ≈ |det J|·h^2
//   f3 秩亏   (x, x^2)                   det≡0      —— 平面被压成一条抛物线（维数坍缩）
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l1_geometry.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

using V = vector<double>;
using M = vector<vector<double>>;

static V f1(const V& p){ return {2*p[0]+0.3*p[1], 0.5*p[0]+1.5*p[1]}; }
static M J1(const V&){ return {{2, 0.3},{0.5, 1.5}}; }

static V f2(const V& p){ return {p[0]*p[0], p[1]}; }
static M J2(const V& p){ return {{2*p[0], 0},{0, 1}}; }

static V f3(const V& p){ return {p[0], p[0]*p[0]}; }
static M J3(const V& p){ return {{1, 0},{2*p[0], 0}}; }

static double det2(const M& A){ return A[0][0]*A[1][1] - A[0][1]*A[1][0]; }
// 二维叉积（带符号面积）
static double cross(const V& a, const V& b){ return a[0]*b[1] - a[1]*b[0]; }

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    // (1) 线性映射：单位正方形面积精确 = |det J|
    V a = f1({1,0}), b = f1({0,1}), o = f1({0,0});
    V A = {a[0]-o[0], a[1]-o[1]};
    V B = {b[0]-o[0], b[1]-o[1]};
    double area1 = fabs(cross(A,B));
    double detJ1 = fabs(det2(J1({0,0})));
    cout << "[L1.2] 线性 f1 面积缩放 = " << area1 << "  |det J| = " << detJ1 << "\n";
    if (!(fabs(area1 - detJ1) < 1e-9)){ pass = false; cout << "  [失败] 线性面积≠|det|\n"; }

    // (2) 非线性映射：局部面积 ≈ |det J|·h^2 （在 p=(3,4) 处）
    V p = {3,4}; double h = 1e-3;
    V e1 = f2({p[0]+h, p[1]}); V e2 = f2({p[0], p[1]+h}); V e0 = f2(p);
    V A2 = {e1[0]-e0[0], e1[1]-e0[1]};
    V B2 = {e2[0]-e0[0], e2[1]-e0[1]};
    double area2 = fabs(cross(A2,B2));
    double area2_expected = fabs(det2(J2(p))) * h * h;
    cout << "[L1.2] 非线性 f2 局部面积 = " << area2
         << "  近似 |det J|·h^2 = " << area2_expected << "\n";
    if (!(fabs(area2 - area2_expected) < 1e-7)){ pass = false; cout << "  [失败] 局部面积≠|det|·h^2\n"; }

    // (3) 秩亏映射：det≡0 —— 维数坍缩为曲线
    bool rankDef = true;
    for (double x : {0.0, 1.0, 2.5, -3.0}){
        if (fabs(det2(J3({x, 0}))) > 1e-9) rankDef = false;
    }
    cout << "[L1.2] 秩亏 f3 各点 det J = " << det2(J3({1,0}))
         << "  (应≡0 → 平面被压成抛物线)\n";
    if (!rankDef){ pass = false; cout << "  [失败] f3 不应满秩\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] det=局部缩放因子 / 秩=维数 直觉成立" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
