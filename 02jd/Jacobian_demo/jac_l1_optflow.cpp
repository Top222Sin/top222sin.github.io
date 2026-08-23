// jac_l1_optflow.cpp
// ============================================================================
// L1.5 图像处理中的雅可比：Lucas-Kanade 光流（迭代式）
// ----------------------------------------------------------------------------
// 合成图像 I1(x,y) = sin(0.3x)·cos(0.2y)，第 2 帧相对第 1 帧平移 (DX,DY)=(2,1)：
//     I2(x,y) = I1(x+DX, y+DY)
// LK 把"像素灰度对位移的敏感度"写成雅可比 [Ix; Iy]，用前向叠加（forward-additive）
// 迭代最小化 Σ(I1(x+u,y+v) - I2(x,y))²：
//     每次迭代解 2×2 法方程  A·Δ = b,  A = Σ[Ix;Iy][Ix,Iy],  b = Σ[Ix;Iy]·res
// 多步迭代后 (u,v) → (DX,DY)。验证：恢复位移 ≈ (2,1)。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l1_optflow.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

static double I1(double x, double y){ return sin(0.3*x)*cos(0.2*y); }
static const double DX = 2.0, DY = 1.0;
static double I2(double x, double y){ return I1(x+DX, y+DY); }   // 第2帧 = 第1帧平移 (DX,DY)

static double gIx(double x, double y){ return (I1(x+1,y)-I1(x-1,y))/2.0; }
static double gIy(double x, double y){ return (I1(x,y+1)-I1(x,y-1))/2.0; }

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    double u = 0.0, v = 0.0;          // 位移估计 (初值 0)
    int iters = 0;
    const int MAXIT = 20;
    for (iters = 0; iters < MAXIT; iters++){
        double A00=0,A01=0,A11=0,b0=0,b1=0;
        for (int xi = 5; xi <= 25; xi++){
            for (int yi = 5; yi <= 25; yi++){
                double gx = gIx(xi+u, yi+v);     // 在当前估计处取梯度
                double gy = gIy(xi+u, yi+v);
                double res = I2(xi,yi) - I1(xi+u, yi+v);
                A00 += gx*gx; A01 += gx*gy; A11 += gy*gy;
                b0  += gx*res; b1  += gy*res;
            }
        }
        double det = A00*A11 - A01*A01;
        double du = ( A11*b0 - A01*b1)/det;
        double dv = (-A01*b0 + A00*b1)/det;
        u += du; v += dv;
        if (fabs(du) < 1e-9 && fabs(dv) < 1e-9) break;
    }

    cout << "[L1.5] 迭代 LK 估计位移 = (" << u << ", " << v << ")  迭代 " << iters << " 次\n";
    cout << "[L1.5] 真实平移         = (" << DX << ", " << DY << ")\n";
    if (!(fabs(u-DX) < 1e-3 && fabs(v-DY) < 1e-3)){
        pass = false; cout << "  [失败] 光流位移估计偏差过大\n";
    }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 光流雅可比迭代恢复出真实平移" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
