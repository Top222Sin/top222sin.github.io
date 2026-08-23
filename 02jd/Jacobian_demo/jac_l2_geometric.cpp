// jac_l2_geometric.cpp
// ============================================================================
// L2.1 几何雅可比（旋量/twist） vs 解析雅可比
// ----------------------------------------------------------------------------
// 平面二连杆：末端位姿 ξ = (x, y, φ)，φ = θ1+θ2。
//   · 解析雅可比 Ja = ∂(x,y,φ)/∂q  （用"最小坐标"描述姿态，存在表示奇异性）
//   · 几何雅可比 Jg = 由各关节旋量轴按 J_i = [z×(p_e-p_{i-1}); z] 拼成（描述末端
//     速度旋量 [v;ω]），不受欧拉角表示奇异性影响。
// 本 demo 验证两点：
//   (1) 该平面情形下 Jg·q̇ 与 Ja·q̇ 都精确等于位姿的有限差分 → 二者一致；
//   (2) 3D 单轴回转 + ZYX 欧拉角：解析雅可比含 1/cosβ 项，在 β→90° 时病态
//       爆炸，而几何雅可比恒为 [0;0;1]（有界）——即"表示奇异"只害解析 J。
// 运行即出 PASS/FAIL。
//
// 编译: g++ -O3 -std=c++17 jac_l2_geometric.cpp -o t && ./t
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

static M transpose(const M& A){ int m=A.size(),n=A[0].size(); M T(n,V(m,0)); for(int i=0;i<m;i++)for(int j=0;j<n;j++)T[j][i]=A[i][j]; return T; }
static V matVec(const M& A, const V& x){ int m=A.size(),n=A[0].size(); V y(m,0); for(int i=0;i<m;i++){double s=0;for(int j=0;j<n;j++)s+=A[i][j]*x[j];y[i]=s;} return y; }
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

// 平面二连杆：末端位姿 (x,y,φ)
static V fkPose(const V& q){
    double t1=q[0], t2=q[0]+q[1];
    return { L1*cos(t1)+L2*cos(t2), L1*sin(t1)+L2*sin(t2), t2 };
}
// 解析雅可比 Ja = ∂(x,y,φ)/∂q
static M Ja(const V& q){
    double t1=q[0], t2=q[0]+q[1];
    double s1=sin(t1), c1=cos(t1), s12=sin(t2), c12=cos(t2);
    return {
        { -L1*s1 - L2*s12, -L2*s12 },
        {  L1*c1 + L2*c12,  L2*c12 },
        {  1.0,              1.0     }
    };
}
// 几何雅可比 Jg（旋量）：J_i = [ z × (p_e - p_{i-1}); z ], z=[0,0,1]
static M Jg(const V& q){
    double t1=q[0], t2=q[0]+q[1];
    double x = L1*cos(t1)+L2*cos(t2);
    double y = L1*sin(t1)+L2*sin(t2);
    double s12 = sin(t2), c12 = cos(t2);
    // col1: r = p_e - 原点 = (x,y)   → [-y, x, 1]
    // col2: r = p_e - p1 = (x-p1x, y-p1y) = (L2 c12, L2 s12) → [-L2 s12, L2 c12, 1]
    return {
        { -y,        -L2*s12 },
        {  x,         L2*c12 },
        {  1.0,       1.0     }
    };
}

// 3D：ZYX 欧拉角 -> 角速度的矩阵 B(α,β)（det=cosβ，β→90° 病态）
static M Bmat(double a, double b){
    double ca=cos(a), sa=sin(a), cb=cos(b), sb=sin(b);
    return { {1, 0, -sb}, {0, ca, sa*cb}, {0, -sa, ca*cb} };
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(6);

    // (1) 平面情形：Jg 与 Ja 都匹配位姿有限差分，且二者一致
    V q={0.6,1.1}, qd={0.4,-0.7}; double h=1e-6;
    V xg = matVec(Jg(q), qd);
    V xa = matVec(Ja(q), qd);
    V qp={q[0]+h*qd[0], q[1]+h*qd[1]};
    V f0=fkPose(q), f1=fkPose(qp);
    V xfd={(f1[0]-f0[0])/h,(f1[1]-f0[1])/h,(f1[2]-f0[2])/h};
    double eG = norm2({xg[0]-xfd[0],xg[1]-xfd[1],xg[2]-xfd[2]});
    double eA = norm2({xa[0]-xfd[0],xa[1]-xfd[1],xa[2]-xfd[2]});
    double eGA=0; for(int i=0;i<3;i++)for(int j=0;j<2;j++) eGA=max(eGA,fabs(Jg(q)[i][j]-Ja(q)[i][j]));
    cout << "[L2.1] Jg·q̇ vs 差分 误差=" << eG << "   Ja·q̇ vs 差分 误差=" << eA
         << "   Jg vs Ja 差异=" << eGA << "\n";
    if(!(eG<1e-4 && eA<1e-4 && eGA<1e-12)){ pass=false; cout<<"  [失败] 平面情形 Jg/Ja 不符\n"; }

    // (2) 3D 欧拉：解析雅可比含 1/cosβ 项，β=85° 时病态爆炸，几何 J 恒有界
    double b1=10*M_PI/180, b2=85*M_PI/180;
    M B1=Bmat(0,b1), B2=Bmat(0,b2);
    // 条件数（用 det + 范数近似）：cond ≈ |det(B)|^-1 * ‖B‖ (对称近似)
    double detB1=fabs(B1[0][0]*(B1[1][1]*B1[2][2]-B1[1][2]*B1[2][1])
                    -B1[0][1]*(B1[1][0]*B1[2][2]-B1[1][2]*B1[2][0])
                    +B1[0][2]*(B1[1][0]*B1[2][1]-B1[1][1]*B1[2][0]));
    double detB2=fabs(B2[0][0]*(B2[1][1]*B2[2][2]-B2[1][2]*B2[2][1])
                    -B2[0][1]*(B2[1][0]*B2[2][2]-B2[1][2]*B2[2][0])
                    +B2[0][2]*(B2[1][0]*B2[2][1]-B2[1][1]*B2[2][0]));
    // 解析角速度雅可比（角速度->欧拉率）：J_analytic = B^{-1} * [0;0;1]
    V axis={0,0,1};
    V jaA1 = matVec(inv(B1), axis);
    V jaA2 = matVec(inv(B2), axis);
    double nA1=norm2(jaA1), nA2=norm2(jaA2);
    cout << "[L2.1] β=10° : |det B|=" << detB1 << "  解析J角‖=" << nA1 << "\n";
    cout << "[L2.1] β=85° : |det B|=" << detB2 << "  解析J角‖=" << nA2
         << "   几何J角‖=1 (恒有界)\n";
    if(!(detB2 < 0.1*detB1)){ pass=false; cout<<"  [失败] β=85° 时 B 未明显病态\n"; }
    if(!(nA2 > 5.0*nA1)){ pass=false; cout<<"  [失败] 解析J在β→90°未爆炸\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 几何J=旋量不受表示奇异；解析J在欧拉奇点爆炸" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
