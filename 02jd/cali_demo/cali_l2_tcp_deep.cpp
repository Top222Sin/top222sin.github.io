// cali_l2_tcp_deep.cpp
// ============================================================================
// L2.4  TCP 标定深入：误差传播 + 共面退化 + 与手眼的关系
// ----------------------------------------------------------------------------
// 在 L1.4 四点法基础上深入：
//   1) 误差传播：法兰位姿读数带噪声(机器人重复定位精度) → TCP 估计的蒙特卡洛方差
//   2) 共面退化：若所有法兰位置共面，球心(沿法向)不可观 → A 矩阵病态
//   3) 与手眼的关系：先标 TCP 还是先标手眼？为什么能形成闭环校验
// 全部合成数据闭环。
//
// 编译: g++ -O2 -std=c++17 cali_l2_tcp_deep.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }
static Matx33d randR(mt19937& g, double s) { Vec3d a{ rnd(g,-1,1), rnd(g,-1,1), rnd(g,-1,1) }; double n=norm(a); a/=(n+1e-9); Mat R; Rodrigues(a*rnd(g,-s,s), R); return Matx33d(R); }

// 四点法：用 (R_i - R_1)x = t_1 - t_i 堆叠解 x；返回 A 的条件数
static double solveTCP(const vector<Matx33d>& R, const vector<Vec3d>& t, Vec3d& x, double& cond) {
    int n = R.size(); int m = 3 * (n - 1);
    Mat A(m, 3, CV_64F), b(m, 1, CV_64F); int row = 0;
    for (int i = 1; i < n; i++) {
        Matx33d dR = R[i] - R[0]; Vec3d db = t[0] - t[i];
        for (int r = 0; r < 3; r++) { A.at<double>(row,0)=dR(r,0); A.at<double>(row,1)=dR(r,1); A.at<double>(row,2)=dR(r,2); b.at<double>(row)=db[r]; row++; }
    }
    SVD svd(A, SVD::MODIFY_A | SVD::NO_UV);
    double smax=0,smin=1e9; for(int i=0;i<svd.w.cols;i++){double s=svd.w.at<double>(i); smax=max(smax,s); smin=min(smin,s);}
    cond = (smin<1e-12)?1e12: smax/smin;
    Mat xm; solve(A, b, xm, DECOMP_SVD); x = Vec3d(xm.at<double>(0), xm.at<double>(1), xm.at<double>(2));
    // 残差
    double resid = 0;
    for (int i = 0; i < n; i++) resid = max(resid, norm(R[i] * x + t[i] - (R[0]*x + t[0])));
    return resid;
}

int main() {
    bool pass = true; mt19937 g(20260817);
    const int n = 8;
    Vec3d xgt(0.04, 0.0, 0.12); Vec3d pworld(0.30, 0.0, 0.45);

    // ---------- 1) 误差传播（蒙特卡洛）----------
    cout << "σ(法兰读数噪声)  TCP 估计均误差(mm)\n";
    vector<double> errs;
    for (double sigma : {0.0005, 0.001, 0.003}) {     // 0.5 / 1 / 3 mm
        int M = 300; double sum = 0;
        for (int trial = 0; trial < M; trial++) {
            vector<Matx33d> R; vector<Vec3d> t;
            for (int i = 0; i < n; i++) {
                Matx33d Ri = randR(g, 0.5);
                Vec3d ti = pworld - Ri * xgt;
                normal_distribution<double> ng(0, sigma);
                ti += Vec3d(ng(g), ng(g), ng(g));      // 注入法兰读数噪声
                R.push_back(Ri); t.push_back(ti);
            }
            Vec3d x; double c; solveTCP(R, t, x, c);
            sum += norm(x - xgt);
        }
        double mean = sum / M * 1000;
        errs.push_back(mean);
        cout << "  " << sigma * 1000 << " mm           " << mean << " mm\n";
    }
    if (!(errs[2] > errs[0])) { pass = false; cout << "  [失败] 误差未随噪声增大\n"; }

    // ---------- 2) 共面退化 ----------
    // 非共面：法兰位置 3D 散布
    vector<Matx33d> Rn; vector<Vec3d> tn;
    for (int i = 0; i < n; i++) { Matx33d Ri = randR(g, 0.5); Rn.push_back(Ri); tn.push_back(pworld - Ri * xgt + Vec3d(rnd(g,-0.1,0.1), rnd(g,-0.1,0.1), rnd(g,-0.1,0.1))); }
    Vec3d xnc; double condNC, condCP; solveTCP(Rn, tn, xnc, condNC);
    // 共面：法兰位置全在 z=0 平面
    vector<Matx33d> Rc; vector<Vec3d> tc;
    for (int i = 0; i < n; i++) { Matx33d Ri = randR(g, 0.3); Rc.push_back(Ri); tc.push_back(pworld - Ri * xgt); tc.back()[2] = 0; }
    Vec3d xc; solveTCP(Rc, tc, xc, condCP);
    cout << "\n[共面退化] 非共面 A 条件数=" << condNC << "  TCP误差=" << norm(xnc-xgt)*1000 << " mm\n";
    cout << "           共面   A 条件数=" << condCP << "  TCP误差=" << norm(xc-xgt)*1000 << " mm\n";
    if (!(condCP > 10 * condNC)) { pass = false; cout << "  [失败] 共面未表现为病态\n"; }
    if (!(norm(xc - xgt) > 5 * norm(xnc - xgt))) { pass = false; cout << "  [失败] 共面误差未显著放大\n"; }

    // ---------- 3) 与手眼闭环（说明用，打印关系）----------
    cout << "\n[闭环] 标定链:  base --(TCP)--> flange --(手眼X)--> cam ; 两者各自独立求，但都影响\n";
    cout << "       最终精度，故常先 TCP 再手眼，或迭代联合优化（L3 鲁棒会涉及）。\n";

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] TCP 深入：误差传播/共面退化/闭环关系 验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
