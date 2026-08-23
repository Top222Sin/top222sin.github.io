// cali_l1_tcp.cpp
// ============================================================================
// L1.4  TCP 标定（Tool Center Point，四点法）
// ----------------------------------------------------------------------------
// TCP = 工具坐标系原点（抓取/作业参考点）在法兰(frame 0)下的坐标 x。
// 四点法原理：让工具尖以 4+ 个不同姿态去碰"空间中同一个固定点"P。
//   每姿态 i 下：P = R_i·x + t_i  （R_i,t_i = 法兰在基座下的位姿）
//   => (R_i - R_1)·x = t_1 - t_i ，堆叠最小二乘解出 x。
// 这里用合成法兰位姿闭环验证：先造 x_gt，反算 t_i 使尖都落在同一点，再反估 x。
//
// 编译: g++ -O2 -std=c++17 cali_l1_tcp.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }
static Matx33d randRot(mt19937& g) {
    Vec3d axis{ rnd(g, -1, 1), rnd(g, -1, 1), rnd(g, -1, 1) };
    double n = norm(axis); axis /= (n + 1e-9);
    double ang = rnd(g, -0.6, 0.6);
    Mat R; Rodrigues(axis * ang, R);
    return Matx33d(R);
}

int main() {
    bool pass = true;
    const int n = 6;                          // 6 个姿态（≥4 且不共轴）
    Vec3d xgt(0.05, 0.0, 0.15);               // 真值 TCP（法兰系，沿 z 15cm）
    Vec3d pworld(0.30, 0.0, 0.50);            // 空间中固定的那个点 P
    mt19937 g(20260817);

    vector<Matx33d> R(n); vector<Vec3d> t(n);
    for (int i = 0; i < n; i++) { R[i] = randRot(g); t[i] = pworld - R[i] * xgt; }

    // ---- 用 (R_i - R_1)x = t_1 - t_i 堆叠解 x ----
    int m = 3 * (n - 1);
    Mat A(m, 3, CV_64F), b(m, 1, CV_64F);
    int row = 0;
    for (int i = 1; i < n; i++) {
        Matx33d dR = R[i] - R[0];
        Vec3d db = t[0] - t[i];
        for (int r = 0; r < 3; r++) {
            A.at<double>(row, 0) = dR(r, 0); A.at<double>(row, 1) = dR(r, 1); A.at<double>(row, 2) = dR(r, 2);
            b.at<double>(row) = db[r]; row++;
        }
    }
    Mat xm; solve(A, b, xm, DECOMP_SVD);
    Vec3d xest(xm.at<double>(0), xm.at<double>(1), xm.at<double>(2));

    double e = norm(xest - xgt);
    double consist = 0;
    for (int i = 0; i < n; i++) consist = max(consist, norm(R[i] * xest + t[i] - pworld));
    cout << "[TCP 四点法] 姿态数=" << n << "  x_gt=" << xgt.t() << "  x_est=" << xest.t() << "\n";
    cout << "        位置误差=" << e * 1000 << " mm  一致性(尖偏移)max=" << consist * 1000 << " mm\n";
    if (e > 1e-6) { pass = false; cout << "  [失败] 无噪声下应精确恢复\n"; }

    // ---- 加 1mm 噪声看鲁棒性 ----
    vector<Vec3d> t2 = t;
    normal_distribution<double> mm(0, 0.001);
    for (int i = 0; i < n; i++) t2[i] += Vec3d(mm(g), mm(g), mm(g));
    Mat A2(m, 3, CV_64F), b2(m, 1, CV_64F); row = 0;
    for (int i = 1; i < n; i++) {
        Matx33d dR = R[i] - R[0]; Vec3d db = t2[0] - t2[i];
        for (int r = 0; r < 3; r++) { A2.at<double>(row, 0) = dR(r, 0); A2.at<double>(row, 1) = dR(r, 1); A2.at<double>(row, 2) = dR(r, 2); b2.at<double>(row) = db[r]; row++; }
    }
    Mat xm2; solve(A2, b2, xm2, DECOMP_SVD);
    Vec3d xest2(xm2.at<double>(0), xm2.at<double>(1), xm2.at<double>(2));
    cout << "        +1mm 噪声后 位置误差=" << norm(xest2 - xgt) * 1000 << " mm（应与噪声同量级）\n";
    if (norm(xest2 - xgt) > 0.02) { pass = false; cout << "  [失败] 噪声下误差过大\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] TCP 四点法验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
