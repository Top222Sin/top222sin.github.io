// cali_l1_intrinsics.cpp
// ============================================================================
// L1.3  内参标定（Intrinsics，张正友法）
// ----------------------------------------------------------------------------
// 用已知真值 K + 畸变，合成 N 张"棋盘图"的 2D 角点（带像素噪声），
// 再调 OpenCV calibrateCamera 反估 K + 畸变，与真值对拍，并打印重投影 RMS。
// 无需真实相机：整条链路在内存里闭环，运行即出 PASS/FAIL。
//
// 编译: g++ -O2 -std=c++17 cali_l1_intrinsics.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }

int main() {
    bool pass = true;
    const int cn = 9, rn = 6;
    const double sq = 0.025;                 // 真实格子 25 mm
    vector<Point3f> obj;
    for (int r = 0; r < rn; r++) for (int c = 0; c < cn; c++) obj.push_back(Point3f(c * sq, r * sq, 0));

    const int N = 24;
    Size imgSize(640, 480);
    Matx33d Kgt(520, 0, imgSize.width / 2.0, 0, 520, imgSize.height / 2.0, 0, 0, 1);  // 真值内参
    Vec4d dgt(0.09, -0.035, 0.001, 0.0006);  // 真值畸变 (k1,k2,p1,p2)
    Mat K(Kgt), D(4, 1, CV_64F, {dgt[0], dgt[1], dgt[2], dgt[3]});

    vector<vector<Point3f>> vobj(N, obj);
    vector<vector<Point2f>> vimg;
    mt19937 g(20260817);
    normal_distribution<double> pix(0, 0.6);   // 0.6 px 角点噪声
    for (int i = 0; i < N; i++) {
        Vec3d rvec(rnd(g, -0.5, 0.5), rnd(g, -0.5, 0.5), rnd(g, -0.25, 0.25));
        Vec3d tvec(rnd(g, -0.15, 0.15), rnd(g, -0.15, 0.15), rnd(g, 0.35, 0.9));
        vector<Point2f> pts;
        projectPoints(obj, rvec, tvec, K, D, pts);
        for (auto& p : pts) { p.x += pix(g); p.y += pix(g); }
        vimg.push_back(pts);
    }

    // ---------- 反估 ----------
    Mat Kc, Dc; vector<Mat> rvecs, tvecs;
    double rms = calibrateCamera(vobj, vimg, imgSize, Kc, Dc, rvecs, tvecs);
    cout << "[内参标定] 图像数=" << N << "  重投影 RMS=" << rms << " px\n";
    cout << "  真值 K = " << Mat(Kgt) << "\n";
    cout << "  估计 K = " << Kc << "\n";
    cout << "  真值 dist(k1,k2,p1,p2) = " << dgt << "\n";
    cout << "  估计 dist = " << Dc.t() << "\n";

    // ---------- 对拍 ----------
    double kerr = norm(Mat(Kgt) - Kc, NORM_INF) / norm(Mat(Kgt), NORM_INF);
    double derr = max({ abs(Dc.at<double>(0) - dgt[0]), abs(Dc.at<double>(1) - dgt[1]),
                        abs(Dc.at<double>(2) - dgt[2]), abs(Dc.at<double>(3) - dgt[3]) });
    cout << "  内参相对误差=" << kerr * 100 << "%  畸变最大绝对误差=" << derr << "\n";

    if (rms > 2.0) { pass = false; cout << "  [失败] 重投影 RMS 过大\n"; }
    if (kerr > 0.05) { pass = false; cout << "  [失败] 内参相对误差 >5%\n"; }
    if (derr > 0.05) { pass = false; cout << "  [失败] 畸变误差过大\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 内参标定(张正友法)验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
