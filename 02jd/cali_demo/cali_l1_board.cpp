// cali_l1_board.cpp
// ============================================================================
// L1.2  标定板（Calibration board）
// ----------------------------------------------------------------------------
// 标定板是"相机→世界"的锚点：没有它就没有 T_target2cam 这个输入。
// 本 demo 用合成图像验证两类最常用板的角点/圆心检测：
//   1) 棋盘格 (chessboard)        -> findChessboardCorners + cornerSubPix
//   2) 圆点阵列 (circles grid)    -> findCirclesGrid (对称阵)
// 并演示"角点 → 带物理尺寸的 3D objectPoints"这一关键映射（标定必需）。
// Charuco 因依赖 aruco 模块，仅在文档中说明，不在本可运行 demo 内实例化。
//
// 编译: g++ -O2 -std=c++17 cali_l1_board.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    bool pass = true;
    const int cn = 9, rn = 6;          // 内角点/圆心 列、行
    const double sq = 40.0;            // 演示用像素间距（真实标定应换成 mm）
    const int margin = 40;
    const int W = margin * 2 + cn * sq, H = margin * 2 + rn * sq;

    // ---------- 1) 棋盘格：画 + 检测 ----------
    Mat img(H, W, CV_8UC3, Scalar(255, 255, 255));
    for (int r = 0; r < rn + 1; r++)
        for (int c = 0; c < cn + 1; c++) {
            bool black = (r + c) % 2 == 0;
            rectangle(img, Point(margin + c * sq, margin + r * sq),
                      Point(margin + (c + 1) * sq, margin + (r + 1) * sq),
                      black ? Scalar(0, 0, 0) : Scalar(255, 255, 255), FILLED);
        }

    vector<Point2f> corners;
    bool found = findChessboardCorners(img, Size(cn, rn), corners,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    cout << "[棋盘格] found=" << (found ? "true" : "false") << " corners=" << corners.size() << "\n";
    if (!found) { pass = false; }
    else {
        // cornerSubPix 要求单通道图, 彩色棋盘格需先转灰度
        Mat imgGray;
        cvtColor(img, imgGray, COLOR_BGR2GRAY);
        cornerSubPix(imgGray, corners, Size(5, 5), Size(-1, -1),
                     TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 1e-6));
        double maxe = 0;
        for (int r = 0; r < rn; r++)
            for (int c = 0; c < cn; c++) {
                Point2f ideal(margin + (c + 1) * sq, margin + (r + 1) * sq);  // 内交点位置
                maxe = max(maxe, norm(corners[r * cn + c] - ideal));
            }
        cout << "        亚像素后 maxErr=" << maxe << " px\n";
        if (corners.size() != (size_t)cn * rn || maxe > 2.0) pass = false;
    }

    // ---------- 2) 圆点阵：画 + 检测 ----------
    Mat img2(H, W, CV_8UC1, Scalar(255));
    vector<Point2f> drawn;
    for (int r = 0; r < rn; r++)
        for (int c = 0; c < cn; c++) {
            Point2f ctr(margin + (c + 0.5) * sq, margin + (r + 0.5) * sq);
            drawn.push_back(ctr);
            circle(img2, ctr, sq * 0.32, Scalar(0), FILLED);
        }
    vector<Point2f> cc;
    bool f2 = findCirclesGrid(img2, Size(cn, rn), cc, CALIB_CB_SYMMETRIC_GRID);
    cout << "[圆点阵] found=" << (f2 ? "true" : "false") << " centers=" << cc.size() << "\n";
    if (!f2) { pass = false; }
    else {
        double maxe = 0;
        for (size_t i = 0; i < cc.size(); i++) maxe = max(maxe, norm(cc[i] - drawn[i]));
        cout << "        maxErr=" << maxe << " px\n";
        if (cc.size() != (size_t)cn * rn || maxe > 2.0) pass = false;
    }

    // ---------- 3) 角点 → 带物理尺寸的 3D objectPoints（标定关键映射）----------
    const double sq_mm = 25.0;        // 真实标定板格子边长，单位 mm
    vector<Point3f> obj;
    for (int r = 0; r < rn; r++)
        for (int c = 0; c < cn; c++)
            obj.push_back(Point3f(c * sq_mm, r * sq_mm, 0));
    cout << "[映射] 生成 objectPoints=" << obj.size()
         << " 个，间距=" << sq_mm << " mm，例: " << obj[0] << " .. " << obj.back() << "\n";

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 标定板检测与 3D 映射验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
