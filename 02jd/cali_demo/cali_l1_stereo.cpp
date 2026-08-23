// cali_l1_stereo.cpp
// ============================================================================
// 双目标定（Stereo Calibration）· 基础：求两相机相对位姿
// ----------------------------------------------------------------------------
// 单目标定（见 cali_l1_monocular / L1.3）只标定"一个相机"。
// 双目视觉还需要知道 左相机→右相机 的相对位姿 R,T（||T||=基线 B），
// 这样才能把两幅图像对齐到同一坐标系、做立体匹配与测距。
//
// 本 demo：
//   1. 合成一个双目 rig（已知 K_L,K_R, 真值 R,T, 基线 B）与同步棋盘观测；
//   2. 用 OpenCV stereoCalibrate（CALIB_FIX_INTRINSIC，内参已单目标定好）
//      反估相对位姿 R,T；
//   3. 与真值对拍：旋转角差、平移相对误差、三角化重投影一致性；
//   4. 打印 PASS/FAIL。
// 无需真实相机：整条链路在内存闭环。
//
// 编译: g++ -O2 -std=c++17 cali_l1_stereo.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
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
    const double sq = 0.025;
    vector<Point3f> obj;
    for (int r = 0; r < rn; r++) for (int c = 0; c < cn; c++) obj.push_back(Point3f(c * sq, r * sq, 0));

    const int N = 20;
    Size img(640, 480);
    Matx33d KL(520, 0, img.width / 2.0, 0, 520, img.height / 2.0, 0, 0, 1);
    Matx33d KR(505, 0, img.width / 2.0 + 8, 0, 505, img.height / 2.0 - 6, 0, 0, 1);
    Vec4d dL(0.08, -0.03, 0.001, 0.0005), dR(0.075, -0.028, 0.0012, 0.0004);

    // 真值相对位姿：右相机 = R·左相机 + T ；基线沿 x
    double B = 0.12;                       // 12 cm 基线
    double ay = 15 * CV_PI / 180.0;        // 两相机间 ~15° 偏航，让 R 非平凡
    Matx33d Ry(cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay));
    Matx33d Rgt = Ry;
    Vec3d Tgt(B, 0.002, -0.004);           // 主要沿 x，轻微 y/z 偏移

    Mat KLm(KL), dLm(4, 1, CV_64F, {dL[0], dL[1], dL[2], dL[3]});
    Mat KRm(KR), dRm(4, 1, CV_64F, {dR[0], dR[1], dR[2], dR[3]});

    vector<vector<Point3f>> vobj(N, obj);
    vector<vector<Point2f>> vL, vR;
    mt19937 g(20260817);
    normal_distribution<double> pix(0, 0.5);
    for (int i = 0; i < N; i++) {
        Vec3d rl(rnd(g, -0.5, 0.5), rnd(g, -0.5, 0.5), rnd(g, -0.25, 0.25));
        Vec3d tl(rnd(g, -0.1, 0.1), rnd(g, -0.1, 0.1), rnd(g, 0.4, 0.9));
        Matx33d Rl; Rodrigues(rl, Rl);
        Matx33d Rr = Rgt * Rl;             // 右 = Rgt·Rl（保证两视图共视同一 3D 板）
        Vec3d rr; Rodrigues(Rr, rr);
        Vec3d tr = Rgt * tl + Tgt;         // 右平移 = Rgt·tl + Tgt
        vector<Point2f> pL, pR;
        projectPoints(obj, rl, tl, KLm, dLm, pL);
        projectPoints(obj, rr, tr, KRm, dRm, pR);
        for (size_t k = 0; k < pL.size(); k++) {
            pL[k].x += pix(g); pL[k].y += pix(g);
            pR[k].x += pix(g); pR[k].y += pix(g);
        }
        vL.push_back(pL); vR.push_back(pR);
    }

    // ---------- 双目标定：固定内参，求相对位姿 R,T ----------
    Mat K1 = KLm.clone(), D1 = dLm.clone(), K2 = KRm.clone(), D2 = dRm.clone();
    Mat R, T, E, F;
    double err = stereoCalibrate(vobj, vL, vR, K1, D1, K2, D2, img, R, T, E, F,
                                 CALIB_FIX_INTRINSIC,
                                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 60, 1e-7));
    cout << "[双目标定] 立体重投影误差=" << err << " px\n";
    cout << "  真值 基线 B=" << norm(Tgt) << "  T=(" << Tgt[0] << "," << Tgt[1] << "," << Tgt[2] << ")\n";
    cout << "  估计  T=(" << T.at<double>(0) << "," << T.at<double>(1) << "," << T.at<double>(2) << ")\n";

    // ---------- 对拍：旋转角差 / 平移相对误差 ----------
    Matx33d Rest; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Rest(a, b) = R.at<double>(a, b);
    Matx33d Re = Rest * Rgt.t();
    double ang = acos(max(-1.0, min(1.0, (Re(0, 0) + Re(1, 1) + Re(2, 2) - 1.0) / 2.0))) * 180.0 / CV_PI;
    double tErr = norm(T - Mat(Tgt)) / norm(Mat(Tgt));
    cout << "  旋转角差=" << ang << " deg   平移相对误差=" << tErr * 100 << "%\n";

    // ---------- 对拍：三角化一致性（极线几何闭环）----------
    Mat I0 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    Mat P1 = KLm * I0;                         // 左投影矩阵 [K_L | 0]
    Mat htmp; hconcat(Mat(Rest), Mat(T), htmp); // 右投影矩阵 [K_R·R | K_R·T]
    Mat P2 = KRm * htmp;
    Mat pts4d; triangulatePoints(P1, P2, Mat(vL[0]), Mat(vR[0]), pts4d);
    vector<Point3f> wpts;
    for (int k = 0; k < pts4d.cols; k++) {
        double w = pts4d.at<float>(3, k);
        wpts.push_back(Point3f(pts4d.at<float>(0, k) / w, pts4d.at<float>(1, k) / w, pts4d.at<float>(2, k) / w));
    }
    Vec3d rvecP2; Rodrigues(Rest, rvecP2);
    vector<Point2f> rp1, rp2;
    projectPoints(wpts, Mat::zeros(3, 1, CV_64F), Mat::zeros(3, 1, CV_64F), KLm, Mat::zeros(4, 1, CV_64F), rp1);
    projectPoints(wpts, rvecP2, T, KRm, Mat::zeros(4, 1, CV_64F), rp2);
    double triRMS = 0; int cnt = 0;
    for (size_t k = 0; k < wpts.size(); k++) {
        triRMS += norm(vL[0][k] - rp1[k]) + norm(vR[0][k] - rp2[k]); cnt++;
    }
    triRMS /= cnt;
    cout << "  三角化重投影 RMS=" << triRMS << " px\n";

    // ---------- PASS/FAIL ----------
    if (err > 2.0)  { pass = false; cout << "  [失败] 立体重投影误差过大\n"; }
    if (ang > 2.0)  { pass = false; cout << "  [失败] 旋转角差 >2°\n"; }
    if (tErr > 0.05) { pass = false; cout << "  [失败] 平移相对误差 >5%\n"; }
    if (triRMS > 2.0) { pass = false; cout << "  [失败] 三角化一致性差（极线几何未闭合）\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 双目标定：相对位姿 R,T 恢复通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
