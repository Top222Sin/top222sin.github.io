// cali_l2_stereo_rect.cpp
// ============================================================================
// 双目标定（Stereo）· 深入：立体校正 + 极线对齐 + 视差→深度
// ----------------------------------------------------------------------------
// 上篇 cali_l1_stereo 求出了相对位姿 R,T。本篇接着做双目的"最后一公里"：
//   - stereoRectify 把两图极线拉平成水平且行对齐（把二维匹配降成一维搜索）；
//   - 用校正后几何验证"同一 3D 点两图落在同一行"（垂直视差≈0）；
//   - 视差→深度 Z = f·B / d，并演示 ① 基线过小 / ② 标定 R,T 误差 如何放大深度误差。
// 无需真实相机：整条链路在内存闭环，运行即出 PASS/FAIL。
//
// 编译: g++ -O2 -std=c++17 cali_l2_stereo_rect.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }

// 构造一个双目 rig 的真值（内参 + 相对位姿 + 基线 B）
static void makeRig(Size img, Matx33d& KL, Mat& dL, Matx33d& KR, Mat& dR,
                    Matx33d& Rgt, Vec3d& Tgt, double B) {
    KL = Matx33d(520, 0, img.width / 2.0, 0, 520, img.height / 2.0, 0, 0, 1);
    KR = Matx33d(505, 0, img.width / 2.0 + 8, 0, 505, img.height / 2.0 - 6, 0, 0, 1);
    dL = Mat(4, 1, CV_64F, {0.08, -0.03, 0.001, 0.0005});
    dR = Mat(4, 1, CV_64F, {0.075, -0.028, 0.0012, 0.0004});
    double ay = 15 * CV_PI / 180.0;
    Rgt = Matx33d(cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay));
    Tgt = Vec3d(B, 0.002, -0.004);
}

int main() {
    bool pass = true;
    Size img(640, 480);
    Matx33d KL, KR; Mat dL, dR; Matx33d Rgt; Vec3d Tgt;
    makeRig(img, KL, dL, KR, dR, Rgt, Tgt, 0.12);   // 基线 B = 12 cm
    Mat KLm(KL), KRm(KR);

    // ---------- 立体校正（用上篇得到的相对位姿 R,T）----------
    Mat R1, R2, P1, P2, Q;
    stereoRectify(KLm, dL, KRm, dR, img, Mat(Rgt), Mat(Tgt), R1, R2, P1, P2, Q, 0, 1.0);

    // ---------- 验证：极线对齐（垂直视差 ≈ 0） + 深度----------
    // 在"左相机坐标系"下取一批 3D 点，经校正投影矩阵 P1/P2 投影：
    // 校正良好时，同一点的两图 y 坐标应一致（极线水平、行对齐）。
    vector<Point3f> wpts;
    mt19937 g(7);
    for (int i = 0; i < 40; i++) {
        double z = rnd(g, 0.3, 0.8);
        double x = rnd(g, -0.25, 0.25) * z;
        double y = rnd(g, -0.18, 0.18) * z;
        wpts.push_back(Point3f(x, y, z));
    }
    double f = P1.at<double>(0, 0);
    double Bfx = -P2.at<double>(0, 3) / f;          // 由 P2 反推基线 B
    double vdMax = 0, zErrMax = 0;
    double demoZ = 0, demoD = 0, demoZest = 0;
    for (auto& X : wpts) {
        Mat x1 = P1 * Mat(Matx41d(X.x, X.y, X.z, 1));
        Mat x2 = P2 * Mat(Matx41d(X.x, X.y, X.z, 1));
        Point2f p1(x1.at<double>(0) / x1.at<double>(2), x1.at<double>(1) / x1.at<double>(2));
        Point2f p2(x2.at<double>(0) / x2.at<double>(2), x2.at<double>(1) / x2.at<double>(2));
        vdMax = max(vdMax, (double)abs(p1.y - p2.y));        // 垂直视差
        double d = p1.x - p2.x;                      // 视差
        if (d > 1e-3) {
            double Zest = f * Bfx / d;               // 深度 = f·B / d
            zErrMax = max(zErrMax, abs(Zest - X.z) / X.z);
            if (demoD == 0) { demoZ = X.z; demoD = d; demoZest = Zest; }
        }
    }
    cout << "[双目校正] 最长垂直视差=" << vdMax << " px   最大深度相对误差=" << zErrMax * 100 << "%\n";
    cout << "  焦距 f=" << f << " px   由 P2 反推基线 B=" << Bfx << " m\n";
    cout << "  示例点：真值 Z=" << demoZ << " m  视差 d=" << demoD << " px  估得 Z=" << demoZest << " m\n";

    // ---------- 退化①：基线过小 ----------
    Matx33d KL2, KR2; Mat dL2, dR2; Matx33d Rgt2; Vec3d Tgt2;
    makeRig(img, KL2, dL2, KR2, dR2, Rgt2, Tgt2, 0.03);   // 基线缩到 3 cm
    Mat R1s, R2s, P1s, P2s, Qs;
    stereoRectify(Mat(KL2), dL2, Mat(KR2), dR2, img, Mat(Rgt2), Mat(Tgt2), R1s, R2s, P1s, P2s, Qs, 0, 1.0);
    double fS = P1s.at<double>(0, 0), BS = -P2s.at<double>(0, 3) / fS, zErrSmallB = 0;
    for (auto& X : wpts) {
        Mat x1 = P1s * Mat(Matx41d(X.x, X.y, X.z, 1));
        Mat x2 = P2s * Mat(Matx41d(X.x, X.y, X.z, 1));
        Point2f p1(x1.at<double>(0) / x1.at<double>(2), x1.at<double>(1) / x1.at<double>(2));
        Point2f p2(x2.at<double>(0) / x2.at<double>(2), x2.at<double>(1) / x2.at<double>(2));
        double d = p1.x - p2.x;
        if (d > 1e-3) zErrSmallB = max(zErrSmallB, abs(fS * BS / d - X.z) / X.z);
    }
    cout << "  [退化①] 基线 3cm 时最大深度相对误差=" << zErrSmallB * 100
         << "%  （基线越小，同样视差误差→深度误差越大）\n";

    // ---------- 退化②：标定 R,T 误差 5%（基线方向 +5%）----------
    Vec3d Tbad = Tgt * 1.05;
    Mat R1b, R2b, P1b, P2b, Qb;
    stereoRectify(KLm, dL, KRm, dR, img, Mat(Rgt), Mat(Tbad), R1b, R2b, P1b, P2b, Qb, 0, 1.0);
    double vdBad = 0;
    for (auto& X : wpts) {
        Mat x1 = P1b * Mat(Matx41d(X.x, X.y, X.z, 1));
        Mat x2 = P2b * Mat(Matx41d(X.x, X.y, X.z, 1));
        Point2f p1(x1.at<double>(0) / x1.at<double>(2), x1.at<double>(1) / x1.at<double>(2));
        Point2f p2(x2.at<double>(0) / x2.at<double>(2), x2.at<double>(1) / x2.at<double>(2));
        vdBad = max(vdBad, (double)abs(p1.y - p2.y));
    }
    cout << "  [退化②] R,T 误差 5% 时最长垂直视差=" << vdBad << " px  （极线不再行对齐，匹配将失败）\n";

    // ---------- PASS/FAIL ----------
    if (vdMax > 1e-2) { pass = false; cout << "  [失败] 校正后垂直视差过大（极线未对齐）\n"; }
    if (zErrMax > 0.05) { pass = false; cout << "  [失败] 深度相对误差 >5%\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 双目校正 + 视差→深度 验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
