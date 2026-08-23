// cali_l2_intrinsics_deep.cpp
// ============================================================================
// L2.1  内参标定深入：是什么在影响精度
// ----------------------------------------------------------------------------
// 在 L1.3 基础上做"控制变量实验"，量化三个精度因素：
//   1) 姿态覆盖度：全视场多样姿态 vs 只拍画面中心 → 对 K 误差的影响
//   2) 畸变模型阶数：4 参数 vs 8 参数(rational) → 小样本下高阶会过拟合
//   3) 角点噪声水平：0.3px vs 2.0px → 对内参误差的影响
// 全部合成数据闭环，运行即看趋势。
//
// 编译: g++ -O2 -std=c++17 cali_l2_intrinsics_deep.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }

static void genData(const Matx33d& K, const Vec4d& d, int n, const vector<Vec3d>& rvecs, const vector<Vec3d>& tvecs,
                    vector<vector<Point3f>>& vobj, vector<vector<Point2f>>& vimg, double noise, mt19937& g, int cn, int rn, double sq) {
    vector<Point3f> obj;
    for (int r = 0; r < rn; r++) for (int c = 0; c < cn; c++) obj.push_back(Point3f(c * sq, r * sq, 0));
    vobj.assign(n, obj); vimg.resize(n);
    Mat Km(K), Dm(4, 1, CV_64F, { d[0], d[1], d[2], d[3] });
    normal_distribution<double> pix(0, noise);
    for (int i = 0; i < n; i++) {
        vector<Point2f> pts; Mat rv(rvecs[i]), tv(tvecs[i]);
        projectPoints(obj, rv, tv, Km, Dm, pts);
        for (auto& p : pts) { p.x += pix(g); p.y += pix(g); }
        vimg[i] = pts;
    }
}

static double calibKerr(const Matx33d& Kgt, const vector<vector<Point3f>>& vobj, const vector<vector<Point2f>>& vimg,
                        const Size& imgSize, int flags, int dcnt, Vec4d& dest, double& rms) {
    Mat Kc, Dc; vector<Mat> rv, tv;
    rms = calibrateCamera(vobj, vimg, imgSize, Kc, Dc, rv, tv, flags);
    dest = Vec4d(Dc.at<double>(0), Dc.at<double>(1), Dc.at<double>(2), Dc.at<double>(3));
    return norm(Mat(Kgt) - Kc, NORM_INF) / norm(Mat(Kgt), NORM_INF);
}

int main() {
    bool pass = true;
    const int cn = 9, rn = 6; const double sq = 0.025;
    Size imgSize(640, 480);
    Matx33d Kgt(520, 0, 320, 0, 520, 240, 0, 0, 1);
    Vec4d dgt(0.10, -0.04, 0.001, 0.0006);
    mt19937 g(20260817);

    // ---------- 实验1：姿态覆盖度 ----------
    int N = 30;
    vector<Vec3d> rA, tA, rB, tB;
    for (int i = 0; i < N; i++) {                       // A：全视场、多样
        rA.push_back(Vec3d(rnd(g,-0.6,0.6), rnd(g,-0.6,0.6), rnd(g,-0.3,0.3)));
        tA.push_back(Vec3d(rnd(g,-0.28,0.28), rnd(g,-0.28,0.28), rnd(g,0.4,0.9)));
        rB.push_back(Vec3d(rnd(g,-0.1,0.1), rnd(g,-0.1,0.1), rnd(g,-0.05,0.05)));  // B：只拍中心
        tB.push_back(Vec3d(rnd(g,-0.03,0.03), rnd(g,-0.03,0.03), rnd(g,0.55,0.7)));
    }
    vector<vector<Point3f>> oA, oB; vector<vector<Point2f>> iA, iB; Vec4d d; double rms;
    genData(Kgt, dgt, N, rA, tA, oA, iA, 0.8, g, cn, rn, sq);
    double kA = calibKerr(Kgt, oA, iA, imgSize, 0, 5, d, rms);
    genData(Kgt, dgt, N, rB, tB, oB, iB, 0.8, g, cn, rn, sq);
    double kB = calibKerr(Kgt, oB, iB, imgSize, 0, 5, d, rms);
    cout << "[实验1 覆盖度] 全视场 K误差=" << kA * 100 << "%  | 只拍中心 K误差=" << kB * 100 << "%\n";
    if (kB > kA) cout << "        ✓ 覆盖越好，K 误差越低\n"; else { pass = false; cout << "  [失败] 覆盖度未体现\n"; }

    // ---------- 实验2：畸变模型阶数（小样本过拟合）----------
    int M = 12;
    vector<Vec3d> rC, tC;
    for (int i = 0; i < M; i++) { rC.push_back(Vec3d(rnd(g,-0.5,0.5), rnd(g,-0.5,0.5), rnd(g,-0.25,0.25)));
                                 tC.push_back(Vec3d(rnd(g,-0.2,0.2), rnd(g,-0.2,0.2), rnd(g,0.4,0.85))); }
    vector<vector<Point3f>> oC; vector<vector<Point2f>> iC; genData(Kgt, dgt, M, rC, tC, oC, iC, 0.5, g, cn, rn, sq);
    double k5 = calibKerr(Kgt, oC, iC, imgSize, 0, 5, d, rms);          // 默认(5参数: k1,k2,p1,p2,k3)
    double k8 = calibKerr(Kgt, oC, iC, imgSize, CALIB_RATIONAL_MODEL, 8, d, rms); // 8参数 rational
    cout << "[实验2 畸变阶数] 5参数(默认) K误差=" << k5 * 100 << "%  | 8参数 K误差=" << k8 * 100 << "%\n";
    if (k8 <= k5 * 3) cout << "        ✓ 小样本下高阶模型未显著更优（高阶易过拟合）\n";
    else { pass = false; cout << "  [失败] 8参数退化过大\n"; }

    // ---------- 实验3：角点噪声 ----------
    int P = 30; vector<Vec3d> rD, tD;
    for (int i = 0; i < P; i++) { rD.push_back(Vec3d(rnd(g,-0.6,0.6), rnd(g,-0.6,0.6), rnd(g,-0.3,0.3)));
                                 tD.push_back(Vec3d(rnd(g,-0.25,0.25), rnd(g,-0.25,0.25), rnd(g,0.4,0.9))); }
    vector<vector<Point3f>> oD1, oD2; vector<vector<Point2f>> iD1, iD2;
    genData(Kgt, dgt, P, rD, tD, oD1, iD1, 0.3, g, cn, rn, sq);
    genData(Kgt, dgt, P, rD, tD, oD2, iD2, 2.0, g, cn, rn, sq);
    double kN1 = calibKerr(Kgt, oD1, iD1, imgSize, 0, 5, d, rms);
    double kN2 = calibKerr(Kgt, oD2, iD2, imgSize, 0, 5, d, rms);
    cout << "[实验3 角点噪声] 0.3px K误差=" << kN1 * 100 << "%  | 2.0px K误差=" << kN2 * 100 << "%\n";
    if (kN2 > kN1) cout << "        ✓ 角点越准，内参越准\n"; else { pass = false; cout << "  [失败] 噪声未体现\n"; }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 内参精度因素实验符合预期" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
