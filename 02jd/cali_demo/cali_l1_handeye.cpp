// cali_l1_handeye.cpp
// ============================================================================
// L1.5  手眼标定（眼在手上 / 眼在手外）
// ----------------------------------------------------------------------------
// 手眼标定求解"相机系 ↔ 机械臂系"的固定刚体变换 X，核心是 AX = XB：
//   A = 法兰(夹爪)在基座下的相对运动，B = 标定板在相机下的相对运动。
// 本 demo 自实现 Park–Martin 对偶四元数(SVD)求解器，并用合成数据 + 真值 X_gt 闭环；
// 同时调用 OpenCV calibrateHandEye 作交叉验证。
// 关键点：眼在手 vs 眼在外，A、B 的"相对运动方向"恰好相反（代码里显式标注）。
//
// 编译: g++ -O2 -std=c++17 cali_l1_handeye.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }
static Matx33d randR(mt19937& g) {
    Vec3d axis{ rnd(g, -1, 1), rnd(g, -1, 1), rnd(g, -1, 1) };
    double n = norm(axis); axis /= (n + 1e-9);
    Mat R; Rodrigues(axis * rnd(g, -0.7, 0.7), R); return Matx33d(R);
}
static Matx44d matRT(const Matx33d& R, const Vec3d& t) {
    Matx44d M; M = Matx44d::eye();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) M(i, j) = R(i, j);
    for (int i = 0; i < 3; i++) M(i, 3) = t[i];
    return M;
}
static Matx44d inv44(const Matx44d& T) { Mat M(T), Mi; invert(M, Mi, DECOMP_SVD); return Matx44d(Mi); }

// ---- 旋转 ↔ 四元数(w,x,y,z) ----
static Vec4d quatFromR(const Matx33d& R) {
    double tr = R(0, 0) + R(1, 1) + R(2, 2); double w, x, y, z;
    if (tr > 0) { double s = 0.5 / sqrt(tr + 1); w = 0.25 / s; x = (R(2, 1) - R(1, 2)) * s; y = (R(0, 2) - R(2, 0)) * s; z = (R(1, 0) - R(0, 1)) * s; }
    else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) { double s = 2 * sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2)); x = 0.25 * s; y = (R(0, 1) + R(1, 0)) / s; z = (R(0, 2) + R(2, 0)) / s; w = (R(2, 1) - R(1, 2)) / s; }
    else if (R(1, 1) > R(2, 2)) { double s = 2 * sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2)); y = 0.25 * s; x = (R(0, 1) + R(1, 0)) / s; z = (R(1, 2) + R(2, 1)) / s; w = (R(0, 2) - R(2, 0)) / s; }
    else { double s = 2 * sqrt(1 + R(2, 2) - R(0, 0) - R(1, 1)); z = 0.25 * s; x = (R(0, 2) + R(2, 0)) / s; y = (R(1, 2) + R(2, 1)) / s; w = (R(1, 0) - R(0, 1)) / s; }
    return Vec4d(w, x, y, z);
}
static Matx33d RfromQuat(const Vec4d& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    return Matx33d(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                   2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                   2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
}
// S(q)·p = q⊗p ; W(q)·p = p⊗q
static Matx44d Sq(const Vec4d& q) { double w=q[0],x=q[1],y=q[2],z=q[3]; return Matx44d(w,-x,-y,-z, x,w,-z,y, y,z,w,-x, z,-y,x,w); }
static Matx44d Wq(const Vec4d& q) { double w=q[0],x=q[1],y=q[2],z=q[3]; return Matx44d(w,-x,-y,-z, x,w,z,-y, y,-z,w,x, z,y,-x,w); }

// 解 AX = XB（A、B 为相对运动序列），返回 X = [Rx tx; 0 1]
static void solveHandEye(const vector<Matx33d>& RA, const vector<Vec3d>& tA,
                         const vector<Matx33d>& RB, const vector<Vec3d>& tB,
                         Matx33d& Rx, Vec3d& tx) {
    int k = (int)RA.size();
    Mat M(4 * k, 4, CV_64F);
    for (int i = 0; i < k; i++) {
        Vec4d qa = quatFromR(RA[i]), qb = quatFromR(RB[i]);
        Matx44d blk = Sq(qa) - Wq(qb);
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) M.at<double>(4 * i + r, c) = blk(r, c);
    }
    SVD svd(M, SVD::MODIFY_A | SVD::FULL_UV);
    Vec4d qx(svd.vt.at<double>(3, 0), svd.vt.at<double>(3, 1), svd.vt.at<double>(3, 2), svd.vt.at<double>(3, 3));
    qx = qx / norm(qx);
    Rx = RfromQuat(qx);
    Mat NA(3 * k, 3, CV_64F), nb(3 * k, 1, CV_64F);
    for (int i = 0; i < k; i++) {
        Matx33d blk = RA[i] - Matx33d::eye();
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) NA.at<double>(3 * i + r, c) = blk(r, c);
        Vec3d rhs = Rx * tB[i] - tA[i];
        for (int r = 0; r < 3; r++) nb.at<double>(3 * i + r) = rhs[r];
    }
    Mat txm; solve(NA, nb, txm, DECOMP_SVD);
    tx = Vec3d(txm.at<double>(0), txm.at<double>(1), txm.at<double>(2));
}

// 评估 X_est 相对 X_gt：旋转角(度) + 平移误差(米)
static void evalX(const Matx44d& Xest, const Matx44d& Xgt, double& angDeg, double& tErr) {
    Matx44d E = Xgt.inv() * Xest;            // 应≈单位阵
    Matx33d Re = Matx33d(E(0,0),E(0,1),E(0,2), E(1,0),E(1,1),E(1,2), E(2,0),E(2,1),E(2,2));
    Vec3d rvec; Rodrigues(Mat(Re), rvec);
    angDeg = norm(rvec) * 180.0 / CV_PI;
    tErr = norm(Vec3d(E(0,3), E(1,3), E(2,3)));
}

int main() {
    bool pass = true;
    mt19937 g(20260817);
    const int N = 14;                          // 采集位姿数

    // ---------- 眼在手上 (eye-in-hand): X = cam2gripper ----------
    Matx44d Xei = matRT(randR(g), Vec3d(0.03, 0.02, 0.08));   // 真值
    vector<Matx44d> G(N), T(N);
    for (int i = 0; i < N; i++) { G[i] = matRT(randR(g), Vec3d(rnd(g,-0.3,0.3), rnd(g,-0.3,0.3), rnd(g,0.4,0.8)));
                                 T[i] = inv44(Xei) * inv44(G[i]); }   // target2base = I
    vector<Matx33d> RA(N-1), RB(N-1); vector<Vec3d> tA(N-1), tB(N-1);
    for (int i = 0; i < N-1; i++) {                       // 约定: A=G_i⁻¹G_{i+1}, B=T_i T_{i+1}⁻¹
        Matx44d Ai = inv44(G[i]) * G[i+1], Bi = T[i] * inv44(T[i+1]);
        for (int r=0;r<3;r++) for(int c=0;c<3;c++){ RA[i](r,c)=Ai(r,c); RB[i](r,c)=Bi(r,c); }
        tA[i] = Vec3d(Ai(0,3),Ai(1,3),Ai(2,3)); tB[i] = Vec3d(Bi(0,3),Bi(1,3),Bi(2,3));
    }
    Matx33d Rx; Vec3d tx; solveHandEye(RA, tA, RB, tB, Rx, tx);
    Matx44d Xei_est = matRT(Rx, tx);
    double a1, e1; evalX(Xei_est, Xei, a1, e1);
    cout << "[眼在手上] 旋转误差=" << a1 << "°  平移误差=" << e1 * 1000 << " mm\n";
    if (a1 > 0.5 || e1 > 0.01) pass = false;

    // 与 OpenCV 交叉验证
    vector<Mat> Rg, tg, Rt, tt;
    for (int i = 0; i < N; i++) {
        Matx33d Rg_i, Rt_i; Vec3d tg_i, tt_i;
        for (int r=0;r<3;r++) for(int c=0;c<3;c++){ Rg_i(r,c)=G[i](r,c); Rt_i(r,c)=T[i](r,c); }
        tg_i=Vec3d(G[i](0,3),G[i](1,3),G[i](2,3)); tt_i=Vec3d(T[i](0,3),T[i](1,3),T[i](2,3));
        Rg.push_back(Mat(Rg_i)); tg.push_back(Mat(tg_i)); Rt.push_back(Mat(Rt_i)); tt.push_back(Mat(tt_i));
    }
    Mat Rcg, tcg;
    calibrateHandEye(Rg, tg, Rt, tt, Rcg, tcg, CALIB_HAND_EYE_TSAI);
    Matx44d Xcv = matRT(Matx33d(Rcg), Vec3d(tcg.at<double>(0),tcg.at<double>(1),tcg.at<double>(2)));
    double a2, e2; evalX(Xcv, Xei, a2, e2);
    cout << "[眼在手上|OpenCV] 旋转误差=" << a2 << "°  平移误差=" << e2 * 1000 << " mm\n";

    // ---------- 眼在手外 (eye-to-hand): X = cam2base ----------
    Matx44d Xet = matRT(randR(g), Vec3d(0.10, -0.05, 0.40));
    Matx44d C = matRT(randR(g), Vec3d(0.0, 0.0, 0.0));    // 板固定在夹爪上(target2gripper)
    for (int i = 0; i < N; i++) { G[i] = matRT(randR(g), Vec3d(rnd(g,-0.3,0.3), rnd(g,-0.3,0.3), rnd(g,0.4,0.8)));
                                 T[i] = inv44(Xet) * G[i] * C; }
    for (int i = 0; i < N-1; i++) {                       // 约定相反: A=G_{i+1}G_i⁻¹, B=T_{i+1}T_i⁻¹
        Matx44d Ai = G[i+1] * inv44(G[i]), Bi = T[i+1] * inv44(T[i]);
        for (int r=0;r<3;r++) for(int c=0;c<3;c++){ RA[i](r,c)=Ai(r,c); RB[i](r,c)=Bi(r,c); }
        tA[i] = Vec3d(Ai(0,3),Ai(1,3),Ai(2,3)); tB[i] = Vec3d(Bi(0,3),Bi(1,3),Bi(2,3));
    }
    solveHandEye(RA, tA, RB, tB, Rx, tx);
    Matx44d Xet_est = matRT(Rx, tx);
    double a3, e3; evalX(Xet_est, Xet, a3, e3);
    cout << "[眼在手外] 旋转误差=" << a3 << "°  平移误差=" << e3 * 1000 << " mm\n";
    if (a3 > 0.5 || e3 > 0.01) pass = false;

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 手眼标定(眼外/手) 自实现求解器 + OpenCV 交叉验证通过" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
