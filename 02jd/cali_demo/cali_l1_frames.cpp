// cali_l1_frames.cpp
// ============================================================================
// L1.1  坐标变换与刚体变换基础
// ----------------------------------------------------------------------------
// 标定的一切都建立在"坐标系 + 刚体变换"上。本 demo 用合成数据验证三件事：
//   1) 齐次变换矩阵 T 的 复合 / 求逆 正确（核心中的核心）；
//   2) 四种旋转表示（旋转矩阵 / 轴角(旋转向量) / 四元数 / 欧拉角）互转闭环；
//   3) 坐标链 target -> cam -> gripper -> base 的一致性
//      T_target2base = T_gripper2base * T_cam2gripper * T_target2cam
// 全程不依赖真机/真相机：随机生成位姿即可复现，运行即出 PASS/FAIL。
//
// 编译 (Linux):
//   g++ -O2 -std=c++17 cali_l1_frames.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// 编译 (MSVC):
//   cl /O2 /std:c++17 cali_l1_frames.cpp /I<opencv>/include <opencv>/lib/opencv_world*.lib
// ============================================================================
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }

// 随机生成一个刚体位姿 T ∈ SE(3)：小幅随机旋转 + 随机平移
static Matx44d randPose(mt19937& g) {
    Vec3d axis{ rnd(g, -1, 1), rnd(g, -1, 1), rnd(g, -1, 1) };
    double n = norm(axis); axis = axis / (n + 1e-9);
    double ang = rnd(g, -0.8, 0.8);                 // 小角度，远离欧拉奇点
    Mat R; Rodrigues(axis * ang, R);                // 3x3 旋转矩阵
    Vec3d t{ rnd(g, -0.5, 0.5), rnd(g, -0.5, 0.5), rnd(g, 0.3, 1.0) };
    Matx44d T = Matx44d::eye();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) T(i, j) = R.at<double>(i, j);
    for (int i = 0; i < 3; i++) T(i, 3) = t[i];
    return T;
}

static Matx44d inv44(const Matx44d& T) { Mat M(T), Mi; invert(M, Mi, DECOMP_SVD); return Matx44d(Mi); }

static Vec3d applyPose(const Matx44d& T, const Vec3d& p) {
    Vec4d h(p[0], p[1], p[2], 1.0);
    Vec4d r = T * h;
    return Vec3d(r[0] / r[3], r[1] / r[3], r[2] / r[3]);
}

// ---- 四种旋转表示互转（自己实现，便于看清数学）----
static Vec4d rvec2quat(const Vec3d& r) {                 // 旋转向量 -> 四元数 (w,x,y,z)
    double a = norm(r);
    if (a < 1e-12) return Vec4d(1, 0, 0, 0);
    Vec3d u = r / a; double s = sin(a / 2);
    return Vec4d(cos(a / 2), u[0] * s, u[1] * s, u[2] * s);
}
static Matx33d quat2R(const Vec4d& q) {                   // 四元数 -> 旋转矩阵
    double w = q[0], x = q[1], y = q[2], z = q[3];
    return Matx33d(
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y));
}
static Vec3d R2euler(const Matx33d& R) {                  // 旋转矩阵 -> ZYX 欧拉角 (yaw,pitch,roll)
    double pitch = asin(-clamp(R(2, 0), -1.0, 1.0));
    double yaw = atan2(R(1, 0), R(0, 0));
    double roll = atan2(R(2, 1), R(2, 2));
    return Vec3d(yaw, pitch, roll);
}
static Matx33d euler2R(const Vec3d& e) {                  // ZYX 欧拉角 -> 旋转矩阵
    double y = e[0], p = e[1], r = e[2];
    Matx33d Rz(cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1);
    Matx33d Ry(cos(p), 0, sin(p), 0, 1, 0, -sin(p), 0, cos(p));
    Matx33d Rx(1, 0, 0, 0, cos(r), -sin(r), 0, sin(r), cos(r));
    return Rz * Ry * Rx;
}

int main() {
    mt19937 g(20260817);
    bool pass = true;

    // ---------- 测试 1：齐次矩阵 复合 / 求逆 ----------
    Matx44d I = Matx44d::eye();
    for (int k = 0; k < 20; k++) {
        Matx44d T = randPose(g);
        Matx44d Tinv = inv44(T);
        Matx44d P = T * Tinv;                 // 应为单位阵
        double err = norm(Mat(P - I), NORM_INF);
        if (err > 1e-9) { pass = false; cout << "  [失败] 复合/求逆 err=" << err << endl; }
    }
    cout << "[测试1] 齐次矩阵 T·T⁻¹=I  (20 组随机位姿)\n";

    // ---------- 测试 2：四种旋转表示互转闭环 ----------
    for (int k = 0; k < 20; k++) {
        Matx44d T = randPose(g);
        Matx33d R; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R(i, j) = T(i, j);

        // 旋转矩阵 -> 旋转向量 -> 四元数 -> 旋转矩阵
        Vec3d rvec; Rodrigues(Mat(R), rvec);
        Vec4d q = rvec2quat(rvec);
        Matx33d Rback = quat2R(q);
        double e1 = norm(Mat(R - Rback), NORM_INF);

        // 旋转矩阵 -> 欧拉 -> 旋转矩阵
        Vec3d eu = R2euler(R);
        Matx33d Rback2 = euler2R(eu);
        double e2 = norm(Mat(R - Rback2), NORM_INF);

        if (e1 > 1e-6 || e2 > 1e-6) { pass = false; cout << "  [失败] 旋转互转 e1=" << e1 << " e2=" << e2 << endl; }
    }
    cout << "[测试2] 旋转矩阵↔轴角↔四元数↔欧拉 闭环 (20 组)\n";

    // ---------- 测试 3：坐标链一致性 ----------
    // 任取三个固定变换：G=gripper2base, X=cam2gripper, Ttc=target2cam
    Matx44d G = randPose(g), X = randPose(g), Ttc = randPose(g);
    // 标定板在基座下的位姿由链式定义：Tb = G * X * Ttc
    Matx44d Tb = G * X * Ttc;
    // 取标定板上的一个点 p，两种路径变换到基座系，必须一致
    Vec3d p{ 0.12, -0.05, 0.0 };
    Vec3d v_direct = applyPose(Tb, p);
    // 路径二：target -> cam -> gripper -> base
    Vec3d v_chain = applyPose(G, applyPose(X, applyPose(Ttc, p)));
    double e3 = norm(v_direct - v_chain, NORM_INF);
    if (e3 > 1e-9) { pass = false; cout << "  [失败] 坐标链 err=" << e3 << endl; }
    cout << "[测试3] T_target2base = T_gripper2base·T_cam2gripper·T_target2cam  (链一致性)\n";
    cout << "        点 p=" << p << " -> 基座系 direct=" << v_direct.t()
         << " chain=" << v_chain.t() << "  err=" << e3 << "\n";

    // ---------- 总结 ----------
    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 坐标变换基础全部验证通过" : " [FAIL] 存在错误，见上") << "\n";
    cout << "=================================================\n";
    return pass ? 0 : 1;
}
