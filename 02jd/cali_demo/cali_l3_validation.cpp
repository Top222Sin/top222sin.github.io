// cali_l3_validation.cpp
// ============================================================================
// L3.1  验证方案（Validation）⭐
// ----------------------------------------------------------------------------
// 标定完得到 K / 手眼 X / TCP，怎么证明"这套参数真的有用"？
// 本 demo 做端到端验证（非重投影，而是"机器人按这套参数去够目标，偏多少"）：
//   ① 投影验证：用估计参数把已知 3D 目标投成像素，对比相机实际看到的像素
//   ② 闭环够取验证：相机看到像素 → 用估计参数反投影到已知高度平面 → 指挥 TCP 到达
//     目标在已知高度平面 z=z_t 上，比较"到达点"与"真值目标"的 3D 偏差
// 用两组标定质量（好 / 差）对比，证明验证指标能"抓出"坏标定。
// 全程合成数据（已知真值），运行即出 PASS/FAIL + 分指标。
//
// 编译: g++ -O2 -std=c++17 cali_l3_validation.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b){ return uniform_real_distribution<double>(a,b)(g); }
static Matx44d matRT(const Matx33d& R, const Vec3d& t){ Matx44d M=Matx44d::eye(); for(int i=0;i<3;i++){for(int j=0;j<3;j++)M(i,j)=R(i,j);M(i,3)=t[i];} return M; }
static Matx44d inv44(const Matx44d& T){ Mat M(T),Mi; invert(M,Mi,DECOMP_SVD); return Matx44d(Mi); }
static Matx33d randR(mt19937& g, double s){ Vec3d a{rnd(g,-1,1),rnd(g,-1,1),rnd(g,-1,1)}; double n=norm(a); a/=(n+1e-9); Mat R; Rodrigues(a*rnd(g,-s,s),R); return Matx33d(R); }
static Vec3d applyT(const Matx44d& T, const Vec3d& p){ return Vec3d(
    T(0,0)*p[0]+T(0,1)*p[1]+T(0,2)*p[2]+T(0,3),
    T(1,0)*p[0]+T(1,1)*p[1]+T(1,2)*p[2]+T(1,3),
    T(2,0)*p[0]+T(2,1)*p[1]+T(2,2)*p[2]+T(2,3)); }

// 已知真值标定链：相机在基座系位姿 = inv(G * X)（X=gripper2cam, G=base2gripper）
static Point2f projectPx(const Matx33d& K, const Matx44d& T_cam2base, const Vec3d& pw){
    Vec3d pc = applyT(T_cam2base, pw);
    return Point2f((float)(K(0,0)*pc[0]/pc[2]+K(0,2)), (float)(K(1,1)*pc[1]/pc[2]+K(1,2)));
}
// 相机看到像素 u → 用估计参数反投到平面 z=z_t 的 3D 点（base 系）
static Vec3d backprojPlane(const Matx33d& K, const Matx44d& T_cam2base, const Point2f& u, double zt){
    Vec3d pc_ray((u.x-K(0,2))/K(0,0), (u.y-K(1,2))/K(1,1), 1.0); // K^-1 * [u;1]，相机系射线(z=1)
    Vec3d o = applyT(T_cam2base, Vec3d(0,0,0));
    Matx33d Rc2b(T_cam2base(0,0),T_cam2base(0,1),T_cam2base(0,2),
                 T_cam2base(1,0),T_cam2base(1,1),T_cam2base(1,2),
                 T_cam2base(2,0),T_cam2base(2,1),T_cam2base(2,2));
    Vec3d dir = Rc2b * pc_ray;
    double t = (zt - o[2]) / dir[2];
    return o + t*dir;
}

// 用一组 (K,X,quality) 在 N 个目标上做两类验证，返回 (前向像素误差mm->px, 闭环3D误差)
static void validate(const Matx33d& Ktrue, const Matx44d& Tcb_true,
                     const Matx33d& Kest,  const Matx44d& Tcb_est,
                     const vector<Vec3d>& targets, double zt,
                     double& fwdPx, double& reachMM){
    double se=0, sm=0;
    for(auto& p: targets){
        Point2f u = projectPx(Ktrue, Tcb_true, p);            // 相机实际看到的像素
        Point2f uest = projectPx(Kest, Tcb_est, p);           // 估计参数前向投影
        fwdPx += norm(u-uest);
        Vec3d reach = backprojPlane(Kest, Tcb_est, u, zt);     // 反投到平面
        reachMM += norm(reach - p);
        se=max(se, (double)norm(u-uest)); sm=max(sm, norm(reach-p));
    }
    fwdPx/=targets.size(); reachMM = (reachMM/targets.size())*1000.0;
    (void)se;(void)sm;
}

int main(){
    bool pass=true; mt19937 g(20260817);
    const double zt=0.0;                       // 目标在 z=0 平面
    Matx33d Ktrue(520,0,320, 0,520,240, 0,0,1);
    // 机器人/相机真值位姿：gripper 在 z=-0.5，相机再往前 0.1 → 相机在 z=-0.6 看向 +z
    Matx44d G = matRT(Matx33d::eye(), Vec3d(0,0,-0.5));        // base2gripper
    Matx44d Xtrue = matRT(Matx33d::eye(), Vec3d(0,0,-0.1));    // gripper2cam
    Matx44d Tcb_true = inv44(G * Xtrue);                       // cam2base（真值）

    // 生成 N 个世界目标点（z=0 平面，落在视场内）
    int N=60; vector<Vec3d> targets;
    for(int i=0;i<N;i++) targets.push_back(Vec3d(rnd(g,-0.18,0.18), rnd(g,-0.14,0.14), zt));

    // ---- 好的标定：极小扰动 ----
    Matx44d Xgood = matRT(randR(g,0.2*CV_PI/180.0), Vec3d(0.0003,-0.0002,0.0002)); // rot≈0.2°, t≈0.3mm
    Matx44d Tcb_good = inv44(G * (Xtrue * Xgood));   // 把扰动叠加到真值 X 上
    Matx33d Kgood = Ktrue * 1.003;                   // 焦距偏 0.3%
    double fwdG, reachG; validate(Ktrue,Tcb_true, Kgood,Tcb_good, targets, zt, fwdG, reachG);

    // ---- 差的标定：大扰动 ----
    Matx44d Xbad = matRT(randR(g,3.0*CV_PI/180.0), Vec3d(0.005,-0.003,0.004)); // rot≈3°, t≈5mm
    Matx44d Tcb_bad = inv44(G * (Xtrue * Xbad));
    Matx33d Kbad = Ktrue * 1.05;                     // 焦距偏 5%
    double fwdB, reachB; validate(Ktrue,Tcb_true, Kbad,Tcb_bad, targets, zt, fwdB, reachB);

    cout<<"[L3.1 验证] 好的标定: 前向像素误差="<<fwdG<<"px  闭环够取误差="<<reachG<<"mm\n";
    cout<<"            差的标定: 前向像素误差="<<fwdB<<"px  闭环够取误差="<<reachB<<"mm\n";

    if(fwdG>3.0 || reachG>3.0){pass=false; cout<<"  [失败] 好标定验证误差过大\n";}
    if(!(fwdB > fwdG*2.0) || !(reachB > reachG*2.0)){pass=false; cout<<"  [失败] 验证指标未拉开好/坏标定\n";}

    cout<<"\n=================================================\n";
    cout<<(pass?" [PASS] 端到端验证指标可区分标定质量":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
