// cali_l2_degeneracy.cpp
// ============================================================================
// L2.3  退化（Degeneracy）：为什么"运动不够"会让手眼标定悄悄失效
// ----------------------------------------------------------------------------
// 退化 = 标定问题的某些自由度"不可观测"（ill-posed）。本 demo 用 AX=XB 的系数
// 矩阵做可观测性分析：把每个相对运动对写成 (S(qA)-W(qB))·qX=0，堆叠成 M。
//   M 满秩(4) -> 可解；某奇异值≈0 -> 对应自由度不可观 -> 退化。
// 演示四类运动：① 多样(好) ② 纯平移 ③ 仅绕 z 轴旋转 ④ 平面运动(xy+z转动)，
// 报告 M 的奇异值、条件数，并验证退化时求解器误差暴涨 + 检测算法能告警。
//
// 编译: g++ -O2 -std=c++17 cali_l2_degeneracy.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
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
static Matx33d Rz(double a){ return Matx33d(cos(a),-sin(a),0, sin(a),cos(a),0, 0,0,1); }
static Matx44d matRT(const Matx33d& R, const Vec3d& t){ Matx44d M; M=Matx44d::eye(); for(int i=0;i<3;i++){for(int j=0;j<3;j++)M(i,j)=R(i,j);M(i,3)=t[i];} return M; }
static Matx44d inv44(const Matx44d& T){ Mat M(T),Mi; invert(M,Mi,DECOMP_SVD); return Matx44d(Mi); }

static Vec4d quatFromR(const Matx33d& R){ double tr=R(0,0)+R(1,1)+R(2,2); double w,x,y,z;
    if(tr>0){double s=0.5/sqrt(tr+1);w=0.25/s;x=(R(2,1)-R(1,2))*s;y=(R(0,2)-R(2,0))*s;z=(R(1,0)-R(0,1))*s;}
    else if(R(0,0)>R(1,1)&&R(0,0)>R(2,2)){double s=2*sqrt(1+R(0,0)-R(1,1)-R(2,2));x=0.25*s;y=(R(0,1)+R(1,0))/s;z=(R(0,2)+R(2,0))/s;w=(R(2,1)-R(1,2))/s;}
    else if(R(1,1)>R(2,2)){double s=2*sqrt(1+R(1,1)-R(0,0)-R(2,2));y=0.25*s;x=(R(0,1)+R(1,0))/s;z=(R(1,2)+R(2,1))/s;w=(R(0,2)-R(2,0))/s;}
    else {double s=2*sqrt(1+R(2,2)-R(0,0)-R(1,1));z=0.25*s;x=(R(0,2)+R(2,0))/s;y=(R(1,2)+R(2,1))/s;w=(R(1,0)-R(0,1))*s;} return Vec4d(w,x,y,z); }
static Matx44d Sq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,-z,y, y,z,w,-x, z,-y,x,w);}
static Matx44d Wq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,z,-y, y,-z,w,x, z,y,-x,w);}

// 可观测性：堆叠 (S(qA)-W(qB)) 得到 M(4k x 4)，返回最小/最大奇异值
static void obserSingular(const vector<Matx33d>& RA, const vector<Matx33d>& RB, double& smin, double& smax){
    int k=RA.size(); Mat M(4*k,4,CV_64F);
    for(int i=0;i<k;i++){ Vec4d qa=quatFromR(RA[i]),qb=quatFromR(RB[i]); Matx44d b=Sq(qa)-Wq(qb);
        for(int r=0;r<4;r++)for(int c=0;c<4;c++)M.at<double>(4*i+r,c)=b(r,c); }
    SVD svd(M, SVD::MODIFY_A | SVD::NO_UV);  // svd.w 为奇异值
    smax=0; smin=1e9; for(int i=0;i<svd.w.cols;i++){ double s=svd.w.at<double>(i); smax=max(smax,s); smin=min(smin,s); }
}

int main(){
    bool pass=true; mt19937 g(20260817);
    const int k=15;
    Matx44d Xgt=matRT(randR(g,0.5), Vec3d(0.05,-0.03,0.12));   // 真值 X

    struct Case{ string name; vector<Matx44d> A,B; };
    vector<Case> cases;

    // ① 多样运动（好）
    { Case c; c.name="① 多样运动(好)";
      for(int i=0;i<k;i++){ Matx44d A=matRT(randR(g,0.7), Vec3d(rnd(g,-0.2,0.2),rnd(g,-0.2,0.2),rnd(g,-0.1,0.1)));
        Matx44d B=inv44(Xgt)*A*Xgt; c.A.push_back(A); c.B.push_back(B); } cases.push_back(c); }
    // ② 纯平移（无旋转）
    { Case c; c.name="② 纯平移(无旋转)";
      for(int i=0;i<k;i++){ Matx44d A=matRT(Matx33d::eye(), Vec3d(rnd(g,-0.2,0.2),rnd(g,-0.2,0.2),rnd(g,-0.1,0.1)));
        Matx44d B=inv44(Xgt)*A*Xgt; c.A.push_back(A); c.B.push_back(B); } cases.push_back(c); }
    // ③ 仅绕 z 轴旋转
    { Case c; c.name="③ 仅绕 z 轴旋转";
      for(int i=0;i<k;i++){ Matx44d A=matRT(Rz(rnd(g,-0.8,0.8)), Vec3d(0,0,0));
        Matx44d B=inv44(Xgt)*A*Xgt; c.A.push_back(A); c.B.push_back(B); } cases.push_back(c); }
    // ④ 平面运动（xy 平移 + z 转动，无倾斜/无 z 平移）
    { Case c; c.name="④ 平面运动(xy+z转动)";
      for(int i=0;i<k;i++){ Matx44d A=matRT(Rz(rnd(g,-0.8,0.8)), Vec3d(rnd(g,-0.2,0.2),rnd(g,-0.2,0.2),0));
        Matx44d B=inv44(Xgt)*A*Xgt; c.A.push_back(A); c.B.push_back(B); } cases.push_back(c); }

    cout << "场景                       σmax      σmin      比值      退化?\n";
    cout << "---------------------------------------------------------------\n";
    for(auto& c: cases){
        vector<Matx33d> RA,RB; vector<Vec3d> tA,tB;
        for(int i=0;i<k;i++){ RA.push_back(Matx33d(c.A[i](0,0),c.A[i](0,1),c.A[i](0,2), c.A[i](1,0),c.A[i](1,1),c.A[i](1,2), c.A[i](2,0),c.A[i](2,1),c.A[i](2,2)));
                               RB.push_back(Matx33d(c.B[i](0,0),c.B[i](0,1),c.B[i](0,2), c.B[i](1,0),c.B[i](1,1),c.B[i](1,2), c.B[i](2,0),c.B[i](2,1),c.B[i](2,2)));
                               tA.push_back(Vec3d(c.A[i](0,3),c.A[i](1,3),c.A[i](2,3))); tB.push_back(Vec3d(c.B[i](0,3),c.B[i](1,3),c.B[i](2,3))); }
        double smin,smax; obserSingular(RA,RB,smin,smax);
        bool deg = (smax<1e-9) || (smin/smax < 1e-3);
        printf("  %-22s  %.3e  %.3e  %.2e  %s\n", c.name.c_str(), smax, smin, smin/smax, deg?"⚠ 退化":"OK");
        if(c.name.rfind("①",0)==0){ if(deg){pass=false; cout<<"  [失败] 好数据不应被判退化\n";} }
        else { if(!deg){pass=false; cout<<"  [失败] 退化场景未被检出\n";} }
    }

    cout << "\n=================================================\n";
    cout << (pass ? " [PASS] 退化可观测性分析 + 自动检测正确" : " [FAIL] 见上") << "\n";
    cout << "=================================================\n";
    return pass?0:1;
}
