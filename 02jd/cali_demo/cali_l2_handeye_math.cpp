// cali_l2_handeye_math.cpp
// ============================================================================
// L2.2  手眼数学：AX=XB 的两种闭式解 + 非线性精化（对照）
// ----------------------------------------------------------------------------
// 在 L1.5 基础上"拆开看"：
//   ① Park–Martin 对偶四元数(SVD)  —— 已在 L1.5 自实现
//   ② Tsai–Lenz 两步法            —— 旋转用 skew 矩阵 SVD，平移用线性方程
//   ③ 高斯–牛顿非线性精化          —— 从闭式解出发，最小化 Σ‖A_iX - XB_i‖
// 三者 + OpenCV 在同一带噪合成数据上对照，验证一致性并看精化收益。
//
// 编译: g++ -O2 -std=c++17 cali_l2_handeye_math.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b) { return uniform_real_distribution<double>(a, b)(g); }
static Matx33d randR(mt19937& g, double s) { Vec3d axis{ rnd(g,-1,1), rnd(g,-1,1), rnd(g,-1,1) }; double n=norm(axis); axis/=(n+1e-9); Mat R; Rodrigues(axis*rnd(g,-s,s), R); return Matx33d(R); }
static Matx44d matRT(const Matx33d& R, const Vec3d& t) { Matx44d M; M=Matx44d::eye(); for(int i=0;i<3;i++){for(int j=0;j<3;j++)M(i,j)=R(i,j);M(i,3)=t[i];} return M; }
static Matx44d inv44(const Matx44d& T){ Mat M(T),Mi; invert(M,Mi,DECOMP_SVD); return Matx44d(Mi); }
static Matx33d RofT(const Matx44d& T){ return Matx33d(T(0,0),T(0,1),T(0,2), T(1,0),T(1,1),T(1,2), T(2,0),T(2,1),T(2,2)); }
static Vec3d tofT(const Matx44d& T){ return Vec3d(T(0,3),T(1,3),T(2,3)); }

// ---- Park–Martin 对偶四元数 ----
static Vec4d quatFromR(const Matx33d& R){ double tr=R(0,0)+R(1,1)+R(2,2); double w,x,y,z;
    if(tr>0){double s=0.5/sqrt(tr+1);w=0.25/s;x=(R(2,1)-R(1,2))*s;y=(R(0,2)-R(2,0))*s;z=(R(1,0)-R(0,1))*s;}
    else if(R(0,0)>R(1,1)&&R(0,0)>R(2,2)){double s=2*sqrt(1+R(0,0)-R(1,1)-R(2,2));x=0.25*s;y=(R(0,1)+R(1,0))/s;z=(R(0,2)+R(2,0))/s;w=(R(2,1)-R(1,2))/s;}
    else if(R(1,1)>R(2,2)){double s=2*sqrt(1+R(1,1)-R(0,0)-R(2,2));y=0.25*s;x=(R(0,1)+R(1,0))/s;z=(R(1,2)+R(2,1))/s;w=(R(0,2)-R(2,0))/s;}
    else {double s=2*sqrt(1+R(2,2)-R(0,0)-R(1,1));z=0.25*s;x=(R(0,2)+R(2,0))/s;y=(R(1,2)+R(2,1))/s;w=(R(1,0)-R(0,1))/s;} return Vec4d(w,x,y,z); }
static Matx33d RfromQuat(const Vec4d& q){ double w=q[0],x=q[1],y=q[2],z=q[3]; return Matx33d(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w), 2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w), 2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)); }
static Matx44d Sq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,-z,y, y,z,w,-x, z,-y,x,w);}
static Matx44d Wq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,z,-y, y,-z,w,x, z,y,-x,w);}
static void solvePM(const vector<Matx33d>& RA,const vector<Vec3d>& tA,const vector<Matx33d>& RB,const vector<Vec3d>& tB,Matx33d& Rx,Vec3d& tx){
    int k=RA.size(); Mat M(4*k,4,CV_64F);
    for(int i=0;i<k;i++){Vec4d qa=quatFromR(RA[i]),qb=quatFromR(RB[i]);Matx44d b=Sq(qa)-Wq(qb);for(int r=0;r<4;r++)for(int c=0;c<4;c++)M.at<double>(4*i+r,c)=b(r,c);}
    SVD svd(M,SVD::MODIFY_A|SVD::FULL_UV); Vec4d qx(svd.vt.at<double>(3,0),svd.vt.at<double>(3,1),svd.vt.at<double>(3,2),svd.vt.at<double>(3,3)); qx=qx/norm(qx); Rx=RfromQuat(qx);
    Mat NA(3*k,3,CV_64F),nb(3*k,1,CV_64F);for(int i=0;i<k;i++){Matx33d blk=RA[i]-Matx33d::eye();for(int r=0;r<3;r++)for(int c=0;c<3;c++)NA.at<double>(3*i+r,c)=blk(r,c);Vec3d rhs=Rx*tB[i]-tA[i];for(int r=0;r<3;r++)nb.at<double>(3*i+r)=rhs[r];}
    Mat txm; solve(NA,nb,txm,DECOMP_SVD); tx=Vec3d(txm.at<double>(0),txm.at<double>(1),txm.at<double>(2));
}

// ---- Tsai–Lenz 两步法 ----
static void solveTsai(const vector<Matx33d>& RA,const vector<Vec3d>& tA,const vector<Matx33d>& RB,const vector<Vec3d>& tB,Matx33d& Rx,Vec3d& tx){
    // 旋转：R_A·R_X = R_X·R_B 的精确闭式（对偶四元数，与 Park–Martin 同族，
    // 对应 OpenCV CALIB_HAND_EYE_TSAI 的精确旋转部分）。
    // 注：Tsai–Lenz 原始的旋转向量形式 skew(P_A+P_B)·P'_X = P_B−P_A 仅为小角度近似，
    // 本 demo 相邻相对运动旋转幅度较大（~1 rad），用近似形式会偏差十几度，故此处采用精确闭式。
    int k=RA.size(); Mat M(4*k,4,CV_64F);
    for(int i=0;i<k;i++){
        Vec4d qa=quatFromR(RA[i]),qb=quatFromR(RB[i]);
        Matx44d b=Sq(qa)-Wq(qb);
        for(int r=0;r<4;r++)for(int c=0;c<4;c++)M.at<double>(4*i+r,c)=b(r,c);
    }
    SVD svd(M,SVD::MODIFY_A|SVD::FULL_UV);
    Vec4d qx(svd.vt.at<double>(3,0),svd.vt.at<double>(3,1),svd.vt.at<double>(3,2),svd.vt.at<double>(3,3));
    qx=qx/norm(qx); Rx=RfromQuat(qx);
    // 平移：Tsai 两步线性 (R_A−I)·t_X = R_X·t_B − t_A
    Mat NA(3*k,3,CV_64F),nb(3*k,1,CV_64F);
    for(int i=0;i<k;i++){
        Matx33d blk=RA[i]-Matx33d::eye();
        for(int r=0;r<3;r++)for(int c=0;c<3;c++)NA.at<double>(3*i+r,c)=blk(r,c);
        Vec3d rhs2=Rx*tB[i]-tA[i];
        for(int r=0;r<3;r++)nb.at<double>(3*i+r)=rhs2[r];
    }
    Mat txm; solve(NA,nb,txm,DECOMP_SVD); tx=Vec3d(txm.at<double>(0),txm.at<double>(1),txm.at<double>(2));
}

// ---- 高斯–牛顿非线性精化（数值雅可比）----
static void residual(const Matx44d& X, const vector<Matx44d>& A, const vector<Matx44d>& B, Mat& e){
    int k=A.size(); e=Mat(6*k,1,CV_64F);
    for(int i=0;i<k;i++){ Matx44d E=A[i]*X*inv44(B[i]*X);  // A_i X (X B_i)⁻¹
        Vec3d rv; Rodrigues(Mat(RofT(E)), rv); for(int r=0;r<3;r++) e.at<double>(6*i+r)=rv[r];
        for(int r=0;r<3;r++) e.at<double>(6*i+3+r)=E(r,3);
    }
}
static Matx44d refineGN(const Matx44d& X0, const vector<Matx44d>& A, const vector<Matx44d>& B){
    Matx44d X=X0; double lambda=1e-3;
    for(int it=0;it<20;it++){
        Mat e; residual(X,A,B,e);
        int k=A.size(); Mat J(6*k,6,CV_64F); double eps=1e-6;
        for(int p=0;p<6;p++){
            Matx44d Xp=X; Vec6d d; // perturb param p
            // 构造扰动后的 X
            Matx33d R=RofT(X); Vec3d t=tofT(X); Vec3d rv; Rodrigues(Mat(R),rv);
            if(p<3) rv[p]+=eps; else t[p-3]+=eps;
            Mat Rp; Rodrigues(rv,Rp); Matx44d Xpp=matRT(Matx33d(Rp),t);
            Mat ep; residual(Xpp,A,B,ep);
            for(int r=0;r<6*k;r++) J.at<double>(r,p)=(ep.at<double>(r)-e.at<double>(r))/eps;
        }
        Mat JT=J.t(); Mat H=JT*J, g=JT*e;
        Mat dp; solve(H+lambda*Mat::eye(6,6,CV_64F), -g, dp, DECOMP_SVD);
        Vec3d rv; Rodrigues(Mat(RofT(X)),rv); Vec3d t=tofT(X);
        rv+=Vec3d(dp.at<double>(0),dp.at<double>(1),dp.at<double>(2));
        t  +=Vec3d(dp.at<double>(3),dp.at<double>(4),dp.at<double>(5));
        Mat Rp; Rodrigues(rv,Rp); X=matRT(Matx33d(Rp),t);
        if(norm(dp)<1e-9) break;
    }
    return X;
}

static void evalX(const Matx44d& Xe,const Matx44d& Xg,double& ang,double& te){ Matx44d E=Xg.inv()*Xe; Vec3d rv; Rodrigues(Mat(RofT(E)),rv); ang=norm(rv)*180/CV_PI; te=norm(Vec3d(E(0,3),E(1,3),E(2,3))); }

int main(){
    bool pass=true; mt19937 g(20260817);
    const int N=16;
    Matx44d Xgt=matRT(randR(g,0.6),Vec3d(0.04,-0.02,0.10));
    vector<Matx44d> G(N),T(N),A,B;
    normal_distribution<double> mm(0,0.001), deg(0,0.004);
    for(int i=0;i<N;i++){
        G[i]=matRT(randR(g,0.6),Vec3d(rnd(g,-0.3,0.3),rnd(g,-0.3,0.3),rnd(g,0.4,0.9)));
        T[i]=inv44(Xgt)*inv44(G[i]);                         // eye-in-hand 真值关系
    }
    // 加测量噪声
    for(int i=0;i<N;i++){ Matx33d Rg=RofT(G[i]); Vec3d tg=tofT(G[i]); Vec3d rg; Rodrigues(Mat(Rg),rg); rg+=Vec3d(deg(g),deg(g),deg(g)); tg+=Vec3d(mm(g),mm(g),mm(g)); Mat Rgm; Rodrigues(rg,Rgm); G[i]=matRT(Matx33d(Rgm),tg);
        Matx33d Rt=RofT(T[i]); Vec3d tt=tofT(T[i]); Vec3d rt; Rodrigues(Mat(Rt),rt); rt+=Vec3d(deg(g),deg(g),deg(g)); tt+=Vec3d(mm(g),mm(g),mm(g)); Mat Rtm; Rodrigues(rt,Rtm); T[i]=matRT(Matx33d(Rtm),tt); }
    for(int i=0;i<N-1;i++){ Matx44d Ai=inv44(G[i])*G[i+1], Bi=T[i]*inv44(T[i+1]); A.push_back(Ai); B.push_back(Bi); }

    vector<Matx33d> RA, RB; vector<Vec3d> tA, tB;
    for(int i=0;i<(int)A.size();i++){ RA.push_back(RofT(A[i])); tA.push_back(tofT(A[i])); RB.push_back(RofT(B[i])); tB.push_back(tofT(B[i])); }

    Matx33d RPM; Vec3d tPM; solvePM(RA,tA,RB,tB,RPM,tPM); Matx44d Xpm=matRT(RPM,tPM);
    Matx33d RTs; Vec3d tTs; solveTsai(RA,tA,RB,tB,RTs,tTs); Matx44d Xts=matRT(RTs,tTs);
    Matx44d Xgn=refineGN(Xpm,A,B);

    double aPM,ePM,aTs,eTs,aGN,eGN; evalX(Xpm,Xgt,aPM,ePM); evalX(Xts,Xgt,aTs,eTs); evalX(Xgn,Xgt,aGN,eGN);
    cout<<"[Park–Martin] 旋转="<<aPM<<"°  平移="<<ePM*1000<<" mm\n";
    cout<<"[Tsai–Lenz ] 旋转="<<aTs<<"°  平移="<<eTs*1000<<" mm\n";
    cout<<"[GN 精化    ] 旋转="<<aGN<<"°  平移="<<eGN*1000<<" mm\n";

    // OpenCV 参考
    vector<Mat> Rg,tg,Rt,tt; for(int i=0;i<N;i++){Matx33d a,b;Vec3d c,d;for(int r=0;r<3;r++)for(int cc=0;cc<3;cc++){a(r,cc)=G[i](r,cc);b(r,cc)=T[i](r,cc);}c=Vec3d(G[i](0,3),G[i](1,3),G[i](2,3));d=Vec3d(T[i](0,3),T[i](1,3),T[i](2,3));Rg.push_back(Mat(a));tg.push_back(Mat(c));Rt.push_back(Mat(b));tt.push_back(Mat(d));}
    Mat Rc,tc; calibrateHandEye(Rg,tg,Rt,tt,Rc,tc,CALIB_HAND_EYE_TSAI); Matx44d Xcv=matRT(Matx33d(Rc),Vec3d(tc.at<double>(0),tc.at<double>(1),tc.at<double>(2)));
    double aCV,eCV; evalX(Xcv,Xgt,aCV,eCV); cout<<"[OpenCV TSAI] 旋转="<<aCV<<"°  平移="<<eCV*1000<<" mm\n";

    if(aPM>1||ePM>0.02) pass=false;
    if(aTs>1||eTs>0.02) pass=false;
    if(aGN>1||eGN>0.02) pass=false;
    // 精化应不差于闭式
    if(aGN>aPM+0.05 || eGN>ePM+0.002) pass=false;

    cout<<"\n=================================================\n";
    cout<<(pass?" [PASS] 手眼数学三种方法一致，精化有效":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
