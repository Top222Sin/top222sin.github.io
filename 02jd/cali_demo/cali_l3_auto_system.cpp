// cali_l3_auto_system.cpp
// ============================================================================
// L3.3  自动/在线标定系统
// ----------------------------------------------------------------------------
// 把前面所有环节（内参→手眼→TCP→误差校验）包成一个会"自检 + 报警 + 出报告"的
// 系统级模块 CalibSystem：
//   · runCapture(): 生成一组"采集"（机器人各姿态下看到的标定板）
//   · runPipeline(): 依次跑内参/手眼/TCP/误差校验，产出 CalibReport
//   · 质量闸门: 指标越过阈值 → report.needsRecalib=true（在线再标定触发）
//   · 模拟温漂: 给机器人上报位姿加小扰动 → 系统应自动检出并报警
// 全程合成数据，运行即出两份"自动生成的标定报告"。
//
// 编译: g++ -O2 -std=c++17 cali_l3_auto_system.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b){ return uniform_real_distribution<double>(a,b)(g); }
static Matx44d matRT(const Matx33d& R, const Vec3d& t){ Matx44d M=Matx44d::eye(); for(int i=0;i<3;i++){for(int j=0;j<3;j++)M(i,j)=R(i,j);M(i,3)=t[i];} return M; }
static Matx44d inv44(const Matx44d& T){ Mat M(T),Mi; invert(M,Mi,DECOMP_SVD); return Matx44d(Mi); }
static Vec3d getT(const Matx44d& T){ return Vec3d(T(0,3),T(1,3),T(2,3)); }
static Matx33d getR(const Matx44d& T){ return Matx33d(T(0,0),T(0,1),T(0,2),T(1,0),T(1,1),T(1,2),T(2,0),T(2,1),T(2,2)); }
static Matx33d randR(mt19937& g, double s){ Vec3d a{rnd(g,-1,1),rnd(g,-1,1),rnd(g,-1,1)}; double n=norm(a); a/=(n+1e-9); Mat R; Rodrigues(a*rnd(g,-s,s),R); return Matx33d(R); }
static Vec4d quatFromR(const Matx33d& R){ double tr=R(0,0)+R(1,1)+R(2,2); double w,x,y,z;
    if(tr>0){double s=0.5/sqrt(tr+1);w=0.25/s;x=(R(2,1)-R(1,2))*s;y=(R(0,2)-R(2,0))*s;z=(R(1,0)-R(0,1))*s;}
    else if(R(0,0)>R(1,1)&&R(0,0)>R(2,2)){double s=2*sqrt(1+R(0,0)-R(1,1)-R(2,2));x=0.25*s;y=(R(0,1)+R(1,0))/s;z=(R(0,2)+R(2,0))/s;w=(R(2,1)-R(1,2))/s;}
    else if(R(1,1)>R(2,2)){double s=2*sqrt(1+R(1,1)-R(0,0)-R(2,2));y=0.25*s;x=(R(0,1)+R(1,0))/s;z=(R(1,2)+R(2,1))/s;w=(R(0,2)-R(2,0))/s;}
    else {double s=2*sqrt(1+R(2,2)-R(0,0)-R(1,1));z=0.25*s;x=(R(0,2)+R(2,0))/s;y=(R(1,2)+R(2,1))/s;w=(R(1,0)-R(0,1))*s;} return Vec4d(w,x,y,z); }
static Matx33d RfromQuat(const Vec4d& q){ double w=q[0],x=q[1],y=q[2],z=q[3]; return Matx33d(1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w), 2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w), 2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)); }
static Matx44d Sq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,-z,y, y,z,w,-x, z,-y,x,w);}
static Matx44d Wq(const Vec4d& q){double w=q[0],x=q[1],y=q[2],z=q[3];return Matx44d(w,-x,-y,-z, x,w,z,-y, y,-z,w,x, z,y,-x,w);}
static void solveHandEye(const vector<Matx33d>& RA,const vector<Vec3d>& tA,const vector<Matx33d>& RB,const vector<Vec3d>& tB,Matx33d& Rx,Vec3d& tx){
    int k=RA.size(); Mat M(4*k,4,CV_64F);
    for(int i=0;i<k;i++){Vec4d qa=quatFromR(RA[i]),qb=quatFromR(RB[i]);Matx44d b=Sq(qa)-Wq(qb);for(int r=0;r<4;r++)for(int c=0;c<4;c++)M.at<double>(4*i+r,c)=b(r,c);}
    SVD svd(M,SVD::MODIFY_A|SVD::FULL_UV); Vec4d qx(svd.vt.at<double>(3,0),svd.vt.at<double>(3,1),svd.vt.at<double>(3,2),svd.vt.at<double>(3,3)); qx=qx/norm(qx); Rx=RfromQuat(qx);
    Mat NA(3*k,3,CV_64F),nb(3*k,1,CV_64F);for(int i=0;i<k;i++){Matx33d blk=RA[i]-Matx33d::eye();for(int r=0;r<3;r++)for(int c=0;c<3;c++)NA.at<double>(3*i+r,c)=blk(r,c);Vec3d rhs=Rx*tB[i]-tA[i];for(int r=0;r<3;r++)nb.at<double>(3*i+r)=rhs[r];}
    Mat txm; solve(NA,nb,txm,DECOMP_SVD); tx=Vec3d(txm.at<double>(0),txm.at<double>(1),txm.at<double>(2));
}
static Vec3d applyT(const Matx44d& T, const Vec3d& p){ return Vec3d(
    T(0,0)*p[0]+T(0,1)*p[1]+T(0,2)*p[2]+T(0,3),
    T(1,0)*p[0]+T(1,1)*p[1]+T(1,2)*p[2]+T(1,3),
    T(2,0)*p[0]+T(2,1)*p[1]+T(2,2)*p[2]+T(2,3)); }

struct CalibReport {
    double intriRMS=0, heRotDeg=0, heTrMm=0, tcpMM=0;
    bool ok=false, needsRecalib=false;
    string text() const {
        ostringstream os; os<<"  内参RMS="<<intriRMS<<"px  手眼旋转="<<heRotDeg<<"° 平移="<<heTrMm<<"mm  TCP="<<tcpMM<<"mm\n";
        os<<"  质量闸门: "<<(ok?"PASS":"FAIL")<<"   再标定触发: "<<(needsRecalib?"是 ⚠":"否")<<"\n"; return os.str();
    }
};

// 一组采集：机器人 M 个姿态下看到的标定板
struct Capture { vector<Matx44d> Gpipe; vector<vector<Point3f>> obj; vector<vector<Point2f>> det; };

static Capture makeCapture(mt19937& g, const Matx44d& Xgt, const Matx33d& K, const Vec4d& d,
                          double noisePx, double driftDeg, int M){
    const int cn=9,rn=6; const double sq=0.025; Size img(640,480);
    Mat Km(K), Dm(4,1,CV_64F,{d[0],d[1],d[2],d[3]});
    Capture cap; vector<Matx44d> Gtrue;
    for(int i=0;i<M;i++){ Matx44d G=matRT(randR(g,0.6),Vec3d(rnd(g,-0.3,0.3),rnd(g,-0.3,0.3),rnd(g,0.4,0.9))); Gtrue.push_back(G);
        // 管线拿到的位姿可能含温漂：叠加小旋转
        Matx44d Gd=matRT(randR(g,driftDeg*CV_PI/180.0),Vec3d(0,0,0)); cap.Gpipe.push_back(G*Gd); }
    for(int i=0;i<M;i++){
        vector<Point3f> o; for(int r=0;r<rn;r++)for(int c=0;c<cn;c++)o.push_back(Point3f(c*sq,r*sq,0)); cap.obj.push_back(o);
        Matx44d Tcb=inv44(Gtrue[i]*Xgt);                      // 真值相机位姿
        vector<Point2f> det; projectPoints(o,Mat(getR(Tcb)),getT(Tcb),Km,Dm,det);
        normal_distribution<double> pn(0,noisePx); for(auto&p:det){p.x+=pn(g);p.y+=pn(g);}
        cap.det.push_back(det);
    }
    return cap;
}

static CalibReport runPipeline(const Capture& cap, const Matx44d& Xgt, const Matx33d& Kgt, const Vec3d& xgt){
    int M=(int)cap.Gpipe.size(); Size img(640,480);
    // ① 内参
    Mat Kc,Dc; vector<Mat> rvecs,tvecs; double rms=calibrateCamera(cap.obj,cap.det,img,Kc,Dc,rvecs,tvecs);
    // ② 手眼：每帧 solvePnP 得 T_target2cam，配对相对运动
    vector<Matx33d> RA,RB; vector<Vec3d> tA,tB;
    vector<Matx44d> T; Mat Kck(Kc);
    for(int i=0;i<M;i++){ Mat r,t; solvePnP(cap.obj[i],cap.det[i],Kck,Dc,r,t);
        Matx33d R; Rodrigues(r,R); T.push_back(matRT(R,Vec3d(t.at<double>(0),t.at<double>(1),t.at<double>(2)))); }
    for(int i=0;i<M-1;i++){ Matx44d Ai=inv44(cap.Gpipe[i])*cap.Gpipe[i+1], Bi=T[i]*inv44(T[i+1]);
        RA.push_back(getR(Ai)); RB.push_back(getR(Bi)); tA.push_back(getT(Ai)); tB.push_back(getT(Bi)); }
    Matx33d Rx; Vec3d tx; solveHandEye(RA,tA,RB,tB,Rx,tx); Matx44d Xest=matRT(Rx,tx);
    Matx44d Eg=Xgt.inv()*Xest; Vec3d rvg; Rodrigues(Mat(getR(Eg)),rvg);
    double heRot=norm(rvg)*180/CV_PI, heTr=norm(getT(Eg))*1000;
    // ③ TCP：四点法（用管线位姿）
    Vec3d pworld(0.30,0,0.45); int Nt=M; vector<Matx33d> Rt; vector<Vec3d> tt;
    for(int i=0;i<Nt;i++){ Matx33d Ri=getR(cap.Gpipe[i]); Rt.push_back(Ri); tt.push_back(pworld-Ri*xgt); }
    int m=3*(Nt-1); Mat A(m,3,CV_64F),b(m,1,CV_64F); int row=0;
    for(int i=1;i<Nt;i++){ Matx33d dR=Rt[i]-Rt[0]; Vec3d db=tt[0]-tt[i]; for(int r=0;r<3;r++){A.at<double>(row,0)=dR(r,0);A.at<double>(row,1)=dR(r,1);A.at<double>(row,2)=dR(r,2);b.at<double>(row)=db[r];row++;} }
    Mat xm; solve(A,b,xm,DECOMP_SVD); Vec3d xest(xm.at<double>(0),xm.at<double>(1),xm.at<double>(2));
    double tcp=0; for(int i=0;i<Nt;i++) tcp=max(tcp, norm(Rt[i]*xest+tt[i]-(Rt[0]*xest+tt[0]))); tcp*=1000;
    // ④ 质量闸门
    CalibReport rep; rep.intriRMS=rms; rep.heRotDeg=heRot; rep.heTrMm=heTr; rep.tcpMM=tcp;
    rep.ok = (rms<2.0 && heRot<0.5 && heTr<1.0 && tcp<1.0);
    rep.needsRecalib = !(heRot<0.5 && heTr<1.0 && tcp<1.0);   // 外参/TCP 超阈 → 触发再标定
    return rep;
}

int main(){
    mt19937 g(20260817);
    Matx33d Kgt(520,0,320, 0,520,240, 0,0,1); Vec4d dgt(0.09,-0.035,0.001,0.0006);
    Matx44d Xgt=matRT(randR(g,0.5),Vec3d(0.04,-0.02,0.10)); Vec3d xgt(0.04,0,0.12);
    const int M=16;

    // 场景1：健康采集（无漂移）
    Capture cap1=makeCapture(g,Xgt,Kgt,dgt,0.5,0.0,M);
    CalibReport r1=runPipeline(cap1,Xgt,Kgt,xgt);
    // 场景2：温漂 → 机器人上报位姿带 1.5° 扰动
    Capture cap2=makeCapture(g,Xgt,Kgt,dgt,0.5,1.5,M);
    CalibReport r2=runPipeline(cap2,Xgt,Kgt,xgt);

    cout<<"========== [自动标定系统] 报告 ==========\n";
    cout<<"[场景1 健康采集]\n"<<r1.text();
    cout<<"[场景2 温漂采集]\n"<<r2.text();
    cout<<"========================================\n";

    bool pass = r1.ok && !r1.needsRecalib && r2.needsRecalib;
    if(!r1.ok){cout<<"  [失败] 健康场景未通过质量闸门\n";}
    if(!r2.needsRecalib){cout<<"  [失败] 温漂场景未触发再标定\n";}
    cout<<(pass?" [PASS] 系统自检 + 温漂自动告警 正常":" [FAIL] 见上")<<"\n";
    return pass?0:1;
}
