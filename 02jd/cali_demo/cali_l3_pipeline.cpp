// cali_l3_pipeline.cpp
// ============================================================================
// L3.4  全流程工具链（capstone）
// ----------------------------------------------------------------------------
// 把整个标定链路串成一条龙，并内置"退化检测"闸门：
//   角点检测(合成) → 内参 → 手眼(AX=XB + 退化条件数) → TCP → 误差校验 → 验证
// 跑两种采集：
//   ① 多样运动（好）：全流程 PASS，端到端投影误差小
//   ② 纯平移运动（退化注入）：退化检测器在"手眼"前就告警并拒绝，链路 FAIL
// 全程合成数据，运行即出完整报告 + PASS/FAIL。
//
// 编译: g++ -O2 -std=c++17 cali_l3_pipeline.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
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
// 退化条件数：堆叠 M=(S(qA)-W(qB))，SVD 取 σmax/σmin；越大越退化
static double degCondition(const vector<Matx33d>& RA,const vector<Vec3d>& tA,const vector<Matx33d>& RB,const vector<Vec3d>& tB){
    int k=RA.size(); Mat M(4*k,4,CV_64F);
    for(int i=0;i<k;i++){Vec4d qa=quatFromR(RA[i]),qb=quatFromR(RB[i]);Matx44d b=Sq(qa)-Wq(qb);for(int r=0;r<4;r++)for(int c=0;c<4;c++)M.at<double>(4*i+r,c)=b(r,c);}
    SVD svd(M); return svd.w.at<double>(0)/max(svd.w.at<double>(3),1e-9);
}

// 投影已知 3D 点到像素（给定 T_cam2base 与 K）
static Vec3d applyTtrue(const Matx44d& T, const Vec3d& p, const Matx33d& K){
    Vec3d pc=getR(T)*p+getT(T);
    return Vec3d(K(0,0)*pc[0]/pc[2]+K(0,2), K(1,1)*pc[1]/pc[2]+K(1,2), 0);
}

struct Report { double rms=0,heRot=0,heTr=0,tcp=0,fwd=0; bool deg=false,pass=false; string msg; };

static Report run(const vector<Matx44d>& G, const vector<vector<Point3f>>& obj, const vector<vector<Point2f>>& det,
                  const Matx44d& Xgt, const Matx33d& Kgt, const Vec3d& xgt){
    int M=(int)G.size(); Size img(640,480); Report rep;
    // 内参
    Mat Kc,Dc; vector<Mat> rv,tv; double rms=calibrateCamera(obj,det,img,Kc,Dc,rv,tv); rep.rms=rms;
    Mat Kck(Kc);
    // 手眼：逐帧 solvePnP → T_target2cam，配对相对运动
    vector<Matx44d> T;
    for(int i=0;i<M;i++){ Mat r,t; solvePnP(obj[i],det[i],Kck,Dc,r,t); Matx33d R; Rodrigues(r,R);
        T.push_back(matRT(R,Vec3d(t.at<double>(0),t.at<double>(1),t.at<double>(2)))); }
    vector<Matx33d> RA,RB; vector<Vec3d> tA,tB;
    for(int i=0;i<M-1;i++){ Matx44d Ai=inv44(G[i])*G[i+1], Bi=T[i]*inv44(T[i+1]);
        RA.push_back(getR(Ai)); RB.push_back(getR(Bi)); tA.push_back(getT(Ai)); tB.push_back(getT(Bi)); }
    // 退化闸门（手眼前先查）：条件数爆炸 或 矩阵整体坍塌(纯平移→奇异值全0) 均判退化
    double cond=degCondition(RA,tA,RB,tB);
    if(cond>1e3 || cond<1e-4){ rep.deg=true; rep.msg="⚠ 退化检测：相对运动条件数="+to_string(cond)+" → 手眼不可靠，链路终止"; rep.pass=false; return rep; }
    // 手眼求解
    Matx33d Rx; Vec3d tx; solveHandEye(RA,tA,RB,tB,Rx,tx); Matx44d Xest=matRT(Rx,tx);
    Matx44d Eg=Xgt.inv()*Xest; Vec3d rvg; Rodrigues(Mat(getR(Eg)),rvg);
    rep.heRot=norm(rvg)*180/CV_PI; rep.heTr=norm(getT(Eg))*1000;
    // TCP 四点法
    Vec3d pworld(0.30,0,0.45); vector<Matx33d> Rt; vector<Vec3d> tt;
    for(int i=0;i<M;i++){ Matx33d Ri=getR(G[i]); Rt.push_back(Ri); tt.push_back(pworld-Ri*xgt); }
    int m=3*(M-1); Mat A(m,3,CV_64F),b(m,1,CV_64F); int row=0;
    for(int i=1;i<M;i++){ Matx33d dR=Rt[i]-Rt[0]; Vec3d db=tt[0]-tt[i]; for(int r=0;r<3;r++){A.at<double>(row,0)=dR(r,0);A.at<double>(row,1)=dR(r,1);A.at<double>(row,2)=dR(r,2);b.at<double>(row)=db[r];row++;} }
    Mat xm; solve(A,b,xm,DECOMP_SVD); Vec3d xest(xm.at<double>(0),xm.at<double>(1),xm.at<double>(2));
    rep.tcp=0; for(int i=0;i<M;i++) rep.tcp=max(rep.tcp, norm(Rt[i]*xest+tt[i]-(Rt[0]*xest+tt[0]))); rep.tcp*=1000;
    // 端到端验证：已知 3D 点前向投影误差
    Vec3d P0(0.10,0.10,0.0); double fe=0;
    for(int i=0;i<M;i++){
        Matx44d Tcb_true=inv44(G[i]*Xgt), Tcb_est=inv44(G[i]*Xest);
        Vec3d pct=applyTtrue(Tcb_true,P0,Kgt), pce=applyTtrue(Tcb_est,P0,Matx33d(Kc));
        fe+=norm(pct-pce);
    }
    rep.fwd=fe/M;
    rep.pass = (rms<2.0 && rep.heRot<0.5 && rep.heTr<1.0 && rep.tcp<1.0 && rep.fwd<2.0);
    rep.msg = rep.pass? "全流程 PASS" : "指标超阈 FAIL";
    return rep;
}

int main(){
    bool pass=true; mt19937 g(20260817);
    const int cn=9,rn=6; const double sq=0.025; Size img(640,480);
    Matx33d Kgt(520,0,320, 0,520,240, 0,0,1); Vec4d dgt(0.09,-0.035,0.001,0.0006);
    Mat Km(Kgt), Dm(4,1,CV_64F,{dgt[0],dgt[1],dgt[2],dgt[3]});
    Matx44d Xgt=matRT(randR(g,0.5),Vec3d(0.04,-0.02,0.10)); Vec3d xgt(0.04,0,0.12);
    const int M=16;

    // ---- 采集生成 ----
    auto makeCap=[&](bool pureTrans)->tuple<vector<Matx44d>,vector<vector<Point3f>>,vector<vector<Point2f>>>{
        vector<Matx44d> G; vector<vector<Point3f>> obj; vector<vector<Point2f>> det;
        Matx33d R0 = pureTrans? randR(g,0.0) : randR(g,0.6);   // 纯平移时所有帧同旋转
        for(int i=0;i<M;i++){
            Matx33d Ri = pureTrans? R0 : randR(g,0.6);
            Vec3d ti(rnd(g,-0.3,0.3),rnd(g,-0.3,0.3),rnd(g,0.4,0.9));
            G.push_back(matRT(Ri,ti));
            vector<Point3f> o; for(int r=0;r<rn;r++)for(int c=0;c<cn;c++)o.push_back(Point3f(c*sq,r*sq,0)); obj.push_back(o);
            Matx44d Tcb=inv44(G.back()*Xgt);
            vector<Point2f> d; projectPoints(o,Mat(getR(Tcb)),getT(Tcb),Km,Dm,d);
            normal_distribution<double> pn(0,0.5); for(auto&p:d){p.x+=pn(g);p.y+=pn(g);} det.push_back(d);
        }
        return {G,obj,det};
    };

    auto [GA,objA,detA]=makeCap(false);
    auto [GB,objB,detB]=makeCap(true);

    Report rA=run(GA,objA,detA,Xgt,Kgt,xgt);
    Report rB=run(GB,objB,detB,Xgt,Kgt,xgt);

    cout<<"====== [L3.4 全流程工具链] 报告 ======\n";
    cout<<"[① 多样运动] "<<rA.msg<<"\n";
    cout<<"  内参RMS="<<rA.rms<<"px 手眼="<<rA.heRot<<"°/"<<rA.heTr<<"mm TCP="<<rA.tcp<<"mm 前向验证="<<rA.fwd<<"px\n";
    cout<<"[② 纯平移(退化注入)] "<<rB.msg<<"\n";
    if(rB.deg) cout<<"  → 退化闸门在求解前拦下，链路安全终止\n";
    cout<<"======================================\n";

    if(!rA.pass){pass=false; cout<<"  [失败] 健康链路未 PASS\n";}
    if(!rB.deg){pass=false; cout<<"  [失败] 未检出退化运动\n";}
    cout<<(pass?" [PASS] 全流程 + 退化告警 正常":" [FAIL] 见上")<<"\n";
    return pass?0:1;
}
