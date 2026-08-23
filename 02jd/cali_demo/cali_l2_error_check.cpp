// cali_l2_error_check.cpp
// ============================================================================
// L2.5  误差校验方案（Error check）⭐
// ----------------------------------------------------------------------------
// 标定完了怎么知道"标得有多准"？本 demo 给出一套量化指标并在合成数据上闭环：
//   ① 内参：逐图重投影误差（均值/最大/标准差）
//   ② 手眼：AX=XB 一致性残差（旋转°+平移mm），并与真值比对
//   ③ TCP ：多点一致性（尖是否落在同一点）
//   ④ 系统性 vs 随机残差：按像点到主点半径分箱，区分"模型错(边缘大)"与"纯噪声(均匀)"
// 全程合成数据，运行即出 PASS/FAIL + 分指标。
//
// 编译: g++ -O2 -std=c++17 cali_l2_error_check.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
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
static Matx44d matRT(const Matx33d& R, const Vec3d& t){ Matx44d M; M=Matx44d::eye(); for(int i=0;i<3;i++){for(int j=0;j<3;j++)M(i,j)=R(i,j);M(i,3)=t[i];} return M; }
static Matx44d inv44(const Matx44d& T){ Mat M(T),Mi; invert(M,Mi,DECOMP_SVD); return Matx44d(Mi); }
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

int main(){
    bool pass=true; mt19937 g(20260817);
    const int cn=9, rn=6; const double sq=0.025; Size imgSize(640,480);
    Matx33d Kgt(520,0,320, 0,520,240, 0,0,1); Vec4d dgt(0.09,-0.035,0.001,0.0006);
    Mat Km(Kgt), Dm(4,1,CV_64F,{dgt[0],dgt[1],dgt[2],dgt[3]});

    // ---- ① 内参：逐图重投影误差 ----
    int N=24; vector<vector<Point3f>> vobj(N); vector<vector<Point2f>> vdet, vgt;
    for(int i=0;i<N;i++){ vector<Point3f> o; for(int r=0;r<rn;r++)for(int c=0;c<cn;c++)o.push_back(Point3f(c*sq,r*sq,0)); vobj[i]=o;
        Vec3d rv(rnd(g,-0.5,0.5),rnd(g,-0.5,0.5),rnd(g,-0.25,0.25)); Vec3d tv(rnd(g,-0.15,0.15),rnd(g,-0.15,0.15),rnd(g,0.35,0.9));
        vector<Point2f> gt, det; projectPoints(o,rv,tv,Km,Dm,gt); det=gt;
        normal_distribution<double> pn(0,0.6); for(auto&p:det){p.x+=pn(g);p.y+=pn(g);}
        vgt.push_back(gt); vdet.push_back(det); }
    Mat Kc,Dc; vector<Mat> rvecs,tvecs;
    double rms=calibrateCamera(vobj,vdet,imgSize,Kc,Dc,rvecs,tvecs);
    double sumErr=0,maxErr=0; vector<double> perImg;
    for(int i=0;i<N;i++){ vector<Point2f> proj; projectPoints(vobj[i],rvecs[i],tvecs[i],Kc,Dc,proj);
        double e=0; for(size_t j=0;j<proj.size();j++) e+=norm(proj[j]-vdet[i][j]); e/=proj.size();
        perImg.push_back(e); sumErr+=e; maxErr=max(maxErr,e); }
    double meanErr=sumErr/N;
    cout<<"[①内参] 重投影 RMS="<<rms<<"px  逐图均值="<<meanErr<<"px  最大="<<maxErr<<"px\n";
    if(rms>2.0||maxErr>3.0){pass=false;cout<<"  [失败] 内参残差过大\n";}

    // ---- ② 手眼：AX=XB 一致性残差 + 真值比对 ----
    Matx44d Xgt=matRT(randR(g,0.5),Vec3d(0.04,-0.02,0.10));
    int Mh=14; vector<Matx44d> G,T;
    for(int i=0;i<Mh;i++){ G.push_back(matRT(randR(g,0.6),Vec3d(rnd(g,-0.3,0.3),rnd(g,-0.3,0.3),rnd(g,0.4,0.9)))); T.push_back(inv44(Xgt)*inv44(G[i])); }
    vector<Matx33d> RA,RB; vector<Vec3d> tA,tB;
    for(int i=0;i<Mh-1;i++){ Matx44d Ai=inv44(G[i])*G[i+1], Bi=T[i]*inv44(T[i+1]);
        RA.push_back(Matx33d(Ai(0,0),Ai(0,1),Ai(0,2),Ai(1,0),Ai(1,1),Ai(1,2),Ai(2,0),Ai(2,1),Ai(2,2)));
        RB.push_back(Matx33d(Bi(0,0),Bi(0,1),Bi(0,2),Bi(1,0),Bi(1,1),Bi(1,2),Bi(2,0),Bi(2,1),Bi(2,2)));
        tA.push_back(Vec3d(Ai(0,3),Ai(1,3),Ai(2,3))); tB.push_back(Vec3d(Bi(0,3),Bi(1,3),Bi(2,3))); }
    Matx33d Rx; Vec3d tx; solveHandEye(RA,tA,RB,tB,Rx,tx); Matx44d Xest=matRT(Rx,tx);
    double maxRot=0,maxTr=0,sumRot=0;
    for(int i=0;i<(int)RA.size();i++){ Matx44d Ai=matRT(RA[i],tA[i]), Bi=matRT(RB[i],tB[i]); Matx44d E=inv44(Ai*Xest)*(Xest*Bi);
        Vec3d rv; Rodrigues(Mat(Matx33d(E(0,0),E(0,1),E(0,2),E(1,0),E(1,1),E(1,2),E(2,0),E(2,1),E(2,2))),rv);
        double deg=norm(rv)*180/CV_PI, tr=norm(Vec3d(E(0,3),E(1,3),E(2,3)));
        maxRot=max(maxRot,deg); maxTr=max(maxTr,tr*1000); sumRot+=deg; }
    Matx44d Eg=Xgt.inv()*Xest; Vec3d rvg; Rodrigues(Mat(Matx33d(Eg(0,0),Eg(0,1),Eg(0,2),Eg(1,0),Eg(1,1),Eg(1,2),Eg(2,0),Eg(2,1),Eg(2,2))),rvg);
    double gtAng=norm(rvg)*180/CV_PI, gtTr=norm(Vec3d(Eg(0,3),Eg(1,3),Eg(2,3)))*1000;
    cout<<"[②手眼] 一致性: 最大旋转残差="<<maxRot<<"°  最大平移残差="<<maxTr<<"mm\n";
    cout<<"        真值比对: 旋转误差="<<gtAng<<"°  平移误差="<<gtTr<<"mm\n";
    if(maxRot>0.1||maxTr>1.0){pass=false;cout<<"  [失败] 手眼一致性差\n";}

    // ---- ③ TCP：多点一致性 ----
    Vec3d xgt(0.04,0,0.12), pworld(0.30,0,0.45); int Nt=8;
    vector<Matx33d> Rt; vector<Vec3d> tt;
    for(int i=0;i<Nt;i++){ Matx33d Ri=randR(g,0.5); Rt.push_back(Ri); tt.push_back(pworld-Ri*xgt); }
    // 解
    int m=3*(Nt-1); Mat A(m,3,CV_64F),b(m,1,CV_64F); int row=0;
    for(int i=1;i<Nt;i++){ Matx33d dR=Rt[i]-Rt[0]; Vec3d db=tt[0]-tt[i]; for(int r=0;r<3;r++){A.at<double>(row,0)=dR(r,0);A.at<double>(row,1)=dR(r,1);A.at<double>(row,2)=dR(r,2);b.at<double>(row)=db[r];row++;} }
    Mat xm; solve(A,b,xm,DECOMP_SVD); Vec3d xest(xm.at<double>(0),xm.at<double>(1),xm.at<double>(2));
    double tcpMax=0; for(int i=0;i<Nt;i++) tcpMax=max(tcpMax, norm(Rt[i]*xest+tt[i]-(Rt[0]*xest+tt[0])));
    cout<<"[③TCP ] 多点一致性 max="<<tcpMax*1000<<"mm\n";
    if(tcpMax*1000>1.0){pass=false;cout<<"  [失败] TCP 一致性差\n";}

    // ---- ④ 系统性 vs 随机残差（按半径分箱）----
    auto radialTest=[&](const Vec4d& dreal, bool addRand, double& centerE, double& edgeE){
        int P=24; vector<vector<Point3f>> vo(P); vector<vector<Point2f>> vd; vector<double> radii;
        for(int i=0;i<P;i++){ vector<Point3f> o; for(int r=0;r<rn;r++)for(int c=0;c<cn;c++)o.push_back(Point3f(c*sq,r*sq,0)); vo[i]=o;
            Vec3d rv(rnd(g,-0.6,0.6),rnd(g,-0.6,0.6),rnd(g,-0.3,0.3)); Vec3d tv(rnd(g,-0.28,0.28),rnd(g,-0.28,0.28),rnd(g,0.35,0.9));
            vector<Point2f> det; Mat Dreal(4,1,CV_64F,{dreal[0],dreal[1],dreal[2],dreal[3]});
            projectPoints(o,rv,tv,Km,Dreal,det);
            for(auto&p:det){ radii.push_back(norm(p-Point2f(320,240))); if(addRand){normal_distribution<double> pn(0,0.6); p.x+=pn(g);p.y+=pn(g);} }
            vd.push_back(det); }
        Mat K2,D2; vector<Mat> rv2,tv2; calibrateCamera(vo,vd,imgSize,K2,D2,rv2,tv2);
        // 分箱：中心(<0.35Rmax) vs 边缘(>0.7Rmax)
        double rmax=*max_element(radii.begin(),radii.end()); centerE=0; edgeE=0; int nc=0,ne=0;
        for(int i=0;i<P;i++){ vector<Point2f> proj; projectPoints(vo[i],rv2[i],tv2[i],K2,D2,proj);
            for(size_t j=0;j<proj.size();j++){ double e=norm(proj[j]-vd[i][j]); double rr=norm(vd[i][j]-Point2f(320,240))/rmax;
                if(rr<0.35){centerE+=e;nc++;} else if(rr>0.7){edgeE+=e;ne++;} } }
        centerE/=nc; edgeE/=ne;
    };
    double cR,eR,cS,eS;
    radialTest(dgt, true, cR,eR);                 // 随机噪声：模型匹配，应均匀
    radialTest(Vec4d(0.10,-0.04,0.06,0.0), false, cS,eS); // 模型错：真实含 k3 高阶径向畸变，却只标 4 参数(k1,k2,p1,p2) → 边缘系统性偏大
    cout<<"[④残差] 随机噪声: 中心="<<cR<<"px 边缘="<<eR<<"px (≈均匀)\n";
    cout<<"       模型错  : 中心="<<cS<<"px 边缘="<<eS<<"px (边缘>>中心=系统性)\n";
    if(!(eR < cR*1.5)){pass=false;cout<<"  [失败] 随机情形异常\n";}      // 随机应较均匀
    if(!(eS > cS*1.5)){pass=false;cout<<"  [失败] 未检出系统性残差(模型错)\n";}

    cout<<"\n=================================================\n";
    cout<<(pass?" [PASS] 误差校验指标 + 系统性/随机区分 通过":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
