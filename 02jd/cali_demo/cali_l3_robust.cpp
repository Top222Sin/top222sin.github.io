// cali_l3_robust.cpp
// ============================================================================
// L3.2  鲁棒估计与退化自适应 ⭐
// ----------------------------------------------------------------------------
// 真实采集常混有坏帧（检测抖动、运动模糊、板被遮挡一部分）。本 demo 演示：
//   · 基线：把坏帧一起喂给 calibrateCamera → 内参被系统性带偏
//   · 鲁棒：RANSAC 多子集采样 + 中位数评分（对离群不敏感）→ 只在"好帧"上精修
//   并对比两者恢复焦距的误差，证明鲁棒法能把被坏帧带偏的内参拉回真值。
// 同时给出"退化/坏数据"的自动告警思路（中位数残差异常即报警）。
// 全程合成数据，运行即出 PASS/FAIL。
//
// 编译: g++ -O2 -std=c++17 cali_l3_robust.cpp -o t $(pkg-config --cflags --libs opencv4) && ./t
// ============================================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

static double rnd(mt19937& g, double a, double b){ return uniform_real_distribution<double>(a,b)(g); }
static double focalErr(const Matx33d& K, double f0){ return abs(K(0,0)-f0)/f0*100.0; }
static double median(vector<double> v){ if(v.empty())return 0; sort(v.begin(),v.end()); size_t n=v.size(); return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }

int main(){
    bool pass=true; mt19937 g(20260817);
    const int cn=9, rn=6; const double sq=0.025; Size imgSize(640,480);
    Matx33d Kgt(520,0,320, 0,520,240, 0,0,1); Vec4d dgt(0.09,-0.035,0.001,0.0006);
    Mat Km(Kgt), Dm(4,1,CV_64F,{dgt[0],dgt[1],dgt[2],dgt[3]});

    // ---- 生成 N 帧；其中 OUT 帧为"坏帧"（检测被系统性 +6px 平移）----
    int N=30, OUT=4; vector<int> bad;
    while((int)bad.size()<OUT) { int k=(int)rnd(g,0,N); if(find(bad.begin(),bad.end(),k)==bad.end()) bad.push_back(k); }
    vector<vector<Point3f>> vobj(N); vector<vector<Point2f>> vdet(N);
    for(int i=0;i<N;i++){
        vector<Point3f> o; for(int r=0;r<rn;r++)for(int c=0;c<cn;c++)o.push_back(Point3f(c*sq,r*sq,0)); vobj[i]=o;
        Vec3d rv(rnd(g,-0.5,0.5),rnd(g,-0.5,0.5),rnd(g,-0.25,0.25)); Vec3d tv(rnd(g,-0.15,0.15),rnd(g,-0.15,0.15),rnd(g,0.35,0.9));
        vector<Point2f> det; projectPoints(o,rv,tv,Km,Dm,det);
        normal_distribution<double> pn(0,0.5); for(auto&p:det){p.x+=pn(g);p.y+=pn(g);}
        if(find(bad.begin(),bad.end(),i)!=bad.end()) for(auto&p:det){ p.x+=6.0; } // 系统性坏帧
        vdet[i]=det;
    }

    // ---- 基线：全部帧一起标 ----
    Mat Kb,Db; vector<Mat> rvb,tvb;
    calibrateCamera(vobj,vdet,imgSize,Kb,Db,rvb,tvb);
    double fErrBase = focalErr(Matx33d(Kb), 520.0);

    // ---- 鲁棒：RANSAC ----
    int iters=60, sample=15; double bestMed=1e9; vector<int> bestInliers;
    for(int it=0; it<iters; it++){
        vector<vector<Point3f>> so; vector<vector<Point2f>> sd;
        for(int s=0;s<sample;s++){ int idx=(int)rnd(g,0,N); so.push_back(vobj[idx]); sd.push_back(vdet[idx]); }
        Mat Ks,Ds; vector<Mat> rvs,tvs; calibrateCamera(so,sd,imgSize,Ks,Ds,rvs,tvs);
        // 用候选 Ks 在全部帧上算每帧 RMS（逐帧 solvePnP 求外参），取中位数评分
        vector<double> rmsAll(N,0);
        for(int i=0;i<N;i++){
            Mat rvi,tvi; solvePnP(vobj[i],vdet[i],Ks,Ds,rvi,tvi);
            vector<Point2f> proj; projectPoints(vobj[i],rvi,tvi,Ks,Ds,proj);
            double e=0; for(size_t j=0;j<proj.size();j++) e+=norm(proj[j]-vdet[i][j]); e/=proj.size(); rmsAll[i]=e;
        }
        double med=median(rmsAll);
        if(med<bestMed){ bestMed=med;
            bestInliers.clear(); for(int i=0;i<N;i++) if(rmsAll[i] < 2.0*med + 0.5) bestInliers.push_back(i); }
    }
    // 在 inlier 上精修
    vector<vector<Point3f>> io; vector<vector<Point2f>> id;
    for(int i: bestInliers){ io.push_back(vobj[i]); id.push_back(vdet[i]); }
    Mat Kr,Dr; vector<Mat> rvr,tvr;
    calibrateCamera(io,id,imgSize,Kr,Dr,rvr,tvr);
    double fErrRob = focalErr(Matx33d(Kr), 520.0);

    cout<<"[L3.2 鲁棒] 坏帧数="<<OUT<<"/"<<N<<"\n";
    cout<<"  基线(含坏帧) 焦距误差 = "<<fErrBase<<"%\n";
    cout<<"  RANSAC(好帧精修) 焦距误差 = "<<fErrRob<<"%\n";
    cout<<"  RANSAC 选中 inlier = "<<bestInliers.size()<<" 帧（应为 "<<N-OUT<<" 左右）\n";

    if(fErrRob>5.0){pass=false; cout<<"  [失败] 鲁棒法未恢复真值\n";}
    if(!(fErrBase > fErrRob)){pass=false; cout<<"  [失败] 鲁棒法未优于基线\n";}
    if((int)bestInliers.size() < N-OUT-1){pass=false; cout<<"  [失败] 未正确剔出坏帧\n";}

    cout<<"\n=================================================\n";
    cout<<(pass?" [PASS] RANSAC 鲁棒标定剔坏帧 + 拉回真值":" [FAIL] 见上")<<"\n";
    cout<<"=================================================\n";
    return pass?0:1;
}
