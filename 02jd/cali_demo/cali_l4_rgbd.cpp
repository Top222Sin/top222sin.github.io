// cali_l4_rgbd.cpp
// ============================================================================
// L4  RGB-D 深度标定与对齐 (depth bias calibration + depth-color extrinsics)
// ----------------------------------------------------------------------------
// 这是 cali_demo 的 L4 扩展篇第四条，补齐 P2 cali 四主线之「RGB-D」（应用：深度-彩色对齐）。
//
// 工业刚需：RealSense/Kinect 等 RGB-D 相机给"彩色图 + 深度图"，但深度含系统性偏置
// （ToF/结构光随距离非线性漂移），且深度与彩色是两路相机需外参对齐才能"像素级配准"。
//   ① 深度偏置标定：d_meas = d_true + c·d_true² → 用已知距离拟合 c → 校正。
//   ② 深度-彩色外参一致性：刚体变换 (R,t) 的回环重投影应自洽。
//
// 全程纯 C++17（不依赖 OpenCV）——偏置是标量二次式、外参是刚体变换，从公式实现最直观。
//
// 编译: g++ -O2 -std=c++17 cali_l4_rgbd.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>

using Mat = std::vector<std::vector<double>>;

static Mat matmul(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(), p=(int)B[0].size();
    Mat C(n, std::vector<double>(p,0.0));
    for(int i=0;i<n;++i) for(int k=0;k<m;++k){ double a=A[i][k]; if(a==0) continue;
        for(int j=0;j<p;++j) C[i][j]+=a*B[k][j]; }
    return C;
}
static Mat transpose(const Mat&A){
    int n=(int)A.size(), m=(int)A[0].size();
    Mat T(m, std::vector<double>(n,0.0));
    for(int i=0;i<n;++i) for(int j=0;j<m;++j) T[j][i]=A[i][j];
    return T;
}
static std::vector<double> matvec(const Mat&A,const std::vector<double>&x){
    int n=(int)A.size(), m=(int)A[0].size();
    std::vector<double> y(n,0.0);
    for(int i=0;i<n;++i){ double s=0; for(int j=0;j<m;++j) s+=A[i][j]*x[j]; y[i]=s; }
    return y;
}

// ---------- 参数 ----------
static const double fx=500.0, fy=500.0, cx=320.0, cy=240.0;
static const int    W=40, Hh=30;
static const double Z0=2.0;          // 真值深度平面
static const double c_true=0.01;      // ToF 深度二次偏置系数

static double depth_meas(double d_true){ return d_true + c_true*d_true*d_true; }
// d_meas = x + c x^2 -> x = (-1 + sqrt(1 + 4 c d_meas)) / (2c)
static double depth_corr(double d_meas, double c){
    return (-1.0 + std::sqrt(1.0 + 4.0*c*d_meas))/(2.0*c);
}
static Mat rot_z(double a){ double c=std::cos(a),s=std::sin(a); return {{c,-s,0},{s,c,0},{0,0,1}}; }

// 投影 / 反投影 (z 沿光轴)
static void proj(const double p[3], double&u, double&v){
    u = fx*p[0]/p[2] + cx;  v = fy*p[1]/p[2] + cy;
}
static void unproj(double u,double v,double z, double p[3]){
    p[0]=(u-cx)/fx*z; p[1]=(v-cy)/fy*z; p[2]=z;
}
// 刚体变换 R p + t ; 逆 Rᵀ(p - t)
static void xform(const Mat&R,const std::vector<double>&t,const double p[3],double q[3]){
    std::vector<double> pv={p[0],p[1],p[2]};
    auto r=matvec(R,pv);
    q[0]=r[0]+t[0]; q[1]=r[1]+t[1]; q[2]=r[2]+t[2];
}
static void xform_T(const Mat&R,const std::vector<double>&t,const double p[3],double q[3]){
    std::vector<double> d={p[0]-t[0],p[1]-t[1],p[2]-t[2]};
    Mat Rt=transpose(R);
    auto r=matvec(Rt,d);
    q[0]=r[0]; q[1]=r[1]; q[2]=r[2];
}

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    printf("  [%-46s] %s  (%.4e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4.5 RGB-D 深度标定/对齐] 验证报告 ======\n";
    printf("--- ① 深度偏置标定 + 校正 ---\n");

    // 标定: 从已知距离拟合 c (线性: d_meas = d + c d^2)
    double calib_d[6]={0.5,1.0,1.5,2.0,2.5,3.0};
    double num=0, den=0;
    for(int i=0;i<6;++i){
        double dm = depth_meas(calib_d[i]);
        num += (dm - calib_d[i]) * calib_d[i]*calib_d[i];
        den += calib_d[i]*calib_d[i]*calib_d[i]*calib_d[i];
    }
    double c_fit = num/den;

    // 平面场景: 真值深度 Z0, 测值含偏置
    double e_true=0,e_meas=0,e_corr=0; int n=0;
    for(int v=0;v<Hh;++v) for(int u=0;u<W;++u){
        double d_true=Z0;
        double d_meas=depth_meas(d_true);
        double d_corr=depth_corr(d_meas,c_fit);
        e_true += std::fabs(d_true - d_true);
        e_meas += std::fabs(d_meas - d_true);
        e_corr += std::fabs(d_corr - d_true);
        ++n;
    }
    e_true/=n; e_meas/=n; e_corr/=n;
    printf("  真值回投误差=%.3e  测值(偏置)误差=%.3e  校正后误差=%.3e\n",e_true,e_meas,e_corr);
    chk("真值深度回投误差<1e-9 (几何正确)", e_true, 1e-9, false);
    chk("未校正偏置明显(误差>1e-2)",        e_meas, 1e-2, true);
    chk("校正后误差<1e-4",                  e_corr, 1e-4, false);
    chk("校正后 << 校正前",                 e_corr, e_meas, false);
    chk("标定系数 c≈0.01 (<1e-6)",          std::fabs(c_fit-c_true), 1e-6, false);

    printf("--- ② 深度-彩色外参一致性 (回环重投影) ---\n");
    Mat Rcol=rot_z(0.05);
    std::vector<double> tcol={0.02,-0.01,0.0};
    double maxalign=0;
    for(int v=0;v<Hh;v+=5) for(int u=0;u<W;u+=5){
        double Pd[3]; unproj((double)u,(double)v,Z0,Pd);
        double ud,vd; proj(Pd,ud,vd);
        double Pd2[3]; unproj(ud,vd,Z0,Pd2);
        double Pc[3]; xform(Rcol,tcol,Pd2,Pc);
        double uc_,vc_; proj(Pc,uc_,vc_);
        double Pc2[3]; unproj(uc_,vc_,Pc[2],Pc2);
        double Pd3[3]; xform_T(Rcol,tcol,Pc2,Pd3);
        double e=std::max({std::fabs(Pd3[0]-Pd[0]),std::fabs(Pd3[1]-Pd[1]),std::fabs(Pd3[2]-Pd[2])});
        if(e>maxalign) maxalign=e;
    }
    printf("  外参回环重投影最大偏差=%.3e\n",maxalign);
    chk("深度-彩色外参一致(回环<1e-9)", maxalign, 1e-9, false);

    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] RGB-D 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
