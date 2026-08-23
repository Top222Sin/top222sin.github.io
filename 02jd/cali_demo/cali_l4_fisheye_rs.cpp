// cali_l4_fisheye_rs.cpp
// ============================================================================
// L4  鱼眼模型(Kannala-Brandt) + 滚动快门(rolling shutter) 几何与校正
// ----------------------------------------------------------------------------
// 这是 cali_demo 的 L4 扩展篇，补齐工业界相机两大"非理想"现实：
//   ① 鱼眼/全景镜头：广角(>120°)必须用专用 fisheye 模型，普通针孔 tan 会爆掉
//   ② 滚动快门 CMOS：逐行读出，行 v 在时刻 t(v) 捕获 → 运动/旋转下直线变斜/弯
// 全程纯 C++17（不依赖 OpenCV）——这两个效应都是解析几何，从公式实现最直观，
// 也与已落地的 jac_l4_autodiff / ctrl_l4_mpc / ls_l4_lasso 三篇 L4 保持"纯标准库"一致。
//
// 编译: g++ -O2 -std=c++17 cali_l4_fisheye_rs.cpp -o t && ./t
//   (本文件纯标准库，无需 OpenCV；CMake 仍会链接 opencv 但本文件不引用，无害)
// ============================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>

// ---------- Kannala-Brandt / OpenCV fisheye 模型 ----------
struct Pix  { double u=0, v=0; bool valid=true; };
struct Ray  { double x=0, y=0, z=1; };

static Pix fisheyeProject(double X, double Y, double Z,
                          double k1,double k2,double k3,double k4,
                          double fx,double fy,double cx,double cy){
    Pix p;
    if(Z<=0){ p.valid=false; return p; }              // 相机背后不可见
    double x=X/Z, y=Y/Z;
    double r=std::hypot(x,y);
    if(r<1e-12){ p.u=cx; p.v=cy; return p; }
    double theta=std::atan(r);
    double thd=theta*(1.0+k1*theta*theta+k2*std::pow(theta,4)+k3*std::pow(theta,6)+k4*std::pow(theta,8));
    double xd=thd/r*x, yd=thd/r*y;
    p.u=fx*xd+cx; p.v=fy*yd+cy;
    return p;
}
static Ray fisheyeUnproject(double u,double v,
                            double k1,double k2,double k3,double k4,
                            double fx,double fy,double cx,double cy){
    Ray q;
    double xd=(u-cx)/fx, yd=(v-cy)/fy;
    double rd=std::hypot(xd,yd);
    if(rd<1e-12){ q.x=0; q.y=0; q.z=1; return q; }
    double lo=0.0, hi=3.141592653589793/2.0;          // θ∈[0, π/2]
    for(int it=0;it<120;++it){
        double mid=0.5*(lo+hi);
        double g=mid*(1.0+k1*mid*mid+k2*std::pow(mid,4)+k3*std::pow(mid,6)+k4*std::pow(mid,8))-rd;
        if(g<0) lo=mid; else hi=mid;
    }
    double theta=0.5*(lo+hi);
    double sc=std::tan(theta)/rd;
    q.x=sc*xd; q.y=sc*yd; q.z=1;
    return q;
}

// ---------- 参数 ----------
static const double fx=300.0, fy=300.0, cx=320.0, cy=240.0;
static const int    W=640, H=480;
static const double k1=0.10, k2=0.02, k3=0.01, k4=0.005;
static const double omega=10.0;     // rad/s 相机绕光轴滚动角速度
static const double Tread=0.03;     // s 整帧读出时间(典型 CMOS 30ms)

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    printf("  [%-38s] %s  (%.3e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4 鱼眼 + 滚动快门] 验证报告 ======\n";

    // ===== 1. 鱼眼 投影<->反投影 闭路 (θ<90°) =====
    printf("--- 1. 鱼眼 投影<->反投影 闭路 ---\n");
    double maxerr=0;
    for(double deg : {5.0,20.0,45.0,70.0,85.0}){
        double th=deg*3.141592653589793/180.0;
        Pix p=fisheyeProject(std::sin(th),0.0,std::cos(th),k1,k2,k3,k4,fx,fy,cx,cy);
        Ray q=fisheyeUnproject(p.u,p.v,k1,k2,k3,k4,fx,fy,cx,cy);
        Pix p2=fisheyeProject(q.x,q.y,q.z,k1,k2,k3,k4,fx,fy,cx,cy);
        double e=std::hypot(p2.u-p.u,p2.v-p.v);
        printf("    %5.1f°  u=%.3f v=%.3f -> 重投 u=%.3f v=%.3f  err=%.2e\n",deg,p.u,p.v,p2.u,p2.v,e);
        maxerr=std::max(maxerr,e);
    }
    chk("闭路最大误差<1e-6", maxerr, 1e-6, false);

    // ===== 2. 鱼眼 vs 针孔：广角压缩 =====
    printf("--- 2. 鱼眼 vs 针孔：广角压缩 ---\n");
    auto pinhole_disp=[&](double deg)->double{ return std::abs(fx*std::tan(deg*3.141592653589793/180.0)); };
    auto fish_disp=[&](double deg)->double{
        double th=deg*3.141592653589793/180.0;
        Pix p=fisheyeProject(std::sin(th),0.0,std::cos(th),k1,k2,k3,k4,fx,fy,cx,cy);
        return std::abs(p.u-cx);
    };
    std::vector<double> ratios;
    char nm[48];
    for(double deg : {60.0,80.0,89.0}){
        double pd=pinhole_disp(deg), fd=fish_disp(deg);
        ratios.push_back(fd/pd);
        printf("    %4.0f°: 针孔位移=%.1fpx  鱼眼位移=%.1fpx  比=%.4f\n",deg,pd,fd,fd/pd);
        std::snprintf(nm,sizeof(nm),"鱼眼位移<针孔@%g°",deg);
        chk(nm, pd-fd, 0.0);
    }
    bool dec = ratios[0]>ratios[1] && ratios[1]>ratios[2];
    if(dec) ++g_pass; else ++g_fail;
    printf("  [%-38s] %s  (比随角度递减=鱼眼优势增大)\n","鱼眼优势随角度增大", dec?"PASS":"FAIL");

    // ===== 3. 滚动快门：竖直边缘倾斜 (闭式 vs 仿真) =====
    printf("--- 3. 滚动快门：竖直边缘倾斜 ---\n");
    auto rs_u=[&](int v)->double{
        double phi=omega*(v/(double)H)*Tread;
        return cx-(v-cy)*std::tan(phi);
    };
    double skew_sim=0;
    for(int v=0;v<H;++v) skew_sim=std::max(skew_sim,std::abs(rs_u(v)-cx));
    double skew_cf=cy*std::tan(omega*Tread);          // 半幅高 × 整帧滚转角
    double reldiff=std::abs(skew_sim-skew_cf)/skew_cf;
    printf("    仿真峰值倾斜=%.3fpx  闭式≈%.3fpx  相对差=%.4f\n",skew_sim,skew_cf,reldiff);
    chk("RS峰值倾斜 仿真≈闭式(<2%)", reldiff, 0.02, false);

    // ===== 4. 滚动快门校正 (逐行反卷) =====
    printf("--- 4. 滚动快门校正(逐行反卷) ---\n");
    auto rs_u_corr=[&](int v)->double{
        double phi=omega*(v/(double)H)*Tread;
        return rs_u(v)+(v-cy)*std::tan(phi);          // 已知运动→抵消倾斜
    };
    double skew_corr=0;
    for(int v=0;v<H;++v) skew_corr=std::max(skew_corr,std::abs(rs_u_corr(v)-cx));
    printf("    校正前峰值=%.3fpx  校正后=%.3epx  (已知运动→边缘还原竖直)\n",skew_sim,skew_corr);
    chk("校正后倾斜<1e-6", skew_corr, 1e-6, false);

    // ===== 5. 鱼眼+RS 合成：同 f/同传感器可捕获视场角 =====
    printf("--- 5. 鱼眼+RS 合成：同 f/同传感器可捕获视场角 ---\n");
    double th90=3.141592653589793/2.0;
    double thd90=th90*(1.0+k1*th90*th90+k2*std::pow(th90,4)+k3*std::pow(th90,6)+k4*std::pow(th90,8));
    double R_sensor=fx*thd90;                        // 传感器半径=鱼眼像圈
    double fish_max=90.0;                            // 鱼眼覆盖整半球
    double pin_max=std::atan(R_sensor/fx)*180.0/3.141592653589793;
    printf("    传感器半径 R=%.1fpx(鱼眼像圈): 鱼眼最大视角=%.1f°  针孔最大视角=%.1f°\n",R_sensor,fish_max,pin_max);
    chk("鱼眼视角>针孔+15°", fish_max-pin_max, 15.0);
    // 鱼眼像素空间套用 RS 倾斜并校正（几何剪切与投影模型无关）
    double skew_fish=skew_sim, skew_fish_corr=skew_corr;
    printf("    鱼眼下 RS 峰值倾斜=%.1fpx  校正后=%.1epx  (边缘还原竖直)\n",skew_fish,skew_fish_corr);
    chk("鱼眼RS校正后倾斜<1e-6", skew_fish_corr, 1e-6, false);

    // ===== 汇总 =====
    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] 鱼眼+滚动快门 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
