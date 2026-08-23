// cali_l4_structured_light.cpp
// ============================================================================
// L4  结构光 3D 扫描 (structured light / phase-shifting profilometry)
// ----------------------------------------------------------------------------
// 这是 cali_demo 的 L4 扩展篇第三条，补齐 P2 cali 四主线之「结构光」（应用：3D 扫描）。
//
// 工业刚需：单相机+投影仪即可对物体做高精 3D 重建（工业检测、人脸/模具扫描、流水线体积）。
// 核心：投影 N 步相移正弦条纹 → 相机解码每像素包裹相位 → 解包裹得绝对相位
//       → 反查投影列 → 三角测量恢复高度场。
//
// 全程纯 C++17（不依赖 OpenCV）——相移解码是标准三角函数，从公式实现最直观，
// 且 OpenCV 的 structured_light 模块 API 繁琐、本机无 OpenCV 也能验证。
//
// 编译: g++ -O2 -std=c++17 cali_l4_structured_light.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <cmath>
#include <cstdio>

// ---------- 参数 ----------
static const int   N   = 4;        // 相移步数
static const double P   = 24.0;    // 条纹周期 (投影列/周期)
static const double A   = 0.5;     // 条纹强度偏置
static const double B   = 0.4;     // 条纹强度幅度
static const int   W   = 40, Hh = 30;
static const double k   = 1.0;     // 高度 -> 相位位移 比例
static const double uc  = 20.0, vc = 15.0;
static const double sigma2 = 40.0;

static double p_flat(double u){ return 1.8*u; }                                   // 参考平面投影列映射
static double height(double u,double v){ return 2.0*std::exp(-(((u-uc)*(u-uc)+(v-vc)*(v-vc))/sigma2)); }
static double p_cam(double u,double v){ return p_flat(u) + k*height(u,v); }       // 物体存在时投影列

static double captured_pcam(double pc, int n){
    double theta = 2.0*M_PI*pc/P;
    return A + B*std::cos(theta + 2.0*M_PI*(double)n/(double)N);
}
// 解码包裹相位 phi_w = atan2(S,C)
static double raw_phase_pcam(double pc){
    double C=0.0,S=0.0;
    for(int n=0;n<N;++n){
        double ph = 2.0*M_PI*(double)n/(double)N;
        double In = captured_pcam(pc,n);
        C += In*std::cos(ph); S += In*std::sin(ph);
    }
    return std::atan2(S,C);
}
// 解包裹: 用参考平面 p_flat 定整周期, 得绝对投影列
static double unwrap_pcam(double pc, double u){
    double phi_w = raw_phase_pcam(pc);
    double psi = -phi_w;                       // 解码得 -θ, θ = 2π·pc/P
    double p_w = psi * P / (2.0*M_PI);         // ∈ (-P/2, P/2]
    double m = std::round((p_flat(u) - p_w)/P);
    return p_w + m*P;
}

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    printf("  [%-46s] %s  (%.4e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4.4 结构光 3D 扫描] 验证报告 ======\n";
    printf("--- 相位解码 + 解包裹 + 高度恢复 ---\n");

    // ① 高度恢复最大误差
    double maxerr=0;
    for(int v=0;v<Hh;++v) for(int u=0;u<W;++u){
        double pc=p_cam((double)u,(double)v);
        double p_est=unwrap_pcam(pc,(double)u);
        double e=std::fabs((p_est-p_flat((double)u))/k - height((double)u,(double)v));
        if(e>maxerr) maxerr=e;
    }
    printf("  最大高度恢复误差=%.3e\n",maxerr);
    chk("高度恢复 最大误差<1e-4", maxerr, 1e-4, false);

    // ② 平坦平面子测试 (无物体): 恢复高度应≈0
    double maxflat=0;
    for(int v=0;v<Hh;++v) for(int u=0;u<W;++u){
        double pc=p_flat((double)u);
        double p_est=unwrap_pcam(pc,(double)u);
        double e=std::fabs((p_est-p_flat((double)u))/k);
        if(e>maxflat) maxflat=e;
    }
    printf("  平坦平面恢复最大|h|=%.3e\n",maxflat);
    chk("平坦平面恢复高度≈0 (<1e-9)", maxflat, 1e-9, false);

    // ③ 凸起峰值
    double pk=unwrap_pcam(p_cam(uc,vc), uc);
    double pk_h=(pk-p_flat(uc))/k;
    printf("  凸起峰值 h=%.4f\n",pk_h);
    chk("凸起峰值≈2.0 (<1e-3)", std::fabs(pk_h-2.0), 1e-3, false);

    // ④ 包裹相位解码正确性: phi_w == wrapped(-θ)
    double maxpw=0;
    for(int v=0;v<Hh;v+=5) for(int u=0;u<W;u+=5){
        double pc=p_cam((double)u,(double)v);
        double theta=2.0*M_PI*pc/P;
        double phi_w=raw_phase_pcam(pc);
        // wrapped(-θ) ∈ (-π,π]
        double x=-theta+M_PI; x = x - 2.0*M_PI*std::floor(x/(2.0*M_PI)); double exp=x-M_PI;
        double d=std::fabs(phi_w-exp);
        if(d>maxpw) maxpw=d;
    }
    printf("  包裹相位解码最大偏差=%.3e\n",maxpw);
    chk("包裹相位解码=wrapped(-θ) (<1e-9)", maxpw, 1e-9, false);

    // ⑤ 解包裹连续性: 相邻像素无整周期(=P)跳变
    double maxjump=0;
    for(int v=0;v<Hh;++v) for(int u=1;u<W;++u){
        double d=std::fabs(unwrap_pcam(p_cam((double)u,(double)v),(double)u)
                          -unwrap_pcam(p_cam((double)(u-1),(double)v),(double)(u-1)));
        if(d>maxjump) maxjump=d;
    }
    printf("  相邻像素 p_est 最大跳变=%.3e (真斜率1.8, 周期%.0f)\n",maxjump,P);
    chk("解包裹无整周期跳变(<P/2=12)", maxjump, 12.0, false);

    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] 结构光 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
