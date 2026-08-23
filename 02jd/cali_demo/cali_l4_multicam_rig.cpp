// cali_l4_multicam_rig.cpp
// ============================================================================
// L4  多相机 rig (环视 BEV 拼接 + 回环一致性)
// ----------------------------------------------------------------------------
// 这是 cali_demo 的 L4 扩展篇第五条（最后一条），补齐 P2 cali 四主线之「多相机 rig」
// （应用：环视 surround-view / BEV 拼接）。
//
// 工业刚需：自动驾驶/AGV 用 4~6 路相机环绕做 360° 环视拼接。两块必须标定：
//   ① 各相机外参 T_i(rig->cam_i) 自洽——绕 rig 一圈的"回环闭合"应=单位变换（否则拼接错位）；
//   ② 逆透视映射 IPM——把每路图像地面像素投回 rig 地面，跨相机拼接成统一 BEV。
//
// 全程纯 C++17（不依赖 OpenCV）——刚体变换/透视投影/平面求交都是标准几何，
// 与已落地的七篇 L4 保持"纯标准库"一致。
//
// 编译: g++ -O2 -std=c++17 cali_l4_multicam_rig.cpp -o t && ./t
// ============================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

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
static Mat rot_z(double a){ double c=std::cos(a),s=std::sin(a); return {{c,-s,0},{s,c,0},{0,0,1}}; }
static Mat rot_x(double a){ double c=std::cos(a),s=std::sin(a); return {{1,0,0},{0,c,-s},{0,s,c}}; }

// ---------- SE3 (rig->cam) ----------
struct SE3 { Mat R; std::vector<double> t; };
static SE3 se3_mul(const SE3&A,const SE3&B){
    SE3 C; C.R=matmul(A.R,B.R);
    auto rb=matvec(A.R,B.t);
    C.t={rb[0]+A.t[0], rb[1]+A.t[1], rb[2]+A.t[2]};
    return C;
}
static SE3 se3_inv(const SE3&A){
    SE3 C; C.R=transpose(A.R);
    auto rt=matvec(C.R,A.t);
    C.t={-rt[0],-rt[1],-rt[2]};
    return C;
}
static std::vector<double> apply_se3(const SE3&T,double x,double y,double z){
    std::vector<double> p={x,y,z}; auto r=matvec(T.R,p);
    return {r[0]+T.t[0], r[1]+T.t[1], r[2]+T.t[2]};
}

// ---------- 参数 ----------
static const double fx=500.0, fy=500.0, cx=320.0, cy=240.0;
static const double Rrig=1.5, h=1.0, tilt=2.094;   // 半径/高度/120°俯仰(朝外+下)

static void proj(const double p[3], double&u, double&v){
    u=fx*p[0]/p[2]+cx; v=fy*p[1]/p[2]+cy;
}
// 相机系射线 K^-1(u,v,1) 与地面(rig z=0)交点 -> rig 系地面点
static bool ipm_to_ground(const SE3&T, double u,double v, double G[3]){
    double dx=(u-cx)/fx, dy=(v-cy)/fy, dz=1.0;
    SE3 Tinv=se3_inv(T);
    double znum = Tinv.R[2][0]*dx + Tinv.R[2][1]*dy + Tinv.R[2][2]*dz;
    if(std::fabs(znum)<1e-12) return false;
    double s = -Tinv.t[2]/znum;
    if(s<=0) return false;
    double Pc[3]={s*dx,s*dy,s*dz};
    auto Gr=apply_se3(Tinv, Pc[0],Pc[1],Pc[2]);
    G[0]=Gr[0]; G[1]=Gr[1]; G[2]=Gr[2];
    return true;
}

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    printf("  [%-46s] %s  (%.4e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4.6 多相机 rig 环视/BEV] 验证报告 ======\n";

    // 4 相机环绕 rig
    SE3 Ts[4];
    for(int i=0;i<4;++i){
        double phi_pos=i*M_PI/2.0;
        std::vector<double> cp={Rrig*std::cos(phi_pos), Rrig*std::sin(phi_pos), h};
        double phi_rot=i*M_PI/2.0 - M_PI/2.0;
        Mat Ri=matmul(rot_z(phi_rot), rot_x(-tilt));
        auto rb=matvec(Ri,cp);
        Ts[i].R=Ri; Ts[i].t={-rb[0],-rb[1],-rb[2]};
    }

    printf("--- ① 回投: 世界点 -> 各相机投影 -> 反投 -> 回 rig ---\n");
    double P[3]={0.3,-0.2,0.5};
    double maxrt=0;
    for(int i=0;i<4;++i){
        auto Pcam=apply_se3(Ts[i], P[0],P[1],P[2]);
        double u,v; proj(Pcam.data(),u,v);
        double Pcam2[3]={(u-cx)/fx*Pcam[2],(v-cy)/fy*Pcam[2],Pcam[2]};
        auto Prig=apply_se3(se3_inv(Ts[i]), Pcam2[0],Pcam2[1],Pcam2[2]);
        double e=std::max({std::fabs(Prig[0]-P[0]),std::fabs(Prig[1]-P[1]),std::fabs(Prig[2]-P[2])});
        if(e>maxrt) maxrt=e;
    }
    printf("  回投最大偏差=%.3e\n",maxrt);
    chk("世界点回投自洽(各相机<1e-9)", maxrt, 1e-9, false);

    printf("--- ② 回环闭合: cam0->cam1->cam2->cam3->cam0 = I ---\n");
    SE3 rel01=se3_mul(se3_inv(Ts[0]),Ts[1]);
    SE3 rel12=se3_mul(se3_inv(Ts[1]),Ts[2]);
    SE3 rel23=se3_mul(se3_inv(Ts[2]),Ts[3]);
    SE3 rel30=se3_mul(se3_inv(Ts[3]),Ts[0]);
    SE3 loop=se3_mul(se3_mul(se3_mul(rel01,rel12),rel23),rel30);
    double maxR=0,maxt=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        maxR=std::max(maxR, std::fabs(loop.R[i][j]-(i==j?1.0:0.0)));
    for(int i=0;i<3;++i) maxt=std::max(maxt, std::fabs(loop.t[i]));
    printf("  回环 R 偏差=%.3e  t 偏差=%.3e\n",maxR,maxt);
    chk("回环闭合 旋转=I (<1e-9)", maxR, 1e-9, false);
    chk("回环闭合 平移=0 (<1e-9)", maxt, 1e-9, false);

    printf("--- ③ BEV/IPM: 投影像素 -> 地面 -> 重投影像素 (各相机回环) ---\n");
    double maxipm=0; int cnt=0;
    for(int i=0;i<4;++i) for(int du:{-100,-50,0,50,100}) for(int dv:{-80,-40,0,40,80}){
        double u=cx+du, v=cy+dv;
        double G[3];
        if(!ipm_to_ground(Ts[i],u,v,G)) continue;
        if(std::fabs(G[2])>1e-9) continue;
        double u2,v2; proj(apply_se3(Ts[i],G[0],G[1],G[2]).data(), u2,v2);
        double e=std::max(std::fabs(u2-u),std::fabs(v2-v));
        if(e>maxipm) maxipm=e; ++cnt;
    }
    printf("  IPM 回环最大像素误差=%.3e (样本=%d)\n",maxipm,cnt);
    chk("IPM 地面回投像素误差<1e-6", maxipm, 1e-6, false);

    printf("--- ④ 跨相机 BEV 一致: 共享地面点被多相机 IPM 还原相同 ---\n");
    double maxcross=0; int shared=0;
    for(int gi=-6;gi<=6;++gi) for(int gj=-6;gj<=6;++gj){
        double gx=gi*0.4, gy=gj*0.4;
        double G[3]={gx,gy,0.0};
        int nvis=0; double locmax=0;
        for(int i=0;i<4;++i){
            auto Pc=apply_se3(Ts[i], gx,gy,0.0);
            if(Pc[2]<=0) continue;
            double u,v; proj(Pc.data(),u,v);
            if(std::fabs(u-cx)>320 || std::fabs(v-cy)>240) continue;
            double G1[3];
            if(!ipm_to_ground(Ts[i],u,v,G1)) continue;
            ++nvis;
            double e=std::max({std::fabs(G1[0]-G[0]),std::fabs(G1[1]-G[1]),std::fabs(G1[2]-G[2])});
            if(e>locmax) locmax=e;
        }
        if(nvis>=2){ ++shared; if(locmax>maxcross) maxcross=locmax; }
    }
    printf("  跨相机共享地面点=%d  最大 BEV 偏差=%.3e\n",shared,maxcross);
    chk("找到跨相机共享地面点(>=1)", (double)shared, 1.0, true);
    chk("跨相机 BEV 一致(偏差<1e-6)", maxcross, 1e-6, false);

    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] 多相机 rig 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
