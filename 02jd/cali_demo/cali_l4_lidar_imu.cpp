// cali_l4_lidar_imu.cpp
// ============================================================================
// L4  LiDAR-IMU 时空标定（Spatial + Temporal Calibration）
// ----------------------------------------------------------------------------
// 补齐工业界激光雷达/IMU 融合最刚需的两块"标定"：
//   ① 空间外参(X = T_lidar^imu)：Umeyama/Kabsch 刚体对齐（3x3 SVD via Jacobi）
//   ② 时间偏移(temporal offset Δt)：1-D 网格搜索使 IMU 预测相对位姿 == LiDAR 注册相对位姿
//   ③ 联合：外参 X 嵌入后，残差仍在正确 Δt 处最小 —— 证明两标定须协同
// 全程纯 C++17（不依赖 OpenCV）——SVD/Jacobi/SE(3) 都是标准库可实现的数值方法，
// 与已落地的 jac_l4_autodiff / ctrl_l4_mpc / ls_l4_lasso / cali_l4_fisheye_rs 四篇 L4 一致。
//
// 编译: g++ -O2 -std=c++17 cali_l4_lidar_imu.cpp -o t && ./t
//   (本文件纯标准库，无需 OpenCV；CMake 仍会链接 opencv 但本文件不引用，无害)
// ============================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>

using Mat3 = std::vector<std::vector<double>>;
using Vec3 = std::vector<double>;
static const double PI = 3.14159265358979323846;

// ---------- 固定种子 LCG（与受管 Python 验证脚本逐位一致）----------
static unsigned int g_s = 12345u;
static unsigned int lcg_next(){
    g_s = (1664525u * g_s + 1013904223u);   // unsigned 32-bit 自动取模
    return g_s;
}
static double lcg_u(){ return (double)lcg_next() / 4294967296.0; }     // [0,1)
static double lcg_r(double lo, double hi){ return lo + (hi-lo)*lcg_u(); }

// ---------- 3x3 矩阵工具 ----------
static Mat3 matmul3(const Mat3&A, const Mat3&B){
    Mat3 C(3, Vec3(3,0));
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        for(int k=0;k<3;++k) C[i][j]+=A[i][k]*B[k][j];
    return C;
}
static Mat3 trans3(const Mat3&A){
    Mat3 C(3, Vec3(3,0));
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) C[i][j]=A[j][i];
    return C;
}
static Vec3 mul3v(const Mat3&A, const Vec3&v){
    return { A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
             A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
             A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2] };
}
static double det3(const Mat3&A){
    return A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
          -A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
          +A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
}
static Mat3 sub3(const Mat3&A, const Mat3&B){
    Mat3 C(3, Vec3(3,0));
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) C[i][j]=A[i][j]-B[i][j];
    return C;
}
static double fro3(const Mat3&A){
    double s=0; for(int i=0;i<3;++i) for(int j=0;j<3;++j) s+=A[i][j]*A[i][j];
    return std::sqrt(s);
}
static Mat3 I3(){ return {{1,0,0},{0,1,0},{0,0,1}}; }

// ---------- 旋转 ----------
static Mat3 Rx(double a){ double c=std::cos(a),s=std::sin(a); return {{1,0,0},{0,c,-s},{0,s,c}}; }
static Mat3 Ry(double a){ double c=std::cos(a),s=std::sin(a); return {{c,0,s},{0,1,0},{-s,0,c}}; }
static Mat3 Rz(double a){ double c=std::cos(a),s=std::sin(a); return {{c,-s,0},{s,c,0},{0,0,1}}; }

// ---------- Jacobi 特征分解（对称 3x3）----------
static std::vector<double> jacobi_eigen(const Mat3&Ain, Mat3& Vout){
    Mat3 a(3, Vec3(3,0));
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) a[i][j]=Ain[i][j];
    Vout = I3();
    for(int sweep=0; sweep<100; ++sweep){
        int p=0,q=1; double mx=std::fabs(a[0][1]);
        if(std::fabs(a[0][2])>mx){ mx=std::fabs(a[0][2]); p=0; q=2; }
        if(std::fabs(a[1][2])>mx){ mx=std::fabs(a[1][2]); p=1; q=2; }
        if(mx < 1e-15) break;
        double app=a[p][p], aqq=a[q][q], apq=a[p][q];
        double c,s;
        if(std::fabs(apq)<1e-30){ c=1; s=0; }
        else{
            double theta=(aqq-app)/(2.0*apq);
            double t = std::copysign(1.0,theta)/(std::fabs(theta)+std::sqrt(theta*theta+1.0));
            c = 1.0/std::sqrt(1.0+t*t); s = t*c;
        }
        double app_n = c*c*app - 2*s*c*apq + s*s*aqq;
        double aqq_n = s*s*app + 2*s*c*apq + c*c*aqq;
        int r = 3-p-q;
        double arp=a[r][p], arq=a[r][q];
        double arp_n = c*arp - s*arq;
        double arq_n = s*arp + c*arq;
        a[p][p]=app_n; a[q][q]=aqq_n; a[p][q]=0; a[q][p]=0;
        a[r][p]=arp_n; a[p][r]=arp_n; a[r][q]=arq_n; a[q][r]=arq_n;
        for(int i=0;i<3;++i){
            double vip=Vout[i][p], viq=Vout[i][q];
            Vout[i][p]=c*vip - s*viq;
            Vout[i][q]=s*vip + c*viq;
        }
    }
    return { a[0][0], a[1][1], a[2][2] };
}

// ---------- Umeyama 刚体对齐 (R,t) ----------
struct CalibRes { Mat3 R; Vec3 t; };
static CalibRes umeyama(const std::vector<Vec3>& P, const std::vector<Vec3>& Q){
    int N=(int)P.size();
    Vec3 muP{0,0,0}, muQ{0,0,0};
    for(int i=0;i<N;++i){ for(int k=0;k<3;++k){ muP[k]+=P[i][k]; muQ[k]+=Q[i][k]; } }
    for(int k=0;k<3;++k){ muP[k]/=N; muQ[k]/=N; }
    Mat3 H(3, Vec3(3,0));
    for(int i=0;i<N;++i) for(int a=0;a<3;++a) for(int b=0;b<3;++b)
        H[a][b]+=(P[i][a]-muP[a])*(Q[i][b]-muQ[b]);
    Mat3 G = matmul3(H, trans3(H));
    Mat3 U; std::vector<double> eig = jacobi_eigen(G, U);
    std::vector<double> sv{ std::sqrt(std::max(eig[0],0.0)),
                             std::sqrt(std::max(eig[1],0.0)),
                             std::sqrt(std::max(eig[2],0.0)) };
    Mat3 Vt(3, Vec3(3,0));
    for(int b=0;b<3;++b){
        if(sv[b] > 1e-12){
            Vec3 htu = mul3v(trans3(H), {U[0][b],U[1][b],U[2][b]});
            for(int k=0;k<3;++k) Vt[b][k] = htu[k]/sv[b];
        }
    }
    Mat3 V = trans3(Vt);
    double d = det3(matmul3(U, V));
    double sgn = (d>=0)? 1.0 : -1.0;
    Mat3 S = I3(); S[2][2]=sgn;
    Mat3 R = matmul3(matmul3(V, S), trans3(U));
    Vec3 t{ muQ[0]-R[0][0]*muP[0]-R[0][1]*muP[1]-R[0][2]*muP[2],
            muQ[1]-R[1][0]*muP[0]-R[1][1]*muP[1]-R[1][2]*muP[2],
            muQ[2]-R[2][0]*muP[0]-R[2][1]*muP[1]-R[2][2]*muP[2] };
    return { R, t };
}

// ---------- SE(3) 相对位姿（刚性螺旋运动模型）----------
// 返回 {R,t} of T(tb) T(ta)^-1  (T(τ)=[Rz(ωτ)|(vτ,0,0)])
static std::pair<Mat3,Vec3> T_rel(double ta, double tb){
    const double omega=0.3, v=0.5;
    double d = tb-ta;
    Mat3 R = Rz(omega*d);
    Vec3 t = { v*tb, 0.0, 0.0 };
    Vec3 Rta_vta = mul3v(R, { v*ta, 0.0, 0.0 });
    t = { t[0]-Rta_vta[0], t[1]-Rta_vta[1], t[2]-Rta_vta[2] };
    return { R, t };
}
static std::pair<Mat3,Vec3> pose_mul(const std::pair<Mat3,Vec3>& A, const std::pair<Mat3,Vec3>& B){
    Mat3 R = matmul3(A.first, B.first);
    Vec3 t = { A.second[0] + (A.first[0][0]*B.second[0]+A.first[0][1]*B.second[1]+A.first[0][2]*B.second[2]),
               A.second[1] + (A.first[1][0]*B.second[0]+A.first[1][1]*B.second[1]+A.first[1][2]*B.second[2]),
               A.second[2] + (A.first[2][0]*B.second[0]+A.first[2][1]*B.second[1]+A.first[2][2]*B.second[2]) };
    return { R, t };
}
static std::pair<Mat3,Vec3> pose_inv(const std::pair<Mat3,Vec3>& A){
    Mat3 Rt = trans3(A.first);
    Vec3 t = { -(Rt[0][0]*A.second[0]+Rt[0][1]*A.second[1]+Rt[0][2]*A.second[2]),
               -(Rt[1][0]*A.second[0]+Rt[1][1]*A.second[1]+Rt[1][2]*A.second[2]),
               -(Rt[2][0]*A.second[0]+Rt[2][1]*A.second[1]+Rt[2][2]*A.second[2]) };
    return { Rt, t };
}
static double rel_residual(const std::pair<Mat3,Vec3>& C){
    double tr=C.first[0][0]+C.first[1][1]+C.first[2][2];
    double ang=std::acos(std::max(-1.0,std::min(1.0,(tr-1)/2)));
    double tn=std::hypot(C.second[0],C.second[1],C.second[2]);
    return ang+tn;
}

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    std::printf("  [%-40s] %s  (%.3e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4 LiDAR-IMU 时空标定] 验证报告 ======\n";

    // ===== A. 空间外参标定 (Umeyama / Kabsch) =====
    std::cout<<"--- A. 空间外参标定 (Umeyama / Kabsch) ---\n";
    double al=lcg_r(-PI,PI), be=lcg_r(-PI,PI), ga=lcg_r(-PI,PI);
    Mat3 Rtrue = matmul3(Rz(ga), matmul3(Ry(be), Rx(al)));
    Vec3 ttrue = { lcg_r(-1,1)*2, lcg_r(-1,1)*2, lcg_r(-1,1)*2 };

    // A1: 4 点精确
    std::vector<Vec3> P4(4), Q4(4);
    for(int k=0;k<4;++k){
        P4[k] = { lcg_r(-1,1), lcg_r(-1,1), lcg_r(-1,1) };
        Q4[k] = { Rtrue[0][0]*P4[k][0]+Rtrue[0][1]*P4[k][1]+Rtrue[0][2]*P4[k][2]+ttrue[0],
                  Rtrue[1][0]*P4[k][0]+Rtrue[1][1]*P4[k][1]+Rtrue[1][2]*P4[k][2]+ttrue[1],
                  Rtrue[2][0]*P4[k][0]+Rtrue[2][1]*P4[k][1]+Rtrue[2][2]*P4[k][2]+ttrue[2] };
    }
    CalibRes c1 = umeyama(P4, Q4);
    double errR1 = fro3(sub3(c1.R, Rtrue));
    double errt1 = std::hypot(c1.t[0]-ttrue[0], c1.t[1]-ttrue[1], c1.t[2]-ttrue[2]);
    std::printf("    A1 零噪声 4点: ||R-Rtrue||_F=%.2e  ||t-ttrue||=%.2e\n", errR1, errt1);
    chk("零噪声 R 恢复误差<1e-9", errR1, 1e-9, false);
    chk("零噪声 t 恢复误差<1e-9", errt1, 1e-9, false);

    // SVD 自洽 + 正交性
    {
        int N=4; Vec3 muP{0,0,0}, muQ{0,0,0};
        for(int i=0;i<N;++i){ for(int k=0;k<3;++k){ muP[k]+=P4[i][k]; muQ[k]+=Q4[i][k]; } }
        for(int k=0;k<3;++k){ muP[k]/=N; muQ[k]/=N; }
        Mat3 H(3,Vec3(3,0));
        for(int i=0;i<N;++i) for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            H[a][b]+=(P4[i][a]-muP[a])*(Q4[i][b]-muQ[b]);
        Mat3 G = matmul3(H, trans3(H));
        Mat3 U; std::vector<double> eig=jacobi_eigen(G,U);
        std::vector<double> sv{std::sqrt(std::max(eig[0],0.0)),std::sqrt(std::max(eig[1],0.0)),std::sqrt(std::max(eig[2],0.0))};
        Mat3 Vt(3,Vec3(3,0));
        for(int b=0;b<3;++b){
            if(sv[b]>1e-12){
                Vec3 htu=mul3v(trans3(H),{U[0][b],U[1][b],U[2][b]});
                for(int k=0;k<3;++k) Vt[b][k]=htu[k]/sv[b];
            }
        }
        Mat3 V=trans3(Vt);
        Mat3 USVt = matmul3(matmul3(U, {{sv[0],0,0},{0,sv[1],0},{0,0,sv[2]}}), Vt);
        double svd_res = fro3(sub3(USVt,H));
        double orthU = fro3(sub3(matmul3(U,trans3(U)), I3()));
        double orthV = fro3(sub3(matmul3(V,trans3(V)), I3()));
        std::printf("    SVD 重建残差=%.2e  U正交=%.2e  V正交=%.2e\n", svd_res, orthU, orthV);
        chk("SVD 重建残差<1e-9", svd_res, 1e-9, false);
        chk("U 正交<1e-9", orthU, 1e-9, false);
        chk("V 正交<1e-9", orthV, 1e-9, false);
    }

    // A2: 10 点带噪声
    const double noise=0.003;
    std::vector<Vec3> P10(10), Q10(10);
    for(int k=0;k<10;++k){
        P10[k] = { lcg_r(-1,1), lcg_r(-1,1), lcg_r(-1,1) };
        Q10[k] = { Rtrue[0][0]*P10[k][0]+Rtrue[0][1]*P10[k][1]+Rtrue[0][2]*P10[k][2]+ttrue[0]+lcg_r(-noise,noise),
                   Rtrue[1][0]*P10[k][0]+Rtrue[1][1]*P10[k][1]+Rtrue[1][2]*P10[k][2]+ttrue[1]+lcg_r(-noise,noise),
                   Rtrue[2][0]*P10[k][0]+Rtrue[2][1]*P10[k][1]+Rtrue[2][2]*P10[k][2]+ttrue[2]+lcg_r(-noise,noise) };
    }
    CalibRes c2 = umeyama(P10, Q10);
    double recon=0;
    for(int k=0;k<10;++k){
        double ex=(c2.R[0][0]*P10[k][0]+c2.R[0][1]*P10[k][1]+c2.R[0][2]*P10[k][2])+c2.t[0]-Q10[k][0];
        double ey=(c2.R[1][0]*P10[k][0]+c2.R[1][1]*P10[k][1]+c2.R[1][2]*P10[k][2])+c2.t[1]-Q10[k][1];
        double ez=(c2.R[2][0]*P10[k][0]+c2.R[2][1]*P10[k][1]+c2.R[2][2]*P10[k][2])+c2.t[2]-Q10[k][2];
        recon+=std::hypot(ex,ey,ez);
    }
    recon/=10;
    double orthR2 = fro3(sub3(matmul3(c2.R,trans3(c2.R)), I3()));
    std::printf("    A2 噪声: 平均重建误差=%.3e  R正交残差=%.2e\n", recon, orthR2);
    chk("噪声重建误差<3*noise", recon, 3*noise, false);
    chk("R 正交<1e-9", orthR2, 1e-9, false);

    // ===== B. 时间偏移标定 (1-D 网格搜索) =====
    std::cout<<"--- B. 时间偏移标定 (1-D 网格搜索) ---\n";
    double dt_true=0.05, t1=0.5, t2=1.5;
    auto A = T_rel(t1+dt_true, t2+dt_true);   // LiDAR 注册相对位姿 (= 物体真实相对运动, X=I)
    double best_dt=0, best_res=1e18;
    for(int kk=-150; kk<=300; ++kk){
        double dt_hat=kk*0.001;
        auto B = T_rel(t1+dt_hat, t2+dt_hat);
        auto C = pose_mul(pose_inv(A), B);
        double res = rel_residual(C);
        if(res < best_res){ best_res=res; best_dt=dt_hat; }
    }
    std::printf("    B: dt_true=%.3f  回收 dt_hat=%.3f  min残差=%.4f\n", dt_true, best_dt, best_res);
    chk("时间偏移回收误差<0.002", std::fabs(best_dt-dt_true), 0.002, false);

    // ===== C. 联合: 外参 X 嵌入后，残差仍于正确 dt 处最小 =====
    std::cout<<"--- C. 联合: 外参 X 嵌入后残差一致性 ---\n";
    double axc=lcg_r(-PI,PI), bxc=lcg_r(-PI,PI), gxc=lcg_r(-PI,PI);
    Mat3 RX = matmul3(Rz(gxc), matmul3(Ry(bxc), Rx(axc)));
    Vec3 tX = { lcg_r(-1,1)*2, lcg_r(-1,1)*2, lcg_r(-1,1)*2 };
    auto X = std::make_pair(RX, tX);
    auto embed=[&](double dt_hat){
        auto Bc = T_rel(t1+dt_hat, t2+dt_hat);
        return pose_mul(pose_mul(pose_inv(X), Bc), X);
    };
    auto Aemb = embed(dt_true);
    double best2_dt=0, best2_res=1e18;
    for(int kk=-150; kk<=300; ++kk){
        double dt_hat=kk*0.001;
        auto Bemb = embed(dt_hat);
        auto C = pose_mul(pose_inv(Aemb), Bemb);
        double res = rel_residual(C);
        if(res < best2_res){ best2_res=res; best2_dt=dt_hat; }
    }
    std::printf("    C: dt_true=%.3f  回收 dt_hat=%.3f  min残差=%.4f\n", dt_true, best2_dt, best2_res);
    chk("联合时间偏移回收误差<0.002", std::fabs(best2_dt-dt_true), 0.002, false);

    // ===== 汇总 =====
    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] LiDAR-IMU 时空标定 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
