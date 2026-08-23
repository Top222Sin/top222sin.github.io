// cali_l4_self_active.cpp
// ============================================================================
// L4  自标定 / 主动标定 (self-calibration & active calibration)
// ----------------------------------------------------------------------------
// 这是 cali_demo 的 L4 扩展篇第二条，补齐 P2 cali 四主线之「自标定/主动标定」，
// 并把 L3.3 自动/在线标定系统里"蜻蜓点水"的「主动运动规划」章落到可运行的数学上。
//
// 两个工业刚需：
//   ① 自标定：相机绕固定内部结构旋转 → 无穷单应 H∞ = K·R·K⁻¹ 仅含内参。
//      由已知旋转 R + 图像对应点估出的 H∞，可线性回收 K（无需标定板！）。
//   ② 主动标定 / D-最优运动规划：让机器人"主动选运动"，使标定问题信息量最大、
//      避免 L2.3 退化的「只绕单一轴转」导致自由度不可观测。
//
// 全程纯 C++17（不依赖 OpenCV）——自标定是线性代数、主动规划是观测矩阵秩分析，
// 从公式实现最直观，也与 jac_l4_autodiff / ctrl_l4_mpc / ls_l4_lasso /
// cali_l4_fisheye_rs / cali_l4_lidar_imu 五篇 L4 保持"纯标准库"一致。
//
// 编译: g++ -O2 -std=c++17 cali_l4_self_active.cpp -o t && ./t
//   (本文件纯标准库，无需 OpenCV)
// ============================================================================
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <utility>

using Mat = std::vector<std::vector<double>>;

// ---------- 小矩阵库 ----------
static Mat matmul(const Mat&A,const Mat&B){
    int n=(int)A.size(), m=(int)A[0].size(), p=(int)B[0].size();
    Mat C(n, std::vector<double>(p,0.0));
    for(int i=0;i<n;++i){
        const auto&Ai=A[i];
        for(int k=0;k<m;++k){ double a=Ai[k]; if(a==0.0) continue; const auto&Bk=B[k];
            for(int j=0;j<p;++j) C[i][j]+=a*Bk[j]; }
    }
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
static Mat inv3(const Mat&M){
    double a=M[0][0],b=M[0][1],c=M[0][2], d=M[1][0],e=M[1][1],f=M[1][2], g=M[2][0],h=M[2][1],i=M[2][2];
    double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if(std::fabs(det)<1e-18) det=(det>=0?1e-18:-1e-18);
    Mat R(3, std::vector<double>(3));
    R[0][0]=(e*i-f*h)/det; R[0][1]=(c*h-b*i)/det; R[0][2]=(b*f-c*e)/det;
    R[1][0]=(f*g-d*i)/det; R[1][1]=(a*i-c*g)/det; R[1][2]=(c*d-a*f)/det;
    R[2][0]=(d*h-e*g)/det; R[2][1]=(b*g-a*h)/det; R[2][2]=(a*e-b*d)/det;
    return R;
}
static Mat inv(const Mat&M){                       // Gauss-Jordan, 任意阶
    int n=(int)M.size();
    Mat A(n, std::vector<double>(2*n,0.0));
    for(int i=0;i<n;++i){ for(int j=0;j<n;++j) A[i][j]=M[i][j]; A[i][n+i]=1.0; }
    for(int col=0;col<n;++col){
        int piv=col; double best=0;
        for(int r=col;r<n;++r){ double v=std::fabs(A[r][col]); if(v>best){best=v;piv=r;} }
        if(piv!=col) std::swap(A[col],A[piv]);
        double pv=A[col][col]; for(int j=0;j<2*n;++j) A[col][j]/=pv;
        for(int r=0;r<n;++r){ if(r!=col && std::fabs(A[r][col])>0){ double f=A[r][col];
            for(int j=0;j<2*n;++j) A[r][j]-=f*A[col][j]; } }
    }
    Mat R(n, std::vector<double>(n));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) R[i][j]=A[i][n+j];
    return R;
}
static Mat jacobi_eig(const Mat&Ain){              // 对称阵特征值 (Jacobi 旋转)
    int n=(int)Ain.size();
    Mat A(n, std::vector<double>(n));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) A[i][j]=Ain[i][j];
    Mat V(n, std::vector<double>(n,0.0));
    for(int i=0;i<n;++i) V[i][i]=1.0;
    for(int it=0;it<200;++it){
        int p=0,q=1; double mx=0;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){ double v=std::fabs(A[i][j]); if(v>mx){mx=v;p=i;q=j;} }
        if(mx<1e-14) break;
        double app=A[p][p], aqq=A[q][q], apq=A[p][q];
        double phi=0.5*std::atan2(2.0*apq, app-aqq);
        double c=std::cos(phi), s=std::sin(phi);
        for(int i=0;i<n;++i){ double ai=A[i][p], bi=A[i][q]; A[i][p]=c*ai-s*bi; A[i][q]=s*ai+c*bi; }
        for(int i=0;i<n;++i){ double ai=A[p][i], bi=A[q][i]; A[p][i]=c*ai-s*bi; A[q][i]=s*ai+c*bi; }
        for(int i=0;i<n;++i){ double vi=V[i][p], bi=V[i][q]; V[i][p]=c*vi-s*bi; V[i][q]=s*vi+c*bi; }
    }
    Mat R(1, std::vector<double>(n));
    for(int i=0;i<n;++i) R[0][i]=A[i][i];
    return R;                                        // 仅取特征值
}
static std::vector<double> solve4(const Mat&M,const std::vector<double>&r){ // 4×4 Gauss-Jordan
    int n=4;
    Mat X(n, std::vector<double>(n+1,0.0));
    for(int i=0;i<n;++i){ for(int j=0;j<n;++j) X[i][j]=M[i][j]; X[i][n]=r[i]; }
    for(int col=0;col<n;++col){
        int piv=col; double best=0;
        for(int k=col;k<n;++k){ double v=std::fabs(X[k][col]); if(v>best){best=v;piv=k;} }
        if(piv!=col) std::swap(X[col],X[piv]);
        double pv=X[col][col]; for(int j=0;j<n+1;++j) X[col][j]/=pv;
        for(int rr=0;rr<n;++rr){ if(rr!=col){ double f=X[rr][col]; for(int j=0;j<n+1;++j) X[rr][j]-=f*X[col][j]; } }
    }
    std::vector<double> sol(n);
    for(int i=0;i<n;++i) sol[i]=X[i][n];
    return sol;
}

// ---------- 相机 / 旋转 ----------
static const double fx=500.0, fy=500.0, cx=320.0, cy=240.0;
static const Mat K={{fx,0,cx},{0,fy,cy},{0,0,1}};
static Mat rot_z(double a){ double c=std::cos(a),s=std::sin(a); return {{c,-s,0},{s,c,0},{0,0,1}}; }
static Mat rot_x(double a){ double c=std::cos(a),s=std::sin(a); return {{1,0,0},{0,c,-s},{0,s,c}}; }
static Mat rot_y(double a){ double c=std::cos(a),s=std::sin(a); return {{c,0,s},{0,1,0},{-s,0,c}}; }
// 旋转相机无穷单应 H∞ = K·R·K⁻¹ （仅依赖内参 K 与已知旋转 R）
static Mat Hinf(const Mat&R){ return matmul(matmul(K,R), inv3(K)); }

// ---------- Part A: 主动自标定 (K·R = H∞·K, 线性回收 K) ----------
// 对 (fx,fy,cx,cy) 线性; K[2][2]=1 固定投影尺度。每 (i,j) 给出一条线性方程。
struct LinRow { double a[4]; double b; };
static std::vector<LinRow> build_lin(const Mat&H,const Mat&R){
    std::vector<LinRow> rows;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j){
        double kr_cf[4]={0,0,0,0}; double kr_c=0;
        if(i==0){ kr_cf[0]+=R[0][j]; kr_cf[2]+=R[2][j]; }
        else if(i==1){ kr_cf[1]+=R[1][j]; kr_cf[3]+=R[2][j]; }
        else { kr_c=R[2][j]; }
        double hk_cf[4]={0,0,0,0}; double hk_c=0;
        if(j==0) hk_cf[0]+=H[i][0];
        else if(j==1) hk_cf[1]+=H[i][1];
        else { hk_cf[2]+=H[i][0]; hk_cf[3]+=H[i][1]; hk_c=H[i][2]; }
        LinRow row;
        for(int k=0;k<4;++k) row.a[k]=kr_cf[k]-hk_cf[k];
        row.b=hk_c-kr_c;
        rows.push_back(row);
    }
    return rows;
}
static std::vector<double> recover_K(const std::vector<std::pair<Mat,Mat>>&HRs){
    std::vector<std::vector<double>> A; std::vector<double> b;
    for(const auto&pr:HRs){
        auto rows=build_lin(pr.first,pr.second);
        for(const auto&r:rows){ A.push_back({r.a[0],r.a[1],r.a[2],r.a[3]}); b.push_back(r.b); }
    }
    Mat At=transpose(A);
    Mat ATA=matmul(At,A);
    std::vector<double> ATb=matvec(At,b);
    return solve4(ATA,ATb);
}

// ---------- Part B: 主动标定 / D-最优运动规划 ----------
// 4 参数 (fx,fy,cx,cy) 的观测矩阵 AᵀA：奇异值/行列式 = 信息量。
static void observ(const std::vector<std::pair<Mat,Mat>>&HRs, double&svmin, double&det, int&deficit){
    std::vector<std::vector<double>> A; std::vector<double> b;
    for(const auto&pr:HRs){
        auto rows=build_lin(pr.first,pr.second);
        for(const auto&r:rows){ A.push_back({r.a[0],r.a[1],r.a[2],r.a[3]}); b.push_back(r.b); }
    }
    Mat At=transpose(A); Mat ATA=matmul(At,A);
    Mat ev=jacobi_eig(ATA);
    svmin=1e30; det=1.0;
    for(int k=0;k<(int)ev[0].size();++k){ double e=ev[0][k]; double s=std::sqrt(std::max(e,0.0)); if(s<svmin) svmin=s; det*=e; }
    deficit=0; for(int k=0;k<(int)ev[0].size();++k) if(ev[0][k]<=1e-7) ++deficit;
}

// ---------- 校验辅助 ----------
static int g_pass=0, g_fail=0;
static void chk(const char* name, double got, double thr, bool ge=true){
    bool ok = ge ? (got>=thr) : (got<=thr);
    if(ok) ++g_pass; else ++g_fail;
    printf("  [%-46s] %s  (%.4e %s %.3e)\n", name, ok?"PASS":"FAIL", got, ge?">=":"<=", thr);
}

int main(){
    std::cout<<"====== [L4.3 自标定 / 主动标定] 验证报告 ======\n";

    // ===== Part A: 主动自标定 (H∞=K R K⁻¹ → 线性回收 K) =====
    printf("--- Part A: 主动自标定 (H∞=K R K⁻¹ → 线性回收 K) ---\n");
    // 机器人主动旋转相机 3 组异轴旋转 (已知 R; 由图像对应点估计 H∞)
    Mat R1=rot_z(0.30), R2=rot_x(0.43), R3=rot_y(0.25);
    std::vector<std::pair<Mat,Mat>> HRs={{Hinf(R1),R1},{Hinf(R2),R2},{Hinf(R3),R3}};
    auto sol=recover_K(HRs);
    double fxr=sol[0], fyr=sol[1], cxr=sol[2], cyr=sol[3];
    printf("  回收 K: fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n",fxr,fyr,cxr,cyr);
    chk("回收 fx ≈ 真值(误差<1e-4)", std::fabs(fxr-fx), 1e-4, false);
    chk("回收 fy ≈ 真值(误差<1e-4)", std::fabs(fyr-fy), 1e-4, false);
    chk("回收 cx ≈ 真值(误差<1e-4)", std::fabs(cxr-cx), 1e-4, false);
    chk("回收 cy ≈ 真值(误差<1e-4)", std::fabs(cyr-cy), 1e-4, false);
    // 验证 Krec·R = H·Krec (闭环)
    Mat Krec={{fxr,0,cxr},{0,fyr,cyr},{0,0,1}};
    Mat H1=Hinf(R1);
    Mat KrecR=matmul(Krec,R1), HKrec=matmul(H1,Krec);
    double res=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) res=std::max(res, std::fabs(KrecR[i][j]-HKrec[i][j]));
    chk("Krec·R1 = H1·Krec 残差<1e-9", res, 1e-9, false);

    // ===== Part B: 主动标定 / D-最优运动规划 =====
    printf("--- Part B: 主动标定 / D-最优运动规划 ---\n");
    double sv_same,det_same; int def1;
    observ({{Hinf(rot_z(0.20)),rot_z(0.20)},{Hinf(rot_z(0.55)),rot_z(0.55)}}, sv_same,det_same,def1);
    double sv_div,det_div; int d2;
    observ({{Hinf(rot_z(0.20)),rot_z(0.20)},{Hinf(rot_x(0.43)),rot_x(0.43)}}, sv_div,det_div,d2);
    double sv_div3,det_div3; int d3;
    observ({{Hinf(rot_z(0.20)),rot_z(0.20)},{Hinf(rot_x(0.43)),rot_x(0.43)},{Hinf(rot_y(0.31)),rot_y(0.31)}}, sv_div3,det_div3,d3);
    printf("  同z轴 (退化): σ_min=%.3e det=%.3e 亏秩=%d\n",sv_same,det_same,def1);
    printf("  z+x 异轴 (可辨识): σ_min=%.3e det=%.3e\n",sv_div,det_div);
    printf("  z+x+y 三组: σ_min=%.3e det=%.3e\n",sv_div3,det_div3);
    chk("同z轴运动 -> 观测矩阵亏秩(秩<4)", (double)def1, 1.0, true);
    chk("异轴运动 -> 满秩(σ_min>1e-1)", sv_div, 1e-1, true);
    chk("D-最优: 异轴信息量 >> 同z轴 (det 比>10)", det_div/std::max(det_same,1e-30), 10.0, true);
    chk("D-最优: 三组异轴比两组更优 (det 更大)", det_div3, det_div, true);

    std::cout<<"==========================================\n";
    std::cout<<"  PASS="<<g_pass<<"  FAIL="<<g_fail<<"  -> "
             <<(g_fail==0?"[PASS] 自标定/主动标定 全部验证通过":"[FAIL] 见上")<<"\n";
    return g_fail==0?0:1;
}
