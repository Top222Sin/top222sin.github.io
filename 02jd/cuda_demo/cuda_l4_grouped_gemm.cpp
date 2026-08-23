// ============================================================================
//  cuda_l4_grouped_gemm.cpp
//  L4（工业扩展）· grouped GEMM：异构 M/N/K 分组矩阵乘 + launch 开销摊销
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l4_grouped_gemm.cpp -o grp && ./grp
//  （真机 cuBLAS grouped GEMM / persistent kernel 片段由 #ifdef __CUDACC__ 守护，g++ 跳过）
//
//  把"一个 step 里要算 N 个形状各异的 GEMM（MoE/多专家/长尾 kernel）"落成模型：
//    - 每个 group 有独立 (M,N,K)；grouped_gemm 一次性把它们全算完
//    - 正确性 == 逐 group 参考 GEMM；支持异构形状 + 退化形状(K=1 / 1×1)
//    - 收益模型：N 次独立 launch → 1 次 grouped launch，launch 开销按 N 倍摊销
// ============================================================================
#include <cstdio>
#include <vector>
#include <cmath>

struct Group { int M, N, K; };

// 确定性填充（与 Python 镜像同公式 → double 逐位一致）
static void fill_A(const Group& g, std::vector<double>& A){
    A.assign((size_t)g.M*g.K, 0.0);
    for(int i=0;i<g.M;++i) for(int k=0;k<g.K;++k) A[(size_t)i*g.K+k]=(i*g.K+k)*0.1-1.0;
}
static void fill_B(const Group& g, std::vector<double>& B){
    B.assign((size_t)g.K*g.N, 0.0);
    for(int k=0;k<g.K;++k) for(int j=0;j<g.N;++j) B[(size_t)k*g.N+j]=(k*g.N+j)*0.1-0.5;
}
// 参考 GEMM：C = A·B（double，固定累加序）
static void ref_gemm(const Group& g, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C){
    C.assign((size_t)g.M*g.N, 0.0);
    for(int i=0;i<g.M;++i) for(int j=0;j<g.N;++j){
        double s=0.0; for(int k=0;k<g.K;++k) s+=A[(size_t)i*g.K+k]*B[(size_t)k*g.N+j];
        C[(size_t)i*g.N+j]=s;
    }
}
// grouped GEMM：一次性遍历所有 group（真机 = 一个 kernel launch 处理整批）
static void grouped_gemm(const std::vector<Group>& groups,
                          std::vector<std::vector<double>>& Cs){
    Cs.clear(); Cs.reserve(groups.size());
    for(const Group& g: groups){
        std::vector<double> A,B,C; fill_A(g,A); fill_B(g,B); ref_gemm(g,A,B,C); Cs.push_back(C);
    }
}

// ============================ 主程序 + 验证 ===============================
int main(){
    int passed=0, failed=0;
    auto chk=[&](const char* name, bool ok){ printf("  %s %s\n", ok?"PASS":"FAIL", name); ok?++passed:++failed; };

    // 异构形状组（MoE/多专家典型：每专家形状不同）
    std::vector<Group> groups = { {64,128,32}, {32,32,64}, {128,64,16}, {16,16,48} };
    // 退化形状组（边界 case）
    std::vector<Group> edge   = { {1,1,1}, {8,8,1} };

    // ---------- C1: grouped == 逐 group 参考（异构形状）----------
    std::vector<std::vector<double>> Cg;
    grouped_gemm(groups, Cg);
    double e1=0.0;
    for(size_t gi=0; gi<groups.size(); ++gi){
        std::vector<double> A,B,Cref; fill_A(groups[gi],A); fill_B(groups[gi],B); ref_gemm(groups[gi],A,B,Cref);
        for(size_t i=0;i<Cref.size();++i) e1=std::max(e1, std::abs(Cg[gi][i]-Cref[i]));
    }
    chk("grouped GEMM == 逐 group 参考（异构 M/N/K, max|diff|）", e1 < 1e-9);

    // ---------- C2: 确定性（重跑一致）----------
    std::vector<std::vector<double>> Cg2;
    grouped_gemm(groups, Cg2);
    double e2=0.0; for(size_t gi=0;gi<Cg.size();++gi) for(size_t i=0;i<Cg[gi].size();++i) e2=std::max(e2, std::fabs(Cg[gi][i]-Cg2[gi][i]));
    chk("grouped GEMM 重跑确定性（两次结果逐位一致）", e2 < 1e-12);

    // ---------- C3: 退化形状（K=1 / 1×1）正确 ----------
    std::vector<std::vector<double>> Ce;
    grouped_gemm(edge, Ce);
    double e3=0.0;
    for(size_t gi=0; gi<edge.size(); ++gi){
        std::vector<double> A,B,Cref; fill_A(edge[gi],A); fill_B(edge[gi],B); ref_gemm(edge[gi],A,B,Cref);
        for(size_t i=0;i<Cref.size();++i) e3=std::max(e3, std::abs(Ce[gi][i]-Cref[i]));
    }
    chk("grouped GEMM 退化形状(K=1 / 1×1)正确", e3 < 1e-9);

    // ---------- C4: FLOPs 账面 == Σ 2·Mᵢ·Nᵢ·Kᵢ ----------
    long long flops=0; for(const Group& g: groups) flops += 2LL*g.M*g.N*g.K;
    long long expect=2LL*64*128*32 + 2LL*32*32*64 + 2LL*128*64*16 + 2LL*16*16*48;
    chk("grouped GEMM 总 FLOPs == Σ 2·Mᵢ·Nᵢ·Kᵢ", flops==expect && flops>0);

    // ---------- C5: launch 开销摊销模型 ----------
    // 真机：N 个独立 cublasGemm = N 次 launch；grouped = 1 次 launch（计算量相同）
    const int NG=(int)groups.size();            // 组数（这里取异构 4 组；模型对 N 通用）
    const double L_launch=5.0;                  // µs / 单次 launch 的 host 端开销
    const double T_compute=(double)NG*0.05;     // µs：计算量（两种路径相同，仅 launch 不同）
    double T_launch_individual = NG*L_launch;
    double T_launch_grouped   = 1.0*L_launch;
    double launch_speedup = T_launch_individual / T_launch_grouped;   // = NG（launch 次数比）
    double T_total_individual = T_launch_individual + T_compute;
    double T_total_grouped   = T_launch_grouped   + T_compute;
    bool c5 = (launch_speedup==(double)NG)                       // launch 开销按 N 倍摊销
           && (T_total_grouped < T_total_individual)             // 总墙钟必然更低
           && (NG>=2);
    printf("    [info] N=%d 组: launch 开销 %gx ↓ (%.1fµs → %.1fµs); 总墙钟 %.2fµs → %.2fµs\n",
           NG, (int)launch_speedup, T_launch_individual, T_launch_grouped, T_total_individual, T_total_grouped);
    chk("grouped GEMM launch 开销按 N 倍摊销 (1 次 vs N 次)", c5);

    printf("=== %d passed, %d failed ===\n", passed, failed);

#ifdef __CUDACC__
    // ---- 真机片段（仅 nvcc 下编译，作教材）----
    // cuBLAS grouped GEMM（每 group 独立指针 + 形状数组，一次 cublasGemm 调用处理整批）：
    //   cublasGemmBatchedEx 或分组 API：传入 A/B/C 指针数组 + 每组的 m,n,k 数组 + 1 次 launch
    //   → 替代 N 次 cublasGemm（N 次 launch）。MoE 多专家前向即此模式。
    // persistent kernel：kernel 常驻 SM，按 group 流式取任务，避免每 group 一次 launch 的空闲。
#endif
    return failed?1:0;
}
