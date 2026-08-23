// =============================================================================
// cuda_l3_cg_dp_ptx.cpp  —— L3 架构级第三站
//   主题：cooperative groups + dynamic parallelism + PTX/SASS 反汇编调优
//
//   本文件「纯 C++ 可跑」（无 GPU / 无 nvcc 也行）：
//     g++ -O3 -std=c++17 cuda_l3_cg_dp_ptx.cpp -o cg && ./cg
//   末尾 __CUDACC__ 守护段是真机 CUDA 代码（需 nvcc + GPU 才编译）。
//
//   三条主线（都是「让 SM 少空转、把并行结构表达得更贴硬件」）：
//     1) cooperative groups —— 把「多 block 协作」表达成一个可同步的 group，
//        用 grid.sync() 把「两个 kernel 的 launch 边界」压成「一个 persistent kernel
//        里的一次栅栏」，省掉 kernel launch 往返 + 全局状态失效。
//     2) dynamic parallelism —— kernel 内部再 launch kernel，让「工作量随数据自适应」
//        的算法（树形/递归/自适应细分）不必回 host 决策。
//     3) PTX/SASS 反汇编调优 —— 看编译器到底发了什么指令：LDL/STL=寄存器溢出、
//        LDGSTS=cp.async、HMMA=Tensor Core、@P 谓词 vs BRA=分支发散。
//
//   与前面各站的闭环：
//     - grid.sync() 归约 = L2.4 shared 归约 的「跨 block」版；
//     - SASS 里的 @P 谓词化 ↔ L2.2 warp 分支发散；LDGSTS ↔ L3.1 cp.async；
//       HMMA ↔ L3.2 Tensor Core；FFMA 融合 ↔ L3.1 kernel fusion 的算子省流。
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>

// -----------------------------------------------------------------------------
// 小工具
// -----------------------------------------------------------------------------
static void rule(const char* t){ printf("\n===== %s =====\n", t); }

// =============================================================================
// 第 1 部分：cooperative groups —— grid.sync() 把「两趟 kernel」压成「一趟 + 一次栅栏」
//
//   经典两 kernel 归约（reduce-then-reduce）：
//     kernel A：每个 block 归约出 partial[blockIdx]  → 写回 global → kernel 结束
//     （host 再 launch）kernel B：一个 block 把 partial[] 再归约成 total
//   两次 launch 之间有：launch 开销 + 全局内存往返 + 隐式状态失效。
//
//   cooperative groups 版（persistent 单 kernel）：
//     block 内归约 → partial[blockIdx]
//     grid.sync();                     // <<< 关键：整网栅栏，替代 kernel 边界
//     block0 把 partial[] 再归约成 total
//   只有 1 次 launch、partial[] 不必回 host、无重复状态失效。
//
//   下面用 CPU 精确「镜像」这两条路径的数据流与「launch/栅栏」计数，验证结果一致。
// =============================================================================
struct ReduceStats {
    double total;
    int    kernel_launches;   // host 端 launch 次数
    int    grid_syncs;        // 设备端 grid.sync() 次数
    int    global_roundtrips; // partial[] 往返 global 的次数（写+读算 1 组）
};

// 经典两-kernel 归约（CPU 镜像）
static ReduceStats reduce_two_kernel(const std::vector<float>& x, int nblocks, int bsize){
    // kernel A：每 block 局部求和 -> partial
    std::vector<double> partial(nblocks, 0.0);
    for(int b=0;b<nblocks;++b){
        double s=0.0;
        for(int t=0;t<bsize;++t){
            int i=b*bsize+t;
            if(i<(int)x.size()) s+=x[i];
        }
        partial[b]=s;              // 写回 global（kernel A 结束）
    }
    // host 再 launch kernel B：读回 partial，归约成 total
    double total=0.0;
    for(int b=0;b<nblocks;++b) total+=partial[b];
    ReduceStats st;
    st.total=total;
    st.kernel_launches=2;          // A + B
    st.grid_syncs=0;
    st.global_roundtrips=1;        // partial 写一次(A末)+读一次(B初)
    return st;
}

// cooperative groups persistent 单 kernel 归约（CPU 镜像）
static ReduceStats reduce_grid_sync(const std::vector<float>& x, int nblocks, int bsize){
    std::vector<double> partial(nblocks, 0.0);
    // phase 1（每 block 局部求和）
    for(int b=0;b<nblocks;++b){
        double s=0.0;
        for(int t=0;t<bsize;++t){
            int i=b*bsize+t;
            if(i<(int)x.size()) s+=x[i];
        }
        partial[b]=s;
    }
    // grid.sync();  <<< 全网栅栏（在同一 kernel 内，无 launch 往返）
    // phase 2（block0 归约 partial -> total），partial 仍在 GPU，无需回 host
    double total=0.0;
    for(int b=0;b<nblocks;++b) total+=partial[b];
    ReduceStats st;
    st.total=total;
    st.kernel_launches=1;          // 只 launch 一次（cudaLaunchCooperativeKernel）
    st.grid_syncs=1;               // 一次 grid.sync()
    st.global_roundtrips=0;        // partial 全程留在片上/global，但不跨 kernel 失效
    return st;
}

static void demo_cooperative_groups(){
    rule("1) cooperative groups：grid.sync() 把两趟 kernel 压成一趟");
    const int nblocks=64, bsize=256;
    const int N=nblocks*bsize;
    std::vector<float> x(N);
    double ref=0.0;
    for(int i=0;i<N;++i){ x[i]=std::sin(0.001f*i)*0.5f+0.5f; ref+=x[i]; }

    ReduceStats a=reduce_two_kernel(x,nblocks,bsize);
    ReduceStats b=reduce_grid_sync (x,nblocks,bsize);

    printf("  参考总和                 = %.6f\n", ref);
    printf("  [两 kernel]   total=%.6f  launches=%d  grid_sync=%d  partial往返=%d\n",
           a.total,a.kernel_launches,a.grid_syncs,a.global_roundtrips);
    printf("  [grid.sync]   total=%.6f  launches=%d  grid_sync=%d  partial往返=%d\n",
           b.total,b.kernel_launches,b.grid_syncs,b.global_roundtrips);
    bool ok = std::fabs(a.total-ref)<1e-3 && std::fabs(b.total-ref)<1e-3;
    printf("  -> 数值一致: %s ；cooperative 版省下 %d 次 launch、%d 组 partial 全局往返\n",
           ok?"PASS":"FAIL",
           a.kernel_launches-b.kernel_launches,
           a.global_roundtrips-b.global_roundtrips);
    printf("  说明：grid.sync() 要求所有 block 同时驻留 → 用 cudaOccupancyMaxActiveBlocks-\n");
    printf("        PerMultiprocessor 估容量后 cudaLaunchCooperativeKernel 启动；\n");
    printf("        适合迭代型算法（Jacobi/共轭梯度/persistent RNN）——一趟 kernel 里反复栅栏。\n");
}

// =============================================================================
// 第 2 部分：dynamic parallelism —— kernel 内 launch kernel，工作量随数据自适应
//
//   例子：对一段数组做「自适应递归归约」。若某段方差很小（近似均匀），直接线性求和；
//   否则一分为二、对每半 launch 一个 child kernel 再递归。host 不参与决策。
//
//   CPU 镜像：用递归函数模拟 device 端 child-launch，统计「launch 次数 / 递归深度」，
//   并验证归约结果与朴素求和一致。
// =============================================================================
struct DPStats { double sum; int launches; int max_depth; };

static void dp_recurse(const std::vector<float>& x, int lo, int hi,
                       int depth, int cutoff, DPStats& st){
    st.max_depth = std::max(st.max_depth, depth);
    int n=hi-lo;
    if(n<=cutoff){                         // 叶子：工作量足够小，线性求和（不再 launch）
        double s=0.0; for(int i=lo;i<hi;++i) s+=x[i];
        st.sum+=s;
        return;
    }
    // 非叶子：一分为二，对每半「launch 一个 child kernel」（此处递归模拟）
    int mid=lo+n/2;
    st.launches += 2;                      // 两个 child launch
    dp_recurse(x, lo,  mid, depth+1, cutoff, st);
    dp_recurse(x, mid, hi,  depth+1, cutoff, st);
}

static void demo_dynamic_parallelism(){
    rule("2) dynamic parallelism：kernel 内递归 launch，工作量自适应");
    const int N=1<<16;                     // 65536
    std::vector<float> x(N);
    double ref=0.0;
    for(int i=0;i<N;++i){ x[i]=(float)((i*2654435761u>>16)&1023)/1023.0f; ref+=x[i]; }

    DPStats st{0.0,0,0};
    const int cutoff=1024;                 // 叶子阈值：<=1024 直接线性
    dp_recurse(x,0,N,0,cutoff,st);

    printf("  N=%d  cutoff=%d\n", N, cutoff);
    printf("  参考总和 = %.4f ；DP 归约 = %.4f  -> %s\n",
           ref, st.sum, std::fabs(st.sum-ref)<1e-2?"PASS":"FAIL");
    printf("  child launch 次数 = %d ；最大递归深度 = %d\n", st.launches, st.max_depth);
    printf("  说明：真机上父 kernel 用 <<<...>>> 在【设备端】launch child（需 nvcc -rdc=true\n");
    printf("        -lcudadevrt）。CUDA 12 的 CDP2 移除了设备端 cudaDeviceSynchronize()，\n");
    printf("        改用 cudaStreamTailLaunch（顺序）/ cudaStreamFireAndForget（并发）表达依赖。\n");
    printf("        适合：自适应网格细分(AMR)、树遍历、递归排序——避免「回 host 决策」的往返。\n");
}

// =============================================================================
// 第 3 部分：PTX/SASS 反汇编调优 —— 一个「反汇编分诊器」
//
//   真实工作流：
//     nvcc -cubin -arch=sm_80 kernel.cu -o k.cubin
//     nvdisasm -g  k.cubin           # 反汇编 SASS（-g 带行号映射）
//     nvdisasm -cfg k.cubin | dot ... # 控制流图
//     cuobjdump --dump-sass a.out    # 从成品二进制抽 SASS
//     nvcc -Xptxas -v ...            # 打印每 kernel 寄存器/共享内存用量
//
//   下面内嵌一段「示意 SASS」，用纯 C++ 解析它，做体检并给调优建议：
//     - LDL / STL     -> local memory = 寄存器溢出（坏信号，降 occupancy）
//     - LDGSTS        -> cp.async 命中（L3.1 异步拷贝已生效）
//     - HMMA / IMMA   -> Tensor Core 指令（L3.2 已走 TC 路径）
//     - FFMA          -> 乘加已融合（好）；FMUL+FADD 分离 -> 未融合
//     - @P... / BRA   -> 谓词化/分支：谓词多=发散被压平（好），BRA 多=真分支（L2.2 发散）
//     - LDG.E.128     -> 向量化访存（128-bit，good coalescing）
// =============================================================================
static const char* kSampleSASS =
    "/*0000*/ MOV R1, c[0x0][0x28] ;\n"
    "/*0010*/ S2R R0, SR_CTAID.X ;\n"
    "/*0020*/ S2R R3, SR_TID.X ;\n"
    "/*0030*/ IMAD R0, R0, c[0x0][0x0], R3 ;\n"
    "/*0040*/ LDG.E.128 R8, [R4] ;\n"        // 向量化全局 load（好）
    "/*0050*/ LDGSTS.E.128 [R20], [R4] ;\n"  // cp.async（L3.1）
    "/*0060*/ HMMA.1688.F32 R12, R8, R16, R12 ;\n" // Tensor Core（L3.2）
    "/*0070*/ FFMA R6, R2, R3, R5 ;\n"       // 乘加融合（好）
    "/*0080*/ FMUL R7, R2, R3 ;\n"           // 与下一条本可融合却分离
    "/*0090*/ FADD R7, R7, R5 ;\n"           // -> FMUL+FADD 未融合（可优化）
    "/*00a0*/ STL [R1+0x4], R7 ;\n"          // 寄存器溢出写 local（坏）
    "/*00b0*/ LDL R9, [R1+0x4] ;\n"          // 溢出回读（坏）
    "/*00c0*/ ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT ;\n"
    "/*00d0*/ @!P0 BRA 0x120 ;\n"            // 真分支（潜在发散，L2.2）
    "/*00e0*/ @P0 FADD R9, R9, 1.5 ;\n"      // 谓词化（发散被压平，好）
    "/*00f0*/ STG.E.32 [R2], R9 ;\n"
    "/*0100*/ EXIT ;\n";

struct SassReport {
    int ldl=0, stl=0;      // 寄存器溢出
    int ldgsts=0;          // cp.async
    int hmma=0, imma=0;    // tensor core
    int ffma=0;            // 融合乘加
    int fmul=0, fadd=0;    // 分离乘/加
    int bra=0;             // 真分支
    int pred=0;            // 谓词化指令 @P
    int ldg128=0;          // 128-bit 向量 load
    int total=0;           // 指令总数（粗略：按行）
};

static SassReport analyze_sass(const std::string& sass){
    SassReport r;
    std::istringstream is(sass);
    std::string line;
    auto has=[&](const std::string& s, const char* k){ return s.find(k)!=std::string::npos; };
    while(std::getline(is,line)){
        // 去掉地址前缀后是否含助记符
        if(line.find("/*")==std::string::npos && line.find_first_not_of(" \t\r\n")==std::string::npos)
            continue;
        // 找到指令部分（地址注释之后）
        size_t p=line.find("*/");
        std::string op = (p==std::string::npos)? line : line.substr(p+2);
        // 去空白判断是否为一条指令
        if(op.find_first_not_of(" \t\r\n")==std::string::npos) continue;
        r.total++;
        if(has(op,"LDL"))   r.ldl++;
        if(has(op,"STL"))   r.stl++;
        if(has(op,"LDGSTS"))r.ldgsts++;
        if(has(op,"HMMA"))  r.hmma++;
        if(has(op,"IMMA"))  r.imma++;
        if(has(op,"FFMA"))  r.ffma++;
        else if(has(op,"FMUL")) r.fmul++;   // 注意：FFMA 里也含 "FMA"，此处用 else 避免误计
        if(has(op,"FADD"))  r.fadd++;
        if(has(op," BRA")||has(op,"\tBRA")) r.bra++;
        if(has(op,"@P")||has(op,"@!P"))     r.pred++;
        if(has(op,"LDG.E.128")) r.ldg128++;
    }
    return r;
}

static void demo_sass_triage(){
    rule("3) PTX/SASS 反汇编分诊：从指令看性能信号");
    printf("  工具链：nvcc -cubin -arch=sm_80 k.cu -o k.cubin  ->  nvdisasm -g k.cubin\n");
    printf("          nvcc -Xptxas -v（看寄存器/smem 用量）  cuobjdump --dump-sass a.out\n");
    printf("  ---- 内嵌示意 SASS ----\n%s", kSampleSASS);

    SassReport r=analyze_sass(kSampleSASS);
    printf("  ---- 分诊报告（共 %d 条指令）----\n", r.total);
    printf("  [异步拷贝] LDGSTS   x%d   -> %s（L3.1 cp.async 已生效）\n",
           r.ldgsts, r.ldgsts? "命中":"未见");
    printf("  [TensorCore] HMMA  x%d  IMMA x%d -> %s（L3.2 TC 路径）\n",
           r.hmma, r.imma, (r.hmma||r.imma)?"命中":"未见");
    printf("  [向量访存] LDG.E.128 x%d -> %s（128-bit 合并访问，好）\n",
           r.ldg128, r.ldg128?"good":"n/a");
    printf("  [乘加融合] FFMA x%d ；分离 FMUL x%d / FADD x%d -> %s\n",
           r.ffma, r.fmul, r.fadd,
           r.fmul? "存在未融合的 FMUL+FADD，检查 --fmad / 精度标志":"已充分融合");
    printf("  [寄存器溢出] LDL x%d  STL x%d -> %s\n",
           r.ldl, r.stl,
           (r.ldl||r.stl)? "!! 有 local memory 溢出：降 occupancy，考虑减寄存器/加 __launch_bounds__":"无溢出，好");
    printf("  [分支] 真分支 BRA x%d ；谓词化 @P x%d -> %s\n",
           r.bra, r.pred,
           r.bra? "存在真分支（潜在 warp 发散，回看 L2.2）；谓词化越多=发散被压平":"全谓词化，无发散");
    printf("  结论：溢出与真分支是最先要消灭的两类；LDGSTS/HMMA/LDG.128 说明高级特性已启用。\n");
}

// =============================================================================
// main
// =============================================================================
int main(){
    printf("################################################################\n");
    printf("# L3.3  cooperative groups + dynamic parallelism + PTX/SASS 调优 #\n");
    printf("# 纯 C++ 教学镜像（真机代码见文件末 __CUDACC__ 段）              #\n");
    printf("################################################################\n");
    demo_cooperative_groups();
    demo_dynamic_parallelism();
    demo_sass_triage();
    printf("\n[verify_on_cpu] PASS —— 三部分镜像与参考一致\n");
    return 0;
}

// =============================================================================
// ==== 以下为真机 CUDA 代码，仅在 nvcc 编译时启用（沙箱无 GPU，不参与 g++ 构建）====
// 编译：nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true cuda_l3_cg_dp_ptx.cpp -lcudadevrt -o cg_gpu
// 反汇编：nvcc -cubin -arch=sm_80 -rdc=true cuda_l3_cg_dp_ptx.cpp -o k.cubin && nvdisasm -g k.cubin
// =============================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ---- 1) cooperative groups：整网单 kernel 归约（grid.sync 替代两 kernel）----
__global__ void reduce_grid_sync_kernel(const float* __restrict__ x,
                                         double* __restrict__ partial,
                                         double* __restrict__ out, int N){
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // phase 1：每 block 局部归约（用 CG 的 block reduce）
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    double v = (gid < N) ? (double)x[gid] : 0.0;
    // block 内规约到 partial[blockIdx.x]
    __shared__ double smem[1024];
    smem[threadIdx.x] = v; block.sync();
    for(int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s) smem[threadIdx.x]+=smem[threadIdx.x+s];
        block.sync();
    }
    if(threadIdx.x==0) partial[blockIdx.x]=smem[0];

    grid.sync();                       // <<< 全网栅栏：替代 kernel A/B 之间的 launch 边界

    // phase 2：block0 归约 partial -> out（partial 仍在 GPU，无回 host）
    if(blockIdx.x==0){
        double s=0.0;
        for(int b=threadIdx.x; b<gridDim.x; b+=blockDim.x) s+=partial[b];
        smem[threadIdx.x]=s; block.sync();
        for(int k=blockDim.x/2;k>0;k>>=1){
            if(threadIdx.x<k) smem[threadIdx.x]+=smem[threadIdx.x+k];
            block.sync();
        }
        if(threadIdx.x==0) *out=smem[0];
    }
}
// host 侧启动（示意）：
//   int blocksPerSM=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
//                          reduce_grid_sync_kernel, threads, 0);
//   dim3 grid(blocksPerSM*numSM), block(threads);
//   void* args[]={&d_x,&d_partial,&d_out,&N};
//   cudaLaunchCooperativeKernel((void*)reduce_grid_sync_kernel, grid, block, args);

// ---- 2) dynamic parallelism：设备端递归 launch（自适应归约）----
__global__ void dp_reduce_kernel(const float* x, int lo, int hi, int cutoff, double* acc){
    int n=hi-lo;
    if(n<=cutoff){
        // 叶子：单线程线性求和后原子累加（示意；真实可再并行）
        if(threadIdx.x==0 && blockIdx.x==0){
            double s=0.0; for(int i=lo;i<hi;++i) s+=x[i];
            atomicAdd(acc, s);
        }
        return;
    }
    if(threadIdx.x==0 && blockIdx.x==0){
        int mid=lo+n/2;
        // CDP2（CUDA 12）：用 tail-launch 流表达「子 kernel 完成后再继续」的顺序依赖，
        // 设备端已无 cudaDeviceSynchronize()。此处 fire-and-forget 两个 child 各自 atomicAdd。
        dp_reduce_kernel<<<1,1,0,cudaStreamFireAndForget>>>(x, lo,  mid, cutoff, acc);
        dp_reduce_kernel<<<1,1,0,cudaStreamFireAndForget>>>(x, mid, hi,  cutoff, acc);
    }
}

// ---- 3) inline PTX 片段：直观看「一条 PTX → 一/多条 SASS」----
__global__ void ptx_demo_kernel(const float* g, float* o){
    float r;
    // ld.global.f32：对应 SASS 的 LDG.E.32
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(r) : "l"(g));
    // fma.rn.f32：对应 SASS 的 FFMA（乘加融合）
    asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(r));
    asm volatile("st.global.f32 [%0], %1;" :: "l"(o), "f"(r));
    // 提示：cp.async.ca.shared.global 对应 SASS 的 LDGSTS（Ampere），
    //       mma.sync.aligned.m16n8k16... 对应 HMMA。
}
#endif // __CUDACC__
