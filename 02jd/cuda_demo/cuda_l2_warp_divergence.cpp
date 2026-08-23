// ============================================================================
// L2.2 自己写高效 kernel —— warp 概念 / 分支发散 / occupancy / 寄存器压力
// 配套文档：cuda_l2_warp_divergence.md
// 对应笔记本：01CUDA基础学习.docx → L2 能写高效 kernel
//             → 2、warp 概念、分支发散、occupancy、寄存器压力
// 编译（需 NVIDIA GPU + CUDA Toolkit）：
//     nvcc -O3 -std=c++17 cuda_l2_warp_divergence.cpp -o warp && ./warp
// 想看寄存器用量：nvcc -O3 -std=c++17 --ptxas-options=-v cuda_l2_warp_divergence.cpp -o warp
// 沙箱无 GPU/nvcc，本文件为教材级标准代码，逻辑与真机一致。
// ----------------------------------------------------------------------------
// 关键手法：分支发散通常是"看不见的性能黑洞"。本文件用 __ballot_sync 把发散
// 模式"显影"成可校验的数字（不用计时），让 warp 内是否混合一目了然。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

// ============================================================================
// 演示一：warp 概念 —— 硬件调度的最小单位是 warp(32 线程)，不是单线程
// ----------------------------------------------------------------------------
// • 一个 block 被切成若干 warp：warp 数 = ceil(blockDim / 32)。
// • 同一 warp 内 32 个线程**共享一个 PC**（程序计数器），锁步(lockstep)执行同一条
//   指令；不是 SIMD 那种"一条指令广播到所有 lane"，而是"调度器每次发射一个 warp"。
// • 线程在 warp 内的编号叫 lane_id（0..31），block 内的 warp 编号叫 warp_id。
//   等价内建函数：__lane_id()；这里手算以便校验且兼容老架构。
// ============================================================================

__global__ void inspect_warp(int n, int* out_lane, int* out_warp_in_block,
                             int* out_global_warp) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    int lane           = threadIdx.x % 32;            // 线程在 warp 内编号 0..31
    int warp_in_block  = threadIdx.x / 32;            // block 内第几个 warp
    int global_warp    = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // 全局 warp 序
    out_lane[g]          = lane;
    out_warp_in_block[g] = warp_in_block;
    out_global_warp[g]   = global_warp;
}

// ============================================================================
// 演示二：分支发散（branch divergence）—— 同一 warp 走不同路径 = 两条路径串行
// ----------------------------------------------------------------------------
// 若 warp 内一部分 lane 走 if、另一部分走 else，硬件不能并行两条路径，而是
// 「先 masked 执行 true 分支(假分支的 lane 被屏蔽)，再 masked 执行 false 分支」。
// 于是该 warp 耗时 = 两条路径耗时之和（不是取 max！），这就是性能黑洞。
//
// 教学难点：发散是"性能"问题，沙箱无法计时。本文件用 __ballot_sync 把
// "warp 内谓词混合情况"显影成数字，使发散可被校验而无需计时：
//   popc(__ballot_sync(mask, p)) = 该 warp 内 p 为真的 lane 数。
//   - 若每 warp 内部 p 半真半假 → popc≈16 → 一旦以 p 分支就会发散。
//   - 若每 warp 内部 p 全真或全假(popc=32/0) → 不会发散。
// ============================================================================

// ❌ 发散版：谓词依赖 lane(线程在 warp 内编号) → 一个 warp 内偶数 lane 真、奇数假
__global__ void diverge_by_lane(int n, int* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    bool p = (threadIdx.x % 2 == 0);                  // 同一 warp 内 lane 0,2,.. 真；1,3,.. 假
    unsigned mask = __ballot_sync(0xffffffff, p);      // 此时 32 lane 全活跃，统计真值数
    int popc = (int)__popc(mask);                     // 本 warp 应数出 16 个真
    // —— 下面才是真正"会发散"的计算：两条代价不同的路径 ——
    int val = 0;
    if (p) { for (int k = 0; k < 100; ++k) val += k; } // 偶数 lane 走重路径
    else   { val = 1; }                                // 奇数 lane 走轻路径
    out[g] = popc * 1000 + val;                        // 高位记发散模式，低位记分支结果
}

// ✅ 不发散版：谓词按全局序号对齐到 warp 边界(g < n/2) → 一个 warp 全真或全假
__global__ void coherent_by_warp(int n, int* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    bool p = (g < n / 2);                              // n/2 是 warp(32) 的整数倍 → 不对齐问题
    unsigned mask = __ballot_sync(0xffffffff, p);
    int popc = (int)__popc(mask);                     // 整 warp 真→32；整 warp 假→0
    int val = 0;
    if (p) { for (int k = 0; k < 100; ++k) val += k; }
    else   { val = 1; }
    out[g] = popc * 1000 + val;
}

// ============================================================================
// 演示三：occupancy（占用率）—— 一个 SM 上能同时驻留多少 warp
// ----------------------------------------------------------------------------
// occupancy = 活跃 warp 数 / SM 最大 warp 数。它受三道天花板里最紧的那个限制：
//   ① 每线程寄存器数 × 每 block 线程数 × block 数 ≤ SM 寄存器文件总量
//   ② 每 block 用 shared mem × block 数 ≤ SM 共享内存总量
//   ③ 每 block 线程数 / 每 SM 最大线程数 / 每 SM 最大 block 数
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM) 是给编译器的"硬约束"：
// 要求每 SM 至少塞下 minBlocksPerSM 个 block → 编译器被迫把每线程寄存器压到
// (寄存器文件 / (minBlocksPerSM × maxThreadsPerBlock)) 以内，从而换得更高 occupancy。
// ============================================================================

// 无 launch_bounds：编译器可自由使用寄存器 → 可能很多 → 每 SM 驻留 block 少
__global__ void occ_free(int n, float* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float acc = 0;
    for (int k = 0; k < 64; ++k) acc += __sinf((float)(g + k));
    out[g] = acc;
}

// 带 launch_bounds(256, 4)：要求每 SM 至少 4 个 block → 寄存器被压到 4×256 能容纳
__global__ void __launch_bounds__(256, 4) occ_bounded(int n, float* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float acc = 0;
    for (int k = 0; k < 64; ++k) acc += __sinf((float)(g + k));
    out[g] = acc;
}

// ============================================================================
// 演示四：寄存器压力（register pressure）—— 寄存器不够就溢出到 local memory(= 慢!)
// ----------------------------------------------------------------------------
// 每线程可用的寄存器有限（SM 寄存器文件 / 同时驻留线程数）。当 kernel 局部变量
// 太多、编译器不够用时会**溢出(spill)到 local memory**——而 local memory 其实在
// global 显存里！spill 一来，原本应极快的寄存器访问变成显存往返，性能暴跌。
// 高寄存器压力还会压低 occupancy（见演示三），双重打击。
// 用 __launch_bounds__(256, 8) 强迫编译器把每线程寄存器压到更低，避免 spill。
// ============================================================================

// ❌ 高压力：8 个独立累加器 + 循环 → 逼编译器多用寄存器，可能 spill 到 local mem
__global__ void high_pressure(int n, float* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int k = 0; k < 32; ++k) {
        a0 += __sinf((float)(g   + k));
        a1 += __cosf((float)(g   + k));
        a2 += __sinf((float)(g*2 + k));
        a3 += __cosf((float)(g*2 + k));
        a4 += __sinf((float)(g*3 + k));
        a5 += __cosf((float)(g*3 + k));
        a6 += __sinf((float)(g*4 + k));
        a7 += __cosf((float)(g*4 + k));
    }
    out[g] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// ✅ 低压力：同样计算，但 __launch_bounds__(256,8) 逼编译器把每线程寄存器压低
__global__ void __launch_bounds__(256, 8) low_pressure(int n, float* out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int k = 0; k < 32; ++k) {
        a0 += __sinf((float)(g   + k));
        a1 += __cosf((float)(g   + k));
        a2 += __sinf((float)(g*2 + k));
        a3 += __cosf((float)(g*2 + k));
        a4 += __sinf((float)(g*3 + k));
        a5 += __cosf((float)(g*3 + k));
        a6 += __sinf((float)(g*4 + k));
        a7 += __cosf((float)(g*4 + k));
    }
    out[g] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// CPU 参考：与 high/low_pressure 完全相同的算术（用 host 版 sinf/cosf）
static float cpu_ref(int g) {
    float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (int k = 0; k < 32; ++k) {
        a0 += sinf((float)(g   + k));
        a1 += cosf((float)(g   + k));
        a2 += sinf((float)(g*2 + k));
        a3 += cosf((float)(g*2 + k));
        a4 += sinf((float)(g*3 + k));
        a5 += cosf((float)(g*3 + k));
        a6 += sinf((float)(g*4 + k));
        a7 += cosf((float)(g*4 + k));
    }
    return a0+a1+a2+a3+a4+a5+a6+a7;
}

// ============================================================================
// host main（教材级：真机上跑会执行；此处逻辑与真机一致）
// ============================================================================
int main() {
    // ---------------- 演示一：warp / lane 结构 ----------------
    {
        const int N = 1024;                 // 1024 线程 = 4 block × 256 = 32 warps
        const int BS = 256;
        int *d_lane, *d_wib, *d_gw;
        CUDA_CHECK(cudaMalloc(&d_lane, N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_wib,  N*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gw,   N*sizeof(int)));
        inspect_warp<<<N/BS, BS>>>(N, d_lane, d_wib, d_gw);
        CUDA_CHECK(cudaGetLastError());
        std::vector<int> h_lane(N), h_wib(N), h_gw(N);
        CUDA_CHECK(cudaMemcpy(h_lane.data(), d_lane, N*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_wib.data(),  d_wib,  N*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gw.data(),   d_gw,   N*sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int b = 0; b < N/BS; ++b)
            for (int t = 0; t < BS; ++t) {
                int g = b*BS + t;
                if (h_lane[g] != (t % 32)) ok = false;
                if (h_wib[g]  != (t / 32)) ok = false;
                if (h_gw[g]   != (b*BS + t) / 32) ok = false;
            }
        printf("[1] warp 结构: %s (lane=t%%32, warp_in_block=t/32, global_warp 连续)\n", ok?"PASS":"FAIL");
        CUDA_CHECK(cudaFree(d_lane)); CUDA_CHECK(cudaFree(d_wib)); CUDA_CHECK(cudaFree(d_gw));
    }

    // ---------------- 演示二：分支发散（用 ballot 显影） ----------------
    {
        const int N = 256, BS = 64;          // 4 block × 64 = 8 warp
        int *d_out; CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(int)));
        std::vector<int> h(N);

        diverge_by_lane<<<N/BS, BS>>>(N, d_out); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, N*sizeof(int), cudaMemcpyDeviceToHost));
        bool okDiv = true;
        for (int g = 0; g < N; ++g) {
            int popc = h[g] / 1000;          // 高位：该 warp 内真值 lane 数
            int val  = h[g] % 1000;          // 低位：分支结果
            if (popc != 16) okDiv = false;   // 每 warp 半真半假 → 会发散
            bool even = ((g % BS) % 2 == 0); // 注意谓词按 threadIdx.x
            int expectVal = even ? (99*100/2) : 1;   // 重路径 sum 0..99=4950
            if (val != expectVal) okDiv = false;
        }
        printf("[2a] 分支发散(by lane): %s (popc=16 ⇒ warp 内半真半假, 分支必发散)\n", okDiv?"PASS":"FAIL");

        coherent_by_warp<<<N/BS, BS>>>(N, d_out); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, N*sizeof(int), cudaMemcpyDeviceToHost));
        bool okCoh = true;
        for (int g = 0; g < N; ++g) {
            int popc = h[g] / 1000;
            int val  = h[g] % 1000;
            if (g < N/2) { if (popc != 32 || val != 4950) okCoh = false; }   // 全真 warp
            else         { if (popc != 0  || val != 1)    okCoh = false; }   // 全假 warp
        }
        printf("[2b] 不发散(by warp 对齐): %s (popc=32 或 0 ⇒ 整 warp 一致, 不发散)\n", okCoh?"PASS":"FAIL");
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------------- 演示三：occupancy（真机用 Occupancy API 量化） ----------------
    {
        int maxThreadsPerSM = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerSM,
                                         cudaDevAttrMaxThreadsPerMultiProcessor, 0));
        int maxWarps = maxThreadsPerSM / 32;
        int nbFree = 0, nbBound = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nbFree,  (const void*)occ_free,   256, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &nbBound, (const void*)occ_bounded, 256, 0));
        float occFree  = (float)(nbFree  * 256) / (maxWarps * 32);
        float occBound = (float)(nbBound * 256) / (maxWarps * 32);
        printf("[3] occupancy: occ_free=%.2f (blocks/SM=%d)  occ_bounded=%.2f (blocks/SM=%d)  [maxWarps/SM=%d]\n",
               occFree, nbFree, occBound, nbBound, maxWarps);
        printf("    （bounded 因 __launch_bounds__(256,4) 压寄存器，期望 occupancy ≥ free）\n");
    }

    // ---------------- 演示四：寄存器压力（结果一致，用量看 --ptxas-options=-v） ----------------
    {
        const int N = 1 << 14;
        std::vector<float> ref(N); for (int g = 0; g < N; ++g) ref[g] = cpu_ref(g);
        float *d_out; CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(float)));

        high_pressure<<<(N+255)/256, 256>>>(N, d_out); CUDA_CHECK(cudaGetLastError());
        std::vector<float> h(N); CUDA_CHECK(cudaMemcpy(h.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost));
        bool okHi = true; for (int g = 0; g < N; ++g) if (fabsf(h[g]-ref[g]) > 1e-2f) { okHi=false; break; }
        printf("[4a] high_pressure: %s (结果一致；寄存器多, 可能 spill 到 local mem)\n", okHi?"PASS":"FAIL");

        low_pressure<<<(N+255)/256, 256>>>(N, d_out); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost));
        bool okLo = true; for (int g = 0; g < N; ++g) if (fabsf(h[g]-ref[g]) > 1e-2f) { okLo=false; break; }
        printf("[4b] low_pressure(__launch_bounds__ 256,8): %s (结果一致；寄存器被压低, 少 spill)\n", okLo?"PASS":"FAIL");
        CUDA_CHECK(cudaFree(d_out));
        printf("    → 编译加 --ptxas-options=-v，对比 high/low 的 'registers' 字段即见压力差。\n");
    }

    printf("\n全部演示结果校验完成。性能差异需用 Nsight / --ptxas-options=-v 量化。\n");
    return 0;
}

// ============================================================================
// OpenCL 对照（概念一一对应，差异仅在 API）
// ----------------------------------------------------------------------------
// • warp(32 线程锁步)      ↔  wavefront / sub-group（AMD 叫 wavefront=64，Intel=32；
//                              OpenCL 2.0+ 有 sub-group 概念，等价于 warp）
// • lane_id / __ballot_sync ↔  sub_group_ballot / intel_sub_group 内建（部分扩展）
// • 分支发散                ↔  同样存在：sub-group 内谓词混合 → 两条路径串行
// • __launch_bounds__       ↔  无直接等价；靠 work-group size 与编译选项(-cl-opt)间接控制
// • occupancy               ↔  概念同：compute unit 上能驻留多少 wavefront，
//                             受寄存器(=private mem)/local mem/thread 数限制
// • 寄存器压力 / spill      ↔  private memory 溢出到 global（OpenCL 也叫"private 溢出"）
// • __sinf/__cosf           ↔  native_cos/native_sin（fast math 内建）
// • __syncthreads()         ↔  barrier(CLK_LOCAL_MEM_FENCE) / sub_group_barrier
// ============================================================================
