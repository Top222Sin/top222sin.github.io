// ============================================================================
// L2.4 自己写高效 kernel —— 共享内存归约 reduction / 矩阵乘分块 tiling
// 配套文档：cuda_l2_reduction_tiling.md
// 对应笔记本：01CUDA基础学习.docx → L2 能写高效 kernel → 4、共享内存归约、矩阵乘分块
// 编译（需 NVIDIA GPU + CUDA Toolkit）：
//     nvcc -O3 -std=c++17 cuda_l2_reduction_tiling.cpp -o rt && ./rt
// 沙箱无 GPU/nvcc，本文件为教材级标准代码；本文件用 analyze_*() 在 CPU 上
// 确定性量化"shared 带来的收益"，真机段需 GPU 才能跑。
// ----------------------------------------------------------------------------
// 这是 L2 的收官：把 L2.1(shared/bank/pad)、L2.2(occupancy)、L2.3(减 DRAM 流量)
// 一次性串起来——第一个"真正高性能"的手写 kernel。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            return 1;                                                          \
        }                                                                       \
    } while (0)

// ============================================================================
// 第一部分：归约 reduction（求和所有 N 个元素）
// ----------------------------------------------------------------------------
// 朴素错误做法：① 单线程循环（无并行）② atomicAdd 全局求和（所有线程串行抢锁）。
// 正解：每个 block 把一段数据搬进 shared，在 block 内做"树形归约"，
//       每个 block 产出一个 partial，最后再合并 partials。
// 关键坑（呼应 L2.1）：shared 归约在某些 stride 会撞 bank conflict；
//       现代最佳实践是"末段改用 warp shuffle（__shfl_down_sync）彻底无冲突"。
// ============================================================================

// ---- ① 交错寻址归约：经典写法，但在 stride=32 这一步会撞 2 路 bank conflict ----
__global__ void reduce_halving(const float* in, float* partials, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    s[tid] = (gid < n) ? in[gid] : 0.f;
    __syncthreads();
    // 交错寻址：stride 减半，线程 tid 累加 s[tid]+s[tid+stride]
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = s[0];
    // 注：stride=32 时 s[tid] 与 s[tid+32] 落在同一 bank → 2 路冲突（详见 md）
}

// ---- ② 现代最佳实践：树形归约末段改用 warp shuffle，完全避免 bank conflict ----
__device__ float warp_reduce(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__global__ void reduce_warp_shuffle(const float* in, float* partials, int n) {
    __shared__ float warp_sums[256 / 32];   // 8 个 warp，每 warp 一个和
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = (gid < n) ? in[gid] : 0.f;

    int lane = tid % 32;
    int wid  = tid / 32;
    float ws = warp_reduce(v);              // 每 warp 在 lane0 得到本 warp 和
    if (lane == 0) warp_sums[wid] = ws;
    __syncthreads();

    // 第 0 个 warp 把 8 个 warp 和再归约一次
    if (wid == 0) {
        float x = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.f;
        x = warp_reduce(x);
        if (lane == 0) partials[blockIdx.x] = x;
    }
}

// ============================================================================
// 第二部分：矩阵乘分块 tiling（C = A × B，行主序）
// ----------------------------------------------------------------------------
// 朴素：每线程算一个 C[i][j] = Σ_k A[i][k]·B[k][j]，A 的行与 B 的列被重复读 K 次
//       → 全局流量爆炸（呼应 L2.3：memory-bound 的根因就是冗余的 global 取数）。
// 分块：把 A、B 切成 TILE×TILE 小块搬进 shared，每个 block 复用 shared 中的小块
//       算出 C 的一个 TILE——每个全局元素只取 1 次，被 TILE×TILE 个线程复用。
// 关键坑（呼应 L2.1）：shared 里读 B 小块时按列访问会撞 bank conflict；
//       解决：把 B 小块**转置**存入 shared，使读取变成行连续（无冲突）。
// ============================================================================

// ---- ③ 朴素矩阵乘（每线程一个输出，全局流量最大）----
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // 行
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // 列
    if (i < M && j < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
        C[i * N + j] = acc;
    }
}

// ---- ④ 分块矩阵乘（B 正常存 shared，inner loop 读 b_tile 有 2 路 bank conflict）----
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && (t * TILE + tx) < K) a_tile[ty][tx] = A[row * K + t * TILE + tx];
        else                                a_tile[ty][tx] = 0.f;
        if ((t * TILE + ty) < K && col < N) b_tile[ty][tx] = B[(t * TILE + ty) * N + col];
        else                                b_tile[ty][tx] = 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += a_tile[ty][k] * b_tile[k][tx];
        //                                   ↑ b_tile[k][tx]：跨 warp 读同一列 → 2 路 bank conflict
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- ⑤ 分块矩阵乘（B 转置存入 shared，读 b_tile 变行连续 → 无 bank conflict）----
__global__ void matmul_tiled_transposed(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];   // 存 B 的转置：b_tile[tx][ty] = B 原 (ty,tx)
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        if (row < M && (t * TILE + tx) < K) a_tile[ty][tx] = A[row * K + t * TILE + tx];
        else                                a_tile[ty][tx] = 0.f;
        // 转置存：global 读 B[(t*TILE+ty)*N+col]，但存到 b_tile[tx][ty]
        if ((t * TILE + ty) < K && col < N) b_tile[tx][ty] = B[(t * TILE + ty) * N + col];
        else                                b_tile[tx][ty] = 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += a_tile[ty][k] * b_tile[tx][k];
        //                                   ↑ b_tile[tx][k]：行连续 → 无 bank conflict
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ============================================================================
// CPU 参考（用于对拍正确性）
// ============================================================================
static float cpu_reduce(const float* a, int n) {
    double s = 0.0; for (int i = 0; i < n; ++i) s += a[i]; return (float)s;
}
static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// ============================================================================
// (A) 确定性分析：shared 把"全局流量"打下来多少（CPU 可跑，沙箱也能教学）
// ============================================================================
static void analyze_tiling() {
    printf("==================== (A) 确定性分析：tiling 砍掉多少全局流量 ====================\n");
    const int S = 256;   // 方阵 M=N=K=S
    // 朴素：每个 C[i][j] 各读 K 个 A、K 个 B → 全局元素取数次数
    long long naive_fetches = 2LL * S * S * S + 1LL * S * S;          // 2MNK + MN
    // 分块：A、B 每个元素只搬进 shared 一次 → 取数 M*K + K*N + 写 M*N
    long long tiled_fetches = 1LL * S * S + 1LL * S * S + 1LL * S * S; // MK + KN + MN
    double factor = (double)naive_fetches / tiled_fetches;
    printf("方阵 S=%d（M=N=K=%d）：\n", S, S);
    printf("  朴素 matmul 全局取数次数 ≈ %lld 个元素\n", naive_fetches);
    printf("  分块 matmul 全局取数次数 ≈ %lld 个元素（每个元素仅进 shared 一次）\n", tiled_fetches);
    printf("  ▶ 全局流量下降 ≈ %.1f× （FLOPs 不变，纯靠 shared 复用）\n", factor);
    printf("\n要点（呼应 L2.1/L2.2/L2.3）：\n");
    printf("  · 归约：shared 让 block 内并行树形归约 + 合并全局访存，避开 atomic 串行；\n");
    printf("    AI≈0.5 仍属 memory-bound，但延迟/吞吐被 shared + 并行彻底改善（不是改 AI）。\n");
    printf("  · 矩阵乘：tiling 的本质是『数据复用砍冗余全局流量』——AI 不变，\n");
    printf("    但 DRAM 取数降 %.0f×，这正是 L2.3 说的『memory-bound → 减 DRAM 流量』。\n", factor);
    printf("  · 两者都用 shared，务必算 bank 偏移（pad 或转置）防冲突，否则白忙（L2.1）。\n");
    printf("  · 适度 shared 用量 = 好的 occupancy（L2.2），sm__throughput 才能上去。\n");
    printf("=================================================================================\n\n");
}

// ============================================================================
// (B) 真机 GPU 校验（需 CUDA Toolkit；沙箱里这段不会被执行）
// ============================================================================
static int run_gpu_checks() {
    // ---- reduction 校验 ----
    const int N = 1 << 20;
    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_part = (float*)malloc(GRID * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = (float)(i % 7 - 3);   // 含负数，便于发现错误
    float *d_in, *d_part;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_part, GRID * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    reduce_halving<<<GRID, BLOCK>>>(d_in, d_part, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_part, d_part, GRID * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_h = 0.f; for (int i = 0; i < GRID; ++i) sum_h += h_part[i];

    reduce_warp_shuffle<<<GRID, BLOCK>>>(d_in, d_part, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_part, d_part, GRID * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_s = 0.f; for (int i = 0; i < GRID; ++i) sum_s += h_part[i];

    float ref = cpu_reduce(h_in, N);
    printf("[GPU] reduce_halving      = %.3f  (ref %.3f) %s\n", sum_h, ref, fabsf(sum_h - ref) < 1e-2f ? "PASS" : "FAIL");
    printf("[GPU] reduce_warp_shuffle = %.3f  (ref %.3f) %s\n", sum_s, ref, fabsf(sum_s - ref) < 1e-2f ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_part);

    // ---- matmul 校验（S=128，便于真机快速跑）----
    const int S = 128;
    float *hA = (float*)malloc(S * S * sizeof(float));
    float *hB = (float*)malloc(S * S * sizeof(float));
    float *hCref = (float*)malloc(S * S * sizeof(float));
    float *hC1 = (float*)malloc(S * S * sizeof(float));
    float *hC2 = (float*)malloc(S * S * sizeof(float));
    float *hC3 = (float*)malloc(S * S * sizeof(float));
    for (int i = 0; i < S * S; ++i) { hA[i] = (float)((i % 13) - 6); hB[i] = (float)((i % 11) - 5); }
    cpu_matmul(hA, hB, hCref, S, S, S);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, S * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, S * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, S * S * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, S * S * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, S * S * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 gridNaive((S + block.x - 1) / block.x, (S + block.y - 1) / block.y);
    dim3 gridTile((S + TILE - 1) / TILE, (S + TILE - 1) / TILE);
    matmul_naive<<<gridNaive, block>>>(dA, dB, dC, S, S, S);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(hC1, dC, S*S*sizeof(float), cudaMemcpyDeviceToHost));
    matmul_tiled<<<gridTile, block>>>(dA, dB, dC, S, S, S);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(hC2, dC, S*S*sizeof(float), cudaMemcpyDeviceToHost));
    matmul_tiled_transposed<<<gridTile, block>>>(dA, dB, dC, S, S, S);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(hC3, dC, S*S*sizeof(float), cudaMemcpyDeviceToHost));

    auto cmp = [&](const float* g, const char* name) {
        int ok = 1; for (int i = 0; i < S * S; ++i) if (fabsf(g[i] - hCref[i]) > 1e-2f) { ok = 0; break; }
        printf("[GPU] %-24s %s\n", name, ok ? "PASS" : "FAIL");
    };
    cmp(hC1, "matmul_naive");
    cmp(hC2, "matmul_tiled");
    cmp(hC3, "matmul_tiled_transposed");

    free(h_in); free(h_part); free(hA); free(hB); free(hCref); free(hC1); free(hC2); free(hC3);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

// ============================================================================
int main() {
    analyze_tiling();       // (A) 确定性：tiling 砍全局流量（CPU 可跑，沙箱也能教学）
    printf("[提示] 真机运行请取消 main 末尾 run_gpu_checks() 的注释，用：\n");
    printf("        nvcc -O3 -std=c++17 cuda_l2_reduction_tiling.cpp -o rt && ./rt\n");
    // return run_gpu_checks();   // 需真机：归约 + 三种 matmul 均对拍 PASS
    return 0;
}
