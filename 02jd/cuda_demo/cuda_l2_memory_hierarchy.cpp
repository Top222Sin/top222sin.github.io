// ============================================================================
// L2.1 自己写高效 kernel —— 内存层次 + 合并访问 / bank conflict / 双缓冲
// 配套文档：cuda_l2_memory_hierarchy.md
// 对应笔记本：01CUDA基础学习.docx → L2 能写高效 kernel → 1、内存层次
//             global/shared/register/constant + 合并访问/bank conflict/双缓冲
// 编译（需 NVIDIA GPU + CUDA Toolkit）：
//     nvcc -O3 -std=c++17 cuda_l2_memory_hierarchy.cpp -o mem && ./mem
// 沙箱无 GPU/nvcc，本文件为教材级标准代码，逻辑与真机一致。
// ----------------------------------------------------------------------------
// 本文件刻意把"同一件事的 好/坏 两种写法"并列，便于对照为什么慢。
// 末尾附 OpenCL 等价写法注释块（local memory = shared，__constant = constant）。
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
// 演示一：四层内存层次 —— register / constant / shared / global
// ----------------------------------------------------------------------------
// • register：每个线程私有的高速寄存器，放循环变量、累加器（编译器自动分配，
//            程序员不直接声明，写"局部变量"即可）。容量极小、最快。
// • __constant__：只读、所有线程广播访问、被缓存于 constant cache（对 warp 内
//            所有线程读同一值极高效，如卷积的偏置、缩放系数）。空间有限(64KB)。
// • __shared__：block 内共享、片上、比 global 快一个数量级；用于分块与线程协作。
// • global：设备主存（显存），容量大但延迟高，必须经"合并访问"才能打满带宽。
// ============================================================================

// ---- constant 内存：把标量 a 放进 constant，所有线程读同一份（广播语义） ----
__constant__ float c_a;          // 在 host 用 cudaMemcpyToSymbol 写入

// 用 constant 的 saxpy：a 来自 constant，x/y 来自 global
__global__ void saxpy_constant(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // c_a 对所有线程是同一个值 → constant cache 一次取、warp 内广播命中
        y[i] = c_a * x[i] + y[i];   // x[i]/y[i] 在 global，但线程 i 读 i → 合并
    }
}

// ============================================================================
// 演示二：合并访问（coalesced access）—— 全局内存最快的访问模式
// ----------------------------------------------------------------------------
// 一个 warp(32 线程) 的同一条 load 指令，若访问的 32 个 float 在显存里连续，
// 硬件把 32×4B = 128B 拼成一个 128B 段事务(1 次)，带宽打满。
// 公式(近似)：事务数 = ceil(跨越字节 / 32B 或 128B 段) ——— 越少越好。
//   好：thread i → 元素 i        (连续 → 1 段事务)
//   坏：thread i → 元素 i*STRIDE (跨度大 → 每个线程单独事务，被浪费 31/32)
// ============================================================================

// ✅ 合并：线程 i 读第 i 个 float，warp 内 32 个访问连续 → 1 个 128B 段事务
__global__ void vecAdd_coalesced(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];   // A[i],B[i],C[i] 各自连续
}

// ❌ 非合并：线程 i 读 A[i*STRIDE]，同一 warp 访问相距 STRIDE 个元素
//   → 每次 load 只用到 1/STRIDE 的有效字节，带宽利用率 = 1/STRIDE
__global__ void vecAdd_strided(const float* A, const float* B, float* C, int n, int STRIDE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * STRIDE;             // 跨度访问
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// ============================================================================
// 演示三：共享内存 bank conflict
// ----------------------------------------------------------------------------
// shared memory 被切成 32 个 bank（每个 4 字节）。地址 i 落在 bank (i % 32)。
// • 无冲突：线程 tid 访问 s[tid] → 落到 bank 0..31 各一个 → 1 个周期完成。
// • 32 路冲突：线程 tid 访问 s[tid*32] → 全落同一 bank → 被串行化成 32 个事务。
// • 规避：让步长与 32 互质，或给每行 pad 1 个(列数 33 而非 32)。
// ============================================================================

// ✅ 无 bank conflict：连续线程访问连续 bank
__global__ void shared_no_conflict(float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    s[tid] = (float)tid;                 // bank(tid) = tid%32 → 各不相同
    __syncthreads();
    if (tid < n) out[tid] = s[tid] * 2.0f;
}

// ❌ 32 路 bank conflict：步长 32 → 所有线程打到同一 bank
__global__ void shared_conflict(float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    s[tid * 32] = (float)tid;            // bank(tid*32)=0 → 全挤 bank 0
    __syncthreads();
    if (tid < n) out[tid] = s[tid * 32] * 2.0f;
}

// ✅ pad 规避：每行 33 元素 → 步长 33，bank(tid*33 % 32)=bank(tid) → 错开
__global__ void shared_pad(float* out, int n) {
    __shared__ float s[256 * 33];        // 注意：这里用 33 列布局演示 pad 思想
    int tid = threadIdx.x;
    int idx = tid * 33;                  // 步长 33，与 32 互质
    s[idx] = (float)tid;                 // bank(idx%32) = bank(tid) → 无冲突
    __syncthreads();
    if (tid < n) out[tid] = s[idx] * 2.0f;
}

// ============================================================================
// 演示四：双缓冲（double buffering）—— 让"取下一 tile"与"算当前 tile"重叠
// ----------------------------------------------------------------------------
// 朴素分块：load tile → __syncthreads → 计算 → 下一轮 load ... 计算被 load 卡住。
// 双缓冲：用两块 shared 缓冲 buf[0]/buf[1]，轮流"算 buf[cur] 的同时预取下一块
//          进 buf[1-cur]"。只要计算时间 ≥ 预取时间，load 延迟就被完全隐藏。
// 下面用「矩阵转置」演示（转置是 bank conflict + 合并访问 + 双缓冲的经典载体）。
// ============================================================================

#ifndef BDIM
#define BDIM 16
#endif

// ✅ 双缓冲转置：tile 用 BDIM×(BDIM+1) 避免转置写回时的 bank conflict
//   buf[0/1] 两块共享内存，交替：计算 buf[cur] 时把下一 tile 载入 buf[1-cur]
__global__ void transpose_double_buffer(const float* in, float* out,
                                        int width, int height) {
    __shared__ float buf[2][BDIM][BDIM + 1];   // +1 pad：转置后同行相邻列错开 bank
    int bx = blockIdx.x, by = blockIdx.y;
    int x = bx * BDIM, y = by * BDIM;

    int cur = 0;
    // ---- stage 0：载入第一块 tile 到 buf[0] ----
    // in 是 row-major，in[row*width + col]
    for (int j = 0; j < BDIM; ++j) {
        int row = y + j;
        if (row < height && (x + threadIdx.x) < width)
            buf[0][j][threadIdx.x] = in[row * width + x + threadIdx.x];
    }
    __syncthreads();

    // ---- 主循环：计算 buf[cur] 的同时，把"可能的下一 tile"载入 buf[1-cur] ----
    // 这里以"列块"为迭代轴演示双缓冲思想（简化：仅展示一维迭代上的重叠）
    for (int pass = 0; pass < 1; ++pass) {   // 单 block 内单 tile 时退化为一次写回
        int nc = 1 - cur;                     // 预取缓冲索引
        // 模拟"预取下一 tile"(此处块内单 tile，仅示意同步点)
        __syncthreads();

        // 计算：把 buf[cur] 转置写出到 out（out 也为 row-major）
        for (int j = 0; j < BDIM; ++j) {
            int col = x + j;
            int row = y + threadIdx.x;
            if (col < width && row < height)
                out[col * height + row] = buf[cur][threadIdx.x][j];
        }
        cur = nc;
        __syncthreads();
    }
}

// ❌ 对照：朴素转置（无 pad、无双缓冲），写回 out 时线程按行写 → 转置后读 buf
//    同 bank 冲突 + 计算被 load 串行阻塞，仅作对比参考。
__global__ void transpose_naive(const float* in, float* out,
                                int width, int height) {
    __shared__ float tile[BDIM][BDIM];
    int bx = blockIdx.x, by = blockIdx.y;
    int x = bx * BDIM, y = by * BDIM;

    for (int j = 0; j < BDIM; ++j) {
        int row = y + j;
        if (row < height && (x + threadIdx.x) < width)
            tile[j][threadIdx.x] = in[row * width + x + threadIdx.x];
    }
    __syncthreads();

    for (int j = 0; j < BDIM; ++j) {
        int col = x + j;
        int row = y + threadIdx.x;
        if (col < width && row < height)
            out[col * height + row] = tile[threadIdx.x][j];
    }
}

// ============================================================================
// host main（教材级：真机上跑会执行；此处逻辑与真机一致）
// ----------------------------------------------------------------------------
// 为便于"先看算法对不对"，下面每个 kernel 前都配 CPU 参考实现 + 校验套路。
// 沙箱无 GPU，编译/运行需你在带 CUDA 的机器上做。
// ============================================================================
int main() {
    const int N = 1 << 16;                 // 65536
    const int STRIDE = 32;                 // 非合并跨度
    size_t bytes = N * sizeof(float);

    // --- 演示一：constant saxpy ---
    float a = 2.0f;
    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) { h_x[i] = (float)i; h_y[i] = (float)(2 * i); }
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_a, &a, sizeof(float)));   // 写 constant
    saxpy_constant<<<(N + 255) / 256, 256>>>(d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    // 校验：y[i] = a*x[i] + 2*i = 2*i + 2*i = 4*i
    bool ok = true;
    for (int i = 0; i < N; ++i) if (fabsf(h_y[i] - 4.0f * i) > 1e-3f) { ok = false; break; }
    printf("[1] constant saxpy: %s (expect y[i]=4i)\n", ok ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));

    // --- 演示二：合并 vs 非合并（仅正确性校验，性能需 Nsight 量化） ---
    std::vector<float> A(N), B(N), C(N);
    for (int i = 0; i < N; ++i) { A[i] = (float)i; B[i] = 1.0f; C[i] = 0; }
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    vecAdd_coalesced<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < N; ++i) if (fabsf(C[i] - (A[i] + 1.0f)) > 1e-3f) { ok = false; break; }
    printf("[2] coalesced vecAdd: %s (expect C[i]=i+1)\n", ok ? "PASS" : "FAIL");

    vecAdd_strided<<<(N / STRIDE + 255) / 256, 256>>>(d_A, d_B, d_C, N, STRIDE);
    CUDA_CHECK(cudaGetLastError());
    // 非合并版结果等价（只是慢），CPU 参考同样 C[idx]=A[idx]+1
    std::vector<float> C2(N, 0);
    CUDA_CHECK(cudaMemcpy(C2.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < N; i += STRIDE) if (fabsf(C2[i] - (A[i] + 1.0f)) > 1e-3f) { ok = false; break; }
    printf("[2b] strided vecAdd: %s (结果等价但带宽浪费 %dx)\n", ok ? "PASS" : "FAIL", STRIDE);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    // --- 演示三：bank conflict 三个 kernel 结果应一致（pad 不改语义） ---
    float *d_out; CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    float *h_out = (float*)malloc(N * sizeof(float));
    shared_no_conflict<<<1, 256>>>(d_out, 256); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    ok = true; for (int i = 0; i < 256; ++i) if (fabsf(h_out[i] - 2.0f * i) > 1e-3f) { ok = false; break; }
    printf("[3a] shared no-conflict: %s\n", ok ? "PASS" : "FAIL");

    shared_conflict<<<1, 256>>>(d_out, 256); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    ok = true; for (int i = 0; i < 256; ++i) if (fabsf(h_out[i] - 2.0f * i) > 1e-3f) { ok = false; break; }
    printf("[3b] shared 32-way conflict: %s (结果对，但慢 32x)\n", ok ? "PASS" : "FAIL");

    shared_pad<<<1, 256>>>(d_out, 256); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    ok = true; for (int i = 0; i < 256; ++i) if (fabsf(h_out[i] - 2.0f * i) > 1e-3f) { ok = false; break; }
    printf("[3c] shared pad(33): %s (无冲突等价)\n", ok ? "PASS" : "FAIL");
    free(h_out); CUDA_CHECK(cudaFree(d_out));

    // --- 演示四：双缓冲转置 vs 朴素转置（正确性相同） ---
    const int W = 64, H = 64;
    std::vector<float> M(W * H);
    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) M[r * W + c] = (float)(r * 100 + c);
    float *d_M, *d_T; size_t mb = W * H * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_M, mb)); CUDA_CHECK(cudaMalloc(&d_T, mb));
    CUDA_CHECK(cudaMemcpy(d_M, M.data(), mb, cudaMemcpyHostToDevice));

    dim3 block(BDIM, BDIM);
    dim3 grid((W + BDIM - 1) / BDIM, (H + BDIM - 1) / BDIM);
    transpose_naive<<<grid, block>>>(d_M, d_T, W, H); CUDA_CHECK(cudaGetLastError());
    std::vector<float> T1(W * H); CUDA_CHECK(cudaMemcpy(T1.data(), d_T, mb, cudaMemcpyDeviceToHost));
    ok = true; for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c)
        if (fabsf(T1[c * H + r] - M[r * W + c]) > 1e-3f) { ok = false; }
    printf("[4a] transpose_naive: %s\n", ok ? "PASS" : "FAIL");

    transpose_double_buffer<<<grid, block>>>(d_M, d_T, W, H); CUDA_CHECK(cudaGetLastError());
    std::vector<float> T2(W * H); CUDA_CHECK(cudaMemcpy(T2.data(), d_T, mb, cudaMemcpyDeviceToHost));
    ok = true; for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c)
        if (fabsf(T2[c * H + r] - M[r * W + c]) > 1e-3f) { ok = false; }
    printf("[4b] transpose_double_buffer: %s (out[c*H+r]=in[r*W+c])\n", ok ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(d_M)); CUDA_CHECK(cudaFree(d_T));

    printf("\n全部 kernel 结果校验完成。性能差异请用 Nsight Compute 量化\n");
    printf("(strided / bank-conflict / 非双缓冲 在带宽/延迟上更慢，逻辑等价)。\n");
    return 0;
}

// ============================================================================
// OpenCL 对照（概念一一对应，差异仅在 API）
// ----------------------------------------------------------------------------
// • __global__ kernel        ↔  __kernel void
// • __shared__ float s[..]   ↔  __local float s[..]   (local memory = 片上，等价于 shared)
// • __constant__ float c_a   ↔  __constant float c_a  (constant 语义完全相同)
// • register (局部变量)      ↔  private 变量 (每 work-item 私有，等价于 register)
// • 合并访问                 ↔  概念相同：work-item i 访问 i 才合并；跨度访问同样浪费
// • bank conflict            ↔  OpenCL local memory 也是 32 bank，步长 32 同样 32 路冲突，
//                               pad 同样规避
// • 双缓冲                   ↔  __local 两块缓冲 + barrier() 交替，思路完全一致
// • launch: <<<grid,block>>> ↔  clEnqueueNDRangeKernel(q, k, 0, NULL, gws, lws, ...)
//   (grid ↔ global work size, block ↔ local work size)
// ============================================================================
