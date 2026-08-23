// cuda_l1_programming_model.cpp
// CUDA 编程模型第一个教学文件：kernel / grid / block / thread / SIMT
// 对应笔记本: 01CUDA基础学习.docx -> L1 能跑(入门) -> 1、理解编程模型
//
// 编译与运行(需要 NVIDIA GPU + CUDA Toolkit, 本沙箱无 nvcc, 仅作教材):
//   nvcc -O3 -std=c++17 cuda_l1_programming_model.cpp -o pm && ./pm
//
// 想对照 OpenCL? 末尾有 "OpenCL 等价写法" 注释, 概念一一对应:
//   CUDA kernel   <->  OpenCL __kernel
//   grid          <->  NDRange (global work size)
//   block         <->  work-group
//   thread        <->  work-item
//   threadIdx/    <->  get_local_id / get_group_id / get_global_id
//     blockIdx
//   SIMT(warp)    <->  SIMT(wavefront, AMD 为 32/64)
#include <cstdio>
#include <cstdlib>
#include <cmath>

// =====================================================================
// 概念 1: KERNEL
// __global__ 标记的函数叫 kernel —— 它不在 CPU 上顺序执行,
// 而是由 GPU 上"成千上万个线程"各自执行一遍。
// 每个线程拿到的"自己该算哪一份数据"由内置索引决定(见下)。
// (OpenCL 里写作 `__kernel void`, 语法几乎一致)
// =====================================================================

// ---- 教学 kernel A: 向量加(最经典的第一个 kernel) ----
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    // 概念 2: 全局唯一索引
    //   grid  = 所有 block 的集合
    //   block = 一组可协作的线程(可共享 shared memory / __syncthreads)
    //   thread= 单个执行单元
    // 一维时: 全局下标 i = blockIdx.x * blockDim.x + threadIdx.x
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                 // 边界保护: 总线程数通常 > n, 多余线程直接跳过
        C[i] = A[i] + B[i];      // 概念 3: SIMT —— 你只写"一条线程的标量逻辑",
                                 // 硬件自动把 32 条线程编成一个 warp 锁步执行同一条指令,
                                 // 各线程用各自的 i、各自的 A[i]/B[i] 数据。
    }
}

// ---- 教学 kernel B: 把 grid/block/thread 的索引关系"打印"出来 ----
// 每个线程把自己的 (全局下标, 所在 block, 组内 thread) 写进数组,
// CPU 端再读出来, 直观看到三层层次是怎么编号的。
__global__ void indexDemo(int* gOut, int* bOut, int* tOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gOut[i] = i;            // 全局下标
        bOut[i] = blockIdx.x;   // 第几个 block
        tOut[i] = threadIdx.x;  // 该 block 内第几个 thread
    }
}

int main() {
    const int n = 16;           // 故意取小, 方便观察编号
    size_t bytes = n * sizeof(int);

    // ---- host 端内存 ----
    int *h_g = (int*)malloc(bytes), *h_b = (int*)malloc(bytes), *h_t = (int*)malloc(bytes);

    // ---- device 端内存 (cudaMalloc 类比 malloc, 但活在 GPU 显存) ----
    int *d_g, *d_b, *d_t;
    cudaMalloc(&d_g, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_t, bytes);

    // ---- 概念 2(续): 启动配置 <<<grid, block>>> ----
    //   blockSize = 每个 block 多少线程 (这里是 4, 仅为演示清楚; 实战常用 128/256/512)
    //   gridSize  = 一共多少 block = ceil(n / blockSize)
    // 这里 n=16, block=4 -> grid=4 个 block, 共 16 线程, 一一对应 16 个元素。
    int blockSize = 4;
    int gridSize  = (n + blockSize - 1) / blockSize;   // 经典 "向上取整" 写法

    printf("launch: grid=%d blocks, block=%d threads, total=%d\n",
           gridSize, blockSize, gridSize * blockSize);

    // <<<grid, block>>> 就是"把 kernel 撒到 GPU 上, 按这个层次铺开"
    indexDemo<<<gridSize, blockSize>>>(d_g, d_b, d_t, n);
    cudaDeviceSynchronize();   // 等 GPU 算完(教学用; 实战常与 cudaMemcpy 隐式同步)

    // 把结果拷回 host 查看
    cudaMemcpy(h_g, d_g, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t, d_t, bytes, cudaMemcpyDeviceToHost);

    printf("%4s %5s %5s %5s\n", "i", "block", "thread", "global_i");
    for (int i = 0; i < n; i++)
        printf("%4d %5d %5d %5d\n", i, h_b[i], h_t[i], h_g[i]);

    // ---- 概念 3(验证 SIMT 的"线程格局"): 再跑一次向量加, 确认结果 ----
    // (这里省略完整 A/B 初始化, 仅演示调用形态)
    const int m = 1024;
    float *d_A, *d_B, *d_C; size_t fb = m * sizeof(float);
    cudaMalloc(&d_A, fb); cudaMalloc(&d_B, fb); cudaMalloc(&d_C, fb);
    int bs = 256, gs = (m + bs - 1) / bs;
    vecAdd<<<gs, bs>>>(d_A, d_B, d_C, m);   // 1024 线程 / 256 = 4 block, 每线程算 1 个元素
    cudaDeviceSynchronize();
    printf("vecAdd launched with grid=%d block=%d (total threads=%d)\n", gs, bs, gs*bs);

    // ---- 释放 ----
    cudaFree(d_g); cudaFree(d_b); cudaFree(d_t);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_g); free(h_b); free(h_t);
    return 0;
}

/*
 * ============================ OpenCL 等价写法(对照) ============================
 * // OpenCL: kernel 用 __kernel, 索引靠 get_* 内建函数
 * __kernel void vecAdd(__global const float* A,
 *                      __global const float* B,
 *                      __global float* C, int n) {
 *     int i = get_global_id(0);              // = blockIdx.x*blockDim.x + threadIdx.x
 *     if (i < n) C[i] = A[i] + B[i];
 * }
 * // 启动: clEnqueueNDRangeKernel(cmd, kern, 1, NULL, &globalWorkSize, &localWorkSize, ...);
 * //   globalWorkSize == gridDim*blockDim (即 CUDA 的总线程数)
 * //   localWorkSize  == blockDim        (即 CUDA 的 block 大小)
 * // SIMT 概念完全一致: AMD 把 32/64 个 work-item 编成 wavefront 锁步执行。
 * ==============================================================================
 */
