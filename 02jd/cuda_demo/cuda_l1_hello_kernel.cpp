// ============================================================================
// L1.2 会写简单 kernel + 配置 launch bounds + 用 cudaMalloc / cudaMemcpy
// 配套文档：cuda_l1_hello_kernel.md
// 对应笔记本：01CUDA基础学习.docx → L1 能跑（入门）→ 2、会写简单 kernel...
// 编译（需 NVIDIA GPU + CUDA Toolkit）：
//     nvcc -O3 -std=c++17 cuda_l1_hello_kernel.cpp -o hk && ./hk
// 沙箱无 GPU/nvcc，本文件为教材级标准 CUDA，逻辑与真机一致。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

// ----------------------------------------------------------------------------
// CUDA 错误检查宏：每个 CUDA API 调用都包一层，出错立刻报出文件名/行号/原因。
// 这是工业级写法，初学也要养成习惯——否则第一个错误会淹没在几百行后。
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

// ============================================================================
// 一、简单的 kernel：SAXPY  =  z = a * x + y
// ----------------------------------------------------------------------------
// 关键认知（呼应 L1.1）：
//   - kernel 里写的是"一条线程的逻辑"，i 是这条线程负责的下标。
//   - __global__ 表示"从 host 调用、在 device 执行"，无返回值（void）。
//   - 结果通过写入 device 指针 z 带出来。
//   - if (i < n) 做边界保护：总线程数常比 n 大（grid 向上取整），越界线程要跳过。
// ============================================================================
__global__ void saxpy(int n, float a, const float* x, const float* y, float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局下标（L1.1 必背公式）
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}

// ----------------------------------------------------------------------------
// 二、配置 launch bounds（本篇重点之一）
// ----------------------------------------------------------------------------
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//   - 第 1 个参数：本 kernel 启动时每 block 最多多少线程（必须等于或大于实际 blockDim.x）。
//   - 第 2 个参数（可选）：每个 SM 上至少想同时驻留几个 block。
//
// 编译器怎么用它：
//   它告诉编译器"这块代码最多 256 线程、至少 4 个 block 同时跑在 SM 上"。
//   编译器据此反推"每个线程最多能用多少寄存器"——
//     寄存器总数(SM 固定) / (256 线程 * 4 block) → 给每个线程的寄存器上限变小。
//   寄存器用得少 → 能驻留更多 block/线程 → occupancy（SM 利用率）更高 → 掩盖延迟更好。
//   代价：寄存器不够时编译器会把变量"溢出"到 local memory（慢的 off-chip），所以不是越小越好。
//
// 经验值：
//   - 只给第 1 个参数 = 告诉编译器 block 大小上界，足够大多数情况。
//   - 第 2 个参数帮助编译器在"省寄存器换 occupancy"和"多寄存器换性能"间权衡。
//   - 设得比真实 blockDim 小会编译失败或行为未定义，必须 ≥ 实际启动值。
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4) saxpy_bounded(
    int n, float a, const float* x, const float* y, float* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}

// ----------------------------------------------------------------------------
// 三、第二个简单 kernel：向量平方，演示"换个运算、复用同一套 host 流程"。
// 不带 launch_bounds 也能跑——说明它是"可选优化提示"，不是必需语法。
// ----------------------------------------------------------------------------
__global__ void vsqr(int n, const float* x, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * x[i];
}

// ============================================================================
// Host（CPU）端主流程
// ----------------------------------------------------------------------------
// 一个标准 CUDA 程序的"五步数据流"：
//   1) 在 host 准备数据（普通 vector<float>）
//   2) cudaMalloc 在 device 上开显存
//   3) cudaMemcpy H2D 把输入从 host 拷到 device
//   4) 启动 kernel（<<<grid, block>>>）
//   5) cudaMemcpy D2H 把结果拷回 host，cudaFree 释放显存
//
// 重要细节：
//   - cudaMalloc 返回的是"device 指针"，只能在 device 代码里用；host 不能直接解引用。
//   - cudaMemcpy 是"同步"的（默认 stream 0），调用返回时拷贝已完成；
//     它拷的是"主机虚拟地址 ↔ 设备地址"，由 CUDA runtime 区分方向（cudaMemcpyHostToDevice / DeviceToHost）。
//   - kernel 启动 <<<>>> 本身是异步的：CPU 发出后就继续往下走，不会等 GPU 算完。
//     所以 D2H 拷贝前不需要显式同步——cudaMemcpy 会隐式等之前的 kernel 完成（同 stream 顺序保证）。
// ============================================================================
int main() {
    const int n = 1 << 16;            // 65536 个元素
    const float a = 2.0f;

    // ---- 1) host 准备数据 ----
    std::vector<float> h_x(n), h_y(n), h_z(n, 0.f);
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i * 0.5f);
    }

    // ---- 2) device 显存分配 ----
    float *d_x, *d_y, *d_z;
    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));

    // ---- 3) H2D 拷入输入 ----
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    // ---- 4) 配置并启动 kernel ----
    int blockSize = 256;                                   // 每 block 256 线程（= launch_bounds 上限，正好）
    int gridSize  = (n + blockSize - 1) / blockSize;       // 向上取整：ceil(n / blockSize)
    printf("launch: grid=%d blocks, block=%d threads, total=%d (n=%d)\n",
           gridSize, blockSize, gridSize * blockSize, n);

    // 用带 launch_bounds 的版本（演示配置）
    saxpy_bounded<<<gridSize, blockSize>>>(n, a, d_x, d_y, d_z);
    CUDA_CHECK(cudaGetLastError());   // 捕获"启动"阶段的错误（参数非法、kernel 未注册等）

    // ---- 5) D2H 拷回结果 + 校验 ----
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));

    // 校验前几个 + 随机几个
    bool ok = true;
    for (int i = 0; i < 10 && i < n; i++) {
        float expect = a * h_x[i] + h_y[i];
        if (std::fabs(h_z[i] - expect) > 1e-3f) {
            printf("MISMATCH @%d: got %f expect %f\n", i, h_z[i], expect);
            ok = false; break;
        }
    }
    printf("saxpy verify: %s\n", ok ? "PASS" : "FAIL");

    // ---- 第二个 kernel：向量平方（复用 device 指针，省一次分配） ----
    vsqr<<<gridSize, blockSize>>>(n, d_x, d_z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, bytes, cudaMemcpyDeviceToHost));
    ok = (std::fabs(h_z[42] - h_x[42] * h_x[42]) < 1e-3f);
    printf("vsqr verify (z[42]=%f, expect=%f): %s\n", h_z[42], h_x[42] * h_x[42], ok ? "PASS" : "FAIL");

    // ---- 释放显存 ----
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    return 0;
}

// ============================================================================
// OpenCL 对照（双轨学习）
// ----------------------------------------------------------------------------
// 同一个 SAXPY，在 OpenCL 里概念一一对应：
//
//   CUDA                                | OpenCL
//   -----------------------------------|-----------------------------------
//   __global__ void saxpy(...)          | __kernel void saxpy(__global const float* x, ...)
//   __launch_bounds__(256,4)            | 无直接等价；靠 local_work_size 运行时决定，编译器优化靠编译选项
//   cudaMalloc(&d, bytes)               | clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err)
//   cudaMemcpy(d, h, bytes, H2D)        | clEnqueueWriteBuffer(q, d, CL_TRUE, 0, bytes, h, 0, NULL, NULL)
//   saxpy<<<grid, block>>>(args)        | clEnqueueNDRangeKernel(q, k, 0, &gws, &lws, 0, NULL, NULL)
//   cudaMemcpy(h, d, bytes, D2H)        | clEnqueueReadBuffer(q, d, CL_TRUE, 0, bytes, h, 0, NULL, NULL)
//   cudaFree(d)                         | clReleaseMemObject(d)
//
// 差异点：
//   1) OpenCL 没有 __launch_bounds__ 这种"编译期给 occupancy 的提示"，block/warp 占用率
//      完全由你选的 local_work_size 决定。
//   2) OpenCL 显式区分 cl_mem 对象与主机指针，读写都要排队到 command queue；CUDA 的
//      cudaMemcpy 默认同步、用起来更像"普通 memcpy"。
//   3) 底层 SIMT 模型（warp/wavefront）完全一致，差异只在 API 与运行时抽象层。
// ============================================================================
