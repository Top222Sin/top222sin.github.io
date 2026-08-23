// ============================================================================
// L1.3 会调用 cuBLAS / cuDNN 等上层库
// 配套文档：cuda_l1_blas_cudnn.md
// 对应笔记本：01CUDA基础学习.docx → L1 能跑（入门）→ 3、会调用 cuBLAS/cuDNN 等上层库
// 编译（需 NVIDIA GPU + CUDA Toolkit + cuBLAS + cuDNN）：
//     nvcc -O3 -std=c++17 cuda_l1_blas_cudnn.cpp -lcublas -lcudnn -o blas && ./blas
// 沙箱无 GPU/nvcc，本文件为教材级标准代码，逻辑与真机一致。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>

// ----------------------------------------------------------------------------
// 错误检查宏（三个运行时各一套，包住每个 API 调用，出错立刻报位置）
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t s = (call);                                             \
        if (s != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d: %d\n",                      \
                    __FILE__, __LINE__, s);                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

#define CUDNN_CHECK(call)                                                      \
    do {                                                                       \
        cudnnStatus_t s = (call);                                              \
        if (s != CUDNN_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "[CUDNN ERROR] %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudnnGetErrorString(s));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

// ============================================================================
// 示例一：cuBLAS —— SAXPY（z = a*x + y）
// ----------------------------------------------------------------------------
// 步骤：
//   1) cublasCreate(&handle)        —— 建一个 cuBLAS 上下文（类似"开一个计算会话"）
//   2) cublasSetPointerMode(...,HOST) —— 让 alpha/beta 这种标量直接从 host 取
//   3) cublasSaxpy(handle,n,&a,d_x,1,d_y,1) —— 在 device 上算 y = a*x + y，覆写 y
//   4) cublasDestroy(handle)        —— 释放上下文
// 注意：cuBLAS 内部是列主序（column-major），但 SAXPY 只按 stride=1 顺序扫，
//       和主序无关，所以直接当"向量"用即可。
// ============================================================================
void demo_cublas_saxpy() {
    printf("\n===== 示例一：cuBLAS SAXPY =====\n");
    const int n = 1 << 10;
    const float a = 2.0f;

    std::vector<float> h_x(n), h_y(n);
    for (int i = 0; i < n; i++) { h_x[i] = (float)i; h_y[i] = (float)(2 * i); }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 默认 POINTER_MODE_DEVICE：alpha/beta 必须是 device 指针。
    // 这里传的是 host 上的 &a，所以要切到 HOST 模式，否则读野指针。
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    // y = a*x + y  → 期望 d_y[i] = 2*i + 2*i = 4*i
    CUBLAS_CHECK(cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1));

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = (std::fabs(h_y[5] - 4.0f * 5.0f) < 1e-2f);
    printf("  y[5] = %f (expect %f) -> %s\n", h_y[5], 4.0f * 5.0f, ok ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// ============================================================================
// 示例二：cuBLAS —— SGEMM 矩阵乘 C = A * B
// ----------------------------------------------------------------------------
// 关键陷阱：cuBLAS 是"列主序"。而 C/C++ 的二维数组通常是"行主序"。
// 直接把行主序矩阵塞给 cublasSgemm 会得到转置后的结果。
//
// 经典解法（无需转置数据）：调用时"交换 A、B 的位置，并交换 M、N"。
//   数学推导：设 A(M×K) 行主序、B(K×N) 行主序，想算 C(M×N)=A·B 行主序。
//   cuBLAS 在列主序视角下算的是 C_col = (我们的B) · (我们的A)，
//   经过列主序/行主序的重新解读，恰好等价于 C_row = A_row · B_row。
//   所以调用写成：
//       cublasSgemm(handle, N, N, m=N, n=M, k=K,
//                   &alpha, d_B, ldb=N, d_A, lda=K, &beta, d_C, ldc=N);
//
// 本例手算验证（全部整数，便于核对）：
//   A(2×3) = [[1,2,3],[4,5,6]]
//   B(3×2) = [[7,8],[9,10],[11,12]]
//   C = A·B(2×2):
//     C[0][0]=1*7+2*9+3*11 = 58
//     C[0][1]=1*8+2*10+3*12 = 64
//     C[1][0]=4*7+5*9+6*11 = 139
//     C[1][1]=4*8+5*10+6*12 = 154
// ============================================================================
void demo_cublas_sgemm() {
    printf("\n===== 示例二：cuBLAS SGEMM (C = A*B) =====\n");

    // 行主序展开（row-major：一行接一行）
    const float A[6] = {1,2,3, 4,5,6};        // 2x3
    const float B[6] = {7,8,9, 10,11,12};     // 3x2
    float       C[4] = {0};                   // 2x2 结果
    const int M = 2, K = 3, N = 2;
    const float alpha = 1.0f, beta = 0.0f;     // C = 1*A*B + 0*C

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    // 行主序陷阱解法：交换 A/B，交换 M/N。
    //   cublasSgemm(handle, transa, transb, m, n, k, alpha, A_op, lda, B_op, ldb, beta, C, ldc)
    //   m = N (C 的列数), n = M (C 的行数), k = K
    //   A_op = d_B, ldb = N ;  B_op = d_A, lda = K ;  ldc = N
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, K,
                             &beta,
                             d_C, N));

    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    const float expect[4] = {58, 64, 139, 154};
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        if (std::fabs(C[i] - expect[i]) > 1e-2f) { ok = false; break; }
    }
    printf("  C = [%g, %g, %g, %g] (row-major 2x2)\n", C[0], C[1], C[2], C[3]);
    printf("  expect [58, 64, 139, 154] -> %s\n", ok ? "PASS" : "FAIL");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 示例三：cuDNN —— 卷积前向（最典型的 DNN 算子）
// ----------------------------------------------------------------------------
// 步骤（cuDNN 的"描述符"模型，比 cuBLAS 啰嗦，但把形状/精度/算法都显式化了）：
//   1) cudnnCreate(&handle)
//   2) 输入张量描述符  cudnnSetTensor4dDescriptor(NCHW, FLOAT, n,c,h,w)
//   3) 卷积核描述符    cudnnSetFilter4dDescriptor(FLOAT, NCHW, k,c,r,s)
//   4) 卷积描述符      cudnnSetConvolution2dDescriptor(pad,stride,dilation, CONV, FLOAT)
//   5) 由上面三者算输出张量维度  cudnnGetConvolution2dForwardOutputDim
//   6) 输出张量描述符
//   7) 选算法（这里直接指定 IMPLICIT_GEMM，最简单稳定；cuDNN8+ 推荐用启发式 API）
//   8) 开 workspace 显存  cudnnGetConvolutionForwardWorkspaceSize + cudaMalloc
//   9) cudnnConvolutionForward(...)   —— 真正算
//  10) 校验 + 释放全部描述符/workspace/handle
//
// 本例手算验证（全 1，最容易核对）：
//   输入 1x1x4x4 全 1，卷积核 1x1x3x3 全 1，pad=0,stride=1
//   → 无 padding，输出 1x1x2x2，每个点是 3x3=9 个 1 的和 = 9。
// ============================================================================
void demo_cudnn_conv() {
    printf("\n===== 示例三：cuDNN 卷积前向 (1x1x4x4, 3x3 全1核) =====\n");

    const int N = 1, C = 1, H = 4, W = 4;     // 输入：1 张、1 通道、4x4
    const int K = 1, R = 3, S = 3;            // 核：1 输出通道、3x3
    const int pad = 0, stride = 1, dilation = 1;

    // 输入全 1
    std::vector<float> h_in(N * C * H * W, 1.0f);
    // 核全 1
    std::vector<float> h_wt(K * C * R * S, 1.0f);

    float *d_in, *d_wt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  h_in.size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wt,  h_wt.size()  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,  h_in.data(),  h_in.size()  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wt,  h_wt.data(),  h_wt.size()  * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // 2) 输入张量描述符
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           N, C, H, W));
    // 3) 卷积核描述符
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           K, C, R, S));
    // 4) 卷积描述符
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                pad, pad,         // 上下/左右 padding
                                                stride, stride,   // 步长
                                                dilation, dilation,
                                                CUDNN_CONVOLUTION,
                                                CUDNN_DATA_FLOAT));
    // 5) 算输出维度
    int n_, c_, h_, w_;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc,
                                                      &n_, &c_, &h_, &w_));
    printf("  output dim: N=%d C=%d H=%d W=%d\n", n_, c_, h_, w_);

    // 6) 输出张量描述符
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           n_, c_, h_, w_));

    CUDA_CHECK(cudaMalloc(&d_out, n_ * c_ * h_ * w_ * sizeof(float)));

    // 7) 选算法（直接指定，避免版本差异；cuDNN8+ 推荐用启发式 API 选最快算法）
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // 8) 开 workspace 显存
    size_t ws_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, x_desc, w_desc,
                                                        conv_desc, y_desc, algo, &ws_size));
    void *d_workspace = nullptr;
    if (ws_size > 0) CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

    // 9) 真正算卷积
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
                                        &alpha, x_desc, d_in,
                                        w_desc, d_wt,
                                        conv_desc, algo, d_workspace, ws_size,
                                        &beta,  y_desc, d_out));

    // 10) 拷回校验
    std::vector<float> h_out(n_ * c_ * h_ * w_);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = true;
    for (float v : h_out) if (std::fabs(v - 9.0f) > 1e-2f) { ok = false; break; }
    printf("  output (row-major 2x2): [%g, %g, %g, %g] -> %s\n",
           h_out[0], h_out[1], h_out[2], h_out[3], ok ? "PASS" : "FAIL");

    // 释放
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_out));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

// ============================================================================
int main() {
    demo_cublas_saxpy();
    demo_cublas_sgemm();
    demo_cudnn_conv();
    printf("\n全部示例完成（需真实 GPU 才能运行；沙箱下为教材级代码）。\n");
    return 0;
}

// ============================================================================
// OpenCL 对照（双轨学习）
// ----------------------------------------------------------------------------
// cuBLAS / cuDNN 在 OpenCL 这边没有"同名官方孪生库"，但有对应生态：
//
//   CUDA / cuBLAS / cuDNN              | OpenCL 世界
//   ----------------------------------|---------------------------------------
//   cuBLAS SAXPY/SGEMM                 | clBLAS（AMD 主导的 OpenCL BLAS 实现，
//                                     |    API 与 cuBLAS 几乎一一对应：
//                                     |    clblasSaxpy / clblasSgemm）
//   cuDNN 卷积/池化/BN/激活            | 无官方统一库。常见做法：
//                                     |   - ROCm 的 MIOpen 是 HIP 的（非纯 OpenCL）
//                                     |   - 用 OpenCL 手写 kernel 或借助 clDNN（旧 Intel）
//                                     |   - 框架层（如国产推理引擎）自己写 OpenCL 算子
//
// 认知重点：调库的本质是"复用别人调好的 kernel + 他家的 launch 配置 + 他家的
//           内存布局约定（如 cuBLAS 列主序）"。换平台时，这些约定要重新对齐，
//           所以工程上常把"核心算法"和"后端适配层"分开——这正是 NCNN/MNN 的
//           ARM 后端（你之前在 neon_demo 里深挖的那层）做的事：同一份算法，
//           在 NEON / SVE2 / CUDA / OpenCL 各写一份后端，前端算子不变。
// ============================================================================
