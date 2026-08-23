// ============================================================================
//  L3 补充（NEON 线 · 第1站）：ncnn 怎么用 ARM NEON 手写卷积 / 矩阵乘
//
//  读 real source： _ncnn_src/{convolution_arm.cpp, convolution_im2col_gemm.h,
//                              convolution_1x1.h, convolution_3x3.h, arm_usability.h}
//  （Tencent/ncnn@master，src/layer/arm/）
//
//  本文件是「纯 C++ 可跑」的算法镜像（无 ARM 编译器也能跑）：
//      g++ -O3 -std=c++17 cuda_l3_neon_conv_gemm.cpp -o neon && ./neon
//  它把 ncnn 里三段最典型的 NEON kernel 的「算法结构」逐一对拍到标量循环：
//      (A) 直接 3x3 卷积  —— 4 路宽向量同时算 4 个输出像素（对应 conv3x3s1_neon）
//      (B) 1x1 卷积 = GEMM —— 8 输出通道 × 8 输入通道分块（对应 conv1x1s1_neon）
//      (C) im2col+SGEMM 的 8x12 寄存器分块微核（对应 convolution_gemm_transB_packed_tile）
//  每段都和朴素实现对拍，verify() 全 PASS 才退出 0。
//
//  核心结论（见 md）：ncnn 的全部手写 ARM 卷积/矩阵乘，最终都落到同一个原语 ——
//      fmla / vfmaq_laneq_f32 ：把「一个权重标量 lane」广播到 4 路输入向量相乘，
//      累加到多个输出寄存器；再用寄存器分块（8x12 / 8x8 / 4 路）摊薄访存、喂满流水线。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

// ---------- 小工具 ----------
static std::mt19937& rng() { static std::mt19937 r(20260815); return r; }
static std::vector<float> rand_mat(int n, float lo, float hi) {
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = U(rng());
    return v;
}
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// ============================================================================
//  (A) 直接 3x3 卷积 —— NEON 风格：4 路宽向量一次算 4 个输出像素
//  对照：convolution_3x3.h:4  conv3x3s1_neon
//        内核 FMA：vmlaq_f32  convolution_3x3.h:369-381
//        水平求和：vaddvq_f32 convolution_3x3.h:388
//  ncnn 真实做法是把 3x3 核预打包成 3 个「错位向量」(_k00/_k03/_k06)，
//  把 9 次乘累加压成 6 次 FMA；这里用清晰版（9 次 MAC，4 路并行）镜像同一结构。
// ============================================================================
static void naive_conv3x3(const std::vector<float>& in, int H, int W,
                          const float K[9], std::vector<float>& out) {
    out.assign(H * W, 0.f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float s = 0.f;
            for (int dy = 0; dy < 3; ++dy)
                for (int dx = 0; dx < 3; ++dx) {
                    int yy = y + dy - 1, xx = x + dx - 1;
                    if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue; // 默认 pad=0 越界视为 0
                    s += in[yy * W + xx] * K[dy * 3 + dx];
                }
            out[y * W + x] = s;
        }
}
// NEON 风格：每个 SIMD lane 对应一个输出像素，4 路一次推进
static void neon_style_conv3x3(const std::vector<float>& in, int H, int W,
                               const float K[9], std::vector<float>& out) {
    out.assign(H * W, 0.f);
    // pad=0：越界输入按 0 处理，因此边界行的 4-lane 需用 0 填充的临时行
    std::vector<float> pad(W + 4, 0.f); // 略大于一行，便于取 c-1..c+3
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; x += 4) {
            // 三条输入行（y-1, y, y+1）的 [x-1 .. x+4] 共 6 个像素，整块填充一次
            // （4 路输出像素的 3x3 窗口最右列到 x+3+1 = x+4，故需 6 个缓冲，索引 0..5）
            float r0[6], r1[6], r2[6];
            for (int row = 0; row < 3; ++row) {
                int yy = y + row - 1;
                float* dst = (row == 0) ? r0 : (row == 1) ? r1 : r2;
                for (int k = 0; k < 6; ++k) {
                    int xx = (x - 1) + k; // 窗口首列固定为 x-1（不随 lane 变）
                    dst[k] = (yy < 0 || yy >= H || xx < 0 || xx >= W) ? 0.f : in[yy * W + xx];
                }
            }
            // 4 路（lane=0..3）各算一个输出像素，9 次 MAC
            float o[4] = {0, 0, 0, 0};
            for (int dy = 0; dy < 3; ++dy)
                for (int dx = 0; dx < 3; ++dx) {
                    const float* rowvec = (dy == 0) ? r0 : (dy == 1) ? r1 : r2;
                    float w = K[dy * 3 + dx];
                    for (int lane = 0; lane < 4; ++lane)
                        o[lane] += rowvec[dx + lane] * w; // 等价于 NEON 的 vmlaq_f32
                }
            for (int lane = 0; lane < 4 && (x + lane) < W; ++lane)
                out[y * W + x + lane] = o[lane];
        }
    }
}

// ============================================================================
//  (B) 1x1 卷积 = GEMM —— NEON 风格：8 输出通道 × 8 输入通道分块
//  对照：convolution_1x1.h:4  conv1x1s1_neon
//        8 outch 分块：convolution_1x1.h:20-21  (outch >> 3)
//        8 inch 分块：convolution_1x1.h:57      (q += 8)
//        内核 FMA：fmla vX.4s, vY.4s, %Z.s[0]   convolution_1x1.h:134-165
//  1x1 conv 本质：out[H*W][outch] = in[H*W][inch] × W[outch][inch]
// ============================================================================
static void naive_conv1x1(const std::vector<float>& in, int HW, int inch, int outch,
                          const std::vector<float>& W, std::vector<float>& out) {
    out.assign(HW * outch, 0.f);
    for (int o = 0; o < outch; ++o)
        for (int s = 0; s < HW; ++s) {
            float acc = 0.f;
            for (int i = 0; i < inch; ++i) acc += in[s * inch + i] * W[o * inch + i];
            out[o * HW + s] = acc;
        }
}
// NEON 风格：8 输出通道同时累加（每通道 4 路宽向量 => 一次处理 4 个空间点）
static void neon_style_conv1x1(const std::vector<float>& in, int HW, int inch, int outch,
                               const std::vector<float>& W, std::vector<float>& out) {
    out.assign(HW * outch, 0.f);
    int pp = 0;
    for (; pp + 8 <= outch; pp += 8) {        // 8 输出通道分块 (convolution_1x1.h:20)
            for (int s = 0; s < HW; s += 4) {     // 4 路宽向量（一次 4 个空间点）
            float acc[8][4] = {{0}};
            int q = 0;
            for (; q + 8 <= inch; q += 8) {    // 8 输入通道分块 (convolution_1x1.h:57)
                for (int k = 0; k < 8; ++k) {  // 输入通道
                    float x[4];
                    for (int l = 0; l < 4 && s + l < HW; ++l) x[l] = in[(s + l) * inch + q + k];
                    for (int o = 0; o < 8; ++o) {
                        float w = W[(pp + o) * inch + q + k]; // 广播该权重到 4 路（= fmla lane 语义）
                        for (int l = 0; l < 4 && s + l < HW; ++l) acc[o][l] += x[l] * w;
                    }
                }
            }
            // 输入通道余量（inch 非 8 倍数时；对应 convolution_1x1.h:98 的 remain 处理）
            for (; q < inch; ++q) {
                for (int l = 0; l < 4 && s + l < HW; ++l) {
                    float x = in[(s + l) * inch + q];
                    for (int o = 0; o < 8; ++o) acc[o][l] += x * W[(pp + o) * inch + q];
                }
            }
            for (int o = 0; o < 8; ++o)
                for (int l = 0; l < 4 && s + l < HW; ++l)
                    out[(pp + o) * HW + s + l] = acc[o][l];
        }
    }
    // 余量（对应 convolution_1x1.h:98 的 `size & 3` / remain_outch 处理）
    for (int o = pp; o < outch; ++o)
        for (int s = 0; s < HW; ++s) {
            float a = 0.f;
            for (int i = 0; i < inch; ++i) a += in[s * inch + i] * W[o * inch + i];
            out[o * HW + s] = a;
        }
}

// ============================================================================
//  (C) im2col + SGEMM 的 8x12 寄存器分块微核
//  对照：convolution_im2col_gemm.h:178  convolution_gemm_transB_packed_tile
//        分块：ii += 8 行 (:194) ，jj += 12 列 (:206)
//        内含微核 FMA：vfmaq_laneq_f32  convolution_im2col_gemm.h:1569-1592
//                      （aarch64 内联汇编 fmla  Convolution_im2col_gemm.h:266）
//  含义：C[8][12] += A[8][K] × B[12][K]（B 已按卷积需求转置/打包）
// ============================================================================
static void naive_gemm(const std::vector<float>& A, const std::vector<float>& B,
                       int M, int N, int K, std::vector<float>& C) {
    C.assign(M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[j * K + k];
            C[i * N + j] = s;
        }
}
// NEON 风格：固定 8x12 输出分块，K 主循环内把权重 lane 广播到 8 路输入累加
static void neon_style_sgemm_8x12(const std::vector<float>& A, const std::vector<float>& B,
                                  int K, std::vector<float>& C) {
    // C 必须已分配 8*12 且清零（微核只算一个 8x12 tile）
    C.assign(8 * 12, 0.f);
    std::vector<float> acc(8 * 12, 0.f);
    for (int k = 0; k < K; ++k) {                 // K 主循环（对应 max_kk 循环）
        // 取 A 的 8 行第 k 列（8 路输入向量），B 的 12 行第 k 列（12 个权重标量）
        float arow[8], bcol[12];
        for (int i = 0; i < 8; ++i) arow[i] = A[i * K + k];
        for (int j = 0; j < 12; ++j) bcol[j] = B[j * K + k]; // bcol[j] 即被广播的那一 lane
        for (int i = 0; i < 8; ++i)               // 等价于 vfmaq_laneq_f32：a[i] × b[lane] 累加
            for (int j = 0; j < 12; ++j)
                acc[i * 12 + j] += arow[i] * bcol[j];
    }
    for (int i = 0; i < 8 * 12; ++i) C[i] = acc[i];
}

// ============================================================================
//  verify：三段对拍，全 PASS 才返回 0
// ============================================================================
static bool verify() {
    bool ok = true;

    // (A) 3x3
    {
        int H = 16, W = 16;
        float K[9];
        for (int i = 0; i < 9; ++i) K[i] = (float)(i + 1) * 0.1f;
        auto in = rand_mat(H * W, -1.f, 1.f);
        std::vector<float> o1, o2;
        naive_conv3x3(in, H, W, K, o1);
        neon_style_conv3x3(in, H, W, K, o2);
        float d = maxdiff(o1, o2);
        printf("  (A) 3x3 直接卷积  : neon vs naive maxdiff = %.2e  %s\n", d, d < 1e-5f ? "PASS" : "FAIL");
        ok &= (d < 1e-5f);
    }

    // (B) 1x1 = GEMM
    {
        int HW = 24, inch = 20, outch = 18;
        auto in = rand_mat(HW * inch, -1.f, 1.f);
        auto W  = rand_mat(outch * inch, -1.f, 1.f);
        std::vector<float> o1, o2;
        naive_conv1x1(in, HW, inch, outch, W, o1);
        neon_style_conv1x1(in, HW, inch, outch, W, o2);
        float d = maxdiff(o1, o2);
        printf("  (B) 1x1 卷积=GEMM : neon vs naive maxdiff = %.2e  %s\n", d, d < 1e-5f ? "PASS" : "FAIL");
        ok &= (d < 1e-5f);
    }

    // (C) 8x12 SGEMM tile
    {
        int K = 32;
        auto A = rand_mat(8 * K, -1.f, 1.f);
        auto B = rand_mat(12 * K, -1.f, 1.f);
        std::vector<float> o1, o2;
        naive_gemm(A, B, 8, 12, K, o1);
        neon_style_sgemm_8x12(A, B, K, o2);
        float d = maxdiff(o1, o2);
        printf("  (C) 8x12 SGEMM tile: neon vs naive maxdiff = %.2e  %s\n", d, d < 1e-5f ? "PASS" : "FAIL");
        ok &= (d < 1e-5f);
    }
    return ok;
}

int main() {
    printf("########## L3 补充（NEON 线）ncnn 手写 ARM 卷积 / 矩阵乘 ##########\n");
    bool ok = verify();
    printf("\n[summary] 三段 NEON 风格算法镜像 %s\n", ok ? "全部 PASS" : "存在 FAIL");
    printf("  统一原语：fmla / vfmaq_laneq_f32 （权重 lane 广播 × 4 路输入累加）+ 寄存器分块\n");
    return ok ? 0 : 1;
}
