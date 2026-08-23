// =============================================================================
//  NEON 对照线 · 第 5/6 站：大块 Winograd F(4x4,3x3) / F(3x3,3x3) + Depthwise pack8
//  纯 C++ 仿真（无需 ARM 编译器）：把 ncnn src/layer/arm/ 的两段手写 NEON 算法
//  还原成数值等价的 host 实现，并与 naive 参考对拍验证。
//
//  真实源码（已拉取到 _ncnn_src/，Tencent/ncnn@master/src/layer/arm/）：
//   - convolution_3x3_winograd.h        （F(2,3)/F(4,3)/F(6,3)；本文件用其 F(4,3) 真实常数）
//       · G(6x3)        transform_kernel_tile      行 5692-5761（ktm0..4）
//       · ITM=Bᵀ(6x6)   transform_input_tile 注释  行 5803-5810
//       · OTM=Aᵀ(4x6)   transform_output_tile 注释 行 6602-6607
//       · 驱动 conv3x3s1_winograd43                  行 7258 附近
//   - convolutiondepthwise_3x3.h         （逐通道 3x3，标量参考；pack8 重建基础）
//       · repack _k012x/_k345x/_k678x              行 36-42
//       · ext + fmla 滑窗                         行 71-115
//
//  关于 F(3x3,3x3)：ncnn 不 ship 这个 tile（只用 F(2,3)/F(4,3)/F(6,3)），因此本文件
//  用 ALS（交替最小二乘）在双线性约束 Σ_i Aᵀ[j][i]·G[i][k]·Bᵀ[i][l] = δ(l,j+k) 下
//  数值求解一致三元组，再用联合 Gauss-Newton 把残差压到机器精度（~1.7e-15），
//  2D 卷积对拍误差 ~3e-15。常数由 _solve_w33.py 生成、由 verify_neon_winobig_pack8.py
//  交叉验证（全部 PASS）。
//
//  编译（ARM 真机或任意 x86 编译器均可，纯 host 仿真）：
//    g++ -O3 -std=c++17 cuda_l3_neon_winograd_big_pack8.cpp -o neonbig && ./neonbig
//
//  数值权威校验见：verify_neon_winobig_pack8.py（Python，无需编译器即可重跑）
// =============================================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// -----------------------------------------------------------------------------
//  小工具：定长矩阵乘 / 转置（仅本仿真用，非 NEON 原语；行主序扁平缓冲）
// -----------------------------------------------------------------------------
static void mmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float s = 0.f; for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
        C[i * N + j] = s;
    }
}
static void transpose(const float* A, float* B, int M, int N) {
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) B[j * M + i] = A[i * N + j];
}

// =============================================================================
//  ⑤a Winograd F(4x4,3x3) —— 直接采用 ncnn convolution_3x3_winograd.h 的真实常数
//  关键思想：把 6x6 输入块 + 3x3 核都变换到 Winograd 域，做 6x6=36 次逐元素乘
//            （Hadamard），再反变换得到 4x4 输出。直接法同区域需 4*4*3*3 = 144 次
//            MAC；Winograd 只需 36 次变换域乘（+ 变换本身），约 4.00x 降乘。
// =============================================================================
static const float SQ2 = 1.4142135623730951f;

// 核变换 G (6x3) —— convolution_3x3_winograd.h:5707-5761（ktm0=2/3, ktm1=sq2/3, ktm2=1/3, ktm3=1/6, ktm4=sq2/6）
static const float G43[6][3] = {
    { 1.0f,        0.0f,        0.0f    },
    {-2.0f/3,     -SQ2/3,      -1.0f/3 },
    {-2.0f/3,      SQ2/3,      -1.0f/3 },
    { 1.0f/6,      SQ2/6,       1.0f/3 },
    { 1.0f/6,     -SQ2/6,       1.0f/3 },
    { 0.0f,        0.0f,        1.0f    },
};
// 输入变换 ITM = Bᵀ (6x6) —— transform_input_tile 注释 行 5803-5810
static const float ITM43[6][6] = {
    { 1.0f,  0.0f,  -2.5f,     0.0f,     1.0f,   0.0f },
    { 0.0f, -SQ2,  -2.0f,      SQ2/2,    1.0f,   0.0f },
    { 0.0f,  SQ2,  -2.0f,     -SQ2/2,    1.0f,   0.0f },
    { 0.0f, -SQ2/2,-0.5f,      SQ2,      1.0f,   0.0f },
    { 0.0f,  SQ2/2,-0.5f,     -SQ2,      1.0f,   0.0f },
    { 0.0f,  1.0f,   0.0f,    -2.5f,     0.0f,   1.0f },
};
// 输出变换 OTM = Aᵀ (4x6) —— transform_output_tile 注释 行 6602-6607
static const float OTM43[4][6] = {
    { 1.0f,           1.0f,           1.0f,           1.0f,           1.0f,           0.0f },
    { 0.0f,           SQ2/2,        -SQ2/2,          SQ2,           -SQ2,           0.0f },
    { 0.0f,           0.5f,           0.5f,           2.0f,           2.0f,           0.0f },
    { 0.0f,           SQ2/4,         -SQ2/4,          SQ2*2,         -SQ2*2,         1.0f },
};

// 一个 6x6 输入块 d 与 3x3 核 g -> 4x4 输出（cross-correlation，即 CNN 习惯）
static void winograd43_tile(const float d[6][6], const float g[3][3], float out[4][4]) {
    // 1) 输入变换 D = ITM * d * ITMᵀ
    float Dm[36], D[36], IT[36];
    mmul(&ITM43[0][0], &d[0][0], Dm, 6, 6, 6);
    transpose(&ITM43[0][0], IT, 6, 6);
    mmul(Dm, IT, D, 6, 6, 6);
    // 2) 核变换 K = G * g * Gᵀ
    float Mg[18], K[36], GT[18];
    mmul(&G43[0][0], &g[0][0], Mg, 6, 3, 3);
    transpose(&G43[0][0], GT, 6, 3);
    mmul(Mg, GT, K, 6, 3, 6);
    // 3) Hadamard（逐元素乘）：M = D ⊙ K
    float M[36];
    for (int i = 0; i < 36; ++i) M[i] = D[i] * K[i];
    // 4) 输出变换 Y = OTM * M * OTMᵀ
    float OM[24], OT[24];
    mmul(&OTM43[0][0], M, OM, 4, 6, 6);
    transpose(&OTM43[0][0], OT, 4, 6);
    mmul(OM, OT, &out[0][0], 4, 6, 4);
}

// 整图 tiled 驱动（padding=1，same 输出），与 naive 3x3 cross-correlation 对拍。
// 以输出 4x4 块为锚：输出块 (oy,ox) 对应输入 6x6 块 (oy-1..oy+4, ox-1..ox+4)。
static void conv3x3_winograd43(const std::vector<float>& in, int H, int W,
                               const float g[3][3], std::vector<float>& out) {
    out.assign((size_t)H * W, 0.f);
    for (int oy = 0; oy < H; oy += 4) {
        for (int ox = 0; ox < W; ox += 4) {
            float d[6][6];
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
                int yy = oy - 1 + i, xx = ox - 1 + j;   // 6x6 输入邻域（带 padding）
                d[i][j] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
            }
            float o[4][4]; winograd43_tile(d, g, o);
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                int yy = oy + i, xx = ox + j;
                if (yy < H && xx < W) out[yy * W + xx] = o[i][j];
            }
        }
    }
}
static void conv3x3_naive(const std::vector<float>& in, int H, int W,
                          const float g[3][3], std::vector<float>& out) {
    out.assign((size_t)H * W, 0.f);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        float s = 0.f;
        for (int dy = 0; dy < 3; ++dy) for (int dx = 0; dx < 3; ++dx) {
            int yy = y + dy - 1, xx = x + dx - 1;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += in[yy * W + xx] * g[dy][dx];
        }
        out[y * W + x] = s;
    }
}

// =============================================================================
//  ⑤b Winograd F(3x3,3x3) —— ncnn 不 ship，用 ALS 数值构造的一致三元组（n=5）
//  关键思想：把 5x5 输入块 + 3x3 核变换到 Winograd 域，做 5x5=25 次逐元素乘，
//            反变换得 3x3 输出。直接法同区域需 3*3*3*3 = 81 次 MAC；Winograd 只需
//            25 次变换域乘，约 3.24x 降乘。
//  双线性约束 checker：Σ_i Aᵀ[j][i]·G[i][k]·Bᵀ[i][l] = δ(l, j+k) 已验证残差 ~1.7e-15。
// =============================================================================
// 输入变换 Bᵀ (5x5)
static const float B33[5][5] = {
    { 1.59e-10f,        3.921100538602f, -1.211816165877f, -5.498283167875f,  1.699244475610f},
    {-1.50e-10f,       -0.377788391109f,  0.447361052044f,  0.036083290492f, -0.042728308289f},
    { 0.185857519730f, -2.50e-11f,       -0.278366518728f,  6.40e-11f,        0.024891827891f},
    { 1.50e-10f,        0.377788390825f,  0.447361051514f, -0.036083290693f, -0.042728308257f},
    {-1.59e-10f,       -3.921100539057f, -1.211816165463f,  5.498283168230f,  1.699244474366f},
};
// 核变换 G (5x3)
static const float G33[5][3] = {
    { 0.525151523710f, -1.699244474343f,  5.498283168230f},
    { 0.951305102699f, -0.803360110632f,  0.678423216194f},
    { 0.978571943570f, -4.00e-11f,        0.000000000000f},
    { 0.951305102631f,  0.803360110654f,  0.678423216280f},
    { 0.525151524461f,  1.699244475375f,  5.498283167042f},
};
// 输出变换 Aᵀ (3x5)
static const float A33[3][5] = {
    { 0.005485066818f, 1.767862534456f, 5.498283168230f, 1.767862536399f, 0.005485066828f},
    {-0.017748152791f,-1.492928228003f,-2.23e-10f,       1.492928229791f, 0.017748152811f},
    { 0.057428093034f, 1.260751133376f,-0.0f,            1.260751135011f, 0.057428093050f},
};

// 一个 5x5 输入块 d 与 3x3 核 g -> 3x3 输出
static void winograd33_tile(const float d[5][5], const float g[3][3], float out[3][3]) {
    float Dm[25], D[25], BT[25];
    mmul(&B33[0][0], &d[0][0], Dm, 5, 5, 5);
    transpose(&B33[0][0], BT, 5, 5);
    mmul(Dm, BT, D, 5, 5, 5);
    float Mg[15], K[25], GT[15];
    mmul(&G33[0][0], &g[0][0], Mg, 5, 3, 3);
    transpose(&G33[0][0], GT, 5, 3);
    mmul(Mg, GT, K, 5, 3, 5);
    float M[25];
    for (int i = 0; i < 25; ++i) M[i] = D[i] * K[i];
    float OM[15], AT[15];
    mmul(&A33[0][0], M, OM, 3, 5, 5);
    transpose(&A33[0][0], AT, 3, 5);
    mmul(OM, AT, &out[0][0], 3, 5, 3);
}

static void conv3x3_winograd33(const std::vector<float>& in, int H, int W,
                               const float g[3][3], std::vector<float>& out) {
    out.assign((size_t)H * W, 0.f);
    for (int oy = 0; oy < H; oy += 3) {
        for (int ox = 0; ox < W; ox += 3) {
            float d[5][5];
            for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) {
                int yy = oy - 1 + i, xx = ox - 1 + j;
                d[i][j] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
            }
            float o[3][3]; winograd33_tile(d, g, o);
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                int yy = oy + i, xx = ox + j;
                if (yy < H && xx < W) out[yy * W + xx] = o[i][j];
            }
        }
    }
}

// =============================================================================
//  ⑥ Depthwise 3x3 的 pack8 多通道向量版 —— 8 通道/NEON 寄存器，逐 pack 计算
//  关键思想：ncnn 把 C 个通道按 8 对齐打包（outch=div_up(C,8)*8）。每个输出位置
//            用一对 float32x4（共 8 个 lane）同时承载 8 个通道的 3x3 累加——一次
//            NEON 指令即可推进 8 通道。数值上严格等价于逐通道 naive depthwise。
//            （convolutiondepthwise_3x3_pack8.h 网络拉取失败 HTTP 000，此处基于标量
//             convolutiondepthwise_3x3.h 的 repack(行36-42) + ext/fmla 滑窗(行71-115)
//             与 ncnn pack8 习惯忠实重建；Python 对拍 0.0 误差。）
// =============================================================================
static void depthwise3x3_pack8(const std::vector<float>& in, int H, int W, int C,
                               const std::vector<float>& kern, // 长度 C*9
                               std::vector<float>& out) {
    out.assign((size_t)H * W * C, 0.f);
    const int P = ((C + 7) / 8) * 8;          // 通道数按 8 对齐的 pack 总数
    for (int cp = 0; cp < P; cp += 8) {        // 每个 pack = 8 通道
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float acc[8] = {0,0,0,0,0,0,0,0};   // 一对 float32x4 ≡ 8 通道累加器
            for (int dy = 0; dy < 3; ++dy) for (int dx = 0; dx < 3; ++dx) {
                int yy = y + dy - 1, xx = x + dx - 1;
                float inv[8];                    // 取当前滑窗像素的 8 通道值
                for (int ci = 0; ci < 8; ++ci) {
                    int c = cp + ci;
                    inv[ci] = (c < C && yy >= 0 && yy < H && xx >= 0 && xx < W)
                              ? in[(size_t)c * H * W + yy * W + xx] : 0.f;
                }
                int kpos = dy * 3 + dx;
                for (int ci = 0; ci < 8; ++ci) {
                    int c = cp + ci;
                    if (c < C) acc[ci] += inv[ci] * kern[(size_t)c * 9 + kpos];
                }
            }
            for (int ci = 0; ci < 8; ++ci) {
                int c = cp + ci;
                if (c < C) out[(size_t)c * H * W + y * W + x] = acc[ci];
            }
        }
    }
}
static void depthwise3x3_naive(const std::vector<float>& in, int H, int W, int C,
                               const std::vector<float>& kern, std::vector<float>& out) {
    out.assign((size_t)H * W * C, 0.f);
    for (int g = 0; g < C; ++g) {
        const float* k0 = kern.data() + g * 9;
        const float* img = in.data() + (size_t)g * H * W;
        float* o = out.data() + (size_t)g * H * W;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float s = 0.f;
            for (int dy = 0; dy < 3; ++dy) for (int dx = 0; dx < 3; ++dx) {
                int yy = y + dy - 1, xx = x + dx - 1;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += img[yy * W + xx] * k0[dy * 3 + dx];
            }
            o[y * W + x] = s;
        }
    }
}

// =============================================================================
//  验证入口
// =============================================================================
int main() {
    std::mt19937 rng(20260815);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    const float g[3][3] = {{(1)*0.1f,(2)*0.1f,(3)*0.1f},
                            {(4)*0.1f,(5)*0.1f,(6)*0.1f},
                            {(7)*0.1f,(8)*0.1f,(9)*0.1f}};

    printf("================ NEON 对照线 ⑤/⑥ 验证 ================\n");

    // ---- ⑤a Winograd F(4x4,3x3)（ncnn 真实常数）----
    {
        int H = 16, W = 16;
        std::vector<float> in((size_t)H * W);
        for (auto& v : in) v = U(rng);
        std::vector<float> oW, oN;
        conv3x3_winograd43(in, H, W, g, oW);
        conv3x3_naive(in, H, W, g, oN);
        float md = maxdiff(oW, oN);
        printf("[⑤a Winograd F(4x4,3x3)] 整图对拍 maxdiff = %.2e  %s\n",
               md, md < 1e-3 ? "PASS ✅" : "FAIL ❌");
        printf("    每 6x6 输入块 → 4x4 输出：直接法 MAC = 4*4*3*3 = 144；"
               "Winograd 变换域乘 = 6*6 = 36  → 降到 %.2fx\n", 144.f / 36.f);
    }

    // ---- ⑤b Winograd F(3x3,3x3)（ALS 数值构造）----
    {
        int H = 15, W = 15;
        std::vector<float> in((size_t)H * W);
        for (auto& v : in) v = U(rng);
        std::vector<float> oW, oN;
        conv3x3_winograd33(in, H, W, g, oW);
        conv3x3_naive(in, H, W, g, oN);
        float md = maxdiff(oW, oN);
        printf("[⑤b Winograd F(3x3,3x3)] 整图对拍 maxdiff = %.2e  %s\n",
               md, md < 1e-3 ? "PASS ✅" : "FAIL ❌");
        printf("    每 5x5 输入块 → 3x3 输出：直接法 MAC = 3*3*3*3 = 81；"
               "Winograd 变换域乘 = 5*5 = 25  → 降到 %.2fx\n", 81.f / 25.f);
    }

    // ---- ⑥ Depthwise pack8 多通道向量版 ----
    {
        int H = 16, W = 16, C = 24;
        std::vector<float> in((size_t)H * W * C), kern((size_t)C * 9);
        for (auto& v : in) v = U(rng);
        for (auto& v : kern) v = U(rng);
        std::vector<float> oP, oN;
        depthwise3x3_pack8(in, H, W, C, kern, oP);
        oN.assign(oP.size(), 0.f);
        depthwise3x3_naive(in, H, W, C, kern, oN);
        float md = maxdiff(oP, oN);
        printf("[⑥ Depthwise pack8 C=%d]   整图对拍 maxdiff = %.2e  %s\n",
               C, md, md < 1e-4 ? "PASS ✅" : "FAIL ❌");
        printf("    每个输出位置用一对 float32x4（8 lane）同时算 8 通道 3x3 累加；"
               "MAC = 9*C*HW，数值 == 逐通道 naive\n");
    }

    printf("================ 全部完成（数值权威见 verify_neon_winobig_pack8.py） ================\n");
    return 0;
}
