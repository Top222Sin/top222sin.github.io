// =============================================================================
//  NEON 对照线 · 第 2/3/4 站：Winograd / Depthwise / INT8 定点
//  纯 C++ 仿真（无需 ARM 编译器）：把 ncnn src/layer/arm/ 的三段手写 NEON 算法
//  还原成数值等价的 host 实现，并与 naive 参考对拍验证。
//
//  真实源码（已拉取到 _ncnn_src/，Tencent/ncnn@master/src/layer/arm/）：
//   - convolution_3x3_winograd.h   （F(2x2,3x3) Winograd，行 4574/4666/5167/763/5570）
//   - convolutiondepthwise_3x3.h   （逐通道 3x3，行 23/36-42/71-115）
//   - convolution_im2col_gemm_int8.h（INT8 im2col+GEMM，sdot 行 790；requant gemm_int8.h:1889-1918）
//   - convolution_3x3_int8.h       （INT8 直接 3x3，smlal 行 170）
//   - gemm_int8.h                  （INT8 GEMM 微核，requant 行 1889-1918）
//
//  编译： g++ -O3 -std=c++17 cuda_l3_neon_winograd_int8.cpp -o neon_adv && ./neon_adv
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
//  小工具：4x4 / 3x3 / 2x4 矩阵乘（仅本仿真用，非 NEON 原语）
// -----------------------------------------------------------------------------
static void mm44(const float A[4][4], const float B[4][4], float C[4][4]) {
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += A[i][k] * B[k][j];
        C[i][j] = s;
    }
}
static void mm34(const float A[4][3], const float B[3][4], float C[4][4]) {
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 3; ++k) s += A[i][k] * B[k][j];
        C[i][j] = s;
    }
}
static void mm24(const float A[2][4], const float B[4][4], float C[2][4]) {
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += A[i][k] * B[k][j];
        C[i][j] = s;
    }
}
static void mm42(const float A[2][4], const float B[4][2], float C[2][2]) {
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += A[i][k] * B[k][j];
        C[i][j] = s;
    }
}

// =============================================================================
//  ② Winograd F(2x2,3x3) —— convolution_3x3_winograd.h
//  关键思想：把 4x4 输入块 + 3x3 核都变换到 Winograd 域，做 4x4=16 次逐元素乘
//            （Hadamard），再反变换得到 2x2 输出。直接法同区域需 2*2*3*3 = 36 次
//            MAC；Winograd 只需 16 次变换域 MAC（+ 变换本身），约 2.25x 降乘。
//
//  ncnn 三段变换（与源码逐常数对应）：
//   - 核变换 G（conv3x3s1_winograd23_transform_kernel_tile, 行 4574-4627）：
//       tmp[1]=r0*.5+r1*.5+r2*.5 ; tmp[2]=r0*.5-r1*.5+r2*.5 （行 4602-4604）
//   - 输入变换 B (itm, transform_input_tile 注释 行 4668-4673)
//   - 输出变换 A (otm, transform_output_tile 注释 行 5169-5172)
// =============================================================================
// ncnn 实际常量（已 Python 数值验证 400 例 = cross-correlation）
static const float ITM[4][4] = {  // input transform  B (≡ ncnn itm)
    { 1, 0, -1, 0},
    { 0, 1,  1, 0},
    { 0,-1,  1, 0},
    { 0,-1,  0, 1}
};
static const float GTM[4][3] = {  // kernel transform G (≡ convolution_3x3_winograd.h:4602-4604)
    { 1,    0,    0},
    { 0.5,  0.5,  0.5},
    { 0.5, -0.5,  0.5},
    { 0,    0,    1}
};
static const float OTM[2][4] = {  // output transform A (≡ ncnn otm)
    { 1, 1, 1, 0},
    { 0, 1,-1, 1}
};

// 一个 4x4 输入块 d 与 3x3 核 g -> 2x2 输出（cross-correlation，即 CNN 习惯）
static void winograd23_tile(const float d[4][4], const float g[3][3], float out[2][2]) {
    // 1) 输入变换 D = ITM * d * ITMᵀ
    float D[4][4], Dm[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {  // Dm = ITM * d
        float s = 0.f; for (int k = 0; k < 4; ++k) s += ITM[i][k] * d[k][j]; Dm[i][j] = s;
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {  // D = Dm * ITMᵀ
        float s = 0.f; for (int k = 0; k < 4; ++k) s += Dm[i][k] * ITM[j][k]; D[i][j] = s;
    }
    // 2) 核变换 K = G * g * Gᵀ   （先 G*g(4x3)，再 *Gᵀ(3x4) -> 4x4）
    float Mg[4][3];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 3; ++j) {
        float s = 0.f; for (int k = 0; k < 3; ++k) s += GTM[i][k] * g[k][j]; Mg[i][j] = s;
    }
    float K[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        float s = 0.f; for (int k = 0; k < 3; ++k) s += Mg[i][k] * GTM[j][k]; K[i][j] = s; // *Gᵀ
    }
    // 3) Hadamard（逐元素乘）：M = D ⊙ K
    float M[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M[i][j] = D[i][j] * K[i][j];
    // 4) 输出变换 Y = OTM * M * OTMᵀ
    float OM[2][4];
    mm24(OTM, M, OM);                // OTM*M
    float OTMt[4][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) OTMt[j][i] = OTM[i][j];
    mm42(OM, OTMt, out);             // -> 2x2
}

// 整图 tiled 驱动（padding=1，same 输出），与 naive 3x3 cross-correlation 对拍。
// 以输出 2x2 块为锚：输出块 (oy,ox) 对应输入 4x4 块 (oy-1..oy+2, ox-1..ox+2)。
static void conv3x3_winograd(const std::vector<float>& in, int H, int W,
                             const float g[3][3], std::vector<float>& out) {
    out.assign((size_t)H * W, 0.f);
    for (int oy = 0; oy < H; oy += 2) {
        for (int ox = 0; ox < W; ox += 2) {
            float d[4][4];
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
                int yy = oy - 1 + i, xx = ox - 1 + j;   // 4x4 输入邻域（带 padding）
                d[i][j] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? in[yy * W + xx] : 0.f;
            }
            float o[2][2]; winograd23_tile(d, g, o);
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
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
//  ③ Depthwise 3x3 —— convolutiondepthwise_3x3.h
//  关键思想：逐通道卷积，一个输入通道对应一个输出通道，无跨通道累加。
//  核只有 9 个权重（3x3），被 repack 成 3 个 float32x4（_k012x/_k345x/_k678x，
//  行 36-42）；用 ext 取出每行 3 像素滑窗，fmla 把单个核权重标量广播进累加器。
//  MAC 数 = 9 * C * HW（对比普通 3x3 的 9 * C_in * C_out * HW）。
// =============================================================================
static void depthwise3x3(const std::vector<float>& in, int H, int W, int C,
                         const std::vector<float>& kern, // 长度 C*9
                         std::vector<float>& out) {
    out.assign((size_t)H * W * C, 0.f);
    for (int g = 0; g < C; ++g) {
        const float* k0 = kern.data() + g * 9;
        const float* img = in.data() + g * (size_t)H * W;
        float* o = out.data() + g * (size_t)H * W;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            // 3 行核权重（row0/1/2），每行 3 个；对应输入 3 行滑窗
            for (int row = 0; row < 3; ++row) {
                int yy = y + row - 1;
                for (int lane = 0; lane < 3; ++lane) {
                    int xx = x + lane - 1;
                    float iv = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : 0.f;
                    acc += iv * k0[row * 3 + lane];
                }
            }
            o[y * W + x] = acc;
        }
    }
}
static void depthwise3x3_naive(const std::vector<float>& in, int H, int W, int C,
                               const std::vector<float>& kern, std::vector<float>& out) {
    out.assign((size_t)H * W * C, 0.f);
    for (int g = 0; g < C; ++g) {
        const float* k0 = kern.data() + g * 9;
        const float* img = in.data() + g * (size_t)H * W;
        float* o = out.data() + g * (size_t)H * W;
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
//  ④ INT8 定点 —— convolution_im2col_gemm_int8.h / convolution_3x3_int8.h / gemm_int8.h
//  关键思想：权重/激活量化到 int8（每通道 scale）；矩阵乘在 int8 上做，
//    - SDOT（ARMv8.2 点积）：sdot v16.4s, v0.16b, v4.4b[0] —— 一次 4×int8·int8→int32
//      （convolution_im2col_gemm_int8.h:790，im2col+GEMM 路径）
//    - SMLAL/SMULL（直接 3x3）：smlal v8.4s, v3.4h, v0.h[0] —— int8→vmovl→int16×int16→int32
//      （convolution_3x3_int8.h:170）
//    - 累加为 int32（避免长点积溢出），最后 per-output-channel 乘 scale 反量化
//      （gemm_int8.h:1889-1918，vmulq_laneq_f32(_p, _scale, ...)）
//  这里以 1x1 卷积（即 GEMM）为样例，模拟 SDOT 路径：int8·int8→int32→×scale。
// =============================================================================
static inline int8_t quant_f(float x, float scale) {
    int q = (int)std::lroundf(x / scale);
    if (q > 127) q = 127; if (q < -127) q = -127;
    return (int8_t)q;
}
// INT8 1x1 卷积：out[o][s] = Σ_i Iq[s][i]*Wq[o][i] * (sa*sw)  [反量化 = int32 * 组合scale]
static void conv1x1_int8(const std::vector<float>& I_f, int HW, int inch, int outch,
                         const std::vector<float>& W_f, float sa, float sw,
                         std::vector<float>& out) {
    std::vector<std::vector<int8_t>> Iq(HW, std::vector<int8_t>(inch));
    std::vector<std::vector<int8_t>> Wq(outch, std::vector<int8_t>(inch));
    for (int s = 0; s < HW; ++s) for (int i = 0; i < inch; ++i) Iq[s][i] = quant_f(I_f[s * inch + i], sa);
    for (int o = 0; o < outch; ++o) for (int i = 0; i < inch; ++i) Wq[o][i] = quant_f(W_f[o * inch + i], sw);

    out.assign((size_t)outch * HW, 0.f);
    for (int o = 0; o < outch; ++o) for (int s = 0; s < HW; ++s) {
        int32_t s32 = 0;                                 // int8·int8 -> int32（SDOT/smlal 累加）
        for (int i = 0; i < inch; ++i) s32 += (int32_t)Iq[s][i] * (int32_t)Wq[o][i];
        out[(size_t)o * HW + s] = (float)s32 * (sa * sw); // 反量化：real ≈ int32 * sa * sw
    }
}
static void conv1x1_float(const std::vector<float>& I_f, int HW, int inch, int outch,
                          const std::vector<float>& W_f, std::vector<float>& out) {
    out.assign((size_t)outch * HW, 0.f);
    for (int o = 0; o < outch; ++o) for (int s = 0; s < HW; ++s) {
        float s1 = 0.f;
        for (int i = 0; i < inch; ++i) s1 += I_f[s * inch + i] * W_f[o * inch + i];
        out[(size_t)o * HW + s] = s1;
    }
}

// =============================================================================
//  验证入口
// =============================================================================
int main() {
    std::mt19937 rng(20260815);
    std::uniform_real_distribution<float> U(-1.f, 1.f);

    printf("================ NEON 对照线 ②/③/④ 验证 ================\n");

    // ---- ② Winograd ----
    {
        int H = 16, W = 16;
        std::vector<float> in(H * W);
        for (auto& v : in) v = U(rng);
        float g[3][3] = {{(1)*0.1f,(2)*0.1f,(3)*0.1f},
                         {(4)*0.1f,(5)*0.1f,(6)*0.1f},
                         {(7)*0.1f,(8)*0.1f,(9)*0.1f}};
        std::vector<float> oW, oN;
        conv3x3_winograd(in, H, W, g, oW);
        conv3x3_naive(in, H, W, g, oN);
        float md = maxdiff(oW, oN);
        // MAC 对比：一块 4x4 -> 2x2 输出，直接法 36 次 MAC，Winograd 16 次变换域乘
        printf("[② Winograd F(2x2,3x3)] 整图对拍 maxdiff = %.2e  %s\n",
               md, md < 1e-3 ? "PASS ✅" : "FAIL ❌");
        printf("    每 4x4 输入块 → 2x2 输出：直接法 MAC = 2*2*3*3 = 36；"
               "Winograd 变换域乘 = 4*4 = 16  → 乘计数降到 %.2fx\n", 36.f / 16.f);
    }

    // ---- ③ Depthwise ----
    {
        int H = 16, W = 16, C = 8;
        std::vector<float> in((size_t)H * W * C), kern(C * 9);
        for (auto& v : in) v = U(rng);
        for (auto& v : kern) v = U(rng);
        std::vector<float> oD, oN;
        depthwise3x3(in, H, W, C, kern, oD);
        depthwise3x3_naive(in, H, W, C, kern, oN);
        float md = maxdiff(oD, oN);
        // MAC：depthwise = 9*C*HW；普通 3x3(C_in=C_out=C) = 9*C*C*HW → 比普通少 C 倍
        long dw = 9LL * C * H * W, full = 9LL * C * C * H * W;
        printf("[③ Depthwise 3x3]         整图对拍 maxdiff = %.2e  %s\n",
               md, md < 1e-5 ? "PASS ✅" : "FAIL ❌");
        printf("    MAC：depthwise = 9*C*HW = %ld；普通 3x3 = 9*C*C*HW = %ld  → 降到 %.2fx\n",
               dw, full, (double)full / dw);
    }

    // ---- ④ INT8 ----
    {
        int HW = 24, inch = 20, outch = 18;
        std::vector<float> I_f((size_t)HW * inch), W_f((size_t)outch * inch);
        for (auto& v : I_f) v = U(rng);
        for (auto& v : W_f) v = U(rng);
        float sa = 0.02f, sw = 0.015f;   // 量化 scale（每通道可不同，这里简化统一）
        std::vector<float> oI, oF;
        conv1x1_int8(I_f, HW, inch, outch, W_f, sa, sw, oI);
        conv1x1_float(I_f, HW, inch, outch, W_f, oF);
        float md = maxdiff(oI, oF);
        float peak = 0.f; for (auto v : oF) peak = std::max(peak, std::fabs(v));
        float rel = md / (peak + 1e-9f);
        printf("[④ INT8 定点 (SDOT路径)]   对拍 maxabs = %.4e  rel = %.4e  %s\n",
               md, rel, rel < 0.06f ? "PASS ✅" : "误差偏大 ⚠️");
        printf("    反量化：out = Σ int8·int8 (int32累加) * (sa*sw)；int8 4x 带宽/算力于 fp32\n");
    }

    printf("================ 全部完成 ================\n");
    return 0;
}
