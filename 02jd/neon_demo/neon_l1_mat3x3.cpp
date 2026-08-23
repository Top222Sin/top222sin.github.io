// neon_l1_mat3x3.cpp
// 3x3 矩阵乘 (C = A × B)：标量计算 vs NEON 向量化 对照演示
//
//   A = [[1,2,3],      B = [[11,12,13],
//        [4,5,6],           [14,15,16],
//        [7,8,9]]           [17,18,19]]
//
// 编译(x86 标量基线, 验证算法正确):
//   g++ -O3 -std=c++17 neon_l1_mat3x3.cpp -o m3 && ./m3
// 编译(ARM 真跑 NEON):
//   aarch64-linux-gnu-g++ -O3 -march=armv8-a+simd -std=c++17 neon_l1_mat3x3.cpp -o m3 && ./m3
//
// 对应笔记本结构: §X、一个 3x3 乘,把"标量三层循环"与"NEON 行外积"放一起看
// 这是 L3 GEMM 微内核(8x4/rank-1 更新)在最小尺寸上的原型。
//
// ===== NEON 函数速查(看代码前先读) =====
//   float32x4_t  : 128-bit 向量类型, 装 4 个 float32, 即 4 条"车道(lane)" [l0,l1,l2,l3]
//   vld1q_f32(p) : 从内存连续读 4 个 float32 进一个 q 寄存器 (Load, 1=不交织, q=128bit)
//   vdupq_n_f32(s): 把单个标量 s 广播进全部 4 个 lane -> [s,s,s,s] (DUPlicate, n=标量)
//   vmlaq_f32(a,b,c): 逐 lane 算 a + b*c -> [a0+b0*c0, a1+b1*c1, a2+b2*c2, a3+b3*c3]
//                     (Multiply-Accumulate, Lane-wise, q=128bit)
//   vst1q_f32(p,v): 把 q 寄存器的 4 个 lane 连续写回内存 (STore, 1=不交织, q=128bit)
//   关键点: 一条 vmlaq 同时处理 4 个 float, 这就是 SIMD 的并行来源。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

typedef float M3[9]; // row-major: M[i*3+j]

static void print_m3(const char* name, const M3 C) {
    printf("%s =\n", name);
    for (int i = 0; i < 3; i++) {
        printf("  [");
        for (int j = 0; j < 3; j++) printf(" %6.1f", C[i * 3 + j]);
        printf(" ]\n");
    }
}

// ============ 标量: 经典三层循环 ============
// C[i][j] = Σ_k A[i][k] * B[k][j]
// 每个输出元素单独算，27 次乘法-加法，一次只处理 1 个 float。
// 例如 C[0][1] = 1*12 + 2*15 + 3*18 = 12+30+54 = 96
static void scalar_mm3x3(const M3 A, const M3 B, M3 C) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++) 
                s += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = s;
        }
}

// ============ NEON: 行外积法 (rank-1 / 行向量视角) ============
// 第 i 行输出 = Σ_k A[i][k] * (B 的第 k 行)
//   即 C[i] = A[i][0]*B[0] + A[i][1]*B[1] + A[i][2]*B[2]
// 关键: 把标量 A[i][k] 用 vdupq_n 广播成 4-lane 向量,
//       与 B 的第 k 行(也装进 4-lane, 第4元素 pad=0) 逐 lane 乘加,
//       一条 vmlaq 同时算输出行里的 4 个分量(实际用 3 个, 第4忽略)。
// 整体只需 i、k 两层循环 —— j 循环被向量化(4 lane)吞掉了。
//
// ---- 预取阶段结果(本例) ----
//   R[0] = vld1q([11,12,13,0]) = [11,12,13, 0]   // B 的第 0 行
//   R[1] = vld1q([14,15,16,0]) = [14,15,16, 0]   // B 的第 1 行
//   R[2] = vld1q([17,18,19,0]) = [17,18,19, 0]   // B 的第 2 行
//   (pad 第4个 0: vld1q 必读 4 个 float; 最后只取前 3 个输出, 第4 lane 不影响结果)
//
// ---- 计算第 0 行 i=0, A[0]=[1,2,3] 的 acc 演进 ----
//   acc = vdupq_n(0)                  = [0, 0, 0, 0]
//   k=0: aik=vdupq_n(A[0][0]=1)=[1,1,1,1]
//        acc = [0,0,0,0] + [1,1,1,1]*[11,12,13,0] = [11,12,13,0]
//   k=1: aik=vdupq_n(A[0][1]=2)=[2,2,2,2]
//        acc = [11,12,13,0] + [2,2,2,2]*[14,15,16,0] = [39,42,45,0]
//   k=2: aik=vdupq_n(A[0][2]=3)=[3,3,3,3]
//        acc = [39,42,45,0] + [3,3,3,3]*[17,18,19,0] = [90,96,102,0]
//   vst1q -> out=[90,96,102,0] -> C[0]=[90,96,102]  ✓ 与手算一致
//   (第 1、2 行同理, 系数换成 [4,5,6] / [7,8,9])
#ifdef __aarch64__
#include <arm_neon.h>
static void neon_mm3x3(const M3 A, const M3 B, M3 C) {
    // 预取 B 的 3 行, 各 pad 到 4 float, 之后内层循环不再反复加载
    float32x4_t R[3];
    for (int k = 0; k < 3; k++) {
        // buf 末尾补 0: 凑满 vld1q_f32 需要的 4 个 float
        float buf[4] = {B[k * 3 + 0], B[k * 3 + 1], B[k * 3 + 2], 0.f};
        R[k] = vld1q_f32(buf);   // 连续读 4 个 float32 -> 128-bit 寄存器(4 lane)
    }
    for (int i = 0; i < 3; i++) {
        float32x4_t acc = vdupq_n_f32(0.f);   // 输出行累加器: [0,0,0,0]
        for (int k = 0; k < 3; k++) {
            // 广播标量 A[i][k] 到 4 个 lane, 例如 A[0][0]=1 -> [1,1,1,1]
            float32x4_t aik = vdupq_n_f32(A[i * 3 + k]);
            // 逐 lane 乘加: acc[l] += aik[l] * R[k][l]
            // 一条指令同时推进输出行的 4 个分量(实际用 3 个)
            acc = vmlaq_f32(acc, aik, R[k]);
        }
        float out[4];
        vst1q_f32(out, acc);   // 把 4 个 lane 写回内存
        C[i * 3 + 0] = out[0]; // 只取前 3 个有效分量
        C[i * 3 + 1] = out[1];
        C[i * 3 + 2] = out[2];
    }
}
#endif

int main() {
    M3 A = {1, 2, 3,
            4, 5, 6,
            7, 8, 9};
    M3 B = {11, 12, 13,
            14, 15, 16,
            17, 18, 19};
    M3 Cs, Cn;
    print_m3("A", A);
    print_m3("B", B);

    scalar_mm3x3(A, B, Cs);
    print_m3("C = A*B  (scalar)", Cs);

#ifdef __aarch64__
    neon_mm3x3(A, B, Cn);
    print_m3("C = A*B  (NEON) ", Cn);
    bool ok = true;
    for (int t = 0; t < 9; t++) if (fabsf(Cs[t] - Cn[t]) > 1e-3f) { ok = false; break; }
    printf("scalar / NEON 结果一致: %s\n", ok ? "YES" : "NO");
#else
    printf("(本机非 aarch64, NEON 内核已跳过, 仅展示标量基线)\n");
#endif

    // 计时: 反复跑让 NEON 的向量化优势(减少循环/取指开销)显现
    const int N = 2000000;
    M3 tmp;
    auto t0 = std::chrono::steady_clock::now();
    for (int n = 0; n < N; n++) scalar_mm3x3(A, B, tmp);
    auto t1 = std::chrono::steady_clock::now();
    double ts = std::chrono::duration<double>(t1 - t0).count();
    printf("scalar: %.3f s / %d calls  (%.1f ns/call)\n", ts, N, ts / N * 1e9);

#ifdef __aarch64__
    auto t2 = std::chrono::steady_clock::now();
    for (int n = 0; n < N; n++) neon_mm3x3(A, B, tmp);
    auto t3 = std::chrono::steady_clock::now();
    double tn = std::chrono::duration<double>(t3 - t2).count();
    printf("NEON  : %.3f s / %d calls  (%.1f ns/call)  speedup=%.2fx\n",
           tn, N, tn / N * 1e9, ts / tn);
#endif
    return 0;
}
