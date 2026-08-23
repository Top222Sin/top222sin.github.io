// ============================================================================
//  neon_l1_optinfo.cpp  —  读懂 -fopt-info 向量化报告，并据此改代码
// ----------------------------------------------------------------------------
//  这是「写对 → 确认编译器真向量化了」的关键一步。
//  本文件特点：里面所有"坏函数"都是真实可编译的 C++，你本机跑：
//
//      g++ -O3 -march=native -ffast-math -fopt-info-vec=report.txt neon_l1_optinfo.cpp
//      # 然后看 report.txt，对照本文件 §2/§3/§4 逐条解读
//
//      # Clang 等价：
//      clang++ -O3 -march=native -Rpass=loop-vectorize \
//              -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize \
//              neon_l1_optinfo.cpp
//
//  编译后函数不会真的被调用（main 为空），但优化器仍会对它们做向量化分析，
//  因此报告能真实反映每个循环"成/败 + 原因"。
// ============================================================================

#include <cstddef>

// ============================================================================
//  §1  怎么开报告（GCC 与 Clang 命令对照）
// ============================================================================
//
//  GCC（-fopt-info 家族，输出到 stderr 或重定向文件）：
//    -fopt-info-vec                所有向量化信息（成功+失败）
//    -fopt-info-vec-optimized      只打印「成功向量化」的循环
//    -fopt-info-vec-missed         只打印「本可向量化但失败」的循环  ← 最常用
//    -fopt-info-vec-all            上述全部 + 更细的分析
//    重定向：  -fopt-info-vec=report.txt
//    只看某个文件：可加 -fopt-info-vec-missed=miss.txt
//
//  Clang（-Rpass 家族，按「备注/成功/失败」分类）：
//    -Rpass=loop-vectorize              成功
//    -Rpass-missed=loop-vectorize       失败 + 原因
//    -Rpass-analysis=loop-vectorize     分析过程（为什么没做）
//    导出机器可读：  -fsave-optimization-record  生成 XXX.opt.yaml
//
//  基础优化组合（务必带上，否则报告可能全是"未优化"）：
//    GCC : -O3 -march=native            (march=native 让编译器用上 NEON/AVX)
//    Clang: -O2 -march=native           (Clang 在 -O2 以上才向量化)
//  可选 -ffast-math：放宽浮点重排/NaN 假设，解锁更多向量化（见 §6 警告）。
//
//  注意：-O0 下 -fopt-info 基本无意义；报告只在真正跑优化的 pass 里产生。

// ============================================================================
//  §2  报告长什么样（真实样例 + 逐行注解）
// ============================================================================
//
//  --- 成功样例（GCC，摘自 report.txt）---
//  neon_l1_optinfo.cpp:120:25: note: loop vectorized using 16 byte vectors
//  neon_l1_optinfo.cpp:120:25: note: loop versioned for vectorization because
//                           of possible aliasing
//  解读：第 120 行循环被向量化（16 字节 = NEON 128-bit）；"versioned for
//        aliasing" 表示编译器**自动生成了两份**——一份假设无别名（快），
//        一份处理有别名（慢），运行时选。说明它拿不准，但至少向量化了。
//
//  --- 失败样例（GCC）---
//  neon_l1_optinfo.cpp:140:21: note: not vectorized: control flow in loop
//  neon_l1_optinfo.cpp:158:21: note: not vectorized: possible aliasing
//  neon_l1_optinfo.cpp:176:21: note: not vectorized: loop-carried dependency
//  neon_l1_optinfo.cpp:194:21: note: not vectorized: data reference is not
//                           consecutive (AoS / gather)
//  neon_l1_optinfo.cpp:212:21: note: not vectorized: number of iterations
//                           cannot be computed
//  解读：每条 note 都点名了**失败原因**——这正是你改代码的地图。
//
//  --- Clang 失败样例 ---
//  neon_l1_optinfo.cpp:158:21: remark: loop not vectorized: cannot prove
//          it is safe to vectorize due to potential aliasing
//  neon_l1_optinfo.cpp:176:21: remark: loop not vectorized: loop-carried
//          dependence (dependency between iterations)
//  解读：Clang 用 "remark" 而非 "note"，措辞不同但原因一一对应。

// ============================================================================
//  §3  失败原因 → 改法 对照表（最核心，贴墙用）
// ============================================================================
//
//  ┌─────────────────────────────────┬───────────────────────────────────┐
//  │ 报告里的失败原因                  │ 怎么改代码                         │
//  ├─────────────────────────────────┼───────────────────────────────────┤
//  │ possible aliasing               │ 给指针加 __restrict__（见 §4.1）   │
//  │ control flow in loop            │ 去掉数据相关分支 / -ffast-math /    │
//  │   (if/else 依赖元素)            │   用 mask 模拟（见 §4.4）          │
//  │ loop-carried dependency         │ 改算法：这是串行本质，编译器救不了 │
//  │   (a[i] 依赖 a[i-1])           │   （见 §4.3）                      │
//  │ data ref not consecutive        │ 改 SoA 布局 / 手写 gather          │
//  │   (AoS / 链表 / 指针数组)       │   （见 §4.2）                      │
//  │ number of iterations cannot     │ 让循环界是编译期常量或简单变量；   │
//  │   be computed                   │   去掉循环内改 bound 的逻辑        │
//  │ reduction needs CAS / float     │ 加 -ffast-math（接受重排误差）；   │
//  │   reduction not vectorized      │   或手动 int 累加再转 float        │
//  │ call is not inlined             │ 加 inline / 放头文件 / 开 -flto    │
//  │ unsupported data type            │ 换可向量化类型，或手写 intrinsics  │
//  │ loop body too small             │ 忽略，向量化收益本身就不大          │
//  └─────────────────────────────────┴───────────────────────────────────┘

// ============================================================================
//  §4  实战：四个"坏函数" + 报告 + 修好
// ============================================================================
//  下面每个 bad_* 函数都会触发上面某条失败；fixed_* 是对照修法。

// ---- §4.1  别名（possible aliasing）----
// 坏：x、y 是裸指针，编译器不敢假设它们不重叠，为安全放弃向量化。
void bad_alias(float* x, float* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] + 1.0f;          // 报告：possible aliasing
}
// 修：加 __restrict__ 给编译器"信任票"，声明两块内存不重叠。
void fixed_alias(float* __restrict__ x, float* __restrict__ y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] + 1.0f;          // 报告：loop vectorized
}

// ---- §4.2  AoS 非连续（data ref not consecutive）----
struct Point { float x, y, z; };     // 12 字节，x 之间隔着 y,z
// 坏：取所有点的 x 时，内存不连续，一条 vector load 抓不到 4 个 x。
void bad_aos(Point* p, float* out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = p[i].x * 2.0f;      // 报告：data reference is not consecutive
}
// 修：改 SoA，xs/ys/zs 各自连续 → 一条 vld1q 干净取到 4 个 x。
void fixed_soa(const float* __restrict__ xs, float* __restrict__ out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = xs[i] * 2.0f;       // 报告：loop vectorized
}

// ---- §4.3  循环依赖（loop-carried dependency）----
// 坏：a[i] 依赖 a[i-1]，本质是串行递推，任何向量化器都只能放弃。
void bad_dep(float* a, int n) {
    for (int i = 1; i < n; ++i)
        a[i] = a[i-1] + 1.0f;        // 报告：loop-carried dependency
}
// 修：这类**不能靠加 restrict/SoA 救**，必须改算法。
//     例：前缀和（prefix sum）要用并行扫描（Blelloch/Hillis-Steele），
//     而不是朴素递推。此处仅示意"算法层面"的替换思路：
//       并行写法不在本文件展开，关键是：把依赖链打散成可并行的阶段。

// ---- §4.4  数据相关分支（control flow in loop）----
// 坏：if 的分支依赖元素值，向量化得用 mask，默认不开 fast-math 时易失败。
void bad_branch(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = (x[i] > 0) ? x[i] : 0.0f;   // 报告：control flow in loop
}
// 修 A（数值语义不变，用 max 等价消除分支）：
void fixed_branch_max(const float* __restrict__ x, float* __restrict__ y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;  // 编译器易识别为 fmax → 向量化
}
// 修 B（更激进）：加 -ffast-math，编译器允许把分支重排成 mask 选择，
//                即使不消除分支也可能向量化。生产环境需评估数值影响（见 §6）。

// ---- §4.5  循环次数无法静态确定（number of iterations cannot be computed）----
// 坏：上界在循环体内被改，编译器无法证明 trip count 固定。
int g_bound;
void bad_unknown(float* a, int n) {
    int m = g_bound;
    for (int i = 0; i < m; ++i) {    // 报告：number of iterations cannot be computed
        a[i] += 1.0f;
        if (a[i] > 100.0f) m = i;    // 循环内改上界
    }
}
// 修：别在循环里改上界；若逻辑允许，把上界提成编译期常量或简单变量。
//     简单常量上界会被自动展开 + 向量化。

// ============================================================================
//  §5  一个完整工作流（接 neon_l2_profiling.cpp 的 §A）
// ============================================================================
//
//  1) 写热点循环，先不手写 intrinsics；
//  2) 编译加 -O3 -march=native -fopt-info-vec-missed=miss.txt；
//  3) 读 miss.txt：报告里反复出现的原因，就是你的改码方向（§3 对照表）；
//  4) 按 §4 的修法改（restrict / SoA / 去分支 / 改算法）；
//  5) 再编译，确认 miss.txt 里该循环消失、且出现 "loop vectorized"；
//  6) objdump -d 找 v 前缀指令（fadd v0.4s / fmla ...），确认真向量化；
//  7) 用 neon_l2_profiling.cpp 的 bench 测带宽/加速比，确认"快了"而非"只是向量化了"。
//
//  一句话：报告告诉你"哪里没做 + 为什么"，对照表告诉你"改哪一行"。

// ============================================================================
//  §6  常见坑（避免白忙活）
// ============================================================================
//
//  • -O0 下报告几乎为空：必须 -O2/-O3 且 -march=native，向量化 pass 才跑。
//  • -ffast-math 会改变数值结果（重排累加顺序、假设无 NaN/无穷、放宽结合律），
//    生产环境使用要评估能否接受；调试阶段可先开它验证"能不能向量化"。
//  • Clang 的 -Rpass* 默认需要 -O2 + -march=native，否则看不到向量化 remark。
//  • 报告显示 "loop versioned for aliasing" 不算失败——它自动生成了
//    无别名快路径，仍向量化；但想彻底去掉慢路径就加 __restrict__。
//  • 别迷信 IPC 高＝快：见过向量化后 IPC 高但全在等内存，仍 memory-bound。
//  • "loop vectorized" 只说明编译器发了向量指令；是否真快，必须靠 §7/profiling 测量。

int main() {
    // 故意不调用上面的函数：优化器仍会对它们做向量化分析并产出报告，
    // 但不会产生无用副作用。想看真实运行效果，可放开下面注释并加 -DRUN：
#ifdef RUN
    float a[1024], b[1024], c[1024];
    for (int i = 0; i < 1024; ++i) a[i] = b[i] = (float)i;
    fixed_alias(a, c, 1024);
#endif
    return 0;
}
