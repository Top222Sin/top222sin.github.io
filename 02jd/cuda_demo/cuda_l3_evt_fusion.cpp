/*
 * cuda_l3_evt_fusion.cpp
 * ============================================================================
 * 工业级源码线 · 第 4 站
 * ORT FusedMatMul（带激活 / Scale / Bias 的融合 GEMM）
 *      vs
 * CUTLASS EVT 访客树（Epilogue Visitor Tree）
 * ----------------------------------------------------------------------------
 * 端到端对照：一个 ONNX 算子如何映射到
 *        "一棵 epilogue 访客树  +  一组 MMA 形状"
 *
 * 沙箱无 GPU / 无 nvcc。本文件核心部分（mini-EVT + ORT 枚举 epilogue + CPU 对拍）
 * 是纯 C++，可直接 g++ 跑：
 *     g++ -O3 -std=c++17 cuda_l3_evt_fusion.cpp -o evt && ./evt
 * 末尾 __CUDACC__ 守护段给出"真机 CUTLASS 映射"示意（需 nvcc + CUTLASS，本机不编译）。
 *
 * 真实源码落点（已备份于 _ort_src/ 与 _cutlass_src/）：
 *   ORT   : contrib_ops/cuda/math/fused_matmul.cc:19,27-30
 *           —— FusedMatMul 注册到 onnxruntime::cuda::MatMul<T> 内核；
 *           ComputeInternal 再派发到 TunableMatMul / ComputeDefault
 *           (core/providers/cuda/math/matmul.cc:133-138)。
 *           融合经由 cuBLASLt 的【枚举 epilogue】完成（见正文）。
 *   CUTLASS: include/cutlass/epilogue/collective/default_epilogue.hpp:63,239,248
 *           —— DefaultEpilogue 持有一个 ThreadEpilogueOp（即 EVT 树），
 *           对累加器逐元素调用 epilogue_op(accum, fragC)。
 * ============================================================================
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <cassert>

// ============================ 数值 / 激活 ============================
static float h_gelu(float x){ static const float c=0.7978845608028654f;
  return 0.5f*x*(1.0f+std::tanh(c*(x+0.044715f*x*x*x))); }
static float h_relu(float x){ return x>0.f?x:0.f; }
static float h_swish(float x){ return x/(1.0f+std::exp(-x)); }

// ===================== 极简 EVT：epilogue 访客树 =====================
// 模仿 CUTLASS 3.x 的 Epilogue Visitor Tree（仅教学，纯 C++ 可跑）：
//   AuxLoad   : 仿 VisitorAuxLoad   —— 从某张量取数（bias=列广播 / C=逐元素）
//   UnaryOp   : 仿 VisitorCompute(一元) —— 对累加器做激活
//   BinaryOp  : 仿 VisitorCompute(二元) —— 累加器(左) ⊕ 孩子节点(右)
//   StoreD    : 仿 VisitorAuxStore  —— 把结果写回 D
// 树的后序求值：根=StoreD；accum 来自 MMA 累加器（逐元素）自顶向下贯穿；
// 叶子 AuxLoad 忽略 accum；BinaryOp 的左操作数永远是 accum，右操作数是孩子结果。
struct Visitor {
  virtual float eval(float accum, int m, int n) const = 0;
  virtual ~Visitor() = default;
};

struct AuxLoad : Visitor {                 // 仿 VisitorAuxLoad
  enum Kind { BIAS, C_TENSOR };
  Kind kind; const std::vector<float>& data; int N;
  AuxLoad(Kind k, const std::vector<float>& d, int n):kind(k),data(d),N(n){}
  float eval(float, int m, int n) const override {
    return kind==BIAS ? data[n] : data[m*N+n];   // BIAS：列广播；C：逐元素
  }
};

struct UnaryOp : Visitor {                 // 仿 VisitorCompute（一元激活）
  std::function<float(float)> f; const Visitor* child;   // child==nullptr ⇒ 直接作用于 accum
  UnaryOp(std::function<float(float)> g, const Visitor* c=nullptr):f(g),child(c){}
  float eval(float accum,int m,int n) const override {
    return f(child ? child->eval(accum,m,n) : accum);
  }
};

struct BinaryOp : Visitor {                // 仿 VisitorCompute（二元：accum ⊕ child）
  std::function<float(float,float)> f; const Visitor* child;
  BinaryOp(std::function<float(float,float)> g, const Visitor* c):f(g),child(c){}
  float eval(float accum,int m,int n) const override {
    return f(accum, child->eval(accum,m,n));
  }
};

struct StoreD : Visitor {                  // 仿 VisitorAuxStore
  std::vector<float>& D; int N; const Visitor* child;
  StoreD(std::vector<float>& d,int n,const Visitor* c):D(d),N(n),child(c){}
  float eval(float accum,int m,int n) const override {
    float r = child->eval(accum,m,n); D[m*N+n]=r; return r;
  }
};

// 用访客树表达 "D = activation( A·B + bias )"
//   树： StoreD( UnaryOp(act, BinaryOp(plus, AuxLoad(BIAS))) )
static std::vector<float> run_evt(const std::vector<float>& Accum, int M, int N,
                                  const std::vector<float>* bias,
                                  std::function<float(float)> act) {
  std::unique_ptr<Visitor> biasLeaf, combiner, actNode;
  const Visitor* inner = nullptr;          // 喂给激活节点的：accum 还是 (accum+bias)
  if (bias) {
    biasLeaf = std::make_unique<AuxLoad>(AuxLoad::BIAS,*bias,N);
    combiner = std::make_unique<BinaryOp>([](float a,float b){return a+b;}, biasLeaf.get());
    inner = combiner.get();
  }
  actNode = std::make_unique<UnaryOp>(act, inner);
  std::vector<float> D(M*N);
  StoreD store(D,N, actNode.get());
  for (int m=0;m<M;++m) for(int n=0;n<N;++n)
    store.eval(Accum[m*N+n], m, n);          // accum 来自 MMA 累加器
  return D;
}

// ===================== ORT：闭集枚举 epilogue =====================
// cublasLt 的 epilogue 是【固定枚举】，ORT FusedMatMul 只能从中选一个：
//   CUBLASLT_EPILOGUE_DEFAULT / BIAS / RELU / BIAS_RELU / GELU / BIAS_GELU / DRELU
// MMA 的形状/调度由 cuBLASLt 内部 autotuner 决定 —— ORT 不指定 tile。
enum OrtEpilogueKind {
  EPI_DEFAULT, EPI_BIAS, EPI_RELU, EPI_BIAS_RELU,
  EPI_GELU, EPI_BIAS_GELU, EPI_DRELU, EPI_UNSUPPORTED
};
struct OrtFusedMatMul {
  float alpha=1.f; bool transA=false,transB=false;
  OrtEpilogueKind epi=EPI_DEFAULT; const std::vector<float>* bias=nullptr;
};

// 把 "有没有 bias + 激活名" 映射到 cuBLASLt 的【枚举】epilogue（闭集）
static OrtEpilogueKind ort_map_epi(bool has_bias, const char* activation){
  if (!activation || activation[0]=='\0' || strcmp(activation,"None")==0)
    return has_bias ? EPI_BIAS : EPI_DEFAULT;
  if (strcmp(activation,"Relu")==0)  return has_bias ? EPI_BIAS_RELU : EPI_RELU;
  if (strcmp(activation,"Gelu")==0)  return has_bias ? EPI_BIAS_GELU : EPI_GELU;
  // 闭集之外：cublasLt 没有原生 epilogue → 无法融合，必须退化为独立 kernel
  return EPI_UNSUPPORTED;   // 例：FastGelu / Swish / 任意自定义
}

// 枚举 epilogue 的"执行"（CPU 模拟 cublasLt 在 GEMM 末尾做的事）
static std::vector<float> ort_run_epilogue(const std::vector<float>& G, int M,int N,
                                           OrtEpilogueKind epi, const std::vector<float>* bias){
  std::vector<float> D(M*N);
  bool need_bias = (epi==EPI_BIAS||epi==EPI_BIAS_RELU||epi==EPI_BIAS_GELU);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){
    float v = G[m*N+n];
    if (need_bias) v += (*bias)[n];
    switch(epi){
      case EPI_RELU: case EPI_BIAS_RELU: v=h_relu(v); break;
      case EPI_GELU: case EPI_BIAS_GELU: v=h_gelu(v); break;
      case EPI_DRELU: v = (v>0.f)? v : 0.f; break;   // 训练用
      default: break;                                // DEFAULT / BIAS：无激活
    }
    D[m*N+n]=v;
  }
  return D;
}

// ORT FusedMatMul 驱动：GEMM(α·A·B) → 套用枚举 epilogue
// 返回 true 表示成功融合；false 表示闭集外、必须退化成独立 kernel
static bool ort_fused_matmul(const std::vector<float>& A, const std::vector<float>& B,
                             int M,int N,int K, OrtFusedMatMul op,
                             const char* activation, std::vector<float>& out){
  std::vector<float> G(M*N, 0.f);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){
    float s=0.f; for(int k=0;k<K;++k) s += A[m*K+k]*B[k*N+n];
    G[m*N+n] = op.alpha * s;          // α·A·B（MMA 由 cuBLASLt 内部选形状）
  }
  OrtEpilogueKind epi = ort_map_epi(op.bias!=nullptr, activation);
  if (epi == EPI_UNSUPPORTED) return false;   // 退化：MatMul + 独立激活 kernel
  out = ort_run_epilogue(G, M, N, epi, op.bias);
  return true;
}

// ============================ 工具 ============================
static std::vector<float> naive_gemm(const std::vector<float>& A,const std::vector<float>& B,
                                     int M,int N,int K){
  std::vector<float> C(M*N,0.f);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){
    float s=0.f; for(int k=0;k<K;++k) s+=A[m*K+k]*B[k*N+n];
    C[m*N+n]=s;
  }
  return C;
}
static float max_abs_diff(const std::vector<float>& a,const std::vector<float>& b){
  float d=0.f; for(size_t i=0;i<a.size();++i) d=std::max(d,std::fabs(a[i]-b[i])); return d;
}
static void print_vec(const char* name,const std::vector<float>& v,int cols){
  printf("  %-10s = [", name);
  for(size_t i=0;i<v.size();++i){ printf("%.3f%s", v[i], ((i+1)%cols==0)?" | ":" "); }
  printf("]\n");
}

// ============================ 对拍 ============================
static int g_fail=0;
static void check(const char* tag,const std::vector<float>& got,
                  const std::vector<float>& ref,int M,int N){
  float d=max_abs_diff(got,ref);
  bool ok = d < 1e-4f;
  printf("  [%-10s] max|Δ| = %.2e  → %s\n", tag, d, ok?"PASS":"FAIL");
  if(!ok) g_fail++;
}

int main(){
  const int M=4, N=3, K=5;
  std::vector<float> A(M*K), B(K*N), bias(N);
  for(int i=0;i<M*K;++i) A[i] = 0.1f*i - 0.3f;
  for(int i=0;i<K*N;++i) B[i] = 0.2f*i - 0.5f;
  for(int n=0;n<N;++n)   bias[n] = 0.5f*n - 0.2f;
  float alpha = 1.5f;

  printf("==================== 端到端对照：FusedMatMul = GELU( α·A·B + bias ) ====================\n");
  printf("M=%d N=%d K=%d  α=%.2f  (bias 列广播)\n\n", M,N,K,alpha);

  // 累加器（真实场景下由 MMA tile 产出 α·A·B；此处用朴素 GEMM 代劳并乘 α）
  std::vector<float> Accum = naive_gemm(A,B,M,N,K);
  for(auto& x: Accum) x *= alpha;            // α 属于 MMA 输出，在 epilogue 之前生效

  // —— 路线 A：CUTLASS 风格 —— 一棵 epilogue 访客树
  std::vector<float> D_evt = run_evt(Accum, M, N, &bias, h_gelu);

  // —— 路线 B：ORT 风格 —— 闭集枚举 epilogue（cublasLt）
  OrtFusedMatMul op; op.alpha=alpha; op.bias=&bias;
  std::vector<float> D_ort; bool fused = ort_fused_matmul(A,B,M,N,K,op,"Gelu",D_ort);

  // —— 参考：直接公式 ——
  std::vector<float> D_ref(M*N);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n)
    D_ref[m*N+n] = h_gelu(Accum[m*N+n] + bias[n]);

  print_vec("α·A·B", Accum, N);
  print_vec("bias", bias, N);
  print_vec("D(EVT)", D_evt, N);
  print_vec("D(ORT)", D_ort, N);
  print_vec("D(ref)", D_ref, N);
  printf("\n  ORT FusedMatMul(\"Gelu\") 融合成功? %s  (枚举 epilogue = EPI_BIAS_GELU)\n", fused?"YES":"NO");
  check("EVT",   D_evt, D_ref, M, N);
  check("ORT",   D_ort, D_ref, M, N);

  // =================== 关键对照：开放 vs 闭集（Swish 测试） ===================
  printf("\n==================== 开放性对照：新增 Swish 激活 ====================\n");
  // CUTLASS：只需换一个访客节点，MMA 形状一行不动（开放组合）
  std::vector<float> D_evt_swish = run_evt(Accum, M, N, &bias, h_swish);
  // ORT：Swish 不在 cublasLt 枚举内 → 闭集外 → 融合失败，必须退化成 MatMul + 独立 kernel
  std::vector<float> D_ort_swish;
  bool fused_sw = ort_fused_matmul(A,B,M,N,K,op,"Swish",D_ort_swish);
  printf("  CUTLASS EVT : 加一个 UnaryOp(swish) 节点即可，MMA(GemmShape)不变 → D 形状 = %zu  ✅ 可融合\n",
         D_evt_swish.size());
  printf("  ORT Fused   : ort_map_epi(\"Swish\") = %s → 闭集外，必须退化为 [MatMul] + [独立 Swish kernel] ❌ 不能融合\n",
         fused_sw ? "OK" : "EPI_UNSUPPORTED");

  // =================== 闭集清单（cublasLt 原生支持） ===================
  printf("\n==================== ORT/cuBLASLt 可融合 epilogue 闭集 ====================\n");
  printf("  { DEFAULT, BIAS, RELU, BIAS_RELU, GELU, BIAS_GELU, DRELU }\n");
  printf("  → 任意不在此集合的激活/算子链 = 无法单 kernel 融合，需要图优化拆成独立 kernel。\n");
  printf("  → MMA tile / 调度由 cuBLASLt 内部 autotuner 选择，ORT 不暴露、不可控。\n");

  printf("\n==================== 结论 ====================\n");
  printf("  同一 ONNX FusedMatMul：\n");
  printf("    ORT     → 一个【枚举 epilogue】(选有限种之一，现代 cuBLASLt 10+ 种) + bias 指针；MMA 形状对 ORT 不透明。\n");
  printf("    CUTLASS → 一棵【epilogue 访客树】(任意组合) + 一组 GemmShape<BM,BN,BK>/<WM,WN,WK>/<IM,IN,IK>。\n");
  printf("  差异收敛到一点：输出算子是否可被【用户任意组合】。\n");
  if (g_fail==0) printf("\n>>> 全部对拍 PASS ✅\n");
  else           printf("\n>>> 有 %d 项 FAIL ❌\n", g_fail);
  return g_fail? 1 : 0;
}

/* ============================================================================
 * __CUDACC__ 守护段：真机 CUTLASS 映射示意（需 nvcc + CUTLASS，本机不编译）
 * ----------------------------------------------------------------------------
 * 同一个 FusedMatMul = GELU( α·A·B + bias )，在 CUTLASS 3.x 里被拆成
 *   (1) 一组 MMA 形状 —— 全部显式、用户可控
 *   (2) 一棵 epilogue 访客树（EVT）—— 用户任意组合
 *
 *   using Gemm = cutlass::gemm::device::Gemm<
 *     ElementA, LayoutA, ElementB, LayoutB,
 *     ElementC, LayoutC,
 *     ElementAccumulator,
 *     cutlass::arch::OpClassTensorOp,
 *     cutlass::arch::Sm80,
 *     cutlass::gemm::GemmShape<128,128,32>,   // ← Threadblock 级 MMA 形状
 *     cutlass::gemm::GemmShape< 64, 64,32>,   // ← Warp 级
 *     cutlass::gemm::GemmShape< 16, 8, 16>,   // ← Instruction 级 (Tensor Core)
 *     // ↓ epilogue：一棵 EVT 访客树（示意类型名，以官方头文件为准）
 *     cutlass::epilogue::collective::Sm80EVT<
 *       cutlass::epilogue::collective::VisitorAuxStore<ElementD, StrideD, ThreadMapD, CopyOp>, // 根：写 D
 *       cutlass::epilogue::collective::Sm80EVT<
 *         cutlass::epilogue::collective::VisitorCompute<cm::gelu, ElementCompute, ElementD>,   // GELU
 *         cutlass::epilogue::collective::Sm80EVT<
 *           cutlass::epilogue::collective::VisitorCompute<plus, ElementCompute, ElementD>,     // α·acc + bias
 *           cutlass::epilogue::collective::VisitorAuxLoad<ElementBias, StrideBias, ...>,       // 加载 bias(列广播)
 *           /* 累加器 acc 作为左操作数隐式传入 *\/
 *         >
 *       >
 *     >,
 *     cutlass::gemm::NoSwizzle, 2,   // kStages：双缓冲
 *
 *   而 DefaultEpilogue（default_epilogue.hpp:63,239,248）对累加器逐元素调用：
 *       FragDType fragD = epilogue_op(accumulators(i), fragC);   // epilogue_op 即上面那棵 EVT
 *   即：MMA 段对所有 GEMM 系算子【固定不变】，差异全在 epilogue 访客树这一个可替换点。
 * ==========================================================================*/
#ifdef __CUDACC__
// （本段仅在有 CUDA 工具链 + CUTLASS 时参与编译，此处留空，示意见上方注释。）
#endif
