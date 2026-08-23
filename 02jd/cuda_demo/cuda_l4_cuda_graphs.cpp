// ============================================================================
//  cuda_l4_cuda_graphs.cpp
//  L4（工业扩展）· CUDA Graphs：把整条 kernel 链固化成「可重放图」
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l4_cuda_graphs.cpp -o cg4 && ./cg4
//  （真机 CUDA Graphs API 由 #ifdef __CUDACC__ 守护，g++ 下跳过，仅作教材）
//
//  讲清楚一件事：推理/训练里「同一条 kernel 链被 launch 成千上万次」时，
//  per-launch 的 host 端开销会主导。CUDA Graphs 把这条链**捕获成一张图**，
//  之后每次「重放」只付一次 cudaGraphLaunch 的代价——把 host 端的
//  N·K 次 launch 压成 1 次 capture + N 次 graph-launch。
//
//  本文件用纯 C++ 复现：
//    Part A 正确性 —— graph 重放 N 次 == stream 逐次 launch N 次（且==闭式解）
//    Part B 收益   —— host 端 launch 开销模型，量化加速比与渐近上界
//    Part C 真机   —— #ifdef __CUDACC__ 守护的 stream-capture / graph API 片段
// ============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// -------------------- 模型参数（CUDA Graphs 教学常量）--------------------
// 单次 kernel launch 的 host 端开销（CPU→GPU 派发 + 依赖检查 + 入队），
// 文献常见量级 ~3-10 µs，这里取 5 µs。
static const double L_STREAM   = 5.0;   // µs / 每次逐 kernel launch
// 单次 cudaGraphLaunch 的 host 端开销（一图打包 K 个 kernel，只一次派发），
// 量级 ~1-3 µs，这里取 1 µs。
static const double L_GRAPH    = 1.0;   // µs / 每次图重放
// 一次性 capture 开销（建图、记录节点与依赖），量级 ~几十~几百 µs，取 100 µs。
static const double C_CAPTURE  = 100.0; // µs（一次性，被 N 次重放摊薄）

// -------------------- 一条 kernel 链（一个 step 内的 K 个 kernel）----------
// 每个 kernel 都是对向量 y 的就地变换；整链构成一个可重复的 step。
// 教学常量选 ac=A*C=0.4（保证迭代稳定、且非零）。
static const double A_ = 0.5, B_ = 0.3, C_ = 0.8, D_ = 0.1;

static void k0_scale (std::vector<double>& y){ for(double& v:y) v = A_ * v; }   // y = a·y
static void k1_shift (std::vector<double>& y){ for(double& v:y) v = v + B_; }   // y = y + b
static void k2_scale (std::vector<double>& y){ for(double& v:y) v = C_ * v; }   // y = c·y
static void k3_shift (std::vector<double>& y){ for(double& v:y) v = v + D_; }   // y = y + d

// 一个 step = 顺序执行 4 个 kernel（k0→k1→k2→k3 线性依赖）
static void chain_step(std::vector<double>& y){
  k0_scale(y); k1_shift(y); k2_scale(y); k3_shift(y);
}

// 闭式解：单步 y' = ac·y + (c·b + d)，ac = A·C
static std::vector<double> closed_form(const std::vector<double>& y0, int N){
  const double ac = A_ * C_, off = C_ * B_ + D_;
  std::vector<double> r(y0.size());
  for(size_t i=0;i<y0.size();++i)
    r[i] = std::pow(ac, N) * y0[i] + off * (1.0 - std::pow(ac, N)) / (1.0 - ac);
  return r;
}

// ============================ Part A: 正确性 ==============================
// 最小「图」抽象：节点 + 依赖，捕获一次后按拓扑序重放。
// 这里用显式依赖表 + Kahn 拓扑排序，忠实模拟「capture 记录依赖、replay 按依赖执行」。
struct Graph {
  using Op = void(*)(std::vector<double>&);
  std::vector<Op>         nodes;        // 节点函数
  std::vector<std::vector<int>> deps;   // deps[i] = 节点 i 的前驱
  std::vector<int>        topo;         // 拓扑序

  void build(){
    // 线性链：k0 ← k1 ← k2 ← k3（每个依赖前一个，deps[i] = 节点 i 的前驱表）
    nodes = {k0_scale, k1_shift, k2_scale, k3_shift};
    deps  = {{}, {0}, {1}, {2}};
    // Kahn 拓扑排序：节点 i 的入度 = 其前驱个数
    std::vector<int> indeg(deps.size(), 0);
    for(size_t i=0;i<deps.size();++i) indeg[i] = (int)deps[i].size();
    std::vector<int> q;
    for(size_t i=0;i<indeg.size();++i) if(indeg[i]==0) q.push_back((int)i);
    topo.clear();
    while(!q.empty()){
      int u = q.back(); q.pop_back();
      topo.push_back(u);
      // 找出以 u 为前驱的节点 v（即 u ∈ deps[v]），减其入度
      for(size_t v=0; v<deps.size(); ++v){
        auto it = std::find(deps[v].begin(), deps[v].end(), u);
        if(it != deps[v].end() && --indeg[v]==0) q.push_back((int)v);
      }
    }
  }
  void replay(std::vector<double>& y) const {
    for(int idx : topo) nodes[idx](y);   // 按拓扑序执行节点
  }
};

// ============================ Part B: 收益模型 ============================
static double t_stream(int n, int k){ return (double)n * k * L_STREAM; }
static double t_graph (int n, int k){ return C_CAPTURE + (double)n * L_GRAPH; }
static double speedup (int n, int k){ return t_stream(n,k) / t_graph(n,k); }
static double asym    (int    k){ return (double)k * L_STREAM / L_GRAPH; }

int main(){
  printf("===== L4  CUDA Graphs：把 kernel 链固化成可重放图 =====\n");

  int passed = 0, failed = 0;
  auto chk = [&](const char* name, bool ok){
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    ok ? ++passed : ++failed;
  };

  // ---------- Part A：正确性 ----------
  const int K = 4;
  const int N_REP = 1000;
  // 确定性输入（无 RNG → C++ 与 Python 逐位一致）
  std::vector<double> y0;
  for(int i=0;i<256;++i) y0.push_back(((i % 7) - 3) * 0.37 + 0.1);

  // stream 路径：N_REP 次，每次把链从头跑一遍（每 kernel 一次 launch）
  std::vector<double> y_stream = y0;
  for(int r=0;r<N_REP;++r) chain_step(y_stream);

  // graph 路径：捕获一次（建图），重放 N_REP 次（每次 cudaGraphLaunch 内部按拓扑序跑 K 节点）
  Graph g; g.build();
  std::vector<double> y_graph = y0;
  for(int r=0;r<N_REP;++r) g.replay(y_graph);

  std::vector<double> y_cf = closed_form(y0, N_REP);

  double max_sg = 0, max_sc = 0, max_gc = 0;
  for(size_t i=0;i<y0.size();++i){
    max_sg = std::max(max_sg, std::fabs(y_stream[i] - y_graph[i]));
    max_sc = std::max(max_sc, std::fabs(y_stream[i] - y_cf[i]));
    max_gc = std::max(max_gc, std::fabs(y_graph[i]  - y_cf[i]));
  }
  printf("\n--- Part A: 正确性（重放 N=%d 次, 每图 K=%d kernel）---\n", N_REP, K);
  printf("  stream vs graph  max|diff| = %.2e\n", max_sg);
  printf("  stream vs 闭式   max|diff| = %.2e\n", max_sc);
  printf("  graph  vs 闭式   max|diff| = %.2e\n", max_gc);
  chk("stream 与 graph 重放结果逐元素一致", max_sg < 1e-9);
  chk("stream 与闭式解一致",               max_sc < 1e-9);
  chk("graph  与闭式解一致",               max_gc < 1e-9);

  // ---------- Part B：收益 ----------
  printf("\n--- Part B: host 端 launch 开销模型（µs）---\n");
  printf("  [随重放次数 N，K=%d]\n", K);
  for(int n : {100, 1000, 10000}){
    printf("    N=%-6d  stream=%.0f  graph=%.0f  加速比=%.2fx (渐近上界=%.0fx)\n",
           n, t_stream(n,K), t_graph(n,K), speedup(n,K), asym(K));
  }
  printf("  [随每图 kernel 数 K，N=1000]\n");
  for(int k : {1, 4, 10, 20}){
    printf("    K=%-3d  stream=%.0f  graph=%.0f  加速比=%.2fx (渐近上界=%.0fx)\n",
           k, t_stream(1000,k), t_graph(1000,k), speedup(1000,k), asym(k));
  }
  chk("N=1000 加速比 > 10x",           speedup(1000,K) > 10.0);
  chk("K=20 渐近加速比 > 50x",         asym(20) > 50.0);
  chk("graph host 开销 < stream (N=100)", t_graph(100,K) < t_stream(100,K));
  chk("加速比随 N 单调增(100→1e4)",     speedup(10000,K) >= speedup(100,K));

  // ---------- 解读 ----------
  printf("\n[解读] 加速比来自「把 N·K 次逐 kernel launch 压成 1 次 capture + N 次 graph-launch」。\n"
         "       N 越大、每图 kernel 越多，收益越大；渐近上界 = K·L_stream / L_graph。\n"
         "       K=4 时上界≈%gx；K=20 时上界≈%gx（多小 kernel 反复 launch 的场景最划算）。\n",
         (int)asym(K), (int)asym(20));

  // ---------- Part C：真机片段（g++ 下不编译）----------
#ifdef __CUDACC__
  // —— 片段 1：stream capture（最常用，几乎零改造）——
  //   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  //   for (int i=0;i<N;i++) myKernel<<<grid,block,0,stream>>>(...);  // 照常写 launch
  //   cudaStreamEndCapture(stream, &graph);
  //   cudaGraphInstantiate(&graphExec, graph, NULL);
  //   for (int i=0;i<N_REP;i++) cudaGraphLaunch(graphExec, stream);  // 每次重放只 1 次派发
  //   cudaStreamSynchronize(stream);
  //   cudaGraphExecDestroy(graphExec); cudaGraphDestroy(graph);
  //
  // —— 片段 2：Graph API（显式节点 + 依赖，适合动态构图）——
  //   cudaGraphCreate(&graph, 0);
  //   cudaGraphNode_t n[4]; cudaKernelNodeParams p[4];  // 填 func/args/grid/block/shared
  //   cudaGraphAddKernelNode(&n[0], graph, nullptr, 0, &p[0]);
  //   cudaGraphAddKernelNode(&n[1], graph, &n[0], 1, &p[1]);   // n[1] 依赖 n[0]
  //   cudaGraphAddKernelNode(&n[2], graph, &n[1], 1, &p[2]);
  //   cudaGraphAddKernelNode(&n[3], graph, &n[2], 1, &p[3]);
  //   cudaGraphInstantiate(&graphExec, graph, NULL);
  //   cudaGraphLaunch(graphExec, stream);
  //
  // —— 片段 3：图更新（关键工程点）——静态图不能改，但可**廉价更新节点参数**（如改指针）：
  //   cudaGraphExecKernelNodeSetParams(graphExec, n[0], &p0_new);  // 改输入指针/常量，免重 capture
  //
  // —— 片段 4：条件/循环节点（CUDA 12+）——把控制流也固化进图（如 while 循环、if 分支）：
  //   cudaGraphConditionalHandle h; cudaGraphConditionalHandleCreate(&h, graph, 0, 0);
  //   cudaGraphAddWhileNode(&wn, graph, &n[3], 1, &whileParams, h);
#endif // __CUDACC__

  printf("\n(纯 C++ 校验完成；CUDA Graphs API 片段由 __CUDACC__ 守护，需真机 nvcc 编译)\n");
  printf("=== %d passed, %d failed ===\n", passed, failed);
  return failed ? 1 : 0;
}
