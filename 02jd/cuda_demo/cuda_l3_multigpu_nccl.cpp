// =============================================================================
//  L3.4  多 GPU / NVLink / NCCL 通信  ——  架构级收官站
// -----------------------------------------------------------------------------
//  纯 C++ 可跑（无 GPU / 无 NCCL 也能跑全部模型与对拍）：
//    g++ -O3 -std=c++17 cuda_l3_multigpu_nccl.cpp -o mg && ./mg
//  末尾 __CUDACC__ 守护段给真机骨架：每卡本地计算 kernel（DP 步的前向/反向）
//  以及 HAVE_NCCL 守护的真机 NCCL 集合通信调用（需 nccl.h + 多卡环境）。
//
//  主线：L2.4 shared 归约（一个 tile 内）→ L3.4 跨 GPU 归约（一个 GPU 即一个 tile）；
//        L3.1 pipeline/fusion（用计算藏访存）→ L3.4 用计算藏通信（comm-compute overlap）；
//        L3.2 Tensor Core + 混合精度 → L3.4 张量并行把一层 GEMM 切到多卡 + NCCL 求和( fp32 累加 )。
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>
#include <sstream>
#include <algorithm>

// ----------------------------------------------------------------------------
// 一、ring AllReduce 的纯 CPU 模型 + 正确性对拍
//    N 张卡，每张卡持有长度 K 的向量 v[rank]；ring all-reduce 让每张卡最终都拿到
//    sum_{r} v[r]。用分块环算法（scatter-reduce + all-gather）在 host 端模拟，
//    验证结果与朴素求和一致。这正是 L2.4 跨线程归约的「跨卡」放大版。
// ----------------------------------------------------------------------------
static std::vector<float> ring_allreduce_host(const std::vector<std::vector<float>>& in) {
    const int N = (int)in.size();
    const size_t K = in.empty() ? 0 : in[0].size();
    const size_t cs = K / (size_t)N;        // 每卡把向量切成 N 块，每块 cs 个元素
    // buf[r] = 该卡持有的 N 块（每块 cs 个 float）
    std::vector<std::vector<std::vector<float>>> buf(N);
    for (int r = 0; r < N; ++r) {
        buf[r].resize(N);
        for (int c = 0; c < N; ++c)
            buf[r][c].assign(in[r].begin() + c * cs, in[r].begin() + (c + 1) * cs);
    }
    // scatter-reduce：第 step 步，rank r 把第 (r-step+N)%N 块 累加 给 (r+1)%N
    for (int step = 0; step < N - 1; ++step) {
        auto next = buf;
        for (int r = 0; r < N; ++r) {
            int sc = (r - step + N) % N;
            int sendTo = (r + 1) % N;
            for (size_t e = 0; e < cs; ++e) next[sendTo][sc][e] += buf[r][sc][e];
        }
        buf = next;
    }
    // all-gather：把已规约好的块沿环扩散（覆盖式），使每张卡都拿到全量
    for (int step = 0; step < N - 1; ++step) {
        auto next = buf;
        for (int r = 0; r < N; ++r) {
            int sc = (r + 1 - step + N) % N;   // 本步广播的块（= 本卡已持有完整的一块）
            int sendTo = (r + 1) % N;
            next[sendTo][sc] = buf[r][sc];
        }
        buf = next;
    }
    // 展平：所有卡结果相同，取 rank0
    std::vector<float> out;
    for (int c = 0; c < N; ++c)
        for (size_t e = 0; e < cs; ++e) out.push_back(buf[0][c][e]);
    return out;
}

static void verify_ring_allreduce() {
    printf("\n[verify_ring_allreduce] N=4 张卡, 向量长度 K=256\n");
    const int N = 4; const size_t K = 256;
    std::vector<std::vector<float>> in(N);
    std::vector<float> ref(K, 0.0f);
    srand(20260815);
    for (int r = 0; r < N; ++r) {
        in[r].resize(K);
        for (size_t i = 0; i < K; ++i) {
            in[r][i] = (float)(rand() % 1000) / 7.0f;
            ref[i] += in[r][i];            // 参考：朴素求和
        }
    }
    std::vector<float> out = ring_allreduce_host(in);
    float maxdiff = 0.0f;
    for (size_t i = 0; i < K; ++i) maxdiff = std::max(maxdiff, std::fabs(out[i] - ref[i]));
    printf("  维数 K=%zu, max|ring - naive| = %.6g  ->  %s\n", K, maxdiff,
           maxdiff < 1e-3f ? "PASS（跨卡归约正确）" : "FAIL");
}

// ----------------------------------------------------------------------------
// 二、带宽模型：ring AllReduce 的“线上字节数” → busbw / algobw
//    bus_bytes / GPU = 2*(N-1)/N * K     （N 张卡 K 字节，环算法）
//    algobw = K / time                    （把 K 字节“广播到所有卡”的有效速率）
//    busbw  = bus_bytes / time = 2*(N-1)/N * algobw  （链路真实占用，看是否压满 NVLink）
//    用 NVLink(H100 单向 450 GB/s) / PCIe Gen5(64) / PCIe Gen4(32) 对比。
// ----------------------------------------------------------------------------
static void bandwidth_model(int N, double K_GB, double link_GBs) {
    const double bus_bytes = 2.0 * (N - 1) / (double)N * K_GB;   // 每卡线上字节(GB)
    const double time_s = bus_bytes / link_GBs;                  // 秒
    const double algobw = K_GB / time_s;                         // GB/s
    const double busbw  = bus_bytes / time_s;                    // GB/s
    printf("  N=%d K=%.0f GB  link=%.0f GB/s(单向):  bus_bytes/GPU=%.3f GB  "
           "t=%.2f ms  algobw=%.1f GB/s  busbw=%.1f GB/s\n",
           N, K_GB, link_GBs, bus_bytes, time_s * 1e3, algobw, busbw);
}

static void report_bandwidth() {
    printf("\n[report_bandwidth] ring AllReduce 带宽模型（N=8, K=1 GB）\n");
    const int N = 8; const double K = 1.0;
    printf("  -- NVLink 4.0 (H100) 单向 450 GB/s --\n");
    bandwidth_model(N, K, 450.0);
    printf("  -- PCIe Gen5 x16 单向 64 GB/s --\n");
    bandwidth_model(N, K, 64.0);
    printf("  -- PCIe Gen4 x16 单向 32 GB/s --\n");
    bandwidth_model(N, K, 32.0);
    printf("  => NVLink 用满时 busbw≈450 GB/s，是 PCIe Gen4 的 ~14×；\n"
           "     大 N 时 algobw→link/2（环算法每个块要走 N-1 跳）。\n");
}

// ----------------------------------------------------------------------------
// 三、数据并行(DP) step 的 comm-compute overlap 仿真
//    = L3.1 pipeline 思想的跨卡版：用“计算”去藏“通信”。
//    一步 = 本地前向/反向(compute_ms) + 梯度 ring AllReduce(comm_ms)。
//    不重叠：step = compute + comm；重叠：step = max(compute, comm)。
// ----------------------------------------------------------------------------
static void data_parallel_overlap(double compute_ms, double K_GB, double link_GBs) {
    const int N = 8;
    const double bus_GB = 2.0 * (N - 1) / (double)N * K_GB;
    const double comm_ms = bus_GB / link_GBs * 1e3;
    const double non_overlap = compute_ms + comm_ms;
    const double overlap = std::max(compute_ms, comm_ms);
    printf("  compute=%.0f ms grad=%.1f GB link=%.0f GB/s: comm=%.1f ms  "
           "no-overlap=%.1f ms  overlap=%.1f ms  通信被藏比例=%.1f%%\n",
           compute_ms, K_GB, link_GBs, comm_ms, non_overlap, overlap,
           100.0 * (1.0 - overlap / non_overlap));
}

static void report_overlap() {
    printf("\n[report_overlap] 数据并行一步：计算 vs 梯度 AllReduce（模型梯度 14 GB）\n");
    const double compute_ms = 100.0, K_GB = 14.0;
    printf("  -- NVLink 450 GB/s --\n"); data_parallel_overlap(compute_ms, K_GB, 450.0);
    printf("  -- PCIe Gen4 32 GB/s --\n"); data_parallel_overlap(compute_ms, K_GB, 32.0);
    printf("  => 梯度越大/模型越偏通信，NVLink 与 PCIe 的差距越致命；\n"
           "     张量并行(tensor parallel)通信最频繁，必须待在 NVLink 域内。\n");
}

// ----------------------------------------------------------------------------
// 四、拓扑 → 传输选型（解析 `nvidia-smi topo -m` 的简化输出）
//    NCCL 自动在 NVLink > PCIe P2P(GPUDirect) > RDMA(InfiniBand) > socket 里挑最快。
// ----------------------------------------------------------------------------
static std::string pick_transport(const std::string& topo) {
    if (topo.find("NVL") != std::string::npos) return "NVLink（直连/经 NVSwitch，全速）";
    if (topo.find("PIX") != std::string::npos) return "PCIe 同桥 P2P（GPUDirect，接近 PCIe 上限）";
    if (topo.find("PXB") != std::string::npos) return "PCIe 跨桥（经 switch，带宽打折）";
    if (topo.find("NODE") != std::string::npos) return "跨 NUMA 节点（经 CPU interconnect，最慢）";
    return "socket/IB-RDMA（跨机）";
}

static void analyze_topology() {
    printf("\n[analyze_topology] 解析 nvidia-smi topo -m 片段\n");
    const char* lines[] = {
        "GPU0  GPU1  GPU2  GPU3",
        "GPU0   X    NVL   NVL   NVL",
        "GPU1  NVL    X    NVL   NVL",
        "GPU3  NVL   NVL   NVL    X",
    };
    for (int i = 1; i < 4; ++i) {
        std::string line(lines[i]);
        size_t p = line.find("GPU0");
        // 取第 i 张卡对 GPU0 的关系列
        std::istringstream iss(line);
        std::string tok; std::vector<std::string> cols;
        while (iss >> tok) cols.push_back(tok);
        // cols: [GPUi, relTo0, relTo1, relTo2, relTo3]
        if (cols.size() >= 2)
            printf("  %s <-> GPU0 : %s  => NCCL 选 %s\n",
                   cols[0].c_str(), cols[1].c_str(), pick_transport(cols[1]).c_str());
    }
}

// ----------------------------------------------------------------------------
// 五、NCCL 集合通信语义速查（打印，非执行）
// ----------------------------------------------------------------------------
static void nccl_api_cheatsheet() {
    printf("\n[nccl_api_cheatsheet] 集合通信原语（API 仿 MPI）\n");
    printf("  ncclAllReduce : 各卡 sendbuff 求和 -> 每卡 recvbuff 都拿到相同结果（DP 梯度同步）\n");
    printf("  ncclBroadcast : root 的数据 -> 所有卡\n");
    printf("  ncclReduce    : 各卡求和 -> 仅 root 拿到（如 loss 聚合）\n");
    printf("  ncclAllGather : 各卡贡献一段 -> 每卡拿到拼接全量（张量并行收集分片）\n");
    printf("  ncclReduceScatter: 求和后把结果按卡切分 -> 每卡只拿自己那 1/N（Megatron 张量并行）\n");
    printf("  ncclSend/ncclRecv : 点对点（P2P），常配合 ncclGroupStart/End 双向交换\n");
    printf("  op: ncclSum/ncclProd/ncclMax/ncclMin/ncclAvg；type: ncclFloat16/32, ncclBfloat16, ncclInt8...\n");
    printf("  redOp 在 fp16/bf16 输入上仍用 fp32 累加（呼应 L3.2 混合精度：低精度输入 × 高精度累加）\n");
}

// =============================================================================
//  真机骨架（不会在纯 g++ 路径下编译，需对应硬件/库）
// =============================================================================
#ifdef __CUDACC__
// 每卡本地的“前向+反向”计算（DP 步里真正吃算力的部分）。
// 这里用一个占位 GEMM+激活代表，真实场景是模型的一层。
__global__ void local_compute_kernel(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      float* __restrict__ grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 0.0f;
        // 占位：把 w 当一维权重做一次乘加（真实是矩阵乘 + 反向）
        s += x[i] * w[i];
        grad[i] = s;   // 反向得到的本卡梯度，待 AllReduce
    }
}

#ifdef HAVE_NCCL
// 真机 NCCL 调用示意（需 #include <nccl.h> 与多卡环境）。
// 多进程标准写法（单进程可用 ncclCommInitAll）：
//   ncclGetUniqueId(&id);  // rank0，再 MPI_Bcast 给其余 rank
//   cudaSetDevice(rank);
//   ncclCommInitRank(&comm, worldSize, id, rank);
//   ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);
//   cudaStreamSynchronize(stream);
//   ncclCommDestroy(comm);
// 把多个集合调用用 ncclGroupStart()/ncclGroupEnd() 包起来可一次性排程，
// 让 NCCL 在底层把环/树通道合并，进一步压低延迟。
__attribute__((unused)) static void device_nccl_allreduce_demo(
        ncclComm_t comm, cudaStream_t stream,
        float* send, float* recv, size_t count) {
    ncclGroupStart();
    ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);
    ncclGroupEnd();
}
#endif // HAVE_NCCL
#endif // __CUDACC__

// =============================================================================
int main() {
    printf("================ L3.4 多 GPU / NVLink / NCCL 通信 ================\n");
    verify_ring_allreduce();     // 一、跨卡归约正确性（= L2.4 放大版）
    report_bandwidth();          // 二、ring AllReduce 带宽模型
    report_overlap();            // 三、comm-compute overlap（= L3.1 跨卡版）
    analyze_topology();          // 四、拓扑 → 传输选型
    nccl_api_cheatsheet();       // 五、NCCL 集合通信语义
    printf("\n[说明] 纯 C++ 路径已跑完全部模型与对拍；真机 __CUDACC__/HAVE_NCCL 段需 GPU+NCCL。\n");
    return 0;
}
