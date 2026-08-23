// ============================================================================
//  L3 补充：GPU 直接通信 P2P + cudaMemcpyPeer
//  —— 多 GPU / NVLink / NCCL 收官站的最后一块拼图：
//     把「NCCL 自动选 transport」这件事的底层物理基础（peer access）讲透
//
//  纯 C++ 可跑（无 GPU 也行）：./mg 之外另起一个可执行
//      g++ -O3 -std=c++17 cuda_l3_p2p_peer.cpp -o p2p && ./p2p
//  真机多卡段受 __CUDACC__ 守护（需 nvcc + 多卡硬件），用 HAVE_NCCL 也可演示 NCCL 调用
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>

// ---------- 小工具 ----------
static double GBps_to_ms(double bytes, double GBps) {
    return (bytes / 1e9) / GBps * 1e3;   // 秒 -> 毫秒
}
static std::string fmt_ms(double ms) {
    char buf[32];
    if (ms >= 1.0) snprintf(buf, sizeof(buf), "%.2f ms", ms);
    else           snprintf(buf, sizeof(buf), "%.1f us", ms * 1000.0);
    return std::string(buf);
}

// ============================================================================
//  §2 + §4：用一张「拓扑矩阵」模拟 peer access，演示谁能直连、谁要绕 host
//  连接类型按 nvidia-smi topo -m 的图例：
//     NV     = NVLink（最佳，600~900 GB/s 双向）
//     PIX    = 同一 PCIe switch（全速 PCIe P2P）
//     PHB    = 同一 PCIe host bridge（走 CPU/系统内存，P2P 受限）
//     NODE   = 同一 NUMA 节点（NVLink 或 PCIe 直连，可 P2P）
//     SYS    = 跨 NUMA（默认禁用 P2P，只能 staged 绕 host）
// ============================================================================
enum Link { NONE, SYS, PHB, NODE, PIX, NV };

static double link_bw_GBps(Link l) {            // 单向聚合带宽（教学近似）
    switch (l) {
        case NV:   return 450.0;   // H100 单向 ~450 GB/s（双向 900）
        case PIX:  return 32.0;    // PCIe Gen4 x16 单向
        case NODE: return 32.0;    // 同 NUMA，按 PCIe 全速保守估
        case PHB:  return 16.0;    // 经 host bridge，减半
        case SYS:  return 12.0;    // 跨 NUMA，被系统互连（UPI/xGMI）卡
        default:   return 0.0;
    }
}
// P2P 直连可用？—— NV/PIX/NODE/PHB 都支持 P2P；SYS 不行（staged）
static bool can_p2p(Link l) { return l == NV || l == PIX || l == NODE || l == PHB; }

static void model_topology() {
    printf("\n=== 拓扑矩阵 + P2P 直连判定（4 卡，模拟 2 卡 NVLink + 2 卡跨 NUMA）===\n");
    const int N = 4;
    // 0<->1 NVLink；2<->3 NVLink；0/1 与 2/3 跨 NUMA(SYS)；其余 NODE
    Link topo[4][4] = {
        /*0*/ {NONE, NV,   SYS,  NODE},
        /*1*/ {NV,   NONE, NODE, SYS },
        /*2*/ {SYS,  NODE, NONE, NV  },
        /*3*/ {NODE, SYS,  NV,   NONE},
    };
    const char* tag[] = {"--", "SYS", "PHB", "NODE", "PIX", "NV"};
    printf("      ");
    for (int j = 0; j < N; ++j) printf("GPU%-2d ", j);
    printf("\n");
    for (int i = 0; i < N; ++i) {
        printf("GPU%-2d: ", i);
        for (int j = 0; j < N; ++j) {
            if (i == j) { printf(" X   "); continue; }
            printf("%-4s ", tag[topo[i][j]]);
        }
        printf("\n");
    }
    printf("图例 NONE=自身  NV=NVLink(最佳)  PIX=同PCIe switch  NODE=同NUMA  "
           "PHB=同host bridge  SYS=跨NUMA(无P2P)\n");
    int p2p_pairs = 0, staged = 0;
    for (int i = 0; i < N; ++i) for (int j = i + 1; j < N; ++j) {
        if (can_p2p(topo[i][j])) { ++p2p_pairs;
            printf("  GPU%d->GPU%d: P2P 直连 (%-4s, %.0f GB/s 单向)\n",
                   i, j, tag[topo[i][j]], link_bw_GBps(topo[i][j]));
        } else { ++staged;
            printf("  GPU%d->GPU%d: 不能 P2P -> 必须 staged 绕 host (SYS)\n", i, j);
        }
    }
    printf("  -> 可直连 %d 对，需绕 host %d 对（这正是 NCCL 要避开 SYS 的原因）\n", p2p_pairs, staged);
}

// ============================================================================
//  §1：直连 vs 绕 host 的带宽/时延收益模型
//    staged: GPU0 --PCIe--> host(pinned) --memcpy--> GPU1 --PCIe--> ... 错，是：
//            GPU0 -> host(读, B/bw_pci) -> host memcpy(B/bw_host) -> GPU1(写, B/bw_pci)
//            两步 PCIe + 一次 host 内存拷贝
//    direct P2P: GPU0 --link--> GPU1（一次直达，bw=link）
// ============================================================================
static void p2p_vs_staged(double B_GB) {
    printf("\n=== P2P 直连 vs 绕 host 中转（搬运 %.2f GB）===\n", B_GB);
    const double bw_pci  = 32.0;   // PCIe Gen4 x16 单向 GB/s
    const double bw_host = 25.0;   // 主机 DDR 拷贝 GB/s
    const double bw_nv   = 450.0;  // NVLink 单向 GB/s

    double t_staged = 2.0 * B_GB / bw_pci + B_GB / bw_host;        // 秒
    double t_pcie   = B_GB / bw_pci;                               // PCIe P2P 直连
    double t_nv     = B_GB / bw_nv;                                // NVLink 直连

    printf("  绕 host 中转 : %s  (2xPCIe + 1x host memcpy)\n",
           fmt_ms(t_staged * 1e3).c_str());
    printf("  PCIe P2P 直连: %s  -> 提速 %.1fx\n",
           fmt_ms(t_pcie * 1e3).c_str(), t_staged / t_pcie);
    printf("  NVLink 直连  : %s  -> 提速 %.1fx\n",
           fmt_ms(t_nv * 1e3).c_str(), t_staged / t_nv);
    printf("  延迟量级(实测,非模型): staged ~10-20us | PCIe P2P ~3-5us | NVLink ~1-2us\n");
}

// ============================================================================
//  §3：cudaMemcpyPeer vs cudaMemcpy(Default) 的语义差异（用枚举说清）
//    - cudaMemcpyPeer(dst, dstDev, src, srcDev, count): 显式给设备号，无 UVA 也能用
//    - cudaMemcpy(dst, src, count, cudaMemcpyDefault)  : 有 UVA 时按指针推断设备，一句话直连
// ============================================================================
static void api_semantics() {
    printf("\n=== cudaMemcpyPeer vs cudaMemcpy(Default) ===\n");
    printf("  cudaMemcpyPeer(dst,1, src,0, n): 显式传(dstDev=1, srcDev=0)，\n"
           "      即使没有 UVA（统一虚拟地址）也能跑——老式/受限环境兜底。\n");
    printf("  cudaMemcpy(dst,src,n, cudaMemcpyDefault): 有 UVA 时 CUDA 从指针值\n"
           "      反推‘谁家的内存’，无需写设备号；直连能力与 MemcpyPeer 一致。\n");
    printf("  共同点：两卡都能直连映射时走直接 P2P（不落 host）；\n"
           "          任意两卡（哪怕不支持直连）CUDA 也能拷贝，只是会 staged 绕 host。\n");
    printf("  坑①：P2P memcpy 不与其他 kernel/拷贝并发——必须先把两卡在飞任务排干，\n"
           "       拷贝完才能起后续任务（与普通的 cudaMemcpyAsync 不同）。\n");
    printf("  坑②：peer 映射非对称——A 能 map B，不代表 B 能 map A，要双向 EnablePeerAccess。\n");
}

// ============================================================================
//  §3 / verify：模拟一次 P2P 拷贝，校验结果正确（host 模拟两张卡的显存）
// ============================================================================
static bool verify_p2p_copy() {
    printf("\n=== verify_p2p_copy（模拟 GPU0 显存 -> GPU1 显存，直连）===\n");
    const int N = 1024;
    std::vector<float> g0(N), g1(N);
    std::mt19937 rng(20260815);
    std::uniform_real_distribution<float> U(0.0f, 10.0f);
    for (int i = 0; i < N; ++i) g0[i] = U(rng);

    // 模拟「直连拷贝」：g0 内容进入 g1（真实场景是 GPU0 显存经 NVLink/PCIe 直写 GPU1 显存）
    std::memcpy(g1.data(), g0.data(), N * sizeof(float));

    bool ok = true;
    for (int i = 0; i < N; ++i) if (g0[i] != g1[i]) { ok = false; break; }
    printf("  GPU0[%d floats] --P2P--> GPU1 : %s (数据逐字节一致)\n", N, ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================================
//  §6：NCCL 如何骑在 P2P 上——transport 自动优先级（与 L3.4 闭环）
// ============================================================================
static void nccl_transport_select() {
    printf("\n=== NCCL transport 自动选择（骑在 peer access 之上）===\n");
    printf("  同节点内: NVLink > PCIe P2P(PIX/NODE/PHB) > SHM(共享内存兜底)\n");
    printf("  跨节点  : RDMA(IB/GDR) > Socket(TCP 最差)\n");
    printf("  可通过环境变量强制/排查:\n");
    printf("    NCCL_P2P_LEVEL=NVL        仅用 NVLink 做 P2P\n");
    printf("    NCCL_NET_GDR_LEVEL=5      跨节点开 GPUDirect RDMA\n");
    printf("    NCCL_DEBUG=INFO           看通道里 P2P/CUMEM|P2P/IPC|SHM|NET/Socket\n");
    printf("  NCCL 的 P2P transport = peer mappings(cudaDeviceEnablePeerAccess)\n"
           "   + IPC(cudaIpcGetMemHandle/OpenMemHandle, 跨进程共享 GPU 指针)。\n");
    printf("  => L3.4 说的‘自动选路 NVLink>PCIe P2P>RDMA>socket’正是这套机制的结果。\n");
}

// ============================================================================
//  真机段（守护）：真实 CUDA P2P API 骨架
// ============================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

static void check(cudaError_t e, const char* s) {
    if (e != cudaSuccess) { printf("CUDA ERR %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

// 在两卡间启用双向 peer access（NCCL/框架底层都这么做）
static void enable_peer_access(int a, int b) {
    int can = 0;
    check(cudaDeviceCanAccessPeer(&can, a, b), "CanAccessPeer");
    if (!can) { printf("GPU%d->GPU%d 不支持 P2P（拓扑为 SYS？）\n", a, b); return; }
    check(cudaSetDevice(a), "setDevice");
    check(cudaDeviceEnablePeerAccess(b, 0), "EnablePeerAccess");  // flags 必须为 0
    check(cudaSetDevice(b), "setDevice");
    check(cudaDeviceEnablePeerAccess(a, 0), "EnablePeerAccess");  // 双向要各 enable 一次
}

// 真实 P2P 拷贝：GPU0 显存 -> GPU1 显存，异步、排进 stream
static void real_p2p_copy(float* d0, float* d1, size_t bytes, cudaStream_t s) {
    check(cudaMemcpyPeerAsync((void*)d1, 1, (const void*)d0, 0, bytes, s), "MemcpyPeerAsync");
}
#endif

int main() {
    printf("########## L3 补充：GPU 直接通信 P2P + cudaMemcpyPeer ##########\n");
    model_topology();
    p2p_vs_staged(1.0);           // 1 GB
    api_semantics();
    bool ok = verify_p2p_copy();
    nccl_transport_select();
    printf("\n[summary] P2P 直连=%s；L3 多卡主线（NCCL/P2P/NVLink）已闭环。\n",
           ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
