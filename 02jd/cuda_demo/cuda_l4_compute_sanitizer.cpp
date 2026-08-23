// ============================================================================
//  cuda_l4_compute_sanitizer.cpp
//  L4（工业扩展）· compute-sanitizer 运行时诊断：捕获 OOB / 未初始化 / 数据竞争
//
//  纯 C++ 可跑：  g++ -O3 -std=c++17 cuda_l4_compute_sanitizer.cpp -o san && ./san
//  （compute-sanitizer 真机调用 + CUDA kernel 片段由 #ifdef __CUDACC__ 守护，g++ 跳过）
//
//  真实 kernel 的 bug 大多不在编译期暴露，而在运行时：越界访问、读未初始化
//  共享内存、跨线程数据竞争。compute-sanitizer 靠"插桩内存访问"抓这三类。
//  本文件在 CPU 上复刻一台极简"device 模拟器 + sanitizer"：
//    - 每个访问原语(gstore/gload/sstore/sload/gatomic)都经插桩，记录越界、未初始化读、竞争
//    - 三个 buggy 程序被精准抓出；fixed 程序干净通过 + 数值 == 参考
// ============================================================================
#include <cstdio>
#include <vector>
#include <map>

// -------------------- 极简 device 模拟器 + sanitizer --------------------
// 模型：N 长度全局内存 gmem + 一块共享内存(每 block 一份，所有线程可见)
// 插桩原语忠实记录三类运行时错误，与 compute-sanitizer 的诊断一一对应。
struct Sanitizer {
    int N;                          // 全局内存长度
    std::vector<int> gmem;          // 全局内存（值）
    std::vector<bool> gwritten;     // 全局元素是否曾被写（未写为"未初始化"）
    std::map<int,int> gwriter;      // idx -> 上次写的 tid（自上次 barrier 起）
    std::vector<bool> swritten;     // 共享内存元素是否曾被写
    bool oob=false, uninit=false, race=false;

    explicit Sanitizer(int n): N(n), gmem(n,0), gwritten(n,false), swritten(8,false) {}

    void gstore(int idx, int tid, int val){
        if(idx<0 || idx>=N){ oob=true; return; }            // 越界写
        if(gwriter.count(idx) && gwriter[idx]!=tid) race=true; // 不同线程写同址且未同步 → 竞争
        gwriter[idx]=tid; gwritten[idx]=true; gmem[idx]=val;
    }
    void gload(int idx, int tid){
        if(idx<0 || idx>=N){ oob=true; return; }
        if(!gwritten[idx]) uninit=true;                      // 读从未写过的全局 → 未初始化
    }
    void sstore(int sidx, int /*tid*/, int /*val*/){ swritten[sidx]=true; }   // 共享写（全线程可见）
    void sload(int sidx, int /*tid*/){ if(!swritten[sidx]) uninit=true; }     // 读未写过的共享 → 未初始化
    void gatomic(int idx, int tid, int val){
        if(idx<0 || idx>=N){ oob=true; return; }
        gwriter[idx]=tid; gwritten[idx]=true; gmem[idx]+=val;  // atomic 是有意并发写，不报 race
    }
    void barrier(){ gwriter.clear(); }                       // 跨线程同步点：清空"上次写者"记录
    bool clean() const { return !(oob||uninit||race); }
};

// -------------------- 三个 buggy 程序 + 一个 fixed 程序 --------------------
// 约定：NT 个线程 tid∈[0,NT)，顺序 tid=0..NT-1 执行各自语句（SIMT 简化模型）

// Buggy 1：越界写（常见：索引算错 / 缺 i<N 保护）
static Sanitizer buggy_oob(){
    const int NT=8, N=8; Sanitizer s(N);
    for(int tid=0; tid<NT; ++tid)
        if(tid==0) s.gstore(N+5, tid, 1);   // 写超出 gmem 末尾 → OOB
    return s;
}
// Buggy 2：读未初始化共享内存（常见：__shared__ 未先写就跨线程读）
static Sanitizer buggy_uninit(){
    const int NT=8; Sanitizer s(8);
    for(int tid=0; tid<NT; ++tid){
        if(tid==0) s.sload(0, tid);         // tid0 在任何人写前就读共享槽 0 → uninit
        if(tid==1) s.sstore(0, tid, 1);
    }
    return s;
}
// Buggy 3：数据竞争（常见：直方图/归约无 atomic，多 tid 写同一 bin）
static Sanitizer buggy_race(){
    const int NT=8, N=8; Sanitizer s(N);
    for(int tid=0; tid<NT; ++tid){
        int bin = tid % 4;                   // 多个 tid 映射到同一 bin
        s.gstore(bin, tid, 1);               // 不同 tid 写同址且未同步 → race
    }
    return s;
}
// Fixed：直方图用 atomic 累加（compute-sanitizer 视 atomic 为有意的并发写）
static Sanitizer fixed_hist(){
    const int NT=8, N=8; Sanitizer s(N);
    for(int tid=0; tid<NT; ++tid){
        int bin = tid % 4;
        s.gatomic(bin, tid, 1);              // atomicAdd：不报 race
    }
    return s;
}

// -------------------- 主程序 + 验证 --------------------
int main(){
    int passed=0, failed=0;
    auto chk=[&](const char* name, bool ok){ printf("  %s %s\n", ok?"PASS":"FAIL", name); ok?++passed:++failed; };

    Sanitizer soob = buggy_oob();
    chk("sanitizer 捕获越界写(OOB): buggy 程序被标记", soob.oob && !soob.clean());

    Sanitizer su = buggy_uninit();
    chk("sanitizer 捕获未初始化读(uninit): buggy 程序被标记", su.uninit && !su.clean());

    Sanitizer sr = buggy_race();
    chk("sanitizer 捕获数据竞争(race): buggy 程序被标记", sr.race && !sr.clean());

    Sanitizer sf = fixed_hist();
    chk("fixed(atomic) 程序: sanitizer 干净通过", sf.clean());

    // fixed 程序数值 == 参考直方图（每个 bin = tid%4==bin 的线程数）
    int ref[4]={0}; for(int tid=0;tid<8;++tid) ref[tid%4]++;
    bool match=true; for(int b=0;b<4;++b) if(sf.gmem[b]!=ref[b]) match=false;
    chk("fixed(atomic) 直方图 == 参考计数(ref[tid%4])", match && sf.gmem[0]==ref[0] && sf.gmem[3]==ref[3]);

    printf("=== %d passed, %d failed ===\n", passed, failed);

#ifdef __CUDACC__
    // ---- 真机片段（仅 nvcc 下编译，作教材）----
    // 编译期查不出、运行时才暴露的 bug 用 compute-sanitizer 抓：
    //   compute-sanitizer --tool memcheck   ./your_app   # 抓 OOB / 未初始化
    //   compute-sanitizer --tool racecheck  ./your_app   # 抓数据竞争
    //   compute-sanitizer --tool initcheck  ./your_app   # 专抓未初始化
    // 典型 buggy kernel（缺 i<N）：
    //   __global__ void buggy_oob_kernel(int* d, int n){ int i=threadIdx.x; d[i+n]=1; }  // 越界
    // 修复后：
    //   __global__ void fixed_hist_kernel(int* bins, int n){
    //       int tid=blockIdx.x*blockDim.x+threadIdx.x;
    //       if(tid<n) atomicAdd(&bins[tid%4], 1);   // atomic 消除竞争
    //   }
#endif
    return failed?1:0;
}
