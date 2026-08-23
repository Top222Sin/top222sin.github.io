// jac_l4_autodiff.cpp
// ============================================================================
// L4 自动微分（Automatic Differentiation）：前向模式 + 反向模式
// ----------------------------------------------------------------------------
// 雅可比矩阵 J=∂f/∂x 用"手算/有限差分"在深层组合函数下既不精确也不可扩展。
// 工业界（PyTorch/JAX/Ceres）用自动微分：把求导规则嵌进"计算"本身。
//   • 前向模式 (forward / tangent / JVP)：一次前向传播携带切向量，得到 J·t
//   • 反向模式 (reverse / adjoint / VJP)：记录计算图，一次反向传播得到 Jᵀ·u
//   复杂度：前向 ∝ 输入维数 n，反向 ∝ 输出维数 m。
//           训练是"标量损失(m=1) 对百万参数(n 很大)" → 反向模式碾压 → 即 backprop。
//
// 本 demo 用同一函数 f: R^2 → R^2
//   f1 = sin(x)·exp(y) + x·y
//   f2 = x^2 - cos(y)
// 验证：
//   (1) 前向模式 → 全雅可比 J（切向量=单位矩阵，1 次前向）
//   (2) 反向模式 → 全雅可比 J（每个输出种子 1 次，共 2 次反向）
//   (3) JVP：前向模式携带方向 t → J·t
//   (4) VJP：反向模式携带输出种子 u → Jᵀ·u
//   (5) 标量损失 L=f1²+f2² 梯度：反向模式 1 次 → ∇L
//   以上全部与有限差分 (FD) 一致（误差 < 1e-6）。
//
// 编译: g++ -O3 -std=c++17 jac_l4_autodiff.cpp -o t && ./t
// ============================================================================
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

static const double X = 0.7, Y = -0.3;
static const double H = 1e-6;

// ---------------- 有限差分（ground truth） ----------------
static void f_eval(double x, double y, double& f1, double& f2){
    f1 = sin(x)*exp(y) + x*y;
    f2 = x*x - cos(y);
}
static double Lfun(double x, double y){
    double a,b; f_eval(x,y,a,b); return a*a + b*b;
}
static void jacobian_fd(double J[2][2]){
    double f1p,f2p,f1m,f2m;
    f_eval(X+H,Y,f1p,f2p); f_eval(X-H,Y,f1m,f2m);   // ∂/∂x
    J[0][0]=(f1p-f1m)/(2*H); J[1][0]=(f2p-f2m)/(2*H);
    f_eval(X,Y+H,f1p,f2p); f_eval(X,Y-H,f1m,f2m);   // ∂/∂y
    J[0][1]=(f1p-f1m)/(2*H); J[1][1]=(f2p-f2m)/(2*H);
}

// ---------------- 前向模式（dual numbers，切向量） ----------------
struct Fwd { double v; vector<double> d; };
static Fwd f_add(const Fwd& a, const Fwd& b){ Fwd r; r.v=a.v+b.v; r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=a.d[i]+b.d[i]; return r; }
static Fwd f_sub(const Fwd& a, const Fwd& b){ Fwd r; r.v=a.v-b.v; r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=a.d[i]-b.d[i]; return r; }
static Fwd f_mul(const Fwd& a, const Fwd& b){ Fwd r; r.v=a.v*b.v; r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=a.d[i]*b.v + a.v*b.d[i]; return r; }
static Fwd f_sin(const Fwd& a){ Fwd r; r.v=sin(a.v); r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=cos(a.v)*a.d[i]; return r; }
static Fwd f_cos(const Fwd& a){ Fwd r; r.v=cos(a.v); r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=-sin(a.v)*a.d[i]; return r; }
static Fwd f_exp(const Fwd& a){ Fwd r; r.v=exp(a.v); r.d.resize(a.d.size()); for(size_t i=0;i<a.d.size();++i) r.d[i]=exp(a.v)*a.d[i]; return r; }
static void eval_fwd(const Fwd& x, const Fwd& y, Fwd& f1, Fwd& f2){
    f1 = f_add(f_mul(f_sin(x), f_exp(y)), f_mul(x, y));
    f2 = f_sub(f_mul(x, x), f_cos(y));
}

// ---------------- 反向模式（tape / 计算图） ----------------
enum { OP_VAR=0, OP_ADD, OP_SUB, OP_MUL, OP_SIN, OP_COS, OP_EXP };
struct RevNode { double val; double adj; int op; int a; int b; };
static vector<RevNode> tape;
struct Rev { int id; };
static Rev make_var(double v){
    RevNode n; n.val=v; n.adj=0; n.op=OP_VAR; n.a=-1; n.b=-1;
    int id=(int)tape.size(); tape.push_back(n); return Rev{id};
}
static Rev r_add(const Rev& a, const Rev& b){ RevNode n; n.val=tape[a.id].val+tape[b.id].val; n.adj=0; n.op=OP_ADD; n.a=a.id; n.b=b.id; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }
static Rev r_sub(const Rev& a, const Rev& b){ RevNode n; n.val=tape[a.id].val-tape[b.id].val; n.adj=0; n.op=OP_SUB; n.a=a.id; n.b=b.id; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }
static Rev r_mul(const Rev& a, const Rev& b){ RevNode n; n.val=tape[a.id].val*tape[b.id].val; n.adj=0; n.op=OP_MUL; n.a=a.id; n.b=b.id; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }
static Rev r_sin(const Rev& a){ RevNode n; n.val=sin(tape[a.id].val); n.adj=0; n.op=OP_SIN; n.a=a.id; n.b=-1; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }
static Rev r_cos(const Rev& a){ RevNode n; n.val=cos(tape[a.id].val); n.adj=0; n.op=OP_COS; n.a=a.id; n.b=-1; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }
static Rev r_exp(const Rev& a){ RevNode n; n.val=exp(tape[a.id].val); n.adj=0; n.op=OP_EXP; n.a=a.id; n.b=-1; int id=(int)tape.size(); tape.push_back(n); return Rev{id}; }

static void reset_adj(){ for(auto& n: tape) n.adj=0; }
static void propagate(){
    for(int i=(int)tape.size()-1; i>=0; --i){
        RevNode& n = tape[i];
        double a = n.adj;
        switch(n.op){
            case OP_ADD: tape[n.a].adj += a; tape[n.b].adj += a; break;
            case OP_SUB: tape[n.a].adj += a; tape[n.b].adj -= a; break;
            case OP_MUL: tape[n.a].adj += a*tape[n.b].val; tape[n.b].adj += a*tape[n.a].val; break;
            case OP_SIN: tape[n.a].adj += a*cos(tape[n.a].val); break;
            case OP_COS: tape[n.a].adj += a*(-sin(tape[n.a].val)); break;
            case OP_EXP: tape[n.a].adj += a*exp(tape[n.a].val); break;
            case OP_VAR: break;
        }
    }
}
static void eval_rev(const Rev& x, const Rev& y, Rev& f1, Rev& f2){
    f1 = r_add(r_mul(r_sin(x), r_exp(y)), r_mul(x, y));
    f2 = r_sub(r_mul(x, x), r_cos(y));
}

int main(){
    bool pass = true;
    cout << fixed << setprecision(9);

    double Jfd[2][2]; jacobian_fd(Jfd);

    // (1) 前向模式 → 全雅可比（切向量=I_2，1 次前向）
    Fwd xf = {X, {1,0}};
    Fwd yf = {Y, {0,1}};
    Fwd f1f, f2f; eval_fwd(xf, yf, f1f, f2f);
    double Jfwd[2][2] = {{f1f.d[0], f1f.d[1]}, {f2f.d[0], f2f.d[1]}};
    double ef = 0; for(int i=0;i<2;++i) for(int j=0;j<2;++j) ef = max(ef, fabs(Jfwd[i][j]-Jfd[i][j]));
    cout << "[L4] 前向模式 全雅可比 J:\n";
    cout << "     [" << Jfwd[0][0] << ", " << Jfwd[0][1] << "]\n";
    cout << "     [" << Jfwd[1][0] << ", " << Jfwd[1][1] << "]   与FD最大误差=" << ef << "\n";
    if(!(ef < 1e-6)){ pass=false; cout << "  [失败] 前向模式雅可比与FD不符\n"; }

    // (3) JVP 前向模式：方向 t
    double tx=0.3, ty=-0.7;
    Fwd xj = {X, {tx}};
    Fwd yj = {Y, {ty}};
    Fwd f1j, f2j; eval_fwd(xj, yj, f1j, f2j);
    double JVP[2] = {f1j.d[0], f2j.d[0]};
    double JVP_ref[2] = {Jfd[0][0]*tx + Jfd[0][1]*ty, Jfd[1][0]*tx + Jfd[1][1]*ty};
    double ej = 0; for(int i=0;i<2;++i) ej = max(ej, fabs(JVP[i]-JVP_ref[i]));
    cout << "[L4] 前向 JVP J·t = (" << JVP[0] << ", " << JVP[1] << ")   与FD参考误差=" << ej << "\n";
    if(!(ej < 1e-6)){ pass=false; cout << "  [失败] JVP 与FD参考不符\n"; }

    // (2) 反向模式 → 全雅可比（每个输出种子 1 次）
    Rev xr = make_var(X), yr = make_var(Y);
    Rev f1r, f2r; eval_rev(xr, yr, f1r, f2r);
    double Jrev[2][2];
    reset_adj(); tape[f1r.id].adj = 1; propagate();
    Jrev[0][0] = tape[xr.id].adj; Jrev[0][1] = tape[yr.id].adj;
    reset_adj(); tape[f2r.id].adj = 1; propagate();
    Jrev[1][0] = tape[xr.id].adj; Jrev[1][1] = tape[yr.id].adj;
    double er = 0; for(int i=0;i<2;++i) for(int j=0;j<2;++j) er = max(er, fabs(Jrev[i][j]-Jfd[i][j]));
    cout << "[L4] 反向模式 全雅可比 J:\n";
    cout << "     [" << Jrev[0][0] << ", " << Jrev[0][1] << "]\n";
    cout << "     [" << Jrev[1][0] << ", " << Jrev[1][1] << "]   与FD最大误差=" << er << "\n";
    if(!(er < 1e-6)){ pass=false; cout << "  [失败] 反向模式雅可比与FD不符\n"; }

    // 前向 ≡ 反向
    double econs = 0; for(int i=0;i<2;++i) for(int j=0;j<2;++j) econs = max(econs, fabs(Jfwd[i][j]-Jrev[i][j]));
    cout << "[L4] 前向≡反向 最大差=" << econs << "\n";
    if(!(econs < 1e-12)){ pass=false; cout << "  [失败] 两种模式不一致\n"; }

    // (4) VJP 反向模式：输出种子 u → Jᵀ·u（1 次）
    double u0=0.4, u1=1.1;
    reset_adj(); tape[f1r.id].adj = u0; tape[f2r.id].adj = u1; propagate();
    double VJP[2] = {tape[xr.id].adj, tape[yr.id].adj};
    double VJP_ref[2] = {Jfd[0][0]*u0 + Jfd[1][0]*u1, Jfd[0][1]*u0 + Jfd[1][1]*u1};
    double ev = 0; for(int i=0;i<2;++i) ev = max(ev, fabs(VJP[i]-VJP_ref[i]));
    cout << "[L4] 反向 VJP Jᵀ·u = (" << VJP[0] << ", " << VJP[1] << ")   与FD参考误差=" << ev << "\n";
    if(!(ev < 1e-6)){ pass=false; cout << "  [失败] VJP 与FD参考不符\n"; }

    // (5) 标量损失 L=f1²+f2² 梯度（反向 1 次）
    Rev L = r_add(r_mul(f1r,f1r), r_mul(f2r,f2r));
    reset_adj(); tape[L.id].adj = 1; propagate();
    double gL[2] = {tape[xr.id].adj, tape[yr.id].adj};
    double gx = (Lfun(X+H,Y)-Lfun(X-H,Y))/(2*H);
    double gy = (Lfun(X,Y+H)-Lfun(X,Y-H))/(2*H);
    double el = max(fabs(gL[0]-gx), fabs(gL[1]-gy));
    cout << "[L4] 标量损失 ∇L = (" << gL[0] << ", " << gL[1] << ")   与FD误差=" << el << "\n";
    if(!(el < 1e-6)){ pass=false; cout << "  [失败] 标量损失梯度与FD不符\n"; }

    cout << "\n========================================\n";
    cout << (pass ? " [PASS] 前向/反向 AD 与有限差分一致；JVP/VJP/标量梯度正确" : " [FAIL] 见上") << "\n";
    cout << "========================================\n";
    return pass ? 0 : 1;
}
