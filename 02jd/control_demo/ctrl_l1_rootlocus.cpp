// ============================================================
// control_demo L1.3 根轨迹：增益扫描与模角条件
// 纯 C++17。G = K/(s(s+2))，闭环特征 s²+2s+K=0。
// 验证：① 分离点 K=1 (s=-1)；② 幅值条件 |s||s+2|=K(s)；
//       ③ 阻尼线 ζ=0.707 交点 (s=-1±j, K=2) —— 根轨迹设计法核心
// ============================================================
#include <cstdio>
#include <cmath>
#include <complex>
using namespace std;

static void roots2(double a, double b, double c, complex<double>& r1, complex<double>& r2) {
    double d = b * b - 4 * a * c;
    if (d >= 0) {
        double s = sqrt(d);
        r1 = {(-b + s) / (2 * a), 0};
        r2 = {(-b - s) / (2 * a), 0};
    } else {
        double s = sqrt(-d);
        r1 = {(-b) / (2 * a), s / (2 * a)};
        r2 = {(-b) / (2 * a), -s / (2 * a)};
    }
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L1.3 根轨迹：G = K/(s(s+2)) ===\n");

    // ① 分离点: K<1 两实根, K>1 共轭复根 → 分离点在 K=1, s=-1
    complex<double> r099a, r099b, r101a, r101b;
    roots2(1, 2, 0.99, r099a, r099b);
    roots2(1, 2, 1.01, r101a, r101b);
    bool sep_ok = (r099a.imag() == 0 && r099b.imag() == 0) && (r101a.imag() != 0);
    chk("分离点在 K=1 (s=-1)", sep_ok ? 0.0 : 1.0, 0.5);

    // ② 幅值条件: 根轨迹上的点 s 满足 K = |s||s+2| = -(s²+2s)
    complex<double> s(-1.0, 1.5);
    double K_mag = abs(s) * abs(s + 2.0);
    complex<double> K_char = -((s + 2.0) * s);
    chk("幅值条件 |s||s+2| = K(s)", abs(K_mag - K_char.real()), 1e-9);

    // ③ 阻尼线 ζ=0.707 (s=-a±ja) 交点: 代入特征方程 → a=1, K=2
    complex<double> z1, z2;
    roots2(1, 2, 2.0, z1, z2);
    chk("ζ=0.707 交点 (s=-1±j, K=2)",
        abs(z1.real() + 1.0) + abs(z1.imag() - 1.0), 1e-9);

    // 根轨迹扫描表
    printf("  [demo] K 扫描（闭环极点轨迹）:\n");
    for (double K : {0.25, 0.5, 1.0, 2.0, 4.0, 10.0}) {
        complex<double> a, b;
        roots2(1, 2, K, a, b);
        printf("        K=%5.2f  s = %6.3f ± %6.3fj\n", K, a.real(), fabs(a.imag()));
    }
    printf("  [结论] K=1 前沿实轴分离为复根对; 想要 ζ=0.707 就取 K=2（设计即读图）。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
