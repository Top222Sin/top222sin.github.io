// ============================================================
// control_demo L2.2 能控性与能观性（+ 对偶性）
// 纯 C++17。能控性矩阵 Wc=[B AB ...]，能观性矩阵 Wo=[C;CA;...]
// 验证：① 典型二阶系统能控能观；② 不能控反例（并联模态单输入）；
//       ③ 对偶性 (A,C)能观 ⇔ (Aᵀ,Cᵀ)能控
// （与 KF_demo L3.2 的可观测性判据同源：滤波器能观 / 控制器能控）
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
using Mat = vector<vector<double>>;

static Mat matmul(const Mat& A, const Mat& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Mat C(n, vector<double>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < k; ++p) { double a = A[i][p]; if (a) for (int j = 0; j < m; ++j) C[i][j] += a * B[p][j]; }
    return C;
}
static int rank_of(Mat A, double tol = 1e-8) {
    int m = A.size(), n = A[0].size(), r = 0;
    for (int c = 0; c < n; ++c) {
        int piv = -1;
        for (int i = r; i < m; ++i) if (fabs(A[i][c]) > tol) { piv = i; break; }
        if (piv < 0) continue;
        swap(A[r], A[piv]);
        for (int i = 0; i < m; ++i) if (i != r && A[i][c] != 0) {
            double f = A[i][c] / A[r][c];
            for (int j = 0; j < n; ++j) A[i][j] -= f * A[r][j];
        }
        ++r;
    }
    return r;
}

int main() {
    int pass = 0, fail = 0;
    auto chk = [&](const char* name, double err, double tol) {
        bool ok = err < tol;
        printf("  %-38s err=%.3e tol=%.0e  %s\n", name, err, tol, ok ? "PASS" : "FAIL");
        ok ? ++pass : ++fail;
    };
    printf("=== L2.2 能控性 / 能观性 ===\n");

    Mat A = {{0, 1}, {-4.0, -0.8}};
    Mat B = {{0.0}, {1.0}};
    Mat C = {{1.0, 0.0}};
    Mat AB = matmul(A, B);
    Mat Wc = {{B[0][0], AB[0][0]}, {B[1][0], AB[1][0]}};
    Mat CA = matmul(C, A);
    Mat Wo = {{C[0][0], C[0][1]}, {CA[0][0], CA[0][1]}};
    int rc = rank_of(Wc), ro = rank_of(Wo);
    chk("[B AB] 满秩（能控）", rc == 2 ? 0.0 : 1.0, 0.5);
    chk("[C;CA] 满秩（能观）", ro == 2 ? 0.0 : 1.0, 0.5);
    printf("  [demo] rank Wc=%d, rank Wo=%d\n", rc, ro);

    // 反例: 并联模态只激得起一路
    Mat Wc2 = {{1.0, -1.0}, {0.0, 0.0}};   // B=[1;0], A=diag(-1,-2)
    chk("反例 rank=1（不能控）", rank_of(Wc2) == 1 ? 0.0 : 1.0, 0.5);
    printf("  [demo] 反例 A=diag(-1,-2), B=[1;0]: 模态 -2 完全不受控\n");

    // 对偶性: Wo 的秩 = (Aᵀ,Cᵀ) 能控性矩阵的秩（互为转置）
    Mat WoT = {{Wo[0][0], Wo[1][0]}, {Wo[0][1], Wo[1][1]}};
    chk("对偶性: 能观秩=能控秩(转置)", abs(rank_of(WoT) - ro), 0.5);
    printf("  [结论] 能控⇔极点可任意配置(L2.3)；能观⇔观测器可任意配置(L2.4)；\n");
    printf("         对偶性让两套理论共用一套证明。卡尔曼分解是总纲。\n");
    printf("=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
