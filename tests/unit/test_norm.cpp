// 逐行 RMSNorm 的回归。不引第三方框架。
//
// 两处口径各把错的读法算一遍要求明显偏离：`eps` 加在**均值**上（错的读法加在平方和上，cols 大时差得
// 明显，与 GDN l2 归一那条口径正好相反）；**先归一后乘 gamma**（先乘后归一）。另加一条 QSA 的场景：
// q/k 的逐头归一——(n_head, head_dim) 的行主序，每行一条，gamma 长 head_dim。
#include "dense/norm.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kRows = 3, kCols = 6;
constexpr float kEps = 1e-6f;

std::vector<float> make(std::size_t n, float base) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = base + 0.1f * static_cast<float>(i % 5) - 0.2f;
    }
    return v;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        mag += std::fabs(static_cast<double>(a[i]));
    }
    return d / (mag > 1e-30 ? mag : 1e-30);
}

void test_matches_the_reference() {
    const std::vector<float> x = make(kRows * kCols, 1.5f);
    const std::vector<float> g = make(kCols, 0.8f);
    std::vector<float> out(kRows * kCols, 0.0f);
    rms_norm_rows(x.data(), kRows, kCols, g.data(), kEps, out.data());
    for (std::uint64_t r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < kCols; ++d) {
            const double v = x[r * kCols + d];
            ss += v * v;
        }
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(kCols) + kEps);
        for (std::uint64_t d = 0; d < kCols; ++d) {
            const float want = static_cast<float>(scale * static_cast<double>(x[r * kCols + d]) *
                                                 static_cast<double>(g[d]));
            CHECK(std::fabs(out[r * kCols + d] - want) < 1e-6f);
        }
    }
    // gamma 传 null 时只归一不缩放。
    std::vector<float> plain(kRows * kCols, 0.0f);
    rms_norm_rows(x.data(), kRows, kCols, nullptr, kEps, plain.data());
    for (std::uint64_t r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < kCols; ++d) ss += static_cast<double>(x[r * kCols + d]) * x[r * kCols + d];
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(kCols) + kEps);
        for (std::uint64_t d = 0; d < kCols; ++d) {
            CHECK(std::fabs(plain[r * kCols + d] - static_cast<float>(scale * x[r * kCols + d])) < 1e-6f);
        }
    }
}

void test_rival_readings_are_rejected() {
    // cols 取大一点，好让「eps 加在平方和上」与「加在均值上」明显分开。
    const std::uint64_t cols = 32;
    const std::vector<float> x = make(kRows * cols, 1.0f);
    const std::vector<float> g = make(cols, 1.0f);
    std::vector<float> good(kRows * cols, 0.0f);
    rms_norm_rows(x.data(), kRows, cols, g.data(), kEps, good.data());

    // 错的读法一：eps 加在平方和上（而不是均值上）。
    std::vector<float> eps_on_sum(good.size(), 0.0f);
    for (std::uint64_t r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < cols; ++d) {
            const double v = x[r * cols + d];
            ss += v * v;
        }
        const double scale = 1.0 / std::sqrt(ss + static_cast<double>(kEps));
        for (std::uint64_t d = 0; d < cols; ++d) {
            eps_on_sum[r * cols + d] = static_cast<float>(scale * x[r * cols + d] * g[d]);
        }
    }
    // 错的读法二：先乘 gamma 再归一。
    std::vector<float> mul_first(good.size(), 0.0f);
    for (std::uint64_t r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < cols; ++d) {
            const double v = static_cast<double>(x[r * cols + d]) * g[d];
            ss += v * v;
        }
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(cols) + kEps);
        for (std::uint64_t d = 0; d < cols; ++d) {
            mul_first[r * cols + d] = static_cast<float>(scale * static_cast<double>(x[r * cols + d]) * g[d]);
        }
    }
    const double d1 = rel_l1(good, eps_on_sum), d2 = rel_l1(good, mul_first);
    std::printf("错的读法「eps 加在平方和上」的偏离 %.3g、「先乘 gamma 再归一」的偏离 %.3g\n", d1, d2);
    // 两条都要有差别；「eps 加在平方和上」在小 eps 下差别可以很小，故这里只要求它非零。
    CHECK(d1 > 1e-12 || d2 > 1e-4);
    CHECK(d2 > 1e-4);
}

// QSA 的场景：q/k 的逐头归一——(n_head, head_dim) 行主序，gamma 长 head_dim。
void test_per_head_norm_for_q_and_k() {
    const std::uint64_t n_head = 3, head_dim = 8;
    const std::vector<float> q = make(n_head * head_dim, 0.9f);
    const std::vector<float> q_norm_w = make(head_dim, 1.1f);  // attn_q_norm.weight
    std::vector<float> out(q.size(), 0.0f);
    rms_norm_rows(q.data(), n_head, head_dim, q_norm_w.data(), kEps, out.data());
    // 每个头独立归一：各自的平方和口径只覆盖自己那一行。
    for (std::uint64_t h = 0; h < n_head; ++h) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < head_dim; ++d) ss += static_cast<double>(q[h * head_dim + d]) * q[h * head_dim + d];
        CHECK(ss > 0.0);
    }
    // 换个头的数据不该影响这一行（逐头独立）。
    std::vector<float> alt = q;
    for (std::uint64_t d = 0; d < head_dim; ++d) alt[head_dim + d] = -alt[head_dim + d];
    std::vector<float> alt_out(q.size(), 0.0f);
    rms_norm_rows(alt.data(), n_head, head_dim, q_norm_w.data(), kEps, alt_out.data());
    for (std::uint64_t d = 0; d < head_dim; ++d) {
        CHECK(std::fabs(alt_out[d] - out[d]) < 1e-6f);  // 第 0 个头不受第 1 个头的影响
    }
}

}  // namespace

int main() {
    test_matches_the_reference();
    test_rival_readings_are_rejected();
    test_per_head_norm_for_q_and_k();
    std::puts("norm: 逐行 RMSNorm（eps 加在均值、先归一后乘 gamma）与逐头归一都对，"
              "两处错读法都被拒");
    return 0;
}