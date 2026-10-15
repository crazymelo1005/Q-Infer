// QSA 选择的回归。不引第三方框架。
//
// 四条：宽度公式（含真模型的 2048 + 4 − 1 = 2051 那个上界）；选的确实是最大的 width 个
// （与「全排序后取前 width」的朴素参考逐位比）；并列取索引小的、返回升序；低于上界时**恒等**。
#include "dense/qsa_select.hpp"

#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr QsaShapes kReal{2048, 4};  // 真模型：idx_top_k 2048、idx_block 4

void test_width_formula() {
    // 上界：2048 + 4 − 1 = 2051。短上下文时是恒等，长上下文时钳在上界。
    CHECK(qsa_selection_width(0, kReal) == 0);
    CHECK(qsa_selection_width(1, kReal) == 1);
    CHECK(qsa_selection_width(2050, kReal) == 2050);
    CHECK(qsa_selection_width(2051, kReal) == 2051);
    CHECK(qsa_selection_width(2052, kReal) == 2051);
    CHECK(qsa_selection_width(262144, kReal) == 2051);
    // 块大小为 0 按 1 处理（免得减出下溢）：宽度退化成 idx_top_k 本身。
    CHECK(qsa_selection_width(5, QsaShapes{3, 0}) == 3);
    CHECK(qsa_selection_width(2, QsaShapes{3, 0}) == 2);
}

// 朴素参考：全排序（分数降序、并列索引升序）后取前 width 个，再按索引升序。
std::vector<std::uint64_t> naive(const std::vector<float>& s, std::uint64_t width) {
    std::vector<std::uint64_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), std::uint64_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&s](std::uint64_t a, std::uint64_t b) {
        if (s[a] != s[b]) return s[a] > s[b];
        return a < b;
    });
    std::vector<std::uint64_t> out(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(width));
    std::sort(out.begin(), out.end());
    return out;
}

void test_selects_the_largest_and_orders_by_index() {
    // 分数里刻意埋并列（0.5 出现三次）与负值。
    std::vector<float> s{0.1f, 0.5f, -0.2f, 0.9f, 0.5f, 0.0f, 0.5f, 0.3f, -0.7f};
    const std::uint64_t width = 5;
    std::vector<std::uint64_t> got;
    std::string err;
    CHECK(qsa_select(s.data(), s.size(), width, got, err));
    CHECK(err.empty());
    CHECK(got.size() == width);
    // 升序。
    for (std::size_t i = 1; i < got.size(); ++i) CHECK(got[i - 1] < got[i]);
    // 与朴素参考一致。
    const std::vector<std::uint64_t> want = naive(s, width);
    CHECK(got == want);
    // 前 5 名是 0.9(索引3)、0.5(索引1,4,6)、0.3(索引7) —— 并列那三个都要进来且按索引升序。
    const std::vector<std::uint64_t> expect{1, 3, 4, 6, 7};
    CHECK(got == expect);

    // 并列恰好卡在边界上时，取索引小的那些：分数 [1,1,1,1,1]、width 3 → 取 0,1,2。
    std::vector<float> tie(5, 1.0f);
    CHECK(qsa_select(tie.data(), tie.size(), 3, got, err));
    CHECK((got == std::vector<std::uint64_t>{0, 1, 2}));
}

void test_identity_below_the_bound() {
    // 契约要求低于上界时是**恒等**（不是近似）：n_kv ≤ width 时结果必须是 [0..n_kv-1]。
    std::vector<float> s(100);
    for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = static_cast<float>(i % 7) - 3.0f;  // 故意乱序且有负值
    }
    for (std::uint64_t n : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{17},
                            std::uint64_t{99}, std::uint64_t{100}}) {
        std::vector<std::uint64_t> got;
        std::string err;
        CHECK(qsa_select(s.data(), n, n, got, err));
        CHECK(got.size() == n);
        for (std::uint64_t i = 0; i < n; ++i) CHECK(got[static_cast<std::size_t>(i)] == i);
    }
    // 真模型上「上下文 ≤ 2051 就不必选」：n_kv = 2051 时宽度正好等于 n_kv。
    CHECK(qsa_selection_width(2051, kReal) == 2051);
}

void test_edge_and_failures() {
    std::vector<std::uint64_t> out;
    std::string err;
    std::vector<float> s{1.0f, 2.0f};
    // width 为 0：什么都不选，合法。
    CHECK(qsa_select(s.data(), s.size(), 0, out, err));
    CHECK(out.empty());
    // width > n_kv：明确失败（调用方应当用 qsa_selection_width 求宽度，它会自己钳住）。
    CHECK(!qsa_select(s.data(), s.size(), 3, out, err));
    CHECK(!err.empty());
    err.clear();
    // 空输入 + 非零宽度：n_kv 为 0 时宽度只能是 0，故这里也失败。
    CHECK(!qsa_select(nullptr, 0, 1, out, err));
    CHECK(!err.empty());
    err.clear();
    // 指针为空而 n_kv 非零。
    CHECK(!qsa_select(nullptr, 4, 2, out, err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_width_formula();
    test_selects_the_largest_and_orders_by_index();
    test_identity_below_the_bound();
    test_edge_and_failures();
    std::puts("qsa_select: 宽度公式、取最大的 width 个、并列取小索引且返回升序、上界之下恒等，"
              "边界与失败路径都明确");
    return 0;
}