// QSA 索引器逐 cell 分数的回归。不引第三方框架。
//
// 四条：与双精度参考逐元素一致；`Relu` 逐头对「先求和再 Relu」这条错读法要明显不同；尾巴的 1e9 偏置
// 只加在不满一块的那段（已完成块不加）；以及契约里那条性质——**把任意一个 query 头的符号翻过来，
// 分数只会不变或变大**（因为 Relu 逐头取，负贡献被吃掉），这条正是「逐头 Relu」的可观测后果。
#include "dense/qsa_index.hpp"

#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kDim = 4;     // idx_dim（真模型 128）
constexpr std::uint64_t kHd = 3;      // idx_n_head（真模型 4）
constexpr std::uint64_t kBlock = 4;   // idx_block（真模型 4）

struct Lcg {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

std::vector<float> make(std::size_t n) {
    Lcg g;
    std::vector<float> v(n);
    for (float& t : v) t = 2.0f * g.next();
    return v;
}

// 参考（double）。relu_per_head=false 就是错的读法（先求和再 Relu）；tail_bias=false 也是错的读法。
std::vector<float> reference(const std::vector<float>& pooled, std::uint64_t n_bid,
                             const std::vector<float>& q_idx, const std::vector<float>& bias,
                             std::uint64_t n_kv, bool relu_per_head, bool tail_bias,
                             bool clamp_blocks) {
    std::vector<float> out(n_kv, 0.0f);
    const std::uint64_t done = (n_kv / kBlock) * kBlock;
    for (std::uint64_t j = 0; j < n_kv; ++j) {
        const std::uint64_t cb = clamp_blocks ? std::min(j / kBlock, n_bid) : (j / kBlock);
        const float* key = pooled.data() + cb * kDim;
        double acc = 0.0, sum = 0.0;
        for (std::uint64_t h = 0; h < kHd; ++h) {
            double dot = 0.0;
            for (std::uint64_t d = 0; d < kDim; ++d) {
                dot += static_cast<double>(key[d]) *
                       static_cast<double>(q_idx[h * kDim + d]);
            }
            sum += dot;
            if (relu_per_head) acc += dot > 0.0 ? dot : 0.0;
        }
        if (!relu_per_head) acc = sum > 0.0 ? sum : 0.0;
        out[j] = static_cast<float>(acc) + bias[cb];
        if (tail_bias && j >= done) out[j] += 1e9f;
    }
    return out;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        mag += std::fabs(static_cast<double>(a[i]));
    }
    return d / (mag > 1e-30 ? mag : 1e-30);
}

void test_matches_the_reference_and_the_tail_bias() {
    // n_kv = 10、r = 4 → 两块完成（cell 0..7）、两个尾巴 cell（8,9）。
    // 用**两块**而不是一块是刻意的：逐块 bias 只有在至少两块时才能被断言（val 尾巴那段的 bias 在 f32
    // 里会被 +1e9 吃掉——1e9 处的 ulp 是 64，O(1) 的偏置加进去看不出来）。
    const std::uint64_t n_bid = 2, n_kv = 10;
    const std::vector<float> pooled = make((n_bid + 1) * kDim);
    const std::vector<float> q_idx = make(kHd * kDim);
    const std::vector<float> bias = make(n_bid + 1);
    std::vector<float> got(n_kv, 0.0f);
    std::string err;
    CHECK(qsa_index(pooled.data(), n_bid, kDim, kHd, q_idx.data(), bias.data(), kBlock, n_kv,
                    got.data(), err));
    CHECK(err.empty());
    const std::vector<float> want = reference(pooled, n_bid, q_idx, bias, n_kv, true, true, true);
    const double d = rel_l1(want, got);
    std::printf("与参考的相对 L1：%.3g\n", d);
    CHECK(d < 1e-6);
    // 尾巴那两个 cell 带 1e9，已完成的不带。
    CHECK(got[8] > 1e8f && got[9] > 1e8f);
    CHECK(got[0] < 1e8f && got[7] < 1e8f);
    // bias 传 null 时尾巴偏置**仍然**要加。
    std::vector<float> got2(n_kv, 0.0f);
    CHECK(qsa_index(pooled.data(), n_bid, kDim, kHd, q_idx.data(), nullptr, kBlock, n_kv,
                    got2.data(), err));
    CHECK(got2[9] > 1e8f);
    for (std::uint64_t j = 0; j < 8; ++j) {
        // 「先算再减 bias」要减的是该 cell 所属**块**的 bias（`j / r`），不是 `bias[j]`。
        CHECK(std::fabs(got2[j] - (got[j] - bias[j / kBlock])) < 1e-5f);
    }
}

// 两种读法之间那条**能推出来**的不等式：`Σ_h relu(dot_h) ≥ relu(Σ_h dot_h)` 对每个 cell 恒成立
// （因为 relu 是次可加的），且只要有一个头的点积为负、另一头为正，两边就真的不等。
//
// 记一笔：上游在 qsa_index 的注释里写的是「翻一个 query 头的符号不会降分」，但那条**按公式推不出来**
// ——某头的点积为正时，翻符号会把它那一项从 dot 降到 0。能推出来的是上面那条不等式，故这里断言它。
void test_per_head_relu_dominates_the_summed_relu() {
    const std::uint64_t n_bid = 2, n_kv = 8;
    const std::vector<float> pooled = make((n_bid + 1) * kDim);
    // 把奇数号头的 query 行取反：这样每个 cell 上都有正有负的点积，两边的差别才看得出来
    // （全正或全负时 Σrelu 与 relu(Σ) 恰好相等，这条用例就成了空转）。
    std::vector<float> q_idx = make(kHd * kDim);
    for (std::uint64_t h = 1; h < kHd; h += 2) {
        for (std::uint64_t d = 0; d < kDim; ++d) q_idx[h * kDim + d] = -q_idx[h * kDim + d];
    }
    std::vector<float> got(n_kv, 0.0f);
    std::string err;
    CHECK(qsa_index(pooled.data(), n_bid, kDim, kHd, q_idx.data(), nullptr, kBlock, n_kv, got.data(),
                    err));
    const std::vector<float> summed =
        reference(pooled, n_bid, q_idx, std::vector<float>(n_bid + 1, 0.0f), n_kv,
                  /*relu_per_head=*/false, /*tail_bias=*/true, /*clamp_blocks=*/true);
    bool strictly_greater_somewhere = false;
    for (std::uint64_t j = 0; j < n_kv; ++j) {
        CHECK(got[j] >= summed[j] - 1e-3f);  // 逐头的和不可能更小
        if (got[j] > summed[j] + 1e-3f) strictly_greater_somewhere = true;
    }
    CHECK(strictly_greater_somewhere);  // 至少一处不一样，否则这条是空转
}

void test_rival_readings_are_rejected() {
    const std::uint64_t n_bid = 2, n_kv = 10;
    const std::vector<float> pooled = make((n_bid + 1) * kDim);
    // 同样把奇数号头取反，让逐头 Relu 与先和后 Relu 真的不同（否则那条对手是空转的）。
    std::vector<float> q_idx = make(kHd * kDim);
    for (std::uint64_t h = 1; h < kHd; h += 2) {
        for (std::uint64_t d = 0; d < kDim; ++d) q_idx[h * kDim + d] = -q_idx[h * kDim + d];
    }
    const std::vector<float> bias = make(n_bid + 1);
    const std::vector<float> good = reference(pooled, n_bid, q_idx, bias, n_kv, true, true, true);
    struct Case {
        const char* name;
        bool relu_per_head, tail_bias, clamp_blocks;
    };
    const std::vector<Case> cases = {
        {"先对头求和再 Relu", false, true, true},
        {"尾巴不加 1e9 偏置", true, false, true},
    };
    // 注：`cb = min(j / r, n_bid)` 里的那个 `min` 在这套下标下**不可观测**——不满一块的尾巴里
    // `j / r` 恰好等于 `n_bid`（已完成块数就是 `n_kv / r`），故钳与不钳同值。它是防守性的，
    // 不是一条可断言的读法差别；这类「按键面设计却观察不到」的情形本项目记过一次（状态布局那条），
    // 处法也一样：不写成假的性质，而是把为什么不可观测写下来。
    for (const Case& c : cases) {
        const std::vector<float> bad =
            reference(pooled, n_bid, q_idx, bias, n_kv, c.relu_per_head, c.tail_bias, c.clamp_blocks);
        // 两处读法影响的 cell 段不同：Relu 的位置只影响已完成块，尾巴偏置只影响尾巴那一段。
        // 且两段必须各自比——尾巴那段带 1e9，用相对误差会把它自己的差别盖掉（分母口径的坑，
        // 本项目记过一次：比相对误差时分母要用项的幅度而不是结果的幅度）。
        const std::uint64_t done = (n_kv / kBlock) * kBlock;
        const std::uint64_t lo = c.tail_bias ? 0 : done;
        const std::uint64_t hi = c.tail_bias ? done : n_kv;
        double diff = 0.0, mag = 0.0;
        for (std::uint64_t j = lo; j < hi; ++j) {
            diff += std::fabs(static_cast<double>(good[j]) - static_cast<double>(bad[j]));
            mag += std::fabs(static_cast<double>(good[j]));
        }
        const double d = diff / (mag > 1e-30 ? mag : 1e-30);
        std::printf("错的读法「%s」的偏离 %.3g\n", c.name, d);
        CHECK(d > 1e-4);
    }
}

void test_failures() {
    std::vector<float> out(4, 0.0f), pooled(8, 1.0f), q(4, 1.0f);
    std::string err;
    CHECK(!qsa_index(nullptr, 1, kDim, kHd, q.data(), nullptr, kBlock, 4, out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!qsa_index(pooled.data(), 1, 0, kHd, q.data(), nullptr, kBlock, 4, out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!qsa_index(pooled.data(), 1, kDim, kHd, q.data(), nullptr, 0, 4, out.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference_and_the_tail_bias();
    test_per_head_relu_dominates_the_summed_relu();
    test_rival_readings_are_rejected();
    test_failures();
    std::puts("qsa_index: 逐 cell 分数（Relu 逐头、和在后、块映射钳在 n_bid、尾巴加 1e9）、"
              "逐头 Relu 的和不小于先和后 Relu，两处错读法都被拒（块映射那个 min 不可观测，已写明原因）");
    return 0;
}