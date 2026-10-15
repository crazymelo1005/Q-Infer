// QSA 索引器键的追加与池化的回归。不引第三方框架。
//
// 四条：跑完两块之后 pooled[0]/pooled[1] 与独立参考一致；备用槽（最后一格）等于序列 cell 0 的键
// 且整条序列不变；块键的旋转位置是该块**首个** cell 的位置（用 pos_base ≠ 0 的夹具把这条与「按索引」
// 区分开）；块边界按 R 对齐。另把四处错读法各按错的算一遍要求明显偏离。
#include "dense/qsa_indexer.hpp"

#include "dense/norm.hpp"
#include "dense/rope.hpp"

#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kDim = 4;    // idx_dim（真模型 128）
constexpr std::uint64_t kR = 4;      // idx_block（真模型 4）
constexpr std::uint64_t kRot = 4;    // rope 的旋转维数（<= head_dim）
constexpr std::uint64_t kHead = 4;   // head_dim
constexpr double kTheta = 1.0e7;     // 真模型的 rope.freq_base（[S-57]）
constexpr std::uint64_t kPosBase = 100;  // 序列首个 cell 的位置（非 0，用来区分 pos_base 与索引）
constexpr float kEps = 1e-6f;

struct Lcg {
    std::uint64_t s = 0xA0761D6478BD642Full;
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

QsaShapes shapes() { return QsaShapes{2048, kR, kHead, 24, 2}; }

// 独立参考：对给定的若干个 cell 原始键，按 (旋转位置) 算池化键。角度现算（不查表）。
std::vector<float> ref_pooled(const std::vector<std::vector<float>>& cells, const std::vector<float>& gamma,
                              std::uint64_t rope_pos, bool rotate_at_last_cell) {
    std::vector<double> mean(kDim, 0.0);
    for (const auto& c : cells) {
        for (std::uint64_t d = 0; d < kDim; ++d) mean[d] += static_cast<double>(c[d]) / static_cast<double>(kR);
    }
    double ss = 0.0;
    for (std::uint64_t d = 0; d < kDim; ++d) ss += mean[d] * mean[d];
    const double scale = 1.0 / std::sqrt(ss / static_cast<double>(kDim) + kEps);
    std::vector<float> key(kDim, 0.0f);
    for (std::uint64_t d = 0; d < kDim; ++d) {
        key[d] = static_cast<float>(scale * mean[d] * static_cast<double>(gamma[d]));
    }
    // NEOX 部分 rope：前 kRot 维按 (i, i + kRot/2) 配对，角度现算。
    const std::uint64_t last = cells.empty() ? rope_pos : rope_pos + (rotate_at_last_cell ? kR - 1 : 0);
    const std::uint64_t p = rotate_at_last_cell ? last : rope_pos;
    for (std::uint64_t i = 0; i < kRot / 2; ++i) {
        const double ang = static_cast<double>(p) *
                           std::pow(kTheta, -2.0 * static_cast<double>(i) / static_cast<double>(kRot));
        const float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
        float oa = 0.0f, ob = 0.0f;
        rope_neox_pair(key[i], key[i + kRot / 2], c, s, oa, ob);
        key[i] = oa;
        key[i + kRot / 2] = ob;
    }
    return key;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        mag += std::fabs(static_cast<double>(a[i]));
    }
    return d / (mag > 1e-30 ? mag : 1e-30);
}

// 跑 8 个 token（两块），返回每块的原始键与跑完后的状态。
struct Run {
    std::vector<std::vector<std::vector<float>>> blocks;  // [块][cell][dim]
    QsaIndexerState st;
};

Run run_all(const std::vector<float>& gamma, const std::vector<float>& cos_tab,
            const std::vector<float>& sin_tab, bool rotate_at_last_cell) {
    const QsaShapes s = shapes();
    Run r;
    r.st.reset(s, kDim);
    // 8 个 cell 的键**一次生成**（每个都不同）：早先的写法是每个 token 现调一次 make()，
    // 而 make() 每次从同一个种子起算，于是 8 帧完全相同，「块边界错位一格」那条对手就被
    // 相同数据抹平了（变异测试里表现为偏离 0）。
    std::vector<std::vector<float>> cells;
    Lcg g;
    for (std::uint64_t i = 0; i < 2 * kR; ++i) {
        std::vector<float> c(kDim);
        for (float& t : c) t = 2.0f * g.next();
        cells.push_back(c);
    }
    std::string err;
    for (std::uint64_t pos = 0; pos < 2 * kR; ++pos) {
        const std::vector<float>& raw = cells[static_cast<std::size_t>(pos)];
        const std::uint64_t slot = pos % kR;
        // 错的读法：旋转位置取该块**末个** cell。为了让它真的不同，实现侧仍走原函数、
        // 参考侧用不同的位置——故这里只在参考里体现（见调用处）。
        CHECK(indexer_key_append(s, kDim, kRot, pos, kPosBase, raw.data(), gamma.data(), kEps, cos_tab,
                                 sin_tab, r.st, err));
        CHECK(err.empty());
        if (slot + 1 == kR) {
            r.blocks.push_back(std::vector<std::vector<float>>(
                cells.begin() + static_cast<long>(pos + 1 - kR), cells.begin() + static_cast<long>(pos + 1)));
        }
        (void) rotate_at_last_cell;
    }
    return r;
}

void test_matches_the_reference_and_the_spare_slot() {
    const std::vector<float> gamma = make(kDim);
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kRot, kTheta, 4096, cos_tab, sin_tab, err));
    CHECK(err.empty());
    const Run r = run_all(gamma, cos_tab, sin_tab, false);

    // 两块都完成了。
    CHECK(r.blocks.size() == 2);
    CHECK(r.st.n_bid == 2);
    CHECK(r.st.pooled.size() == (2 + 1) * kDim);

    // pooled[b] 与独立参考一致：旋转位置取该块首个 cell 的位置 = pos_base + R*b。
    for (std::uint64_t b = 0; b < 2; ++b) {
        const std::vector<float> want =
            ref_pooled(r.blocks[static_cast<std::size_t>(b)], gamma, kPosBase + kR * b, false);
        std::vector<float> got(r.st.pooled.begin() + static_cast<long>(b * kDim),
                               r.st.pooled.begin() + static_cast<long>((b + 1) * kDim));
        const double d = rel_l1(want, got);
        std::printf("块 %llu 与参考的相对 L1：%.3g\n", static_cast<unsigned long long>(b), d);
        CHECK(d < 1e-5);
    }

    // 备用槽（最后一格）等于序列 cell 0 的键（位置 0 处算出来的），且与 dead 一致。
    const std::vector<float> cell0_key = ref_pooled({r.blocks[0][0]}, gamma, /*pos=*/0, false);
    std::vector<float> spare(r.st.pooled.begin() + static_cast<long>(2 * kDim),
                             r.st.pooled.end());
    CHECK(rel_l1(cell0_key, spare) < 1e-5);
    CHECK(rel_l1(cell0_key, r.st.dead) < 1e-5);
}

void test_rival_readings_are_rejected() {
    const std::vector<float> gamma = make(kDim);
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kRot, kTheta, 4096, cos_tab, sin_tab, err));
    const Run r = run_all(gamma, cos_tab, sin_tab, false);
    const std::vector<float> good =
        ref_pooled(r.blocks[0], gamma, kPosBase + kR * 0, false);

    struct Case {
        const char* name;
        std::uint64_t pos;
        bool rotate_last;
    };
    const std::vector<Case> cases = {
        {"旋转位置取该块末个 cell", kPosBase, true},
        {"把 cell 的位置当成索引（忽略 pos_base）", 0, false},
    };
    for (const Case& c : cases) {
        const std::vector<float> bad = ref_pooled(r.blocks[0], gamma, c.pos, c.rotate_last);
        const double d = rel_l1(good, bad);
        std::printf("错的读法「%s」的偏离 %.3g\n", c.name, d);
        CHECK(d > 1e-4);
    }
    // 块边界按 R 对齐：把第 0 块换成「前 3 个 + 第 4 个」以外的组合（错位一格）应当不同。
    std::vector<std::vector<float>> shifted{r.blocks[0][1], r.blocks[0][2], r.blocks[0][3]};
    shifted.push_back(r.blocks[1][0]);
    const std::vector<float> misaligned = ref_pooled(shifted, gamma, kPosBase + kR * 0, false);
    const double d = rel_l1(good, misaligned);
    std::printf("错的读法「块边界错位一格」的偏离 %.3g\n", d);
    CHECK(d > 1e-4);
}

// 索引器的 q 预处理：按头归一 + rope；并钉住「键是**原样**存的」——把每个原始键先归一再池化，// 得到的池化键与正确做法不同（norm∘mean ≠ mean∘norm），这一条把那条错读法挡在外面。
void test_q_prep_and_raw_keys_stay_raw() {
    const QsaShapes s = shapes();
    const std::uint64_t idx_n_head = 4, idx_dim = kDim;
    const std::vector<float> q_proj = make(idx_n_head * idx_dim);
    const std::vector<float> q_norm_w = make(idx_dim);
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kRot, kTheta, 4096, cos_tab, sin_tab, err));
    std::vector<float> q_idx(idx_n_head * idx_dim, 0.0f);
    CHECK(qsa_indexer_prep(idx_n_head, idx_dim, kRot, 3, q_proj.data(), q_norm_w.data(), kEps,
                           cos_tab, sin_tab, q_idx.data(), err));
    CHECK(err.empty());
    // 与「自己归一、自己现算角度」的参考比（逐头）。
    for (std::uint64_t h = 0; h < idx_n_head; ++h) {
        std::vector<double> row(idx_dim, 0.0);
        for (std::uint64_t d = 0; d < idx_dim; ++d) row[d] = q_proj[h * idx_dim + d];
        double ss = 0.0;
        for (std::uint64_t d = 0; d < idx_dim; ++d) ss += row[d] * row[d];
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(idx_dim) + kEps);
        for (std::uint64_t d = 0; d < idx_dim; ++d) row[d] = scale * row[d] * q_norm_w[d];
        for (std::uint64_t i = 0; i < kRot / 2; ++i) {
            const double ang = 3.0 * std::pow(kTheta, -2.0 * static_cast<double>(i) / static_cast<double>(kRot));
            float oa = 0.0f, ob = 0.0f;
            rope_neox_pair(static_cast<float>(row[i]), static_cast<float>(row[i + kRot / 2]),
                           static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)), oa, ob);
            row[i] = oa;
            row[i + kRot / 2] = ob;
        }
        for (std::uint64_t d = 0; d < idx_dim; ++d) {
            CHECK(std::fabs(q_idx[h * idx_dim + d] - static_cast<float>(row[d])) < 1e-6f);
        }
    }
    // 键必须**原样**存：拿「先逐个归一」的键去池化，结果与用原始键不同。
    QsaIndexerState st_raw, st_normed;
    st_raw.reset(s, idx_dim);
    st_normed.reset(s, idx_dim);
    Lcg g;
    for (std::uint64_t pos = 0; pos < kR; ++pos) {
        std::vector<float> raw(idx_dim);
        for (float& t : raw) t = 2.0f * g.next();
        std::vector<float> prenorm(idx_dim, 0.0f);
        rms_norm_rows(raw.data(), 1, idx_dim, nullptr, kEps, prenorm.data());
        CHECK(indexer_key_append(s, idx_dim, kRot, pos, kPosBase, raw.data(), q_norm_w.data(), kEps,
                                 cos_tab, sin_tab, st_raw, err));
        CHECK(indexer_key_append(s, idx_dim, kRot, pos, kPosBase, prenorm.data(), q_norm_w.data(), kEps,
                                 cos_tab, sin_tab, st_normed, err));
    }
    CHECK(rel_l1(st_raw.pooled, st_normed.pooled) > 1e-4);
}

void test_failures() {
    const QsaShapes s = shapes();
    QsaIndexerState st;
    std::vector<float> raw(kDim, 1.0f), gamma(kDim, 1.0f);
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kRot, kTheta, 64, cos_tab, sin_tab, err));
    // 状态没开。
    CHECK(!indexer_key_append(s, kDim, kRot, 0, kPosBase, raw.data(), gamma.data(), kEps, cos_tab, sin_tab,
                              st, err));
    CHECK(!err.empty());
    err.clear();
    st.reset(s, kDim);
    CHECK(!indexer_key_append(s, kDim, kRot, 0, kPosBase, nullptr, gamma.data(), kEps, cos_tab, sin_tab, st,
                              err));
    CHECK(!err.empty());
    err.clear();
    CHECK(indexer_key_append(s, kDim, kRot, 0, kPosBase, raw.data(), gamma.data(), kEps, cos_tab, sin_tab,
                             st, err));
    CHECK(err.empty());
    CHECK(st.pooled.size() >= kDim);
}

}  // namespace

int main() {
    test_matches_the_reference_and_the_spare_slot();
    test_rival_readings_are_rejected();
    test_q_prep_and_raw_keys_stay_raw();
    test_failures();
    std::puts("qsa_indexer: 池化（mean→归一→rope，旋转在块首 cell）与备用槽（序列 cell 0 的键）"
              "都与独立参考一致，三处错读法都被拒");
    return 0;
}