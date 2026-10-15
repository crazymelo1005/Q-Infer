// NEOX 部分 RoPE 的回归。不引第三方框架。
//
// 四条：与「用 double 现算角度、按 (i, i+n_rot/2) 配对」的独立参考逐元素一致（参考不查表，故也
// 顺带验了表的构造）；位置 0 是恒等；尾巴（≥ n_rot 的那些维）原样穿过；就地（out == x）与非就地
// 结果相同。另把四处错读法各按错的算一遍要求明显偏离。
#include "dense/rope.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kHd = 8;     // head_dim
constexpr std::uint64_t kNrot = 4;   // 只转前 4 维（真模型是 256 里转 64，比例同为 1/4）
constexpr double kTheta = 10000.0;
constexpr std::uint64_t kPos = 6;

std::vector<float> make_x(std::size_t n) {
    std::vector<float> v(n);
    std::uint64_t s = 12345;
    for (float& t : v) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        t = static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f - 1.0f;
    }
    return v;
}

// 独立参考：角度现算（double），NEOX 配对，尾巴穿过。不查表。
std::vector<float> ref_apply(const std::vector<float>& x, std::uint64_t rows, std::uint64_t head_dim,
                             std::uint64_t n_rot, const std::vector<std::uint64_t>& pos,
                             bool adjacent, bool rotate_all, bool flip_sign) {
    const std::uint64_t rot = rotate_all ? head_dim : n_rot;
    const std::uint64_t half = rot / 2;
    std::vector<float> out(x.size());
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t d = 0; d < head_dim; ++d) {
            out[r * head_dim + d] = x[r * head_dim + d];
        }
        for (std::uint64_t i = 0; i < half; ++i) {
            const std::uint64_t ia = adjacent ? 2 * i : i;
            const std::uint64_t ib = adjacent ? 2 * i + 1 : i + half;
            const double ang = static_cast<double>(pos[r]) *
                               std::pow(kTheta, -2.0 * static_cast<double>(i) /
                                                    static_cast<double>(rot));
            const double c = std::cos(ang), s = std::sin(ang) * (flip_sign ? -1.0 : 1.0);
            const double a = x[r * head_dim + ia], b = x[r * head_dim + ib];
            out[r * head_dim + ia] = static_cast<float>(a * c - b * s);
            out[r * head_dim + ib] = static_cast<float>(a * s + b * c);
        }
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

void test_matches_the_independent_reference() {
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kNrot, kTheta, kPos, cos_tab, sin_tab, err));
    CHECK(err.empty());
    CHECK(cos_tab.size() == kPos * (kNrot / 2));
    // 位置 0：所有角都是 0，故 cos 全 1、sin 全 0。
    for (std::uint64_t i = 0; i < kNrot / 2; ++i) {
        CHECK(cos_tab[i] == 1.0f);
        CHECK(sin_tab[i] == 0.0f);
    }
    // 第 0 对的角就是 p（theta^0 = 1）。
    CHECK(std::fabs(cos_tab[1 * (kNrot / 2)] - static_cast<float>(std::cos(1.0))) < 1e-6f);

    const std::uint64_t rows = 3;
    const std::vector<std::uint64_t> pos{0, 3, 5};
    const std::vector<float> x = make_x(rows * kHd);
    std::vector<float> out(x.size(), 0.0f);
    CHECK(rope_neox_apply(x.data(), rows, kHd, kNrot, pos.data(), cos_tab, sin_tab, out.data(), err));
    CHECK(err.empty());
    const std::vector<float> want =
        ref_apply(x, rows, kHd, kNrot, pos, /*adjacent=*/false, /*rotate_all=*/false, /*flip=*/false);
    const double d = rel_l1(want, out);
    std::printf("与独立参考（现算角度）的相对 L1：%.3g\n", d);
    CHECK(d < 1e-6);

    // 位置 0 那一行是恒等；尾巴（≥ n_rot）在所有行都原样。
    for (std::uint64_t d = 0; d < kHd; ++d) CHECK(out[d] == x[d]);
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t d = kNrot; d < kHd; ++d) {
            CHECK(out[r * kHd + d] == x[r * kHd + d]);
        }
    }
    // 就地与非就地一致。
    std::vector<float> inplace = x;
    CHECK(rope_neox_apply(inplace.data(), rows, kHd, kNrot, pos.data(), cos_tab, sin_tab,
                          inplace.data(), err));
    for (std::size_t i = 0; i < inplace.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(inplace[i]) == std::bit_cast<std::uint32_t>(out[i]));
    }
}

void test_rival_readings_are_rejected() {
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(build_rope_table(kNrot, kTheta, kPos, cos_tab, sin_tab, err));
    const std::uint64_t rows = 3;
    const std::vector<std::uint64_t> pos{0, 3, 5};
    const std::vector<float> x = make_x(rows * kHd);
    std::vector<float> got(x.size(), 0.0f);
    CHECK(rope_neox_apply(x.data(), rows, kHd, kNrot, pos.data(), cos_tab, sin_tab, got.data(), err));
    const std::vector<float> good =
        ref_apply(x, rows, kHd, kNrot, pos, false, false, false);

    struct Case {
        const char* name;
        bool adjacent, rotate_all, flip;
    };
    const std::vector<Case> cases = {
        {"配对用相邻 (2i, 2i+1)", true, false, false},
        {"把整条头都转了（n_rot 当 head_dim）", false, true, false},
        {"一对里的符号取反", false, false, true},
    };
    for (const Case& c : cases) {
        const std::vector<float> bad = ref_apply(x, rows, kHd, kNrot, pos, c.adjacent, c.rotate_all,
                                                 c.flip);
        const double d = rel_l1(good, bad);
        std::printf("错的读法「%s」的最大偏离 %.3g\n", c.name, d);
        CHECK(d > 1e-4);
    }
}

void test_failures() {
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    CHECK(!build_rope_table(0, kTheta, kPos, cos_tab, sin_tab, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!build_rope_table(3, kTheta, kPos, cos_tab, sin_tab, err));  // 奇数
    CHECK(!err.empty());
    err.clear();
    CHECK(!build_rope_table(kNrot, 0.0, kPos, cos_tab, sin_tab, err));  // theta <= 0
    CHECK(!err.empty());
    err.clear();
    CHECK(build_rope_table(kNrot, kTheta, kPos, cos_tab, sin_tab, err));
    const std::vector<std::uint64_t> pos{7};  // 超出表长
    std::vector<float> x(kHd, 1.0f), out(kHd, 0.0f);
    CHECK(!rope_neox_apply(x.data(), 1, kHd, kNrot, pos.data(), cos_tab, sin_tab, out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!rope_neox_apply(x.data(), 1, kHd, kHd + 2, pos.data(), cos_tab, sin_tab, out.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_independent_reference();
    test_rival_readings_are_rejected();
    test_failures();
    std::puts("rope: NEOX 配对、部分旋转（尾巴穿过）、表在 double 里算、就地与非就地一致，"
              "三处错读法都被拒");
    return 0;
}