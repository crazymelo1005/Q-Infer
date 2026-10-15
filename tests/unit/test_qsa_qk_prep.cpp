// QSA 的 q/k 预处理的回归。不引第三方框架。
//
// 四条：与独立参考一致（参考自己拆、自己归一、自己现算角度做 rope）；门原样（不归一不 rope）；
// 尾巴（≥ n_rot 的维）在归一后仍按归一值穿过、不被 rope 动；四处错读法各按错的算一遍要求明显偏离
// （交错拆、先 rope 后归一、门也归一、整条头都转）。
#include "dense/qsa_qk_prep.hpp"

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

constexpr std::uint64_t kNh = 4;    // q 头数
constexpr std::uint64_t kNkv = 2;   // kv 头数
constexpr std::uint64_t kHd = 8;    // head_dim
constexpr std::uint64_t kNrot = 4;  // 只转前 4 维
constexpr std::uint64_t kPos = 5;
constexpr double kTheta = 1.0e7;  // 真模型的 rope.freq_base
constexpr float kEps = 1e-6f;

struct Lcg {
    std::uint64_t s = 0x8EBC6AF09C88C6E3ull;
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

// 独立参考：交错/块拆、是否先 rope、门是否归一、是否整条转，各一个开关。角度现算，不查表。
struct Opts {
    bool interleaved_split = false;
    bool rope_before_norm = false;
    bool normalize_gate = false;
    bool rotate_all = false;
};

void reference(const Opts& o, std::uint64_t pos, const std::vector<float>& q_full,
               const std::vector<float>& k_raw, const std::vector<float>& qg,
               const std::vector<float>& kg, std::vector<float>& q_out,
               std::vector<float>& k_out, std::vector<float>& gate_out) {
    const std::uint64_t rot = o.rotate_all ? kHd : kNrot;
    auto rot_row = [&](std::vector<double>& row) {
        for (std::uint64_t i = 0; i < rot / 2; ++i) {
            const double ang = static_cast<double>(pos) *
                               std::pow(kTheta, -2.0 * static_cast<double>(i) / static_cast<double>(rot));
            const float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
            float oa = 0.0f, ob = 0.0f;
            rope_neox_pair(static_cast<float>(row[i]), static_cast<float>(row[i + rot / 2]), c, s, oa, ob);
            row[i] = oa;
            row[i + rot / 2] = ob;
        }
    };
    auto norm_row = [&](std::vector<double>& row, const std::vector<float>& g) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < kHd; ++d) ss += row[d] * row[d];
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(kHd) + kEps);
        for (std::uint64_t d = 0; d < kHd; ++d) row[d] = scale * row[d] * g[d];
    };
    auto head = [&](const float* src, std::uint64_t h, bool interleaved) {
        std::vector<double> row(kHd, 0.0);
        for (std::uint64_t d = 0; d < kHd; ++d) {
            row[d] = interleaved ? src[h * 2 * kHd + 2 * d] : src[h * 2 * kHd + d];
        }
        return row;
    };
    q_out.assign(kNh * kHd, 0.0f);
    gate_out.assign(kNh * kHd, 0.0f);
    for (std::uint64_t h = 0; h < kNh; ++h) {
        std::vector<double> q = head(q_full.data(), h, o.interleaved_split);
        for (std::uint64_t d = 0; d < kHd; ++d) {
            gate_out[h * kHd + d] = o.interleaved_split ? q_full[h * 2 * kHd + 2 * d + 1]
                                                        : q_full[h * 2 * kHd + kHd + d];
        }
        if (o.rope_before_norm) rot_row(q);
        norm_row(q, qg);
        if (!o.rope_before_norm) rot_row(q);
        for (std::uint64_t d = 0; d < kHd; ++d) q_out[h * kHd + d] = static_cast<float>(q[d]);
    }
    if (o.normalize_gate) {
        for (std::uint64_t h = 0; h < kNh; ++h) {
            std::vector<double> g(kHd, 0.0);
            for (std::uint64_t d = 0; d < kHd; ++d) g[d] = gate_out[h * kHd + d];
            norm_row(g, qg);
            for (std::uint64_t d = 0; d < kHd; ++d) gate_out[h * kHd + d] = static_cast<float>(g[d]);
        }
    }
    k_out.assign(kNkv * kHd, 0.0f);
    for (std::uint64_t h = 0; h < kNkv; ++h) {
        std::vector<double> k(kHd, 0.0);
        for (std::uint64_t d = 0; d < kHd; ++d) k[d] = k_raw[h * kHd + d];
        if (o.rope_before_norm) rot_row(k);
        norm_row(k, kg);
        if (!o.rope_before_norm) rot_row(k);
        for (std::uint64_t d = 0; d < kHd; ++d) k_out[h * kHd + d] = static_cast<float>(k[d]);
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        mag += std::fabs(static_cast<double>(a[i]));
    }
    return d / (mag > 1e-30 ? mag : 1e-30);
}

struct Fixture {
    std::vector<float> q_full = make(kNh * 2 * kHd);
    std::vector<float> k_raw = make(kNkv * kHd);
    std::vector<float> qg = make(kHd);
    std::vector<float> kg = make(kHd);
    std::vector<float> cos_tab, sin_tab;
    Fixture() {
        std::string err;
        CHECK(build_rope_table(kNrot, kTheta, 64, cos_tab, sin_tab, err));
        CHECK(err.empty());
    }
    QsaQkWeights weights() const { return QsaQkWeights{qg.data(), kg.data(), kEps}; }
};

void test_matches_the_reference() {
    const Fixture f;
    std::vector<float> q(kNh * kHd, 0.0f), k(kNkv * kHd, 0.0f), g(kNh * kHd, 0.0f);
    std::string err;
    CHECK(qsa_qk_prep(kNh, kNkv, kHd, kNrot, kPos, f.q_full.data(), f.k_raw.data(), f.weights(),
                      f.cos_tab, f.sin_tab, q.data(), k.data(), g.data(), err));
    CHECK(err.empty());
    std::vector<float> rq, rk, rg;
    reference(Opts{}, kPos, f.q_full, f.k_raw, f.qg, f.kg, rq, rk, rg);
    const double dq = rel_l1(rq, q), dk = rel_l1(rk, k), dg = rel_l1(rg, g);
    std::printf("与参考的相对 L1：q %.3g、k %.3g、gate %.3g\n", dq, dk, dg);
    CHECK(dq < 1e-6 && dk < 1e-6 && dg < 1e-6);

    // 门原样：就是那个块的后半段，没被动过。
    for (std::uint64_t h = 0; h < kNh; ++h) {
        for (std::uint64_t d = 0; d < kHd; ++d) {
            CHECK(g[h * kHd + d] == f.q_full[h * 2 * kHd + kHd + d]);
        }
    }
    // 尾巴（≥ n_rot）在归一后仍等于「归一值」（被 rope 动过就不同）。
    for (std::uint64_t h = 0; h < kNh; ++h) {
        double ss = 0.0;
        for (std::uint64_t d = 0; d < kHd; ++d) ss += static_cast<double>(f.q_full[h * 2 * kHd + d]) * f.q_full[h * 2 * kHd + d];
        const float scale = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(kHd) + kEps));
        for (std::uint64_t d = kNrot; d < kHd; ++d) {
            const float want = scale * f.q_full[h * 2 * kHd + d] * f.qg[d];
            CHECK(std::fabs(q[h * kHd + d] - want) < 1e-6f);
        }
    }
}

void test_rival_readings_are_rejected() {
    const Fixture f;
    std::vector<float> q(kNh * kHd, 0.0f), k(kNkv * kHd, 0.0f), g(kNh * kHd, 0.0f);
    std::string err;
    CHECK(qsa_qk_prep(kNh, kNkv, kHd, kNrot, kPos, f.q_full.data(), f.k_raw.data(), f.weights(),
                      f.cos_tab, f.sin_tab, q.data(), k.data(), g.data(), err));
    std::vector<float> rq, rk, rg;
    reference(Opts{}, kPos, f.q_full, f.k_raw, f.qg, f.kg, rq, rk, rg);

    struct Case {
        const char* name;
        Opts o;
        bool gate_side;
    };
    std::vector<Case> cases;
    {
        Opts o;
        o.interleaved_split = true;
        cases.push_back({"q/门按元素交错拆", o, true});
    }
    {
        Opts o;
        o.rope_before_norm = true;
        cases.push_back({"先 rope 后归一", o, false});
    }
    {
        Opts o;
        o.normalize_gate = true;
        cases.push_back({"门也归一", o, true});
    }
    {
        Opts o;
        o.rotate_all = true;
        cases.push_back({"整条头都转（n_rot 当 head_dim）", o, false});
    }
    for (const Case& c : cases) {
        std::vector<float> bq, bk, bg;
        reference(c.o, kPos, f.q_full, f.k_raw, f.qg, f.kg, bq, bk, bg);
        const double worst = std::max({rel_l1(rq, bq), rel_l1(rk, bk), rel_l1(rg, bg)});
        std::printf("错的读法「%s」的偏离 %.3g\n", c.name, worst);
        CHECK(worst > 1e-4);
        (void) c.gate_side;
    }
}

void test_failures() {
    const Fixture f;
    std::vector<float> q(kNh * kHd, 0.0f), k(kNkv * kHd, 0.0f), g(kNh * kHd, 0.0f);
    std::string err;
    CHECK(!qsa_qk_prep(0, kNkv, kHd, kNrot, kPos, f.q_full.data(), f.k_raw.data(), f.weights(),
                       f.cos_tab, f.sin_tab, q.data(), k.data(), g.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!qsa_qk_prep(kNh, kNkv, kHd, kHd + 2, kPos, f.q_full.data(), f.k_raw.data(), f.weights(),
                       f.cos_tab, f.sin_tab, q.data(), k.data(), g.data(), err));
    CHECK(!err.empty());
    err.clear();
    QsaQkWeights bad = f.weights();
    bad.q_norm = nullptr;
    CHECK(!qsa_qk_prep(kNh, kNkv, kHd, kNrot, kPos, f.q_full.data(), f.k_raw.data(), bad, f.cos_tab,
                       f.sin_tab, q.data(), k.data(), g.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!qsa_qk_prep(kNh, kNkv, kHd, kNrot, kPos, nullptr, f.k_raw.data(), f.weights(), f.cos_tab,
                       f.sin_tab, q.data(), k.data(), g.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference();
    test_rival_readings_are_rejected();
    test_failures();
    std::puts("qsa_qk_prep: q/门的块拆分、按头归一、部分 NEOX rope 都与独立参考一致，"
              "门原样、尾巴穿过，四处错读法都被拒");
    return 0;
}