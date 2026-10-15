// QSA 的注意力与门：softmax GQA（只在选中的 cell 上）与 `attn · sigmoid(gate)`。不引第三方框架。
//
// 四处「都产出形状正确的结果」的地方各按错的算一遍要求明显偏离：q→kv 头用取模而不是整除、
// 门用 silu 而不是 sigmoid、门取自交错位置而不是每头块的后半段、分数不除 sqrt(head_dim)。
// 另有三条范围：ids 为空 → attn 全 0（合法）；n_head 不是 n_head_kv 的整数倍 → 明确失败；
// softmax 的权重与 attn 自洽（权重和为 1、且用权重重算 attn 与输出一致）。
#include "dense/qsa_attend.hpp"

#include "check.hpp"
#include "kernels/fp16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kHd = 4;    // 头宽
constexpr std::uint64_t kNh = 4;    // q 头数
constexpr std::uint64_t kNkv = 2;   // kv 头数（GQA 4:2）
constexpr std::uint64_t kIds = 3;   // 选中的 cell 数

struct Lcg {
    std::uint64_t s = 0xD1B54A32D192ED03ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

QsaShapes shapes() { return QsaShapes{2048, 4, kHd, kNh, kNkv}; }

std::vector<std::uint16_t> make_f16(std::size_t n) {
    Lcg g;
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = f32_to_f16_bits(g.next());
    return v;
}

std::vector<float> make_f32(std::size_t n) {
    std::vector<float> v(n);
    Lcg g;
    for (float& t : v) t = g.next();
    return v;
}

// 参考实现（double），每处陷阱一个开关。
struct Opts {
    bool divide_head_map = true;   // false：用取模
    bool sigmoid_gate = true;      // false：用 silu
    bool gate_second_half = true;  // false：按元素交错取门
    bool scale_scores = true;      // false：不除 sqrt(head_dim)
};

void reference(const Opts& o, const QsaShapes& s, const std::vector<float>& q,
               const std::vector<std::uint16_t>& k, const std::vector<std::uint16_t>& v,
               std::uint64_t n_ids, std::vector<float>& attn, std::vector<float>& w) {
    const std::uint64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
    attn.assign(nh * hd, 0.0f);
    w.assign(nh * n_ids, 0.0f);
    if (n_ids == 0) return;
    const double scale = o.scale_scores ? 1.0 / std::sqrt(static_cast<double>(hd)) : 1.0;
    for (std::uint64_t h = 0; h < nh; ++h) {
        const std::uint64_t kvh = o.divide_head_map ? h / (nh / nkv) : h % nkv;
        std::vector<double> sc(n_ids, 0.0);
        double mx = -1e300;
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            double acc = 0.0;
            for (std::uint64_t d = 0; d < hd; ++d) {
                acc += static_cast<double>(f16_bits_to_f32(k[(j * nkv + kvh) * hd + d])) *
                       static_cast<double>(q[h * hd + d]);
            }
            sc[j] = acc * scale;
            if (sc[j] > mx) mx = sc[j];
        }
        double sum = 0.0;
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            sc[j] = std::exp(sc[j] - mx);
            sum += sc[j];
        }
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            w[h * n_ids + j] = static_cast<float>(sc[j] / sum);
            for (std::uint64_t d = 0; d < hd; ++d) {
                attn[h * hd + d] += static_cast<float>(sc[j] / sum *
                                                       static_cast<double>(f16_bits_to_f32(
                                                           v[(j * nkv + kvh) * hd + d])));
            }
        }
    }
}

void reference_gate(const Opts& o, const QsaShapes& s, const std::vector<float>& attn,
                    const std::vector<float>& q_full, std::vector<float>& out) {
    const std::uint64_t hd = s.head_dim;
    out.assign(s.n_head * hd, 0.0f);
    for (std::uint64_t h = 0; h < s.n_head; ++h) {
        for (std::uint64_t d = 0; d < hd; ++d) {
            const std::size_t at = o.gate_second_half ? h * 2 * hd + hd + d : h * 2 * hd + 2 * d;
            const double g = static_cast<double>(q_full[at]);
            const double act = o.sigmoid_gate ? 1.0 / (1.0 + std::exp(-g)) : g / (1.0 + std::exp(-g));
            out[h * hd + d] = static_cast<float>(static_cast<double>(attn[h * hd + d]) * act);
        }
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

void test_matches_the_reference_and_the_weights_are_consistent() {
    const QsaShapes s = shapes();
    const std::vector<float> q = make_f32(kNh * kHd);
    const std::vector<std::uint16_t> k = make_f16(kIds * kNkv * kHd), v = make_f16(kIds * kNkv * kHd);
    std::vector<float> attn(kNh * kHd, 0.0f), w(kNh * kIds, 0.0f);
    std::string err;
    CHECK(qsa_attend(q.data(), k.data(), v.data(), kIds, s, attn.data(), w.data(), err));
    CHECK(err.empty());
    std::vector<float> ra, rw;
    reference(Opts{}, s, q, k, v, kIds, ra, rw);
    const double da = rel_l1(ra, attn), dw = rel_l1(rw, w);
    std::printf("与参考的相对 L1：attn %.3g、weights %.3g\n", da, dw);
    CHECK(da < 1e-6 && dw < 1e-6);
    // softmax 权重每个头各自和为 1。
    for (std::uint64_t h = 0; h < kNh; ++h) {
        double sum = 0.0;
        for (std::uint64_t j = 0; j < kIds; ++j) sum += w[h * kIds + j];
        CHECK(std::fabs(sum - 1.0) < 1e-5);
    }
    // 用权重重算 attn 应当得到同一结果（权重与注意力自洽）。
    for (std::uint64_t h = 0; h < kNh; ++h) {
        const std::uint64_t kvh = h / (kNh / kNkv);
        for (std::uint64_t d = 0; d < kHd; ++d) {
            double acc = 0.0;
            for (std::uint64_t j = 0; j < kIds; ++j) {
                acc += static_cast<double>(w[h * kIds + j]) *
                       static_cast<double>(f16_bits_to_f32(v[(j * kNkv + kvh) * kHd + d]));
            }
            CHECK(std::fabs(acc - static_cast<double>(attn[h * kHd + d])) < 1e-5);
        }
    }
}

void test_gate_matches_the_reference() {
    const QsaShapes s = shapes();
    const std::vector<float> attn = make_f32(kNh * kHd);
    const std::vector<float> q_full = make_f32(kNh * 2 * kHd);
    std::vector<float> out(kNh * kHd, 0.0f);
    std::string err;
    CHECK(qsa_gate_apply_f32(attn.data(), q_full.data(), s, out.data(), err));
    CHECK(err.empty());
    std::vector<float> want;
    reference_gate(Opts{}, s, attn, q_full, want);
    const double d = rel_l1(want, out);
    std::printf("门与参考的相对 L1：%.3g\n", d);
    CHECK(d < 1e-6);
}

void test_rival_readings_are_rejected() {
    const QsaShapes s = shapes();
    const std::vector<float> q = make_f32(kNh * kHd);
    const std::vector<std::uint16_t> k = make_f16(kIds * kNkv * kHd), v = make_f16(kIds * kNkv * kHd);
    std::vector<float> good_a, good_w;
    reference(Opts{}, s, q, k, v, kIds, good_a, good_w);
    const std::vector<float> attn = make_f32(kNh * kHd);
    const std::vector<float> q_full = make_f32(kNh * 2 * kHd);
    std::vector<float> good_g;
    reference_gate(Opts{}, s, attn, q_full, good_g);

    struct Case {
        const char* name;
        Opts o;
        bool gate;
    };
    std::vector<Case> cases;
    {
        Opts o;
        o.divide_head_map = false;
        cases.push_back({"q→kv 头用取模而不是整除", o, false});
    }
    {
        Opts o;
        o.scale_scores = false;
        cases.push_back({"分数不除 sqrt(head_dim)", o, false});
    }
    {
        Opts o;
        o.sigmoid_gate = false;
        cases.push_back({"门用 silu 而不是 sigmoid", o, true});
    }
    {
        Opts o;
        o.gate_second_half = false;
        cases.push_back({"门按元素交错取而不是每头块的后半段", o, true});
    }
    for (const Case& c : cases) {
        double worst = 0.0;
        if (!c.gate) {
            std::vector<float> a, w;
            reference(c.o, s, q, k, v, kIds, a, w);
            worst = std::max(rel_l1(good_a, a), rel_l1(good_w, w));
        } else {
            std::vector<float> g;
            reference_gate(c.o, s, attn, q_full, g);
            worst = rel_l1(good_g, g);
        }
        std::printf("错的读法「%s」的最大偏离 %.3g\n", c.name, worst);
        CHECK(worst > 1e-4);
    }
}

// 减最大值不是可选的等价变形：分数一大，不减它 exp 会溢出。这里把 q 放大到 1e3 量级，
// 使分数约 1e3（exp 溢出），要求输出仍然有限且与参考一致——这条正是「为什么要减最大值」的证明。
void test_large_scores_do_not_overflow() {
    const QsaShapes s = shapes();
    std::vector<float> q = make_f32(kNh * kHd);
    for (float& v : q) v *= 1.0e3f;
    const std::vector<std::uint16_t> k = make_f16(kIds * kNkv * kHd), v = make_f16(kIds * kNkv * kHd);
    std::vector<float> attn(kNh * kHd, 0.0f), w(kNh * kIds, 0.0f);
    std::string err;
    CHECK(qsa_attend(q.data(), k.data(), v.data(), kIds, s, attn.data(), w.data(), err));
    CHECK(err.empty());
    for (float x : attn) CHECK(std::isfinite(x));
    for (float x : w) CHECK(std::isfinite(x) && x >= 0.0f && x <= 1.0f);
    std::vector<float> ra, rw;
    reference(Opts{}, s, q, k, v, kIds, ra, rw);
    CHECK(rel_l1(ra, attn) < 1e-5);
    for (std::uint64_t h = 0; h < kNh; ++h) {
        double sum = 0.0;
        for (std::uint64_t j = 0; j < kIds; ++j) sum += w[h * kIds + j];
        CHECK(std::fabs(sum - 1.0) < 1e-4);  // 大分数下仍要是概率
    }
}

void test_empty_ids_and_failures() {
    const QsaShapes s = shapes();
    const std::vector<float> q = make_f32(kNh * kHd);
    std::vector<float> attn(kNh * kHd, 5.0f);  // 先塞非零，验证会被清零
    std::string err;
    CHECK(qsa_attend(q.data(), nullptr, nullptr, 0, s, attn.data(), nullptr, err));
    CHECK(err.empty());
    for (float v : attn) CHECK(v == 0.0f);

    err.clear();
    CHECK(!qsa_attend(q.data(), nullptr, nullptr, 1, s, attn.data(), nullptr, err));
    CHECK(!err.empty());  // 有 cell 却没有 scratch
    err.clear();
    CHECK(!qsa_attend(q.data(), nullptr, nullptr, 0, QsaShapes{2048, 4, kHd, kNh, 3}, attn.data(),
                      nullptr, err));
    CHECK(!err.empty());  // n_head 不是 n_head_kv 的整数倍
    err.clear();
    CHECK(!qsa_gate_apply_f32(nullptr, nullptr, s, nullptr, err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference_and_the_weights_are_consistent();
    test_gate_matches_the_reference();
    test_rival_readings_are_rejected();
    test_large_scores_do_not_overflow();
    test_empty_ids_and_failures();
    std::puts("qsa_attend: 只选中的 cell 上做 softmax GQA（整除的 q→kv 映射、除 sqrt(head_dim)）、"
              "权重与注意力自洽、门取每头块后半段并乘 sigmoid，四处错读法都被拒");
    return 0;
}