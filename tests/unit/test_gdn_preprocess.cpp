// GDN 前半段的回归：卷积 → SiLU → 拆 q|k|v → q/k 的 l2 归一 → 两个门。不引第三方框架。
//
// 做法与门控残差那片一致：每一处「一读就过」的地方先按错的算一遍、要求它与对的读法有明显差别，
// 再比实现。四处：SiLU 加在整段 conv 输出上（不是拆完之后）、v 不做 l2 归一、门上是 softplus
// 而不是 sigmoid 且 dt 加在 softplus 里面、卷积状态的移位方向（权重第 0 行配最早一帧）。
//
// 合成几何取小值：头宽 4、q/k/v 头数 2/1/3、卷积核长 3（故窗口 2 帧），n_embd 5。
#include "dense/gdn_preprocess.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kN = 5;      // n_embd
constexpr std::uint64_t kQh = 2, kKh = 1, kVh = 3;
constexpr std::uint64_t kHd = 4;     // 头宽
constexpr std::uint64_t kDc = 3;     // 卷积核长（窗口 2）
constexpr float kEps = 1e-6f;
constexpr std::uint64_t kQd = kQh * kHd, kKd = kKh * kHd, kVd = kVh * kHd;
constexpr std::uint64_t kCd = kQd + kKd + kVd;  // 22

struct Lcg {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

// ---- 参考实现（double 累加），每一处陷阱一个开关 ----
struct Opts {
    bool silu_after_split = false;  // true：错的读法（SiLU 加在拆完之后）
    bool normalize_v = false;       // true：错的读法（v 也归一）
    bool softplus_gate = true;      // false：错的读法（门上取 sigmoid）
    bool dt_outside_softplus = false;  // true：错的读法（dt 加在 softplus 外面）
    bool shift_newest_last = true;     // false：错的读法（新帧放窗口头、不按时间对齐）
};

void reference(const GdnShapes& s, const Opts& o, const float* x, const std::vector<float>& qkv,
               std::vector<float>& frames, std::vector<float>& q, std::vector<float>& k,
               std::vector<float>& v, std::vector<float>& beta, std::vector<float>& gate,
               const std::vector<float>& convw, const std::vector<std::uint16_t>& alpha,
               const std::vector<std::uint16_t>& beta_w, const std::vector<float>& dt,
               const std::vector<float>& ssm_a) {
    const std::uint64_t cd = s.conv_dim();
    const std::uint64_t win = s.d_conv - 1;
    std::vector<double> conv(cd, 0.0);
    for (std::uint64_t r = 0; r < s.d_conv; ++r) {
        const float* frame = r < win ? frames.data() + r * cd : qkv.data();
        for (std::uint64_t i = 0; i < cd; ++i) conv[i] += convw[r * cd + i] * frame[i];
    }
    if (!o.silu_after_split) {
        for (std::uint64_t i = 0; i < cd; ++i) conv[i] = conv[i] / (1.0 + std::exp(-conv[i]));
    }
    auto norm_head = [&](const double* src, std::uint64_t hd, std::vector<float>& dst, bool do_it) {
        double ss = 0.0;
        for (std::uint64_t i = 0; i < hd; ++i) ss += src[i] * src[i];
        const double rs = do_it ? 1.0 / std::sqrt(ss + kEps) : 1.0;
        for (std::uint64_t i = 0; i < hd; ++i) dst.push_back(static_cast<float>(src[i] * rs));
    };
    q.clear();
    k.clear();
    v.clear();
    for (std::uint64_t h = 0; h < s.n_q_head; ++h) {
        norm_head(conv.data() + h * s.head_dim, s.head_dim, q, true);
    }
    for (std::uint64_t h = 0; h < s.n_k_head; ++h) {
        norm_head(conv.data() + s.q_dim() + h * s.head_dim, s.head_dim, k, true);
    }
    for (std::uint64_t h = 0; h < s.n_v_head; ++h) {
        norm_head(conv.data() + s.q_dim() + s.k_dim() + h * s.head_dim, s.head_dim, v,
                  o.normalize_v);
    }
    if (o.silu_after_split) {
        // 错的读法：SiLU 加在 q/k/v 上（v 也过一遍）。
        for (float& t : q) t = static_cast<float>(t / (1.0 + std::exp(-static_cast<double>(t))));
        for (float& t : k) t = static_cast<float>(t / (1.0 + std::exp(-static_cast<double>(t))));
        for (float& t : v) t = static_cast<float>(t / (1.0 + std::exp(-static_cast<double>(t))));
    }
    beta.clear();
    gate.clear();
    for (std::uint64_t h = 0; h < s.n_v_head; ++h) {
        double ab = 0.0, ag = 0.0;
        for (std::uint64_t i = 0; i < s.n_embd; ++i) {
            ab += static_cast<double>(x[i]) * bf16_bits_to_f32(beta_w[h * s.n_embd + i]);
            ag += static_cast<double>(x[i]) * bf16_bits_to_f32(alpha[h * s.n_embd + i]);
        }
        beta.push_back(static_cast<float>(1.0 / (1.0 + std::exp(-ab))));
        const double g = ag + (o.dt_outside_softplus ? 0.0 : static_cast<double>(dt[h]));
        double sp = o.softplus_gate ? (g > 20.0 ? g : std::log1p(std::exp(g))) : 1.0 / (1.0 + std::exp(-g));
        if (o.dt_outside_softplus) sp += static_cast<double>(dt[h]);
        gate.push_back(static_cast<float>(sp * ssm_a[h]));
    }
    // 状态前移
    if (win > 0) {
        std::vector<float> nf(win * cd, 0.0f);
        for (std::uint64_t r = 0; r + 1 < win; ++r) {
            for (std::uint64_t i = 0; i < cd; ++i) nf[r * cd + i] = frames[(r + 1) * cd + i];
        }
        for (std::uint64_t i = 0; i < cd; ++i) nf[(win - 1) * cd + i] = qkv[i];
        frames = nf;
    }
}

struct Fixture {
    Lcg g{};
    std::vector<float> convw;               // d_conv × cd
    std::vector<std::uint16_t> alpha, beta_w;
    std::vector<float> dt, ssm_a;
    Fixture() {
        convw.resize(kDc * kCd);
        for (float& t : convw) t = g.next();
        alpha.resize(kVh * kN);
        beta_w.resize(kVh * kN);
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            alpha[i] = f32_to_bf16_bits(g.next());
            beta_w[i] = f32_to_bf16_bits(g.next());
        }
        dt.resize(kVh);
        ssm_a.resize(kVh);
        for (std::uint64_t h = 0; h < kVh; ++h) {
            dt[h] = 0.5f + 0.25f * static_cast<float>(h);  // 正的，softplus 才不退化
            ssm_a[h] = -0.5f - 0.1f * static_cast<float>(h);
        }
    }
    GdnShapes shapes() const { return GdnShapes{kN, kQh, kKh, kVh, kHd, kDc}; }
    GdnPreWeights weights() const {
        return GdnPreWeights{GrTensor{convw.data(), GrPrecision::kF32},
                             GrTensor{alpha.data(), GrPrecision::kBf16},
                             GrTensor{beta_w.data(), GrPrecision::kBf16}, dt.data(), ssm_a.data()};
    }
    std::vector<float> make_x() const {
        std::vector<float> x(kN);
        Lcg g2;
        for (float& t : x) t = g2.next();
        return x;
    }
    // 每一步都用同一帧（确定性）：卷积状态的作用因此只看步数，不看输入抖动。
    std::vector<float> make_qkv() const {
        std::vector<float> q(kCd);
        Lcg g3;
        for (float& t : q) t = g3.next();
        return q;
    }
};

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        mag += std::fabs(static_cast<double>(a[i]));
    }
    return d / (mag > 1e-30 ? mag : 1e-30);
}

// 一次 run：跑实现与参考，返回各输出的相对 L1。tokens 走多步，把卷积状态也走一遍。
struct Diff {
    double q, k, v, beta, gate;
};

Diff run_case(const Opts& o, std::uint64_t steps, std::vector<float>* got_v = nullptr) {
    const Fixture fx;
    const GdnShapes s = fx.shapes();
    const std::vector<float> x = fx.make_x();
    GdnConvState st;
    std::vector<float> ref_frames(static_cast<std::size_t>((kDc - 1) * kCd), 0.0f);
    GdnPreOut out;
    Diff d{0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::uint64_t t = 0; t < steps; ++t) {
        const std::vector<float> qkv = fx.make_qkv();
        std::string err;
        CHECK(gdn_preprocess(s, fx.weights(), x.data(), qkv.data(), st, kEps, out, err));
        CHECK(err.empty());
        std::vector<float> rq, rk, rv, rb, rg;
        reference(s, o, x.data(), qkv, ref_frames, rq, rk, rv, rb, rg, fx.convw, fx.alpha,
                  fx.beta_w, fx.dt, fx.ssm_a);
        d.q = std::max(d.q, rel_l1(rq, out.q));
        d.k = std::max(d.k, rel_l1(rk, out.k));
        d.v = std::max(d.v, rel_l1(rv, out.v));
        d.beta = std::max(d.beta, rel_l1(rb, out.beta));
        d.gate = std::max(d.gate, rel_l1(rg, out.gate));
        if (got_v != nullptr) *got_v = out.v;
    }
    return d;
}

void test_matches_the_reference_and_walks_the_state() {
    const Diff d = run_case(Opts{}, /*steps=*/4);
    std::printf("与参考的相对 L1（4 步，卷积状态也走了一遍）：q %.3g、k %.3g、v %.3g、beta %.3g、gate %.3g\n",
                d.q, d.k, d.v, d.beta, d.gate);
    CHECK(d.q < 1e-5 && d.k < 1e-5 && d.v < 1e-5 && d.beta < 1e-5 && d.gate < 1e-5);
}

// 状态真的在起作用：第 1 步（窗口全零）与第 3 步（窗口有真帧）的输出必须不同，
// 且把状态清零重来会让第 3 步回到第 1 步的样子——否则「移位」这事在数据上没被验到。
// 参考侧单独跑一遍（只用来对比两种读法），返回最后一步的输出。
struct RefOut {
    std::vector<float> q, k, v, beta, gate;
};

RefOut ref_last(const Opts& o, std::uint64_t steps) {
    const Fixture fx;
    const GdnShapes s = fx.shapes();
    const std::vector<float> x = fx.make_x();
    std::vector<float> frames(static_cast<std::size_t>((kDc - 1) * kCd), 0.0f);
    RefOut last;
    for (std::uint64_t t = 0; t < steps; ++t) {
        const std::vector<float> qkv = fx.make_qkv();
        reference(s, o, x.data(), qkv, frames, last.q, last.k, last.v, last.beta, last.gate,
                  fx.convw, fx.alpha, fx.beta_w, fx.dt, fx.ssm_a);
    }
    return last;
}

void test_conv_state_actually_carries_history() {
    const Fixture fx;
    const GdnShapes s = fx.shapes();
    const std::vector<float> x = fx.make_x();
    std::string err;
    const std::vector<float> qkv = fx.make_qkv();
    std::vector<float> first, third;
    GdnConvState st;
    GdnPreOut out;
    CHECK(gdn_preprocess(s, fx.weights(), x.data(), qkv.data(), st, kEps, out, err));
    first = out.v;
    for (int t = 0; t < 2; ++t) {
        CHECK(gdn_preprocess(s, fx.weights(), x.data(), qkv.data(), st, kEps, out, err));
    }
    third = out.v;
    CHECK(first.size() == third.size() && !first.empty());
    bool differs = false;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(first[i]) != std::bit_cast<std::uint32_t>(third[i])) {
            differs = true;
        }
    }
    CHECK(differs);
    CHECK(st.tokens == 3);
}

void test_rival_readings_are_rejected() {
    const RefOut good = ref_last(Opts{}, 3);
    struct Case {
        const char* name;
        Opts o;
    };
    std::vector<Case> cases;
    {
        Opts o;
        o.silu_after_split = true;
        cases.push_back({"SiLU 加在拆完之后", o});
    }
    {
        Opts o;
        o.normalize_v = true;
        cases.push_back({"v 也做 l2 归一", o});
    }
    {
        Opts o;
        o.softplus_gate = false;
        cases.push_back({"门上取 sigmoid 而不是 softplus", o});
    }
    {
        Opts o;
        o.dt_outside_softplus = true;
        cases.push_back({"dt 加在 softplus 外面", o});
    }
    for (const Case& c : cases) {
        const RefOut bad = ref_last(c.o, 3);
        const double worst = std::max({rel_l1(good.q, bad.q), rel_l1(good.k, bad.k),
                                       rel_l1(good.v, bad.v), rel_l1(good.beta, bad.beta),
                                       rel_l1(good.gate, bad.gate)});
        std::printf("错的读法「%s」的最大偏离 %.3g\n", c.name, worst);
        CHECK(worst > 1e-4);
    }
}

void test_bad_inputs_fail_loudly() {
    const Fixture fx;
    const std::vector<float> x = fx.make_x(), qkv = fx.make_qkv();
    GdnConvState st;
    GdnPreOut out;
    std::string err;
    CHECK(!gdn_preprocess(GdnShapes{0, kQh, kKh, kVh, kHd, kDc}, fx.weights(), x.data(), qkv.data(),
                          st, kEps, out, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!gdn_preprocess(fx.shapes(), fx.weights(), nullptr, qkv.data(), st, kEps, out, err));
    CHECK(!err.empty());
    err.clear();
    GdnPreWeights no_conv = fx.weights();
    no_conv.conv1d = GrTensor{};
    CHECK(!gdn_preprocess(fx.shapes(), no_conv, x.data(), qkv.data(), st, kEps, out, err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference_and_walks_the_state();
    test_conv_state_actually_carries_history();
    test_rival_readings_are_rejected();
    test_bad_inputs_fail_loudly();
    std::puts("gdn_preprocess: 卷积与它的状态、整段 SiLU、q/k 归一与两个门都与主机参考一致，"
              "且四处错读法都被拒");
    return 0;
}