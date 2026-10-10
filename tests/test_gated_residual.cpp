// 门控残差（超连接）的回归。不引第三方框架。
//
// 做法照参考引擎自己的 parity 测试：**每一种错的读法都先按错的算一遍，要求它与对的读法有明显差别**，
// 然后才把实现与对的读法比。只做后一半的测试对两种读法都通过。六处陷阱（S-53）：
//   1. 逐流一条 RMS，而不是整栈一条；
//   2. 除 hc 在 silu **里面**；
//   3. lo 上取 silu、门上取 sigmoid（写反是常见笔误，两个函数都单调有界，量级还差不多）；
//   4. 按流求平均，不是求和（差一个 hc 倍，看着像尺度问题而不是结构问题）；
//   5. 激活按 bf16 舍入（参考实现的默认契约，不是因为权重是 bf16）；
//   6. gr_write 的 2·sigmoid 把门心定在 1 —— 零注入必须是普通残差相加（按性质断言，不按数值）。
#include "dense/gated_residual.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kN = 8;    // n_embd
constexpr std::uint64_t kHc = 3;   // 流数
constexpr std::uint64_t kLr = 5;   // 瓶颈秩
constexpr float kEps = 1e-6f;

// 确定性伪随机（不用 <random>，免得实现差异进到夹具里）。
struct Lcg {
    std::uint64_t s = 0x243F6A8885A308D3ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const std::uint32_t hi = static_cast<std::uint32_t>(s >> 32);
        return (static_cast<float>(hi) / 2147483648.0f) - 1.0f;  // [-1, 1)
    }
};

// 权重生成成 bf16 再读回 f32：与实现侧看到的完全一致，差别只留在算术上。
std::vector<std::uint16_t> rand_weights(std::uint64_t n, Lcg& g) {
    std::vector<std::uint16_t> w(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = f32_to_bf16_bits(g.next());
    return w;
}

float as_f32(std::uint16_t bits) { return bf16_bits_to_f32(bits); }

// 各条流的幅度不同（第 c 条约 c+1 倍）：整栈一条 RMS 的错读法因此可观测。
std::vector<float> make_R(Lcg& g) {
    std::vector<float> R(static_cast<std::size_t>(kN * kHc));
    for (std::uint64_t c = 0; c < kHc; ++c) {
        for (std::uint64_t d = 0; d < kN; ++d) {
            R[static_cast<std::size_t>(c * kN + d)] =
                g.next() * static_cast<float>(c + 1) + 0.5f * static_cast<float>(c);
        }
    }
    return R;
}

// ---- 参考实现（主机侧、double 累加），每一处陷阱都带一个开关 ----
struct Opts {
    bool per_stream_norm = true;    // false：整栈一条 RMS
    bool scale_inside_silu = true;  // false：silu(p)/hc
    bool sigmoid_on_gate = true;    // false：门上取 silu、lo 上取 sigmoid
    bool mean_over_streams = true;  // false：按流求和
    bool round_activation = true;   // false：不按 bf16 舍入
};

void reference(const Opts& o, const std::vector<float>& R, const std::vector<std::uint16_t>& wn,
               const std::vector<std::uint16_t>& wd, const std::vector<std::uint16_t>& wu,
               const std::vector<std::uint16_t>& wi, std::vector<float>& mixed,
               std::vector<float>& inject) {
    const std::size_t hc_dim = kN * kHc;
    std::vector<double> xn(hc_dim, 0.0);
    if (o.per_stream_norm) {
        for (std::uint64_t c = 0; c < kHc; ++c) {
            double ss = 0.0;
            for (std::uint64_t d = 0; d < kN; ++d) {
                const double v = R[static_cast<std::size_t>(c * kN + d)];
                ss += v * v;
            }
            const double rs = 1.0 / std::sqrt(ss / static_cast<double>(kN) + kEps);
            for (std::uint64_t d = 0; d < kN; ++d) {
                const std::size_t i = static_cast<std::size_t>(c * kN + d);
                xn[i] = static_cast<double>(R[i]) * rs * as_f32(wn[i]);
            }
        }
    } else {
        double ss = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) ss += static_cast<double>(R[i]) * R[i];
        const double rs = 1.0 / std::sqrt(ss / static_cast<double>(hc_dim) + kEps);
        for (std::size_t i = 0; i < hc_dim; ++i) xn[i] = static_cast<double>(R[i]) * rs * as_f32(wn[i]);
    }
    std::vector<double> act(hc_dim);
    for (std::size_t i = 0; i < hc_dim; ++i) {
        act[i] = o.round_activation ? bf16_bits_to_f32(f32_to_bf16_bits(static_cast<float>(xn[i])))
                                    : xn[i];
    }

    std::vector<double> lo(static_cast<std::size_t>(kLr), 0.0), lq(static_cast<std::size_t>(kLr), 0.0);
    for (std::uint64_t k = 0; k < kLr; ++k) {
        double a = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) a += act[i] * as_f32(wd[static_cast<std::size_t>(k) * hc_dim + i]);
        const double p = a;
        const double scaled = p / static_cast<double>(kHc);
        if (o.sigmoid_on_gate) {
            lo[static_cast<std::size_t>(k)] = o.scale_inside_silu
                                                  ? scaled / (1.0 + std::exp(-scaled))
                                                  : (p / (1.0 + std::exp(-p))) / static_cast<double>(kHc);
        } else {
            lo[static_cast<std::size_t>(k)] = o.scale_inside_silu
                                                  ? 1.0 / (1.0 + std::exp(-scaled))
                                                  : (1.0 / (1.0 + std::exp(-p))) / static_cast<double>(kHc);
        }
    }
    for (std::uint64_t k = 0; k < kLr; ++k) {
        lq[static_cast<std::size_t>(k)] =
            o.round_activation ? bf16_bits_to_f32(f32_to_bf16_bits(static_cast<float>(lo[static_cast<std::size_t>(k)])))
                               : lo[static_cast<std::size_t>(k)];
    }

    mixed.assign(static_cast<std::size_t>(kN), 0.0f);
    for (std::uint64_t d = 0; d < kN; ++d) {
        double m = 0.0;
        for (std::uint64_t c = 0; c < kHc; ++c) {
            const std::size_t i = static_cast<std::size_t>(c * kN + d);
            double a = 0.0;
            for (std::uint64_t k = 0; k < kLr; ++k) {
                a += lq[static_cast<std::size_t>(k)] * as_f32(wu[i * kLr + k]);
            }
            const double s = o.sigmoid_on_gate ? 1.0 / (1.0 + std::exp(-a)) : a / (1.0 + std::exp(-a));
            m += xn[i] * s;
        }
        mixed[static_cast<std::size_t>(d)] =
            static_cast<float>(o.mean_over_streams ? m / static_cast<double>(kHc) : m);
    }

    inject.assign(static_cast<std::size_t>(kHc), 0.0f);
    for (std::uint64_t c = 0; c < kHc; ++c) {
        double a = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) a += act[i] * as_f32(wi[static_cast<std::size_t>(c) * hc_dim + i]);
        inject[static_cast<std::size_t>(c)] = static_cast<float>(a);
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
    Lcg g{};
    std::vector<std::uint16_t> wn, wd, wu, wi;
    std::vector<float> R;
    Fixture()
        : wn(rand_weights(kN * kHc, g)),
          wd(rand_weights(kLr * kN * kHc, g)),
          wu(rand_weights(kN * kHc * kLr, g)),
          wi(rand_weights(kHc * kN * kHc, g)) {
        R = make_R(g);
    }
    GrShapes shapes() const { return GrShapes{kN, kHc, kLr}; }
    GrWeights weights() const {
        return GrWeights{wn.data(), wd.data(), wu.data(), wi.data()};
    }
};

void test_matches_the_reference() {
    const Fixture fx;
    std::vector<float> mixed(kN, 0.0f), inject(kHc, 0.0f);
    GrScratch sc;
    std::string err;
    CHECK(gr_read(fx.shapes(), fx.weights(), fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    CHECK(err.empty());

    std::vector<float> rm, ri;
    reference(Opts{}, fx.R, fx.wn, fx.wd, fx.wu, fx.wi, rm, ri);
    const double dm = rel_l1(rm, mixed), di = rel_l1(ri, inject);
    std::printf("与参考的相对 L1：mixed %.3g、inject %.3g\n", dm, di);
    CHECK(dm < 1e-5);
    CHECK(di < 1e-5);

    // 三条流的幅度不同、注入值也应各不相同（抓「三条流算成同一条」）。
    CHECK(std::fabs(inject[0] - inject[1]) > 1e-6f || std::fabs(inject[1] - inject[2]) > 1e-6f);
}

// 六处陷阱：每处都按错的读法算一遍，要求它离对的读法**有明显差别**；否则这条用例是空转。
void test_rival_readings_are_rejected() {
    const Fixture fx;
    std::vector<float> mixed(kN, 0.0f), inject(kHc, 0.0f);
    GrScratch sc;
    std::string err;
    CHECK(gr_read(fx.shapes(), fx.weights(), fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    std::vector<float> rm, ri;
    reference(Opts{}, fx.R, fx.wn, fx.wd, fx.wu, fx.wi, rm, ri);

    struct Case {
        const char* name;
        Opts o;
    };
    std::vector<Case> cases;
    {
        Opts o;
        o.per_stream_norm = false;
        cases.push_back({"整栈一条 RMS", o});
    }
    {
        Opts o;
        o.scale_inside_silu = false;
        cases.push_back({"把 /hc 挪到 silu 外面", o});
    }
    {
        Opts o;
        o.sigmoid_on_gate = false;
        cases.push_back({"lo 上 sigmoid、门上 silu", o});
    }
    {
        Opts o;
        o.mean_over_streams = false;
        cases.push_back({"按流求和而不是平均", o});
    }
    {
        Opts o;
        o.round_activation = false;
        cases.push_back({"激活不做 bf16 舍入", o});
    }
    for (const Case& c : cases) {
        std::vector<float> wm, wi2;
        reference(c.o, fx.R, fx.wn, fx.wd, fx.wu, fx.wi, wm, wi2);
        const double dm = rel_l1(rm, wm), di = rel_l1(ri, wi2);
        std::printf("错的读法「%s」相对差别：mixed %.3g、inject %.3g\n", c.name, dm, di);
        CHECK(dm > 1e-4 || di > 1e-4);
    }
}

void test_zero_injection_is_a_plain_residual_add() {
    const Fixture fx;
    const std::vector<float> inject(static_cast<std::size_t>(kHc), 0.0f);
    std::vector<float> block(kN);
    for (std::uint64_t d = 0; d < kN; ++d) block[static_cast<std::size_t>(d)] = 0.25f * static_cast<float>(d) - 0.75f;
    std::vector<float> out(static_cast<std::size_t>(kN * kHc), 0.0f);
    std::string err;
    CHECK(gr_write(fx.shapes(), fx.R.data(), block.data(), inject.data(), out.data(), err));
    // 2·sigmoid(0) 恰好是 1，故每条流都是 R + block，逐位相同。
    for (std::uint64_t c = 0; c < kHc; ++c) {
        for (std::uint64_t d = 0; d < kN; ++d) {
            const std::size_t i = static_cast<std::size_t>(c * kN + d);
            const float want = fx.R[i] + block[static_cast<std::size_t>(d)];
            CHECK(std::bit_cast<std::uint32_t>(out[i]) == std::bit_cast<std::uint32_t>(want));
        }
    }
}

void test_write_weight_is_per_stream_and_centred_on_one() {
    const Fixture fx;
    // 大正/大负注入 -> 门饱和到 2 / 0；零 -> 门恰好 1；2.0 是**中间值**，用来钉住注入在被 sigmoid
    // 之前要先除 hc（饱和区里除不除都一样，只有中间值才看得出来）。
    const std::vector<float> inject{40.0f, -40.0f, 2.0f};
    std::vector<float> block(kN, 1.5f);
    std::vector<float> out(static_cast<std::size_t>(kN * kHc), 0.0f);
    std::string err;
    CHECK(gr_write(fx.shapes(), fx.R.data(), block.data(), inject.data(), out.data(), err));
    const double mid = 2.0 * (1.0 / (1.0 + std::exp(-(2.0 / 3.0))));  // 2·sigmoid(inject/hc)
    const double rival = 2.0 * (1.0 / (1.0 + std::exp(-2.0)));        // 忘了除 hc
    CHECK(std::fabs(mid - rival) > 1e-2);
    for (std::uint64_t d = 0; d < kN; ++d) {
        const std::size_t i = static_cast<std::size_t>(d);
        CHECK(std::fabs(out[i] - (fx.R[i] + 3.0f)) < 1e-5f);                 // 门 ~ 2
        CHECK(std::fabs(out[i + kN] - fx.R[i + kN]) < 1e-5f);               // 门 ~ 0
        const float want = fx.R[i + 2 * kN] + 1.5f * static_cast<float>(mid);
        CHECK(std::fabs(out[i + 2 * kN] - want) < 1e-4f);
    }
}

void test_null_inject_leaves_the_buffer_untouched() {
    const Fixture fx;
    std::vector<float> mixed(kN, 0.0f), inject(kHc, 0.0f);
    for (std::uint64_t c = 0; c < kHc; ++c) inject[static_cast<std::size_t>(c)] = 7.0f;
    GrWeights w = fx.weights();
    w.w_inject = nullptr;  // 最后那个 mixer 没有回写门
    GrScratch sc;
    std::string err;
    CHECK(gr_read(fx.shapes(), w, fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    for (std::uint64_t c = 0; c < kHc; ++c) CHECK(inject[static_cast<std::size_t>(c)] == 7.0f);
}

void test_bad_inputs_fail_loudly() {
    const Fixture fx;
    std::vector<float> mixed(kN, 0.0f), inject(kHc, 0.0f), out(kN * kHc, 0.0f);
    GrScratch sc;
    GrWeights w = fx.weights();
    std::string err;

    CHECK(!gr_read(GrShapes{0, kHc, kLr}, w, fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    CHECK(!err.empty());
    err.clear();
    GrWeights no_down = w;
    no_down.w_down = nullptr;
    CHECK(!gr_read(fx.shapes(), no_down, fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!gr_read(fx.shapes(), w, nullptr, kEps, mixed.data(), inject.data(), sc, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!gr_write(GrShapes{kN, 0, kLr}, fx.R.data(), mixed.data(), inject.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!gr_write(fx.shapes(), fx.R.data(), nullptr, inject.data(), out.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference();
    test_rival_readings_are_rejected();
    test_zero_injection_is_a_plain_residual_add();
    test_write_weight_is_per_stream_and_centred_on_one();
    test_null_inject_leaves_the_buffer_untouched();
    test_bad_inputs_fail_loudly();
    std::puts("gated_residual: 超连接的两半——按流归一与门控平均、以及把块输出按流权重加回——"
              "与主机参考一致，且六处错读法都被拒");
    return 0;
}