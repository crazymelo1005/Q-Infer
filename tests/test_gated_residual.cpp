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

#include "artifact/gguf_table.hpp"
#include "check.hpp"
#include "kernels/bf16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

std::vector<float> to_f32(const std::vector<std::uint16_t>& v) {
    std::vector<float> r(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) r[i] = bf16_bits_to_f32(v[i]);
    return r;
}

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

void reference(const GrShapes& s, const Opts& o, const std::vector<float>& R,
               const std::vector<float>& wn, const std::vector<float>& wd,
               const std::vector<float>& wu, const std::vector<float>& wi,
               std::vector<float>& mixed, std::vector<float>& inject) {
    const std::uint64_t N = s.n_embd, H = s.hc, L = s.hc_lr;
    const std::size_t hc_dim = static_cast<std::size_t>(N * H);
    std::vector<double> xn(hc_dim, 0.0);
    if (o.per_stream_norm) {
        for (std::uint64_t c = 0; c < H; ++c) {
            double ss = 0.0;
            for (std::uint64_t d = 0; d < N; ++d) {
                const double v = R[static_cast<std::size_t>(c * N + d)];
                ss += v * v;
            }
            const double rs = 1.0 / std::sqrt(ss / static_cast<double>(N) + kEps);
            for (std::uint64_t d = 0; d < N; ++d) {
                const std::size_t i = static_cast<std::size_t>(c * N + d);
                xn[i] = static_cast<double>(R[i]) * rs * wn[i];
            }
        }
    } else {
        double ss = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) ss += static_cast<double>(R[i]) * R[i];
        const double rs = 1.0 / std::sqrt(ss / static_cast<double>(hc_dim) + kEps);
        for (std::size_t i = 0; i < hc_dim; ++i) xn[i] = static_cast<double>(R[i]) * rs * wn[i];
    }
    std::vector<double> act(hc_dim);
    for (std::size_t i = 0; i < hc_dim; ++i) {
        act[i] = o.round_activation ? bf16_bits_to_f32(f32_to_bf16_bits(static_cast<float>(xn[i])))
                                    : xn[i];
    }

    std::vector<double> lo(static_cast<std::size_t>(L), 0.0), lq(static_cast<std::size_t>(L), 0.0);
    for (std::uint64_t k = 0; k < L; ++k) {
        double a = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) a += act[i] * wd[static_cast<std::size_t>(k) * hc_dim + i];
        const double p = a;
        const double scaled = p / static_cast<double>(H);
        if (o.sigmoid_on_gate) {
            lo[static_cast<std::size_t>(k)] = o.scale_inside_silu
                                                  ? scaled / (1.0 + std::exp(-scaled))
                                                  : (p / (1.0 + std::exp(-p))) / static_cast<double>(H);
        } else {
            lo[static_cast<std::size_t>(k)] = o.scale_inside_silu
                                                  ? 1.0 / (1.0 + std::exp(-scaled))
                                                  : (1.0 / (1.0 + std::exp(-p))) / static_cast<double>(H);
        }
    }
    for (std::uint64_t k = 0; k < L; ++k) {
        lq[static_cast<std::size_t>(k)] =
            o.round_activation ? bf16_bits_to_f32(f32_to_bf16_bits(static_cast<float>(lo[static_cast<std::size_t>(k)])))
                               : lo[static_cast<std::size_t>(k)];
    }

    mixed.assign(static_cast<std::size_t>(N), 0.0f);
    for (std::uint64_t d = 0; d < N; ++d) {
        double m = 0.0;
        for (std::uint64_t c = 0; c < H; ++c) {
            const std::size_t i = static_cast<std::size_t>(c * N + d);
            double a = 0.0;
            for (std::uint64_t k = 0; k < L; ++k) {
                a += lq[static_cast<std::size_t>(k)] * wu[i * static_cast<std::size_t>(L) + k];
            }
            const double s = o.sigmoid_on_gate ? 1.0 / (1.0 + std::exp(-a)) : a / (1.0 + std::exp(-a));
            m += xn[i] * s;
        }
        mixed[static_cast<std::size_t>(d)] =
            static_cast<float>(o.mean_over_streams ? m / static_cast<double>(H) : m);
    }

    inject.assign(static_cast<std::size_t>(H), 0.0f);
    if (wi.empty()) return;  // w_inject 为 null：inject 不动
    for (std::uint64_t c = 0; c < H; ++c) {
        double a = 0.0;
        for (std::size_t i = 0; i < hc_dim; ++i) a += act[i] * wi[static_cast<std::size_t>(c) * hc_dim + i];
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
    std::vector<std::uint16_t> wn, wd, wu, wi;  // bf16 位型（合成夹具统一用 bf16）
    std::vector<float> wn_f, wd_f, wu_f, wi_f;  // 同一个值的 f32 像，供主机参考用
    std::vector<float> R;
    Fixture()
        : wn(rand_weights(kN * kHc, g)),
          wd(rand_weights(kLr * kN * kHc, g)),
          wu(rand_weights(kN * kHc * kLr, g)),
          wi(rand_weights(kHc * kN * kHc, g)) {
        R = make_R(g);
        wn_f = to_f32(wn);
        wd_f = to_f32(wd);
        wu_f = to_f32(wu);
        wi_f = to_f32(wi);
    }
    GrShapes shapes() const { return GrShapes{kN, kHc, kLr}; }
    GrWeights weights() const {
        return GrWeights{GrTensor{wn.data()}, GrTensor{wd.data()}, GrTensor{wu.data()},
                         GrTensor{wi.data()}};
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
    reference(fx.shapes(), Opts{}, fx.R, fx.wn_f, fx.wd_f, fx.wu_f, fx.wi_f, rm, ri);
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
    reference(fx.shapes(), Opts{}, fx.R, fx.wn_f, fx.wd_f, fx.wu_f, fx.wi_f, rm, ri);

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
        reference(fx.shapes(), c.o, fx.R, fx.wn_f, fx.wd_f, fx.wu_f, fx.wi_f, wm, wi2);
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
    w.w_inject = GrTensor{};  // 最后那个 mixer 没有回写门
    GrScratch sc;
    std::string err;
    CHECK(gr_read(fx.shapes(), w, fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    for (std::uint64_t c = 0; c < kHc; ++c) CHECK(inject[static_cast<std::size_t>(c)] == 7.0f);
}

// 精度是逐张力声明的，不是全局的：真模型上 norm 是 F32、三个投影是 BF16。合成夹具全是 bf16，
// 故这里单造一个「norm 是 F32 且取值带 bf16 装不下的低位」的用例，并顺手把「无视声明、一律按 bf16
// 读」这条错读法算出来要求它明显偏离——否则这条路径在 CI 里没人看着。
void test_f32_tensor_is_read_by_its_declared_precision() {
    const Fixture fx;
    std::vector<float> wn_f(static_cast<std::size_t>(kN * kHc));
    Lcg g2;
    for (std::size_t i = 0; i < wn_f.size(); ++i) {
        // 1 + 小扰动：像 gamma，且扰动带 bf16 装不下的位（bf16 只有 8 位尾数）。
        wn_f[i] = 1.0f + 0.0009765625f * static_cast<float>(i) + g2.next() * 1e-4f;
    }
    std::size_t lossy = 0;
    for (const float v : wn_f) {
        if (bf16_bits_to_f32(f32_to_bf16_bits(v)) != v) ++lossy;
    }
    CHECK(lossy > wn_f.size() / 2);  // 夹具本身要真的装不下，否则这条用例是空转

    std::vector<float> mixed(kN, 0.0f), inject(kHc, 0.0f);
    GrScratch sc;
    std::string err;
    const GrWeights w{GrTensor{wn_f.data(), GrPrecision::kF32}, GrTensor{fx.wd.data()},
                      GrTensor{fx.wu.data()}, GrTensor{fx.wi.data()}};
    CHECK(gr_read(fx.shapes(), w, fx.R.data(), kEps, mixed.data(), inject.data(), sc, err));
    std::vector<float> rm, ri;
    reference(fx.shapes(), Opts{}, fx.R, wn_f, fx.wd_f, fx.wu_f, fx.wi_f, rm, ri);
    const double dm = rel_l1(rm, mixed);
    std::printf("F32 norm 按声明读：与参考的相对 L1 %.3g\n", dm);
    CHECK(dm < 1e-5);

    // 错的读法：同一块缓冲当成 bf16 位型读。
    const GrWeights wrong{GrTensor{wn_f.data(), GrPrecision::kBf16}, GrTensor{fx.wd.data()},
                          GrTensor{fx.wu.data()}, GrTensor{fx.wi.data()}};
    std::vector<float> m2(kN, 0.0f), i2(kHc, 0.0f);
    GrScratch sc2;
    CHECK(gr_read(fx.shapes(), wrong, fx.R.data(), kEps, m2.data(), i2.data(), sc2, err));
    const double dw = rel_l1(rm, m2);
    std::printf("错读法「无视声明一律按 bf16 读」的相对差别：%.3g\n", dw);
    CHECK(dw > 1e-4);
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
    no_down.w_down = GrTensor{};
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

// ---- 真模型手工入口 ----
//
// 用真权重跑一遍，并顺手把几何从**数据**里推出来（而不是照抄引擎契约头里那句「the real model」）：
//   w_norm 的元素数 = hc·n_embd；w_inject 的元素数 = hc·hc·n_embd，两者相除即 hc；
//   down 的元素数 ÷ (hc·n_embd) 即 hc_lr。四个张力的类型应当是 BF16（码 30）。
// 布局按引擎契约头（= llama.cpp 的清单布局）核：down 的行长是 hc·n_embd、up 的行长是 hc_lr、
// inject 的行长是 hc·n_embd，故 ggml 的 ne 序分别是 {hc·n_embd, hc_lr}、{hc_lr, hc·n_embd}、
// {hc·n_embd, hc}。布局不对就明确失败——按错的布局读会把权重读成另一组，且仍然给出可信的数。
int manual_model_entry(const char* model, int layer, const char* grp, std::uint64_t hc_hint) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    const std::string p = (std::string(grp) == "output_hc")
                              ? "output_hc_"
                              : "blk." + std::to_string(layer) + "." + grp + "_";
    const artifact::GgufTensorInfo* tn = g.find(p + "norm.weight");
    const artifact::GgufTensorInfo* td = g.find(p + "down.weight");
    const artifact::GgufTensorInfo* tu = g.find(p + "up.weight");
    const artifact::GgufTensorInfo* ti = g.find(p + "inject.weight");
    const bool has_inject = ti != nullptr;
    if (tn == nullptr || td == nullptr || tu == nullptr) {
        std::printf("层 %d 缺 %s 组的张力（%snorm / down / up）\n", layer, grp, p.c_str());
        return 1;
    }
    if (!has_inject && std::string(grp) != "output_hc") {
        std::printf("层 %d 的 %s 组没有 inject 张力\n", layer, grp);
        return 1;
    }
    const std::uint64_t norm_elems = tn->elements();
    std::uint64_t hc = 0, n_embd = 0;
    if (has_inject) {
        const std::uint64_t inj_elems = ti->elements();
        if (norm_elems == 0 || inj_elems % norm_elems != 0) {
            std::printf("元素数推不出 hc：norm %llu、inject %llu\n",
                        static_cast<unsigned long long>(norm_elems),
                        static_cast<unsigned long long>(inj_elems));
            return 1;
        }
        hc = inj_elems / norm_elems;
        n_embd = hc == 0 ? 0 : norm_elems / hc;
    } else {
        // 最后那个 mixer 没有 inject 张力，除不出 hc；而 hc 在模型里是元数据（hyper-connection
        // count），本读取器目前只枚举张力、不暴露元数据，故这一组的 hc 只能由调用方给。
        hc = hc_hint;
        n_embd = hc == 0 ? 0 : norm_elems / hc;
        std::printf("（output_hc 没有 inject，hc 由调用方给：%llu；本读取器不暴露元数据）\n",
                    static_cast<unsigned long long>(hc));
    }
    if (hc == 0 || n_embd == 0) {
        std::printf("元素数推不出 n_embd / hc\n");
        return 1;
    }
    const std::uint64_t hc_dim = norm_elems;
    if (td->elements() == 0 || td->elements() % hc_dim != 0) {
        std::printf("down 的元素数 %llu 不是 hc·n_embd %llu 的整数倍\n",
                    static_cast<unsigned long long>(td->elements()),
                    static_cast<unsigned long long>(hc_dim));
        return 1;
    }
    const std::uint64_t hc_lr = td->elements() / hc_dim;
    GrShapes s{n_embd, hc, hc_lr};
    std::printf("层 %d %s 组：n_embd %llu、hc %llu、hc_lr %llu；类型 norm/down/up = %u/%u/%u%s\n",
                layer, grp, static_cast<unsigned long long>(n_embd),
                static_cast<unsigned long long>(hc), static_cast<unsigned long long>(hc_lr),
                tn->type, td->type, tu->type,
                has_inject ? ("/inject " + std::to_string(ti->type)).c_str() : "（无 inject）");

    // 布局核对（ggml 的 ne 序 = 文件里的 dims 序，ne0 最快）。norm 允许一维 hc·n_embd，也允许
    // 二维 [n_embd, hc]——两者展平后是同一个序，故都算对。
    const bool norm_ok =
        (tn->dims.size() == 1 && tn->dims[0] == hc_dim) ||
        (tn->dims.size() == 2 && tn->dims[0] == n_embd && tn->dims[1] == hc);
    const bool proj_ok = tu->dims.size() == 2 && td->dims.size() == 2 && td->dims[0] == hc_dim &&
                         td->dims[1] == hc_lr && tu->dims[0] == hc_lr && tu->dims[1] == hc_dim;
    const bool inj_ok = !has_inject || (ti->dims.size() == 2 && ti->dims[0] == hc_dim && ti->dims[1] == hc);
    if (!norm_ok || !proj_ok || !inj_ok) {
        std::printf("布局与清单不符：norm %zu 维、down %zu 维、up %zu 维\n", tn->dims.size(),
                    td->dims.size(), tu->dims.size());
        return 1;
    }

    // 按每个张力自己声明的类型读：这份 GGUF 里 norm 是 F32（码 0）、三个投影是 BF16（码 30）。
    // 别照着上游那句「这些张力都是 BF16」写死——那说的是引擎自带的原生 pack，不是这份 GGUF。
    // 读成 f32 一律精确，故两种精度共用同一条路径；顺带记下「若硬截断到 bf16 会丢多少位」，
    // 这个数就是「norm 的 F32 必须被尊重」的证据（实测 40960 个里有 7520 个丢位）。
    const std::uint32_t kF32 = 0, kBF16 = 30;
    std::uint64_t lossy = 0;
    // 两种精度各留一份：实现侧按声明精度读原字节（bf16 必须是 uint16 位型，不能给它转好的 float，
    // 那会被当成位型读进去）；参考侧用 f32 像，两种精度都是精确转换。
    auto load = [&](const artifact::GgufTensorInfo* t, std::vector<float>& f,
                    std::vector<std::uint16_t>& b, GrPrecision& prec) {
        f.resize(static_cast<std::size_t>(t->elements()));
        if (t->type == kBF16) {
            prec = GrPrecision::kBf16;
            b.resize(f.size());
            g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(b.data()), b.size() * 2);
            for (std::size_t i = 0; i < f.size(); ++i) f[i] = bf16_bits_to_f32(b[i]);
            return true;
        }
        if (t->type == kF32) {
            prec = GrPrecision::kF32;
            g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(f.data()), f.size() * 4);
            for (const float v : f) {
                if (bf16_bits_to_f32(f32_to_bf16_bits(v)) != v) ++lossy;
            }
            return true;
        }
        std::printf("张力 %s 的类型码 %u 既不是 F32 也不是 BF16\n", t->name.c_str(), t->type);
        return false;
    };
    std::vector<float> wn, wd, wu, wi;
    std::vector<std::uint16_t> wnb, wdb, wub, wib;
    GrPrecision pn = GrPrecision::kBf16, pd = pn, pu = pn, pi = pn;
    if (!load(tn, wn, wnb, pn) || !load(td, wd, wdb, pd) || !load(tu, wu, wub, pu)) return 1;
    if (has_inject && !load(ti, wi, wib, pi)) return 1;
    std::printf("精度：norm %s、down %s、up %s、inject %s；F32 的那些若硬截断到 bf16 会丢 %llu 个值\n",
                pn == GrPrecision::kF32 ? "F32" : "BF16", pd == GrPrecision::kF32 ? "F32" : "BF16",
                pu == GrPrecision::kF32 ? "F32" : "BF16", pi == GrPrecision::kF32 ? "F32" : "BF16",
                static_cast<unsigned long long>(lossy));

    // 一组观测量（不是断言）：w_norm 在包里存的是 1+w，故应聚在 1 附近。
    float lo_v = wn[0], hi_v = lo_v;
    double sum = 0.0;
    for (const float v : wn) {
        if (v < lo_v) lo_v = v;
        if (v > hi_v) hi_v = v;
        sum += v;
    }
    std::printf("w_norm：min %.4f、max %.4f、均值 %.4f\n", static_cast<double>(lo_v),
                static_cast<double>(hi_v), sum / static_cast<double>(wn.size()));

    // 真字节上跑两半，与同一个主机双精度参考比（几何用从数据推出的那一份）。
    std::vector<float> R(static_cast<std::size_t>(hc_dim), 0.0f);
    Lcg rng;
    for (std::size_t i = 0; i < R.size(); ++i) R[i] = rng.next();
    std::vector<float> mixed(static_cast<std::size_t>(n_embd), 0.0f);
    // 没有 inject 的那一组（最后那个 mixer）用哨兵值起头，好断言实现确实没碰它。
    std::vector<float> inject(static_cast<std::size_t>(hc), 7.0f);
    GrScratch sc;
    const GrWeights w{
        pn == GrPrecision::kF32 ? GrTensor{wn.data(), pn} : GrTensor{wnb.data(), pn},
        pd == GrPrecision::kF32 ? GrTensor{wd.data(), pd} : GrTensor{wdb.data(), pd},
        pu == GrPrecision::kF32 ? GrTensor{wu.data(), pu} : GrTensor{wub.data(), pu},
        has_inject ? (pi == GrPrecision::kF32 ? GrTensor{wi.data(), pi} : GrTensor{wib.data(), pi})
                   : GrTensor{}};
    if (!gr_read(s, w, R.data(), kEps, mixed.data(), inject.data(), sc, err)) {
        std::printf("gr_read 失败：%s\n", err.c_str());
        return 1;
    }
    std::vector<float> rm, ri;
    bool inject_touched = true;
    if (!has_inject) {
        inject_touched = false;
        for (std::uint64_t c = 0; c < hc; ++c) {
            if (inject[static_cast<std::size_t>(c)] != 7.0f) inject_touched = true;
        }
        std::printf("没有 inject 张力时实现是否碰了 inject：%s\n", inject_touched ? "碰了" : "没碰");
    }
    reference(s, Opts{}, R, wn, wd, wu, wi, rm, ri);
    const double dm = rel_l1(rm, mixed), di = rel_l1(ri, inject);
    if (has_inject) {
        std::printf("与主机参考的相对 L1：mixed %.3g、inject %.3g\n", dm, di);
    } else {
        // 没有 inject 时参考不填、实现也不该碰，故这一项不比对（比了会比出哨兵值）。
        std::printf("与主机参考的相对 L1：mixed %.3g（无 inject，不比对）\n", dm);
    }

    std::vector<float> block(static_cast<std::size_t>(n_embd), 1.0f);
    std::vector<float> out(static_cast<std::size_t>(hc_dim), 0.0f);
    if (!gr_write(s, R.data(), block.data(), inject.data(), out.data(), err)) {
        std::printf("gr_write 失败：%s\n", err.c_str());
        return 1;
    }
    double cancels = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) cancels += std::fabs(static_cast<double>(out[i]));
    std::printf("gr_write 合计 |R_out| = %.6g（有回写门，故与 |R| 不同）\n", cancels);

    const bool ok = dm < 1e-5 && (has_inject ? di < 1e-5 : !inject_touched);
    std::printf("层 %d %s 组：%s\n", layer, grp, ok ? "与主机参考一致" : "不一致");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 0;
        std::uint64_t hc_hint = 4;
        std::string grp = "hc_attn";
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--group") grp = argv[i + 1];
            if (std::string(argv[i]) == "--hc") hc_hint = static_cast<std::uint64_t>(std::atoi(argv[i + 1]));
        }
        return manual_model_entry(argv[2], layer, grp.c_str(), hc_hint);
    }
    test_matches_the_reference();
    test_rival_readings_are_rejected();
    test_zero_injection_is_a_plain_residual_add();
    test_write_weight_is_per_stream_and_centred_on_one();
    test_null_inject_leaves_the_buffer_untouched();
    test_f32_tensor_is_read_by_its_declared_precision();
    test_bad_inputs_fail_loudly();
    std::puts("gated_residual: 超连接的两半——按流归一与门控平均、以及把块输出按流权重加回——"
              "与主机参考一致，且六处错读法都被拒");
    return 0;
}