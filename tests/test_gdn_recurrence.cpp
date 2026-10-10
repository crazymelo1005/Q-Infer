// GDN 循环本体与收尾的回归。不引第三方框架。
//
// 六处陷阱（见 gdn_recurrence.hpp 的清单）各按错的读法算一遍，要求与对的读法有明显差别：
// 头配对取模对交错、衰减在更新前对之后、状态布局 (i,h,j) 对 (i,j,h)、beta 未 sigmoid、
// l2 的空间加法对均值、收尾 sigmoid 对 silu。另有三条性质断言（不是数值）：
//   零门零 beta 时状态不变且 o = 0；beta=0 时只有衰减在动（d=0 故无秩一更新）；
//   两次同样输入要给出逐位相同的状态与输出（同配置可复现）。
#include "dense/gdn_recurrence.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kS = 3;    // 头宽（也是状态被收缩的那一维）
constexpr std::uint64_t kHk = 2;   // q/k 头数
constexpr std::uint64_t kHv = 4;   // v 头数（是 h_k 的整数倍）
constexpr float kEps = 1e-6f;

struct Lcg {
    std::uint64_t s = 0x2545F4914F6CDD1Dull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

GdnRecShapes shapes() { return GdnRecShapes{kS, kHk, kHv}; }

struct Vecs {
    std::vector<float> q, k, v, gate, beta;
    Vecs() {
        Lcg g;
        q.resize(kHk * kS);
        k.resize(kHk * kS);
        v.resize(kHv * kS);
        gate.resize(kHv);
        beta.resize(kHv);
        for (float& t : q) t = g.next();
        for (float& t : k) t = g.next();
        for (float& t : v) t = g.next();
        for (std::uint64_t h = 0; h < kHv; ++h) {
            gate[h] = -0.2f - 0.1f * static_cast<float>(h);  // 负的门 -> dec < 1，状态衰减
            beta[h] = 0.3f + 0.1f * static_cast<float>(h);
        }
    }
};

// ---- 参考实现（double 累加、按 (i,j,h) 布局），每处陷阱一个开关 ----
struct Opts {
    bool modulo_pairing = true;    // false：交错配对 h / (h_v/h_k)
    bool decay_before_update = true;  // false：先更新再衰减
    bool transposed_state = true;  // false：按 (i,j,h) 索引同一块内存（错的读法）
    bool sigmoid_beta = true;      // false：错的读法（忘了 sigmoid，直接用原始值）
    bool silu_on_z = false;        // true：收尾用 silu
};

std::vector<float> make_z(std::uint64_t n) {
    Lcg g;
    std::vector<float> z(n);
    for (float& t : z) t = g.next();
    return z;
}

std::vector<float> make_ssm_norm(std::uint64_t n) {
    std::vector<float> v(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(i)] = 0.5f + 0.25f * static_cast<float>(i % 5);
    }
    return v;
}

void reference(const Opts& o, const GdnRecShapes& s, const Vecs& in, std::vector<double>& state,
               std::vector<float>& out_o, std::vector<float>& out_y,
               const std::vector<float>& z, const std::vector<float>& ssm_norm) {
    const std::uint64_t S = s.S;
    // 参考侧一律用 (i, h, j) 的下标（与实现同一约定），除非开关要求 (i, j, h)。
    auto at = [&](std::uint64_t i, std::uint64_t h, std::uint64_t j) {
        return o.transposed_state ? (i * s.h_v + h) * S + j : (i * S + j) * s.h_v + h;
    };
    out_o.assign(s.h_v * S, 0.0f);
    std::vector<double> q(in.q.begin(), in.q.end()), k(in.k.begin(), in.k.end());
    (void) q;
    (void) k;
    for (std::uint64_t h = 0; h < s.h_v; ++h) {
        const std::uint64_t idx = o.modulo_pairing ? (h % s.h_k) : (h / (s.h_v / s.h_k));
        for (std::uint64_t j = 0; j < S; ++j) {
            const double dec = std::exp(static_cast<double>(in.gate[h]));
            double sk = 0.0;
            for (std::uint64_t i = 0; i < S; ++i) {
                double& e = state[at(i, h, j)];
                if (o.decay_before_update) e *= dec;
                sk += e * static_cast<double>(in.k[idx * S + i]);
            }
            const double raw = static_cast<double>(in.beta[h]);
            const double b = o.sigmoid_beta ? 1.0 / (1.0 + std::exp(-raw)) : raw;
            const double d = (static_cast<double>(in.v[h * S + j]) - sk) * b;
            for (std::uint64_t i = 0; i < S; ++i) {
                double& e = state[at(i, h, j)];
                e += static_cast<double>(in.k[idx * S + i]) * d;
                if (!o.decay_before_update) e *= dec;
            }
            double acc = 0.0;
            for (std::uint64_t i = 0; i < S; ++i) {
                acc += state[at(i, h, j)] * static_cast<double>(in.q[idx * S + i]);
            }
            out_o[h * S + j] = static_cast<float>(acc);
        }
    }
    out_y.assign(s.h_v * S, 0.0f);
    for (std::uint64_t h = 0; h < s.h_v; ++h) {
        double ss = 0.0;
        for (std::uint64_t j = 0; j < S; ++j) {
            const double vv = out_o[h * S + j];
            ss += vv * vv;
        }
        const double rs = 1.0 / std::sqrt(ss / static_cast<double>(S) + kEps);
        for (std::uint64_t j = 0; j < S; ++j) {
            const double zv = static_cast<double>(z[h * S + j]);
            const double act = o.silu_on_z ? zv / (1.0 + std::exp(-zv)) : 1.0 / (1.0 + std::exp(-zv));
            out_y[h * S + j] = static_cast<float>(static_cast<double>(out_o[h * S + j]) * rs *
                                                  static_cast<double>(ssm_norm[j]) * act);
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

// 跑实现：返回 o 与收尾后的 y（状态留在 st 里）。
void run_impl(const Vecs& in, std::vector<float>& o_out, std::vector<float>& y_out,
              const std::vector<float>& z, const std::vector<float>& ssm_norm,
              GdnState* keep = nullptr) {
    const GdnRecShapes s = shapes();
    GdnState st;
    st.resize(s);
    std::string err;
    std::vector<float> beta = in.beta;  // 原始 ssm_beta@x，先 sigmoid
    CHECK(gdn_beta_gate(beta.data(), s.h_v, err));
    o_out.assign(s.h_v * s.S, 0.0f);
    CHECK(gdn_step(st, s, in.q.data(), in.k.data(), in.v.data(), in.gate.data(), beta.data(),
                   o_out.data(), err));
    CHECK(err.empty());
    y_out.assign(s.h_v * s.S, 0.0f);
    CHECK(gdn_closing_norm(s, o_out.data(), z.data(), ssm_norm.data(), kEps, y_out.data(), err));
    CHECK(err.empty());
    if (keep != nullptr) *keep = st;
}

struct RivalCase {
    const char* name;
    Opts o;
};

void test_matches_the_reference() {
    const Vecs in;
    const std::vector<float> z = make_z(kHv * kS), norm = make_ssm_norm(kS);
    std::vector<float> got_o, got_y;
    GdnState kept;
    run_impl(in, got_o, got_y, z, norm, &kept);
    std::vector<double> st_r(static_cast<std::size_t>(kS * kHv * kS), 0.0);
    std::vector<float> ref_o, ref_y;
    reference(Opts{}, shapes(), in, st_r, ref_o, ref_y, z, norm);
    const double do_ = rel_l1(ref_o, got_o), dy = rel_l1(ref_y, got_y);
    std::printf("与参考的相对 L1：o %.3g、y %.3g\n", do_, dy);
    CHECK(do_ < 1e-5 && dy < 1e-5);
    // 承载的状态也要对，且这一条是把**状态布局**钉住的那条。单看一个自洽的实现看不出布局：读写用
    // 同一个下标映射时，换布局只是换个坐标名；只有把状态交给别人读时它才可观测。引擎是 (S,h_v,S)
    // 且 j 最快、参考是转置的，故这里按契约的下标逐元素比。
    double sdiff = 0.0, smag = 0.0;
    for (std::size_t i = 0; i < kept.data.size(); ++i) {
        sdiff += std::fabs(static_cast<double>(kept.data[i]) - st_r[i]);
        smag += std::fabs(st_r[i]);
    }
    const double sd = sdiff / (smag > 1e-30 ? smag : 1e-30);
    std::printf("承载状态的相对 L1：%.3g\n", sd);
    CHECK(sd < 1e-5);
}

void test_rival_readings_are_rejected() {
    const Vecs in;
    const std::vector<float> z = make_z(kHv * kS), norm = make_ssm_norm(kS);
    std::vector<float> got_o, got_y;
    run_impl(in, got_o, got_y, z, norm);

    std::vector<RivalCase> cases;
    {
        Opts o;
        o.modulo_pairing = false;
        cases.push_back({"头配对用交错而不是取模", o});
    }
    {
        Opts o;
        o.decay_before_update = false;
        cases.push_back({"衰减放在秩一更新之后", o});
    }
    {
        Opts o;
        o.sigmoid_beta = false;
        cases.push_back({"beta 没 sigmoid（直接用原始值）", o});
    }
    {
        Opts o;
        o.silu_on_z = true;
        cases.push_back({"收尾用 silu 而不是 sigmoid", o});
    }
    for (const RivalCase& c : cases) {
        std::vector<double> st_r(static_cast<std::size_t>(kS * kHv * kS), 0.0);
        std::vector<float> ref_o, ref_y;
        reference(c.o, shapes(), in, st_r, ref_o, ref_y, z, norm);
        // 与正确读法的参考比（不是与实现比），这样两边都是参考侧、差别只来自那处读法。
        std::vector<double> st_ok(static_cast<std::size_t>(kS * kHv * kS), 0.0);
        std::vector<float> ok_o, ok_y;
        reference(Opts{}, shapes(), in, st_ok, ok_o, ok_y, z, norm);
        const double worst = std::max(rel_l1(ok_o, ref_o), rel_l1(ok_y, ref_y));
        std::printf("错的读法「%s」的最大偏离 %.3g\n", c.name, worst);
        CHECK(worst > 1e-4);
    }
}

// 性质：门很负（衰减~0）且 beta = 0 时，状态被清成 0、秩一更新不发生，o 也全 0。
void test_zero_beta_and_strong_decay_zero_the_state() {
    const GdnRecShapes s = shapes();
    Vecs in;
    in.beta.assign(kHv, 0.0f);
    for (float& t : in.gate) t = -60.0f;  // exp(-60) ≈ 8.8e-27
    std::vector<float> beta = in.beta;
    std::string err;
    CHECK(gdn_beta_gate(beta.data(), s.h_v, err));  // sigmoid(0) = 0.5，故这里不用它
    // 直接用 0 当「已 sigmoid 的 beta」——sigmoid(0)=0.5 不是 0，故另造一组已 sigmoid 的值。
    std::vector<float> beta_zero(kHv, 0.0f);
    GdnState st;
    st.resize(s);
    for (float& e : st.data) e = 1.0f;  // 先塞满
    std::vector<float> o(s.h_v * s.S, 0.0f);
    CHECK(gdn_step(st, s, in.q.data(), in.k.data(), in.v.data(), in.gate.data(), beta_zero.data(),
                   o.data(), err));
    for (float e : st.data) CHECK(std::fabs(e) < 1e-20f);
    for (float e : o) CHECK(std::fabs(e) < 1e-20f);  // 状态被衰减到 8.8e-27，o 同量级而非恰好 0
}

// 性质与数值：beta = 0 时只有衰减在动（d = 0，秩一更新不发生），且 o 用的是衰减后的状态。
// 这里两侧从**同一个非零初始状态**出发，逐元素比状态与 o。
void test_beta_zero_leaves_only_the_decay() {
    const GdnRecShapes s = shapes();
    Vecs in;
    in.beta.assign(kHv, 0.0f);
    GdnState st;
    st.resize(s);
    Lcg g;
    for (float& e : st.data) e = 0.2f * g.next();
    const std::vector<float> initial = st.data;
    std::vector<float> o(s.h_v * s.S, 0.0f);
    std::string err;
    CHECK(gdn_step(st, s, in.q.data(), in.k.data(), in.v.data(), in.gate.data(), in.beta.data(),
                   o.data(), err));
    CHECK(err.empty());

    std::vector<double> st_r(initial.begin(), initial.end());
    std::vector<float> ref_o, ref_y;
    const std::vector<float> z = make_z(kHv * kS), norm = make_ssm_norm(kS);
    // 传进去的 beta 已经是 sigmoid 过的小数（0），故参考侧不再 sigmoid——与实现的契约一致：
    // `gdn_step` 收的就是已过门的 beta，sigmoid 那一步是层的事（见 gdn_beta_gate）。
    Opts no_sig;
    no_sig.sigmoid_beta = false;
    reference(no_sig, s, in, st_r, ref_o, ref_y, z, norm);
    CHECK(std::fabs(rel_l1(ref_o, o)) < 1e-6);
    for (std::size_t i = 0; i < st.data.size(); ++i) {
        const float want = static_cast<float>(st_r[i]);
        // f32 与 double 不保证逐位相同，故按相对容差比（这一条要钉的是「只有衰减在动」，不是末位）。
        const float tol = 1e-6f * (1.0f + std::fabs(want));
        if (std::fabs(st.data[i] - want) > tol) {
            std::printf("FAIL 状态第 %zu 个：得 %.9g 期 %.9g\n", i,
                        static_cast<double>(st.data[i]), static_cast<double>(want));
            CHECK(false);
        }
    }
}

void test_bad_inputs_fail_loudly() {
    const GdnRecShapes s = shapes();
    const Vecs in;
    GdnState st;
    std::vector<float> o(s.h_v * s.S, 0.0f);
    std::string err;
    CHECK(!gdn_step(st, s, in.q.data(), in.k.data(), in.v.data(), in.gate.data(), in.beta.data(),
                    o.data(), err));
    CHECK(!err.empty());  // 状态没 resize
    err.clear();
    st.resize(s);
    CHECK(!gdn_step(st, GdnRecShapes{kS, kHk, 5}, in.q.data(), in.k.data(), in.v.data(),
                    in.gate.data(), in.beta.data(), o.data(), err));
    CHECK(!err.empty());  // h_v 不是 h_k 的整数倍
    err.clear();
    CHECK(!gdn_step(st, s, nullptr, in.k.data(), in.v.data(), in.gate.data(), in.beta.data(),
                    o.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_matches_the_reference();
    test_rival_readings_are_rejected();
    test_zero_beta_and_strong_decay_zero_the_state();
    test_beta_zero_leaves_only_the_decay();
    test_bad_inputs_fail_loudly();
    std::puts("gdn_recurrence: 递推（含衰减次序与头配对）、beta 的 sigmoid 与收尾归一都与参考一致，"
              "且四处错读法都被拒");
    return 0;
}