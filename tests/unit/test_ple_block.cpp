// PLE 块（第 1 层的 per-layer embedding）的回归。不引第三方框架。
//
// 几何刻意取小（n_embd 8、hc 2 条流，历史仍是 9 行），但**每个位置的值都不一样**——这一条有来历：
// 夹具有过「重复的数据把对手抹平」的教训，转置、错位的读法在重复数据上会碰巧对上。
//
// 五路证据：
//   1. 与「照公式直写的独立参考」逐段逐位一致（key / value / gate / gated / normalized / conv / result）；
//   2. 引擎点名的三处错读法各按错的算一遍、要求可观测：卷积的输入用 gated、卷积核按转置读、
//      历史按通道慢读；另加两条本地读法（门不带符号的平方根、结果不套 silu）；
//   3. 历史推进是逐通道错位拷贝——「整块 memmove」那条错法可观测；
//   4. 门的两处性质：点积恰好为 0 那一路的门正好是 0.5（sign(0) = 0），且 s → -s 时门关于 0.5 对称；
//   5. 「投影已算好」那条入口与整块那条逐位一致（投影重算一遍不该改变结果）。
#include "ple/block.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include "kernels/fp16.hpp"
#include "ple/spec.hpp"
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer::ple;
using namespace qinfer::kernels;

namespace {

constexpr std::size_t kE = 8;   // n_embd
constexpr std::size_t kHc = 2;  // 残差流条数
constexpr std::size_t kH = kE * kHc;

struct Lcg {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

float silu_g(float x) { return x / (1.0f + std::exp(-x)); }

struct Fixture {
    std::vector<std::uint16_t> key, value, conv1d;
    std::vector<float> nk, nq, nc;
    std::vector<float> emb, hidden, hist;
};

// 视图**每次现算**，不存进 Fixture：存进去的话，一次拷贝/移动就会留下指着旧缓冲的裸指针
// （写这条用例时就踩过：改完 key 再跑，块读的还是老地址上的那份）。
BlockWeights weights_of(const Fixture& f) {
    return BlockWeights{f.key.data(), f.value.data(), f.nk.data(), f.nq.data(), f.nc.data(),
                        f.conv1d.data()};
}

BlockShapes shapes_of(const Fixture&) { return BlockShapes{kE, kHc}; }

Fixture make_fixture() {
    Fixture f;
    Lcg r;
    f.key.resize(kH * kE);
    for (std::uint16_t& t : f.key) t = f32_to_bf16_bits(r.next());
    f.value.resize(kE * kE);
    for (std::uint16_t& t : f.value) t = f32_to_bf16_bits(r.next());
    f.conv1d.resize(static_cast<std::size_t>(kConvKernel) * kH);
    for (std::uint16_t& t : f.conv1d) t = f32_to_f16_bits(0.25f * r.next());
    f.nk.resize(kH);
    f.nq.resize(kH);
    f.nc.resize(kH);
    for (std::size_t i = 0; i < kH; ++i) {
        f.nk[i] = 1.0f + 0.1f * static_cast<float>(i % 3);
        f.nq[i] = 0.5f + 0.2f * static_cast<float>(i % 5);
        f.nc[i] = 2.0f - 0.1f * static_cast<float>(i % 4);
    }
    f.emb.resize(kE);
    for (std::size_t i = 0; i < kE; ++i) f.emb[i] = 0.3f * static_cast<float>(i) - 1.0f;
    f.hidden.resize(kH);
    for (std::size_t i = 0; i < kH; ++i) f.hidden[i] = 0.5f - 0.05f * static_cast<float>(i);
    // 历史：每个 (通道, 行) 一个互不相同的值——同一行各通道相同的话，「通道慢」那条会被抹平。
    f.hist.resize(kH * kHistRows);
    for (std::size_t c = 0; c < kH; ++c) {
        for (int row = 0; row < kHistRows; ++row) {
            f.hist[c * kHistRows + static_cast<std::size_t>(row)] =
                0.01f * static_cast<float>(static_cast<int>(c) * 100 + row * 7 + 3);
        }
    }
    return f;
}

// 逐流的 RMS 归一的独立写法（同一套算术，故可逐位比）。
std::vector<float> gnorm(const std::vector<float>& x, const std::vector<float>& gamma) {
    std::vector<float> y(kH, 0.0f);
    for (std::size_t c = 0; c < kHc; ++c) {
        double sum = 0.0;
        for (std::size_t d = 0; d < kE; ++d) {
            const float xv = x[c * kE + d];
            sum += static_cast<double>(xv * xv);  // 乘积先舍 f32 再放宽
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(kE)) +
                                             kPleRmsEps);
        for (std::size_t d = 0; d < kE; ++d) {
            y[c * kE + d] = x[c * kE + d] * scale * gamma[c * kE + d];
        }
    }
    return y;
}

// 卷积那一段，参数化到「用哪一份 terms / 核怎么索引 / 历史怎么索引 / 要不要 silu」——三条错读法
// 各自只换一个开关，就都在同一个参考里显形。
std::vector<float> conv_stage(const Fixture& f, const float* terms, bool transposed_kernel,
                              bool channel_slow_hist, bool no_silu) {
    std::vector<float> conv(kH, 0.0f);
    for (std::size_t c = 0; c < kH; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < kConvKernel; ++k) {
            const int row = kHistRows - (kConvKernel - 1 - k) * kNgramSize;
            const float v =
                (row == kHistRows)
                    ? terms[c]
                    : (channel_slow_hist ? f.hist[static_cast<std::size_t>(row) * kH + c]
                                         : f.hist[c * kHistRows + static_cast<std::size_t>(row)]);
            const std::size_t ki = transposed_kernel
                                       ? static_cast<std::size_t>(k) * kH + c
                                       : static_cast<std::size_t>(k) + kConvKernel * c;
            acc += f16_bits_to_f32(f.conv1d[ki]) * v;
        }
        conv[c] = no_silu ? acc : silu_g(acc);
    }
    return conv;
}

struct Stages {
    std::vector<float> key, value, gate, gated, normalized, conv, result;
};

// 独立参考。两个投影调的是同一份已经单独测过的 `bf16_gemv`（本用例要核的是块自己的算术与
// 布局，不是那个投影）；归一、门、广播、卷积、残差都直写。
void reference_block(const Fixture& f, Stages& s) {
    s.key.resize(kH, 0.0f);
    s.value.resize(kE, 0.0f);
    bf16_gemv(f.key.data(), kH, kE, f.emb.data(), s.key.data());
    bf16_gemv(f.value.data(), kE, kE, f.emb.data(), s.value.data());

    s.key = gnorm(s.key, f.nk);
    const std::vector<float> query = gnorm(f.hidden, f.nq);

    s.gate.resize(kHc, 0.0f);
    for (std::size_t c = 0; c < kHc; ++c) {
        double acc = 0.0;
        for (std::size_t d = 0; d < kE; ++d) {
            acc += static_cast<double>(s.key[c * kE + d] * query[c * kE + d]);
        }
        const float sv = static_cast<float>(acc) / std::sqrt(static_cast<float>(kE));
        const float mag = std::sqrt(std::fmax(std::fabs(sv), 1e-6f));
        const float sgn = (sv > 0.0f) ? 1.0f : ((sv < 0.0f) ? -1.0f : 0.0f);
        s.gate[c] = 1.0f / (1.0f + std::exp(-(sgn * mag)));
    }
    s.gated.resize(kH, 0.0f);
    for (std::size_t i = 0; i < kH; ++i) s.gated[i] = s.value[i % kE] * s.gate[i / kE];

    s.normalized = gnorm(s.gated, f.nc);
    s.conv = conv_stage(f, s.normalized.data(), false, false, false);
    s.result.resize(kH, 0.0f);
    for (std::size_t i = 0; i < kH; ++i) s.result[i] = f.hidden[i] + s.gated[i] + s.conv[i];
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
    }
    return true;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        den += std::fabs(static_cast<double>(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

// 把七段跑出来（段缓冲由调用方给，模拟调用方按需取中间结果）。
void run_all(Fixture& f, std::vector<std::vector<float>>& out_bufs, BlockOut& out, BlockScratch& sc,
             std::string& err) {
    const std::size_t lens[7] = {kH, kE, kHc, kH, kH, kH, kH};
    for (int i = 0; i < 7; ++i) out_bufs[static_cast<std::size_t>(i)].assign(lens[i], 0.0f);
    out.key = out_bufs[0].data();
    out.value = out_bufs[1].data();
    out.gate = out_bufs[2].data();
    out.gated = out_bufs[3].data();
    out.normalized = out_bufs[4].data();
    out.conv = out_bufs[5].data();
    out.result = out_bufs[6].data();
    std::vector<float> hidden = f.hidden;  // 输入的那一份（就地写 R 那条另有单独一例）
    CHECK(ple_block(f.emb.data(), hidden.data(), f.hist.data(), weights_of(f), shapes_of(f), out, sc, err));
}

void test_matches_the_reference() {
    Fixture f = make_fixture();
    std::vector<std::vector<float>> bufs(7);
    BlockOut out;
    BlockScratch sc;
    std::string err;
    run_all(f, bufs, out, sc, err);

    Stages r;
    reference_block(f, r);
    CHECK(same_bits(bufs[0], r.key));
    CHECK(same_bits(bufs[1], r.value));
    CHECK(same_bits(bufs[2], r.gate));
    CHECK(same_bits(bufs[3], r.gated));
    CHECK(same_bits(bufs[4], r.normalized));
    CHECK(same_bits(bufs[5], r.conv));
    CHECK(same_bits(bufs[6], r.result));
    std::printf("  与独立参考：七段逐位一致（key/value/gate/gated/normalized/conv/result）\n");

    // 残差可以就地写：result 与 hidden 同址时结果不变（引擎明记这一条是良定义的）。
    std::vector<float> inplace = f.hidden;
    BlockOut o2;
    o2.result = inplace.data();
    CHECK(ple_block(f.emb.data(), inplace.data(), f.hist.data(), weights_of(f), shapes_of(f), o2, sc, err));
    CHECK(same_bits(inplace, r.result));
    std::printf("  残差就地写（result 与 hidden 同址）结果不变\n");
}

void test_rival_readings_are_observable() {
    Fixture f = make_fixture();
    std::vector<std::vector<float>> bufs(7);
    BlockOut out;
    BlockScratch sc;
    std::string err;
    run_all(f, bufs, out, sc, err);
    Stages r;
    reference_block(f, r);

    // 三条引擎点名的错读法 + 两条本地的，各只换一处，再从正确的中段重算到底。
    struct Rival {
        const char* name;
        std::vector<float> result;
    };
    std::vector<Rival> riv;

    {  // 卷积的输入用 gated 而不是 normalized
        const std::vector<float> cv = conv_stage(f, r.gated.data(), false, false, false);
        std::vector<float> res(kH);
        for (std::size_t i = 0; i < kH; ++i) res[i] = f.hidden[i] + r.gated[i] + cv[i];
        riv.push_back({"卷积的输入用 gated", res});
    }
    {  // 卷积核按转置读（`k·hc_dim + c` 而不是 `k + 4·c`）
        const std::vector<float> cv = conv_stage(f, r.normalized.data(), true, false, false);
        std::vector<float> res(kH);
        for (std::size_t i = 0; i < kH; ++i) res[i] = f.hidden[i] + r.gated[i] + cv[i];
        riv.push_back({"卷积核按转置读", res});
    }
    {  // 历史按通道慢读（`row·hc_dim + c` 而不是 `c·9 + row`）
        const std::vector<float> cv = conv_stage(f, r.normalized.data(), false, true, false);
        std::vector<float> res(kH);
        for (std::size_t i = 0; i < kH; ++i) res[i] = f.hidden[i] + r.gated[i] + cv[i];
        riv.push_back({"历史按通道慢读", res});
    }
    {  // 门不带符号的平方根
        std::vector<float> gate_w(kHc);
        for (std::size_t c = 0; c < kHc; ++c) {
            double acc = 0.0;
            const std::vector<float> query = gnorm(f.hidden, f.nq);
            for (std::size_t d = 0; d < kE; ++d) {
                acc += static_cast<double>(r.key[c * kE + d] * query[c * kE + d]);
            }
            const float sv = static_cast<float>(acc) / std::sqrt(static_cast<float>(kE));
            gate_w[c] = 1.0f / (1.0f + std::exp(-sv));
        }
        std::vector<float> gated_w(kH);
        for (std::size_t i = 0; i < kH; ++i) gated_w[i] = r.value[i % kE] * gate_w[i / kE];
        const std::vector<float> nm_w = gnorm(gated_w, f.nc);
        const std::vector<float> cv = conv_stage(f, nm_w.data(), false, false, false);
        std::vector<float> res(kH);
        for (std::size_t i = 0; i < kH; ++i) res[i] = f.hidden[i] + gated_w[i] + cv[i];
        riv.push_back({"门不带符号的平方根", res});
    }
    {  // 结果不套 silu
        const std::vector<float> cv = conv_stage(f, r.normalized.data(), false, false, true);
        std::vector<float> res(kH);
        for (std::size_t i = 0; i < kH; ++i) res[i] = f.hidden[i] + r.gated[i] + cv[i];
        riv.push_back({"结果不套 silu", res});
    }

    for (const Rival& v : riv) {
        const double rel = rel_l1(v.result, r.result);
        std::printf("  错读法「%s」相对偏离 %.3g\n", v.name, rel);
        CHECK(rel > 1e-2);  // 结构性错读是 O(1)；这个门限与「末位之差」隔三个数量级
    }
}

void test_history_advance_is_per_channel() {
    const Fixture f = make_fixture();
    std::vector<float> hist = f.hist;
    std::vector<float> normalized(kH);
    for (std::size_t i = 0; i < kH; ++i) normalized[i] = 0.5f + 0.125f * static_cast<float>(i);
    const std::vector<float> before = hist;
    ple_history_advance(hist.data(), normalized.data(), shapes_of(f));

    // 期望：每个通道左移一格、末行写 normalized。
    for (std::size_t c = 0; c < kH; ++c) {
        for (int row = 0; row + 1 < kHistRows; ++row) {
            CHECK(hist[c * kHistRows + static_cast<std::size_t>(row)] ==
                  before[c * kHistRows + static_cast<std::size_t>(row) + 1]);
        }
        CHECK(hist[c * kHistRows + kHistRows - 1] == normalized[c]);
    }
    // 错读法：把历史当成一整块 memmove。逐通道的错位拷贝与它在同样的数据上必须不同。
    std::vector<float> flat = before;
    for (std::size_t i = 0; i + 1 < flat.size(); ++i) flat[i] = flat[i + 1];
    flat[flat.size() - 1] = normalized[kH - 1];
    int diff = 0;
    for (std::size_t i = 0; i < flat.size(); ++i) {
        if (flat[i] != hist[i]) ++diff;
    }
    std::printf("  历史推进：逐通道错位拷贝；「整块 memmove」那条错法有 %d/%zu 个位置不同\n", diff,
                flat.size());
    CHECK(diff > 0);
}

void test_gate_is_half_when_the_dot_is_zero() {
    Fixture f = make_fixture();
    // 让第 0 条流整条为零：那一条流的 8 个元素各自是 key 矩阵第 0..7 行与自己那一行的点积，
    // 故要零掉 8 行 × 8 列（写成「零掉前 8 个 u16」只会零掉第 0 个元素——这里踩过一次）。
    for (std::size_t i = 0; i < kE * kE; ++i) f.key[i] = 0;
    std::vector<std::vector<float>> bufs(7);
    BlockOut out;
    BlockScratch sc;
    std::string err;
    run_all(f, bufs, out, sc, err);
    CHECK(bufs[2][0] == 0.5f);
    CHECK(bufs[2][1] != 0.5f);

    // s → -s 时门关于 0.5 对称：把第 1 条流取反，门应当落在 0.5 的另一侧且和为 1。
    const float g_pos = bufs[2][1];
    Fixture f2 = f;
    for (std::size_t i = kE * kE; i < 2 * kE * kE; ++i) {
        f2.key[i] = static_cast<std::uint16_t>(f2.key[i] ^ 0x8000u);  // 翻符号位
    }
    std::vector<std::vector<float>> bufs2(7);
    BlockOut out2;
    BlockScratch sc2;
    run_all(f2, bufs2, out2, sc2, err);
    const float g_neg = bufs2[2][1];
    std::printf("  门：点积为零那一路正好 0.5；翻符号后 %.6f 与 %.6f（和 %.6f）\n", g_pos, g_neg,
                static_cast<double>(g_pos) + static_cast<double>(g_neg));
    CHECK(g_pos > 0.5f && g_neg < 0.5f);
    CHECK(std::fabs(static_cast<double>(g_pos) + static_cast<double>(g_neg) - 1.0) < 1e-6);
}

void test_projected_entry_agrees_with_the_whole_block() {
    Fixture f = make_fixture();
    std::vector<std::vector<float>> bufs(7);
    BlockOut out;
    BlockScratch sc;
    std::string err;
    run_all(f, bufs, out, sc, err);

    // 投影已算好的那条：把两个投影自己算一遍喂进去，结果必须逐位相同。
    std::vector<float> key_raw(kH, 0.0f), value_raw(kE, 0.0f);
    bf16_gemv(f.key.data(), kH, kE, f.emb.data(), key_raw.data());
    bf16_gemv(f.value.data(), kE, kE, f.emb.data(), value_raw.data());
    std::vector<std::vector<float>> bufs2(7);
    BlockOut out2;
    BlockScratch sc2;
    std::string err2;
    const std::size_t lens[7] = {kH, kE, kHc, kH, kH, kH, kH};
    for (int i = 0; i < 7; ++i) bufs2[static_cast<std::size_t>(i)].assign(lens[i], 0.0f);
    out2.key = bufs2[0].data();
    out2.value = bufs2[1].data();
    out2.gate = bufs2[2].data();
    out2.gated = bufs2[3].data();
    out2.normalized = bufs2[4].data();
    out2.conv = bufs2[5].data();
    out2.result = bufs2[6].data();
    std::vector<float> hid = f.hidden;  // 输入的那一份（result 另给一处；同址那条已在上面的用例里核过）
    CHECK(ple_block_projected(key_raw.data(), value_raw.data(), hid.data(), f.hist.data(),
                              weights_of(f), shapes_of(f), out2, sc2, err2));
    for (int i = 0; i < 7; ++i) CHECK(same_bits(bufs[static_cast<std::size_t>(i)],
                                                bufs2[static_cast<std::size_t>(i)]));
    std::printf("  两条入口（整块 / 投影已算好）七段逐位一致\n");
}

}  // namespace

// 真权重逐段对照：读 `measure/ple_block_oracle.py` 写出的夹具（它自读同两份 GGUF 的六个张力，按
// numpy 的写法重算一遍），逐段比相对 L1。输入与期望值都从夹具里来，故两边喂的是同一份。
//
// 卷积那一段的相对差要对着**项**量（normalized 的 |.|）报：四项求和会相消，对着结果（|.| 小一个多
// 数量级）会把误差放大（引擎的注释里记过这一条）。其余段对着自己报。
int manual_oracle_entry(const std::string& model, int layer, const std::string& oracle) {
    std::string err;
    qinfer::artifact::GgufFile g;
    if (!g.open(model, err)) {
        std::printf("打开模型失败：%s\n", err.c_str());
        return 1;
    }
    PleSpec spec;
    if (!read_ple_spec(g, layer, 2560, 4, spec, err) || !spec.present) {
        std::printf("这一层的 PLE 登记读不出来：%s\n", err.c_str());
        return 1;
    }
    std::printf("层 %d 的六张张力都读到了\n", layer);
    PleBuffers buf;
    if (!load_ple_weights(g, spec, buf, err)) {
        std::printf("读权重失败：%s\n", err.c_str());
        return 1;
    }
    std::ifstream f(oracle, std::ios::binary);
    if (!f) {
        std::printf("打不开夹具 %s\n", oracle.c_str());
        return 1;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "PLEO") {
        std::printf("夹具的魔数不对\n");
        return 1;
    }
    int hdr[6] = {0};
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const int fl_layer = hdr[0], n_embd = hdr[1], hc = hdr[2];
    const int n_arrays = hdr[5];
    if (fl_layer != layer || n_embd != 2560 || hc != 4 || n_arrays != 10) {
        std::printf("夹具的几何与本仓库这套对不上（层 %d / n_embd %d / hc %d / %d 段）\n", fl_layer,
                    n_embd, hc, n_arrays);
        return 1;
    }
    std::vector<std::vector<float>> arr(10);
    for (int i = 0; i < 10; ++i) {
        int n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        arr[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n));
        f.read(reinterpret_cast<char*>(arr[static_cast<std::size_t>(i)].data()),
               static_cast<std::streamsize>(n) * 4);
    }
    const char* names[10] = {"emb", "hidden", "hist", "key", "value", "gate",
                             "gated", "normalized", "conv", "result"};
    for (int i = 0; i < 10; ++i) {
        std::printf("  夹具 %-10s %zu 个 float\n", names[i], arr[static_cast<std::size_t>(i)].size());
    }

    const BlockShapes shapes{static_cast<std::uint64_t>(n_embd), static_cast<std::uint64_t>(hc)};
    std::vector<std::vector<float>> got(7);
    const std::size_t lens[7] = {static_cast<std::size_t>(n_embd * hc),
                                 static_cast<std::size_t>(n_embd),
                                 static_cast<std::size_t>(hc),
                                 static_cast<std::size_t>(n_embd * hc),
                                 static_cast<std::size_t>(n_embd * hc),
                                 static_cast<std::size_t>(n_embd * hc),
                                 static_cast<std::size_t>(n_embd * hc)};
    for (int i = 0; i < 7; ++i) got[static_cast<std::size_t>(i)].assign(lens[i], 0.0f);
    BlockOut out;
    out.key = got[0].data();
    out.value = got[1].data();
    out.gate = got[2].data();
    out.gated = got[3].data();
    out.normalized = got[4].data();
    out.conv = got[5].data();
    out.result = got[6].data();
    std::vector<float> hidden = arr[1];
    BlockScratch sc;
    if (!ple_block(arr[0].data(), hidden.data(), arr[2].data(), buf.view(), shapes, out, sc, err)) {
        std::printf("跑块失败：%s\n", err.c_str());
        return 1;
    }

    int bad = 0;
    const char* stage[7] = {"key", "value", "gate", "gated", "normalized", "conv", "result"};
    // 卷积那一段的项量（对着它报相对差）。
    double term_mag = 0.0;
    for (float v : arr[7]) term_mag += std::fabs(static_cast<double>(v));
    for (int i = 0; i < 7; ++i) {
        const std::vector<float>& want = arr[static_cast<std::size_t>(i + 3)];
        const std::vector<float>& have = got[static_cast<std::size_t>(i)];
        if (want.size() != have.size()) {
            std::printf("  %-11s 长度对不上\n", stage[i]);
            ++bad;
            continue;
        }
        double num = 0.0, den = 0.0;
        for (std::size_t k = 0; k < want.size(); ++k) {
            num += std::fabs(static_cast<double>(have[k]) - static_cast<double>(want[k]));
            den += std::fabs(static_cast<double>(want[k]));
        }
        const char* base = (i == 5) ? "（对着项量）" : "";
        const double rel = num / ((i == 5) ? term_mag : (den > 0.0 ? den : 1.0));
        std::printf("  %-11s 相对 L1 %.3e%s  |期望|合计 %.6g\n", stage[i], rel, base, den);
        if (!(rel < 1e-4)) ++bad;
    }
    if (bad != 0) {
        std::printf("真权重逐段对照：%d 段不过（阈值 1e-4）\n", bad);
        return 1;
    }
    std::puts("真权重逐段对照：七段都在 1e-4 以内（真权重 + 合成输入，独立实现重算）");
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 1;
        std::string oracle;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--layer" && i + 1 < argc) {
                layer = std::atoi(argv[++i]);
            } else if (a == "--oracle" && i + 1 < argc) {
                oracle = argv[++i];
            } else if (a != argv[2]) {
                std::printf("不认识的参数：%s\n", a.c_str());
                return 2;
            }
        }
        if (oracle.empty()) {
            std::printf("真权重那条要 --oracle <夹具>\n");
            return 2;
        }
        return manual_oracle_entry(argv[2], layer, oracle);
    }
    test_matches_the_reference();
    test_rival_readings_are_observable();
    test_history_advance_is_per_channel();
    test_gate_is_half_when_the_dot_is_zero();
    test_projected_entry_agrees_with_the_whole_block();
    std::puts("ple_block: 七段与独立参考逐位一致，三处引擎点名的错读法与两条本地错读法都可观测，"
              "历史推进与门的性质成立");
    return 0;
}