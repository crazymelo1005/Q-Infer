// 顶层步循环（`step_loop`）的回归。不引第三方框架。
//
// 合成夹具是一个极小的整套模型：层 0 是 GDN、层 1 是 QSA（n_embd 256 / hc 2 / hc_lr 3、MoE hidden 同宽、
// 两种注意力都用头宽 32），加收尾的 `output_hc_*` 与 `output.weight`（词表 8）。
//
// 钉子：
//   1. 建得起来、判型对（层 0 GDN、层 1 QSA），一个 token 跑出长度=词表的有限 logits；
//   2. 层间**串的是那个 hc 条流的栈**：把层序对调、少跑最后一层、或干脆不跑循环都给出明显不同的 logits；
//   3. 逐层状态各归各层：同一个 scratch 连跑两个 token 时第二个 token 的输出不同（GDN 状态与 QSA 的 KV
//      都前进了）；新 scratch 从头跑则与第一次逐位一致；
//   4. 接错必须报错：层数与文件不符、缺输出头、没预备 scratch 就跑。
// 真权重在 --model 入口跑（环境2）。
#include "dense/kv_store.hpp"
#include "dense/kv_window.hpp"
#include "dense/spec_decode.hpp"
#include "dense/step_loop.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include "kernels/fp16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kN = 256;
constexpr std::uint64_t kHc = 2;
constexpr std::uint64_t kLr = 3;
constexpr std::uint64_t kFfn = 64;
constexpr std::uint64_t kExpert = 4;
constexpr int kTopK = 2;
constexpr std::uint64_t kHd = 32;  // 两种注意力都用这个头宽
constexpr std::uint64_t kQk = kHd, kVd = kHd, kCd = kQk + kQk + kVd;
constexpr std::uint64_t kDc = 4;
constexpr std::uint64_t kQsaNh = 4, kQsaNkv = 2, kIdx = 4, kIdxH = 2, kNrot = 4;
constexpr std::uint64_t kIdxTopK = 2048, kIdxBlock = 4;
constexpr std::uint64_t kVocab = 8;
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    std::vector<std::uint8_t> b;
    auto u32 = [&b](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u64 = [&b](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    const auto put_str = [&](const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    put_str("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        put_str(t.name);
        u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (b.size() % kAlign != 0) b.push_back(0);
    for (const TensorSpec& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        while (b.size() % kAlign != 0) b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p;
}

std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols, int seed,
                                    std::uint16_t d = 0x3000u) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t blk = 0; blk < cols / 32; ++blk) {
            out.push_back(static_cast<std::uint8_t>(d & 0xFF));
            out.push_back(static_cast<std::uint8_t>(d >> 8));
            for (std::uint64_t c = 0; c < 32; ++c) {
                out.push_back(static_cast<std::uint8_t>(((r + blk * 32 + c + seed) % 7) - 3));
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

std::vector<std::uint8_t> f32_lin(std::uint64_t n, float base) {
    std::vector<float> f(n);
    for (std::uint64_t i = 0; i < n; ++i) f[static_cast<std::size_t>(i)] = base + 0.01f * (float) i;
    return f32_vec(f);
}

std::vector<std::uint8_t> bf16_sym(std::uint64_t n, float amp) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::uint64_t i = 0; i < n; ++i) {
        const float v = amp * (static_cast<float>(static_cast<int>(i % 7) - 3) / 3.0f);
        const std::uint16_t bits = kernels::f32_to_bf16_bits(v);
        out[static_cast<std::size_t>(2 * i)] = static_cast<std::uint8_t>(bits & 0xFF);
        out[static_cast<std::size_t>(2 * i + 1)] = static_cast<std::uint8_t>(bits >> 8);
    }
    return out;
}

std::vector<std::uint8_t> router_tensor(std::uint64_t cols, std::uint64_t experts) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(cols * experts * 2), 0);
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        data[i] = 0x80;  // bf16 1.0
        data[i + 1] = 0x3F;
    }
    return data;
}

std::vector<TensorSpec> moe_tensors(const std::string& b) {
    const std::uint16_t d = 0x2800u;
    std::vector<TensorSpec> ts;
    ts.push_back({b + "ffn_gate_exps.weight", {kN, kFfn, kExpert}, 8,
                  q8_0_rows(kFfn * kExpert, kN, 1, d)});
    ts.push_back({b + "ffn_up_exps.weight", {kN, kFfn, kExpert}, 8,
                  q8_0_rows(kFfn * kExpert, kN, 2, d)});
    ts.push_back({b + "ffn_down_exps.weight", {kFfn, kN, kExpert}, 8,
                  q8_0_rows(kN * kExpert, kFfn, 3, d)});
    ts.push_back({b + "ffn_gate_inp.weight", {kN, kExpert}, 30, router_tensor(kN, kExpert)});
    return ts;
}

std::vector<TensorSpec> gr_group(const std::string& p, float norm_base, float inj_amp) {
    std::vector<TensorSpec> ts;
    ts.push_back({p + "norm.weight", {kN * kHc}, 0, f32_lin(kN * kHc, norm_base)});
    ts.push_back({p + "down.weight", {kN * kHc, kLr}, 30, bf16_sym(kN * kHc * kLr, 0.05f)});
    ts.push_back({p + "up.weight", {kLr, kN * kHc}, 30, bf16_sym(kLr * kN * kHc, 0.5f)});
    ts.push_back({p + "inject.weight", {kN * kHc, kHc}, 30, bf16_sym(kN * kHc * kHc, inj_amp)});
    return ts;
}

std::vector<TensorSpec> gdn_attn(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_qkv.weight", {kN, kCd}, 8, q8_0_rows(kCd, kN, 1)});
    ts.push_back({b + "attn_gate.weight", {kN, kVd}, 8, q8_0_rows(kVd, kN, 2)});
    ts.push_back({b + "ssm_out.weight", {kVd, kN}, 8, q8_0_rows(kN, kVd, 3)});
    ts.push_back({b + "ssm_conv1d.weight", {kDc, kCd}, 0,
                  f32_vec(std::vector<float>(static_cast<std::size_t>(kDc * kCd), 0.25f))});
    ts.push_back({b + "ssm_alpha.weight", {kN, 1}, 30, bf16_sym(kN, 1.0f)});
    ts.push_back({b + "ssm_beta.weight", {kN, 1}, 30, bf16_sym(kN, 0.9f)});
    ts.push_back({b + "ssm_a", {1}, 0, f32_vec(std::vector<float>{-0.5f})});
    ts.push_back({b + "ssm_dt.bias", {1}, 0, f32_vec(std::vector<float>{0.5f})});
    ts.push_back({b + "ssm_norm.weight", {kHd}, 0, f32_vec(std::vector<float>(kHd, 1.0f))});
    return ts;
}

std::vector<TensorSpec> qsa_attn(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back(
        {b + "attn_q.weight", {kN, 2 * kQsaNh * kHd}, 8, q8_0_rows(2 * kQsaNh * kHd, kN, 1)});
    ts.push_back({b + "attn_k.weight", {kN, kQsaNkv * kHd}, 8, q8_0_rows(kQsaNkv * kHd, kN, 2)});
    ts.push_back({b + "attn_v.weight", {kN, kQsaNkv * kHd}, 8, q8_0_rows(kQsaNkv * kHd, kN, 3)});
    ts.push_back({b + "attn_output.weight", {kQsaNh * kHd, kN}, 8, q8_0_rows(kN, kQsaNh * kHd, 4)});
    ts.push_back({b + "attn_q_norm.weight", {kHd}, 0, f32_vec(std::vector<float>(kHd, 1.0f))});
    ts.push_back({b + "attn_k_norm.weight", {kHd}, 0, f32_vec(std::vector<float>(kHd, 1.0f))});
    ts.push_back({b + "indexer.q_proj.weight", {kN, kIdxH * kIdx}, 30,
                  bf16_sym(kN * kIdxH * kIdx, 0.1f)});
    ts.push_back({b + "indexer.k_proj.weight", {kN, kIdx}, 30, bf16_sym(kN * kIdx, 0.1f)});
    ts.push_back({b + "indexer.q_norm.weight", {kIdx}, 0, f32_vec(std::vector<float>(kIdx, 1.0f))});
    ts.push_back({b + "indexer.k_norm.weight", {kIdx}, 0, f32_vec(std::vector<float>(kIdx, 1.0f))});
    return ts;
}

void append(std::vector<TensorSpec>& base, std::vector<TensorSpec> more) {
    for (auto& t : more) base.push_back(std::move(t));
}

std::vector<TensorSpec> layer_tensors(const std::string& b, bool gdn, float attn_norm_base) {
    std::vector<TensorSpec> ts = gr_group(b + "hc_attn_", attn_norm_base, 0.4f);
    append(ts, gr_group(b + "hc_ffn_", attn_norm_base + 1.0f, -0.4f));
    append(ts, gdn ? gdn_attn(b) : qsa_attn(b));
    append(ts, moe_tensors(b));
    return ts;
}

std::vector<TensorSpec> head_tensors() {
    std::vector<TensorSpec> ts;
    ts.push_back({"output_hc_norm.weight", {kN * kHc}, 0, f32_lin(kN * kHc, 1.0f)});
    ts.push_back({"output_hc_down.weight", {kN * kHc, kLr}, 30, bf16_sym(kN * kHc * kLr, 0.05f)});
    ts.push_back({"output_hc_up.weight", {kLr, kN * kHc}, 30, bf16_sym(kLr * kN * kHc, 0.5f)});
    ts.push_back({"output.weight", {kN, kVocab}, 8, q8_0_rows(kVocab, kN, 5)});
    return ts;
}

// token_embd.weight：IQ4_XS，[256 × 8]。d = fp16(0.5 + 0.1·r)、尺度字节固定成 ls = 33（dl = d）、
// 码字节按 (j·7 + r·13 + 5 + ib) 变化（与 test_token_embed 同一套构造）。
void push_row_block(std::vector<std::uint8_t>& out, std::uint64_t r) {
    const std::uint16_t d = kernels::f32_to_f16_bits(0.5f + 0.1f * static_cast<float>(r));
    out.push_back(static_cast<std::uint8_t>(d & 0xFF));
    out.push_back(static_cast<std::uint8_t>(d >> 8));
    out.push_back(0xAA);
    out.push_back(0xAA);
    for (int i = 0; i < 4; ++i) out.push_back(0x11);
    for (int ib = 0; ib < 8; ++ib) {
        for (int j = 0; j < 16; ++j) {
            out.push_back(
                static_cast<std::uint8_t>((j * 7 + static_cast<int>(r) * 13 + 5 + ib) & 0xFF));
        }
    }
}

std::vector<std::uint8_t> token_embd_data() {
    std::vector<std::uint8_t> data;
    for (std::uint64_t r = 0; r < kVocab; ++r) push_row_block(data, r);
    return data;
}

// 两层的小模型：层 0 GDN、层 1 QSA。两层的 hc 权重不同（norm 基值 1.0），靠层类型差异把「层序对调」暴露。
std::vector<TensorSpec> model_tensors() {
    std::vector<TensorSpec> ts = layer_tensors("blk.0.", /*gdn=*/true, 1.0f);
    append(ts, layer_tensors("blk.1.", /*gdn=*/false, 2.0f));
    append(ts, head_tensors());
    ts.push_back({"token_embd.weight", {kN, kVocab}, 23, token_embd_data()});
    return ts;
}

StepConfig two_layer_config() {
    StepConfig cfg;
    cfg.n_embd = kN;
    cfg.hc = kHc;
    cfg.hc_lr = kLr;
    cfg.n_layers = 2;
    cfg.top_k = kTopK;
    cfg.gdn_head_dim = kHd;
    cfg.qsa_head_dim = kHd;
    cfg.n_rot = kNrot;
    cfg.idx_dim = kIdx;
    cfg.idx_top_k = kIdxTopK;
    cfg.idx_block = kIdxBlock;
    return cfg;
}

std::vector<float> make_embd() {
    std::vector<float> x(kN);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.1f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }
    return x;
}

// 按给定层序跑一遍（每层一份新 scratch），再走输出头。用来算各种错的组合。
std::vector<float> logits_with_order(const artifact::GgufFile& g, const StepModel& m,
                                     const std::vector<int>& order) {
    std::string err;
    std::vector<float> R(kN * kHc, 0.0f);
    const std::vector<float> x = make_embd();
    hc_broadcast_init(x.data(), kN, kHc, R.data());
    for (int l : order) {
        StepScratch s;
        CHECK(prepare_step_scratch(m, 2, 0, s, err));
        CHECK(run_block_layer(g, m.blocks[static_cast<std::size_t>(l)], R.data(),
                              s.blocks[static_cast<std::size_t>(l)], err));
    }
    HeadScratch hs;
    CHECK(run_model_head(g, m.head, R.data(), hs, err));
    return hs.logits;
}

double rel_diff(const std::vector<float>& right, const std::vector<float>& wrong) {
    double diff = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < right.size(); ++i) {
        diff += std::fabs(static_cast<double>(right[i]) - static_cast<double>(wrong[i]));
        mag += std::fabs(static_cast<double>(right[i]));
    }
    return mag > 0.0 ? diff / mag : (diff > 0.0 ? 999.0 : 0.0);
}

void test_build_and_run() {
    const std::filesystem::path p = write_gguf("qinfer_step_two.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    StepModel m;
    CHECK(build_step_model(g, two_layer_config(), m, err));
    CHECK(err.empty());
    CHECK(m.blocks.size() == 2u);
    CHECK(m.map[0].kind == LayerKind::kGdn && m.map[1].kind == LayerKind::kQsa);
    CHECK(m.head.out.rows == kVocab && m.head.out.cols == kN);

    const std::vector<float> x = make_embd();
    StepScratch s;
    CHECK(prepare_step_scratch(m, 2, 0, s, err));
    CHECK(s.ready);
    CHECK(run_step_model(g, m, x.data(), s, err));
    CHECK(err.empty());
    CHECK(s.head.logits.size() == kVocab);
    bool finite = true, nonzero = false;
    for (float v : s.head.logits) {
        if (!std::isfinite(v)) finite = false;
        if (v != 0.0f) nonzero = true;
    }
    CHECK(finite && nonzero);

    // 层间确实串了栈：末态 R 与初值（广播）不同。
    bool moved = false;
    for (std::uint64_t i = 0; i < kN * kHc; ++i) {
        if (s.R[static_cast<std::size_t>(i)] != x[static_cast<std::size_t>(i % kN)]) moved = true;
    }
    CHECK(moved);

    // 新 scratch 从头跑 → logits 逐位一致。
    StepScratch s2;
    CHECK(prepare_step_scratch(m, 2, 0, s2, err));
    CHECK(run_step_model(g, m, x.data(), s2, err));
    for (std::size_t i = 0; i < s.head.logits.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(s.head.logits[i]) ==
              std::bit_cast<std::uint32_t>(s2.head.logits[i]));
    }

    // 正确的层序（0,1）与 run_step_model 的 logits 逐位一致（同一套子模块、同一层序）。
    const std::vector<float> ref = logits_with_order(g, m, {0, 1});
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(ref[i]) != std::bit_cast<std::uint32_t>(s.head.logits[i])) {
            std::printf("FAIL 显式层序与 run_step_model 不一致\n");
            CHECK(false);
        }
    }
}

void test_order_and_layers_matter() {
    const std::filesystem::path p = write_gguf("qinfer_step_rivals.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    StepModel m;
    CHECK(build_step_model(g, two_layer_config(), m, err));

    const std::vector<float> ref = logits_with_order(g, m, {0, 1});
    const double d_rev = rel_diff(ref, logits_with_order(g, m, {1, 0}));
    const double d_skip = rel_diff(ref, logits_with_order(g, m, {0}));
    const double d_head0 = rel_diff(ref, logits_with_order(g, m, {}));
    std::printf("  层序对调 %.4f、少跑末层 %.4f、不跑循环 %.4f\n", d_rev, d_skip, d_head0);
    CHECK(d_rev > 0.05);
    CHECK(d_skip > 0.05);
    CHECK(d_head0 > 0.05);
}

void test_state_advances() {
    const std::filesystem::path p = write_gguf("qinfer_step_seq.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    StepModel m;
    CHECK(build_step_model(g, two_layer_config(), m, err));
    StepScratch s;
    CHECK(prepare_step_scratch(m, 2, 0, s, err));

    const std::vector<float> x = make_embd();
    CHECK(run_step_model(g, m, x.data(), s, err));
    const std::vector<float> first = s.head.logits;
    CHECK(run_step_model(g, m, x.data(), s, err));  // 同一个 scratch：状态前进
    bool differs = false;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(first[i]) !=
            std::bit_cast<std::uint32_t>(s.head.logits[i])) {
            differs = true;
        }
    }
    CHECK(differs);
    // 每层各持自己那份状态：QSA 的 KV 前进了两个 cell、GDN 的循环状态与卷积状态在自己的 scratch 里。
    CHECK(s.blocks[1].qsa.pos == 2);
    CHECK(s.blocks[0].gdn.state.data.size() > 0);
    CHECK(s.blocks[0].gdn.conv.tokens == 2);
}

void test_failures() {
    std::string err;
    // 层数与文件不符：要 3 层，文件里只有 2 层。
    {
        const std::filesystem::path p = write_gguf("qinfer_step_short.gguf", model_tensors());
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        StepModel m;
        StepConfig cfg = two_layer_config();
        cfg.n_layers = 3;
        CHECK(!build_step_model(g, cfg, m, err));
        CHECK(!err.empty());
    }
    // 缺输出头。
    {
        std::vector<TensorSpec> ts = layer_tensors("blk.0.", true, 1.0f);
        append(ts, layer_tensors("blk.1.", false, 2.0f));
        const std::filesystem::path p = write_gguf("qinfer_step_nohead.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        StepModel m;
        CHECK(!build_step_model(g, two_layer_config(), m, err));
        CHECK(err.find("output_hc_norm") != std::string::npos);
    }
    // 没预备 scratch 就跑。
    {
        const std::filesystem::path p = write_gguf("qinfer_step_noprep.gguf", model_tensors());
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        StepModel m;
        CHECK(build_step_model(g, two_layer_config(), m, err));
        StepScratch s;
        const std::vector<float> x = make_embd();
        CHECK(!run_step_model(g, m, x.data(), s, err));
        CHECK(err.find("预备") != std::string::npos);
    }
}

// token id → embedding 行 → 步循环 → logits → 贪心下一个 token：最小的生成环。
void test_token_generation() {
    const std::filesystem::path p = write_gguf("qinfer_step_gen.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    StepModel m;
    CHECK(build_step_model(g, two_layer_config(), m, err));
    TokenEmbed e;
    CHECK(build_token_embed(g, kN, e, err));
    CHECK(e.vocab == kVocab);

    const auto generate = [&](std::vector<std::uint64_t>& seq) {
        StepScratch s;
        CHECK(prepare_step_scratch(m, 4, 0, s, err));
        std::uint64_t tok = 3;  // 起点
        for (int i = 0; i < 3; ++i) {
            CHECK(run_token_step(g, m, e, tok, s, err));
            CHECK(err.empty());
            for (float v : s.head.logits) CHECK(std::isfinite(v));
            tok =
                static_cast<std::uint64_t>(greedy_argmax(s.head.logits.data(), s.head.logits.size()));
            CHECK(tok < e.vocab);
            seq.push_back(tok);
        }
    };
    std::vector<std::uint64_t> seq0, seq1;
    generate(seq0);
    generate(seq1);
    CHECK(seq0.size() == 3 && seq0 == seq1);  // 两次独立跑给出同一串
    std::printf("  贪心生成 3 个 token：%llu → %llu → %llu\n", (unsigned long long) seq0[0],
                (unsigned long long) seq0[1], (unsigned long long) seq0[2]);

    // run_token_step 与「先读 embedding 再 run_step_model」等价（同配置逐位同 logits）。
    StepScratch a, b;
    CHECK(prepare_step_scratch(m, 4, 0, a, err));
    CHECK(prepare_step_scratch(m, 4, 0, b, err));
    CHECK(run_token_step(g, m, e, 5, a, err));
    std::vector<float> row(kN, 0.0f);
    CHECK(read_token_embedding(g, e, 5, row.data(), err));
    CHECK(run_step_model(g, m, row.data(), b, err));
    for (std::size_t i = 0; i < a.head.logits.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(a.head.logits[i]) ==
              std::bit_cast<std::uint32_t>(b.head.logits[i]));
    }
}

// 平路贪心：从 start 起生成 n 个 token（不做投机）。作为「投机不改变序列」这条性质的基准。
std::vector<std::uint64_t> plain_greedy(const artifact::GgufFile& g, const StepModel& m,
                                        const TokenEmbed& e, std::uint64_t start, std::size_t n) {
    std::string err;
    StepScratch s;
    CHECK(prepare_step_scratch(m, 32, 0, s, err));
    std::uint64_t tok = start;
    std::vector<std::uint64_t> seq;
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(run_token_step(g, m, e, tok, s, err));
        tok = static_cast<std::uint64_t>(greedy_argmax(s.head.logits.data(), s.head.logits.size()));
        seq.push_back(tok);
    }
    return seq;
}

// 投机解码：接受/拒绝只影响模型步数，不改变贪心轨迹；草稿全对时一轮 k 次前向产出 k+1 个 token。
void test_spec_decode() {
    const std::filesystem::path p = write_gguf("qinfer_step_spec.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    StepModel m;
    CHECK(build_step_model(g, two_layer_config(), m, err));
    TokenEmbed e;
    CHECK(build_token_embed(g, kN, e, err));
    const std::uint64_t start = 3;
    const std::vector<std::uint64_t> base = plain_greedy(g, m, e, start, 14);
    CHECK(base.size() == 14);

    const auto fresh = [&](StepScratch& s, SpecDecoder& d) {
        CHECK(prepare_step_scratch(m, 32, 0, s, err));
        CHECK(spec_commit(g, m, e, start, s, d, err));  // 先喂入起点，得到对下一格的预测
    };
    const auto steps = [&](const StepScratch& s) { return s.blocks[0].gdn.conv.tokens; };

    // 未 prime 就跑：明确报错。
    {
        StepScratch s;
        SpecDecoder d;
        CHECK(prepare_step_scratch(m, 32, 0, s, err));
        std::vector<std::uint64_t> out;
        CHECK(!spec_step(g, m, e, nullptr, 0, s, d, out, err));
        CHECK(err.find("spec_commit") != std::string::npos);
    }
    // 草稿 0 个：一轮正好 1 个 token、1 次前向。
    {
        StepScratch s;
        SpecDecoder d;
        fresh(s, d);
        const std::uint64_t before = steps(s);
        std::vector<std::uint64_t> out;
        CHECK(spec_step(g, m, e, nullptr, 0, s, d, out, err));
        CHECK(out.size() == 1u && out[0] == base[0]);
        CHECK(steps(s) - before == 1u);
    }
    // 草稿全对（k=3）：一轮产出 4 个（3 个接受的草稿 + 1 个 bonus），且与基准的前 4 个逐位一致。
    {
        StepScratch s;
        SpecDecoder d;
        fresh(s, d);
        const std::uint64_t before = steps(s);
        const std::vector<std::uint64_t> draft{base[0], base[1], base[2]};
        std::vector<std::uint64_t> out;
        CHECK(spec_step(g, m, e, draft.data(), draft.size(), s, d, out, err));
        CHECK(out.size() == 4u);
        for (std::size_t i = 0; i < 4; ++i) CHECK(out[i] == base[i]);
        // 顺序验证：每提交一个 token 一次前向（4 个 token = 4 次）。**投机带来的省法在并行验证**
        // （k 个草稿一次前向），那条路要批式前向，不在本片；本片钉的是接受规则与状态纪律。
        CHECK(steps(s) - before == 4u);
    }
    // 第一格就对不上：接受 0 个，产出 1 个（目标模型自己的），与基准一致。
    {
        StepScratch s;
        SpecDecoder d;
        fresh(s, d);
        const std::vector<std::uint64_t> draft{(base[0] + 1) % kVocab, base[1], base[2]};
        std::vector<std::uint64_t> out;
        CHECK(spec_step(g, m, e, draft.data(), draft.size(), s, d, out, err));
        CHECK(out.size() == 1u && out[0] == base[0]);
    }
    // 前两个对、第三个错：接受 2 个 + 1 个 bonus，与基准前 3 个一致。
    {
        StepScratch s;
        SpecDecoder d;
        fresh(s, d);
        const std::vector<std::uint64_t> draft{base[0], base[1], (base[2] + 1) % kVocab, base[3]};
        std::vector<std::uint64_t> out;
        CHECK(spec_step(g, m, e, draft.data(), draft.size(), s, d, out, err));
        CHECK(out.size() == 3u);
        for (std::size_t i = 0; i < 3; ++i) CHECK(out[i] == base[i]);
    }
    // 序列不变性：连续多轮（草稿长度与对错刻意变化）跑出的整串必须与基准逐位一致。
    {
        StepScratch s;
        SpecDecoder d;
        fresh(s, d);
        const std::size_t ks[5] = {1, 3, 0, 2, 4};
        std::vector<std::uint64_t> produced;
        int round = 0;
        while (produced.size() < base.size() && round < 100) {
            const std::size_t k = ks[round % 5];
            const std::size_t at = produced.size();
            std::vector<std::uint64_t> draft(k, 0);
            for (std::size_t i = 0; i < k; ++i) {
                const std::size_t idx = at + i;
                draft[i] = idx < base.size() ? base[idx] : 0;
                if (round % 3 == 0 && i + 1 == k) draft[i] = (draft[i] + 1) % kVocab;  // 有意错一格
            }
            std::vector<std::uint64_t> got;
            CHECK(spec_step(g, m, e, draft.data(), k, s, d, got, err));
            CHECK(!got.empty());
            for (std::uint64_t t : got) produced.push_back(t);
            ++round;
        }
        CHECK(produced.size() >= base.size());
        for (std::size_t i = 0; i < base.size(); ++i) CHECK(produced[i] == base[i]);
        std::printf("  投机：%d 轮产出 %zu 个 token（基准 %zu 个，逐位一致）\n", round, produced.size(),
                    base.size());
        // 每次前向恰好对应一个已提交的 token（+1 是最初喂入起点那次），故投机不省前向总数，
        // 省的是「轮数」：轮均产出 15/6 = 2.5 个 token。
        CHECK(steps(s) == produced.size() + 1);
    }
}

// 专家缓存与每步观测接进步循环：缓存对数值透明，观测记的是本步增量与步末占用；给了需求就走仲裁器。
void test_cache_and_budget() {
    const std::filesystem::path p = write_gguf("qinfer_step_cache.gguf", model_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));

    StepConfig plain = two_layer_config();
    StepConfig cached = two_layer_config();
    cached.cache_slots = 4;
    cached.cache_ways = 4;

    StepModel mp, mc;
    CHECK(build_step_model(g, plain, mp, err));
    CHECK(build_step_model(g, cached, mc, err));
    CHECK(mp.blocks[0].moe.cache_slots == 0);   // 直读
    CHECK(mc.blocks[0].moe.cache_slots == 4);   // 缓存配置逐层套用
    CHECK(mc.blocks[1].moe.cache_slots == 4);

    const std::vector<float> x = make_embd();
    StepScratch sp, sc;
    CHECK(prepare_step_scratch(mp, 4, 0, sp, err));
    CHECK(prepare_step_scratch(mc, 4, 0, sc, err));
    CHECK(run_step_model(g, mp, x.data(), sp, err));
    CHECK(run_step_model(g, mc, x.data(), sc, err));
    // 缓存对数值透明：两条路径逐位一致。
    for (std::size_t i = 0; i < sp.head.logits.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(sp.head.logits[i]) ==
              std::bit_cast<std::uint32_t>(sc.head.logits[i]));
    }
    // 直读那条路没有缓存可观测 → 记录里缓存侧全 0。
    CHECK(sp.has_record);
    CHECK(sp.last_record.cache_hits == 0 && sp.last_record.cache_misses == 0);
    // 缓存那条路：第一步是首次填充（全未命中），步末有常驻。
    CHECK(sc.has_record);
    CHECK(sc.last_record.cache_misses == 4u);  // 2 层 × top-2 专家
    CHECK(sc.last_record.cache_hits == 0u);
    CHECK(sc.last_record.hit_rate == 0.0);
    CHECK(sc.last_record.residents == 4u && sc.last_record.bytes_resident > 0u);

    // 第二步：路由与上一步相同（路由器全 1 → top-2 恒为 {0,1}）→ 全部命中。
    CHECK(run_step_model(g, mc, x.data(), sc, err));
    CHECK(sc.last_record.cache_hits == 4u);
    CHECK(sc.last_record.cache_misses == 0u);
    CHECK(sc.last_record.hit_rate == 1.0);
    CHECK(sc.last_record.cache_evictions == 0u);

    // 只给一个槽位：同一个 token 内的第二个专家换不掉第一个 → 被拒 + 走临时缓冲，数值仍不变。
    StepConfig tiny = cached;
    tiny.cache_slots = 1;
    tiny.cache_ways = 1;
    StepModel mt;
    CHECK(build_step_model(g, tiny, mt, err));
    StepScratch st;
    CHECK(prepare_step_scratch(mt, 4, 0, st, err));
    CHECK(run_step_model(g, mt, x.data(), st, err));
    CHECK(st.last_record.cache_blocked >= 1u || st.last_record.cache_transient >= 1u);
    for (std::size_t i = 0; i < st.head.logits.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(st.head.logits[i]) ==
              std::bit_cast<std::uint32_t>(sp.head.logits[i]));
    }

    // 预算侧：给了需求且画像里有带宽 → 走仲裁器；必需项超出预算记超支，预测性项放不下记迟到。
    StepConfig budgeted = two_layer_config();
    budgeted.profile.pcie_h2d_gbps = 28.64;
    budgeted.profile.step_ms = 12.74;
    budgeted.profile.io_share = 0.5;
    StepModel mb;
    CHECK(build_step_model(g, budgeted, mb, err));
    StepScratch sb;
    CHECK(prepare_step_scratch(mb, 4, 0, sb, err));
    StepModel mm;
    scheduling::StepDemand demand;
    demand.step = 7;
    demand.kv_bytes = 400u * 1024 * 1024;  // 必需项远超一步预算（环境2 画像约 186.8 MiB）
    demand.table = scheduling::TableDemand{16, 90, 1.133};
    for (std::uint32_t e = 0; e < 4; ++e) {
        demand.experts.push_back(scheduling::ExpertCandidate{e, 0.5 - 0.1 * e, 200u * 1024 * 1024});
    }
    CHECK(run_step_model(g, mb, x.data(), sb, err, &demand));
    const experts::StepRecord& r = sb.last_record;
    CHECK(r.step == 7);
    CHECK(r.budget_bytes == scheduling::step_budget_bytes(budgeted.profile));
    CHECK(r.budget_bytes > 0u && r.admitted_bytes > 0u);
    CHECK(r.overspend_bytes > 0u);       // 必需项超预算
    CHECK(r.late_requests > 0u);         // 预测性项放不下
    CHECK(r.budget_used > 0.0);
    // 不给需求 → 预算那几项留 0（表示这一步没做仲裁），缓存侧照记。
    StepScratch s2;
    CHECK(prepare_step_scratch(mc, 4, 0, s2, err));
    CHECK(run_step_model(g, mc, x.data(), s2, err));
    CHECK(s2.last_record.budget_bytes == 0u && s2.last_record.budget_used == 0.0);
    CHECK(s2.last_record.cache_misses == 4u);
    (void) mm;
}

// 真模型手工入口：整模型跑几个 token，打印判型摘要与 logits 概况。
// 给了 `ple_table`（分片2，装着 `per_layer_token_embd.weight`）就再跑一条配上 PLE 的：那条按
// **真 token 序列**走 `run_token_step`（表行由当前 token 与前两个决定），并与不配那条比相对偏离。
// 给了 `kv_slots > 0` 就再跑一条**KV 换出/换入**的：每个 QSA 层挂一个小窗口 + 一份权威副本，与
// 「不挂（恒等、全常驻）」那条比逐位一致。`idx_top_k` 默认取真模型的值（2048），小值只用于让选择变
// 稀疏——真模型要到 2,051 个 cell 以上选择才稀疏，那在主机参考路径上跑不到，故这一段用小的值把
// 「驱逐」这条路在几十个 token 内打开（两条路用同一份配置，比较仍是同口径的）。
int manual_model_entry(const char* model, int tokens, const char* ple_table, int kv_slots,
                       std::uint64_t idx_top_k) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    // 真模型几何（[S-54]/[S-57]/[S-61]）：48 层、n_embd 2560、hc 4、hc_lr 320、top_k 10、
    // GDN 头宽 128 / QSA 头宽 256、n_rot 64、idx_dim 128、idx_top_k 2048、idx_block 4。
    StepConfig cfg;
    cfg.n_embd = 2560;
    cfg.hc = 4;
    cfg.hc_lr = 320;
    cfg.n_layers = 48;
    cfg.top_k = 10;
    cfg.gdn_head_dim = 128;
    cfg.qsa_head_dim = 256;
    cfg.n_rot = 64;
    cfg.idx_dim = 128;
    cfg.idx_top_k = 2048;
    cfg.idx_block = 4;
    StepModel m;
    if (!build_step_model(g, cfg, m, err)) {
        std::printf("建整模型失败：%s\n", err.c_str());
        return 1;
    }
    int n_gdn = 0, n_qsa = 0;
    for (const LayerMapRow& r : m.map) {
        if (r.kind == LayerKind::kGdn) ++n_gdn;
        if (r.kind == LayerKind::kQsa) ++n_qsa;
    }
    std::printf("48 层：GDN %d、QSA %d；词表 %llu\n", n_gdn, n_qsa,
                (unsigned long long) m.head.out.rows);

    StepScratch s;
    if (!prepare_step_scratch(m, static_cast<std::uint64_t>(tokens) + 2, /*pos_base=*/0, s, err)) {
        std::printf("预备 scratch 失败：%s\n", err.c_str());
        return 1;
    }
    std::vector<float> x(2560);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.05f * (float) ((int) (i % 17) - 8);
    }
    double mag = 0.0;
    bool finite = true;
    std::size_t argmax = 0;
    for (int t = 0; t < tokens; ++t) {
        if (!run_step_model(g, m, x.data(), s, err)) {
            std::printf("第 %d 个 token 跑不动：%s\n", t, err.c_str());
            return 1;
        }
    }
    for (std::size_t i = 0; i < s.head.logits.size(); ++i) {
        if (!std::isfinite(s.head.logits[i])) finite = false;
        mag += std::fabs((double) s.head.logits[i]);
        if (s.head.logits[i] > s.head.logits[argmax]) argmax = i;
    }
    std::printf("  %d 个 token：logits |值| 合计 %.6g、有限：%s、argmax %zu\n", tokens, mag,
                finite ? "是" : "否", argmax);

    if (ple_table == nullptr) return finite ? 0 : 1;

    // ---- 配上 PLE 的那一条（真表行的来源是分片2）----
    artifact::GgufTable table;
    if (!table.open(ple_table, err)) {
        std::printf("打开表（PLE）失败：%s\n", err.c_str());
        return 1;
    }
    ple::GgufTableRowSource src(table);
    storage::RowCache rows(4096);
    ple::PleSpec pspec;
    ple::PleBuffers pw;
    if (!ple::read_ple_spec(g, /*layer=*/1, 2560, 4, pspec, err) || !pspec.present ||
        !ple::load_ple_weights(g, pspec, pw, err)) {
        std::printf("读第 1 层的 PLE 登记/权重失败：%s\n", err.c_str());
        return 1;
    }
    PleRun run;
    run.layer = 1;
    run.spec = pspec;
    run.w = pw;
    run.table = &src;
    run.cache = &rows;
    StepConfig cfg_ple = cfg;
    cfg_ple.ple = &run;
    StepModel mp;
    if (!build_step_model(g, cfg_ple, mp, err)) {
        std::printf("建模型（含 PLE）失败：%s\n", err.c_str());
        return 1;
    }
    TokenEmbed te;
    if (!build_token_embed(g, 2560, te, err)) {
        std::printf("读 token embedding 失败：%s\n", err.c_str());
        return 1;
    }
    StepScratch sp;
    if (!prepare_step_scratch(mp, static_cast<std::uint64_t>(tokens) + 2, 0, sp, err)) {
        std::printf("预备 scratch（含 PLE）失败：%s\n", err.c_str());
        return 1;
    }
    double mag_ple = 0.0;
    bool finite_ple = true;
    std::size_t argmax_ple = 0;
    // 逐 token 的取表行（会话状态里的 prev + 推进时机）必须与「整段重算」那条给出**同一份向量**。
    // 这一条把 prev 的顺序与推进时机钉住，是合成夹具照不到的地方（那个模型的 n_embd 与表行向量宽
    // 不一致，走不了自动取表行那条）：整段那条用的是 `build_prev`，与会话状态完全独立。
    std::vector<std::int32_t> seq(static_cast<std::size_t>(tokens));
    for (int t = 0; t < tokens; ++t) seq[static_cast<std::size_t>(t)] = 1 + (t % 7);
    std::vector<float> whole(static_cast<std::size_t>(tokens) * ple::kVectorDim, 0.0f);
    {
        storage::RowCache cw(4096);
        ple::PathStats stw;
        if (!ple::ple_vectors_for_tokens(src, seq.data(), tokens, cw, 0, whole.data(), stw, err)) {
            std::printf("整段算表行失败：%s\n", err.c_str());
            return 1;
        }
    }
    int emb_same = 0, emb_total = 0;
    // 真表上跑：token 序列用重复的 1..tokens（前驱会随会话状态推进，故哈希的输入是变的）。
    for (int t = 0; t < tokens; ++t) {
        const std::uint64_t tok = static_cast<std::uint64_t>(1 + (t % 7));
        if (!run_token_step(g, mp, te, tok, sp, err)) {
            std::printf("（含 PLE）第 %d 个 token 跑不动：%s\n", t, err.c_str());
            return 1;
        }
        for (int i = 0; i < ple::kVectorDim; ++i) {
            ++emb_total;
            const float a = sp.ple.emb[static_cast<std::size_t>(i)];
            const float b = whole[static_cast<std::size_t>(t) * ple::kVectorDim + i];
            if (std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b)) ++emb_same;
        }
    }
    std::printf("  逐 token 取的表行向量 vs 整段重算：逐位相同 %d/%d\n", emb_same, emb_total);
    if (emb_same != emb_total) return 1;
    for (std::size_t i = 0; i < sp.head.logits.size(); ++i) {
        if (!std::isfinite(sp.head.logits[i])) finite_ple = false;
        mag_ple += std::fabs((double) sp.head.logits[i]);
        if (sp.head.logits[i] > sp.head.logits[argmax_ple]) argmax_ple = i;
    }
    std::printf("  含 PLE：logits |值| 合计 %.6g、有限：%s、argmax %zu；表行读 %llu 行、命中 %llu\n",
                mag_ple, finite_ple ? "是" : "否", argmax_ple,
                (unsigned long long) rows.misses(), (unsigned long long) rows.hits());
    // 与「不配 PLE、同一条 token 序列」比：那条也要按 token 走，否则比的是两份不同的输入。
    StepScratch s2;
    if (!prepare_step_scratch(m, static_cast<std::uint64_t>(tokens) + 2, 0, s2, err)) return 1;
    for (int t = 0; t < tokens; ++t) {
        const std::uint64_t tok = static_cast<std::uint64_t>(1 + (t % 7));
        if (!run_token_step(g, m, te, tok, s2, err)) {
            std::printf("（不含 PLE）第 %d 个 token 跑不动：%s\n", t, err.c_str());
            return 1;
        }
    }
    const double rel = rel_diff(s2.head.logits, sp.head.logits);
    std::printf("  PLE 开/关（同一条 token 序列）：logits 相对偏离 %.3g；PLE 的逐层同一步进\n", rel);
    if (kv_slots <= 0) return (finite && finite_ple && rel > 1e-3) ? 0 : 1;

    // ---- KV 换出/换入那一条：每个 QSA 层一个小窗口 + 一份权威副本 ----
    {
        StepConfig cfg_kv = cfg;
        cfg_kv.idx_top_k = idx_top_k;
        StepModel mk;
        if (!build_step_model(g, cfg_kv, mk, err)) {
            std::printf("建模型（含 KV 窗口）失败：%s\n", err.c_str());
            return 1;
        }
        TokenEmbed tek;
        if (!build_token_embed(g, 2560, tek, err)) {
            std::printf("读 token embedding（KV 那条）失败：%s\n", err.c_str());
            return 1;
        }
        const std::uint64_t blocks = static_cast<std::uint64_t>(tokens) / 4 + 2;  // 逻辑页数（够用即可）
        const std::uint64_t pages_needed = blocks;
        std::vector<KvWindow> wins(mk.map.size());
        std::vector<KvHostStore> stores(mk.map.size());
        StepScratch sw;
        if (!prepare_step_scratch(mk, pages_needed, 0, sw, err)) {
            std::printf("预备 scratch（KV 那条）失败：%s\n", err.c_str());
            return 1;
        }
        int attached = 0;
        for (std::size_t l = 0; l < mk.map.size(); ++l) {
            if (mk.map[l].kind != LayerKind::kQsa) continue;
            QsaScratch& q = sw.blocks[l].qsa;
            q.kv.n_pages = static_cast<std::uint64_t>(kv_slots);  // 槽位只给这么多
            q.kv.reset();
            stores[l].init(q.kv.page_size, q.kv.n_head_kv, q.kv.head_dim, blocks);
            wins[l].page_bytes = 2 * q.kv.n_head_kv * q.kv.page_size * q.kv.head_dim * 2;
            wins[l].quota_bytes = static_cast<std::uint64_t>(kv_slots) * wins[l].page_bytes;
            wins[l].store = &stores[l];
            attach_kv_window(q, wins[l], blocks);
            ++attached;
        }
        std::uint64_t ev = 0, moved = 0;
        for (std::size_t l = 0; l < mk.map.size(); ++l) {
            if (mk.map[l].kind != LayerKind::kQsa) continue;
            ev += wins[l].evictions;
            moved += stores[l].bytes_moved;
        }
        for (int t = 0; t < tokens; ++t) {
            const std::uint64_t tok = static_cast<std::uint64_t>(1 + (t % 7));
            if (!run_token_step(g, mk, tek, tok, sw, err)) {
                std::printf("（KV 窗口）第 %d 个 token 跑不动：%s\n", t, err.c_str());
                return 1;
            }
        }
        std::uint64_t ev2 = 0, moved2 = 0;
        for (std::size_t l = 0; l < mk.map.size(); ++l) {
            if (mk.map[l].kind != LayerKind::kQsa) continue;
            ev2 += wins[l].evictions;
            moved2 += stores[l].bytes_moved;
        }
        // 不挂窗口那条：同一份配置（含同一个 idx_top_k）、同一条 token 序列。
        StepScratch s_plain;
        if (!prepare_step_scratch(mk, pages_needed, 0, s_plain, err)) return 1;
        for (int t = 0; t < tokens; ++t) {
            const std::uint64_t tok = static_cast<std::uint64_t>(1 + (t % 7));
            if (!run_token_step(g, mk, tek, tok, s_plain, err)) {
                std::printf("（无窗口）第 %d 个 token 跑不动：%s\n", t, err.c_str());
                return 1;
            }
        }
        int same = 0;
        for (std::size_t i = 0; i < s_plain.head.logits.size() && i < sw.head.logits.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(s_plain.head.logits[i]) ==
                std::bit_cast<std::uint32_t>(sw.head.logits[i])) {
                ++same;
            }
        }
        std::printf("  KV 窗口：%d 个 QSA 层各挂 %d 槽位（逻辑 %llu 页）→ 驱逐 %llu 次、搬入 %llu 字节；"
                    "与不挂窗口逐位相同 %d/%zu\n",
                    attached, kv_slots, (unsigned long long) blocks, (unsigned long long) ev2,
                    (unsigned long long) moved2, same, s_plain.head.logits.size());
        (void) ev;
        (void) moved;
        CHECK(same == static_cast<int>(s_plain.head.logits.size()));
        CHECK(ev2 > 0);
        CHECK(moved2 > 0);
    }
    return (finite && finite_ple && rel > 1e-3) ? 0 : 1;
}

// ---- PLE（第 1 层那一步，[S-63]）----
//
// 合成替身表与行号口径同 test_ple_read_path：行数取真表的行数（哈希常量是按那个几何转录的，故行号
// 必然落在那个范围内），行字节只由行号决定，五块都设尺度。
std::uint16_t ple_scale_bits(std::uint32_t row) {
    return static_cast<std::uint16_t>(0x1000u + (row % 0x2000u));
}

struct PleSynthTable : ple::RowSource {
    std::uint64_t rows() const override { return 320001536ull; }
    bool read_row(std::uint32_t row, std::uint8_t* out) const override {
        const std::uint16_t sb = ple_scale_bits(row);
        std::memset(out, 0, kernels::kIq4nlRowBytes);
        for (int blk = 0; blk < 5; ++blk) {
            out[blk * 18] = static_cast<std::uint8_t>(sb & 0xFF);
            out[blk * 18 + 1] = static_cast<std::uint8_t>(sb >> 8);
        }
        return true;
    }
};

template <typename T>
std::vector<std::uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<std::uint8_t> out(v.size() * sizeof(T));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

std::vector<std::uint8_t> bf16_lin(std::uint64_t n, float base, float step) {
    std::vector<std::uint16_t> b(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(i)] =
            kernels::f32_to_bf16_bits(base + step * static_cast<float>(i));
    }
    return bytes_of(b);
}

std::vector<std::uint8_t> f16_lin(std::uint64_t n, float base, float step) {
    std::vector<std::uint16_t> b(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(i)] =
            kernels::f32_to_f16_bits(base + step * static_cast<float>(i));
    }
    return bytes_of(b);
}

// 第 1 层的六张 PLE 张力（形状按 ne 序：key 是 [n_embd, hc_dim]、卷积核是 [4, hc_dim]）。
std::vector<TensorSpec> ple_tensors() {
    const std::uint64_t hc_dim = kN * kHc;
    std::vector<TensorSpec> ts;
    ts.push_back({"blk.1.ple_key.weight", {kN, hc_dim}, 30, bf16_lin(kN * hc_dim, -0.02f, 0.00007f)});
    ts.push_back({"blk.1.ple_value.weight", {kN, kN}, 30, bf16_lin(kN * kN, 0.004f, 0.0001f)});
    ts.push_back({"blk.1.ple_norm_key.weight", {hc_dim}, 0, f32_lin(hc_dim, 1.0f)});
    ts.push_back({"blk.1.ple_norm_query.weight", {hc_dim}, 0, f32_lin(hc_dim, 0.7f)});
    ts.push_back({"blk.1.ple_norm_conv.weight", {hc_dim}, 0, f32_lin(hc_dim, 1.3f)});
    ts.push_back({"blk.1.ple_conv1d.weight", {4, hc_dim}, 1, f16_lin(4 * hc_dim, -0.03f, 0.0005f)});
    return ts;
}

std::vector<TensorSpec> model_tensors_with_ple() {
    std::vector<TensorSpec> ts = model_tensors();
    append(ts, ple_tensors());
    return ts;
}

// PLE 那一套的持有者：登记 + 常驻权重 + 挂进 StepConfig 的那份配置。表行来源**不给**：这条用例走
// 「调用方自己把 emb 填好」（引擎那侧的 `ple_stage_token` 与图里那一半也是分开的）。
struct PleRig {
    ple::PleSpec spec;
    ple::PleBuffers w;
    PleRun run;
    bool init(const artifact::GgufFile& g, int layer, std::string& err) {
        if (!ple::read_ple_spec(g, layer, kN, kHc, spec, err)) return false;
        if (!spec.present) {
            err = "这一层没有 PLE";
            return false;
        }
        if (!ple::load_ple_weights(g, spec, w, err)) return false;
        run.layer = layer;
        run.spec = spec;
        run.w = w;
        return true;
    }
};

// 填一条确定的 emb（宽度 = 模型的 n_embd；真模型上它由表行路径给出，长度 2560）。
void fill_ple_emb(StepScratch& s, int seed) {
    for (std::size_t i = 0; i < s.ple.emb.size(); ++i) {
        s.ple.emb[i] = 0.02f * static_cast<float>(static_cast<int>((i + seed * 7) % 23) - 11);
    }
    s.ple.has_token = true;
}

// 跑若干步（每步由调用方填 emb），返回最后一步的 logits 与每步之后的历史快照。
std::vector<float> run_ple_steps(const artifact::GgufFile& g, const StepModel& m,
                                 std::vector<std::vector<float>>* hist_log = nullptr) {
    std::string err;
    const std::vector<float> x = make_embd();
    StepScratch s;
    CHECK(prepare_step_scratch(m, 4, 0, s, err));
    for (int t = 0; t < 3; ++t) {
        fill_ple_emb(s, t);
        CHECK(run_step_model(g, m, x.data(), s, err));
        if (hist_log != nullptr) hist_log->push_back(s.ple.hist);
    }
    return s.head.logits;
}

void test_ple_step() {
    const std::filesystem::path p = write_gguf("qinfer_step_ple.gguf", model_tensors_with_ple());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    TokenEmbed te;
    CHECK(build_token_embed(g, kN, te, err));

    // 不配 PLE：那六张张力在文件里也**不该改变任何东西**（引擎那条「这一层没有 PLE 就整块跳过」
    // 的对称面：有张力但没配 PleRun，同样一步都不跑）。
    StepConfig off = two_layer_config();
    StepModel m_off;
    CHECK(build_step_model(g, off, m_off, err));
    const std::vector<float> base = run_ple_steps(g, m_off);

    const std::filesystem::path p0 = write_gguf("qinfer_step_nople.gguf", model_tensors());
    artifact::GgufFile g0;
    CHECK(g0.open(p0.string(), err));
    StepModel m0;
    CHECK(build_step_model(g0, two_layer_config(), m0, err));
    const std::vector<float> base0 = run_ple_steps(g0, m0);
    for (std::size_t i = 0; i < base.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(base[i]) == std::bit_cast<std::uint32_t>(base0[i]));
    }
    std::printf("  不配 PLE：带张力与不带张力两份模型逐位一致（整块跳过）\n");

    // 配上（第 1 层）：跑得动、logits 有限，且与不配那一条**明显不同**（PLE 不是空操作）。
    PleRig rig;
    CHECK(rig.init(g, /*layer=*/1, err));
    StepConfig on = two_layer_config();
    on.ple = &rig.run;
    StepModel m_on;
    CHECK(build_step_model(g, on, m_on, err));
    std::vector<std::vector<float>> hist;
    const std::vector<float> with_ple = run_ple_steps(g, m_on, &hist);
    CHECK(with_ple.size() == kVocab);
    bool finite = true;
    for (float v : with_ple) {
        if (!std::isfinite(v)) finite = false;
    }
    CHECK(finite);
    const double rel = rel_diff(base, with_ple);
    std::printf("  配上第 1 层的 PLE：logits 与不配那条相对偏离 %.3g\n", rel);
    CHECK(rel > 1e-3);

    // 历史确实在动：连续三步之后，每次的 9 行都是「左移一格 + 追加一行」（结构判据）。
    CHECK(hist.size() == 3u);
    for (std::size_t t = 1; t < hist.size(); ++t) {
        for (std::size_t c = 0; c < kN * kHc; ++c) {
            for (int r = 0; r + 1 < ple::kHistRows; ++r) {
                CHECK(hist[t][c * ple::kHistRows + static_cast<std::size_t>(r)] ==
                      hist[t - 1][c * ple::kHistRows + static_cast<std::size_t>(r) + 1]);
            }
            CHECK(hist[t][c * ple::kHistRows + ple::kHistRows - 1] !=
                  hist[t - 1][c * ple::kHistRows + ple::kHistRows - 1]);
        }
    }
    std::printf("  历史：9 行每次左移一格并追加（末行与上一次不同）\n");

    // 同一个 scratch 再跑一遍同一组 emb → 逐位一致；换一组 emb → 不同（表行向量确实进了这一层）。
    const std::vector<float> again = run_ple_steps(g, m_on);
    for (std::size_t i = 0; i < again.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(again[i]) == std::bit_cast<std::uint32_t>(with_ple[i]));
    }
    std::printf("  同一组 emb 两次独立跑：logits 逐位一致\n");

    // 挂错层：把 layer 改成 0（权重仍读的是第 1 层那六张）→ 在层 0 上就地写 R → logits 明显不同。
    PleRig wrong;
    CHECK(wrong.init(g, /*layer=*/1, err));
    wrong.run.layer = 0;
    StepConfig off_layer = two_layer_config();
    off_layer.ple = &wrong.run;
    StepModel m_wrong;
    CHECK(build_step_model(g, off_layer, m_wrong, err));
    const std::vector<float> at0 = run_ple_steps(g, m_wrong);
    std::printf("  挂到第 0 层（权重仍是第 1 层那六张）：与第 1 层那条相对偏离 %.3g\n",
                rel_diff(with_ple, at0));
    CHECK(rel_diff(with_ple, at0) > 1e-3);

    // 配了 PLE 但这一层没有喂 token（has_token 为假）→ 明确报错，不静默跳过。
    {
        const std::vector<float> x = make_embd();
        StepScratch s;
        CHECK(prepare_step_scratch(m_on, 4, 0, s, err));
        CHECK(!run_step_model(g, m_on, x.data(), s, err));
        CHECK(!err.empty());
        std::printf("  配了 PLE 却没喂 token：明确报错（%s）\n", err.c_str());
    }

    // 配了 PLE 又走 `run_token_step`（自动取表行）：这条合成模型的 n_embd 是 256，而表行向量固定
    // 2560 宽 → 明确失败，不截断不补零。（真模型的 n_embd 正是 2560，那条在 --model 入口上。）
    {
        PleRig staged;
        CHECK(staged.init(g, /*layer=*/1, err));
        PleSynthTable table;
        storage::RowCache cache(64);
        staged.run.table = &table;
        staged.run.cache = &cache;
        StepConfig cfg = two_layer_config();
        cfg.ple = &staged.run;
        StepModel ms;
        CHECK(build_step_model(g, cfg, ms, err));
        StepScratch s;
        CHECK(prepare_step_scratch(ms, 4, 0, s, err));
        CHECK(!run_token_step(g, ms, te, 1, s, err));
        CHECK(!err.empty());
        std::printf("  表行向量宽（2560）与 n_embd（256）不符：明确报错（%s）\n", err.c_str());
    }

    // 表行那条路径的两个入口（单 token 带前驱 / 整段重算）必须给出同一份向量——用替身表直接核，
    // 不经过模型（真几何：16 头 × 160 宽）。
    {
        PleSynthTable table;
        storage::RowCache c1(64);
        ple::PathStats st;
        const std::int32_t seq[3] = {1, 2, 3};
        std::vector<float> seq_out(3 * ple::kVectorDim, 0.0f);
        CHECK(ple::ple_vectors_for_tokens(table, seq, 3, c1, 0, seq_out.data(), st, err));
        const std::int32_t prevs[3][2] = {{-1, -1}, {-1, 1}, {1, 2}};
        for (int t = 0; t < 3; ++t) {
            storage::RowCache c2(64);
            ple::PathStats st1;
            std::vector<float> one(ple::kVectorDim, 0.0f);
            CHECK(ple::ple_vector_with_prev(table, seq[t], prevs[t][0], prevs[t][1], c2, 0,
                                            one.data(), st1, err));
            for (int i = 0; i < ple::kVectorDim; ++i) {
                CHECK(one[static_cast<std::size_t>(i)] ==
                      seq_out[static_cast<std::size_t>(t) * ple::kVectorDim +
                              static_cast<std::size_t>(i)]);
            }
        }
        std::printf("  表行：单 token（带前驱）与整段重算逐位一致\n");

        // 前驱的**顺序**是有意义的（哈希按位置取不同的乘子）：两个前驱对调必须是另一组行号。
        // 这条把「接线把 prev[0]/prev[1] 传反了」这类错钉在行号这一级——真权重入口上也照得住。
        std::vector<float> a(ple::kVectorDim, 0.0f), b(ple::kVectorDim, 0.0f);
        storage::RowCache ca(64), cb(64);
        ple::PathStats sa, sb;
        CHECK(ple::ple_vector_with_prev(table, 5, 3, 9, ca, 0, a.data(), sa, err));
        CHECK(ple::ple_vector_with_prev(table, 5, 9, 3, cb, 0, b.data(), sb, err));
        int diff = 0;
        for (int i = 0; i < ple::kVectorDim; ++i) {
            if (a[static_cast<std::size_t>(i)] != b[static_cast<std::size_t>(i)]) ++diff;
        }
        std::printf("  表行：两个前驱对调后 %d/%d 个值不同（顺序有意义）\n", diff, ple::kVectorDim);
        CHECK(diff > 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int tokens = 1;
        int kv_slots = 0;
        std::uint64_t idx_top_k = 2048;
        const char* ple_table = nullptr;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) {
                tokens = std::atoi(argv[++i]);
            } else if (a == "--ple" && i + 1 < argc) {
                ple_table = argv[++i];
            } else if (a == "--kv-slots" && i + 1 < argc) {
                kv_slots = std::atoi(argv[++i]);
            } else if (a == "--idx-top-k" && i + 1 < argc) {
                idx_top_k = static_cast<std::uint64_t>(std::atoi(argv[++i]));
            } else if (a != argv[2]) {
                std::printf("不认识的参数：%s\n", a.c_str());
                return 2;
            }
        }
        return manual_model_entry(argv[2], tokens, ple_table, kv_slots, idx_top_k);
    }
    test_build_and_run();
    test_order_and_layers_matter();
    test_state_advances();
    test_failures();
    test_token_generation();
    test_spec_decode();
    test_cache_and_budget();
    test_ple_step();
    std::puts("step_loop: 按层表串起 GDN+QSA 两层并出 logits；层序对调 / 少跑一层 / 不跑循环都可观测，"
              "逐层状态各归各层，token→embedding→logits→贪心的最小生成环可复现，三种接错都明确报错；"
              "投机解码接受/拒绝只改步数、不改序列");
    return 0;
}
