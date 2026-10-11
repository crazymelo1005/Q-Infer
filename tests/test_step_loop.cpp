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
#include "dense/step_loop.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
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

// 两层的小模型：层 0 GDN、层 1 QSA。两层的 hc 权重不同（norm 基值 1.0），靠层类型差异把「层序对调」暴露。
std::vector<TensorSpec> model_tensors() {
    std::vector<TensorSpec> ts = layer_tensors("blk.0.", /*gdn=*/true, 1.0f);
    append(ts, layer_tensors("blk.1.", /*gdn=*/false, 2.0f));
    append(ts, head_tensors());
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

// 真模型手工入口：整模型跑几个 token，打印判型摘要与 logits 概况。
int manual_model_entry(const char* model, int tokens) {
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
    return finite ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int tokens = 1;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--tokens") tokens = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], tokens);
    }
    test_build_and_run();
    test_order_and_layers_matter();
    test_state_advances();
    test_failures();
    std::puts("step_loop: 按层表串起 GDN+QSA 两层并出 logits；层序对调 / 少跑一层 / 不跑循环都可观测，"
              "逐层状态各归各层，三种接错都明确报错");
    return 0;
}
