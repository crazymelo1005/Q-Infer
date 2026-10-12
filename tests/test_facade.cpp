// 装配门面（`serve::Engine`）的回归。不引第三方框架。
//
// 合成夹具与 tests/test_step_loop.cpp 同款（同一个两层的极小型整套模型：层 0 GDN、层 1 QSA、词表 8），
// 唯一改动是 `write_gguf` 多收一份元数据：分词器语料（3 个特殊 token + 字母 a..e）、eos、chat 模板。
//
// 钉子：
//   1. 打不开的文件 / 缺 chat 模板 → open 明确失败；
//   2. 正常打开：模型名、词表大小、模板就位；generate 从 prompt 一路贪心到长度或 eos；回调逐 token 交出
//      解码字节且与 res.text 一致；
//   3. seed 可复现；不同 seed 的采样式分歧；stop 串从不出现/尾 piece 命中两种口径；
//   4. 参数校验（max_tokens 0）明确报错；
//   5. 挂了 KV 窗口的那台与不挂窗口的那台逐 token、逐字节一致，且窗口侧有驱逐与未命中；
//   6. PLE 错误路径：ple_table 指向没有 PLE 六张张力的 GGUF → open 明确失败（err 含 PLE）。
#include "serve/facade.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include "kernels/fp16.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;
using namespace qinfer::serve;

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

struct MetaKV {
    std::string key;
    std::uint32_t type;             // GGUF 类型码：4=UINT32、8=STRING、9=ARRAY
    std::vector<std::uint8_t> val;  // 值字节（ARRAY 含元素类型与计数，不含值类型码）
};

std::vector<std::uint8_t> meta_u32(std::uint64_t x) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>((x >> (8 * i)) & 0xFF));
    return v;
}
std::vector<std::uint8_t> meta_str(const std::string& s) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((s.size() >> (8 * i)) & 0xFF));
    v.insert(v.end(), s.begin(), s.end());
    return v;
}
std::vector<std::uint8_t> meta_str_array(const std::vector<std::string>& a) {
    std::vector<std::uint8_t> v;
    const std::vector<std::uint8_t> elem = meta_u32(8);  // 元素类型 STRING（LE u32）
    v.insert(v.end(), elem.begin(), elem.end());
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((a.size() >> (8 * i)) & 0xFF));
    for (const std::string& s : a) {
        const std::vector<std::uint8_t> m = meta_str(s);
        v.insert(v.end(), m.begin(), m.end());
    }
    return v;
}
std::vector<std::uint8_t> meta_u32_array(const std::vector<std::uint64_t>& a) {
    std::vector<std::uint8_t> v;
    const std::vector<std::uint8_t> elem = meta_u32(4);  // 元素类型 UINT32（LE u32）
    v.insert(v.end(), elem.begin(), elem.end());
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((a.size() >> (8 * i)) & 0xFF));
    for (std::uint64_t x : a) {
        const std::vector<std::uint8_t> m = meta_u32(x);
        v.insert(v.end(), m.begin(), m.end());
    }
    return v;
}

// 合成夹具的分词器语料：3 个特殊 token + 字母 a..e。词表 8 = token_embd 的行数。
std::vector<MetaKV> serve_metas() {
    return {
        {"general.alignment", 4, meta_u32(32)},
        {"general.name", 8, meta_str("qinfer-serve-fixture")},
        {"tokenizer.ggml.model", 8, meta_str("gpt2")},
        {"tokenizer.ggml.tokens", 9,
         meta_str_array({"<|im_start|>", "<|im_end|>", "<|endoftext|>", "a", "b", "c", "d", "e"})},
        // merges 不能为空：现读取器对空数组返回「不存在」（meta_strings 看 strs.empty()）。给一条
        // 引用词表内 token 的最小规则即可，不新增词条（init 只填合并秩，词表保持 8）。
        {"tokenizer.ggml.merges", 9, meta_str_array({"a b"})},
        {"tokenizer.ggml.token_type", 9, meta_u32_array({3, 3, 3, 1, 1, 1, 1, 1})},
        {"tokenizer.ggml.eos_token_id", 4, meta_u32(2)},
        {"tokenizer.chat_template", 8,
         meta_str("{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}")},
    };
}

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts,
                                 const std::vector<MetaKV>& metas) {
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
    u64(metas.size());
    for (const MetaKV& m : metas) {
        put_str(m.key);
        u32(m.type);
        b.insert(b.end(), m.val.begin(), m.val.end());
    }
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
    // output.weight 的 8 行取同一行（种子 5 的第 0 行）：8 个 logits 逐位相等，温度 1 的采样在词表上
    // 均匀展开、seed 不同必然分歧。原夹具此行是尖峰分布（logits 极差约 16），top_p 折叠成单 token，
    // 任何 seed 都采出同一串——「温度 1 两 seed 分歧」那条就钉不住。贪心不变（并列取小索引 → 恒 token 0，
    // 不撞 eos；长度到 24 仍成立）。
    const std::vector<std::uint8_t> row0 = q8_0_rows(1, kN, 5);
    std::vector<std::uint8_t> flat;
    for (std::uint64_t r = 0; r < kVocab; ++r) flat.insert(flat.end(), row0.begin(), row0.end());
    ts.push_back({"output.weight", {kN, kVocab}, 8, flat});
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

dense::StepConfig two_layer_config() {
    dense::StepConfig cfg;
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

}  // namespace

int main() {
    const std::filesystem::path p = write_gguf("qinfer_facade.gguf", model_tensors(), serve_metas());
    // 打不开的文件 → 明确错误。
    {
        Engine e; std::string err; FacadeConfig cfg; cfg.model_path = "/nonexistent/qinfer.gguf";
        CHECK(!Engine::open(cfg, e, err)); CHECK(!err.empty());
    }
    // 缺 chat 模板 → 拒绝启动（拒绝裸拼接）。
    {
        std::vector<MetaKV> ms;
        for (const MetaKV& m : serve_metas())
            if (m.key != "tokenizer.chat_template") ms.push_back(m);
        const std::filesystem::path q = write_gguf("qinfer_facade_notpl.gguf", model_tensors(), ms);
        Engine e; std::string err; FacadeConfig cfg; cfg.model_path = q.string();
        CHECK(!Engine::open(cfg, e, err));
        CHECK(err.find("chat_template") != std::string::npos);
    }
    // 正常打开：模型名来自 general.name、词表 8、模板就位。
    Engine eng; std::string err;
    FacadeConfig cfg; cfg.model_path = p.string();
    cfg.n_embd = 256; cfg.hc = 2; cfg.hc_lr = 3; cfg.n_layers = 2; cfg.top_k = 2;
    cfg.gdn_head_dim = 32; cfg.qsa_head_dim = 32; cfg.n_rot = 4;
    cfg.idx_dim = 4; cfg.idx_top_k = 2048; cfg.idx_block = 4;
    CHECK(Engine::open(cfg, eng, err));
    CHECK(eng.model_name() == "qinfer-serve-fixture");
    CHECK(eng.tokenizer().vocab_size() == 8u);
    CHECK(!eng.chat_template().empty());

    // 长度到：贪心 24 token。
    GenOptions g; g.temperature = 0.0f; g.seed = 7; g.max_tokens = 24;
    GenResult a; std::vector<std::string> pieces;
    auto collect = [&](const std::string& s) { pieces.push_back(s); return true; };
    CHECK(eng.generate({0, 1}, g, collect, a, err));
    CHECK(a.n_prompt == 2); CHECK(a.n_gen >= 4); CHECK(a.n_gen <= 24);
    CHECK(pieces.size() == a.n_gen); CHECK(a.tokens.size() == a.n_gen);
    if (a.finish == Finish::kLength) CHECK(a.n_gen == 24);
    for (std::int32_t t : a.tokens) CHECK(t >= 0 && t < 8);
    std::string joined;
    for (const std::string& s : pieces) joined += s;
    CHECK(joined == a.text); CHECK(!a.tokens.empty());
    std::printf("生成 %llu token（finish %d）：text %zu 字节\n", (unsigned long long) a.n_gen,
                (int) a.finish, a.text.size());

    // seed 可复现：同 seed 同参数 → 同一串 token 与文本。
    {
        GenResult b;
        CHECK(eng.generate({0, 1}, g, nullptr, b, err));
        CHECK(b.tokens == a.tokens); CHECK(b.text == a.text);
    }
    // 采样式（温度 1、seed 1/2）在长跑里必然分歧（夹具固定，结果确定）。
    {
        GenOptions s1 = g, s2 = g;
        s1.temperature = 1.0f; s2.temperature = 1.0f; s1.seed = 1; s2.seed = 2;
        s1.max_tokens = 256; s2.max_tokens = 256;
        GenResult c1, c2;
        CHECK(eng.generate({0, 1}, s1, nullptr, c1, err));
        CHECK(eng.generate({0, 1}, s2, nullptr, c2, err));
        std::size_t nd = 0;
        for (std::size_t i = 0; i < c1.tokens.size() && i < c2.tokens.size(); ++i)
            if (c1.tokens[i] != c2.tokens[i]) ++nd;
        CHECK(nd > 0 || c1.n_gen < 4 || c2.n_gen < 4);
        std::printf("温度 1 两 seed：%zu/%zu 处不同\n", nd,
                    c1.tokens.size() < c2.tokens.size() ? c1.tokens.size() : c2.tokens.size());
    }
    // stop 串：从不出现的 → 与不设 stop 完全一致；尾 piece 命中的 → 截到命中处。
    {
        GenOptions go; go.temperature = 0.0f; go.seed = 7; go.max_tokens = 64;
        GenResult full;
        CHECK(eng.generate({0, 1}, go, nullptr, full, err));
        CHECK(full.n_gen >= 8);
        GenOptions gz = go; gz.stop = {"zz"};
        GenResult z;
        CHECK(eng.generate({0, 1}, gz, nullptr, z, err));
        CHECK(z.tokens == full.tokens);
        GenOptions gs = go; gs.stop = {full.text.substr(full.text.size() - 2)};
        GenResult s;
        CHECK(eng.generate({0, 1}, gs, nullptr, s, err));
        CHECK(s.finish == Finish::kStopString);
        CHECK(s.n_gen <= full.n_gen);
        CHECK(s.text == full.text.substr(0, s.text.size()));
        std::printf("stop 命中：截到 %zu 字节（全量 %zu）\n", s.text.size(), full.text.size());
    }
    // 参数校验：max_tokens 0 → 报错。
    {
        GenOptions bad = g; bad.max_tokens = 0;
        GenResult r;
        CHECK(!eng.generate({0, 1}, bad, nullptr, r, err)); CHECK(!err.empty());
    }
    // KV 窗口对拍（Task 4 的另一半钉在这里）：同一份几何（idx_top_k 取 8 让选择稀疏），
    // kv_slots=13 那台与不挂窗口那台逐 token 逐位一致；窗口侧驱逐 > 0。
    // 数值由来：宽度 = idx_top_k + idx_block - 1 = 11 → 单步最多 11 个聚集页 + 1 个写入页
    // = 12 个钉住页 < 配额 13 → 永不拒绝；会话写到的逻辑页 = ceil(66/4) = 17 > 13 → 必有驱逐。
    {
        FacadeConfig cw = cfg; cw.kv_slots = 13; cw.idx_top_k = 8;
        FacadeConfig cp = cfg; cp.idx_top_k = 8;
        Engine ew, ep; std::string errw;
        CHECK(Engine::open(cw, ew, errw));
        CHECK(Engine::open(cp, ep, err));
        GenOptions gk; gk.temperature = 0.0f; gk.seed = 7; gk.max_tokens = 64;
        GenResult rw, rp;
        CHECK(ew.generate({0, 1}, gk, nullptr, rw, errw));
        CHECK(ep.generate({0, 1}, gk, nullptr, rp, err));
        CHECK(rw.tokens == rp.tokens);
        CHECK(rw.text == rp.text);
        CHECK(ew.kv_evictions() > 0);
        CHECK(ew.kv_misses() > 0);
        std::printf("KV 窗口：%llu token 与不挂窗口逐位一致；驱逐 %llu、未命中 %llu\n",
                    (unsigned long long) rw.tokens.size(), (unsigned long long) ew.kv_evictions(),
                    (unsigned long long) ew.kv_misses());
    }
    // PLE 错误路径：给了 ple_table 但那份 GGUF 没有 PLE 的六张张力 → open 明确失败。
    {
        FacadeConfig cp = cfg; cp.ple_table = p.string();   // 拿模型自己当表：没有 blk.1.ple_*
        Engine ep; std::string errp;
        CHECK(!Engine::open(cp, ep, errp));
        CHECK(errp.find("PLE") != std::string::npos);
        std::printf("PLE 错误路径：%s\n", errp.c_str());
    }
    std::puts("facade: 通过");
    return 0;
}