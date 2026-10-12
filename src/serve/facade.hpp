// 装配门面：把 GGUF + 分词器 + token embedding + 步模型 +（可选的）PLE 与 KV 窗口装配成一个可生成的
// 服务实体。一次 `generate` = 一份全新会话（prompt 自足，请求之间无共享状态）。
// 装配口径与真权重对拍入口（tests/test_step_loop.cpp 的 --model 路径）一致：同一组几何、同一组调用序。
// 几何缺省即真模型推导值（48 层那组），合成测试用 FacadeConfig 覆盖成小值。
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/kv_store.hpp"
#include "dense/kv_window.hpp"
#include "dense/step_loop.hpp"
#include "dense/token_embed.hpp"
#include "ple/row_path.hpp"
#include "ple/spec.hpp"
#include "serve/prompt.hpp"
#include "serve/sampler.hpp"
#include "storage/row_cache.hpp"
#include "text/tokenizer.hpp"

namespace qinfer::serve {

struct FacadeConfig {
    std::string model_path;          // 必填（主案例 GGUF 分片）
    std::uint64_t n_embd = 2560;
    std::uint64_t hc = 4;
    std::uint64_t hc_lr = 320;
    int n_layers = 48;
    int top_k = 10;
    std::uint64_t gdn_head_dim = 128;
    std::uint64_t qsa_head_dim = 256;
    std::uint64_t n_rot = 64;
    std::uint64_t idx_dim = 128;
    std::uint64_t idx_top_k = 2048;
    std::uint64_t idx_block = 4;
    std::uint64_t max_context = 262144;      // 报告用（/v1/models、/props）
    std::size_t cache_slots = 0;             // 专家缓存槽位；0 = 按需直读
    std::uint64_t cache_byte_budget = 0;
    std::string ple_table;                   // PLE 表行来源（分片2），空 = 不配 PLE
    std::uint64_t ple_budget_bytes = 0;
    std::uint64_t kv_slots = 0;              // 每 QSA 层常驻槽位；0 = 不挂 KV 窗口
};

// 结束原因：长度到 / eos token / stop 串命中。客户端中断（on_token 返回假）→ generate 返回假、err 说明。
enum class Finish { kLength, kStop, kStopString };

struct GenOptions {
    float temperature = 0.8f;
    float top_p = 0.95f;
    std::int64_t seed = -1;
    std::uint64_t max_tokens = 512;
    std::vector<std::string> stop;   // 生成的字节流出现这些子串即终止（截到命中处）
};

struct GenResult {
    Finish finish = Finish::kLength;
    std::uint64_t n_prompt = 0;
    std::uint64_t n_gen = 0;
    std::string text;                   // 全量已生成文本（UTF-8 字节，已去掉 stop 命中段）
    std::vector<std::int32_t> tokens;   // 已生成 token id（不含被 stop 截掉的后缀）
};

class Engine {
public:
    // 装载：GGUF 打开、分词器、chat 模板（缺失即失败——拒绝裸拼接）、token embedding、步模型，
    // 以及（给了 ple_table 时）第 1 层 PLE 的登记/权重/表行来源。任一步失败都返回假并写 err。
    static bool open(const FacadeConfig& cfg, Engine& out, std::string& err);

    // 同步生成。prompt 是已编码的 token id（调用方按 chat 模板渲染 + 分词）。逐 token 回调 on_token
    // = 该 token 解码后的字节片段（stop 命中时只回调保留前缀，前缀为空则免回调；eos 不回调节点）。
    // 回调返回 false = 客户端中断，generate 立即返回假、err 说明。
    bool generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                  const std::function<bool(const std::string&)>& on_token, GenResult& res,
                  std::string& err);

    const text::Tokenizer& tokenizer() const { return tokenizer_; }
    const std::string& chat_template() const { return chat_template_; }
    const std::string& model_name() const { return model_name_; }
    const FacadeConfig& config() const { return cfg_; }
    // 上次 generate 的 KV 窗口统计（未配窗口时全为 0）。只读标量，不碰窗口内部指针。
    std::uint64_t kv_misses() const;
    std::uint64_t kv_evictions() const;
    std::uint64_t kv_bytes_moved() const;

private:
    // KV 窗口接线（cfg_.kv_slots > 0 时由 generate 每次装配）：与真权重入口同一套字段公式。
    void attach_kv_windows(dense::StepScratch& s, std::uint64_t logical_blocks);

    FacadeConfig cfg_;
    artifact::GgufFile g_;
    artifact::GgufTable table_;              // PLE 行来源（可空）
    ple::GgufTableRowSource src_{table_};
    storage::RowCache rows_{0};
    ple::PleSpec pspec_;
    ple::PleBuffers pw_;
    dense::PleRun run_{};
    dense::StepConfig step_cfg_;
    dense::StepModel m_;
    dense::TokenEmbed te_;
    text::Tokenizer tokenizer_;
    std::string chat_template_;
    std::string model_name_;
    std::int32_t eos_ = -1;
    std::vector<dense::KvWindow> wins_;      // 每次 generate 重装；生存期 = generate（表/副本指针指向
    std::vector<dense::KvHostStore> stores_; // 当次 scratch，结束后不得再碰指针成员）
};

}  // namespace qinfer::serve