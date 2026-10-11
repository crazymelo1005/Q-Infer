// 顶层步循环：把「一层的组合」按层表串成整个模型的一个 decode 步，两端接上残差栈的初值与输出头。
//
// 依据 [S-60]/[S-61]：48 层是**每层一个图**，层间只传那个 hc 条流的栈 R；PLE 挂在 `block_layer_pre`
// 上、缺张力时整个跳过（故本实现把它留成可选的一步，先不含）；LM head 在循环之外，由调用方单独计时。
//
//   R = 广播(embedding)                       （[S-61]：真序列首个 token 就是 embedding 广播到每条流）
//   for l in 0..n_layers-1:  R = run_block_layer(l)(R)
//   logits = output @ gr_read(R, output_hc_*, inject=None)
//
// 每个**有状态**的层各持自己的一份状态：GDN 的循环/卷积状态与 QSA 的分页 KV 都是逐层的，不能共用
// 一份 scratch（共用会把第 l 层的状态当成第 l+1 层的）。故 `StepScratch` 每层一份 `BlockScratch`。
//
// 本节是主机侧参考路径：不做批、不做 KV 搬运（分页 KV 的换入换出属于存储层）、不含 MTP。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/block_layer.hpp"
#include "dense/layer_map.hpp"
#include "dense/model_head.hpp"
#include "dense/token_embed.hpp"
#include "experts/step_record.hpp"
#include "scheduling/step_budget.hpp"
#include "storage/expert_cache.hpp"

namespace qinfer::dense {

// 建整模型所需的几何与超参。它们解不出唯一值（见 [S-54]/[S-57]），故由调用方给；两种注意力的头宽不同
// （真模型 GDN 128 / QSA 256）。
struct StepConfig {
    std::uint64_t n_embd = 0;
    std::uint64_t hc = 0;
    std::uint64_t hc_lr = 0;
    int n_layers = 0;
    int top_k = 0;
    std::uint64_t gdn_head_dim = 0;
    std::uint64_t qsa_head_dim = 0;
    std::uint64_t n_rot = 0;
    std::uint64_t idx_dim = 0;
    std::uint64_t idx_top_k = 0;
    std::uint64_t idx_block = 0;

    // 专家缓存（engine §5）：`cache_slots == 0` 表示按需直读（与最早的实现一致）。逐层套用。
    std::size_t cache_slots = 0;
    int cache_ways = 8;
    std::uint64_t cache_byte_budget = 0;
    qinfer::storage::PolicyKind cache_policy = qinfer::storage::PolicyKind::kLru;

    // 每步 PCIe 预算的画像输入。`pcie_h2d_gbps == 0` 表示这一步不做预算仲裁（观测记 0）。
    scheduling::ProfileInputs profile;

    // 可选的「把未命中专家那批交出去」（interfaces §8）：给了就**逐层**交给它指定的候选组。
    // 调用方按层的档建好每层一份（长度须等于 n_layers），每层一份字节来源；不给就还是逐专家的
    // 主机路径。两条路的归并顺序相同，故逐位一样（见 experts/moe_layer.hpp 的 MoeOffload）。
    const std::vector<experts::MoeOffload>* offload_layers = nullptr;
};

struct StepModel {
    StepConfig cfg;
    std::vector<LayerMapRow> map;     // 逐层判型（长度 = n_layers）
    std::vector<BlockLayer> blocks;   // 按层号；两类注意力各自建好
    ModelHead head;
};

// 逐层判型 → 按该层类型建 `BlockLayer`（头宽按类型取）→ 建输出头。任一层建不起来都返回假并写 err。
bool build_step_model(const artifact::GgufFile& gguf, const StepConfig& cfg, StepModel& out,
                      std::string& err);

// 每层一份 scratch（各自的 GDN 状态 / QSA 分页 KV / MoE scratch），加输出头与残差栈。
struct StepScratch {
    std::vector<BlockScratch> blocks;
    HeadScratch head;
    std::vector<float> R;     // hc·n_embd
    std::vector<float> embd;  // n_embd，`run_token_step` 读进来的那一行 token embedding
    bool ready = false;       // QSA 层的 KV 与 rope 是否已配好（未配则 run 明确报错）

    // 每步观测（interfaces §7 的预算与缓存两侧）。`last_record` 由 `run_step_model` 每次覆写。
    experts::StepRecord last_record;
    bool has_record = false;
};

// 给每层 QSA 配分页 KV（`n_pages` 个物理页、恒等页表）与 rope 表；`pos_base` 是序列首个 cell 的位置
// （[S-59]：不能由 cell 号反推）。真模型上页的换入换出属于存储层，这里只给恒等页表。
bool prepare_step_scratch(const StepModel& model, std::uint64_t n_pages, std::uint64_t pos_base,
                          StepScratch& out, std::string& err);

// 跑一个 token：`x_embd` 长 n_embd（残差栈的初值来源）。logits 在 `s.head.logits`（长 = 词表）。
// `demand` 非空且 `cfg.profile.pcie_h2d_gbps > 0` 时，本步经仲裁器记一次预算观测（interfaces §7）；
// 否则预算那几项记 0（预算的**预测性**专家需求来自共现与会话亲和，本仓库尚无该来源）。
// 无论走不走预算，缓存侧的观测都会写进 `s.last_record`。
bool run_step_model(const artifact::GgufFile& gguf, const StepModel& model, const float* x_embd,
                    StepScratch& s, std::string& err,
                    const scheduling::StepDemand* demand = nullptr);

// 便捷入口：从 token id 读 embedding 行（`token_embd.weight`）再跑一步。等价于先
// `read_token_embedding` 到 `s.embd` 再 `run_step_model(..., s.embd.data(), ...)`。
bool run_token_step(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
                    std::uint64_t token, StepScratch& s, std::string& err);

}  // namespace qinfer::dense
