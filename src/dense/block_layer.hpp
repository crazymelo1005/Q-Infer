// 一层的组合：把门控残差、注意力（GDN 或 QSA）、MoE 三块按参考引擎的顺序接起来。
//
// 顺序取自 [S-60]（`block_layer_pre` / `block_layer_post`）：
//
//   pre :  gr_read(R, hc_attn_*) → mixer（该层的注意力：GDN 或 QSA）→ gr_write(R)
//   post:  gr_read(R, hc_ffn_*)  → MoE（路由 → 逐专家前馈 + 共享专家 → 合并）→ gr_write(R)
//
// 两处要点：**每层两次门控残差**（attn 后一次、ffn 后一次），且 `gr_write` 是就地改写 R（hc 条流的栈）；
// `gr_read` 的 inject 只在同一段内用（S-60 说两段之间要原样留住——本实现顺序调用，天然满足）。
// MoE 那一半用 `run_moe_layer`：它已经含共享专家与逐专家的加权累加（[S-50]）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/attn_phases.hpp"
#include "dense/dense_exec.hpp"
#include "dense/gated_residual.hpp"
#include "dense/gdn_layer.hpp"
#include "dense/layer_map.hpp"
#include "dense/qsa_layer.hpp"
#include "experts/moe_layer.hpp"
#include "ple/block.hpp"
#include "ple/row_path.hpp"
#include "ple/spec.hpp"

namespace qinfer::dense {

// 第 1 层那一步的可选配置（引擎 `PleRun` 的对应物，见 [S-63]）：六张张力的登记与常驻权重，加表行来源。
// 由调用方建好、用指针挂进 `StepConfig`（与 `offload_layers` 是同一套做法）；不给或那一层没有六张张力
// 就整块跳过。`layer` 是它挂在哪一层（引擎的 `ple.layers = [1]` 说的是第 1 层）。
//
// 表行来源是**可选**的：给了就由 `run_token_step` 自动按当前 token 取表行（引擎的 `ple_stage_token`），
// 不给则由调用方自己把 `PleState::emb` 填好、`has_token` 置真再跑——引擎那侧也是这么分两半的
// （`ple_issue_token` / `ple_finish_token` 在驱动侧，图里只有 `ple_block`）。
struct PleRun {
    int layer = -1;
    ple::PleSpec spec;
    ple::PleBuffers w;
    ple::RowSource* table = nullptr;
    storage::RowCache* cache = nullptr;
    std::uint64_t budget_bytes = 0;
    bool ready() const { return layer >= 0 && spec.usable(); }
    bool can_stage() const { return table != nullptr && cache != nullptr; }
};

// PLE 的会话侧状态：9 行归一化历史（行最快）、当前 token 的表行向量、以及它的两个前驱（最老在前）。
// 历史是**跨 token** 的（卷积要看的正是它），故它跟着 scratch 走而不是跟着层走。
struct PleState {
    std::vector<float> hist;
    std::vector<float> emb;
    std::int32_t prev[2] = {kernels::kTokenNull, kernels::kTokenNull};
    bool has_token = false;
};

// 一层的静态描述：判型 + 两组 hc_* 张力 + 该走哪条注意力路径 + MoE。
struct BlockLayer {
    int layer = -1;
    LayerKind kind = LayerKind::kUnknown;
    GrShapes gr;
    std::uint64_t n_embd = 0;
    const artifact::GgufTensorInfo* a_norm = nullptr;    // hc_attn_norm（F32）
    const artifact::GgufTensorInfo* a_down = nullptr;    // hc_attn_down（BF16）
    const artifact::GgufTensorInfo* a_up = nullptr;      // hc_attn_up（BF16）
    const artifact::GgufTensorInfo* a_inject = nullptr;  // hc_attn_inject（BF16）
    const artifact::GgufTensorInfo* f_norm = nullptr;    // hc_ffn_norm（F32）
    const artifact::GgufTensorInfo* f_down = nullptr;
    const artifact::GgufTensorInfo* f_up = nullptr;
    const artifact::GgufTensorInfo* f_inject = nullptr;
    GdnLayer gdn;  // kind == kGdn 时有效
    QsaLayer qsa;  // kind == kQsa 时有效
    experts::MoeLayer moe;
};

// 几何与超参由调用方给（它们解不出唯一值，见 [S-54]/[S-57]）：n_embd / hc / hc_lr / top_k /
// head_dim / n_rot / idx_dim / idx_top_k / idx_block。判型失败、缺 hc 张力、注意力或 MoE 建不起来都返回假。
bool build_block_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                       std::uint64_t hc, std::uint64_t hc_lr, int top_k, std::uint64_t head_dim,
                       std::uint64_t n_rot, std::uint64_t idx_dim, std::uint64_t idx_top_k,
                       std::uint64_t idx_block, BlockLayer& out, std::string& err);

struct BlockScratch {
    std::vector<float> a_norm_f32, f_norm_f32;
    // 两组 hc_* 的三个投影一律解成 F32 存（档由文件声明：BF16 放大无损、Q8_0 逐块反量化，ADR-010）。
    std::vector<float> a_down_f32, a_up_f32, a_inject_f32;
    std::vector<float> f_down_f32, f_up_f32, f_inject_f32;
    GrScratch gr;
    std::vector<float> mixed, inject, block_out;
    GdnScratch gdn;
    QsaScratch qsa;
    experts::MoeScratch moe;
    // 可选的「把未命中专家这批算力交出去」（interfaces §8）。为空就走逐专家的主机路径。
    // 由调用方按本层的档给候选组与字节来源；见 `experts/moe_layer.hpp` 的 MoeOffload。
    const experts::MoeOffload* offload = nullptr;

    // 可选的 dense 设备执行器（阶段 A dense 接线）：非空时 GDN/QSA 整层走对应的 *_cuda 入口，
    // 否则（默认）走主机 run_gdn_layer / run_qsa_layer。由调用方（step_loop 挂 StepConfig.dense_exec）
    // 逐层挂上；主机构建该指针恒空（DenseExec 是函数指针结构，主机构建不填即不引用 CUDA 符号）。
    const DenseExec* dense = nullptr;

    // 可选的 PLE：只在 `ple->layer` 那一层挂上（由 `prepare_step_scratch` 挂），别的层为空。
    const PleRun* ple = nullptr;
    PleState* ple_state = nullptr;
    ple::BlockScratch ple_scratch;  // 块自己的中间缓冲（逐层一份，常驻不重分配）
};

// 一层的四段耗时（毫秒）。只作观测：给了 `phases` 才计时，不给就一点开销都不多花。
// 段按 [S-60] 的顺序切：gr_read（第一段门控残差的读路径）→ attn（该层的注意力）→ moe（含共享专家，
// 也含被交出去的那批）→ gr_write（两段门控残差的写回）。interfaces §7 把「每步耗时」列为必测观测，
// 这里是它的一层粒度版本。
struct LayerPhases {
    double load_ms = 0.0;  // 把两组 hc_* 的六个投影解成 F32（**每层一次**：该层的 scratch 第一次调用时读，
                           // 之后恒为 0——权重按层缓存，见 block_layer.cpp 的 load_gr_group）
    double gr_read_ms = 0.0;
    // gr_read_ms 的逐子步切分（两次 gr_read 累加；见 dense/gated_residual.hpp 的 GrPhases）。
    GrPhases gr_split;
    double attn_ms = 0.0;
    // attn_ms 的逐子步切分（互斥且穷尽，见 dense/attn_phases.hpp）。只有 proj 那几段走按行并行的
    // 池子，其余是手写标量循环——拆开才知道该动哪一段。
    AttnPhases attn;
    double moe_ms = 0.0;
    // moe_ms 的四项（互斥且穷尽，见 experts/moe_layer.hpp 的 MoePhases）：取数 / 路由 / 逐专家 / 共享。
    // 四者的修法完全不同（字节常驻、把路由也按行并行、把提交并大、共享专家单独处理），故要分开量。
    double moe_io_ms = 0.0;
    double moe_route_ms = 0.0;
    double moe_expert_ms = 0.0;
    double moe_shared_ms = 0.0;
    // 交出去那条路时 expert_ms 的三段（取数 / 工人 / 归并）；主机逐专家那条路为 0。
    double moe_off_fetch_ms = 0.0;
    double moe_off_worker_ms = 0.0;
    double moe_off_merge_ms = 0.0;
    double gr_write_ms = 0.0;
};

// 跑一层：R 是 (hc, n_embd)，**就地**被两次 gr_write 改写。QSA 层的 KV/rope 由调用方在
// `s.qsa` 里配好（分页 KV 与 rope 表），GDN 层的状态也在 `s.gdn`；MoE 因果地按该层自己的路由走。
bool run_block_layer(const artifact::GgufFile& gguf, const BlockLayer& layer, float* R,
                     BlockScratch& s, std::string& err, LayerPhases* phases = nullptr);

}  // namespace qinfer::dense