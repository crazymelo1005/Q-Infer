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
#include "dense/gated_residual.hpp"
#include "dense/gdn_layer.hpp"
#include "dense/layer_map.hpp"
#include "dense/qsa_layer.hpp"
#include "experts/moe_layer.hpp"

namespace qinfer::dense {

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
};

// 跑一层：R 是 (hc, n_embd)，**就地**被两次 gr_write 改写。QSA 层的 KV/rope 由调用方在
// `s.qsa` 里配好（分页 KV 与 rope 表），GDN 层的状态也在 `s.gdn`；MoE 因果地按该层自己的路由走。
bool run_block_layer(const artifact::GgufFile& gguf, const BlockLayer& layer, float* R,
                     BlockScratch& s, std::string& err);

}  // namespace qinfer::dense