// QSA 一层的静态描述：五个投影的档与块几何 + 四个 norm 的登记项。先建结构，数值路径随后。
//
// 张力名（[S-54] 直读 GGUF）：`attn_q` / `attn_k` / `attn_v` / `attn_output`、
// `attn_q_norm` / `attn_k_norm`、`indexer.q_proj` / `indexer.k_proj` / `indexer.q_norm` / `indexer.k_norm`。
// QSA 层**没有** `attn_qkv`（那是 GDN 层的），故这一条也用来把两种层区分开。
//
// 几何由数据推出并互相核对（真模型：head_dim 256、n_head 24、n_head_kv 2，见 [S-54]）：
//   attn_q 的输出 = 2 · n_head · head_dim（q 与门融合，见 [S-57]）
//   attn_v 的输出 = n_head_kv · head_dim；attn_output 的输入 = n_head · head_dim
// 但 `head_dim` 与 `n_rot` 无法从形状唯一解出（6144 可以是 256×24 也可以是 128×48），故由调用方给，
// 本函数负责核。索引器：`indexer.q_proj` 的输出 = idx_n_head · idx_dim、`indexer.k_proj` 的输出 = idx_dim，
// 两者也要与 `indexer.*_norm` 的长度对上。
#pragma once

#include <cstdint>
#include <string>

#include "artifact/gguf_table.hpp"
#include "dense/qsa_select.hpp"
#include "experts/expert_formats.hpp"

namespace qinfer::dense {

struct QsaLayer {
    int layer = -1;
    QsaShapes sh;                  // idx_top_k / idx_block / head_dim / n_head / n_head_kv
    std::uint64_t n_rot = 0;       // 部分 rope 的维数（真模型 64）
    std::uint64_t idx_dim = 0;     // 索引器的头宽（真模型 128）
    std::uint64_t idx_n_head = 0;  // 索引器的 q 头数（真模型 4）
    experts::MatrixSpec q, k, v, out;      // 四个注意力投影
    // 两个索引器投影：这份 artifact 里是 **BF16**（不是量化权重），故不经「按档分派」那张表，
    // 走 bf16 的点积（与路由器同一条路）；这里只登记它们的几何。
    experts::MatrixSpec idx_q, idx_k;
    bool idx_dense_bf16 = false;
    const artifact::GgufTensorInfo* t_q = nullptr;
    const artifact::GgufTensorInfo* t_k = nullptr;
    const artifact::GgufTensorInfo* t_v = nullptr;
    const artifact::GgufTensorInfo* t_out = nullptr;
    const artifact::GgufTensorInfo* t_idx_q = nullptr;
    const artifact::GgufTensorInfo* t_idx_k = nullptr;
    const artifact::GgufTensorInfo* t_q_norm = nullptr;
    const artifact::GgufTensorInfo* t_k_norm = nullptr;
    const artifact::GgufTensorInfo* t_idx_q_norm = nullptr;
    const artifact::GgufTensorInfo* t_idx_k_norm = nullptr;
};

// 建一层。`head_dim` / `n_rot` / `idx_dim` / `idx_top_k` / `idx_block` 由调用方给（形状解不出唯一值），
// `n_head` / `n_head_kv` / `idx_n_head` 从形状推出并互相核对。任一必需张力缺失、档没内核、几何不自洽
// （含「这一层不是 QSA 层」）都返回假并写 err。
bool build_qsa_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                     std::uint64_t head_dim, std::uint64_t n_rot, std::uint64_t idx_dim,
                     std::uint64_t idx_top_k, std::uint64_t idx_block, QsaLayer& out,
                     std::string& err);

}  // namespace qinfer::dense