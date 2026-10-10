// QSA 的选择（`topk_512`）：从逐 cell 的索引器分数里挑出 `width` 个最大的 cell。
//
// 依据参考引擎 `include/strata/kernels/qsa.hpp`（[S-57]）：`width = min(n_kv, idx_top_k + idx_block - 1)`，
// 其中 `idx_block - 1` 不是余量而是**没填满一个块的尾巴**——预算 = 整块数 × 块大小 + 不满一块的那些 cell。
// 真模型（[S-54]、G-12）：idx_top_k 2048、idx_block 4（索引器把 4 个 token 池化成一个 cell），
// 故宽度的上界是 2,051。**低于这个上界时选择是恒等**，契约要求 `topk_512` 精确复现恒等而不是近似。
//
// 三条规则都要照抄：取分数最大的 `width` 个；并列时**取索引小的**；返回的顺序是**索引升序**
// （不是分数降序——下游按页聚集要的是有序的 cell 列表）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 索引器的选择参数。块大小是「多少个 cell 打成一个块」，与池化粒度同一件事。
struct QsaShapes {
    std::uint64_t idx_top_k = 0;
    std::uint64_t idx_block = 0;
};

// 选择宽度：`min(n_kv, idx_top_k + idx_block - 1)`。`idx_block` 为 0 时按 1 处理（免得减出下溢）。
std::uint64_t qsa_selection_width(std::uint64_t n_kv, const QsaShapes& s);

// 从 n_kv 个分数里选出 width 个 cell，结果写进 out（索引升序）。n_kv < width 时全选。
// width == n_kv 时必须是恒等：[0, 1, …, n_kv-1]。
bool qsa_select(const float* scores, std::uint64_t n_kv, std::uint64_t width,
                std::vector<std::uint64_t>& out, std::string& err);

}  // namespace qinfer::dense