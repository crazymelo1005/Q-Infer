// QSA 索引器的逐 cell 分数（`qsa_index`）。选择（[S-57] 的 `topk_512`）要的就是它的输出。
//
// 逐字取自参考引擎 `include/strata/kernels/qsa.hpp`：
//
//   score[j] = bias[cb] + sum_h Relu(dot(pooled[cb], q_idx[h])),  cb = min(j / r, n_bid)
//   score[j] += 1e9f  对「不满一块的尾巴」那些 cell（j >= (n_kv / r) * r）
//
// 五处「都产出形状正确的结果」的地方，每处都在测试里按错的算一遍要求它明显偏离：
//   1. Relu 是**逐头**取、求和在后——`relu(sum_h dot_h)` 是错的读法，只要有一个头的点积为负，
//      两者就是不同的数；
//   2. 「不满一块的尾巴」那一段要加 `1e9` 的偏置（这一条与 llama.cpp 因果解码的输入一致，
//      且**即使 bias 传 null 也要加**）；
//   3. 块的映射是 `cb = min(j / r, n_bid)`——那个 `min` 就是「尾巴 → 备用槽」的全部内容（解码期
//      没有别的掩码）；少了它，尾巴那些 cell 会去读备用槽的键；
//   4. 这里**没有 softmax 也没有 1/sqrt(d)**；
//   5. 已完成块不加尾巴偏置（只有不满一块的那段加）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/qsa_select.hpp"  // QsaShapes

namespace qinfer::dense {

// 逐 cell 的分数，长度 n_kv（= pos + 1）。
//   pooled   (n_bid + 1, idx_dim)  块键，**备用槽在最后**（由 `indexer_key_append` 维护）；
//   q_idx    (idx_n_head, idx_dim) 调用方已归一、已过 rope（[S-58]）；
//   bias     (n_bid + 1,) 或 null  可选的逐块偏置；
//   cell_scores 长 n_kv，覆盖写。
bool qsa_index(const float* pooled, std::uint64_t n_bid, std::uint64_t idx_dim,
               std::uint64_t idx_n_head, const float* q_idx, const float* bias, std::uint64_t idx_block,
               std::uint64_t n_kv, float* cell_scores, std::string& err);

}  // namespace qinfer::dense