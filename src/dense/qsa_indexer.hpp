// QSA 索引器的键追加与池化（`indexer_key_append`）。它是 `qsa_index` 的输入一侧。
//
// 逐字取自参考引擎 `include/strata/kernels/qsa.hpp` 与 `src/kernels/cuda/native_qsa_indexer.cu`（[S-59]）：
//
//   pos == 0               y = rope_neox(norm(raw, w_k_norm), 0)；`dead = y`、`pooled[0] = y`
//   每个 token             把原始键放进这一块的窗口（窗口长 R = idx_block）
//   块 b 满时（pos % R == R-1）
//                          pooled[b] = rope_neox(norm(mean(该块 R 个 cell), w_k_norm), pos_base + R*b)
//                          pooled[b + 1] = dead          （备用槽只是 dead 的镜像，dead 整条序列不变）
//
// 四处「都产出形状正确、量级可信的结果」的地方，每处都在测试里按错的算一遍要求它明显偏离：
//   1. 块键的旋转位置取 `pos_base + R*b`（该块**首个** cell 的位置），不是取末个 cell 的位置；
//   2. `pos_base` 不能省——把 cell 的位置当成它的索引只对从 0 开始的序列成立（上游在 pos_base = 100
//      上测出两种读法差 514%）；
//   3. 备用槽是**序列 cell 0** 的键（位置 0 处算出来的），不是当前窗口里的某个 cell；
//   4. 均值取的是**该块那 R 个 cell**，块边界按 R 对齐（不是「前 R-1 个 + 当前」这种滑动窗口）。
// 另有一处次序：先归一（除以 sqrt(mean(x²)+eps)）再逐维乘 `w_k_norm`，不是先乘后归一。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dense/qsa_select.hpp"  // QsaShapes

namespace qinfer::dense {

// 跨 token 存活的状态：窗口（这一块已到的 cell）、备用槽的键、块键、已完成块数。
struct QsaIndexerState {
    std::vector<float> window;  // idx_block × idx_dim，这一块的原始键（按 cell 序）
    std::vector<float> dead;    // idx_dim，备用槽的键 = 序列 cell 0 在位置 0 的键
    std::vector<float> pooled;  // (n_bid + 1) × idx_dim，块键；备用槽在最后
    std::uint64_t n_bid = 0;    // 已完成块数
    // 按几何把缓冲开好（窗口清零、dead 与 pooled 置零）。
    void reset(const QsaShapes& s, std::uint64_t idx_dim);
};

bool indexer_key_append(const QsaShapes& s, std::uint64_t idx_dim, std::uint64_t pos,
                        std::uint64_t pos_base, const float* raw, const float* w_k_norm, float eps,
                        const std::vector<float>& cos_tab, const std::vector<float>& sin_tab,
                        QsaIndexerState& st, std::string& err);

}  // namespace qinfer::dense