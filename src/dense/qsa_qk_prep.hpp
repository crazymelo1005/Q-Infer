// QSA 的 q/k 预处理：从融合的 `attn_q` 投影里拆出 q 与门，再把 q/k 各自按头归一、过 rope。
//
// 依据参考引擎 `include/strata/kernels/qsa.hpp` 与 [S-54]/[S-57]：
//
//   `attn_q.weight` 的输出是 (n_head, 2·head_dim)——每个头一个 2·head_dim 的块，其中
//       q    = 块的前 head_dim
//       gate = 块的后 head_dim        （**不是**按元素交错；ref/qsa.py 的 PROPERTY 10 钉这条）
//   而 `qsa_attend` 收到的 q 是「already normed and rotated」，故 q 与 k 都要先
//       rms_norm(按头, gamma = attn_q_norm / attn_k_norm) → 部分 NEOX rope（[S-58]）
//
// 四处「都产出形状正确的结果」的地方，每处都在测试里按错的算一遍要求它明显偏离：
//   1. q/gate 的拆分在 head_dim 处切（块的前后两半），不是按元素交错取；
//   2. rope 在**归一之后**（先 rope 后归一会得到另一个形状同样正确的结果）；
//   3. 门**不**过 rope 也不归一（它不上注意力，只做收尾的 sigmoid）；
//   4. 头宽与 rope 维数分开：只转每头的前 `n_rot` 维，尾巴穿过。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 一次预处理需要的权重与位置：gamma 各长 head_dim（每头共用一份）。
struct QsaQkWeights {
    const float* q_norm = nullptr;  // attn_q_norm.weight
    const float* k_norm = nullptr;  // attn_k_norm.weight
    float eps = 1e-6f;
};

// q_full 是 `attn_q` 的**原始输出**（n_head × 2·head_dim，任何归一之前）；
// k_raw 是 `attn_k` 的原始输出（n_head_kv × head_dim）。
// 输出：q_ready（n_head × head_dim，已归一已 rope）、k_ready（n_head_kv × head_dim，同）、
// gate（n_head × head_dim，**未归一未 rope**，留给收尾的 sigmoid）。
bool qsa_qk_prep(std::uint64_t n_head, std::uint64_t n_head_kv, std::uint64_t head_dim,
                 std::uint64_t n_rot, std::uint64_t pos, const float* q_full, const float* k_raw,
                 const QsaQkWeights& w, const std::vector<float>& cos_tab,
                 const std::vector<float>& sin_tab, float* q_ready, float* k_ready, float* gate,
                 std::string& err);

}  // namespace qinfer::dense