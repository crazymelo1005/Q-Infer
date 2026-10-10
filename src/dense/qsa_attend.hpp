// QSA 的两步：在选择出来的 cell 上做 softmax GQA，以及把 attn 乘上它自己的门。
//
// 逐字取自参考引擎 `include/strata/kernels/qsa.hpp`（[S-57]）：
//
//     scores[j] = dot(k[j][q / (n_head/n_kv_head)], q[h]) / sqrt(head_dim)
//     w         = softmax_j(scores)
//     attn[h]   = sum_j w[j] * v[j][q / (n_head/n_kv_head)]
//
//     gate = q_full[h][head_dim + d]        （每个头的 2·head_dim 块的**后半段**，不是交错）
//     out  = attn · sigmoid(gate)           （sigmoid，不是 silu）
//
// 五处「都产出形状正确的结果」的地方，每一处都在测试里按错的算一遍要求它明显偏离：
//   1. q→kv 头的映射是**整除** `q / (n_head/n_kv_head)`，不是取模 `q % n_kv_head`（24 头 2 组时
//      两者给出 [0×12, 1×12] 与 [0,1,0,1,…]，都合法）；
//   2. 门上是 **sigmoid**（这个模型的 QSA 门是 sigmoid）；
//   3. 门取自每个头那个 2·head_dim 块的**后半段**，不是按元素交错取（`ref/qsa.py` 的 PROPERTY 10 钉这条）；
//   4. 分数要除 sqrt(head_dim)；
//   5. 若末尾要舍入到 fp16，**先乘后舍**（一次舍入而不是两次）。
//
// 两处范围：`ids` 为空是合法的（没有选中的 cell 就没有 softmax 可取），此时 attn 全 0；本文件**不**做
// fp16 的舍入也不做激活量化——引擎给了两个入口，差别只在最后那一次舍入，而本仓库的 `attn_output` 是
// Q8_0（[S-54]），它的激活搭档是 Q8_0 而不是 fp16，故这里只产出 f32，量化交给同一份档分派表。
#pragma once

#include <cstdint>
#include <string>

#include "dense/qsa_select.hpp"  // QsaShapes 与选择（同属 QSA 的几何）

namespace qinfer::dense {

// 只要选出来的那几个 cell 上的 softmax GQA。
//   q         (n_head, head_dim) f32，已归一、已过 rope（都由调用方做）；
//   k_scratch (n_ids, n_head_kv, head_dim) **fp16 位型**（uint16），v_scratch 同形；n_ids 为 0 时可为空；
//   attn      (n_head, head_dim) f32，**未过门**；weights（可空）是 (n_head, n_ids) 的 softmax 权重，
//             给的是「记录用」的那一份（引擎把它写进捕获记录，免得与它声称描述的那次注意力漂移）。
bool qsa_attend(const float* q, const std::uint16_t* k_scratch, const std::uint16_t* v_scratch,
                std::uint64_t n_ids, const QsaShapes& s, float* attn, float* weights,
                std::string& err);

// `out = attn · sigmoid(gate)`，乘在 double 里做（与引擎两个入口一致，故这里与它只差最后那次舍入）。
//   q_full (n_head, 2·head_dim) f32，是 `attn_q` 的**原始投影**（任何归一之前）；gate 取每头块的后半段。
//   out    (n_head, head_dim) f32；要 fp16 或要按档量化由调用方接。
bool qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out,
                        std::string& err);

}  // namespace qinfer::dense