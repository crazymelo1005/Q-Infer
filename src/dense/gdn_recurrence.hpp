// GDN 的循环本体与收尾（参考引擎 `gdn.hpp` 里「两个投影之间」的那一段）。
//
// 逐字取自 `include/strata/kernels/gdn.hpp` 的契约块（它自己转录自 `ref/gdn.py`，后者又来自
// `build_layer_attn_linear` 与 `build_delta_net_autoregressive`）：
//
//     dec      = exp(gate)                          (h_v,)
//     st       = state * dec
//     sk[h,j]  = sum_i st[i,j,h] * k[idx[h]][i]
//     d[h,j]   = (v[h,j] - sk[h,j]) * beta[h]
//     st[i,j,h] += k[idx[h]][i] * d[h,j]
//     o[h,j]   = sum_i st[i,j,h] * q[idx[h]][i]
//
// 六处「读过去很顺但给出可信错值」的地方，每一处都在测试里按错的算一遍要求它明显偏离：
//   1. 头的配对是 `h % h_k`（取模），不是 `h / (h_v / h_k)`（交错）——两种写法都产出形状正确的输出；
//   2. 衰减必须在秩一更新**之前**作用（先 `st = state·dec` 再累加，不是反过来）；
//   3. 状态布局是 (S, h_v, S) 且 **j 最快**，与 `ref/gdn.py` 的 (S, S, h_v) 互为转置——两者索引的是
//      同三个数，写错是静默的；
//   4. `beta` 进到这里之前**必须已经 sigmoid 过**（`gdn_step` 不做，层要做；不做的话写强度无界且有符号）
//      ——本文件提供 `gdn_beta_gate` 就是为这一步；
//   5. l2 归一的 `+ eps` 是加在**平方和**上的绝对下限，不是加在均值上（差 sqrt(S) ≈ 11.3 倍）；
//   6. 收尾是 `y = rms_norm(o, ssm_norm) · sigmoid(z)`——**sigmoid**，不是 silu（qwen3.5 才是 silu）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 一层 GDN 的几何：S 是头宽（也是状态被收缩的那一维），h_k 是 q/k 头数，h_v 是 v 头数。
// 真模型（[S-54]、`gdn.hpp` 的几何注记）：S = 128、h_k = 16、h_v = 48，且 h_v 必须是 h_k 的整数倍。
struct GdnRecShapes {
    std::uint64_t S = 0;
    std::uint64_t h_k = 0;
    std::uint64_t h_v = 0;
    bool sane() const { return S != 0 && h_k != 0 && h_v != 0 && h_v % h_k == 0; }
};

// 循环状态：**行主序 (S, h_v, S)，j 最快**。
struct GdnState {
    std::vector<float> data;
    void resize(const GdnRecShapes& s) { data.assign(static_cast<std::size_t>(s.S * s.h_v * s.S), 0.0f); }
};

// 一步递推，状态**原地**更新。
//   q、k 各 (h_k, S)，已 l2 归一，且 q 已由调用方乘过 1/sqrt(S)（两条 llama.cpp 路径把这步尺度放在
//   不同位置，故归调用方）；
//   v、o 各 (h_v, S)；gate 与 beta 各 (h_v,)。gate 是取 exp **之前**的值。
bool gdn_step(GdnState& st, const GdnRecShapes& s, const float* q, const float* k, const float* v,
              const float* gate, const float* beta, float* o, std::string& err);

// `beta = sigmoid(beta)`，就地作用在 h_v 个值上。`gdn_step` **不做**这一步，层必须做。
bool gdn_beta_gate(float* beta, std::uint64_t h_v, std::string& err);

// 收尾：`y[h,j] = rms_norm(o 的第 h 行, eps) * ssm_norm[j] * sigmoid(z[h,j])`。
//   o 与 z 各 (h_v, S)，ssm_norm 长 S（每 v 头共用一份），y 长 h_v·S。
bool gdn_closing_norm(const GdnRecShapes& s, const float* o, const float* z, const float* ssm_norm,
                      float eps, float* y, std::string& err);

}  // namespace qinfer::dense