// NEOX 部分 RoPE（旋转位置编码）。QSA 的 q/k 与索引器的键都要过它，故先做它。
//
// 依据参考引擎 `include/strata/kernels/rope.hpp`（[S-57] 同族来源）：
//
//   配对是 **NEOX**：`(i, i + n_rot/2)`，不是相邻的 `(2i, 2i+1)`；
//   旋转是**部分**的：只动每个头的前 `n_rot` 维，尾巴原样穿过（这份模型 256 维里只转 64）；
//   cos/sin 表在 **host 上用 float64 建**，只与 (n_rot, theta, 位置) 有关，与 token 无关，故值得缓存；
//   缩放在**表里**而不在内核里——带缩放的表与不带缩放的表在核里看不出区别，因为缩放后的角度
//   （含 YaRN 的 mscale 幅度修正）本身就是那张表。
//
// 四处「都产出形状正确的结果」的地方，每处都在测试里按错的算一遍要求它明显偏离：
//   1. 配对用相邻的 `(2i, 2i+1)` 而不是 `(i, i + n_rot/2)`；
//   2. 旋转整条头（把 n_rot 当 head_dim）而不是只转前 n_rot 维；
//   3. 一对里的符号取反（`a·c + b·s`）；
//   4. 表按 float 算角度而不是 double（差得小，但存在）。
//
// 一处范围：这份模型的 rope 带 **sections**（`rope_sections` 四个值，见 [S-58]），即不同频段可能用
// 不同的行为；本文件只做单段的 NEOX 部分 RoPE，分段那一层要等它的口径取到再补（别猜）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 一对 NEOX 的旋转：`(a, b) -> (a·c − b·s, a·s + b·c)`。
// 一整行的旋转与索引器池化那一行的旋转**必须**用这一份，否则两边配对不同、算出来的键指着别的方向。
// NEOX 下两个半段不相交，故就地写回是安全的。
inline void rope_neox_pair(float a, float b, float c, float s, float& oa, float& ob) {
    oa = a * c - b * s;
    ob = a * s + b * c;
}

// cos/sin 表：`max_pos` 个位置、每个 `n_rot/2` 对，对的下标最快。
// 第 p 个位置、第 i 对的角是 `p · theta^(-2i/n_rot)`，在 double 里算完再存成 f32。
bool build_rope_table(std::uint64_t n_rot, double theta, std::uint64_t max_pos,
                      std::vector<float>& cos_tab, std::vector<float>& sin_tab, std::string& err);

// 部分 NEOX 旋转：`x`/`out` 是 (rows, head_dim)，`pos` 长 rows（每行一个位置）。
// 只动每行的前 `n_rot` 维；`head_dim - n_rot` 那段原样穿过。可以就地（`out == x`）。
bool rope_neox_apply(const float* x, std::uint64_t rows, std::uint64_t head_dim,
                     std::uint64_t n_rot, const std::uint64_t* pos,
                     const std::vector<float>& cos_tab, const std::vector<float>& sin_tab, float* out,
                     std::string& err);

}  // namespace qinfer::dense