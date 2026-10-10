// IQ2_XXS：IQ 家族的 2.0625 bpw 块格式，每块 256 个值、66 字节。它是目标负载里 gate/up 的另一档
// （部署那份 11 层、另一份 IQ3_XXS 档 9 层）。
//
// 与 IQ2_XS / IQ2_S 同族，差别在尺度只有一组：每 32 个值（一个 ib32）共用一个 4 位尺度，而
// IQ2_XS / IQ2_S 是每 16 个值一个。格点索引是 8 位（256 项的 iq2xxs_grid），符号索引 7 位、
// 四个码字各一个，三者都挤在一个 ib32 的 8 字节里。激活搭档同为 Q8_K。逐位细节见 iq2xxs.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq2xxsBlockBytes = 66;
inline constexpr int kIq2xxsValuesPerBlock = 256;

void dequant_iq2xxs_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ2_XXS 块（共 n_blocks × 256 个元素）与 n_blocks 个 Q8_K 块的点积。
float iq2xxs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ2_XXS 块，行距 n_blocks × 66 字节。
void iq2xxs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels