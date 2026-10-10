// IQ2_S：IQ 家族的 2.5625 bpw 块格式，每块 256 个值、82 字节。它是目标负载里 gate/up 投影最大的
// 单一档（部署那份 34 层，另一份 IQ3_XXS 档 10 层）。
//
// 与 IQ2_XS 同族，但有两处不同：格点索引是 10 位（1024 项的 iq2s_grid，高 2 位来自 qh），
// 且符号与尺度各自单独存放（IQ2_XS 把符号合并在码字的高 7 位里）。激活搭档同为 Q8_K。
// 逐位细节与转录来源见 iq2s.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq2sBlockBytes = 82;
inline constexpr int kIq2sValuesPerBlock = 256;

void dequant_iq2s_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ2_S 块（共 n_blocks × 256 个元素）与 n_blocks 个 Q8_K 块的点积。
float iq2s_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ2_S 块，行距 n_blocks × 82 字节。
void iq2s_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels