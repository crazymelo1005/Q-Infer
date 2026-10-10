// Q6_K：6.5625 bpw 的 K-quant 块，每块 256 个值、210 字节。它覆盖部署那份模型共享专家
// gate/up 的一部分层。块内布局与 IQ 家族都不同：ql[128] 低 4 位 + qh[64] 高 2 位 + scales[16]（int8）
// + **fp16 尺度在最后**（不是块首）。逐位细节与转录来源见 q6k.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kQ6kBlockBytes = 210;
inline constexpr int kQ6kValuesPerBlock = 256;

void dequant_q6k_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 Q6_K 块（共 n_blocks × 256 个元素）与 n_blocks 个 Q8_K 块的点积。
float q6k_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 Q6_K 块，行距 n_blocks × 210 字节。
void q6k_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels