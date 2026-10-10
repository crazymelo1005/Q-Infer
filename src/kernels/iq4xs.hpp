// IQ4_XS：4.25 bpw 的 256 值块、136 字节。它覆盖部署那份模型共享专家 gate/up 的一部分层
// （与 IQ3_S / Q6_K 同列），配 Q8_K 做定点点积。它是 IQ4_NL 的 256 值块版本：共用同一张
// kvalues_iq4nl 码本（见 iq4nl_tables.hpp）与同样的半字节序，但尺度是带 -32 偏移的 6 位
// （低 4 位在 scales_l、高 2 位在 scales_h），且没有单独的 fp16 尺度之外的东西。
//
// 块内布局（共 136 字节）：d 在 0（fp16）、scales_h 在 2（u16，每 2 位给一个 32 值组的高 2 位）、
// scales_l 在 4（4 字节，每字节两个 4 位低半尺度）、qs 在 8（128 字节）。逐位细节见 iq4xs.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq4xsBlockBytes = 136;
inline constexpr int kIq4xsValuesPerBlock = 256;

void dequant_iq4xs_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ4_XS 块与 n_blocks 个 Q8_K 块的点积。
float iq4xs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ4_XS 块，行距 n_blocks × 136 字节。
void iq4xs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels