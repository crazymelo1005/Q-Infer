// IQ3_S：3.4375 bpw 的 256 值块、110 字节。它覆盖部署那份模型共享专家 gate/up 的一部分层
// （另有 IQ4_XS 与 Q6_K），与 IQ2 家族一样配 Q8_K 做定点点积。
//
// 块内布局（共 110 字节）：d 在 0（fp16）、qs 在 2（64 字节，每 2 字节一组给 8 个值的格点索引
// 低 8 位）、qh 在 66（8 字节，每个 32 值组出一个字节，低 5 位是 8 个索引的高 1 位）、signs 在 74
// （32 字节，每 4 字节覆盖一个 32 值组的 8 个值）、scales 在 106（4 字节，每字节两个 4 位子尺度）。
// 逐位细节与转录来源见 iq3s.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq3sBlockBytes = 110;
inline constexpr int kIq3sValuesPerBlock = 256;

void dequant_iq3s_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ3_S 块与 n_blocks 个 Q8_K 块的点积。
float iq3s_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ3_S 块，行距 n_blocks × 110 字节。
void iq3s_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels