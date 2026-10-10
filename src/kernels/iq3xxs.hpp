// IQ3_XXS：3.0625 bpw 的 256 值块、98 字节。它覆盖另一档模型（IQ3_XXS 那份）gate/up 的 6 层，
// 配 Q8_K 做定点点积。
//
// 块内布局（共 98 字节）：d 在 0（fp16）、qs 在 2（64 字节，8 个 32 值组各出 8 个 8 位格点索引）、
// scales_and_signs 在 66（32 字节，每个 32 值组一个 u32：高 4 位是子尺度增量 n，低 28 位分成 4 个
// 7 位符号索引）。逐位细节与转录来源见 iq3xxs.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq3xxsBlockBytes = 98;
inline constexpr int kIq3xxsValuesPerBlock = 256;

void dequant_iq3xxs_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ3_XXS 块与 n_blocks 个 Q8_K 块的点积。
float iq3xxs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ3_XXS 块，行距 n_blocks × 98 字节。
void iq3xxs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels