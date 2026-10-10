// IQ2_XS × Q8_K 的定点点积：权重在整数域直接参与点积，不解码成浮点再算。这是 engine §6 里
// 「码本解码融进 GEMM」的一个 IQ2_XS 档实现（也是 G-03 / G-07 两个微基准所测的那种形状）；
// 参考实例的专家档是 IQ2_S / IQ2_XXS / IQ1_M / Q2_0，见 engine §6 与 S-36。
//
// 与逐元素反量化相比有两处等价变形：尺度用 ls = 2·s + 1 配合末尾的 0.125 因子（等价于反量化里的
// (0.5 + s)·0.25），且整块的点积在 int32 里累加完再乘 d·d_y·0.125。转录来源见 iq2xs_dot.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

// 一段 n_blocks 个 IQ2_XS 块（共 n_blocks × 256 个元素）与 n_blocks 个 Q8_K 块的点积。
float iq2xs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ2_XS 块，行距 n_blocks × 74 字节。
void iq2xs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels