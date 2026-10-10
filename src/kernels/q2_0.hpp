// Q2_0：2.25 bpw 的 2-bit 档，每块 64 个值、18 字节。它是参考实例里 down 投影（全 48 层）的格式，
// 也是引擎自身 P1.S3 所实现的那个档。640（expert_feed_forward_length）不是 256 的整数倍，所以只有
// 每块 64 个值的档能覆盖 down 的 640 维（见 S-36 与 engine §6）。
//
// 码 {0,1,2,3} 映射到 {-1,0,+1,+2}，每字节 4 个码、低 2 位在前。激活搭档是 Q8_0：一个 Q2_0 块
// 对应两个 Q8_0 块。转录来源见 q2_0.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8_0.hpp"

namespace qinfer::kernels {

inline constexpr int kQ20BlockElems = 64;
inline constexpr int kQ20BlockBytes = 18;  // 2 字节 fp16 尺度 + 16 字节 2-bit 码

void dequant_q2_0_block(const std::uint8_t* block, float* out64);

// 一段 n_blocks 个 Q2_0 块（共 n_blocks × 64 个元素）与 2·n_blocks 个 Q8_0 块的点积。
float q2_0_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 Q2_0 块，行距 n_blocks × 18 字节。
void q2_0_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out);

}  // namespace qinfer::kernels