// IQ2_XS：路由专家权重的块格式，每块 256 个值、74 字节。格点表与符号表的转录来源、
// 以及 0.25 因子 / 0.5 偏置 / 分裂式尺度这三个陷阱，都写在 iq2xs.cpp 的头部说明里。
#pragma once

#include <cstdint>

#include "kernels/fp16.hpp"

namespace qinfer::kernels {

inline constexpr int kIq2xsBlockBytes = 74;
inline constexpr int kIq2xsValuesPerBlock = 256;

void dequant_iq2xs_block(const std::uint8_t* block, float* out256);

}  // namespace qinfer::kernels