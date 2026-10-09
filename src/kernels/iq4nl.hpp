// IQ4_NL：记忆表（per_layer_token_embd.weight）的行格式。一行 160 个值 = 5 块，
// 每块 = 2 字节 fp16 尺度 + 16 字节半字节，且半字节是分裂式而非交错：
//
//   y[j]      = d · cb[qs[j] & 0x0f]
//   y[j + 16] = d · cb[qs[j] >> 4]
//
// 「分裂式」是关键陷阱：按交错读（y[2j]、y[2j+1]）会得到一组完全可信但错误的 160 个值。
// 码本 cb 是 IQ4_NL 的量化格点（ggml 的 kvalues_iq4nl），16 项、无 -8 偏移。
#pragma once

#include <cstdint>

#include "kernels/fp16.hpp"

namespace qinfer::kernels {

inline constexpr int kIq4nlRowBytes = 90;
inline constexpr int kIq4nlValuesPerRow = 160;

int iq4nl_code(int code);

void dequant_iq4nl_row(const std::uint8_t* row, float* out160);

}  // namespace qinfer::kernels
