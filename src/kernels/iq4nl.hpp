// IQ4_NL：4 位非线性量化的块格式，每块 32 个值、18 字节（2 字节 fp16 尺度 + 16 字节半字节）。
// 它在本项目里有两种身份：记忆表（per_layer_token_embd.weight）的行格式（一行 160 个值 = 5 块），
// 以及通用权重格式（另一档模型里 18 层 down 投影与部分共享专家张量用它）。
//
// 半字节是「分裂式」而非交错：y[j] = d·cb[qs[j] & 0x0f]、y[j + 16] = d·cb[qs[j] >> 4]。
// 按交错读（y[2j]、y[2j+1]）会得到一组完全可信但错误的 32 个值。码本 cb 是 IQ4_NL 的量化格点
// （ggml 的 kvalues_iq4nl），16 项、无 -8 偏移。块的激活搭档是 Q8_0（两者块大小都是 32）。
#pragma once

#include <cstdint>

#include "kernels/fp16.hpp"
#include "kernels/q8_0.hpp"

namespace qinfer::kernels {

inline constexpr int kIq4nlBlockBytes = 18;
inline constexpr int kIq4nlValuesPerBlock = 32;

inline constexpr int kIq4nlRowBytes = 90;        // 记忆表行：5 块
inline constexpr int kIq4nlValuesPerRow = 160;

int iq4nl_code(int code);

void dequant_iq4nl_block(const std::uint8_t* block, float* out32);

void dequant_iq4nl_row(const std::uint8_t* row, float* out160);

// 一段 n_blocks 个 IQ4_NL 块（共 n_blocks × 32 个元素）与 n_blocks 个 Q8_0 块的点积。
float iq4nl_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ4_NL 块，行距 n_blocks × 18 字节。
void iq4nl_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out);

}  // namespace qinfer::kernels