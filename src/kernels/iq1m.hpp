// IQ1_M：IQ 家族的 1.75 bpw 块格式，每块 256 个值、56 字节。它覆盖部署那份模型 gate/up 的最后
// 3 层（8 / 13 / 37），补上它之后该模型的 routed expert 才算各层都有内核。
//
// 三处与 IQ2 家族不同，照搬会错：
//   1. 块里**没有独立的 fp16 尺度字段**——8 个 scales 字节既是 4 个 3 位子尺度、又合起来拼出那个
//      fp16 尺度（gguf-py 的注释称其为「唯一一个把 f16 尺度拆开存的类型」）；
//   2. 格点字节是**有符号** int8（-1 / 0 / 1），不是无符号幅度；
//   3. 除格点项外还有一个逐码字的 ±delta（IQ1M_DELTA = 0.125）加在格点值上。
// 逐位细节与转录来源见 iq1m.cpp 头部。
#pragma once

#include <cstdint>

#include "kernels/q8k.hpp"

namespace qinfer::kernels {

inline constexpr int kIq1mBlockBytes = 56;
inline constexpr int kIq1mValuesPerBlock = 256;

void dequant_iq1m_block(const std::uint8_t* block, float* out256);

// 一段 n_blocks 个 IQ1_M 块（共 n_blocks × 256 个元素）与 n_blocks 个 Q8_K 块的点积。
float iq1m_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 IQ1_M 块，行距 n_blocks × 56 字节。
void iq1m_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out);

}  // namespace qinfer::kernels