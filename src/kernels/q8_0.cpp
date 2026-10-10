// Q8_0 的块布局与反量化转录自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，
// 2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-37）：
//   ggml/src/ggml-common.h  l.251 block_q8_0（QK8_0 = 32）
//   ggml/src/ggml-quants.c  l.553 dequantize_row_q8_0 / l.276 quantize_row_q8_0_ref
// 只取这三处。参考量化器的尺度取绝对值最大者（不带符号）：d = amax/127，存成 fp16；
// 量化值是 roundf(x/d)（四舍五入到最近整数、.5 远离零，即 std::round 的语义）。
#include "kernels/q8_0.hpp"

#include "kernels/fp16.hpp"

#include <cmath>

namespace qinfer::kernels {

void q8_0_quantize_row(const float* x, Q80Block* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * kQ80BlockElems;
        Q80Block& y = out[b];
        float amax = 0.0f;
        for (int j = 0; j < kQ80BlockElems; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y.d_bits = f32_to_f16_bits(d);
        for (int j = 0; j < kQ80BlockElems; ++j) {
            y.qs[j] = static_cast<std::int8_t>(std::round(xb[j] * id));
        }
    }
}

void q8_0_dequant_block(const Q80Block& blk, float* out32) {
    const float d = f16_bits_to_f32(blk.d_bits);
    for (int j = 0; j < kQ80BlockElems; ++j) out32[j] = d * static_cast<float>(blk.qs[j]);
}

}  // namespace qinfer::kernels