// Q2_0 的块布局、反量化与「与 Q8_0 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-37）：
//   ggml/src/ggml-common.h      l.187 block_q2_0（QK2_0 = 64）
//   ggml/src/ggml-quants.c      l.439 dequantize_row_q2_0
//   ggml/src/ggml-cpu/quants.c  l.177 ggml_vec_dot_q2_0_q8_0_generic
// 只取这三处。注意 QK2_0 = 64 与本文档部署模型的字节数一致（引擎自注：llama.cpp 引入该类型的那版
// PR 用的是 128，本产物是 64）——这里以 ggml 当前提交的 64 为准，并用真实张量的字节数核过。
#include "kernels/q2_0.hpp"

#include "kernels/fp16.hpp"

#include <cstddef>

namespace qinfer::kernels {

void dequant_q2_0_block(const std::uint8_t* block, float* out64) {
    const float d = f16_bits_to_f32(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
    const std::uint8_t* qs = block + 2;  // 16 字节，每字节 4 个 2-bit 码
    for (int j = 0; j < kQ20BlockElems; ++j) {
        const int code = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        out64[j] = static_cast<float>(code - 1) * d;  // 00 -> -1，01 -> 0，10 -> +1，11 -> +2
    }
}

float q2_0_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kQ20BlockBytes;
        const float d0 = f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        float sumi = 0.0f;
        // 一个 Q2_0 块（64 个权重）对两个 Q8_0 块（2 × 32）。两组各自带自己的 fp16 尺度。
        for (int k = 0; k < 2; ++k) {
            const Q80Block& yb = y[i * 2 + k];
            const float d1 = f16_bits_to_f32(yb.d_bits);
            int sumi_block = 0;
            const std::uint8_t* qs = blk + 2 + k * 8;
            for (int b = 0; b < 8; ++b) {
                const std::uint8_t byte = qs[b];
                sumi_block += (static_cast<int>((byte >> 0) & 3) - 1) * yb.qs[b * 4 + 0];
                sumi_block += (static_cast<int>((byte >> 2) & 3) - 1) * yb.qs[b * 4 + 1];
                sumi_block += (static_cast<int>((byte >> 4) & 3) - 1) * yb.qs[b * 4 + 2];
                sumi_block += (static_cast<int>((byte >> 6) & 3) - 1) * yb.qs[b * 4 + 3];
            }
            sumi += d1 * static_cast<float>(sumi_block);
        }
        sumf += d0 * sumi;
    }
    return sumf;
}

void q2_0_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kQ20BlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = q2_0_dot_q8_0(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels