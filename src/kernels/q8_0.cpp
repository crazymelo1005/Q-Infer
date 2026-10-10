// Q8_0 的块布局与反量化转录自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，
// 2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-37）：
//   ggml/src/ggml-common.h  l.251 block_q8_0（QK8_0 = 32）
//   ggml/src/ggml-quants.c  l.553 dequantize_row_q8_0
// 只取这两处。
#include "kernels/q8_0.hpp"

#include "kernels/fp16.hpp"

namespace qinfer::kernels {

void q8_0_dequant_block(const Q80Block& blk, float* out32) {
    const float d = f16_bits_to_f32(blk.d_bits);
    for (int j = 0; j < kQ80BlockElems; ++j) out32[j] = d * static_cast<float>(blk.qs[j]);
}

}  // namespace qinfer::kernels