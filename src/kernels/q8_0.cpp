// Q8_0 的块布局与反量化转录自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，
// 2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-37）：
//   ggml/src/ggml-common.h  l.251 block_q8_0（QK8_0 = 32）
//   ggml/src/ggml-quants.c  l.553 dequantize_row_q8_0 / l.276 quantize_row_q8_0_ref
//   ggml/src/ggml-cpu/quants.c  l.451 ggml_vec_dot_q8_0_q8_0_generic（两侧都是 Q8_0 时）
// 只取这几处。参考量化器的尺度取绝对值最大者（不带符号）：d = amax/127，存成 fp16；
// 量化值是 roundf(x/d)（四舍五入到最近整数、.5 远离零，即 std::round 的语义）。
// Q8_0 作权重时（共享专家的 down 投影）先反量化成浮点再点积在本仓库不做——权重侧的
// q8_0_dot_q8_0 保留整数乘加，权重内存流量不翻倍（engine §6 第 1 条）。
#include "kernels/q8_0.hpp"

#include "kernels/fp16.hpp"

#include <cmath>

namespace qinfer::kernels {

namespace {

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

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

void dequant_q8_0_raw(const std::uint8_t* block, float* out32) {
    const float d = f16_bits_to_f32(read_u16_le(block));
    const std::int8_t* qs = reinterpret_cast<const std::int8_t*>(block + 2);
    for (int j = 0; j < kQ80BlockElems; ++j) out32[j] = d * static_cast<float>(qs[j]);
}

float q8_0_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = w_blocks + static_cast<std::size_t>(i) * kQ80BlockBytes;
        const std::int8_t* wq = reinterpret_cast<const std::int8_t*>(block + 2);
        const std::int8_t* yq = y[i].qs;
        std::int32_t sumi = 0;
        for (int j = 0; j < kQ80BlockElems; ++j) {
            sumi += static_cast<std::int32_t>(wq[j]) * static_cast<std::int32_t>(yq[j]);
        }
        sumf += static_cast<float>(sumi) * (f16_bits_to_f32(read_u16_le(block)) *
                                           f16_bits_to_f32(y[i].d_bits));
    }
    return sumf;
}

void q8_0_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kQ80BlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = q8_0_dot_q8_0(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels