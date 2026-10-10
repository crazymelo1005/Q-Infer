// IQ2_XXS 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-41）：
//   ggml/src/ggml-common.h      l.381 block_iq2_xxs（QK_K = 256，66 字节）/ l.560 iq2xxs_grid（256 项）
//   ggml/src/ggml-quants.c      l.2488 dequantize_row_iq2_xxs
//   ggml/src/ggml-cpu/quants.c  l.906 ggml_vec_dot_iq2_xxs_q8_K_generic
// 格点表在 iq2xxs_tables.hpp，符号表与符号位掩码复用 iq2xs_tables.hpp。
//
// 块内偏移（共 66 字节）：d 在 0（fp16），qs 在 2（64 字节）。每 32 个值（ib32）占 8 字节：
// 前 4 字节是 4 个码字的格点索引（8 位，进 256 项表），后 4 字节按小端当一个 u32 用——
// 它的高 4 位是该组唯一的尺度，低 28 位分成 4 个 7 位符号索引。上游用 memcpy 把这两个 u32
// 拷进局部数组再取字节，本实现按小端逐字节拼，等价。
#include "kernels/iq2xxs.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq2xxs_tables.hpp"
#include "kernels/iq2xs_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

const std::uint8_t* group_base(const std::uint8_t* block, int ib) { return block + 2 + 8 * ib; }

}  // namespace

void dequant_iq2xxs_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(detail::read_u16_le(block));
    for (int ib = 0; ib < 8; ++ib) {
        const std::uint8_t* base = group_base(block, ib);
        const std::uint32_t w1 = read_u32_le(base + 4);
        const float db = d * (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
                &detail::kIq2xxsGrid[base[l]]);
            const std::uint8_t sign = detail::kSigns[(w1 >> (7 * l)) & 127u];
            float* y = out256 + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) {
                y[j] = db * static_cast<float>(grid[j]) *
                       ((sign & detail::kSignBit[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

float iq2xxs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kIq2xxsBlockBytes;
        const float d = f16_bits_to_f32(detail::read_u16_le(blk)) * y[i].d;
        const std::int8_t* q8 = y[i].qs;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* base = group_base(blk, ib);
            const std::uint32_t w1 = read_u32_le(base + 4);
            const std::int32_t ls = 2 * static_cast<std::int32_t>(w1 >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
                    &detail::kIq2xxsGrid[base[l]]);
                const std::uint8_t sign = detail::kSigns[(w1 >> (7 * l)) & 127u];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * ((sign & detail::kSignBit[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125f * sumf;
}

void iq2xxs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq2xxsBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq2xxs_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels