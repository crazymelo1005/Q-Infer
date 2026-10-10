// IQ2_S 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-39）：
//   ggml/src/ggml-common.h      l.396 block_iq2_s（QK_K = 256，82 字节）
//   ggml/src/ggml-quants.c      l.2543 dequantize_row_iq2_s
//   ggml/src/ggml-cpu/quants.c  l.998 ggml_vec_dot_iq2_s_q8_K_generic
// 格点表在 iq2s_tables.hpp，符号表与符号位掩码复用 iq2xs_tables.hpp（IQ2_S 与 IQ2_XS 共用）。
//
// 块内偏移（共 82 字节）：d 在 0（fp16）、qs 在 2（前 32 字节是 32 个码字的格点索引低 8 位，
// 后 32 字节是 32 个码字的符号字节）、qh 在 66（8 字节，每个 ib32 出 8 位，分给该组的 4 个码字
// 各 2 位作为索引高 2 位）、scales 在 74（8 字节，每字节两个 4 位尺度）。上游写作 signs = qs + 32。
#include "kernels/iq2s.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq2s_tables.hpp"
#include "kernels/iq2xs_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

// 一个码字的 10 位格点索引：低 8 位来自 qs，高 2 位来自 qh 的 2*l 位处。
std::uint16_t grid_index(const std::uint8_t* qs, const std::uint8_t* qh, int ib, int l) {
    return static_cast<std::uint16_t>(qs[4 * ib + l] |
                                      ((qh[ib] << (8 - 2 * l)) & 0x300));
}

}  // namespace

void dequant_iq2s_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(detail::read_u16_le(block));
    const std::uint8_t* qs = block + 2;
    const std::uint8_t* signs = qs + 32;
    const std::uint8_t* qh = block + 66;
    const std::uint8_t* scales = block + 74;
    for (int ib = 0; ib < 8; ++ib) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib] & 0x0F)) * 0.25f;
        const float db1 = d * (0.5f + static_cast<float>(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
                &detail::kIq2sGrid[grid_index(qs, qh, ib, l)]);
            const std::uint8_t sign = signs[4 * ib + l];
            const float db = (l < 2) ? db0 : db1;
            float* y = out256 + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) {
                y[j] = db * static_cast<float>(grid[j]) *
                       ((sign & detail::kSignBit[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

float iq2s_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kIq2sBlockBytes;
        const float d = f16_bits_to_f32(detail::read_u16_le(blk)) * y[i].d;
        const std::int8_t* q8 = y[i].qs;
        const std::uint8_t* qs = blk + 2;
        const std::uint8_t* signs = qs + 32;
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* scales = blk + 74;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::int32_t ls1 = 1 + 2 * (scales[ib] & 0x0F);
            const std::int32_t ls2 = 1 + 2 * (scales[ib] >> 4);
            std::int32_t sumi1 = 0;
            std::int32_t sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
                    &detail::kIq2sGrid[grid_index(qs, qh, ib, l)]);
                const std::uint8_t sign = signs[4 * ib + l];
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * ((sign & detail::kSignBit[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
                    &detail::kIq2sGrid[grid_index(qs, qh, ib, l)]);
                const std::uint8_t sign = signs[4 * ib + l];
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * ((sign & detail::kSignBit[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125f * sumf;
}

void iq2s_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq2sBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq2s_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels