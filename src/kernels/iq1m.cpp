// IQ1_M 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-45）：
//   ggml/src/ggml-common.h      l.433 block_iq1_m（56 字节）/ l.1131 NGRID_IQ1S = 2048
//                              / l.1132 IQ1S_DELTA = 0.125 / l.1133 IQ1M_DELTA = 0.125 / l.443 iq1m_scale_t
//   ggml/src/ggml-quants.c      l.2679 dequantize_row_iq1_m
//   ggml/src/ggml-cpu/quants.c  ggml_vec_dot_iq1_m_q8_K_generic
// 格点表在 iq1m_tables.hpp（有符号字节）。
//
// 块内偏移（共 56 字节）：qs 在 0（32 字节，32 个码字的格点索引低 8 位）、qh 在 32（16 字节，
// 每 ib 两个：位 0..2 是第 l 个码字的索引高位、位 3 是它的 delta 符号；位 4..6 与位 7 同理给第 l+1 个）、
// scales 在 48（8 字节，当 4 个 u16 用）。**没有 d 字段**：fp16 尺度由这 4 个 u16 的各 4 个高位拼出
// （sc[0]>>12 | (sc[1]>>8)&0xf0 | (sc[2]>>4)&0xf00 | sc[3]&0xf000），低 12 位则是 8 个 3 位子尺度。
#include "kernels/iq1m.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq1m_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

constexpr float kDelta = 0.125f;  // IQ1S_DELTA == IQ1M_DELTA

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

// 由 8 个 scales 字节（当 4 个 u16）拼出那个被拆散的 fp16 尺度。
float assembled_d(const std::uint8_t* scales) {
    const std::uint16_t sc0 = read_u16_le(scales);
    const std::uint16_t sc1 = read_u16_le(scales + 2);
    const std::uint16_t sc2 = read_u16_le(scales + 4);
    const std::uint16_t sc3 = read_u16_le(scales + 6);
    const std::uint16_t u16 = static_cast<std::uint16_t>((sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) |
                                                         ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u));
    return f16_bits_to_f32(u16);
}

// 第 ib 组的两个子尺度（3 位，+1 后成为 1..15 的奇数倍）。
int sub_scale(const std::uint8_t* scales, int ib, int third_pair) {
    const std::uint16_t sc = read_u16_le(scales + 2 * (ib / 2));
    const int shift = 6 * (ib % 2) + third_pair;
    return 2 * static_cast<int>((sc >> shift) & 0x7u) + 1;
}

// 第 l 个码字的 11 位格点索引：低 8 位来自 qs，高 3 位来自 qh 的对应半字节。
std::uint16_t grid_index(const std::uint8_t* qs, const std::uint8_t* qh, int l) {
    return static_cast<std::uint16_t>(
        qs[l] | ((static_cast<std::uint16_t>(qh[l / 2]) << (8 - 4 * (l % 2))) & 0x700u));
}

// 第 l 个码字的 delta 符号：qh 的位 3（l 偶）或位 7（l 奇）。
float delta_of(const std::uint8_t* qh, int l) {
    return (qh[l / 2] & (l % 2 == 0 ? 0x08u : 0x80u)) ? -kDelta : kDelta;
}

}  // namespace

void dequant_iq1m_block(const std::uint8_t* block, float* out256) {
    const std::uint8_t* qs = block;
    const std::uint8_t* qh = block + 32;
    const std::uint8_t* scales = block + 48;
    const float d = assembled_d(scales);
    for (int ib = 0; ib < 8; ++ib) {
        const float dl1 = d * static_cast<float>(sub_scale(scales, ib, 0));
        const float dl2 = d * static_cast<float>(sub_scale(scales, ib, 3));
        const std::uint8_t* qsb = qs + 4 * ib;
        const std::uint8_t* qhb = qh + 2 * ib;
        for (int l = 0; l < 4; ++l) {
            const std::int8_t* grid = reinterpret_cast<const std::int8_t*>(
                &detail::kIq1sGrid[grid_index(qsb, qhb, l)]);
            const float dl = (l < 2) ? dl1 : dl2;
            const float delta = delta_of(qhb, l);
            float* y = out256 + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) {
                y[j] = dl * (static_cast<float>(grid[j]) + delta);
            }
        }
    }
}

float iq1m_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kIq1mBlockBytes;
        const std::uint8_t* qs = blk;
        const std::uint8_t* qh = blk + 32;
        const std::uint8_t* scales = blk + 48;
        const std::int8_t* q8 = y[i].qs;
        std::int32_t sumi1 = 0;
        std::int32_t sumi2 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qsb = qs + 4 * ib;
            const std::uint8_t* qhb = qh + 2 * ib;
            // sum1/sum2 各自按 l/2 分成两半，分别乘该组的两个子尺度。
            std::int32_t sum1[2] = {0, 0};
            std::int32_t sum2[2] = {0, 0};
            for (int l = 0; l < 4; ++l) {
                const std::int8_t* grid = reinterpret_cast<const std::int8_t*>(
                    &detail::kIq1sGrid[grid_index(qsb, qhb, l)]);
                std::int32_t lsum1 = 0;
                std::int32_t lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += (delta_of(qhb, l) > 0.0f) ? lsum2 : -lsum2;
            }
            const std::int32_t ls1 = sub_scale(scales, ib, 0);
            const std::int32_t ls2 = sub_scale(scales, ib, 3);
            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
        }
        sumf += assembled_d(scales) * y[i].d *
                (static_cast<float>(sumi1) + kDelta * static_cast<float>(sumi2));
    }
    return sumf;
}

void iq1m_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq1mBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq1m_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels