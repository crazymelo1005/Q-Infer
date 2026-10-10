// IQ4_XS 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-48）：
//   ggml/src/ggml-common.h      l.454 block_iq4_xs（136 字节）
//   ggml/src/ggml-quants.c      l.2743 dequantize_row_iq4_xs
//   ggml/src/ggml-cpu/quants.c  l.1283 ggml_vec_dot_iq4_xs_q8_K_generic
// 码本复用 iq4nl_tables.hpp 的 kIq4nlCode（IQ4_XS 与 IQ4_NL 共用 kvalues_iq4nl）。
//
// 与 IQ4_NL（32 值块）不同、照搬必错的两处：
//   ① 尺度是 6 位且**带 -32 偏移**：低 4 位来自 scales_l 的半个字节，高 2 位来自 scales_h 的对应
//      2 位（每个 32 值组 2 位），dl = d × (ls - 32)。IQ4_NL 只有一个 fp16 尺度、没有偏移；
//   ② 尺度按 32 值组排布，而 scales_l 是「一字节管两个组」（低半字节给偶数组、高半字节给奇数组），
//      scales_h 是 u16 里每 2 位管一个组——两种不同的排布，混用会把组与组的尺度对调。
// 半字节序与 IQ4_NL 一致（前 16 个取 qs[j] 的低半字节、后 16 个取同一个字节的高半字节），
// 但每 32 个值前进 16 个 qs 字节。
// 点积与反量化同值：整数和乘 dl，没有额外的末尾因子。
#include "kernels/iq4xs.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq4nl_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

constexpr int kScalesHAt = 2;
constexpr int kScalesLAt = 4;
constexpr int kQsAt = 8;

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

// 第 ib 个 32 值组的 6 位尺度：低 4 位在 scales_l 的半个字节，高 2 位在 scales_h 的 2 位处。
int group_scale(const std::uint8_t* scales_l, std::uint16_t scales_h, int ib) {
    const int low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
    const int high = (static_cast<int>(scales_h) >> (2 * ib)) & 3;
    return low | (high << 4);
}

}  // namespace

void dequant_iq4xs_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(read_u16_le(block));
    const std::uint8_t* scales_l = block + kScalesLAt;
    const std::uint16_t scales_h = read_u16_le(block + kScalesHAt);
    const std::uint8_t* qs = block + kQsAt;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl = d * static_cast<float>(group_scale(scales_l, scales_h, ib) - 32);
        float* y = out256 + 32 * ib;
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = dl * static_cast<float>(detail::kIq4nlCode[qs[j] & 0x0F]);
            y[j + 16] = dl * static_cast<float>(detail::kIq4nlCode[qs[j] >> 4]);
        }
        qs += 16;
    }
}

float iq4xs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = w_blocks + static_cast<std::size_t>(i) * kIq4xsBlockBytes;
        const float d = f16_bits_to_f32(read_u16_le(block)) * y[i].d;
        const std::uint8_t* scales_l = block + kScalesLAt;
        const std::uint16_t scales_h = read_u16_le(block + kScalesHAt);
        const std::uint8_t* qs = block + kQsAt;
        const std::int8_t* q8 = y[i].qs;
        for (int ib = 0; ib < 8; ++ib) {
            const float dl = d * static_cast<float>(group_scale(scales_l, scales_h, ib) - 32);
            std::int32_t sumi = 0;
            for (int j = 0; j < 16; ++j) {
                sumi += static_cast<std::int32_t>(q8[j + 0]) *
                        static_cast<std::int32_t>(detail::kIq4nlCode[qs[j] & 0x0F]);
                sumi += static_cast<std::int32_t>(q8[j + 16]) *
                        static_cast<std::int32_t>(detail::kIq4nlCode[qs[j] >> 4]);
            }
            sumf += dl * static_cast<float>(sumi);
            qs += 16;
            q8 += 32;
        }
    }
    return sumf;
}

void iq4xs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq4xsBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq4xs_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels