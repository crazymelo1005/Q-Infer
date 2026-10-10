// IQ3_XXS 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-51）：
//   ggml/src/ggml-common.h      l.407 block_iq3_xxs（98 字节）、l.1017 iq3xxs_grid（256 项）
//   ggml/src/ggml-quants.c      l.2575 dequantize_row_iq3_xxs
//   ggml/src/ggml-cpu/quants.c  l.1050 ggml_vec_dot_iq3_xxs_q8_K_generic
// 格点表在 iq3xxs_tables.hpp，符号表与掩码复用 iq2xs_tables.hpp 的 kSigns / kSignBit。
//
// 与同族的 IQ2_XXS、以及与 IQ3_S 的三处不同，照搬必错：
//   ① 每个 32 值组配一个 u32（在块内偏移 66 + 4·ib32）：高 4 位是子尺度增量 n，低 28 位分成 4 个
//      7 位符号索引（查 ksigns_iq2xs 得 8 位符号模式）——IQ2_XXS 是「高 4 位尺度 + 4 个 7 位符号」
//      的 u32，但它的格点索引是 8 位且表项是 uint64；IQ3_S 的索引是 9 位、尺度是两个组共用一个字节；
//   ② 格点索引就是 qs 的一个字节（8 位，无高位拼接），表是 256 项 uint32；
//   ③ 幅度字节是 4 的倍数那一组（4/12/…/62），子尺度是 d(0.5+n)/2，故点积末尾要乘 0.25
//      （点积里用的是 (2n+1)，与反量化的 d(0.5+n)/2 差 4 倍）——IQ2_* 家族末尾是 0.125，
//      IQ3_S 则没有末尾因子。
#include "kernels/iq3xxs.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq2xs_tables.hpp"
#include "kernels/iq3xxs_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

constexpr int kQhAt = 2;
constexpr int kGasAt = 66;  // scales_and_signs

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) |
                                      (static_cast<std::uint32_t>(p[3]) << 24));
}

// 格点项的第 j 个幅度字节。按位移显式取，不依赖主机端序。
int grid_byte(int index, int j) {
    return static_cast<int>((detail::kIq3xxsGrid[index] >> (8 * j)) & 0xFFu);
}

int sign_int(std::uint8_t byte, int j) { return (byte & detail::kSignBit[j]) ? -1 : 1; }

}  // namespace

void dequant_iq3xxs_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(detail::read_u16_le(block));
    const std::uint8_t* qs = block + kQhAt;
    const std::uint8_t* gas = block + kGasAt;
    float* y = out256;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t aux32 = read_u32_le(gas + 4 * ib32);
        const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = detail::kSigns[(aux32 >> (7 * l)) & 127];
            const int i1 = qs[2 * l + 0];
            const int i2 = qs[2 * l + 1];
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * static_cast<float>(grid_byte(i1, j)) *
                           static_cast<float>(sign_int(signs, j));
                y[j + 4] = db * static_cast<float>(grid_byte(i2, j)) *
                           static_cast<float>(sign_int(signs, j + 4));
            }
            y += 8;
        }
        qs += 8;
    }
}

float iq3xxs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = w_blocks + static_cast<std::size_t>(i) * kIq3xxsBlockBytes;
        const float d = f16_bits_to_f32(detail::read_u16_le(block)) * y[i].d;
        const std::uint8_t* qs = block + kQhAt;
        const std::uint8_t* gas = block + kGasAt;
        const std::int8_t* q8 = y[i].qs;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint32_t aux32 = read_u32_le(gas + 4 * ib32);
            const std::int32_t ls = 2 * static_cast<std::int32_t>(aux32 >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = detail::kSigns[(aux32 >> (7 * l)) & 127];
                const int i1 = qs[2 * l + 0];
                const int i2 = qs[2 * l + 1];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid_byte(i1, j) * static_cast<std::int32_t>(q8[j + 0]) *
                            sign_int(signs, j);
                    sumi += grid_byte(i2, j) * static_cast<std::int32_t>(q8[j + 4]) *
                            sign_int(signs, j + 4);
                }
                q8 += 8;
            }
            qs += 8;
            bsum += sumi * ls;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.25f * sumf;
}

void iq3xxs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                     float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq3xxsBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq3xxs_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels