// IQ3_S 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-47）：
//   ggml/src/ggml-common.h      l.415 block_iq3_s（110 字节）、l.414 IQ3S_N_SCALE = QK_K/64、
//                               l.1052 GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)
//   ggml/src/ggml-quants.c      l.2607 dequantize_row_iq3_s
//   ggml/src/ggml-cpu/quants.c  l.1094 ggml_vec_dot_iq3_s_q8_K_generic
// 格点表在 iq3s_tables.hpp，符号位掩码复用 iq2xs_tables.hpp 的 kSignBit（IQ3_S 与 IQ2 家族共用
// kmask_iq2xs）；小端 u16 读取器也在那里。
//
// 与 IQ2 家族的三处不同，照搬必错：
//   ① 格点表是 uint32_t（每项 4 个幅度字节，取值 1..15 的奇数），不是 uint64 —— 按 8 字节取会
//      读到相邻两项；
//   ② 格点索引是 9 位：低 8 位来自 qs 的 8 个连续字节，高 1 位来自 qh 的同一个字节。上游把这对
//      写作两次位移（`qh << (8-2*l)` 与 `qh << (7-2*l)`），统一起来就是「组内第 m 个索引用
//      qh 的位 m」，即 `(qh << (8-m)) & 256`——照抄两个公式容易把 m 与 l 混掉；
//   ③ 子尺度是「两个 32 值组共用一个字节」：低半字节给前一组、高半字节给后一组，ls = 2n + 1
//      （与 IQ2_XS/IQ2_S 的每 16 值一个尺度、IQ2_XXS 的每 32 值一个都不同）。
// 点积没有 IQ2 家族那个末尾 0.125 因子：整数和乘 d 即得，d = fp16(权重) × Q8_K 的尺度。
#include "kernels/iq3s.hpp"

#include "kernels/fp16.hpp"
#include "kernels/iq2xs_tables.hpp"
#include "kernels/iq3s_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

namespace {

constexpr int kQsAt = 2;
constexpr int kQhAt = 66;
constexpr int kSignsAt = 74;
constexpr int kScalesAt = 106;

// 组内第 m 个（m = 0..7）9 位格点索引：低 8 位是 qs[m]，高 1 位是 qh 字节的位 m。
int grid_index(const std::uint8_t* qs, std::uint8_t qh_byte, int m) {
    return static_cast<int>(qs[m]) | (static_cast<int>(qh_byte << (8 - m)) & 256);
}

// 格点项的第 j 个幅度字节（1..15 的奇数）。按位移显式取，不依赖主机端序。
int grid_byte(int index, int j) {
    return static_cast<int>((detail::kIq3sGrid[index] >> (8 * j)) & 0xFFu);
}

int sign_int(std::uint8_t byte, int j) { return (byte & detail::kSignBit[j]) ? -1 : 1; }

}  // namespace

void dequant_iq3s_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(detail::read_u16_le(block));
    const std::uint8_t* qs = block + kQsAt;
    const std::uint8_t* qh = block + kQhAt;
    const std::uint8_t* signs = block + kSignsAt;
    const std::uint8_t* scales = block + kScalesAt;
    float* y = out256;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] & 0x0F));
        const float db2 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] >> 4));
        for (int half = 0; half < 2; ++half) {
            const float db = (half == 0) ? db1 : db2;
            const std::uint8_t qhb = qh[half];
            for (int l = 0; l < 4; ++l) {
                const int i1 = grid_index(qs, qhb, 2 * l);
                const int i2 = grid_index(qs, qhb, 2 * l + 1);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * static_cast<float>(grid_byte(i1, j)) *
                               static_cast<float>(sign_int(signs[l], j));
                    y[j + 4] = db * static_cast<float>(grid_byte(i2, j)) *
                               static_cast<float>(sign_int(signs[l], j + 4));
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

float iq3s_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = w_blocks + static_cast<std::size_t>(i) * kIq3sBlockBytes;
        const float d = f16_bits_to_f32(detail::read_u16_le(block)) * y[i].d;
        const std::uint8_t* qs = block + kQsAt;
        const std::uint8_t* qh = block + kQhAt;
        const std::uint8_t* signs = block + kSignsAt;
        const std::uint8_t* scales = block + kScalesAt;
        const std::int8_t* q8 = y[i].qs;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const std::int32_t ls1 = 2 * static_cast<std::int32_t>(scales[ib32 / 2] & 0x0Fu) + 1;
            const std::int32_t ls2 = 2 * static_cast<std::int32_t>(scales[ib32 / 2] >> 4) + 1;
            std::int32_t sumi = 0;
            for (int half = 0; half < 2; ++half) {
                const std::uint8_t qhb = qh[half];
                for (int l = 0; l < 4; ++l) {
                    const int i1 = grid_index(qs, qhb, 2 * l);
                    const int i2 = grid_index(qs, qhb, 2 * l + 1);
                    for (int j = 0; j < 4; ++j) {
                        sumi += grid_byte(i1, j) * static_cast<std::int32_t>(q8[j + 0]) *
                                sign_int(signs[l], j);
                        sumi += grid_byte(i2, j) * static_cast<std::int32_t>(q8[j + 4]) *
                                sign_int(signs[l], j + 4);
                    }
                    q8 += 8;
                }
                qs += 8;
                signs += 4;
                bsum += sumi * (half == 0 ? ls1 : ls2);
                sumi = 0;
            }
            qh += 2;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return sumf;
}

void iq3s_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq3sBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq3s_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels