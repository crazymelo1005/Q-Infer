// IQ2_XS：IQ 家族的 2.3125 bpw 块格式。一块 256 个值 = 74 字节 = 2 字节 fp16 尺度 + 32 个 u16 码字 + 8 字节尺度。
// 这一档是 §15 算带宽上限的名义档，参考实例的专家档是 IQ2_S / IQ2_XXS / IQ1_M / Q2_0（见 S-36）。
//
// 与 IQ4_NL 的区别在于它不是线性码本，而是「格点 + 符号」。每个 u16 码字拆成两半：
//   * 低 9 位是格点索引，取出 8 个幅度字节（取值 8 / 25 / 43）；
//   * 高 7 位是符号索引，取出一个 8 位模式，位为 1 则对应元素取负。
// 256 个值按 32 个一组（下称 ib），组内 4 个码字；前两个码字用该组尺度的低半字节、后两个用高半字节。
// 尺度带 0.25 因子与 0.5 偏置：db = d · (0.5 + s) · 0.25。整块共用开头那个 fp16 尺度 d。
//
// 三张查表放在 iq2xs_tables.hpp（与 iq2xs_dot.cpp 共用）。反量化函数转录自 llama.cpp 的 ggml
// （提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-35）：ggml/src/ggml-quants.c l.2516 dequantize_row_iq2_xs。
// 上游其余部分与 AVX2/CUDA 核不入库。
#include "kernels/iq2xs.hpp"

#include "kernels/iq2xs_tables.hpp"

namespace qinfer::kernels {

void dequant_iq2xs_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(detail::read_u16_le(block));
    const std::uint8_t* qs = block + 2;       // 32 个 u16 码字
    const std::uint8_t* scales = block + 66;  // 8 字节，每字节低半字节在前、高半字节在后
    for (int ib = 0; ib < 8; ++ib) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib] & 0x0F)) * 0.25f;
        const float db1 = d * (0.5f + static_cast<float>(scales[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const std::uint16_t code = detail::read_u16_le(qs + 2 * (4 * ib + l));
            const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(&detail::kGrid[code & 511u]);
            const std::uint8_t sign = detail::kSigns[code >> 9];
            const float db = (l < 2) ? db0 : db1;
            float* y = out256 + ib * 32 + l * 8;
            for (int j = 0; j < 8; ++j) {
                y[j] = db * static_cast<float>(grid[j]) * ((sign & detail::kSignBit[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

}  // namespace qinfer::kernels