// ggml_vec_dot_iq2_xs_q8_K_generic 的转录（llama.cpp 提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，
// 2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-35）：
//   ggml/src/ggml-cpu/quants.c l.948
// 与 IQ2_XS 反量化共用 iq2xs_tables.hpp 里的三张表，故此处只引用、不重复表体。
#include "kernels/iq2xs_dot.hpp"

#include "kernels/iq2xs.hpp"
#include "kernels/iq2xs_tables.hpp"

#include <cstddef>

namespace qinfer::kernels {

float iq2xs_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kIq2xsBlockBytes;
        const float d = f16_bits_to_f32(detail::read_u16_le(blk)) * y[i].d;
        const std::uint8_t* q2 = blk + 2;   // 32 个 u16 码字
        const std::uint8_t* sc = blk + 66;  // 8 个尺度字节
        const std::int8_t* q8 = y[i].qs;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::int32_t ls1 = 2 * (sc[ib] & 0x0F) + 1;
            const std::int32_t ls2 = 2 * (sc[ib] >> 4) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const std::uint16_t code = detail::read_u16_le(q2 + 2 * l);
                const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(&detail::kGrid[code & 511u]);
                const std::uint8_t sign = detail::kSigns[code >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * ((sign & detail::kSignBit[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const std::uint16_t code = detail::read_u16_le(q2 + 2 * l);
                const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(&detail::kGrid[code & 511u]);
                const std::uint8_t sign = detail::kSigns[code >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * ((sign & detail::kSignBit[j]) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 8;
        }
        sumf += d * static_cast<float>(bsum);
    }
    return 0.125f * sumf;
}

void iq2xs_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq2xsBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq2xs_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels