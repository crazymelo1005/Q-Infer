// IQ4_NL 的码本、反量化与「与 Q8_0 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-40）：
//   ggml/src/ggml-common.h      l.447 block_iq4_nl（QK4_NL = 32）/ l.1120 kvalues_iq4nl
//   ggml/src/ggml-quants.c      l.2725 dequantize_row_iq4_nl
//   ggml/src/ggml-cpu/quants.c  l.104 ggml_vec_dot_iq4_nl_q8_0_generic（QK4_NL == QK8_0）
// 只取这几处。码本与「分裂式半字节序」这两件事本仓库此前已按其参考引擎的实现记录过（见 S-34），
// 两处口径一致。
#include "kernels/iq4nl.hpp"

#include "kernels/iq4nl_tables.hpp"

namespace qinfer::kernels {

namespace {

// kvalues_iq4nl 在 iq4nl_tables.hpp（IQ4_XS 共用同一张表）。
constexpr const std::int8_t* kCodebook = detail::kIq4nlCode;

std::uint16_t read_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

int iq4nl_code(int code) { return kCodebook[code & 15]; }

void dequant_iq4nl_block(const std::uint8_t* block, float* out32) {
    const float d = f16_bits_to_f32(read_u16(block));
    const std::uint8_t* qs = block + 2;
    for (int j = 0; j < kIq4nlValuesPerBlock / 2; ++j) {
        out32[j] = d * static_cast<float>(kCodebook[qs[j] & 0x0F]);
        out32[j + kIq4nlValuesPerBlock / 2] = d * static_cast<float>(kCodebook[qs[j] >> 4]);
    }
}

void dequant_iq4nl_row(const std::uint8_t* row, float* out160) {
    constexpr int kBlocksPerRow = kIq4nlValuesPerRow / kIq4nlValuesPerBlock;
    for (int b = 0; b < kBlocksPerRow; ++b) {
        dequant_iq4nl_block(row + b * kIq4nlBlockBytes, out160 + b * kIq4nlValuesPerBlock);
    }
}

float iq4nl_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks) {
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = w_blocks + i * kIq4nlBlockBytes;
        const float d = f16_bits_to_f32(y[i].d_bits) * f16_bits_to_f32(read_u16(blk));
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < kIq4nlValuesPerBlock / 2; ++j) {
            sumi1 += y[i].qs[j] * kCodebook[blk[2 + j] & 0x0F];
            sumi2 += y[i].qs[j + kIq4nlValuesPerBlock / 2] * kCodebook[blk[2 + j] >> 4];
        }
        sumf += d * static_cast<float>(sumi1 + sumi2);
    }
    return sumf;
}

void iq4nl_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kIq4nlBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = iq4nl_dot_q8_0(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels