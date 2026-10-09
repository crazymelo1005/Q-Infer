#include "kernels/iq4nl.hpp"

namespace qinfer::kernels {

namespace {

// kvalues_iq4nl：IQ4_NL 的 16 个格点，来自 ggml 的 IQ4_NL 定义（本仓库的记录见 S-34）。
constexpr std::int8_t kCodebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                       1, 13, 25, 38, 53, 69, 89, 113};

}  // namespace

int iq4nl_code(int code) { return kCodebook[code & 15]; }

void dequant_iq4nl_row(const std::uint8_t* row, float* out160) {
    constexpr int kBlockBytes = 18;   // 2 字节尺度 + 16 字节半字节
    constexpr int kBlockValues = 32;
    for (int b = 0; b < kIq4nlValuesPerRow / kBlockValues; ++b) {
        const std::uint8_t* blk = row + b * kBlockBytes;
        const std::uint16_t dbits = static_cast<std::uint16_t>(blk[0] | (blk[1] << 8));
        const float d = f16_bits_to_f32(dbits);
        const std::uint8_t* qs = blk + 2;
        for (int j = 0; j < 16; ++j) {
            out160[b * kBlockValues + j] = d * static_cast<float>(kCodebook[qs[j] & 0x0F]);
            out160[b * kBlockValues + j + 16] = d * static_cast<float>(kCodebook[qs[j] >> 4]);
        }
    }
}

}  // namespace qinfer::kernels
