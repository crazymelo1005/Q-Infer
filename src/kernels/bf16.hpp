// IEEE 半精度位型 ↔ 单精度（fp16.hpp）与 bfloat16 位型 ↔ 单精度（本文件）。
// bf16 只需截断到高 16 位、加上就近舍入到偶数的进位，转换本身是精确可逆的（bf16 是 f32 的高半）。
#pragma once

#include <cstdint>
#include <cstring>

namespace qinfer::kernels {

// bf16 -> f32：左移 16 位即成 f32 位型，无舍入。
inline float bf16_bits_to_f32(std::uint16_t bits) {
    const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// f32 -> bf16 位型。转录自 ggml 的 ggml_compute_fp32_to_bf16（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，MIT；来源登记见 S-44）：ggml/src/ggml-impl.h l.623。
// 就是「加 0x7FFF 再加尾数最低位」做就近舍入到偶数，然后取高 16 位；NaN 强制置为静默 NaN。
// 路由器必须用它而不是 fp16：引擎自注「此处若改用 fp16 激活会引入 8.100e-03 的误差并翻掉选择」。
inline std::uint16_t f32_to_bf16_bits(float s) {
    std::uint32_t u;
    std::memcpy(&u, &s, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN
        return static_cast<std::uint16_t>((u >> 16) | 64u);
    }
    return static_cast<std::uint16_t>((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

}  // namespace qinfer::kernels