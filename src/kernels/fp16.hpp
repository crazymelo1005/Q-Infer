// IEEE 半精度位型 ↔ 单精度。GGUF 里量化块的尺度就是 fp16，反量化第一件事就是把它展开。
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace qinfer::kernels {

// 逐位精确：正规数走 ldexp(1 + m/1024, e-15)，次正规数走 ldexp(m, -24)，两者都无中间舍入。
inline float f16_bits_to_f32(std::uint16_t bits) {
    const int sign = (bits >> 15) ? -1 : 1;
    const int exp = (bits >> 10) & 0x1F;
    const int man = bits & 0x3FF;
    if (exp == 0) {
        if (man == 0) return sign < 0 ? -0.0f : 0.0f;
        return static_cast<float>(sign * std::ldexp(static_cast<double>(man), -24));
    }
    if (exp == 31) {
        if (man != 0) return sign < 0 ? -std::nanf("") : std::nanf("");
        return sign < 0 ? -INFINITY : INFINITY;
    }
    return static_cast<float>(sign * std::ldexp(1.0 + man / 1024.0, exp - 15));
}

// 单精度 -> 半精度位型。转录自 ggml 的 ggml_compute_fp32_to_fp16（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，MIT；来源登记见 S-43）：ggml/src/ggml-impl.h l.419。
// 它靠「加一个魔数再取位」实现就近舍入到偶数，且无分支地覆盖正规、次正规与上溢三种情形；
// 非有限值统一归一成 0x7E00（尾数信息丢失，符号位保留）。这个函数在量化器的尺度转换里要用，
// 不能凭记忆手写舍入规则。
inline std::uint16_t f32_to_f16_bits(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

    std::uint32_t w;
    std::memcpy(&w, &f, 4);
    const std::uint32_t shl1_w = w + w;
    const std::uint32_t sign = w & 0x80000000u;
    std::uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;

    const std::uint32_t magic = (bias >> 1) + 0x07800000u;
    float m;
    std::memcpy(&m, &magic, 4);
    base = m + base;

    std::uint32_t bits;
    std::memcpy(&bits, &base, 4);
    const std::uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const std::uint32_t mantissa_bits = bits & 0x00000FFFu;
    const std::uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<std::uint16_t>((sign >> 16) |
                                      (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

}  // namespace qinfer::kernels
