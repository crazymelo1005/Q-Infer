// IEEE 半精度位型 ↔ 单精度。GGUF 里量化块的尺度就是 fp16，反量化第一件事就是把它展开。
#pragma once

#include <cmath>
#include <cstdint>

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
        if (man != 0) return std::nanf("");
        return sign < 0 ? -INFINITY : INFINITY;
    }
    return static_cast<float>(sign * std::ldexp(1.0 + man / 1024.0, exp - 15));
}

}  // namespace qinfer::kernels
