// IEEE 半精度位型 ↔ 单精度（fp16.hpp）与 bfloat16 位型 ↔ 单精度（本文件）。
// bf16 只需截断到高 16 位、加上就近舍入到偶数的进位，转换本身是精确可逆的（bf16 是 f32 的高半）。
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

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

// bf16 权重的一行点积式投影：`out[o] = Σ_i bf16(w[o·cols+i]) · bf16(x[i])`，累加在 float 上。
//
// 两处口径，都有实测来源：**激活也要按 bf16 舍入**（权重是 bf16 时 ggml 会把激活转到权重的 vec_dot_type；
// 引擎自注此处的路由投影若改用 fp16 会引入 8.100e-03 的误差并翻掉选择，见 S-44）；累加在 float 上。
// 三处用同一份：路由门控（S-44）、以及 QSA 索引器的两个投影（那份 artifact 里它们是 BF16，见 S-54）。
//
// 分成三步是为了**按行并行**（见 `experts/ffn` 的 `run_bf16_gemv`）：并行要切行，而激活的舍入与行数
// 无关，切行时每块各舍一遍太浪费，故把它提成单独一步、一次做好。注意：**这一步本身不带来提速**——
// 单线程实测「提出去」与「逐行各算一遍」没有差别（27 对 28 ms），因为编译器本来就会把那个纯粹计算
// 提到循环外；拆开只为并行服务，且与原来的算术**逐位相同**（同一个舍入值、同一份乘加顺序）。
inline void bf16_round_row(const float* x, std::uint64_t cols, std::uint16_t* dst) {
    for (std::uint64_t i = 0; i < cols; ++i) dst[i] = f32_to_bf16_bits(x[i]);
}

// 用已经舍好的一行激活做若干行的点积（`rows` 行、从 `w_bf16` 起）。
inline void bf16_gemv_rows(const std::uint16_t* w_bf16, std::uint64_t rows, std::uint64_t cols,
                           const std::uint16_t* x_bf16, float* out) {
    for (std::uint64_t o = 0; o < rows; ++o) {
        const std::uint16_t* row = w_bf16 + o * cols;
        float acc = 0.0f;
        for (std::uint64_t i = 0; i < cols; ++i) {
            acc += bf16_bits_to_f32(row[i]) * bf16_bits_to_f32(x_bf16[i]);
        }
        out[o] = acc;
    }
}

inline void bf16_gemv(const std::uint16_t* w_bf16, std::uint64_t rows, std::uint64_t cols,
                      const float* x, float* out) {
    std::vector<std::uint16_t> xb(static_cast<std::size_t>(cols));
    bf16_round_row(x, cols, xb.data());
    bf16_gemv_rows(w_bf16, rows, cols, xb.data(), out);
}

}  // namespace qinfer::kernels