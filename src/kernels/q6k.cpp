// Q6_K 的块布局、反量化与「与 Q8_K 的定点点积」转录自 llama.cpp 的 ggml（提交
// 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026
// The ggml authors；来源登记见 S-46）：
//   ggml/src/ggml-common.h      l.362 block_q6_K（210 字节）
//   ggml/src/ggml-quants.c      l.1939 dequantize_row_q6_K
//   ggml/src/ggml-cpu/quants.c  l.851 ggml_vec_dot_q6_K_q8_K_generic
// 另外参考引擎自己的 include/strata/artifact/dequant.hpp 里也有一份 Q6_K 转写（见 S-34），两处口径一致。
//
// 块内偏移（共 210 字节）：ql 在 0（128 字节，低 4 位）、qh 在 128（64 字节，高 2 位，每字节拆成 4 份
// 按 2 位一组分别补到四个不同的元素上）、scales 在 192（16 个 int8 子尺度）、d 在 208（fp16，**在末尾**）。
// 处理分两个 128 元素半段，每段内 ql 前进 64、qh 前进 32、scales 前进 8；元素 l 用 scales[l/16 + 0]，
// 而同一 l 的另外三个位置分别用 +2、+4、+6 —— 这条最容易记错，故照抄。
#include "kernels/q6k.hpp"

#include "kernels/fp16.hpp"

#include <cstddef>
#include <cstring>

namespace qinfer::kernels {

namespace {

constexpr int kDAt = 208;

}  // namespace

void dequant_q6k_block(const std::uint8_t* block, float* out256) {
    const float d = f16_bits_to_f32(static_cast<std::uint16_t>(block[kDAt] | (block[kDAt + 1] << 8)));
    const std::uint8_t* ql_base = block;        // 128
    const std::uint8_t* qh_base = block + 128;  // 64
    const std::int8_t* sc_base = reinterpret_cast<const std::int8_t*>(block + 192);  // 16
    for (int half = 0; half < 2; ++half) {
        const std::uint8_t* ql = ql_base + 64 * half;
        const std::uint8_t* qh = qh_base + 32 * half;
        const std::int8_t* sc = sc_base + 8 * half;
        float* y = out256 + 128 * half;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = static_cast<int>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
    }
}

float q6k_dot_q8k(const std::uint8_t* w_blocks, const Q8kBlock* y, int n_blocks) {
    // 上游把整数侧按「元素位置 mod 8」分成 8 条部分和（便于向量化），最后再分别乘 d 相加；
    // 这里照抄这个形状，故结果与上游标量版一致。
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = w_blocks + i * kQ6kBlockBytes;
        const std::uint8_t* ql_base = block;
        const std::uint8_t* qh_base = block + 128;
        const std::int8_t* sc_base = reinterpret_cast<const std::int8_t*>(block + 192);
        std::int8_t aux8[kQ6kValuesPerBlock];
        std::int8_t* a = aux8;
        for (int half = 0; half < 2; ++half) {
            const std::uint8_t* ql = ql_base + 64 * half;
            const std::uint8_t* qh = qh_base + 32 * half;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128;
        }
        const std::int8_t* q8 = y[i].qs;
        a = aux8;
        std::int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int is = 0;
        for (int j = 0; j < kQ6kValuesPerBlock / 16; ++j) {
            const int scale = sc_base[is++];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
            q8 += 8;
            a += 8;
        }
        const float d = f16_bits_to_f32(
                            static_cast<std::uint16_t>(block[kDAt] | (block[kDAt + 1] << 8))) *
                        y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * static_cast<float>(aux32[l]);
    }
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

void q6k_gemv_q8k(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out) {
    const std::size_t row_bytes = static_cast<std::size_t>(n_blocks) * kQ6kBlockBytes;
    for (int r = 0; r < n_rows; ++r) {
        out[r] = q6k_dot_q8k(w + static_cast<std::size_t>(r) * row_bytes, y, n_blocks);
    }
}

}  // namespace qinfer::kernels