// IQ2_XXS 块的回归（反量化与 IQ2_XXS × Q8_K 定点点积）。不引第三方框架。
//
// 四路证据：
//   1. 手工四块，期望值是手算的——钉死块内偏移（d/0、qs/2，每个 ib32 占 8 字节）、
//      尺度在组内第二个 u32 的高 4 位、符号索引在其低 28 位的 4 个 7 位槽、以及 0.25 因子与 0.5 偏置。
//   2. 真字节的结构不变量：模型里 blk.1.ffn_gate_exps.weight 的前 16 个块，每个输出值必须是
//      ±db·g（g ∈ {8, 25, 43}）。这条不依赖任何外部期望值。
//   3. 真字节的回归锁：同一批字节，前 16 值与第 7 块切片必须对上 gguf-py 的独立实现（见 S-41），
//      sum / sumabs 一并核。
//   4. 点积与浮点路径交叉核对，另核逐行 GEMV 与单行点积一致（行距 n_blocks × 66 字节）。
#include "kernels/iq2xxs.hpp"

#include "kernels/fp16.hpp"
#include "check.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

// 由 measure/iq2xxs_oracle.py 生成：kIq2xxsRealHex / kIq2xxsRealFirst16 / kIq2xxsRealBlock7_64 /
// kIq2xxsRealSum / kIq2xxsRealSumAbs。
#include "iq2xxs_oracle_data.inc"

constexpr int kRealBlocks = 16;

std::vector<std::uint8_t> from_hex(const std::string& hex) {
    CHECK(hex.size() % 2 == 0);
    std::vector<std::uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        CHECK(false && "bad hex digit");
        return 0;
    };
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(nib(hex[i]) * 16 + nib(hex[i + 1])));
    }
    return out;
}

// 一块：d 为给定 fp16，qs 的 64 个字节全 0（调用方再按需 poke 具体字节）。
std::vector<std::uint8_t> flat_iq2xxs(std::uint16_t d_bits) {
    std::vector<std::uint8_t> b(kIq2xxsBlockBytes, 0);
    b[0] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[1] = static_cast<std::uint8_t>(d_bits >> 8);
    return b;
}

void test_handmade() {
    // 全 0 的 qs：格点索引全 0（全 8）、尺度 0、符号索引全 0（全正）。
    // d = 1.0 -> db = (0.5 + 0)·0.25 = 0.125，故每个值 = 0.125 · 8 = 1.0。
    {
        const auto b = flat_iq2xxs(0x3C00);
        float out[kIq2xxsValuesPerBlock];
        dequant_iq2xxs_block(b.data(), out);
        for (int k = 0; k < kIq2xxsValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != 0x3F800000u) {
                std::printf("FAIL handmade all-1.0 at %d: got %g\n", k, static_cast<double>(out[k]));
                CHECK(false);
            }
        }
    }
    // 尺度 = 15：把每个组第 2 个 u32 的最高字节置 0xF0（即 w1 的高 4 位）。
    // db = 1.0 · 15.5 · 0.25 = 3.875，故每个值 = 3.875 · 8 = 31.0。
    {
        auto b = flat_iq2xxs(0x3C00);
        for (int ib = 0; ib < 8; ++ib) b[2 + 8 * ib + 7] = 0xF0;
        float out[kIq2xxsValuesPerBlock];
        dequant_iq2xxs_block(b.data(), out);
        for (int k = 0; k < kIq2xxsValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(31.0f)) {
                std::printf("FAIL handmade scale-15 at %d: got %g\n", k,
                            static_cast<double>(out[k]));
                CHECK(false);
            }
        }
    }
    // 符号索引 1（w1 最低字节置 0x01）-> ksigns[1] = 0x81 -> 组内第 0 与第 7 个元素取负。
    {
        auto b = flat_iq2xxs(0x3C00);
        for (int ib = 0; ib < 8; ++ib) b[2 + 8 * ib + 4] = 0x01;
        float out[kIq2xxsValuesPerBlock];
        dequant_iq2xxs_block(b.data(), out);
        const std::uint32_t pos = std::bit_cast<std::uint32_t>(1.0f);
        const std::uint32_t neg = std::bit_cast<std::uint32_t>(-1.0f);
        for (int ib = 0; ib < 8; ++ib) {
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t want = (j == 0 || j == 7) ? neg : pos;
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[ib * 32 + j]);
                if (got != want) {
                    std::printf("FAIL handmade sign ib=%d j=%d: got 0x%08x\n", ib, j, got);
                    CHECK(false);
                }
            }
            // 该组第 2/3/4 个码字（元素 8..31）符号索引仍为 0 -> 全正。
            for (int k = ib * 32 + 8; k < ib * 32 + 32; ++k) {
                CHECK(std::bit_cast<std::uint32_t>(out[k]) == pos);
            }
        }
    }
    // 格点索引：把每个组的第 1 个字节置 1 -> 格点 1 = 字节序 [43,8,8,8,8,8,8,8]，
    // 故组内第 0 个元素 = 0.125 · 43 = 5.375，其余 = 1.0。
    {
        auto b = flat_iq2xxs(0x3C00);
        for (int ib = 0; ib < 8; ++ib) b[2 + 8 * ib] = 1;
        float out[kIq2xxsValuesPerBlock];
        dequant_iq2xxs_block(b.data(), out);
        for (int ib = 0; ib < 8; ++ib) {
            const float got0 = out[ib * 32];
            if (std::bit_cast<std::uint32_t>(got0) != std::bit_cast<std::uint32_t>(5.375f)) {
                std::printf("FAIL handmade grid ib=%d: got %g\n", ib, static_cast<double>(got0));
                CHECK(false);
            }
            for (int j = 1; j < 8; ++j) {
                CHECK(std::bit_cast<std::uint32_t>(out[ib * 32 + j]) == 0x3F800000u);
            }
        }
    }
}

void test_handmade_dot() {
    // 权重 d = 1.0、qs 全 0 -> 每个权重 1.0；激活 Q8_K d = 1.0、qs 全 1 -> 每个 1.0。
    // 整数侧每块：8 组 × 4 码字 × 8 元素 × 8（格点） × 1（ls = 2·0+1） = 2048，乘 d·0.125 -> 256。
    const auto w = flat_iq2xxs(0x3C00);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    CHECK(std::bit_cast<std::uint32_t>(iq2xxs_dot_q8k(w.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(256.0f));
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq2xxsRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq2xxsBlockBytes));

    float out[kIq2xxsValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq2xxsBlockBytes;
        dequant_iq2xxs_block(blk, out);
        const float d = f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* base = blk + 2 + 8 * ib;
            const std::uint32_t w1 = static_cast<std::uint32_t>(base[4]) |
                                     (static_cast<std::uint32_t>(base[5]) << 8) |
                                     (static_cast<std::uint32_t>(base[6]) << 16) |
                                     (static_cast<std::uint32_t>(base[7]) << 24);
            const float db = d * (0.5f + static_cast<float>(w1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                for (int j = 0; j < 8; ++j) {
                    const float v = out[ib * 32 + l * 8 + j];
                    const bool ok = std::fabs(v) == std::fabs(db * 8.0f) ||
                                    std::fabs(v) == std::fabs(db * 25.0f) ||
                                    std::fabs(v) == std::fabs(db * 43.0f);
                    if (!ok) {
                        std::printf("FAIL invariant block %d ib %d l %d j %d: v=%.9g db=%.9g\n", b, ib,
                                    l, j, static_cast<double>(v), static_cast<double>(db));
                        CHECK(false);
                    }
                    sum += static_cast<double>(v);
                    sumabs += std::fabs(static_cast<double>(v));
                }
            }
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kIq2xxsRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq2xxsRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq2xxs_block(w.data() + 7 * kIq2xxsBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[64 + j]);
        if (got != kIq2xxsRealBlock7_64[j]) {
            std::printf("FAIL block7_64 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq2xxsRealBlock7_64[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq2xxsRealSumAbs) > 0 ? std::fabs(kIq2xxsRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq2xxsRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq2xxsRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq2xxsRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq2xxsRealSumAbs);
        CHECK(false);
    }
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq2xxsRealHex);
    // 激活逐元素变化：常量激活会让「哪个元素配哪个 q8」不可辨识。
    std::vector<Q8kBlock> y(kRealBlocks);
    for (int b = 0; b < kRealBlocks; ++b) {
        y[b].d = 0.75f;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            y[b].qs[j] = static_cast<std::int8_t>(((b * 11 + j * 3) % 251) - 125);
        }
        for (int g = 0; g < kQ8kBlockElems / 16; ++g) {
            int s = 0;
            for (int i = 0; i < 16; ++i) s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = static_cast<std::int16_t>(s);
        }
    }

    const float fused = iq2xxs_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq2xxsValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq2xxs_block(w.data() + b * kIq2xxsBlockBytes, wv);
        q8k_dequant_block(y[b], yv);
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            const double term = static_cast<double>(wv[j]) * static_cast<double>(yv[j]);
            acc += term;
            mag += std::fabs(term);
        }
    }
    const double rel = std::fabs(static_cast<double>(fused) - acc) / (mag > 0.0 ? mag : 1.0);
    if (!(rel < 1e-6)) {
        std::printf("FAIL fused vs float path: fused %.9g float %.9g rel %.3g\n",
                    static_cast<double>(fused), acc, rel);
        CHECK(false);
    }

    float gemv[2];
    iq2xxs_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq2xxs_dot_q8k(w.data() + r * 8 * kIq2xxsBlockBytes, y.data(), 8);
        if (std::bit_cast<std::uint32_t>(gemv[r]) != std::bit_cast<std::uint32_t>(one)) {
            std::printf("FAIL gemv row %d: got 0x%08x want 0x%08x\n", r,
                        std::bit_cast<std::uint32_t>(gemv[r]), std::bit_cast<std::uint32_t>(one));
            CHECK(false);
        }
    }
}

}  // namespace

int main() {
    test_handmade();
    test_handmade_dot();
    test_real_bytes_invariant_and_regression();
    test_dot_and_row_stride();
    std::puts("iq2xxs: block dequant and IQ2_XXSxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}