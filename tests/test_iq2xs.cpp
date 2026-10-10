// IQ2_XS 单块反量化的回归。不引第三方框架。
//
// 三路证据：
//   1. 手工构造的四块，期望值是手算的——它钉死格点查表、符号掩码、分裂式尺度（同一 32 组内前两个
//      码字用低半字节、后两个用高半字节）以及 db = d·(0.5+s)·0.25 这条带 0.25 因子与 0.5 偏置的尺度。
//   2. 确定性合成块的回归锁，期望值由 gguf-py 的独立实现给出（llama.cpp 检出内的 numpy 路径，
//      见 measure/iq2xs_oracle.py 与 S-35）。逐位比较，不留容差。
//   3. 结构不变量：任意块的每个输出都必须是 ±db·{8,25,43} 之一（db 由该块自己的尺度算出）。
//      这一条不依赖任何外部期望值，专门抓步长、半字节序与符号位置错。
#include "kernels/iq2xs.hpp"

#include "check.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

// 由 measure/iq2xs_oracle.py 生成：kOracleBlockHex / kOracleBits[48] / kOracleL1。
#include "iq2xs_oracle_data.inc"

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

void expect_bits(const float* got, const std::uint32_t* want, const int* idx, int n, const char* what) {
    for (int t = 0; t < n; ++t) {
        const std::uint32_t g = std::bit_cast<std::uint32_t>(got[idx[t]]);
        if (g != want[t]) {
            std::printf("FAIL %s: index %d got 0x%08x want 0x%08x\n", what, idx[t], g, want[t]);
            CHECK(false);
        }
    }
}

// 一块：d 为给定 fp16，32 个码字全是 code，8 个尺度字节全是 scales。
std::vector<std::uint8_t> flat_block(std::uint16_t d_bits, std::uint16_t code, std::uint8_t scales) {
    std::vector<std::uint8_t> b(kIq2xsBlockBytes, scales);
    b[0] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[1] = static_cast<std::uint8_t>(d_bits >> 8);
    for (int w = 0; w < 32; ++w) {
        b[2 + 2 * w] = static_cast<std::uint8_t>(code & 0xFF);
        b[3 + 2 * w] = static_cast<std::uint8_t>(code >> 8);
    }
    return b;
}

void test_handmade_blocks() {
    // 块 A：d = 2.0（fp16 0x4000），code = 0（格点 0 = 全 8，符号 0 = 全正），尺度 0 -> db = 0.25。
    //       => 每个输出 = 0.25 · 8 = 2.0。
    {
        const auto b = flat_block(0x4000, 0, 0x00);
        float out[kIq2xsValuesPerBlock];
        dequant_iq2xs_block(b.data(), out);
        for (int k = 0; k < kIq2xsValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != 0x40000000u) {
                std::printf("FAIL blockA at %d: got %g\n", k, static_cast<double>(out[k]));
                CHECK(false);
            }
        }
    }
    // 块 B：尺度 0x0F -> 低半字节 15（l = 0,1）、高半字节 0（l = 2,3）。
    //       db0 = 2·15.5·0.25 = 7.75 -> 62.0；db1 = 2·0.5·0.25 = 0.25 -> 2.0。
    {
        const auto b = flat_block(0x4000, 0, 0x0F);
        float out[kIq2xsValuesPerBlock];
        dequant_iq2xs_block(b.data(), out);
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                const float want = (l < 2) ? 62.0f : 2.0f;
                for (int j = 0; j < 8; ++j) {
                    const float got = out[ib * 32 + l * 8 + j];
                    if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(want)) {
                        std::printf("FAIL blockB at ib=%d l=%d j=%d: got %g want %g\n", ib, l, j,
                                    static_cast<double>(got), static_cast<double>(want));
                        CHECK(false);
                    }
                }
            }
        }
    }
    // 块 C：code = 1 -> 格点 1 = 字节序 [43, 8, 8, 8, 8, 8, 8, 8]，符号 0 -> 全正。
    //       db = 0.25 -> 组内第 0 个 = 10.75，其余 = 2.0。钉死格点字节序。
    {
        const auto b = flat_block(0x4000, 1, 0x00);
        float out[kIq2xsValuesPerBlock];
        dequant_iq2xs_block(b.data(), out);
        for (int g = 0; g < 32; ++g) {
            const float want0 = 10.75f;
            const float got0 = out[g * 8];
            if (std::bit_cast<std::uint32_t>(got0) != std::bit_cast<std::uint32_t>(want0)) {
                std::printf("FAIL blockC grid[0] at g=%d: got %g\n", g, static_cast<double>(got0));
                CHECK(false);
            }
            for (int j = 1; j < 8; ++j) {
                const float got = out[g * 8 + j];
                if (std::bit_cast<std::uint32_t>(got) != 0x40000000u) {
                    std::printf("FAIL blockC grid[%d] at g=%d: got %g\n", j, g, static_cast<double>(got));
                    CHECK(false);
                }
            }
        }
    }
    // 块 D：code = 512 -> 格点 0（全 8），符号索引 1 -> ksigns[1] = 0x81 -> j = 0 与 j = 7 取负。
    {
        const auto b = flat_block(0x4000, 512, 0x00);
        float out[kIq2xsValuesPerBlock];
        dequant_iq2xs_block(b.data(), out);
        const std::uint32_t neg = std::bit_cast<std::uint32_t>(-2.0f);
        const std::uint32_t pos = std::bit_cast<std::uint32_t>(2.0f);
        for (int g = 0; g < 32; ++g) {
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[g * 8 + j]);
                const std::uint32_t want = (j == 0 || j == 7) ? neg : pos;
                if (got != want) {
                    std::printf("FAIL blockD j=%d at g=%d: got 0x%08x\n", j, g, got);
                    CHECK(false);
                }
            }
        }
    }
}

void test_oracle_regression() {
    const std::vector<std::uint8_t> block = from_hex(kOracleBlockHex);
    CHECK(block.size() == kIq2xsBlockBytes);

    float out[kIq2xsValuesPerBlock];
    dequant_iq2xs_block(block.data(), out);

    int idx[48];
    for (int i = 0; i < 16; ++i) idx[i] = i;
    for (int i = 0; i < 16; ++i) idx[16 + i] = 128 + i;
    for (int i = 0; i < 16; ++i) idx[32 + i] = 240 + i;
    expect_bits(out, kOracleBits, idx, 48, "oracle bits");

    double l1 = 0.0;
    for (int k = 0; k < kIq2xsValuesPerBlock; ++k) l1 += std::fabs(static_cast<double>(out[k]));
    if (std::fabs(l1 - kOracleL1) > 1e-9 * kOracleL1) {
        std::printf("FAIL oracle L1: got %.17g want %.17g\n", l1, kOracleL1);
        CHECK(false);
    }
}

void test_structural_invariant() {
    // 一份伪随机块：LCG 填满 74 字节，再按该块自己的尺度算 db0/db1，断言每个输出落在 ±db·{8,25,43}。
    std::vector<std::uint8_t> b(kIq2xsBlockBytes);
    std::uint32_t s = 0x12345678u;
    for (auto& x : b) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<std::uint8_t>(s >> 24);
    }
    float out[kIq2xsValuesPerBlock];
    dequant_iq2xs_block(b.data(), out);

    const float d = f16_bits_to_f32(static_cast<std::uint16_t>(b[0] | (b[1] << 8)));
    for (int ib = 0; ib < 8; ++ib) {
        const float db0 = d * (0.5f + static_cast<float>(b[66 + ib] & 0x0F)) * 0.25f;
        const float db1 = d * (0.5f + static_cast<float>(b[66 + ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float db = (l < 2) ? db0 : db1;
            const float mags[3] = {db * 8.0f, db * 25.0f, db * 43.0f};
            for (int j = 0; j < 8; ++j) {
                const float v = std::fabs(out[ib * 32 + l * 8 + j]);
                bool ok = false;
                for (const float m : mags) ok = ok || v == std::fabs(m);
                if (!ok) {
                    std::printf("FAIL invariant ib=%d l=%d j=%d: |v|=%.9g db=%g\n", ib, l, j,
                                static_cast<double>(v), static_cast<double>(db));
                    CHECK(false);
                }
            }
        }
    }
}

}  // namespace

int main() {
    test_handmade_blocks();
    test_oracle_regression();
    test_structural_invariant();
    std::puts("iq2xs: block dequant pinned by handmade, oracle and structural checks");
    return 0;
}