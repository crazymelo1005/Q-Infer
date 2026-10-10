// IQ1_M 块的回归（反量化与 IQ1_M × Q8_K 定点点积）。不引第三方框架。
//
// 四处与 IQ2 家族不同，测试针对性钉住：
//   1. 块里没有独立的 d 字段，fp16 尺度由 8 个 scales 字节拼出——手工块里直接构造出 d = 1.0
//      （sc0 的高 4 位 = 3、sc1 的高 4 位 = 0xC，其余 0 -> u16 = 0x3C00）；
//   2. 格点字节是有符号的：格点 0 是八连 0xFF，即八个 -1；
//   3. 每个码字除格点项外还有一个 ±0.125 的 delta（符号位在 qh 的位 3 / 位 7）；
//   4. 每 32 个值一组，组内用两个 3 位子尺度（1 + 2s），组与组之间按 scales 的四个 u16 轮转。
//
// 五路证据：手算四块（基例 / delta 符号 / 子尺度 / 格点高位参与索引）、真字节结构不变量、
// 真字节回归锁（前 16 值与第 7 块切片 + sum/sumabs 对上 gguf-py）、点积手算（-224）与浮点路径交叉核、
// 逐行 GEMV 与单行点积一致。
#include "kernels/iq1m.hpp"

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

// 由 measure/iq1m_oracle.py 生成：kIq1mRealHex / kIq1mRealFirst16 / kIq1mRealBlock7_32 /
// kIq1mRealSum / kIq1mRealSumAbs。
#include "iq1m_oracle_data.inc"

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

// 一块：qs/qh 全 0，scales 由给定的 4 个 u16 小端写入。
std::vector<std::uint8_t> flat_iq1m(std::uint16_t sc0, std::uint16_t sc1, std::uint16_t sc2,
                                    std::uint16_t sc3) {
    std::vector<std::uint8_t> b(kIq1mBlockBytes, 0);
    const std::uint16_t sc[4] = {sc0, sc1, sc2, sc3};
    for (int i = 0; i < 4; ++i) {
        b[48 + 2 * i] = static_cast<std::uint8_t>(sc[i] & 0xFF);
        b[48 + 2 * i + 1] = static_cast<std::uint8_t>(sc[i] >> 8);
    }
    return b;
}

void test_handmade() {
    // 基例：sc3 = 0x3000、sc2 = 0xC000，其余 0 -> 拼出的 u16 = 0x3C00 -> d = 1.0；
    // 四个 u16 的低 12 位全 0 -> 所有子尺度都是 1 -> dl = 1.0；
    // qs/qh 全 0 -> 格点 0 = 八个 -1、delta = +0.125 -> 每个值 = -1 + 0.125 = -0.875。
    const float base = -0.875f;
    {
        const auto b = flat_iq1m(0, 0, 0xC000, 0x3000);
        float out[kIq1mValuesPerBlock];
        dequant_iq1m_block(b.data(), out);
        for (int k = 0; k < kIq1mValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(base)) {
                std::printf("FAIL handmade base at %d: got %g\n", k, static_cast<double>(out[k]));
                CHECK(false);
            }
        }
    }
    // delta 符号：qh 的第 0 个字节置 0x08 -> 第 0 组的第 0 个码字（元素 0..7）取 -0.125，
    // 其余仍是 +0.125。注意位 3 不参与格点索引（索引只用位 0..2 与 4..6），故格点仍是 0。
    {
        auto b = flat_iq1m(0, 0, 0xC000, 0x3000);
        b[32] = 0x08;  // qh[0]
        float out[kIq1mValuesPerBlock];
        dequant_iq1m_block(b.data(), out);
        for (int k = 0; k < 8; ++k) {
            const float want = -1.125f;
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(want)) {
                std::printf("FAIL handmade delta[%d]: got %g\n", k, static_cast<double>(out[k]));
                CHECK(false);
            }
        }
        for (int k = 8; k < kIq1mValuesPerBlock; ++k) {
            CHECK(std::bit_cast<std::uint32_t>(out[k]) == std::bit_cast<std::uint32_t>(base));
        }
    }
    // 子尺度：sc0 的最低位（bit 0）给第 0 组的第一个子尺度，ls = 2·1 + 1 = 3 -> dl1 = 3.0。
    // 高 4 位不变（仍 3）故 d 仍是 1.0。第 1 组的第一个子尺度在 bit 6..8，为 0 -> 仍是 1。
    {
        const auto b = flat_iq1m(0x0001, 0, 0xC000, 0x3000);
        float out[kIq1mValuesPerBlock];
        dequant_iq1m_block(b.data(), out);
        for (int k = 0; k < 16; ++k) {
            const float want = 3.0f * base;
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(want)) {
                std::printf("FAIL handmade subscale[%d]: got %g\n", k, static_cast<double>(out[k]));
                CHECK(false);
            }
        }
        for (int k = 16; k < 32; ++k) {
            CHECK(std::bit_cast<std::uint32_t>(out[k]) == std::bit_cast<std::uint32_t>(base));
        }
        for (int k = 32; k < kIq1mValuesPerBlock; ++k) {
            CHECK(std::bit_cast<std::uint32_t>(out[k]) == std::bit_cast<std::uint32_t>(base));
        }
    }
    // 格点高位参与索引：qh 的第 0 个字节置 0x01 -> 第 0 个码字的索引变成 0x100（不再是 0）。
    // 若实现漏掉这 3 位，输出不会变——所以这条只断言「变了」。
    {
        const auto b0 = flat_iq1m(0, 0, 0xC000, 0x3000);
        auto b1 = b0;
        b1[32] = 0x01;
        float o0[kIq1mValuesPerBlock];
        float o1[kIq1mValuesPerBlock];
        dequant_iq1m_block(b0.data(), o0);
        dequant_iq1m_block(b1.data(), o1);
        bool differs = false;
        for (int k = 0; k < kIq1mValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(o0[k]) != std::bit_cast<std::uint32_t>(o1[k])) differs = true;
        }
        CHECK(differs);
    }
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq1mRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq1mBlockBytes));

    float out[kIq1mValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq1mBlockBytes;
        dequant_iq1m_block(blk, out);
        // 尺度同实现：拼 fp16 + 每组的两个 3 位子尺度。
        const std::uint16_t sc0 = static_cast<std::uint16_t>(blk[48] | (blk[49] << 8));
        const std::uint16_t sc1 = static_cast<std::uint16_t>(blk[50] | (blk[51] << 8));
        const std::uint16_t sc2 = static_cast<std::uint16_t>(blk[52] | (blk[53] << 8));
        const std::uint16_t sc3 = static_cast<std::uint16_t>(blk[54] | (blk[55] << 8));
        const std::uint16_t u16 = static_cast<std::uint16_t>((sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) |
                                                             ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u));
        const float d = f16_bits_to_f32(u16);
        const std::uint16_t scv[4] = {sc0, sc1, sc2, sc3};
        for (int ib = 0; ib < 8; ++ib) {
            const int ls1 = 2 * ((scv[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1;
            const int ls2 = 2 * ((scv[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1;
            for (int l = 0; l < 4; ++l) {
                const float dl = d * static_cast<float>(l < 2 ? ls1 : ls2);
                for (int j = 0; j < 8; ++j) {
                    const float v = out[ib * 32 + l * 8 + j];
                    // |值| 必须是 |dl| × {0.125, 0.875, 1.125} 之一（0.125 是 delta）。
                    bool ok = false;
                    for (float g : {0.125f, 0.875f, 1.125f}) {
                        ok = ok || std::fabs(v) == std::fabs(dl * g);
                    }
                    if (!ok) {
                        std::printf("FAIL invariant block %d ib %d l %d j %d: v=%.9g dl=%.9g\n", b, ib, l,
                                    j, static_cast<double>(v), static_cast<double>(dl));
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
                if (got != kIq1mRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq1mRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq1m_block(w.data() + 7 * kIq1mBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[32 + j]);
        if (got != kIq1mRealBlock7_32[j]) {
            std::printf("FAIL block7_32 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq1mRealBlock7_32[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq1mRealSumAbs) > 0 ? std::fabs(kIq1mRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq1mRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq1mRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq1mRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq1mRealSumAbs);
        CHECK(false);
    }
}

void test_handmade_dot() {
    // 权重：d = 1.0、子尺度全 1、格点全 -1、delta = +0.125 -> 每个权重 -0.875。
    // 激活：Q8_K d = 1.0、qs 全 1 -> 每个 1.0。点积 = 256 × (-0.875) = -224。
    const auto w = flat_iq1m(0, 0, 0xC000, 0x3000);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    const float got = iq1m_dot_q8k(w.data(), &y, 1);
    if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(-224.0f)) {
        std::printf("FAIL handmade dot: got %g want -224\n", static_cast<double>(got));
        CHECK(false);
    }
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq1mRealHex);
    std::vector<Q8kBlock> y(kRealBlocks);
    for (int b = 0; b < kRealBlocks; ++b) {
        y[b].d = 0.5f;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            y[b].qs[j] = static_cast<std::int8_t>(((b * 13 + j * 5) % 251) - 125);
        }
        for (int g = 0; g < kQ8kBlockElems / 16; ++g) {
            int s = 0;
            for (int i = 0; i < 16; ++i) s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = static_cast<std::int16_t>(s);
        }
    }

    const float fused = iq1m_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq1mValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq1m_block(w.data() + b * kIq1mBlockBytes, wv);
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
    iq1m_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq1m_dot_q8k(w.data() + r * 8 * kIq1mBlockBytes, y.data(), 8);
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
    std::puts("iq1m: block dequant and IQ1_MxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}