// IQ4_XS 块的回归（反量化与 IQ4_XS × Q8_K 定点点积）。不引第三方框架。
//
// 与 IQ4_NL（同一个码本、同一种半字节序）不同的两处，测试针对性钉住：
//   1. 尺度是 6 位且带 -32 偏移：低 4 位在 scales_l 的半个字节（一字节管两个组），高 2 位在
//      scales_h 的 2 位处（u16 里每 2 位一组）——两种排布混用会把组间尺度对调；
//   2. 每 32 个值取 16 个 qs 字节，前 16 值用低半字节、后 16 值用高半字节。
//
// 四路证据：手算四块（基例 / 低半字节尺度 / scales_h 的高 2 位 / 半字节配对）、真字节结构不变量
// （|v| / |d(ls-32)| 必须是码本的 16 个绝对值之一）、真字节回归锁（前 16 值与第 5 块中段 16 值 +
// sum/sumabs 对上 gguf-py）、点积浮点交叉核与逐行 GEMV 一致。
#include "kernels/iq4xs.hpp"

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

// 由 measure/iq4xs_oracle.py 生成：kIq4xsRealHex / kIq4xsRealFirst16 / kIq4xsRealBlock5_64 /
// kIq4xsRealSum / kIq4xsRealSumAbs。
#include "iq4xs_oracle_data.inc"

constexpr int kRealBlocks = 16;
constexpr int kScalesHAt = 2;
constexpr int kScalesLAt = 4;
constexpr int kQsAt = 8;

// 码本（与 iq4nl_tables.hpp 一致，测试里独立写一份，避免把被测实现当成期望值来源）。
constexpr int kCode[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

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

std::vector<std::uint8_t> flat_iq4xs(std::uint16_t scales_h, std::uint8_t scales_l0) {
    std::vector<std::uint8_t> b(kIq4xsBlockBytes, 0);
    b[0] = 0x00;
    b[1] = 0x3C;  // fp16 1.0
    b[kScalesHAt] = static_cast<std::uint8_t>(scales_h & 0xFF);
    b[kScalesHAt + 1] = static_cast<std::uint8_t>(scales_h >> 8);
    b[kScalesLAt] = scales_l0;
    return b;
}

void expect(float got, float want, const char* what) {
    if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(want)) {
        std::printf("FAIL %s: got %.9g want %.9g\n", what, static_cast<double>(got),
                    static_cast<double>(want));
        CHECK(false);
    }
}

void test_handmade() {
    float out[kIq4xsValuesPerBlock];
    // 基例：d = 1、两个尺度字段全 0 -> ls = 0、dl = -32；qs 全 0 -> 码本项均为 -127。
    {
        const auto b = flat_iq4xs(0, 0);
        dequant_iq4xs_block(b.data(), out);
        for (int k = 0; k < kIq4xsValuesPerBlock; ++k) expect(out[k], -32.0f * -127.0f, "base");
    }
    // 低 4 位尺度：scales_l[0] = 0x21 -> 组 0 用低半字节 1、组 1 用高半字节 2 -> dl 分别为 -31 / -30。
    {
        const auto b = flat_iq4xs(0, 0x21);
        dequant_iq4xs_block(b.data(), out);
        for (int k = 0; k < 32; ++k) expect(out[k], -31.0f * -127.0f, "scales_l group0");
        for (int k = 32; k < 64; ++k) expect(out[k], -30.0f * -127.0f, "scales_l group1");
        for (int k = 64; k < kIq4xsValuesPerBlock; ++k) expect(out[k], -32.0f * -127.0f, "scales_l rest");
    }
    // scales_h 的高 2 位：位 0..1 给组 0、位 2..3 给组 1。取 0x000D -> 组 0 得 1（ls = 16、dl = -16）、
// 组 1 得 3（ls = 48、dl = +16），其余组为 0（ls = 0、dl = -32）。符号相反这一点能测出位取错。
    {
        const auto b = flat_iq4xs(0x000D, 0);
        dequant_iq4xs_block(b.data(), out);
        for (int k = 0; k < 32; ++k) expect(out[k], -16.0f * -127.0f, "scales_h group0");
        for (int k = 32; k < 64; ++k) expect(out[k], 16.0f * -127.0f, "scales_h group1");
        for (int k = 64; k < kIq4xsValuesPerBlock; ++k) expect(out[k], -32.0f * -127.0f, "scales_h rest");
    }
    // 半字节配对：qs[0] = 0x21 -> 组 0 的第 0 个值取低半字节 1（码本 13）、第 16 个值取高半字节 2
    // （码本 25）。若把两个半字节对调，这两处的期望值就会互换成 25 / 13。
    {
        auto b = flat_iq4xs(0, 0);
        b[kQsAt + 0] = 0x21;
        dequant_iq4xs_block(b.data(), out);
        expect(out[0], -32.0f * static_cast<float>(kCode[1]), "nibble low");
        expect(out[16], -32.0f * static_cast<float>(kCode[2]), "nibble high");
        expect(out[1], -32.0f * static_cast<float>(kCode[0]), "nibble j1");
        expect(out[32], -32.0f * -127.0f, "nibble next group");
    }
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq4xsRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq4xsBlockBytes));

    float out[kIq4xsValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq4xsBlockBytes;
        dequant_iq4xs_block(blk, out);
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        const std::uint16_t sh = static_cast<std::uint16_t>(blk[kScalesHAt] |
                                                           (blk[kScalesHAt + 1] << 8));
        const std::uint8_t* sl = blk + kScalesLAt;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            for (int k = 0; k < 32; ++k) {
                const float v = out[32 * ib + k];
                const float r = (dl == 0.0f) ? 0.0f : std::fabs(v) / std::fabs(dl);
                bool ok = false;
                for (int c : kCode) {
                    ok = ok || std::fabs(r - static_cast<float>(c < 0 ? -c : c)) < 1e-3f;
                }
                if (!ok) {
                    std::printf("FAIL invariant b=%d ib=%d k=%d: v=%.9g dl=%.9g r=%.9g\n", b, ib, k,
                                static_cast<double>(v), static_cast<double>(dl),
                                static_cast<double>(r));
                    CHECK(false);
                }
                sum += static_cast<double>(v);
                sumabs += std::fabs(static_cast<double>(v));
            }
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kIq4xsRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq4xsRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq4xs_block(w.data() + 5 * kIq4xsBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[64 + j]);
        if (got != kIq4xsRealBlock5_64[j]) {
            std::printf("FAIL block5_64 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq4xsRealBlock5_64[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq4xsRealSumAbs) > 0 ? std::fabs(kIq4xsRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq4xsRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq4xsRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq4xsRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq4xsRealSumAbs);
        CHECK(false);
    }
}

void test_handmade_dot() {
    // 权重：d = 1、ls = 0 -> dl = -32；码本项全 -127 -> 每个权重 4064。激活：Q8_K d = 1、qs 全 1
    // -> 每个 1。点积 = 256 × 4064 = 1040384。
    const auto w = flat_iq4xs(0, 0);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    expect(iq4xs_dot_q8k(w.data(), &y, 1), 1040384.0f, "handmade dot");
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq4xsRealHex);
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

    const float fused = iq4xs_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq4xsValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq4xs_block(w.data() + b * kIq4xsBlockBytes, wv);
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
    iq4xs_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq4xs_dot_q8k(w.data() + r * 8 * kIq4xsBlockBytes, y.data(), 8);
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
    std::puts("iq4xs: block dequant and IQ4_XSxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}