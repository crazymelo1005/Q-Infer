// Q8_0 的权重侧回归（从文件字节反量化、Q8_0 × Q8_0 定点点积、逐行 GEMV）。
// 激活侧的量化器与反量化已在 test_q2_0 里随 Q2_0 那条链测过，这里不再重复。
//
// 三路证据：手算四块（基例 / fp16 尺度 / 负码字 / 点积）、真字节结构不变量（v/d 必须是
// [-127, 127] 内的整数）、真字节回归锁（前 16 值与第 5 块中段 16 值 + sum/sumabs 对上 gguf-py）。
// 另有一条等价性：原始字节路径（dequant_q8_0_raw）必须与结构体路径逐位一致。
#include "kernels/q8_0.hpp"

#include "kernels/fp16.hpp"
#include "check.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

// 由 measure/q8_0_oracle.py 生成：kQ80RealHex / kQ80RealFirst16 / kQ80RealBlock5_16 /
// kQ80RealSum / kQ80RealSumAbs。
#include "q8_0_oracle_data.inc"

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

std::vector<std::uint8_t> flat_q80(std::uint16_t d_bits, std::int8_t q) {
    std::vector<std::uint8_t> b(kQ80BlockBytes, 0);
    b[0] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[1] = static_cast<std::uint8_t>(d_bits >> 8);
    for (int j = 0; j < kQ80BlockElems; ++j) b[2 + j] = static_cast<std::uint8_t>(q);
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
    float out[kQ80BlockElems];
    // 基例：d = 1、码字全 3 -> 每个值 3.0。
    {
        const auto b = flat_q80(0x3C00, 3);
        dequant_q8_0_raw(b.data(), out);
        for (int j = 0; j < kQ80BlockElems; ++j) expect(out[j], 3.0f, "base");
    }
    // fp16 尺度：d = 0.5、码字全 3 -> 1.5。
    {
        const auto b = flat_q80(0x3800, 3);
        dequant_q8_0_raw(b.data(), out);
        for (int j = 0; j < kQ80BlockElems; ++j) expect(out[j], 1.5f, "d half");
    }
    // 负码字：d = 1、码字全 -127 -> 每个值 -127.0；再单个码字改到 127 看只有它变。
    {
        const auto b = flat_q80(0x3C00, -127);
        dequant_q8_0_raw(b.data(), out);
        for (int j = 0; j < kQ80BlockElems; ++j) expect(out[j], -127.0f, "negative");
        auto b2 = flat_q80(0x3C00, -127);
        b2[2 + 5] = 127;
        dequant_q8_0_raw(b2.data(), out);
        expect(out[5], 127.0f, "one positive");
        expect(out[4], -127.0f, "one positive neighbour");
    }
}

void test_handmade_dot() {
    // 权重：d = 0.5、码字全 1 -> 每块权重 0.5；激活：d = 1、码字全 1 -> 每块 1。
    // 点积 = 32 × 0.5 × 1 = 16。
    const auto w = flat_q80(0x3800, 1);
    Q80Block y{};
    y.d_bits = 0x3C00;
    for (int j = 0; j < kQ80BlockElems; ++j) y.qs[j] = 1;
    expect(q8_0_dot_q8_0(w.data(), &y, 1), 16.0f, "handmade dot");
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kQ80RealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kQ80BlockBytes));

    float out[kQ80BlockElems];
    float via_struct[kQ80BlockElems];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kQ80BlockBytes;
        dequant_q8_0_raw(blk, out);
        // 原始字节路径必须与结构体路径逐位一致（Q80Block 的布局与文件一致，但这条不假设对齐）。
        Q80Block s{};
        std::memcpy(&s, blk, kQ80BlockBytes);
        q8_0_dequant_block(s, via_struct);
        for (int j = 0; j < kQ80BlockElems; ++j) {
            if (std::bit_cast<std::uint32_t>(out[j]) != std::bit_cast<std::uint32_t>(via_struct[j])) {
                std::printf("FAIL raw vs struct b=%d j=%d: %.9g vs %.9g\n", b, j,
                            static_cast<double>(out[j]), static_cast<double>(via_struct[j]));
                CHECK(false);
            }
        }
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        for (int j = 0; j < kQ80BlockElems; ++j) {
            const float v = out[j];
            const float q = (d == 0.0f) ? 0.0f : v / d;
            const bool is_int = std::fabs(q - std::round(q)) <= 1e-4f;
            if (!is_int || q < -127.0f - 1e-3f || q > 127.0f + 1e-3f) {
                std::printf("FAIL invariant b=%d j=%d: v=%.9g d=%.9g q=%.9g\n", b, j,
                            static_cast<double>(v), static_cast<double>(d), static_cast<double>(q));
                CHECK(false);
            }
            sum += static_cast<double>(v);
            sumabs += std::fabs(static_cast<double>(v));
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kQ80RealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kQ80RealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_q8_0_raw(w.data() + 5 * kQ80BlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[16 + j]);
        if (got != kQ80RealBlock5_16[j]) {
            std::printf("FAIL block5_16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kQ80RealBlock5_16[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kQ80RealSumAbs) > 0 ? std::fabs(kQ80RealSumAbs) : 1.0);
    if (std::fabs(sum - kQ80RealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kQ80RealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kQ80RealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kQ80RealSumAbs);
        CHECK(false);
    }
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kQ80RealHex);
    std::vector<Q80Block> y(kRealBlocks);
    for (int b = 0; b < kRealBlocks; ++b) {
        y[b].d_bits = 0x3800;  // 0.5
        for (int j = 0; j < kQ80BlockElems; ++j) {
            y[b].qs[j] = static_cast<std::int8_t>(((b * 13 + j * 5) % 251) - 125);
        }
    }

    const float fused = q8_0_dot_q8_0(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kQ80BlockElems];
    float yv[kQ80BlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_q8_0_raw(w.data() + b * kQ80BlockBytes, wv);
        q8_0_dequant_block(y[b], yv);
        for (int j = 0; j < kQ80BlockElems; ++j) {
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
    q8_0_gemv_q8_0(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = q8_0_dot_q8_0(w.data() + r * 8 * kQ80BlockBytes, y.data(), 8);
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
    std::puts("q8_0: weight-side dequant and Q8_0xQ8_0 dot pinned by handmade, real bytes and float cross-check");
    return 0;
}