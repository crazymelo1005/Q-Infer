// IQ3_XXS 块的回归（反量化与 IQ3_XXS × Q8_K 定点点积）。不引第三方框架。
//
// 与同族的 IQ2_XXS、以及与 IQ3_S 不同的三处，测试针对性钉住：
//   1. 每个 32 值组配一个 u32（块内偏移 66 + 4·ib32）：高 4 位是子尺度增量 n、低 28 位分成 4 个
//      7 位符号索引（查 128 项的 ksigns 表得 8 位模式）——不是 IQ3_S 那种「两个组共用一个字节」；
//   2. 格点索引就是 qs 的一个字节（8 位、无高位拼接），表是 256 项；
//   3. 幅度字节是 4 的倍数那一组（4/12/…/62），子尺度是 d(0.5+n)/2，故点积末尾乘 0.25。
//
// 四路证据：手算五块（基例 / 子尺度 / 符号查表 / 格点索引 / 点积）、真字节结构不变量（|v|/db 必须是
// 那 8 个幅度值之一）、真字节回归锁（前 16 值与第 5 块中段 16 值 + sum/sumabs 对上 gguf-py）、
// 点积浮点交叉核与逐行 GEMV 一致。
#include "kernels/iq3xxs.hpp"

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

// 由 measure/iq3xxs_oracle.py 生成：kIq3xxsRealHex / kIq3xxsRealFirst16 / kIq3xxsRealBlock5_64 /
// kIq3xxsRealSum / kIq3xxsRealSumAbs。
#include "iq3xxs_oracle_data.inc"

constexpr int kRealBlocks = 16;
constexpr int kGasAt = 66;

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

// d = 1.0、qs 全 0、gas 全 0 的块；第 ib32 组的 u32 由 aux 指定（用来单独调子尺度或符号）。
std::vector<std::uint8_t> flat_iq3xxs(std::uint32_t aux0) {
    std::vector<std::uint8_t> b(kIq3xxsBlockBytes, 0);
    b[0] = 0x00;
    b[1] = 0x3C;  // fp16 1.0
    b[kGasAt + 0] = static_cast<std::uint8_t>(aux0 & 0xFF);
    b[kGasAt + 1] = static_cast<std::uint8_t>((aux0 >> 8) & 0xFF);
    b[kGasAt + 2] = static_cast<std::uint8_t>((aux0 >> 16) & 0xFF);
    b[kGasAt + 3] = static_cast<std::uint8_t>((aux0 >> 24) & 0xFF);
    return b;
}

void expect(float got, float want, const char* what) {
    if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(want)) {
        std::printf("FAIL %s: got %.9g want %.9g\n", what, static_cast<double>(got),
                    static_cast<double>(want));
        CHECK(false);
    }
}

// 内建的幅度值（与上游 C 表一致，见 measure/iq3xxs_oracle.py 的 BUILTIN_MAG）。
constexpr int kMag[8] = {4, 12, 20, 28, 36, 44, 52, 62};

void test_handmade() {
    float out[kIq3xxsValuesPerBlock];
    // 基例：aux32 = 0 -> n = 0、db = 1×(0.5+0)×0.5 = 0.25、符号索引 0 -> 全正；
    // 格点 0 = 0x04040404 -> 每个值 0.25×4 = 1.0。
    {
        const auto b = flat_iq3xxs(0);
        dequant_iq3xxs_block(b.data(), out);
        for (int k = 0; k < kIq3xxsValuesPerBlock; ++k) expect(out[k], 1.0f, "base");
    }
    // 子尺度：高 4 位 = 1 -> n = 1、db = 0.75 -> 每个值 3.0（只作用于第 0 组）。
    {
        const auto b = flat_iq3xxs(0x10000000u);
        dequant_iq3xxs_block(b.data(), out);
        for (int k = 0; k < 32; ++k) expect(out[k], 3.0f, "subscale group0");
        for (int k = 32; k < kIq3xxsValuesPerBlock; ++k) expect(out[k], 1.0f, "subscale rest");
    }
    // 符号查表：低 28 位全是 1 -> 四个 7 位索引都是 127、查表得 0xFF -> 第 0 组的 32 个值全负。
    // 注意一组有四个符号字节（每 l 一个、各管 8 个值），不是一个字节管整组——只写低 7 位只有前 8 个负。
    {
        const auto b = flat_iq3xxs(0x0FFFFFFFu);
        dequant_iq3xxs_block(b.data(), out);
        for (int k = 0; k < 32; ++k) expect(out[k], -1.0f, "signs group0");
        for (int k = 32; k < kIq3xxsValuesPerBlock; ++k) expect(out[k], 1.0f, "signs rest");
    }
    // 符号索引按 l 分段：第 l 个符号字节用 (aux32 >> 7l) & 127。只写第二位 7 位 -> 组内偏移 8..15
    // 取负、其余正。这条抓「符号索引错位」（例如漏掉 7 位步长）。
    {
        const auto b = flat_iq3xxs(127u << 7);
        dequant_iq3xxs_block(b.data(), out);
        for (int k = 0; k < 8; ++k) expect(out[k], 1.0f, "signs l1 up");
        for (int k = 8; k < 16; ++k) expect(out[k], -1.0f, "signs l1");
        for (int k = 16; k < 32; ++k) expect(out[k], 1.0f, "signs l1 rest");
    }
    // 格点索引：qs[0] = 1 -> 格点 1 = 0x04040414，字节 20/4/4/4 -> 第 0 个值 5.0，其余三个 1.0。
    {
        auto b = flat_iq3xxs(0);
        b[2 + 0] = 1;
        dequant_iq3xxs_block(b.data(), out);
        expect(out[0], 0.25f * 20.0f, "grid idx0");
        expect(out[1], 1.0f, "grid idx1");
        expect(out[2], 1.0f, "grid idx2");
        expect(out[3], 1.0f, "grid idx3");
        expect(out[4], 1.0f, "grid idx4");
    }
}

void test_handmade_dot() {
    // 权重：d = 1、n = 0 -> 每个权重 0.25×4 = 1.0；激活：Q8_K d = 1、qs 全 1 -> 每个 1.0。
    // 真值 = 256；点积内部先得 1024（每 32 值组 32 项 × 幅度 4）再乘 0.25。
    const auto w = flat_iq3xxs(0);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    expect(iq3xxs_dot_q8k(w.data(), &y, 1), 256.0f, "handmade dot");
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq3xxsRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq3xxsBlockBytes));

    float out[kIq3xxsValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq3xxsBlockBytes;
        dequant_iq3xxs_block(blk, out);
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* p = blk + kGasAt + 4 * ib32;
            const std::uint32_t aux32 = static_cast<std::uint32_t>(p[0] | (p[1] << 8) |
                                                                  (p[2] << 16) |
                                                                  (static_cast<std::uint32_t>(p[3]) << 24));
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            for (int k = 0; k < 32; ++k) {
                const float v = out[32 * ib32 + k];
                const float r = (db == 0.0f) ? 0.0f : std::fabs(v) / std::fabs(db);
                bool ok = false;
                for (int m : kMag) ok = ok || std::fabs(r - static_cast<float>(m)) < 1e-3f;
                if (!ok) {
                    std::printf("FAIL invariant b=%d ib32=%d k=%d: v=%.9g db=%.9g r=%.9g\n", b, ib32,
                                k, static_cast<double>(v), static_cast<double>(db),
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
                if (got != kIq3xxsRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq3xxsRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq3xxs_block(w.data() + 5 * kIq3xxsBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[64 + j]);
        if (got != kIq3xxsRealBlock5_64[j]) {
            std::printf("FAIL block5_64 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq3xxsRealBlock5_64[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq3xxsRealSumAbs) > 0 ? std::fabs(kIq3xxsRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq3xxsRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq3xxsRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq3xxsRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq3xxsRealSumAbs);
        CHECK(false);
    }
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq3xxsRealHex);
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

    const float fused = iq3xxs_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq3xxsValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq3xxs_block(w.data() + b * kIq3xxsBlockBytes, wv);
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
    iq3xxs_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq3xxs_dot_q8k(w.data() + r * 8 * kIq3xxsBlockBytes, y.data(), 8);
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
    std::puts("iq3xxs: block dequant and IQ3_XXSxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}