// IQ3_S 块的回归（反量化与 IQ3_S × Q8_K 定点点积）。不引第三方框架。
//
// 与 IQ2 家族不同的三处，测试针对性钉住：
//   1. 格点表是 uint32_t（每项 4 个幅度字节，取 1..15 的奇数）——按 uint64 取会读到相邻两项；
//   2. 格点索引 9 位：低 8 位是 qs 的 8 个连续字节，高 1 位是 qh 同一个字节的位 m（m 是组内序号），
//      即 `(qh << (8 - m)) & 256`；上游把它拆成两条位移公式，容易把 m 与 l 混掉；
//   3. 子尺度两个 32 值组共用一个字节（低半字节给前一组、高半字节给后一组），ls = 2·n + 1。
//
// 四路证据：手算五块（基例 / 子尺度 / 符号 / qh 高位落在哪 4 个输出 / qs 参与）、真字节结构不变量
// （|v| / (d(2n+1)) 必须是 8 个奇数之一）、真字节回归锁（前 16 值与第 5 块中段 16 值 + sum/sumabs
// 对上 gguf-py）、点积手算与浮点路径交叉核、逐行 GEMV 一致。
#include "kernels/iq3s.hpp"

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

// 由 measure/iq3s_oracle.py 生成：kIq3sRealHex / kIq3sRealFirst16 / kIq3sRealBlock5_32 /
// kIq3sRealSum / kIq3sRealSumAbs。
#include "iq3s_oracle_data.inc"

constexpr int kRealBlocks = 16;
constexpr int kQhAt = 66;
constexpr int kSignsAt = 74;
constexpr int kScalesAt = 106;

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

// d = 1.0、qs/qh/signs 全 0、scales 给一个值。此时格点索引全 0、符号全正、ls 由 scales 定。
std::vector<std::uint8_t> flat_iq3s(std::uint8_t scales0) {
    std::vector<std::uint8_t> b(kIq3sBlockBytes, 0);
    b[0] = 0x00;
    b[1] = 0x3C;  // fp16 1.0
    b[kScalesAt] = scales0;
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
    float out[kIq3sValuesPerBlock];
    // 基例：格点 0 的四个字节都是 1、符号全正、d = 1、ls = 1 -> 每个值 1.0。
    {
        const auto b = flat_iq3s(0);
        dequant_iq3s_block(b.data(), out);
        for (int k = 0; k < kIq3sValuesPerBlock; ++k) expect(out[k], 1.0f, "base");
    }
    // 子尺度：scales[0] 的低半字节 3 -> 第 0 个 32 值组 ls = 7；高半字节 1 -> 第 1 组 ls = 3。
    {
        const auto b = flat_iq3s(0x13);
        dequant_iq3s_block(b.data(), out);
        for (int k = 0; k < 32; ++k) expect(out[k], 7.0f, "subscale group0");
        for (int k = 32; k < 64; ++k) expect(out[k], 3.0f, "subscale group1");
        for (int k = 64; k < kIq3sValuesPerBlock; ++k) expect(out[k], 1.0f, "subscale rest");
    }
    // 符号：signs[0] = 0xFF -> 第 0 个 32 值组的前 8 个值取负；signs[1] = 0x01 只翻该组第 8 个值。
    {
        auto b = flat_iq3s(0);
        b[kSignsAt + 0] = 0xFF;
        dequant_iq3s_block(b.data(), out);
        for (int k = 0; k < 8; ++k) expect(out[k], -1.0f, "sign l0");
        for (int k = 8; k < kIq3sValuesPerBlock; ++k) expect(out[k], 1.0f, "sign rest");

        auto b2 = flat_iq3s(0);
        b2[kSignsAt + 1] = 0x01;
        dequant_iq3s_block(b2.data(), out);
        expect(out[8], -1.0f, "sign bit0 of l1");
        for (int k = 0; k < 8; ++k) expect(out[k], 1.0f, "sign l1 up");
        for (int k = 9; k < 16; ++k) expect(out[k], 1.0f, "sign l1 rest");
    }
    // qh 的高 1 位落在哪 4 个输出：qh[0] 的位 1 给组内第 1 个索引（l=0 的第二条），即输出 4..7；
    // 位 4 给组内第 4 个索引（l=2 的第一条），即输出 16..19。其余输出不得动。
    {
        const auto base = flat_iq3s(0);
        dequant_iq3s_block(base.data(), out);
        const std::vector<float> b0(out, out + kIq3sValuesPerBlock);

        auto hi1 = base;
        hi1[kQhAt + 0] = 0x02;  // 位 1 -> 组内第 1 个索引
        dequant_iq3s_block(hi1.data(), out);
        for (int k = 0; k < 4; ++k) expect(out[k], b0[k], "qh bit1 keeps 0..3");
        bool ch1 = false;
        for (int k = 4; k < 8; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(b0[k])) ch1 = true;
        }
        CHECK(ch1);
        for (int k = 8; k < kIq3sValuesPerBlock; ++k) expect(out[k], b0[k], "qh bit1 keeps 8..");

        auto hi4 = base;
        hi4[kQhAt + 0] = 0x10;  // 位 4 -> 组内第 4 个索引
        dequant_iq3s_block(hi4.data(), out);
        for (int k = 0; k < 16; ++k) expect(out[k], b0[k], "qh bit4 keeps 0..15");
        bool ch4 = false;
        for (int k = 16; k < 20; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(b0[k])) ch4 = true;
        }
        CHECK(ch4);
        for (int k = 20; k < kIq3sValuesPerBlock; ++k) expect(out[k], b0[k], "qh bit4 keeps 20..");
    }
    // qs 参与：qs[0] = 7 -> 组内第 0 个索引从 0 变 7，只有输出 0..3 会变。
    {
        const auto base = flat_iq3s(0);
        dequant_iq3s_block(base.data(), out);
        const std::vector<float> b0(out, out + kIq3sValuesPerBlock);
        auto q = base;
        q[2 + 0] = 0x07;
        dequant_iq3s_block(q.data(), out);
        bool changed = false;
        for (int k = 0; k < 4; ++k) {
            if (std::bit_cast<std::uint32_t>(out[k]) != std::bit_cast<std::uint32_t>(b0[k])) changed = true;
        }
        CHECK(changed);
        for (int k = 4; k < kIq3sValuesPerBlock; ++k) expect(out[k], b0[k], "qs keeps 4..");
    }
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq3sRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq3sBlockBytes));

    float out[kIq3sValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq3sBlockBytes;
        dequant_iq3s_block(blk, out);
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        const std::uint8_t* scales = blk + kScalesAt;
        for (int g = 0; g < 8; ++g) {
            const int ls = 2 * static_cast<int>((g % 2 == 0) ? (scales[g / 2] & 0x0F)
                                                             : (scales[g / 2] >> 4)) + 1;
            const float db = d * static_cast<float>(ls);
            for (int k = 0; k < 32; ++k) {
                const float v = out[g * 32 + k];
                const float r = (db == 0.0f) ? 0.0f : std::fabs(v) / std::fabs(db);
                bool ok = false;
                for (int odd : {1, 3, 5, 7, 9, 11, 13, 15}) {
                    ok = ok || std::fabs(r - static_cast<float>(odd)) < 1e-3f;
                }
                if (!ok) {
                    std::printf("FAIL invariant b=%d g=%d k=%d: v=%.9g db=%.9g r=%.9g\n", b, g, k,
                                static_cast<double>(v), static_cast<double>(db),
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
                if (got != kIq3sRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq3sRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq3s_block(w.data() + 5 * kIq3sBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[32 + j]);
        if (got != kIq3sRealBlock5_32[j]) {
            std::printf("FAIL block5_32 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq3sRealBlock5_32[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq3sRealSumAbs) > 0 ? std::fabs(kIq3sRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq3sRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq3sRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq3sRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq3sRealSumAbs);
        CHECK(false);
    }
}

void test_handmade_dot() {
    // 权重：d = 1、ls = 1、格点全 1、符号全正 -> 每个权重 1.0。
    // 激活：Q8_K d = 1.0、qs 全 1 -> 每个 1.0。点积 = 256 × 1.0 = 256。
    const auto w = flat_iq3s(0);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    expect(iq3s_dot_q8k(w.data(), &y, 1), 256.0f, "handmade dot");
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq3sRealHex);
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

    const float fused = iq3s_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq3sValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq3s_block(w.data() + b * kIq3sBlockBytes, wv);
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
    iq3s_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq3s_dot_q8k(w.data() + r * 8 * kIq3sBlockBytes, y.data(), 8);
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
    std::puts("iq3s: block dequant and IQ3_SxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}