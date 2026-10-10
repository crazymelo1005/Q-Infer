// IQ4_NL 单行反量化的回归。不引第三方框架。
//
// 两路证据：
//   1. 手工构造的一行（尺度 1.0 + 半字节 0xjj），期望值是手算的——它钉死码本、分裂式半字节序与尺度三件事。
//   2. 模型本体的一行（`per_layer_token_embd.weight` 第 0 行）的字节与反量化结果，由
//      `measure/ple_row_oracle.py` 从 GGUF 取出；该脚本的输出与本仓库记录过的引擎自带 oracle 向量
//      逐位吻合（前 16 值与 L1 和都一致），故这份夹具可信。
#include "kernels/iq4nl.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

// 由 measure/iq4nl_oracle.py 从真实 GGUF 生成：kIq4nlRealHex / kIq4nlRealFirst16 /
// kIq4nlRealBlock7_16 / kIq4nlRealSum / kIq4nlRealSumAbs。真实张量是另一档模型里用 IQ4_NL 作
// 权重的那 18 层 down 之一（blk.0.ffn_down_exps.weight）。
#include "iq4nl_oracle_data.inc"

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

void near(float got, float want, const char* what, int index) {
    const float tol = 1e-6f;
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %s[%d]: got %.9g want %.9g\n", what, index, got, want);
        CHECK(false);
    }
}

constexpr float kCodebook[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};

void test_handmade_row_pins_codebook_and_nibble_order() {
    // 块 0：尺度 1.0（fp16 0x3C00，小端 00 3C），qs[j] = (j<<4)|j → 低半字节 j、高半字节 j。
    // 块 1..4：尺度 0（00 00）、qs 全 0xFF → 值全 0。
    std::vector<std::uint8_t> row;
    row.push_back(0x00); row.push_back(0x3C);
    for (int j = 0; j < 16; ++j) row.push_back(static_cast<std::uint8_t>((j << 4) | j));
    for (int b = 1; b < 5; ++b) {
        row.push_back(0x00); row.push_back(0x00);
        for (int j = 0; j < 16; ++j) row.push_back(0xFF);
    }
    CHECK(row.size() == 90);

    float out[160];
    dequant_iq4nl_row(row.data(), out);

    for (int j = 0; j < 16; ++j) {
        near(out[j], kCodebook[j], "block0 low", j);         // y[j]   = d·cb[qs[j]&0xf]
        near(out[16 + j], kCodebook[j], "block0 high", j);    // y[j+16]= d·cb[qs[j]>>4]
    }
    for (int k = 32; k < 160; ++k) near(out[k], 0.f, "zero-scale blocks", k);
}

void test_model_row_zero_matches_oracle() {
    const std::vector<std::uint8_t> row = from_hex(
        "518c6b846d849ab66ae63757410dbaf31207730c43866d875eb60666bcc9cb0d48961e56510cfa8ecf5409ce789cac3911cc5a5426705e8d5abf604a565ecaa4a565b9bc798b9a860d8c8e79c7834e134b61143c0368d89def7a");
    CHECK(row.size() == 90);

    float out[160];
    dequant_iq4nl_row(row.data(), out);

    const float first16[16] = {-0.0100112f, 0.01290917f, -0.01817822f, 0.01290917f, -0.00658631f,
                               0.00579596f, -0.00658631f, 0.00579596f, 0.00263453f, 0.00263453f,
                               0.02739906f, -0.01817822f, -0.00658631f, 0.01712441f, 0.02186656f,
                               0.00263453f};
    const float last16[16] = {-0.00024724f, 0.0024724f, -0.01310372f, -0.00024724f, 0.01211476f,
                              0.02571297f, 0.01211476f, 0.00543928f, 0.02571297f, 0.0160706f,
                              0.03139949f, 0.00543928f, -0.01705956f, -0.00321412f, -0.02200437f,
                              0.0024724f};
    for (int j = 0; j < 16; ++j) near(out[j], first16[j], "model first16", j);
    for (int j = 0; j < 16; ++j) near(out[144 + j], last16[j], "model last16", j);

    double l1 = 0.0;
    for (int k = 0; k < 160; ++k) l1 += std::fabs(static_cast<double>(out[k]));
    if (std::fabs(l1 - 2.01196) > 1e-4) {
        std::printf("FAIL L1: got %.9g want 2.01196\n", l1);
        CHECK(false);
    }
}

void test_fp16_halfway_values() {
    // fp16 的次正规数与正规数边界：0x0001 = 2^-24，0x0400 = 2^-14，0x3C00 = 1.0。
    near(f16_bits_to_f32(0x0001), 5.9604645e-8f, "fp16 subnormal", 0);
    near(f16_bits_to_f32(0x0400), 6.1035156e-5f, "fp16 min normal", 0);
    near(f16_bits_to_f32(0x3C00), 1.0f, "fp16 one", 0);
    near(f16_bits_to_f32(0xC000), -2.0f, "fp16 minus two", 0);
    near(f16_bits_to_f32(0x7C00), std::numeric_limits<float>::infinity(), "fp16 inf", 0);
}

void test_f32_to_f16() {
    // 全部 65536 个 fp16 位型经「f16 -> f32 -> f16」必须回到原值（正规、次正规、零都算）；
    // 非有限值按 ggml 的约定统一归一成 0x7E00（尾数信息丢失，符号保留）。
    for (std::uint32_t bits = 0; bits < 0x10000u; ++bits) {
        const std::uint16_t h = static_cast<std::uint16_t>(bits);
        const float f = f16_bits_to_f32(h);
        const std::uint16_t back = f32_to_f16_bits(f);
        const bool is_nan = ((h >> 10) & 0x1Fu) == 0x1Fu && (h & 0x3FFu) != 0;
        const std::uint16_t want = is_nan ? static_cast<std::uint16_t>((h & 0x8000u) | 0x7E00u) : h;
        if (back != want) {
            std::printf("FAIL roundtrip 0x%04x -> %g -> 0x%04x (want 0x%04x)\n", h,
                        static_cast<double>(f), back, want);
            CHECK(false);
        }
    }
    // 已知值：1.0、-2.0、fp16 最大正规数、上溢到 Inf、最小次正规。
    CHECK(f32_to_f16_bits(1.0f) == 0x3C00);
    CHECK(f32_to_f16_bits(-2.0f) == 0xC000);
    CHECK(f32_to_f16_bits(65504.0f) == 0x7BFF);
    CHECK(f32_to_f16_bits(65520.0f) == 0x7C00);       // 恰好是上溢的中间点，就近舍入到偶数
    CHECK(f32_to_f16_bits(5.9604645e-8f) == 0x0001);  // 最小次正规
    CHECK(f32_to_f16_bits(0.0f) == 0x0000);
    CHECK(f32_to_f16_bits(-0.0f) == 0x8000);
}

// ---- 通用块路径（IQ4_NL 作权重、Q8_0 作激活） ----

// 一块：d 为给定 fp16，16 个半字节字节全是 qs。
std::vector<std::uint8_t> flat_iq4nl_block(std::uint16_t d_bits, std::uint8_t qs) {
    std::vector<std::uint8_t> b(kIq4nlBlockBytes, qs);
    b[0] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[1] = static_cast<std::uint8_t>(d_bits >> 8);
    return b;
}

Q80Block flat_q80(std::uint16_t d_bits, std::int8_t qs) {
    Q80Block y{};
    y.d_bits = d_bits;
    for (int j = 0; j < kQ80BlockElems; ++j) y.qs[j] = qs;
    return y;
}

void test_block_handmade_and_dot() {
    // d = 1.0、qs[j] = (j<<4)|j -> 低半字节 j 是元素 j，高半字节 j 是元素 j+16。
    const auto b = flat_iq4nl_block(0x3C00, 0);
    float out[kIq4nlValuesPerBlock];
    for (int j = 0; j < 16; ++j) {
        std::vector<std::uint8_t> bb = b;
        bb[2 + j] = static_cast<std::uint8_t>((j << 4) | j);
        dequant_iq4nl_block(bb.data(), out);
        CHECK(std::bit_cast<std::uint32_t>(out[j]) ==
              std::bit_cast<std::uint32_t>(kCodebook[j]));
        CHECK(std::bit_cast<std::uint32_t>(out[j + 16]) ==
              std::bit_cast<std::uint32_t>(kCodebook[j]));
    }

    // 点积：载入的块每半字节都是 0 -> 码本第 0 项 -127，故 32 个权重都是 -127；
    // 激活 d = 1.0、qs 全 1 -> 每个 1.0。点积 = 32 · (-127) = -4064。
    const auto w = flat_iq4nl_block(0x3C00, 0x00);
    const Q80Block y = flat_q80(0x3C00, 1);
    CHECK(std::bit_cast<std::uint32_t>(iq4nl_dot_q8_0(w.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(-4064.0f));

    // 每半字节都是 8 -> 码本第 8 项 1，故 32 个权重都是 1；点积 = 32。
    const auto w2 = flat_iq4nl_block(0x3C00, 0x88);
    CHECK(std::bit_cast<std::uint32_t>(iq4nl_dot_q8_0(w2.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(32.0f));
}

void test_block_real_bytes() {
    const std::vector<std::uint8_t> w = from_hex(kIq4nlRealHex);
    CHECK(w.size() == 16u * static_cast<std::size_t>(kIq4nlBlockBytes));

    float out[kIq4nlValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < 16; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq4nlBlockBytes;
        dequant_iq4nl_block(blk, out);
        const float d = f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        for (int j = 0; j < kIq4nlValuesPerBlock; ++j) {
            // 结构不变量：幅值必须是 |d| 乘码本 16 项之一。
            bool ok = false;
            for (int k = 0; k < 16; ++k) {
                ok = ok || std::fabs(out[j]) == std::fabs(d * kCodebook[k]);
            }
            if (!ok) {
                std::printf("FAIL invariant block %d elem %d: v=%.9g d=%.9g\n", b, j,
                            static_cast<double>(out[j]), static_cast<double>(d));
                CHECK(false);
            }
            sum += static_cast<double>(out[j]);
            sumabs += std::fabs(static_cast<double>(out[j]));
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kIq4nlRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq4nlRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_iq4nl_block(w.data() + 7 * kIq4nlBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[16 + j]);
        if (got != kIq4nlRealBlock7_16[j]) {
            std::printf("FAIL block7_16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq4nlRealBlock7_16[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq4nlRealSumAbs) > 0 ? std::fabs(kIq4nlRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq4nlRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq4nlRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq4nlRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq4nlRealSumAbs);
        CHECK(false);
    }
}

void test_block_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq4nlRealHex);
    // 激活必须逐元素变化：若同一块内所有 qs 相同，点积里「元素 j 与 j+16 的配对」就不可辨识，
    // 把第二半的激活索引写错也测不出来（这正是本用例第一版被变异测试放过的原因）。
    std::vector<Q80Block> y(16);
    for (int b = 0; b < 16; ++b) {
        y[b].d_bits = 0x3C00;
        for (int j = 0; j < kQ80BlockElems; ++j) {
            y[b].qs[j] = static_cast<std::int8_t>(((b * 7 + j * 5) % 251) - 125);
        }
    }

    const float fused = iq4nl_dot_q8_0(w.data(), y.data(), 16);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq4nlValuesPerBlock];
    float yv[kQ80BlockElems];
    for (int b = 0; b < 16; ++b) {
        dequant_iq4nl_block(w.data() + b * kIq4nlBlockBytes, wv);
        q8_0_dequant_block(y[b], yv);
        for (int j = 0; j < kIq4nlValuesPerBlock; ++j) {
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

    // 逐行 GEMV：2 行 × 8 块，所有行共用同一组激活。
    float gemv[2];
    iq4nl_gemv_q8_0(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq4nl_dot_q8_0(w.data() + r * 8 * kIq4nlBlockBytes, y.data(), 8);
        if (std::bit_cast<std::uint32_t>(gemv[r]) != std::bit_cast<std::uint32_t>(one)) {
            std::printf("FAIL gemv row %d: got 0x%08x want 0x%08x\n", r,
                        std::bit_cast<std::uint32_t>(gemv[r]), std::bit_cast<std::uint32_t>(one));
            CHECK(false);
        }
    }
}

}  // namespace

int main() {
    test_handmade_row_pins_codebook_and_nibble_order();
    test_model_row_zero_matches_oracle();
    test_fp16_halfway_values();
    test_f32_to_f16();
    test_block_handmade_and_dot();
    test_block_real_bytes();
    test_block_dot_and_row_stride();
    std::puts("iq4nl: row and block dequant plus IQ4_NLxQ8_0 dot pinned by handmade, real bytes and float cross-check");
    return 0;
}
