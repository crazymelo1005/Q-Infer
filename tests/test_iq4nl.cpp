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

}  // namespace

int main() {
    test_handmade_row_pins_codebook_and_nibble_order();
    test_model_row_zero_matches_oracle();
    test_fp16_halfway_values();
    std::puts("iq4nl: row dequant matches both oracles");
    return 0;
}
