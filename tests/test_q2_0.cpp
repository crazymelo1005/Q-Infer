// Q2_0 块的回归（含其激活搭档 Q8_0 的反量化）。不引第三方框架。
//
// 四路证据：
//   1. 手工两块，期望值是手算的——钉死「码 {0,1,2,3} → {-1,0,+1,+2}」、每字节 4 个码且低 2 位在前、
//      以及尺度是 fp16。
//   2. 真字节的结构不变量：模型里 blk.0.ffn_down_exps.weight 的前 64 个块的每个输出值必须恰好是
//      {-d, 0, +d, +2d} 之一（d 是该块自己的尺度）。这条不依赖任何外部期望值。
//   3. 真字节的回归锁：同一批字节，sum / sumabs 与反量化前 16 值必须对上台架给出的值
//      （measure/q2_0_oracle.py 生成；同一窗口在引擎自身的 stratа-dequant 上得到逐位相同的
//      CHECKSUM，两套实现在真字节上一致）。
//   4. 点积与浮点路径交叉核对：Q2_0 × Q8_0 的结果必须等于「两侧各自反量化后逐元素累加」，
//      容差按 Σ|w·y| 取相对值；另核逐行 GEMV 与单行点积一致（行距 n_blocks × 18 字节）。
#include "kernels/q2_0.hpp"

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

// 由 measure/q2_0_oracle.py 生成：kQ20RealHex / kQ20RealFirst16[16] / kQ20RealSum / kQ20RealSumAbs。
#include "q2_0_oracle_data.inc"

constexpr int kRealBlocks = 64;

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

// 一块：d 为给定 fp16，16 个码字节全是 qs。
std::vector<std::uint8_t> flat_q20(std::uint16_t d_bits, std::uint8_t qs) {
    std::vector<std::uint8_t> b(kQ20BlockBytes, qs);
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

void test_handmade() {
    // 码字节 0xE4 = 0b11100100：4 个码依次是 0、1、2、3 -> 值 -1、0、+1、+2（乘以 d）。
    const auto b = flat_q20(0x3C00, 0xE4);  // d = 1.0
    float out[kQ20BlockElems];
    dequant_q2_0_block(b.data(), out);
    for (int j = 0; j < kQ20BlockElems; ++j) {
        const float want = static_cast<float>((j % 4) - 1);
        if (std::bit_cast<std::uint32_t>(out[j]) != std::bit_cast<std::uint32_t>(want)) {
            std::printf("FAIL handmade j=%d: got %g want %g\n", j, static_cast<double>(out[j]),
                        static_cast<double>(want));
            CHECK(false);
        }
    }

    // 尺度 2.0：值翻倍。
    const auto b2 = flat_q20(0x4000, 0xE4);
    float out2[kQ20BlockElems];
    dequant_q2_0_block(b2.data(), out2);
    for (int j = 0; j < kQ20BlockElems; ++j) {
        const float want = 2.0f * static_cast<float>((j % 4) - 1);
        CHECK(std::bit_cast<std::uint32_t>(out2[j]) == std::bit_cast<std::uint32_t>(want));
    }

    // Q8_0 的反量化：d 由 fp16 展开，值 = d · qs。
    const Q80Block y = flat_q80(0x3C00, -3);
    float yv[kQ80BlockElems];
    q8_0_dequant_block(y, yv);
    for (int j = 0; j < kQ80BlockElems; ++j) {
        CHECK(std::bit_cast<std::uint32_t>(yv[j]) == std::bit_cast<std::uint32_t>(-3.0f));
    }
}

void test_handmade_dot() {
    // 权重：d0 = 1.0、码全 2 -> 每个权重 = +1。激活：两个 Q8_0 块，d = 1.0、qs 全 1 -> 每个 = 1。
    // 整数侧 each 块 = 2 个子块 × 32 个元素 × 1 × 1 = 64，乘 d0 -> 64。
    const auto w = flat_q20(0x3C00, 0xAA);  // 0xAA = 0b10101010 -> 4 个码都是 2
    Q80Block y[2] = {flat_q80(0x3C00, 1), flat_q80(0x3C00, 1)};
    const float got = q2_0_dot_q8_0(w.data(), y, 1);
    CHECK(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(64.0f));

    // 两个 Q2_0 块要配四个 Q8_0 块（每块 64 个权重对应 2×32 个激活）：结果翻倍。
    const auto w2 = flat_q20(0x3C00, 0xAA);
    std::vector<std::uint8_t> ww = w2;
    ww.insert(ww.end(), w2.begin(), w2.end());
    Q80Block y2[4] = {flat_q80(0x3C00, 1), flat_q80(0x3C00, 1), flat_q80(0x3C00, 1),
                      flat_q80(0x3C00, 1)};
    CHECK(std::bit_cast<std::uint32_t>(q2_0_dot_q8_0(ww.data(), y2, 2)) ==
          std::bit_cast<std::uint32_t>(128.0f));
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kQ20RealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kQ20BlockBytes));

    float out[kQ20BlockElems];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_q2_0_block(w.data() + b * kQ20BlockBytes, out);
        const float d = f16_bits_to_f32(static_cast<std::uint16_t>(
            w[b * kQ20BlockBytes] | (w[b * kQ20BlockBytes + 1] << 8)));
        for (int j = 0; j < kQ20BlockElems; ++j) {
            // 结构不变量：值只能是 {-d, 0, +d, +2d}。
            const bool ok = out[j] == -d || out[j] == 0.0f || out[j] == d || out[j] == 2.0f * d ||
                            d == 0.0f;
            if (!ok) {
                std::printf("FAIL invariant block %d elem %d: v=%g d=%g\n", b, j,
                            static_cast<double>(out[j]), static_cast<double>(d));
                CHECK(false);
            }
            sum += static_cast<double>(out[j]);
            sumabs += std::fabs(static_cast<double>(out[j]));
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kQ20RealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kQ20RealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    const double tol = 1e-12 * (std::fabs(kQ20RealSumAbs) > 0 ? std::fabs(kQ20RealSumAbs) : 1.0);
    if (std::fabs(sum - kQ20RealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kQ20RealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kQ20RealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kQ20RealSumAbs);
        CHECK(false);
    }
}

void test_dot_matches_float_path_and_row_stride() {
    // 用真字节做权重（64 块），激活用合成 Q8_0（128 块，每块 32 个值）。
    const std::vector<std::uint8_t> w = from_hex(kQ20RealHex);
    std::vector<Q80Block> y(2 * kRealBlocks);
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = flat_q80(0x3C00, static_cast<std::int8_t>(static_cast<int>(i % 7) - 3));
    }

    const float fused = q2_0_dot_q8_0(w.data(), y.data(), kRealBlocks);

    double acc = 0.0;
    double mag = 0.0;
    float wv[kQ20BlockElems];
    float yv[kQ80BlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_q2_0_block(w.data() + b * kQ20BlockBytes, wv);
        for (int k = 0; k < 2; ++k) {
            q8_0_dequant_block(y[b * 2 + k], yv);
            for (int j = 0; j < kQ80BlockElems; ++j) {
                const double term = static_cast<double>(wv[k * kQ80BlockElems + j]) *
                                    static_cast<double>(yv[j]);
                acc += term;
                mag += std::fabs(term);
            }
        }
    }
    const double rel = std::fabs(static_cast<double>(fused) - acc) / (mag > 0.0 ? mag : 1.0);
    if (!(rel < 1e-6)) {
        std::printf("FAIL fused vs float path: fused %.9g float %.9g rel %.3g\n",
                    static_cast<double>(fused), acc, rel);
        CHECK(false);
    }

    // 逐行 GEMV：把 64 块切成 2 行 × 32 块。矩阵乘向量时所有行共用同一组激活，
// 故每行的结果必须等于「对该行字节、配同一组激活」的单行点积。
    float gemv[2];
    q2_0_gemv_q8_0(w.data(), 2, 32, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = q2_0_dot_q8_0(w.data() + r * 32 * kQ20BlockBytes, y.data(), 32);
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
    test_dot_matches_float_path_and_row_stride();
    std::puts("q2_0: block dequant and Q2_0xQ8_0 dot pinned by handmade, real bytes and float cross-check");
    return 0;
}