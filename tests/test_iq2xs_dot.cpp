// IQ2_XS × Q8_K 定点点积的回归。不引第三方框架。
//
// 四路证据：
//   1. 手工两块，期望值是手算的——钉死 ls = 2s+1、末尾的 0.125 因子、以及组内前两个码字用低半
//      字节尺度后两个用高半字节。
//   2. 数据驱动回归：输入与期望值都来自 measure/iq2xs_oracle.py（ggml 那个 generic 实现的忠实
//      转写），逐位比较，不留容差。
//   3. 与浮点路径交叉核对：点积必须等于「反量化权重 × 反量化激活」的逐元素累加，容差按
//      Σ|w·y| 取相对值。这条独立于整数算法本身，专抓尺度或分组理解错。
//   4. 逐行 GEMV 与对同一段字节的单行点积逐位一致（行距 n_blocks × 74 字节）。
#include "kernels/iq2xs_dot.hpp"

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

// 由 measure/iq2xs_oracle.py 生成：kDotWeightsHex / kDotYD[2] / kDotYQs[512] / kDotOut[4]。
#include "iq2xs_dot_oracle_data.inc"

constexpr int kRows = 4;
constexpr int kBlocks = 2;

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

// 一块激活：尺度 d、所有 qs 都是 q。bsums 留 0——标量点积不用它。
Q8kBlock flat_y(float d, std::int8_t q) {
    Q8kBlock y{};
    y.d = d;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = q;
    return y;
}

void load_activations(Q8kBlock* y) {
    for (int i = 0; i < kBlocks; ++i) {
        y[i].d = std::bit_cast<float>(kDotYD[i]);
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            y[i].qs[j] = kDotYQs[i * kQ8kBlockElems + j];
        }
    }
}

void test_handmade() {
    // 权重：d = 1.0、码字全 0（格点 0 全 8、符号全正）、尺度全 0 -> ls = 1。反量化后每个权重
    //       = 1.0·(0.5+0)·0.25·8 = 1.0；激活 d = 1.0、qs 全 1 -> 每个激活 = 1.0；点积 = 256。
    // 整数侧每块：8 组 × (2+2) 码字 × 8 元素 × 8（格点） = 2048，乘 d·0.125 -> 256。
    const auto w = flat_block(0x3C00, 0, 0x00);
    const Q8kBlock y = flat_y(1.0f, 1);
    CHECK(std::bit_cast<std::uint32_t>(iq2xs_dot_q8k(w.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(256.0f));

    // 尺度 0x0F：低半字节 15（l = 0,1）-> ls1 = 31；高半字节 0（l = 2,3）-> ls2 = 1。
    // 每块 bsum = 8 组 × (128·31 + 128·1) = 32768，乘 0.125 -> 4096。
    const auto w2 = flat_block(0x3C00, 0, 0x0F);
    CHECK(std::bit_cast<std::uint32_t>(iq2xs_dot_q8k(w2.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(4096.0f));
}

void test_oracle_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kDotWeightsHex);
    CHECK(w.size() == static_cast<std::size_t>(kRows * kBlocks * kIq2xsBlockBytes));

    Q8kBlock y[kBlocks] = {};
    load_activations(y);

    float gemv[kRows];
    iq2xs_gemv_q8k(w.data(), kRows, kBlocks, y, gemv);
    for (int r = 0; r < kRows; ++r) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(gemv[r]);
        if (got != kDotOut[r]) {
            std::printf("FAIL gemv row %d: got 0x%08x want 0x%08x\n", r, got, kDotOut[r]);
            CHECK(false);
        }
        const float one = iq2xs_dot_q8k(w.data() + static_cast<std::size_t>(r) * kBlocks * kIq2xsBlockBytes,
                                        y, kBlocks);
        if (std::bit_cast<std::uint32_t>(one) != kDotOut[r]) {
            std::printf("FAIL single-row dot %d: got 0x%08x want 0x%08x\n", r,
                        std::bit_cast<std::uint32_t>(one), kDotOut[r]);
            CHECK(false);
        }
    }
}

void test_matches_float_path() {
    const std::vector<std::uint8_t> w = from_hex(kDotWeightsHex);
    Q8kBlock y[kBlocks] = {};
    load_activations(y);

    float fused[kRows];
    iq2xs_gemv_q8k(w.data(), kRows, kBlocks, y, fused);

    for (int r = 0; r < kRows; ++r) {
        double acc = 0.0;
        double mag = 0.0;
        for (int i = 0; i < kBlocks; ++i) {
            float wv[kIq2xsValuesPerBlock];
            dequant_iq2xs_block(w.data() + static_cast<std::size_t>(r * kBlocks + i) * kIq2xsBlockBytes, wv);
            float yv[kQ8kBlockElems];
            q8k_dequant_block(y[i], yv);
            for (int j = 0; j < kQ8kBlockElems; ++j) {
                const double term = static_cast<double>(wv[j]) * static_cast<double>(yv[j]);
                acc += term;
                mag += std::fabs(term);
            }
        }
        const double rel = std::fabs(static_cast<double>(fused[r]) - acc) / (mag > 0.0 ? mag : 1.0);
        if (!(rel < 1e-5)) {
            std::printf("FAIL row %d vs float path: fused %.9g float %.9g rel %.3g\n", r,
                        static_cast<double>(fused[r]), acc, rel);
            CHECK(false);
        }
    }
}

}  // namespace

int main() {
    test_handmade();
    test_oracle_and_row_stride();
    test_matches_float_path();
    std::puts("iq2xs_dot: fused integer dot pinned by handmade, oracle and float cross-check");
    return 0;
}