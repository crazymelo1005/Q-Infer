// Q8_K 参考激活量化器的回归。不引第三方框架。
//
// 三路证据：
//   1. 手工两块，期望值是手算的——钉死块尺度的负号（iscale = -127/max，d 因此为负）、四舍五入到
//      最近整数、bsums 的 16 个一组。
//   2. 确定性合成向量的逐位回归，期望值来自 ggml 参考量化器的忠实转写（measure/iq2xs_oracle.py）。
//   3. 量化误差不变量：每个元素的重建误差不超过半个量化步长 |d|/2。
#include "kernels/q8k.hpp"

#include "check.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace qinfer::kernels;

namespace {

// 由 measure/iq2xs_oracle.py 生成：kQ8kD[2] / kQ8kQs[512] / kQ8kBsums[32]。
#include "q8k_oracle_data.inc"

constexpr int kBlocks = 2;

void fill_input(float* x) {
    for (int j = 0; j < kBlocks * kQ8kBlockElems; ++j) {
        x[j] = static_cast<float>(((j * 211 + 3) % 1021) - 510) * (1.0f / 128.0f);
    }
}

void test_handmade_block() {
    // x[0] = 1.0、x[1] = -0.25，其余 0。幅值最大者是 x[0] -> max = 1.0，iscale = -127。
    // qs[0] = round(-127·1.0) = -127；qs[1] = round(-127·-0.25) = round(31.75) = 32；其余 0。
    float x[kQ8kBlockElems] = {};
    x[0] = 1.0f;
    x[1] = -0.25f;
    Q8kBlock b;
    q8k_quantize_row(x, &b, 1);

    CHECK(std::bit_cast<std::uint32_t>(b.d) == std::bit_cast<std::uint32_t>(1.0f / -127.0f));
    CHECK(b.qs[0] == -127);
    CHECK(b.qs[1] == 32);
    for (int j = 2; j < kQ8kBlockElems; ++j) CHECK(b.qs[j] == 0);
    CHECK(b.bsums[0] == -95);
    for (int g = 1; g < kQ8kBlockElems / 16; ++g) CHECK(b.bsums[g] == 0);

    float out[kQ8kBlockElems];
    q8k_dequant_block(b, out);
    // 重建值是 d 的整数倍，故 x[1] = -0.25 只能落在 -32/127 = -0.25197 上，偏差不超过半步。
    const float half_step = std::fabs(b.d) * 0.5f;
    CHECK(std::fabs(out[0] - 1.0f) < 1e-6f);
    CHECK(std::fabs(out[1] + 0.25f) <= half_step + 1e-6f);
    CHECK(std::fabs(out[1] + 0.25f) > 0.0f);  // 不是恰好命中，免得上面那条变成恒真
    for (int j = 2; j < kQ8kBlockElems; ++j) CHECK(out[j] == 0.0f);
}

void test_zero_block() {
    float x[kQ8kBlockElems] = {};
    Q8kBlock b;
    q8k_quantize_row(x, &b, 1);
    CHECK(std::bit_cast<std::uint32_t>(b.d) == 0u);
    for (int j = 0; j < kQ8kBlockElems; ++j) CHECK(b.qs[j] == 0);
    for (int g = 0; g < kQ8kBlockElems / 16; ++g) CHECK(b.bsums[g] == 0);
}

void test_oracle() {
    float x[kBlocks * kQ8kBlockElems];
    fill_input(x);
    Q8kBlock b[kBlocks];
    q8k_quantize_row(x, b, kBlocks);

    for (int i = 0; i < kBlocks; ++i) {
        if (std::bit_cast<std::uint32_t>(b[i].d) != kQ8kD[i]) {
            std::printf("FAIL d[%d]: got 0x%08x want 0x%08x\n", i,
                        std::bit_cast<std::uint32_t>(b[i].d), kQ8kD[i]);
            CHECK(false);
        }
    }
    for (int j = 0; j < kBlocks * kQ8kBlockElems; ++j) {
        if (b[j / kQ8kBlockElems].qs[j % kQ8kBlockElems] != kQ8kQs[j]) {
            std::printf("FAIL qs[%d]: got %d want %d\n", j,
                        static_cast<int>(b[j / kQ8kBlockElems].qs[j % kQ8kBlockElems]),
                        static_cast<int>(kQ8kQs[j]));
            CHECK(false);
        }
    }
    for (int g = 0; g < kBlocks * (kQ8kBlockElems / 16); ++g) {
        if (b[g / (kQ8kBlockElems / 16)].bsums[g % (kQ8kBlockElems / 16)] != kQ8kBsums[g]) {
            std::printf("FAIL bsums[%d]: got %d want %d\n", g,
                        static_cast<int>(b[g / (kQ8kBlockElems / 16)].bsums[g % (kQ8kBlockElems / 16)]),
                        static_cast<int>(kQ8kBsums[g]));
            CHECK(false);
        }
    }
}

void test_error_within_half_step() {
    float x[kBlocks * kQ8kBlockElems];
    fill_input(x);
    Q8kBlock b[kBlocks];
    q8k_quantize_row(x, b, kBlocks);

    float out[kQ8kBlockElems];
    for (int i = 0; i < kBlocks; ++i) {
        q8k_dequant_block(b[i], out);
        const float half_step = std::fabs(b[i].d) * 0.5f;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            const float err = std::fabs(out[j] - x[i * kQ8kBlockElems + j]);
            if (!(err <= half_step + 1e-5f)) {
                std::printf("FAIL block %d elem %d: err %.9g half_step %.9g\n", i, j,
                            static_cast<double>(err), static_cast<double>(half_step));
                CHECK(false);
            }
        }
    }
}

}  // namespace

int main() {
    test_handmade_block();
    test_zero_block();
    test_oracle();
    test_error_within_half_step();
    std::puts("q8k: reference activation quantizer pinned by handmade, oracle and error bound");
    return 0;
}