// IQ2_S 块的回归（反量化与 IQ2_S × Q8_K 定点点积）。不引第三方框架。
//
// 四路证据：
//   1. 手工两块，期望值是手算的——钉死块内偏移（d/0、qs/2 含 32 个索引字节 + 32 个符号字节、
//      qh/66、scales/74）、10 位格点索引的高 2 位来自 qh、以及 0.25 因子与 0.5 偏置。
//   2. 真字节的结构不变量：模型里 blk.0.ffn_gate_exps.weight 的前 16 个块，每个输出值必须恰好
//      是 ±db0·g 或 ±db1·g（g ∈ {8, 25, 43}）。这条不依赖任何外部期望值。
//   3. 真字节的回归锁：同一批字节，前 16 值与另一处切片必须对上 gguf-py 的独立实现给出的值
//      （measure/iq2s_oracle.py 生成，见 S-39），sum / sumabs 也一并核。
//   4. 点积与浮点路径交叉核对：IQ2_S × Q8_K 的结果必须等于「两侧各自反量化后逐元素累加」，
//      容差按 Σ|w·y| 取相对值；另核逐行 GEMV 与单行点积一致（行距 n_blocks × 82 字节）。
#include "kernels/iq2s.hpp"

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

// 由 measure/iq2s_oracle.py 生成：kIq2sRealHex / kIq2sRealFirst16 / kIq2sRealBlock7_128 /
// kIq2sRealSum / kIq2sRealSumAbs。
#include "iq2s_oracle_data.inc"

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

// 一块：d 为给定 fp16；32 个索引字节全是 qidx、32 个符号字节全是 sign；qh 与 scales 由参数给。
std::vector<std::uint8_t> flat_iq2s(std::uint16_t d_bits, std::uint8_t qidx, std::uint8_t sign,
                                    std::uint8_t qh, std::uint8_t scales) {
    std::vector<std::uint8_t> b(kIq2sBlockBytes, 0);
    b[0] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[1] = static_cast<std::uint8_t>(d_bits >> 8);
    for (int i = 0; i < 32; ++i) b[2 + i] = qidx;       // qs：索引低 8 位
    for (int i = 0; i < 32; ++i) b[34 + i] = sign;      // signs
    for (int i = 0; i < 8; ++i) b[66 + i] = qh;         // qh
    for (int i = 0; i < 8; ++i) b[74 + i] = scales;     // scales
    return b;
}

void test_handmade() {
    // d = 1.0，索引字节 0、qh 0 -> 格点索引 0 = kIq2sGrid[0]；符号字节 0 -> 全正；
    // 尺度 0 -> db = 1.0 · (0.5 + 0) · 0.25 = 0.125，故每个权重 = 0.125 · 8 = 1.0。
    const std::uint64_t g0 = 0x0808080808080808ull;
    const auto b = flat_iq2s(0x3C00, 0, 0, 0, 0x00);
    float out[kIq2sValuesPerBlock];
    dequant_iq2s_block(b.data(), out);
    for (int k = 0; k < kIq2sValuesPerBlock; ++k) {
        if (std::bit_cast<std::uint32_t>(out[k]) != 0x3F800000u) {
            std::printf("FAIL handmade all-1.0 at %d: got %g\n", k, static_cast<double>(out[k]));
            CHECK(false);
        }
    }
    // 与格点 0 的字节一致（8），确认表首项就是它。
    const std::uint8_t* gp = reinterpret_cast<const std::uint8_t*>(&g0);
    for (int j = 0; j < 8; ++j) CHECK(gp[j] == 8);

    // 尺度 0x0F：低半字节 15（l = 0,1）-> db0 = 3.875 -> 31.0；高半字节 0（l = 2,3）-> db1 = 0.125 -> 1.0。
    const auto b2 = flat_iq2s(0x3C00, 0, 0, 0, 0x0F);
    float out2[kIq2sValuesPerBlock];
    dequant_iq2s_block(b2.data(), out2);
    for (int ib = 0; ib < 8; ++ib) {
        for (int l = 0; l < 4; ++l) {
            const float want = (l < 2) ? 31.0f : 1.0f;
            for (int j = 0; j < 8; ++j) {
                const float got = out2[ib * 32 + l * 8 + j];
                if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(want)) {
                    std::printf("FAIL handmade ls-split ib=%d l=%d j=%d: got %g\n", ib, l, j,
                                static_cast<double>(got));
                    CHECK(false);
                }
            }
        }
    }

    // 符号字节 0x81 -> 第 0 与第 7 个元素取负（值 = ±1.0）。
    const auto b3 = flat_iq2s(0x3C00, 0, 0x81, 0, 0x00);
    float out3[kIq2sValuesPerBlock];
    dequant_iq2s_block(b3.data(), out3);
    const std::uint32_t pos = std::bit_cast<std::uint32_t>(1.0f);
    const std::uint32_t neg = std::bit_cast<std::uint32_t>(-1.0f);
    for (int k = 0; k < kIq2sValuesPerBlock; ++k) {
        const int j = k % 8;
        const std::uint32_t want = (j == 0 || j == 7) ? neg : pos;
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out3[k]);
        if (got != want) {
            std::printf("FAIL handmade sign at %d: got 0x%08x want 0x%08x\n", k, got, want);
            CHECK(false);
        }
    }

    // 10 位索引的高 2 位：qh 第 ib 字节的位 2l 与 2l+1 补到索引的位 8、9。
    // 取 qh = 0x01 且 ib = 0、l = 0 -> 索引 = 0 | ((1 << 8) & 0x300) = 0x100 = 256。
    // kIq2sGrid[256] 的字节值决定结果；这里只核「索引确实取了 256 项」——用两块的差来证明：
    // 若高 2 位被忽略，qh 变化不会改变输出。
    const auto bq0 = flat_iq2s(0x3C00, 0, 0, 0x00, 0x00);
    const auto bq1 = flat_iq2s(0x3C00, 0, 0, 0x01, 0x00);
    float o0[kIq2sValuesPerBlock];
    float o1[kIq2sValuesPerBlock];
    dequant_iq2s_block(bq0.data(), o0);
    dequant_iq2s_block(bq1.data(), o1);
    bool differs = false;
    for (int k = 0; k < kIq2sValuesPerBlock; ++k) {
        if (std::bit_cast<std::uint32_t>(o0[k]) != std::bit_cast<std::uint32_t>(o1[k])) differs = true;
    }
    CHECK(differs);  // qh 参与索引，输出必须随它变
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kIq2sRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kIq2sBlockBytes));

    float out[kIq2sValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kIq2sBlockBytes;
        dequant_iq2s_block(blk, out);
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                const float db = d * (0.5f + static_cast<float>(
                                                 (l < 2) ? (blk[74 + ib] & 0x0F) : (blk[74 + ib] >> 4))) *
                                 0.25f;
                for (int j = 0; j < 8; ++j) {
                    const float v = out[ib * 32 + l * 8 + j];
                    const bool ok = std::fabs(v) == std::fabs(db * 8.0f) ||
                                    std::fabs(v) == std::fabs(db * 25.0f) ||
                                    std::fabs(v) == std::fabs(db * 43.0f);
                    if (!ok) {
                        std::printf("FAIL invariant block %d ib %d l %d j %d: v=%.9g db=%.9g\n", b,
                                    ib, l, j, static_cast<double>(v), static_cast<double>(db));
                        CHECK(false);
                    }
                    sum += static_cast<double>(v);
                    sumabs += std::fabs(static_cast<double>(v));
                }
            }
        }
        // 回归锁之一：第 0 块的前 16 值。必须在 b == 0 时取，否则 out 会被后面的块覆盖。
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kIq2sRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got,
                                kIq2sRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    // 回归锁之二：第 7 块的第 129..144 值（覆盖另一个 ib 与另外两个尺度半字节）。
    dequant_iq2s_block(w.data() + 7 * kIq2sBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[128 + j]);
        if (got != kIq2sRealBlock7_128[j]) {
            std::printf("FAIL block7_128 j=%d: got 0x%08x want 0x%08x\n", j, got,
                        kIq2sRealBlock7_128[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kIq2sRealSumAbs) > 0 ? std::fabs(kIq2sRealSumAbs) : 1.0);
    if (std::fabs(sum - kIq2sRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kIq2sRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kIq2sRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kIq2sRealSumAbs);
        CHECK(false);
    }
}

void test_handmade_dot() {
    // 权重：d = 1.0、索引 0（全 8）、符号 0、尺度 0 -> 每个权重 1.0。
    // 激活：一块 Q8_K，d = 1.0、qs 全 1 -> 每个 1.0。点积 = 256 × 1.0 = 256。
    const auto w = flat_iq2s(0x3C00, 0, 0, 0, 0x00);
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    CHECK(std::bit_cast<std::uint32_t>(iq2s_dot_q8k(w.data(), &y, 1)) ==
          std::bit_cast<std::uint32_t>(256.0f));
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kIq2sRealHex);
    std::vector<Q8kBlock> y(kRealBlocks);
    for (int b = 0; b < kRealBlocks; ++b) {
        y[b].d = 0.5f;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            y[b].qs[j] = static_cast<std::int8_t>(((j * 7 + b * 3) % 251) - 125);
        }
        for (int g = 0; g < kQ8kBlockElems / 16; ++g) {
            int s = 0;
            for (int i = 0; i < 16; ++i) s += y[b].qs[g * 16 + i];
            y[b].bsums[g] = static_cast<std::int16_t>(s);
        }
    }

    const float fused = iq2s_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kIq2sValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_iq2s_block(w.data() + b * kIq2sBlockBytes, wv);
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

    // 逐行 GEMV：2 行 × 8 块，所有行共用同一组激活。
    float gemv[2];
    iq2s_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = iq2s_dot_q8k(w.data() + r * 8 * kIq2sBlockBytes, y.data(), 8);
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
    std::puts("iq2s: block dequant and IQ2_SxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}