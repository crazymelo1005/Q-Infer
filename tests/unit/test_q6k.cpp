// Q6_K 块的回归（反量化与 Q6_K × Q8_K 定点点积）。不引第三方框架。
//
// 这个档与其他 K-quant 不同的三处，测试针对性钉住：
//   1. fp16 尺度在块**末尾**（偏移 208），不在块首；
//   2. 元素位置 l 用 scales[l/16 + 0]，而同一 l 的另外三个输出位（+32 / +64 / +96）分别用
//      scales[l/16 + 2 / +4 / +6]——同一 l 的四个 q 共享 is，但尺度偏移不同，这条最容易记错；
//   3. 码字带 -32 偏移，故 q ∈ [-32, 31]，不是无符号幅度。
//
// 四路证据：手算三块（基例 / 子尺度索引 / 码字抽取与高位参与）、真字节结构不变量、真字节回归锁
// （前 16 值与第 5 块中段 16 值 + sum/sumabs 对上 gguf-py）、点积手算与浮点路径交叉核、逐行 GEMV 一致。
#include "kernels/q6k.hpp"

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

// 由 measure/q6k_oracle.py 生成：kQ6kRealHex / kQ6kRealFirst16 / kQ6kRealBlock5_128 /
// kQ6kRealSum / kQ6kRealSumAbs。
#include "q6k_oracle_data.inc"

constexpr int kRealBlocks = 16;
constexpr int kDAt = 208;

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

void set_d(std::vector<std::uint8_t>& b, std::uint16_t d_bits) {
    b[kDAt] = static_cast<std::uint8_t>(d_bits & 0xFF);
    b[kDAt + 1] = static_cast<std::uint8_t>(d_bits >> 8);
}

void set_scales(std::vector<std::uint8_t>& b, const std::int8_t sc[16]) {
    for (int i = 0; i < 16; ++i) b[192 + i] = static_cast<std::uint8_t>(sc[i]);
}

void expect(std::uint32_t got, float want, const char* what) {
    if (got != std::bit_cast<std::uint32_t>(want)) {
        std::printf("FAIL %s: got %.9g want %.9g\n", what, static_cast<double>(std::bit_cast<float>(got)),
                    static_cast<double>(want));
        CHECK(false);
    }
}

void test_handmade() {
    // 基例：块首 ql/qh 全 0 -> 每个码字 = 0 - 32 = -32；scales 全 1；d = 1.0 -> 每个值 -32。
    {
        std::vector<std::uint8_t> b(kQ6kBlockBytes, 0);
        std::int8_t sc[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        set_scales(b, sc);
        set_d(b, 0x3C00);
        float out[kQ6kValuesPerBlock];
        dequant_q6k_block(b.data(), out);
        for (int k = 0; k < kQ6kValuesPerBlock; ++k) expect(std::bit_cast<std::uint32_t>(out[k]), -32.0f, "base");
    }
    // 子尺度索引：scales = 1..16，ql/qh 全 0（码字恒 -32）。
    // 前半：l=0..15 用 sc[0]，l=16..31 用 sc[1]；+32 用 sc[2]/sc[3]；+64 用 sc[4]/sc[5]；+96 用 sc[6]/sc[7]。
    // 后半：同一 l 的尺度整体后移 8。
    {
        std::vector<std::uint8_t> b(kQ6kBlockBytes, 0);
        std::int8_t sc[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        set_scales(b, sc);
        set_d(b, 0x3C00);
        float out[kQ6kValuesPerBlock];
        dequant_q6k_block(b.data(), out);
        for (int l = 0; l < 32; ++l) {
            const int base = (l < 16 ? 0 : 1);
            expect(std::bit_cast<std::uint32_t>(out[l]), -32.0f * static_cast<float>(sc[base + 0]), "subscale l");
            expect(std::bit_cast<std::uint32_t>(out[l + 32]), -32.0f * static_cast<float>(sc[base + 2]), "subscale l+32");
            expect(std::bit_cast<std::uint32_t>(out[l + 64]), -32.0f * static_cast<float>(sc[base + 4]), "subscale l+64");
            expect(std::bit_cast<std::uint32_t>(out[l + 96]), -32.0f * static_cast<float>(sc[base + 6]), "subscale l+96");
            const int base2 = 8 + base;
            expect(std::bit_cast<std::uint32_t>(out[128 + l]), -32.0f * static_cast<float>(sc[base2 + 0]), "subscale half1");
            expect(std::bit_cast<std::uint32_t>(out[128 + l + 32]), -32.0f * static_cast<float>(sc[base2 + 2]), "subscale half1+32");
            expect(std::bit_cast<std::uint32_t>(out[128 + l + 64]), -32.0f * static_cast<float>(sc[base2 + 4]), "subscale half1+64");
            expect(std::bit_cast<std::uint32_t>(out[128 + l + 96]), -32.0f * static_cast<float>(sc[base2 + 6]), "subscale half1+96");
        }
    }
    // 码字抽取与高位参与：ql[0] = 0x0F、qh[0] 的位 0..1 = 2 -> 第 0 个码字 = 15 | 32 - 32 = 15；
    // 同一 l 的另三路仍取到 0 位的 2 位（位 2..3 / 4..5 / 6..7 都为 0）-> 仍是 -32。
    {
        std::vector<std::uint8_t> b(kQ6kBlockBytes, 0);
        std::int8_t sc[16];
        for (int i = 0; i < 16; ++i) sc[i] = 1;
        set_scales(b, sc);
        set_d(b, 0x3C00);
        b[0] = 0x0F;   // ql[0] 低半字节 = 15
        b[128] = 0x02; // qh[0] 位 0..1 = 2
        float out[kQ6kValuesPerBlock];
        dequant_q6k_block(b.data(), out);
        expect(std::bit_cast<std::uint32_t>(out[0]), 15.0f, "code l0");
        expect(std::bit_cast<std::uint32_t>(out[32]), -32.0f, "code l0+32");
        expect(std::bit_cast<std::uint32_t>(out[64]), -32.0f, "code l0+64");
        expect(std::bit_cast<std::uint32_t>(out[96]), -32.0f, "code l0+96");
    }
    // 高 2 位真的参与：把 qh[5] 的位 6..7 置 3 -> 第 5 个元素的第 4 路码字 +48。实现若漏掉就与基例相同。
    {
        std::vector<std::uint8_t> b0(kQ6kBlockBytes, 0);
        std::int8_t sc[16];
        for (int i = 0; i < 16; ++i) sc[i] = 1;
        set_scales(b0, sc);
        set_d(b0, 0x3C00);
        auto b1 = b0;
        b1[128 + 5] = 0xC0;  // qh[5] 位 6..7 = 3
        float o0[kQ6kValuesPerBlock];
        float o1[kQ6kValuesPerBlock];
        dequant_q6k_block(b0.data(), o0);
        dequant_q6k_block(b1.data(), o1);
        expect(std::bit_cast<std::uint32_t>(o1[5 + 96]), 16.0f, "high bits l5+96");  // 48 - 32
        bool differs = false;
        for (int k = 0; k < kQ6kValuesPerBlock; ++k) {
            if (std::bit_cast<std::uint32_t>(o0[k]) != std::bit_cast<std::uint32_t>(o1[k])) differs = true;
        }
        CHECK(differs);
    }
}

void test_real_bytes_invariant_and_regression() {
    const std::vector<std::uint8_t> w = from_hex(kQ6kRealHex);
    CHECK(w.size() == static_cast<std::size_t>(kRealBlocks * kQ6kBlockBytes));

    float out[kQ6kValuesPerBlock];
    double sum = 0.0;
    double sumabs = 0.0;
    for (int b = 0; b < kRealBlocks; ++b) {
        const std::uint8_t* blk = w.data() + b * kQ6kBlockBytes;
        dequant_q6k_block(blk, out);
        const float d = f16_bits_to_f32(
            static_cast<std::uint16_t>(blk[kDAt] | (blk[kDAt + 1] << 8)));
        const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            for (int half = 0; half < 2; ++half) {
                const std::int8_t* sch = sc + 8 * half;
                const int offs[4] = {0, 32, 64, 96};
                for (int slot = 0; slot < 4; ++slot) {
                    const float dl = d * static_cast<float>(sch[is + 2 * slot]);
                    const float v = out[128 * half + l + offs[slot]];
                    const float q = (dl == 0.0f) ? 0.0f : v / dl;
                    const bool is_int = std::fabs(q - std::round(q)) <= 1e-4f;
                    if (!is_int || q < -32.0f - 1e-4f || q > 31.0f + 1e-4f) {
                        std::printf("FAIL invariant b=%d l=%d half=%d slot=%d: v=%.9g dl=%.9g q=%.9g\n",
                                    b, l, half, slot, static_cast<double>(v), static_cast<double>(dl),
                                    static_cast<double>(q));
                        CHECK(false);
                    }
                    sum += static_cast<double>(v);
                    sumabs += std::fabs(static_cast<double>(v));
                }
            }
        }
        if (b == 0) {
            for (int j = 0; j < 16; ++j) {
                const std::uint32_t got = std::bit_cast<std::uint32_t>(out[j]);
                if (got != kQ6kRealFirst16[j]) {
                    std::printf("FAIL first16 j=%d: got 0x%08x want 0x%08x\n", j, got, kQ6kRealFirst16[j]);
                    CHECK(false);
                }
            }
        }
    }
    dequant_q6k_block(w.data() + 5 * kQ6kBlockBytes, out);
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t got = std::bit_cast<std::uint32_t>(out[128 + j]);
        if (got != kQ6kRealBlock5_128[j]) {
            std::printf("FAIL block5_128 j=%d: got 0x%08x want 0x%08x\n", j, got, kQ6kRealBlock5_128[j]);
            CHECK(false);
        }
    }
    const double tol = 1e-12 * (std::fabs(kQ6kRealSumAbs) > 0 ? std::fabs(kQ6kRealSumAbs) : 1.0);
    if (std::fabs(sum - kQ6kRealSum) > tol) {
        std::printf("FAIL sum: got %.17g want %.17g\n", sum, kQ6kRealSum);
        CHECK(false);
    }
    if (std::fabs(sumabs - kQ6kRealSumAbs) > tol) {
        std::printf("FAIL sumabs: got %.17g want %.17g\n", sumabs, kQ6kRealSumAbs);
        CHECK(false);
    }
}

void test_handmade_dot() {
    // 权重：d = 0.5、scales 全 1、ql/qh 全 0 -> 每个权重 0.5 × 1 × (-32) = -16。
    // 激活：Q8_K d = 1.0、qs 全 1 -> 每个 1.0。点积 = 256 × (-16) = -4096。
    std::vector<std::uint8_t> w(kQ6kBlockBytes, 0);
    std::int8_t sc[16];
    for (int i = 0; i < 16; ++i) sc[i] = 1;
    set_scales(w, sc);
    set_d(w, 0x3800);  // fp16 0.5
    Q8kBlock y{};
    y.d = 1.0f;
    for (int j = 0; j < kQ8kBlockElems; ++j) y.qs[j] = 1;
    expect(std::bit_cast<std::uint32_t>(q6k_dot_q8k(w.data(), &y, 1)), -4096.0f, "handmade dot");
}

void test_dot_and_row_stride() {
    const std::vector<std::uint8_t> w = from_hex(kQ6kRealHex);
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

    const float fused = q6k_dot_q8k(w.data(), y.data(), kRealBlocks);
    double acc = 0.0;
    double mag = 0.0;
    float wv[kQ6kValuesPerBlock];
    float yv[kQ8kBlockElems];
    for (int b = 0; b < kRealBlocks; ++b) {
        dequant_q6k_block(w.data() + b * kQ6kBlockBytes, wv);
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
    q6k_gemv_q8k(w.data(), 2, 8, y.data(), gemv);
    for (int r = 0; r < 2; ++r) {
        const float one = q6k_dot_q8k(w.data() + r * 8 * kQ6kBlockBytes, y.data(), 8);
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
    std::puts("q6k: block dequant and Q6_KxQ8_K dot pinned by handmade, real bytes and float cross-check");
    return 0;
}