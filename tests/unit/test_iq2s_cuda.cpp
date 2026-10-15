// CUDA 侧的 IQ2_S × Q8_K 逐行点积：与主机那份在**真模型字节**上对齐。只有 -DQINFER_ENABLE_CUDA=ON
// 时才构建（ADR-008 第 2 条：CI 在没有 GPU 的 runner 上不编 CUDA）。
//
// 四路证据：
//   1. 真模型字节（iq2s_oracle_data.inc 的块 tile 成 640 行 × 每行 10 块，正是 gate/up 投影的几何）
//      上，设备与主机逐行对照：报出位相同的行数与全局 L1 相对差，后者要求 < 1e-6。
//   2. 与独立双精度参考一致（走反量化那条路，与实现的整数累加不是同一条算术），同一相对判据。
//   3. 两条错读法由变异测试覆盖（尺度漏掉 ×2、qh 的 2 位从低位取），都在本文件的判据上致死——
//      见提交里那次变异记录，不留「判据是空转」的可能。
//   4. 参数不合法要明确失败，不静默给 0。
//
// --bench 是真机上的计时入口（每遍都要重新把权重搬上去，故是含搬运的下界）。
#include "kernels/cuda/iq2s_gemv.hpp"

#include "check.hpp"
#include "kernels/fp16.hpp"
#include "kernels/q8k.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

// 由 measure/iq2s_oracle.py 生成：kIq2sRealHex / kIq2sRealFirst16 / kIq2sRealBlock7_128 /
// kIq2sRealSum / kIq2sRealSumAbs。
#include "iq2s_oracle_data.inc"

constexpr int kRows = 640;        // gate/up 的输出维（ffn 640）
constexpr int kBlocksPerRow = 10; // 2560 / 256

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

// 真字节 tile 成 kRows × kBlocksPerRow 的矩阵（行距 kBlocksPerRow × 82 字节）。
std::vector<std::uint8_t> real_matrix() {
    const std::vector<std::uint8_t> src = from_hex(kIq2sRealHex);
    CHECK(src.size() % static_cast<std::size_t>(kIq2sBlockBytes) == 0);
    const std::size_t n_real = src.size() / kIq2sBlockBytes;
    CHECK(n_real > 0);
    const std::size_t n_blk = static_cast<std::size_t>(kRows) * kBlocksPerRow;
    std::vector<std::uint8_t> m(n_blk * kIq2sBlockBytes);
    for (std::size_t i = 0; i < n_blk; ++i) {
        const std::size_t blk = i % n_real;
        std::memcpy(m.data() + i * kIq2sBlockBytes, src.data() + blk * kIq2sBlockBytes,
                    kIq2sBlockBytes);
    }
    return m;
}

// 激活：走本仓库的 Q8_K 量化器（不是手搓块），量级取模型里常见的那一档。
std::vector<Q8kBlock> real_activation() {
    std::vector<float> x(static_cast<std::size_t>(kBlocksPerRow) * kQ8kBlockElems);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.03125f;
    }
    std::vector<Q8kBlock> y(kBlocksPerRow);
    q8k_quantize_row(x.data(), y.data(), kBlocksPerRow);
    return y;
}

// 独立参考：走反量化那条路（权重逐块反量化、激活按 Q8_K 的 d·qs 展开），全程双精度累加。
// 与实现的「整数域累加再乘尺度」不是同一条算术，故它能抓住实现里的取位/尺度错。
std::vector<double> dequant_reference(const std::vector<std::uint8_t>& w,
                                      const std::vector<Q8kBlock>& y) {
    std::vector<double> out(kRows, 0.0);
    const std::size_t row_bytes = static_cast<std::size_t>(kBlocksPerRow) * kIq2sBlockBytes;
    std::vector<float> wv(kIq2sValuesPerBlock);
    for (int r = 0; r < kRows; ++r) {
        double acc = 0.0;
        for (int i = 0; i < kBlocksPerRow; ++i) {
            dequant_iq2s_block(w.data() + static_cast<std::size_t>(r) * row_bytes +
                                   static_cast<std::size_t>(i) * kIq2sBlockBytes,
                               wv.data());
            const Q8kBlock& yb = y[static_cast<std::size_t>(i)];
            for (int k = 0; k < kQ8kBlockElems; ++k) {
                acc += static_cast<double>(wv[static_cast<std::size_t>(k)]) *
                       static_cast<double>(yb.d) * static_cast<double>(yb.qs[k]);
            }
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
    return out;
}

double rel_l1(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        num += std::fabs(static_cast<double>(got[i]) - want[i]);
        den += std::fabs(want[i]);
    }
    return den > 0.0 ? num / den : num;
}

void test_device_matches_host_on_real_bytes() {
    const std::vector<std::uint8_t> w = real_matrix();
    const std::vector<Q8kBlock> y = real_activation();
    std::vector<float> host(kRows, 0.0f), dev(kRows, 0.0f);
    iq2s_gemv_q8k(w.data(), kRows, kBlocksPerRow, y.data(), host.data());

    std::string err;
    CHECK(iq2s_gemv_q8k_cuda(w.data(), kRows, kBlocksPerRow, y.data(), dev.data(), err));
    CHECK(err.empty());

    int bit_same = 0;
    double num = 0.0, den = 0.0;
    for (int r = 0; r < kRows; ++r) {
        if (std::bit_cast<std::uint32_t>(host[r]) == std::bit_cast<std::uint32_t>(dev[r])) {
            ++bit_same;
        }
        num += std::fabs(static_cast<double>(dev[r]) - static_cast<double>(host[r]));
        den += std::fabs(static_cast<double>(host[r]));
    }
    const double rel = den > 0.0 ? num / den : num;
    std::printf("  设备 vs 主机：位相同的行 %d/%d，L1 相对差 %.3g\n", bit_same, kRows, rel);
    CHECK(den > 0.0);  // 参考值不能是全零，否则判据是空转
    CHECK(rel < 1e-6);
    CHECK(bit_same > 0);
}

void test_matches_dequant_reference() {
    const std::vector<std::uint8_t> w = real_matrix();
    const std::vector<Q8kBlock> y = real_activation();
    std::vector<float> dev(kRows, 0.0f);
    std::string err;
    CHECK(iq2s_gemv_q8k_cuda(w.data(), kRows, kBlocksPerRow, y.data(), dev.data(), err));
    const double rel = rel_l1(dev, dequant_reference(w, y));
    std::printf("  vs 反量化双精度参考 %.3g\n", rel);
    CHECK(rel < 1e-6);
}

void test_bad_inputs_fail_loudly() {
    std::vector<std::uint8_t> w(static_cast<std::size_t>(kIq2sBlockBytes), 0);
    std::vector<Q8kBlock> y(1);
    std::vector<float> out(1, 0.0f);
    std::string err;

    CHECK(!iq2s_gemv_q8k_cuda(nullptr, 1, 1, y.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!iq2s_gemv_q8k_cuda(w.data(), 0, 1, y.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!iq2s_gemv_q8k_cuda(w.data(), 1, 1, y.data(), nullptr, err));
    CHECK(!err.empty());
}

int bench(int iters) {
    const std::vector<std::uint8_t> w = real_matrix();
    const std::vector<Q8kBlock> y = real_activation();
    std::vector<float> out(kRows, 0.0f), ref(kRows, 0.0f);
    std::string err;

    iq2s_gemv_q8k(w.data(), kRows, kBlocksPerRow, y.data(), ref.data());
    CHECK(iq2s_gemv_q8k_cuda(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
    const double rel = rel_l1(out, std::vector<double>(ref.begin(), ref.end()));
    std::printf("几何 %d 行 × %d 块/行（权重 %.3f MiB），%d 遍\n", kRows, kBlocksPerRow,
                static_cast<double>(w.size()) / (1024.0 * 1024.0), iters);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        iq2s_gemv_q8k(w.data(), kRows, kBlocksPerRow, y.data(), out.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        CHECK(iq2s_gemv_q8k_cuda(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
    }
    auto t2 = std::chrono::steady_clock::now();

    const double host_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(iters);
    const double dev_us =
        std::chrono::duration<double, std::micro>(t2 - t1).count() / static_cast<double>(iters);
    const double mib = static_cast<double>(w.size()) / (1024.0 * 1024.0);
    std::printf("主机 %.1f us/次（%.2f GiB/s 等效）；设备 %.1f us/次（%.2f GiB/s 等效，含搬运）\n",
                host_us, mib / host_us * 1e6 / 1024.0, dev_us, mib / dev_us * 1e6 / 1024.0);
    std::printf("一致性（设备 vs 主机 L1 相对差）%.3g\n", rel);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        return bench(argc >= 3 ? std::atoi(argv[2]) : 50);
    }
    test_device_matches_host_on_real_bytes();
    test_matches_dequant_reference();
    test_bad_inputs_fail_loudly();
    std::puts("iq2s_cuda: 设备侧的 IQ2_SxQ8_K 逐行点积与主机那份在真模型字节上一致");
    return 0;
}
