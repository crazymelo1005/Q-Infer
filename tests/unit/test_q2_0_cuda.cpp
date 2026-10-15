// CUDA 侧的 Q2_0 × Q8_0 逐行点积：与主机那份在**真模型字节**上对齐。只有 -DQINFER_ENABLE_CUDA=ON
// 时才构建（ADR-008 第 2 条：CI 在没有 GPU 的 runner 上不编 CUDA）。
//
// 四路证据：
//   1. 真模型字节（q2_0_oracle_data.inc 的 64 块 tile 成 2560 行 × 每行 10 块，正是部署那份 down
//      投影的几何）上，设备与主机逐行对照：报出位相同的行数与全局 L1 相对差，后者要求 < 1e-6。
//   2. 与独立双精度参考一致（同一相对判据）。
//   3. 两条错读法必须可观测：码映射漏掉那 -1、字节内高低位序取反——各自相对差都要 > 1e-2。
//   4. 参数不合法要明确失败，不静默给 0。
//
// --bench 是真机上的计时入口（本入口每遍都要重新把权重搬上去，故是含搬运的下界；权重常驻后的
// 数字要等 CUDA 组那一层）。
#include "kernels/cuda/q2_0_gemv.hpp"

#include "check.hpp"
#include "kernels/fp16.hpp"
#include "kernels/q8_0.hpp"
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

// 由 measure/q2_0_oracle.py 生成：kQ20RealHex / kQ20RealFirst16[16] / kQ20RealSum / kQ20RealSumAbs。
#include "q2_0_oracle_data.inc"

constexpr int kRealBlocks = 64;
constexpr int kRows = 2560;         // down 投影的输出维
constexpr int kBlocksPerRow = 10;   // 640 / 64
constexpr int kActBlocks = 2 * kBlocksPerRow;

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

// 真字节 tile 成 kRows × kBlocksPerRow 的矩阵（行距 kBlocksPerRow × 18 字节）。
std::vector<std::uint8_t> real_matrix() {
    const std::vector<std::uint8_t> src = from_hex(kQ20RealHex);
    CHECK(src.size() == static_cast<std::size_t>(kRealBlocks * kQ20BlockBytes));
    const std::size_t n_blk = static_cast<std::size_t>(kRows) * kBlocksPerRow;
    std::vector<std::uint8_t> m(n_blk * kQ20BlockBytes);
    for (std::size_t i = 0; i < n_blk; ++i) {
        const std::size_t blk = i % kRealBlocks;
        std::memcpy(m.data() + i * kQ20BlockBytes, src.data() + blk * kQ20BlockBytes,
                    kQ20BlockBytes);
    }
    return m;
}

// 激活：走本仓库的量化器（不是手搓块），量级取模型里常见的那一档。
std::vector<Q80Block> real_activation() {
    std::vector<float> x(static_cast<std::size_t>(kActBlocks) * kQ80BlockElems);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.0625f;
    }
    std::vector<Q80Block> y(kActBlocks);
    q8_0_quantize_row(x.data(), y.data(), kActBlocks);
    return y;
}

// 独立双精度参考（逐行）。三条读法：对的那条，以及两条刻意写错的。
enum class Reading { kRight, kDropOffset, kFlipNibbleOrder };

double ref_row(const std::uint8_t* row, const Q80Block* y, int n_blocks, Reading mode) {
    double acc = 0.0;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kQ20BlockBytes;
        const double d0 = static_cast<double>(
            f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8))));
        double sumi = 0.0;
        for (int k = 0; k < 2; ++k) {
            const Q80Block& yb = y[i * 2 + k];
            const double d1 = static_cast<double>(f16_bits_to_f32(yb.d_bits));
            const std::uint8_t* qs = blk + 2 + k * 8;
            double s = 0.0;
            for (int b = 0; b < 8; ++b) {
                const std::uint8_t byte = qs[b];
                int c[4];
                for (int j = 0; j < 4; ++j) {
                    const int shift = (mode == Reading::kFlipNibbleOrder) ? (6 - j * 2) : (j * 2);
                    const int code = (byte >> shift) & 3;
                    c[j] = (mode == Reading::kDropOffset) ? code : code - 1;
                }
                s += static_cast<double>(c[0]) * yb.qs[b * 4 + 0];
                s += static_cast<double>(c[1]) * yb.qs[b * 4 + 1];
                s += static_cast<double>(c[2]) * yb.qs[b * 4 + 2];
                s += static_cast<double>(c[3]) * yb.qs[b * 4 + 3];
            }
            sumi += d1 * s;
        }
        acc += d0 * sumi;
    }
    return acc;
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
    const std::vector<Q80Block> y = real_activation();
    std::vector<float> host(kRows, 0.0f), dev(kRows, 0.0f);
    q2_0_gemv_q8_0(w.data(), kRows, kBlocksPerRow, y.data(), host.data());

    std::string err;
    CHECK(q2_0_gemv_q8_0_cuda(w.data(), kRows, kBlocksPerRow, y.data(), dev.data(), err));
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
    CHECK(den > 0.0);  // 参考值不能是全零，否则下面的判据是空转
    CHECK(rel < 1e-6);
    CHECK(bit_same > 0);
}

void test_matches_double_reference_and_bad_readings_differ() {
    const std::vector<std::uint8_t> w = real_matrix();
    const std::vector<Q80Block> y = real_activation();
    std::vector<float> dev(kRows, 0.0f);
    std::string err;
    CHECK(q2_0_gemv_q8_0_cuda(w.data(), kRows, kBlocksPerRow, y.data(), dev.data(), err));

    const std::size_t row_bytes = static_cast<std::size_t>(kBlocksPerRow) * kQ20BlockBytes;
    std::vector<double> right(kRows), drop(kRows), flip(kRows);
    for (int r = 0; r < kRows; ++r) {
        const std::uint8_t* row = w.data() + static_cast<std::size_t>(r) * row_bytes;
        right[r] = ref_row(row, y.data(), kBlocksPerRow, Reading::kRight);
        drop[r] = ref_row(row, y.data(), kBlocksPerRow, Reading::kDropOffset);
        flip[r] = ref_row(row, y.data(), kBlocksPerRow, Reading::kFlipNibbleOrder);
    }
    const double rel_right = rel_l1(dev, right);
    const double rel_drop = rel_l1(dev, drop);
    const double rel_flip = rel_l1(dev, flip);
    std::printf("  vs 双精度参考 %.3g；错读法 漏 -1 偏移 %.3g、位序取反 %.3g\n", rel_right,
                rel_drop, rel_flip);
    CHECK(rel_right < 1e-6);
    // 两条错读法都必须与本实现有明显差别，否则前一条判据等于没测这件事。
    CHECK(rel_drop > 1e-2);
    CHECK(rel_flip > 1e-2);
}

void test_bad_inputs_fail_loudly() {
    std::vector<std::uint8_t> w(4 * kQ20BlockBytes, 0);
    std::vector<Q80Block> y(8);
    std::vector<float> out(4, 0.0f);
    std::string err;

    CHECK(!q2_0_gemv_q8_0_cuda(nullptr, 1, 1, y.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!q2_0_gemv_q8_0_cuda(w.data(), 0, 1, y.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!q2_0_gemv_q8_0_cuda(w.data(), 1, 1, y.data(), nullptr, err));
    CHECK(!err.empty());
}

int bench(int iters) {
    const std::vector<std::uint8_t> w = real_matrix();
    const std::vector<Q80Block> y = real_activation();
    std::vector<float> out(kRows, 0.0f);
    std::string err;

    // 先各自跑一遍确认一致，再计时——不然测的是一个算错的核。
    std::vector<float> ref(kRows, 0.0f);
    q2_0_gemv_q8_0(w.data(), kRows, kBlocksPerRow, y.data(), ref.data());
    CHECK(q2_0_gemv_q8_0_cuda(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
    const double rel = rel_l1(out, std::vector<double>(ref.begin(), ref.end()));
    std::printf("几何 %d 行 × %d 块/行（权重 %.3f MiB），%d 遍\n", kRows, kBlocksPerRow,
                static_cast<double>(w.size()) / (1024.0 * 1024.0), iters);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        q2_0_gemv_q8_0(w.data(), kRows, kBlocksPerRow, y.data(), out.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        CHECK(q2_0_gemv_q8_0_cuda(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
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
    (void) rel;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        return bench(argc >= 3 ? std::atoi(argv[2]) : 50);
    }
    test_device_matches_host_on_real_bytes();
    test_matches_double_reference_and_bad_readings_differ();
    test_bad_inputs_fail_loudly();
    std::puts("q2_0_cuda: 设备侧的 Q2_0xQ8_0 逐行点积与主机那份在真模型字节上一致");
    return 0;
}
