// CUDA 侧的 IQ2_XXS × Q8_K 与 IQ1_M × Q8_K 逐行点积：两档各自与主机那份在**真模型字节**上对齐。
// 只有 -DQINFER_ENABLE_CUDA=ON 时才构建（ADR-008 第 2 条：CI 在没有 GPU 的 runner 上不编 CUDA）。
//
// 每档四路证据（与 Q2_0、IQ2_S 那两块同法）：
//   1. 真模型字节 tile 成 640 行 × 每行 10 块（gate/up 的几何）后，设备与主机逐行对照：
//      报出位相同的行数与全局 L1 相对差，后者要求 < 1e-6。
//   2. 与独立参考一致：走反量化那条路（权重逐块反量化、激活按 Q8_K 的 d·qs 展开）双精度累加，
//      与实现的整数域累加不是同一条算术。
//   3. 两条错读法由变异测试覆盖，都在本文件的判据上致死（见提交里那次记录）。
//   4. 参数不合法要明确失败。
//
// --bench 是真机上的计时入口（每遍都要重新把权重搬上去，故是含搬运的下界）。
#include "kernels/cuda/iq_gemv.hpp"

#include "check.hpp"
#include "kernels/iq1m.hpp"
#include "kernels/iq2xxs.hpp"
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

// 两份都由对应的 measure/*_oracle.py 生成。
#include "iq1m_oracle_data.inc"
#include "iq2xxs_oracle_data.inc"

constexpr int kRows = 640;         // gate/up 的输出维（ffn 640）
constexpr int kBlocksPerRow = 10;  // 2560 / 256

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

struct Case {
    const char* name;
    int block_bytes;
    const char* hex;
    void (*host)(const std::uint8_t*, int, int, const Q8kBlock*, float*);
    bool (*dev)(const std::uint8_t*, int, int, const Q8kBlock*, float*, std::string&);
    void (*dequant)(const std::uint8_t*, float*);
};

const Case kIq2xxs{"IQ2_XXS", kIq2xxsBlockBytes, kIq2xxsRealHex, iq2xxs_gemv_q8k,
                   iq2xxs_gemv_q8k_cuda, dequant_iq2xxs_block};
const Case kIq1m{"IQ1_M", kIq1mBlockBytes, kIq1mRealHex, iq1m_gemv_q8k, iq1m_gemv_q8k_cuda,
                 dequant_iq1m_block};

// 真字节 tile 成 kRows × kBlocksPerRow 的矩阵。
std::vector<std::uint8_t> tiled(const Case& c) {
    const std::vector<std::uint8_t> src = from_hex(c.hex);
    CHECK(src.size() % static_cast<std::size_t>(c.block_bytes) == 0);
    const std::size_t n_real = src.size() / static_cast<std::size_t>(c.block_bytes);
    CHECK(n_real > 0);
    const std::size_t n_blk = static_cast<std::size_t>(kRows) * kBlocksPerRow;
    std::vector<std::uint8_t> m(n_blk * static_cast<std::size_t>(c.block_bytes));
    for (std::size_t i = 0; i < n_blk; ++i) {
        std::memcpy(m.data() + i * static_cast<std::size_t>(c.block_bytes),
                    src.data() + (i % n_real) * static_cast<std::size_t>(c.block_bytes),
                    static_cast<std::size_t>(c.block_bytes));
    }
    return m;
}

std::vector<Q8kBlock> activation() {
    std::vector<float> x(static_cast<std::size_t>(kBlocksPerRow) * kQ8kBlockElems);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.03125f;
    }
    std::vector<Q8kBlock> y(kBlocksPerRow);
    q8k_quantize_row(x.data(), y.data(), kBlocksPerRow);
    return y;
}

double rel_l1(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        num += std::fabs(static_cast<double>(got[i]) - want[i]);
        den += std::fabs(want[i]);
    }
    return den > 0.0 ? num / den : num;
}

void run_case(const Case& c) {
    const std::vector<std::uint8_t> w = tiled(c);
    const std::vector<Q8kBlock> y = activation();
    std::vector<float> host(kRows, 0.0f), dev(kRows, 0.0f);
    c.host(w.data(), kRows, kBlocksPerRow, y.data(), host.data());

    std::string err;
    CHECK(c.dev(w.data(), kRows, kBlocksPerRow, y.data(), dev.data(), err));
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
    CHECK(den > 0.0);

    // 独立参考：反量化那条路 + 双精度累加。
    const std::size_t row_bytes = static_cast<std::size_t>(kBlocksPerRow) * c.block_bytes;
    std::vector<float> wv(kIq2xxsValuesPerBlock);
    std::vector<double> ref(kRows, 0.0);
    for (int r = 0; r < kRows; ++r) {
        double acc = 0.0;
        for (int i = 0; i < kBlocksPerRow; ++i) {
            c.dequant(w.data() + static_cast<std::size_t>(r) * row_bytes +
                          static_cast<std::size_t>(i) * static_cast<std::size_t>(c.block_bytes),
                      wv.data());
            const Q8kBlock& yb = y[static_cast<std::size_t>(i)];
            for (int k = 0; k < kQ8kBlockElems; ++k) {
                acc += static_cast<double>(wv[static_cast<std::size_t>(k)]) *
                       static_cast<double>(yb.d) * static_cast<double>(yb.qs[k]);
            }
        }
        ref[static_cast<std::size_t>(r)] = acc;
    }
    const double rel_ref = rel_l1(dev, ref);
    std::printf("  %s：设备 vs 主机 位相同的行 %d/%d、L1 相对差 %.3g；vs 反量化双精度参考 %.3g\n",
                c.name, bit_same, kRows, rel, rel_ref);
    CHECK(rel < 1e-6);
    CHECK(rel_ref < 1e-6);
    CHECK(bit_same > 0);
}

void test_bad_inputs_fail_loudly() {
    std::vector<std::uint8_t> w(static_cast<std::size_t>(kIq2xxsBlockBytes), 0);
    std::vector<Q8kBlock> y(1);
    std::vector<float> out(1, 0.0f);
    std::string err;
    for (const Case* c : {&kIq2xxs, &kIq1m}) {
        err.clear();
        CHECK(!c->dev(nullptr, 1, 1, y.data(), out.data(), err));
        CHECK(!err.empty());
        err.clear();
        CHECK(!c->dev(w.data(), 0, 1, y.data(), out.data(), err));
        CHECK(!err.empty());
        err.clear();
        CHECK(!c->dev(w.data(), 1, 1, y.data(), nullptr, err));
        CHECK(!err.empty());
    }
}

int bench(const Case& c, int iters) {
    const std::vector<std::uint8_t> w = tiled(c);
    const std::vector<Q8kBlock> y = activation();
    std::vector<float> out(kRows, 0.0f), ref(kRows, 0.0f);
    std::string err;
    c.host(w.data(), kRows, kBlocksPerRow, y.data(), ref.data());
    CHECK(c.dev(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
    const double rel = rel_l1(out, std::vector<double>(ref.begin(), ref.end()));
    const double mib = static_cast<double>(w.size()) / (1024.0 * 1024.0);
    std::printf("%s 几何 %d 行 × %d 块/行（权重 %.3f MiB），%d 遍\n", c.name, kRows,
                kBlocksPerRow, mib, iters);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) c.host(w.data(), kRows, kBlocksPerRow, y.data(), out.data());
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        CHECK(c.dev(w.data(), kRows, kBlocksPerRow, y.data(), out.data(), err));
    }
    auto t2 = std::chrono::steady_clock::now();

    const double host_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(iters);
    const double dev_us =
        std::chrono::duration<double, std::micro>(t2 - t1).count() / static_cast<double>(iters);
    std::printf("  主机 %.1f us/次（%.2f GiB/s 等效）；设备 %.1f us/次（%.2f GiB/s 等效，含搬运）；"
                "一致性 %.3g\n",
                host_us, mib / host_us * 1e6 / 1024.0, dev_us, mib / dev_us * 1e6 / 1024.0, rel);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        const int iters = argc >= 3 ? std::atoi(argv[2]) : 50;
        bench(kIq2xxs, iters);
        bench(kIq1m, iters);
        return 0;
    }
    run_case(kIq2xxs);
    run_case(kIq1m);
    test_bad_inputs_fail_loudly();
    std::puts("iq_gemv_cuda: IQ2_XXS 与 IQ1_M 的设备侧逐行点积与主机那份在真模型字节上一致");
    return 0;
}
