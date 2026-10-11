// 线程池的回归。两条判据：按行并行时每一行**恰好**被算一次（结构判据，不看浮点），以及
// 并行版与单线程版**逐位相同**（engine §16 那条口径的地基）。
#include "scheduling/thread_pool.hpp"

#include "check.hpp"
#include "experts/ffn.hpp"
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 2560;
constexpr int kCols = 640;  // 一行 10 个 64 值块（Q2_0，部署那份 down 的几何）
constexpr int kBlocksPerRow = kCols / 64;

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

// 由 measure/q2_0_oracle.py 生成的真模型字节。
#include "q2_0_oracle_data.inc"

void test_rows_covered_exactly_once() {
    for (const int threads : {1, 2, 4, 8}) {
        qinfer::scheduling::ThreadPool pool(threads);
        for (const std::uint64_t n : {1ull, 7ull, 64ull, 1000ull, 4096ull}) {
            std::vector<std::atomic<int>> hits(static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < hits.size(); ++i) hits[i].store(0);
            pool.run_rows(n, /*grain=*/8, [&](std::uint64_t b, std::uint64_t e) {
                for (std::uint64_t i = b; i < e; ++i) {
                    hits[static_cast<std::size_t>(i)].fetch_add(1);
                }
            });
            for (std::uint64_t i = 0; i < n; ++i) {
                if (hits[static_cast<std::size_t>(i)].load() != 1) {
                    std::printf("FAIL 行 %llu 被算了 %d 次（n=%llu threads=%d）\n",
                                (unsigned long long) i,
                                hits[static_cast<std::size_t>(i)].load(), (unsigned long long) n,
                                threads);
                    CHECK(false);
                }
            }
        }
    }
    std::printf("  覆盖：1/2/4/8 线程 × n=1/7/64/1000/4096，每行恰好一次\n");
}

void test_gemv_bitwise_same_as_single_thread() {
    qinfer::experts::MatrixSpec m;
    m.format = qinfer::experts::Format::kQ2_0;
    m.rows = kRows;
    m.cols = kCols;
    const std::vector<std::uint8_t> src = from_hex(kQ20RealHex);
    const std::size_t n_blk = static_cast<std::size_t>(kRows) * kBlocksPerRow;
    std::vector<std::uint8_t> w(n_blk * 18);
    for (std::size_t i = 0; i < n_blk; ++i) {
        std::memcpy(w.data() + i * 18, src.data() + (i % (src.size() / 18)) * 18, 18);
    }
    // 破掉一个巧合：tile 出来的块序列有周期（每 32 行重复），而块大小常是行数的整数倍——于是
    // 「切块时忘了按行偏移」会算出**一模一样**的值，变异就活下来了。给每行的尺度字节再抖一下，
    // 行与行就不再有那种周期性等价。
    for (int r = 0; r < kRows; ++r) {
        const std::size_t at = static_cast<std::size_t>(r) * kBlocksPerRow * 18 + 1;
        w[at] = static_cast<std::uint8_t>(w[at] ^ static_cast<std::uint8_t>((r % 7) + 1));
    }
    std::vector<float> x(kCols);
    for (int i = 0; i < kCols; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>((i % 29) - 14) * 0.0625f;
    }

    qinfer::experts::FfnScratch s;
    std::string err;
    CHECK(qinfer::experts::quantize_input(m, x.data(), s, err));

    std::vector<float> a(kRows, 0.0f);
    std::vector<float> b(kRows, 0.0f);
    qinfer::scheduling::set_pool_threads(0);
    CHECK(qinfer::experts::run_gemv(m, w.data(), s, a.data(), err));

    const int nt = 8;
    qinfer::scheduling::set_pool_threads(nt);
    for (int rep = 0; rep < 3; ++rep) {
        std::fill(b.begin(), b.end(), 0.0f);
        CHECK(qinfer::experts::run_gemv(m, w.data(), s, b.data(), err));
    }
    qinfer::scheduling::set_pool_threads(0);

    int same = 0;
    for (int r = 0; r < kRows; ++r) {
        if (std::bit_cast<std::uint32_t>(a[static_cast<std::size_t>(r)]) ==
            std::bit_cast<std::uint32_t>(b[static_cast<std::size_t>(r)])) {
            ++same;
        }
    }
    std::printf("  run_gemv：单线程 对 %d 线程，位相同 %d/%d\n", nt, same, kRows);
    CHECK(same == kRows);
}

}  // namespace

int main() {
    test_rows_covered_exactly_once();
    test_gemv_bitwise_same_as_single_thread();
    std::puts("thread_pool: 每行恰好一次，且按行并行与单线程逐位相同");
    return 0;
}
