// bf16 投影的回归。不引第三方框架。
//
// 三条：与「权重与激活都按 bf16 舍入、float 累加」的独立参考逐位一致；激活**必须**按 bf16 舍入
// （不做的错读法与它对不上，且差值远大于浮点噪声）——这是路由门控那条实测结论（S-44）的对称面，
// 也是 QSA 索引器两个 BF16 投影要用的同一份算术；权重按 bf16 读而不是 fp16。
#include "kernels/bf16.hpp"

#include "check.hpp"
#include "experts/ffn.hpp"
#include "scheduling/thread_pool.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kRows = 5, kCols = 7;

struct Lcg {
    std::uint64_t s = 0x123456789ABCDEFull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

void test_matches_the_reference_and_rounds_the_activation() {
    Lcg g;
    std::vector<std::uint16_t> w(kRows * kCols);
    for (std::uint16_t& t : w) t = f32_to_bf16_bits(g.next());
    std::vector<float> x(kCols);
    for (float& t : x) t = g.next();  // 不刻意取 bf16 可表示的值，好让「有没有舍入」可观测

    std::vector<float> got(kRows, 0.0f);
    bf16_gemv(w.data(), kRows, kCols, x.data(), got.data());
    // 参考：同样先舍激活、再乘、float 累加。
    for (std::uint64_t o = 0; o < kRows; ++o) {
        float acc = 0.0f;
        for (std::uint64_t i = 0; i < kCols; ++i) {
            acc += bf16_bits_to_f32(w[o * kCols + i]) * bf16_bits_to_f32(f32_to_bf16_bits(x[i]));
        }
        CHECK(got[o] == acc);  // 同一条算术，逐位相同
    }

    // 错的读法：激活不舍入（直接乘 fp32 的 x）。
    double diff = 0.0, mag = 0.0;
    for (std::uint64_t o = 0; o < kRows; ++o) {
        float acc = 0.0f;
        for (std::uint64_t i = 0; i < kCols; ++i) {
            acc += bf16_bits_to_f32(w[o * kCols + i]) * x[i];
        }
        diff += std::fabs(static_cast<double>(acc) - static_cast<double>(got[o]));
        mag += std::fabs(static_cast<double>(got[o]));
    }
    const double rel = diff / (mag > 1e-30 ? mag : 1e-30);
    std::printf("错的读法「激活不舍入 bf16」的相对偏离 %.3g\n", rel);
    CHECK(rel > 1e-4);
}

void test_zero_and_shape_edges() {
    std::vector<std::uint16_t> w(4, 0);
    std::vector<float> x(4, 1.0f), out(1, 7.0f);
    bf16_gemv(w.data(), 1, 4, x.data(), out.data());
    CHECK(out[0] == 0.0f);  // 全零权重 -> 0（覆盖写，不留旧值）
}

// `run_bf16_gemv`（experts/ffn 的按行并行入口）与单线程逐位相同，且与 kernels 里那份直接调用一致。
// 真几何（路由是 512 行 × 2560）：5×7 那种规模切不出第二块，用例会空转。
void test_row_parallel_entry_is_bitwise_identical() {
    constexpr std::uint64_t rows = 512, cols = 2560;
    Lcg g;
    std::vector<std::uint16_t> w(rows * cols);
    for (std::uint16_t& t : w) t = f32_to_bf16_bits(g.next());
    std::vector<float> x(cols);
    for (float& t : x) t = g.next();

    std::string err;
    std::vector<float> ref(rows, 0.0f), got(rows, 0.0f);
    qinfer::scheduling::set_pool_threads(0);
    bf16_gemv(w.data(), rows, cols, x.data(), ref.data());
    CHECK(qinfer::experts::run_bf16_gemv(w.data(), rows, cols, x.data(), got.data(), err));
    qinfer::scheduling::set_pool_threads(8);
    std::vector<float> par(rows, 0.0f);
    CHECK(qinfer::experts::run_bf16_gemv(w.data(), rows, cols, x.data(), par.data(), err));
    qinfer::scheduling::set_pool_threads(0);

    int same_ref = 0, same_par = 0;
    for (std::uint64_t o = 0; o < rows; ++o) {
        if (std::bit_cast<std::uint32_t>(ref[o]) == std::bit_cast<std::uint32_t>(got[o])) ++same_ref;
        if (std::bit_cast<std::uint32_t>(ref[o]) == std::bit_cast<std::uint32_t>(par[o])) ++same_par;
    }
    std::printf("  run_bf16_gemv：与 kernels 那份位相同 %d/%llu、8 线程版位相同 %d/%llu\n", same_ref,
                (unsigned long long) rows, same_par, (unsigned long long) rows);
    CHECK(same_ref == static_cast<int>(rows));
    CHECK(same_par == static_cast<int>(rows));
}

}  // namespace

int main() {
    test_matches_the_reference_and_rounds_the_activation();
    test_zero_and_shape_edges();
    test_row_parallel_entry_is_bitwise_identical();
    std::puts("bf16_gemv: 权重与激活都按 bf16 读、float 累加，与参考逐位一致，"
              "「激活不舍入」那条错读法被拒");
    return 0;
}