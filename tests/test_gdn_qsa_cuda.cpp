// 阶段 A 设备路径对照载体：GDN/QSA 稠密段各内核的真字节夹具与判据。
// 本文件由 measure/ 的 oracle 工具生成的 .inc 提供真字节（部署 GGUF 的投影权重前若干块），
// 目标判据照 ADR-009 分布级口径：位相同比例 + L1 相对差小（h 量化在舍入边界可差一步）。
// 后续内核（gdn_proj_gemv / gdn_preprocess_cuda / gdn_step_cuda / qsa_*_cuda）逐个在这里对照。
#include "dense/gdn_preprocess.hpp"
#include "dense/gdn_preprocess_cuda.hpp"
#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"
#include "kernels/cuda/gdn_proj_gemv.hpp"
#include "check.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// oracle .inc 用 std::uint32_t 等类型，须在 <cstdint> 之后引入。
// 这些是阶段 A 的稠密投影档夹具（gdn_ 前缀，避免与主机同档测试的 .inc 撞名）。
#include "gdn_iq3s_oracle_data.inc"
#include "gdn_iq4xs_oracle_data.inc"
#include "gdn_q6k_oracle_data.inc"

namespace {

// hex 字符串 → 字节（oracle .inc 的 kXxxRealHex 是纯 hex 串）。
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

// 分发级判据（ADR-009）：位相同比例 ≥ 0.99 且 L1 相对差 ≤ 1e-4。返回 0 = 通过。
int compare_rows(const std::vector<float>& a, const std::vector<float>& b, double l1_threshold,
                 double bit_ratio_threshold) {
    CHECK(a.size() == b.size());
    std::uint64_t same = 0, total = a.size();
    double diff_sum = 0.0, abs_sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i])) ++same;
        diff_sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        abs_sum += std::abs(static_cast<double>(a[i]));
    }
    const double l1 = abs_sum > 0.0 ? diff_sum / abs_sum : 0.0;
    const double bit_ratio = total > 0 ? static_cast<double>(same) / static_cast<double>(total) : 1.0;
    std::printf("  bit-same %llu/%llu (%.3f), L1 rel %.3e\n",
                static_cast<unsigned long long>(same), static_cast<unsigned long long>(total),
                bit_ratio, l1);
    return bit_ratio >= bit_ratio_threshold && l1 <= l1_threshold ? 0 : 1;
}

// 夹具可用性：三个新档的 oracle .inc 能解码、块数与字节数自洽（Task 1 基线，只钉数据本身）。
void test_oracle_fixtures_load() {
    const auto iq4xs = from_hex(kIq4xsRealHex);
    const auto iq3s = from_hex(kIq3sRealHex);
    const auto q6k = from_hex(kQ6kRealHex);
    // 每块字节数（oracle 头注释写明）：IQ4_XS 136、IQ3_S 110、Q6_K 210；各取 64 块。
    CHECK(iq4xs.size() == 64 * 136);
    CHECK(iq3s.size() == 64 * 110);
    CHECK(q6k.size() == 64 * 210);
    std::printf("oracle 夹具自洽：IQ4_XS %zu B、IQ3_S %zu B、Q6_K %zu B\n", iq4xs.size(),
                iq3s.size(), q6k.size());
}

// 一个档的一次对照：oracle 真字节（64 块）循环 tile 成 rows × blocks 矩阵，
// 主机 quantize_input + run_gemv 对设备 gdn_proj_gemv，compare_rows 判据。
int gemv_against_host(qinfer::experts::Format fmt, const std::string& hex, std::uint64_t cols,
                      std::uint64_t rows, std::uint64_t blocks_per_row) {
    std::printf("== %s: cols %llu rows %llu 块/行 %llu ==\n", qinfer::experts::format_name(fmt),
                static_cast<unsigned long long>(cols), static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(blocks_per_row));
    const std::vector<std::uint8_t> src = from_hex(hex);
    constexpr std::uint64_t kSrcBlocks = 64;
    const std::uint64_t blk_bytes = src.size() / kSrcBlocks;
    CHECK(src.size() % kSrcBlocks == 0);
    const std::uint64_t n_blk = rows * blocks_per_row;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(n_blk * blk_bytes));
    for (std::uint64_t i = 0; i < n_blk; ++i) {
        std::memcpy(w.data() + static_cast<std::size_t>(i) * blk_bytes,
                    src.data() + static_cast<std::size_t>(i % kSrcBlocks) * blk_bytes,
                    static_cast<std::size_t>(blk_bytes));
    }
    qinfer::experts::MatrixSpec m;
    m.format = fmt;
    m.experts = 1;
    m.cols = cols;
    m.rows = rows;
    std::vector<float> x(static_cast<std::size_t>(cols), 0.5f);
    std::vector<float> host_out(static_cast<std::size_t>(rows), 0.0f);
    std::vector<float> dev_out(static_cast<std::size_t>(rows), 0.0f);
    qinfer::experts::FfnScratch sc;
    std::string err;
    if (!qinfer::experts::quantize_input(m, x.data(), sc, err) ||
        !qinfer::experts::run_gemv(m, w.data(), sc, host_out.data(), err)) {
        std::printf("主机 run_gemv: %s\n", err.c_str());
        return 1;
    }
    if (!qinfer::kernels::gdn_proj_gemv(m, w.data(), sc, dev_out.data(), nullptr, err)) {
        std::printf("设备 gdn_proj_gemv: %s\n", err.c_str());
        return 1;
    }
    return compare_rows(host_out, dev_out, 1e-4, 0.99);
}

// 真几何（部署 GGUF 稠密投影，2026-10-13 gguf-py 实测）：GDN attn_qkv 10240×2560 IQ4_XS
// （10 块/行）；GDN attn_gate 6144×2560 IQ3_S（10 块/行）；QSA attn_k 512×2560 Q6_K（10 块/行）。
int test_dense_proj_formats() {
    int bad = 0;
    bad += gemv_against_host(qinfer::experts::Format::kIq4Xs, kIq4xsRealHex, 2560, 10240, 10);
    bad += gemv_against_host(qinfer::experts::Format::kIq3S, kIq3sRealHex, 2560, 6144, 10);
    bad += gemv_against_host(qinfer::experts::Format::kQ6K, kQ6kRealHex, 2560, 512, 10);
    // 专家诸档（Q2_0/IQ2_S/IQ2_XXS/IQ1_M）由 test_q2_0_cuda / test_iq2s_cuda / test_iq_gemv_cuda
    // 各自钉位；gdn_proj_gemv 对它们的转发由本档的 kIq4Xs 一条验证分派表入口不破即可。
    return bad;
}

// ---- GDN 前段设备对照（Task 3）：合成几何与 test_gdn_preprocess 同款，4 步走卷积状态 ----

constexpr std::uint64_t kPrepN = 5;    // n_embd
constexpr std::uint64_t kPrepQh = 2, kPrepKh = 1, kPrepVh = 3;
constexpr std::uint64_t kPrepHd = 4;   // 头宽
constexpr std::uint64_t kPrepDc = 3;   // 卷积核长（窗口 2）
constexpr float kPrepEps = 1e-6f;
constexpr std::uint64_t kPrepQd = kPrepQh * kPrepHd, kPrepKd = kPrepKh * kPrepHd,
                       kPrepVd = kPrepVh * kPrepHd;
constexpr std::uint64_t kPrepCd = kPrepQd + kPrepKd + kPrepVd;

struct PrepLcg {
    std::uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<float>(static_cast<std::uint32_t>(s >> 32)) / 2147483648.0f) - 1.0f;
    }
};

// 合成夹具：权重（conv1d F32、alpha/beta BF16、dt/ssm_a F32）与几何，与主机测试同款。
struct PrepFixture {
    std::vector<float> convw;
    std::vector<std::uint16_t> alpha_w, beta_w;
    std::vector<float> dt, ssm_a;
    PrepFixture() {
        PrepLcg g;
        convw.resize(kPrepDc * kPrepCd);
        for (float& t : convw) t = g.next();
        alpha_w.resize(kPrepVh * kPrepN);
        beta_w.resize(kPrepVh * kPrepN);
        for (std::size_t i = 0; i < alpha_w.size(); ++i) {
            alpha_w[i] = qinfer::kernels::f32_to_bf16_bits(g.next());
            beta_w[i] = qinfer::kernels::f32_to_bf16_bits(g.next());
        }
        dt.resize(kPrepVh);
        ssm_a.resize(kPrepVh);
        for (std::uint64_t h = 0; h < kPrepVh; ++h) {
            dt[h] = 0.5f + 0.25f * static_cast<float>(h);
            ssm_a[h] = -0.5f - 0.1f * static_cast<float>(h);
        }
    }
    qinfer::dense::GdnShapes shapes() const {
        return qinfer::dense::GdnShapes{kPrepN, kPrepQh, kPrepKh, kPrepVh, kPrepHd, kPrepDc};
    }
    qinfer::dense::GdnPreWeights weights() const {
        using qinfer::dense::GrPrecision;
        using qinfer::dense::GrTensor;
        return qinfer::dense::GdnPreWeights{
            GrTensor{convw.data(), GrPrecision::kF32},
            GrTensor{alpha_w.data(), GrPrecision::kBf16},
            GrTensor{beta_w.data(), GrPrecision::kBf16}, dt.data(), ssm_a.data()};
    }
};

// 设备 vs 主机：同一份 x/qkv（每步同一帧，卷积状态只看步数），4 步连跑，各输出逐元素对照。
int test_gdn_preprocess_cuda() {
    const PrepFixture fx;
    const qinfer::dense::GdnShapes s = fx.shapes();
    std::vector<float> x(kPrepN, 0.5f);
    std::vector<float> qkv(kPrepCd);
    PrepLcg g3;
    for (float& t : qkv) t = g3.next();
    qinfer::dense::GdnConvState host_st, dev_st;
    qinfer::dense::GdnPreOut host_out, dev_out;
    qinfer::dense::GdnPreCudaScratch scratch;
    double worst = 0.0;
    for (int step = 0; step < 4; ++step) {
        std::string err;
        CHECK(qinfer::dense::gdn_preprocess(s, fx.weights(), x.data(), qkv.data(), host_st,
                                            kPrepEps, host_out, err));
        CHECK(err.empty());
        CHECK(qinfer::dense::gdn_preprocess_cuda(s, fx.weights(), x.data(), qkv.data(), dev_st,
                                                 kPrepEps, dev_out, scratch, err));
        if (!err.empty()) std::printf("  [设备] %s\n", err.c_str());
        CHECK(err.empty());
        const std::vector<std::vector<float>*> pairs = {
            &host_out.q, &dev_out.q, &host_out.k, &dev_out.k, &host_out.v, &dev_out.v,
            &host_out.beta, &dev_out.beta, &host_out.gate, &dev_out.gate};
        for (std::size_t i = 0; i < pairs.size(); i += 2) {
            const auto& a = *pairs[i];
            const auto& b = *pairs[i + 1];
            CHECK(a.size() == b.size());
            double d = 0.0, mag = 0.0;
            for (std::size_t j = 0; j < a.size(); ++j) {
                d += std::fabs(static_cast<double>(a[j]) - static_cast<double>(b[j]));
                mag += std::fabs(static_cast<double>(a[j]));
            }
            worst = std::max(worst, d / (mag > 1e-30 ? mag : 1e-30));
        }
    }
    std::printf("  GDN 前段 4 步（卷积状态走过）设备 vs 主机最差相对 L1 %.3e\n", worst);
    return worst < 1e-5 ? 0 : 1;
}

}  // namespace

int main() {
    test_oracle_fixtures_load();
    int rc = test_dense_proj_formats();
    rc += test_gdn_preprocess_cuda();
    if (rc == 0) {
        std::puts("gdn_qsa_cuda: 稠密投影档 + GDN 前段设备/主机对照通过");
    }
    return rc;
}