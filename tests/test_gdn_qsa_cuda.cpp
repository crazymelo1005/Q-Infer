// 阶段 A 设备路径对照载体：GDN/QSA 稠密段各内核的真字节夹具与判据。
// 本文件由 measure/ 的 oracle 工具生成的 .inc 提供真字节（部署 GGUF 的投影权重前若干块），
// 目标判据照 ADR-009 分布级口径：位相同比例 + L1 相对差小（h 量化在舍入边界可差一步）。
// 后续内核（gdn_proj_gemv / gdn_preprocess_cuda / gdn_step_cuda / qsa_*_cuda）逐个在这里对照。
#include "artifact/gguf_table.hpp"
#include "dense/gdn_layer.hpp"
#include "dense/gdn_layer_cuda.hpp"
#include "dense/gdn_preprocess.hpp"
#include "dense/gdn_preprocess_cuda.hpp"
#include "dense/gdn_recurrence.hpp"
#include "dense/gdn_recurrence_cuda.hpp"
#include "dense/qsa_attend.hpp"
#include "dense/qsa_index.hpp"
#include "dense/qsa_index_attend_cuda.hpp"
#include "dense/qsa_layer.hpp"
#include "dense/qsa_layer_cuda.hpp"
#include "dense/qsa_qk_prep.hpp"
#include "dense/qsa_qk_prep_cuda.hpp"
#include "dense/rope.hpp"  // build_rope_table
#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"
#include "kernels/cuda/gdn_proj_gemv.hpp"
#include "check.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// oracle .inc 用 std::uint32_t 等类型，须在 <cstdint> 之后引入。
// 这些是阶段 A 的稠密投影档夹具（gdn_ 前缀，避免与主机同档测试的 .inc 撞名）。
#include "gdn_iq3s_oracle_data.inc"
#include "gdn_iq4xs_oracle_data.inc"
#include "gdn_q6k_oracle_data.inc"
// Q8_0 档（合成夹具的稠密权重正是这个档，dense 设备入口必须覆盖）复用专家侧的
// q8_0_oracle_data.inc（真字节，16 块），不与主机 test_q8_0 冲突（只读符号，两个目标各编一份）。
#include "q8_0_oracle_data.inc"

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

// 一个档的一次对照：oracle 真字节（前 src_blocks 块）循环 tile 成 rows × blocks 矩阵，
// 主机 quantize_input + run_gemv 对设备 gdn_proj_gemv，compare_rows 判据。
int gemv_against_host(qinfer::experts::Format fmt, const std::string& hex, std::uint64_t cols,
                      std::uint64_t rows, std::uint64_t blocks_per_row,
                      std::uint64_t src_blocks = 64) {
    std::printf("== %s: cols %llu rows %llu 块/行 %llu（夹具 %llu 块）==\n",
                qinfer::experts::format_name(fmt), static_cast<unsigned long long>(cols),
                static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(blocks_per_row),
                static_cast<unsigned long long>(src_blocks));
    const std::vector<std::uint8_t> src = from_hex(hex);
    const std::uint64_t kSrcBlocks = src_blocks;
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
    // 激活给非均匀分布（不是全 0.5）：常值激活会让各块的整数码趋同，弱化对照强度——某些错读法
    // （如激活尺度泄漏/码字节偏移）在常值下可能被遮掩，变值分布是为了让它们必然现形。
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.5f + 0.01f * static_cast<float>((i * 7 + 3) % 11) *
                          (static_cast<float>(i % 2) * 2.0f - 1.0f);
    }
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
    // Q8_0 档：合成夹具（serve_fixture_gen / test_facade）的稠密权重全是它，--gpu 时设备路径必须
    // 覆盖；oracle 取 q8_0_oracle_data.inc（ffn_down_shexp 真字节，16 块）。Q8_0 的激活搭档是
    // Q8_0（32 值块，quantize_input 按 activation_format 给 act_32）——与 Q2_0 同款布局。
    bad += gemv_against_host(qinfer::experts::Format::kQ8_0, kQ80RealHex, 2560, 640, 80, 16);
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

// ---- GDN 递推与收尾设备对照（Task 4）----
// 几何：S 8、h_k 2、h_v 4（h_v 是 h_k 的 2 倍，覆盖取模配对），真模型 128/16/48 的形状在
// test_gdn_recurrence 里用人手几何即可；这里用 8/2/4 便于全状态可见。

// 合成输入（每步同一份，决定性强）。
int test_gdn_recurrence_cuda() {
    constexpr std::uint64_t kS = 8, kHk = 2, kHv = 4;
    const qinfer::dense::GdnRecShapes s{kS, kHk, kHv};
    std::vector<float> q(static_cast<std::size_t>(kHk * kS)), k(static_cast<std::size_t>(kHk * kS)),
        v(static_cast<std::size_t>(kHv * kS));
    PrepLcg g;
    for (float& t : q) t = 0.25f + 0.1f * g.next();  // 已归一（合成，不真归一）
    for (float& t : k) t = 0.25f + 0.1f * g.next();
    for (float& t : v) t = g.next();
    std::vector<float> gate(kHv), beta(kHv), beta_sig(kHv);
    for (std::uint64_t h = 0; h < kHv; ++h) {
        gate[h] = -0.3f + 0.1f * static_cast<float>(h);
        beta[h] = g.next();
    }
    qinfer::dense::GdnState host_st, dev_st;
    host_st.resize(s);
    dev_st.resize(s);
    std::vector<float> host_o(static_cast<std::size_t>(kHv * kS)),
        dev_o(static_cast<std::size_t>(kHv * kS));
    double worst = 0.0;
    for (int step = 0; step < 3; ++step) {  // 3 步：状态跨 token 累积
        // 主机：beta 需先 sigmoid（层契约），设备同。
        beta_sig = beta;
        std::string beta_err;
        CHECK(qinfer::dense::gdn_beta_gate(beta_sig.data(), kHv, beta_err));
        CHECK(beta_err.empty());
        std::string err;
        CHECK(qinfer::dense::gdn_step(host_st, s, q.data(), k.data(), v.data(), gate.data(),
                                      beta_sig.data(), host_o.data(), err));
        CHECK(err.empty());
        std::vector<float> dev_beta = beta;
        std::string dev_beta_err;
        CHECK(qinfer::dense::gdn_beta_gate(dev_beta.data(), kHv, dev_beta_err));
        CHECK(qinfer::dense::gdn_step_cuda(dev_st, s, q.data(), k.data(), v.data(), gate.data(),
                                           dev_beta.data(), dev_o.data(), nullptr, err));
        CHECK(!dev_beta.empty());
        CHECK(err.empty());
        // o 与状态逐元素对照。
        for (std::size_t i = 0; i < host_o.size(); ++i) {
            const double d = std::fabs(static_cast<double>(host_o[i]) - static_cast<double>(dev_o[i]));
            worst = std::max(worst, d / (std::fabs(host_o[i]) > 1e-30 ? std::fabs(host_o[i]) : 1e-30));
        }
        for (std::size_t i = 0; i < host_st.data.size(); ++i) {
            const double d = std::fabs(static_cast<double>(host_st.data[i]) -
                                       static_cast<double>(dev_st.data[i]));
            worst = std::max(worst,
                             d / (std::fabs(host_st.data[i]) > 1e-30 ? std::fabs(host_st.data[i])
                                                                     : 1e-30));
        }
    }
    std::printf("  GDN 递推 3 步（状态跨 token 累积）o+state 最差相对 L1 %.3e\n", worst);
    return worst < 1e-5 ? 0 : 1;
}

// 收尾对照：设备 gdn_closing_norm_cuda vs 主机 gdn_closing_norm。
int test_gdn_closing_cuda() {
    constexpr std::uint64_t kS = 8, kHk = 2, kHv = 4;
    const qinfer::dense::GdnRecShapes s{kS, kHk, kHv};
    std::vector<float> o(static_cast<std::size_t>(kHv * kS)), z(static_cast<std::size_t>(kHv * kS)),
        norm(kS), host_y(static_cast<std::size_t>(kHv * kS)),
        dev_y(static_cast<std::size_t>(kHv * kS));
    PrepLcg g;
    for (float& t : o) t = g.next();
    for (float& t : z) t = g.next();
    for (float& t : norm) t = 0.5f + 0.1f * g.next();
    std::string err;
    CHECK(qinfer::dense::gdn_closing_norm(s, o.data(), z.data(), norm.data(), 1e-6f,
                                          host_y.data(), err));
    CHECK(err.empty());
    CHECK(qinfer::dense::gdn_closing_norm_cuda(s, o.data(), z.data(), norm.data(), 1e-6f,
                                               dev_y.data(), nullptr, err));
    CHECK(err.empty());
    double worst = 0.0;
    for (std::size_t i = 0; i < host_y.size(); ++i) {
        const double d = std::fabs(static_cast<double>(host_y[i]) - static_cast<double>(dev_y[i]));
        worst = std::max(worst, d / (std::fabs(host_y[i]) > 1e-30 ? std::fabs(host_y[i]) : 1e-30));
    }
    std::printf("  GDN 收尾设备 vs 主机最差相对 L1 %.3e\n", worst);
    return worst < 1e-5 ? 0 : 1;
}

// 真模型整层对照（Task 5）：环境2 上 `--model <部署 GGUF> [--layer N]` 时，主机 run_gdn_layer
// 对设备 run_gdn_layer_cuda 同输入对照；2 步连跑（状态跨 token）。判据 ADR-009 分布级。
int manual_gdn_layer_entry(const char* model, int layer) {
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    qinfer::dense::GdnLayer gl;
    if (!qinfer::dense::build_gdn_layer(g, layer, 2560, 128, 4, gl, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::printf("层 %d：conv_dim %llu、value_dim %llu、q/k 各 %llu 头、v %llu 头、头宽 %llu；档 ",
                layer, static_cast<unsigned long long>(gl.qkv.rows),
                static_cast<unsigned long long>(gl.gate.rows),
                static_cast<unsigned long long>(gl.pre.n_q_head),
                static_cast<unsigned long long>(gl.pre.n_v_head),
                static_cast<unsigned long long>(gl.pre.head_dim));
    std::printf("%s / %s / %s\n", qinfer::experts::format_name(gl.qkv.format),
                qinfer::experts::format_name(gl.gate.format),
                qinfer::experts::format_name(gl.out.format));
    std::vector<float> x(static_cast<std::size_t>(gl.pre.n_embd), 0.5f);
    double worst = 0.0;
    qinfer::dense::GdnScratch host_s, dev_s;
    std::vector<float> host_out(gl.pre.n_embd), dev_out(gl.pre.n_embd);
    for (int step = 0; step < 2; ++step) {  // 2 步：状态跨 token
        if (!qinfer::dense::run_gdn_layer(g, gl, x.data(), host_out.data(), host_s, err)) {
            std::printf("主机层 %d 跑不动（step %d）：%s\n", layer, step, err.c_str());
            return 1;
        }
        if (!qinfer::dense::run_gdn_layer_cuda(g, gl, x.data(), dev_out.data(), dev_s, err)) {
            std::printf("设备层 %d 跑不动（step %d）：%s\n", layer, step, err.c_str());
            return 1;
        }
        double d = 0.0, mag = 0.0;
        for (std::size_t i = 0; i < host_out.size(); ++i) {
            d += std::fabs(static_cast<double>(host_out[i]) - static_cast<double>(dev_out[i]));
            mag += std::fabs(static_cast<double>(host_out[i]));
        }
        worst = std::max(worst, d / (mag > 1e-30 ? mag : 1e-30));
    }
    std::printf("  整层 2 步（状态跨 token）主机 vs 设备最差相对 L1 %.3e\n", worst);
    return worst < 1e-4 ? 0 : 1;
}

// 真模型 QSA 整层对照（Task 8）：`--model <GGUF> --layer N --qsa` 时，主机 run_qsa_layer 对设备
// run_qsa_layer_cuda 同输入对照；2 步连跑（KV/idx 状态跨 token）。判据 ADR-009 分布级。
int manual_qsa_layer_entry(const char* model, int layer) {
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    qinfer::dense::QsaLayer ql;
    // 真模型几何（S-54/S-57）：head_dim 256、n_rot 64、idx_dim 128、idx_top_k 2048、idx_block 4。
    if (!qinfer::dense::build_qsa_layer(g, layer, 2560, 256, 64, 128, 2048, 4, ql, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::printf("QSA 层 %d：n_head %llu、n_head_kv %llu、head_dim %llu、idx_dim %llu、idx_n_head %llu；档 ",
                layer, static_cast<unsigned long long>(ql.sh.n_head),
                static_cast<unsigned long long>(ql.sh.n_head_kv),
                static_cast<unsigned long long>(ql.sh.head_dim),
                static_cast<unsigned long long>(ql.idx_dim),
                static_cast<unsigned long long>(ql.idx_n_head));
    std::printf("%s / %s / %s / %s\n", qinfer::experts::format_name(ql.q.format),
                qinfer::experts::format_name(ql.k.format),
                qinfer::experts::format_name(ql.v.format),
                qinfer::experts::format_name(ql.out.format));
    std::vector<float> x(static_cast<std::size_t>(ql.q.cols), 0.5f);
    double worst = 0.0;
    // 主机与设备的 scratch 各自配 KV / rope 表（与 prepare_step_scratch 同口径：page_size 4、
    // 恒等页表、pos_base 0、rope 表 theta 1e7）。
    auto setup_scratch = [&](qinfer::dense::QsaScratch& s) {
        s.pos_base = 0;
        s.kv.page_size = 4;
        s.kv.n_head_kv = ql.sh.n_head_kv;
        s.kv.head_dim = ql.sh.head_dim;
        s.kv.n_pages = 1024;
        s.kv.reset();
        s.kv.table.resize(1024);
        for (std::uint64_t p = 0; p < 1024; ++p) s.kv.table[p] = static_cast<std::int32_t>(p);
        return qinfer::dense::build_rope_table(ql.n_rot, 1.0e7, 16384, s.cos_tab, s.sin_tab, err);
    };
    qinfer::dense::QsaScratch host_s, dev_s;
    if (!setup_scratch(host_s) || !setup_scratch(dev_s)) {
        std::printf("rope 表建失败：%s\n", err.c_str());
        return 1;
    }
    // Stage B (device KV): the device scratch also gets the exec pointer, so
    // append/gather/attend go through the device pool; host path unchanged.
    qinfer::dense::KvDeviceExec kvex = qinfer::dense::kv_device_exec();
    dev_s.dev_kv = &kvex;
    if (dev_s.dev_kv->reserve == nullptr) {
        std::printf("kv_device_exec 没填（CUDA 构建应填）\n");
        return 1;
    }
    std::vector<float> host_out(ql.q.cols), dev_out(ql.q.cols);
    for (int step = 0; step < 2; ++step) {  // 2 步：KV 与 idx 状态跨 token
        if (!qinfer::dense::run_qsa_layer(g, ql, x.data(), host_out.data(), host_s, err)) {
            std::printf("主机 QSA 层 %d 跑不动（step %d）：%s\n", layer, step, err.c_str());
            return 1;
        }
        if (!qinfer::dense::run_qsa_layer_cuda(g, ql, x.data(), dev_out.data(), dev_s, err)) {
            std::printf("设备 QSA 层 %d 跑不动（step %d）：%s\n", layer, step, err.c_str());
            return 1;
        }
        double d = 0.0, mag = 0.0;
        for (std::size_t i = 0; i < host_out.size(); ++i) {
            d += std::fabs(static_cast<double>(host_out[i]) - static_cast<double>(dev_out[i]));
            mag += std::fabs(static_cast<double>(host_out[i]));
        }
        worst = std::max(worst, d / (mag > 1e-30 ? mag : 1e-30));
    }
    std::printf("  QSA 整层 2 步（KV/idx 状态跨 token）主机 vs 设备最差相对 L1 %.3e\n", worst);
    // Manual entry owns the scratch directly (no StepScratch destructor): free
    // the device pool here.
    if (dev_s.dev_pool != nullptr) {
        std::string rel_err;
        kvex.release(dev_s.dev_pool, rel_err);
    }
    return worst < 1e-4 ? 0 : 1;
}

// ---- QSA 投影与 qk_prep 设备对照（Task 6）----
// 合成几何：n_head 3、n_head_kv 2、head_dim 8、n_rot 4（部分 rope、非整头）、pos 可变。
int test_qsa_qk_prep_cuda() {
    constexpr std::uint64_t kNh = 3, kNhk = 2, kHd = 8, kNrot = 4;
    std::vector<float> cos_tab, sin_tab;
    std::string err;
    if (!qinfer::dense::build_rope_table(kNrot, 1.0e7, 16, cos_tab, sin_tab, err)) {
        std::printf("  rope 表建失败：%s\n", err.c_str());
        return 1;
    }
    std::vector<float> q_norm(kHd), k_norm(kHd);
    PrepLcg g;
    for (float& t : q_norm) t = 0.5f + 0.2f * g.next();
    for (float& t : k_norm) t = 0.5f + 0.2f * g.next();
    qinfer::dense::QsaQkWeights w{q_norm.data(), k_norm.data(), 1e-6f};
    std::vector<float> q_full(static_cast<std::size_t>(kNh * 2 * kHd)),
        k_raw(static_cast<std::size_t>(kNhk * kHd));
    for (float& t : q_full) t = g.next();
    for (float& t : k_raw) t = g.next();
    std::vector<float> hq(static_cast<std::size_t>(kNh * kHd)),
        hk(static_cast<std::size_t>(kNhk * kHd)), hg(static_cast<std::size_t>(kNh * kHd));
    std::vector<float> dq(hq.size()), dk(hk.size()), dg(hg.size());
    double worst = 0.0;
    for (std::uint64_t pos = 0; pos < 4; ++pos) {  // 4 个位置走 rope 表
        CHECK(qinfer::dense::qsa_qk_prep(kNh, kNhk, kHd, kNrot, pos, q_full.data(), k_raw.data(),
                                         w, cos_tab, sin_tab, hq.data(), hk.data(), hg.data(),
                                         err));
        CHECK(err.empty());
        qinfer::dense::QsaQkPrepCudaScratch sc{nullptr, nullptr, 0};
        CHECK(qinfer::dense::qsa_qk_prep_cuda(kNh, kNhk, kHd, kNrot, pos, q_full.data(),
                                              k_raw.data(), w, cos_tab, sin_tab, dq.data(),
                                              dk.data(), dg.data(), sc, nullptr, err));
        CHECK(err.empty());
        const std::vector<std::vector<float>*> pairs = {&hq, &dq, &hk, &dk, &hg, &dg};
        for (std::size_t i = 0; i < pairs.size(); i += 2) {
            const auto& a = *pairs[i];
            const auto& b = *pairs[i + 1];
            double d = 0.0, mag = 0.0;
            for (std::size_t j = 0; j < a.size(); ++j) {
                d += std::fabs(static_cast<double>(a[j]) - static_cast<double>(b[j]));
                mag += std::fabs(static_cast<double>(a[j]));
            }
            worst = std::max(worst, d / (mag > 1e-30 ? mag : 1e-30));
        }
    }
    std::printf("  QSA qk_prep 4 位置（拆头/归一/部分 NEOX rope）设备 vs 主机最差相对 L1 %.3e\n",
                worst);
    return worst < 1e-5 ? 0 : 1;
}

// ---- QSA 索引器与 attend 设备对照（Task 7）----
// 合成几何：idx_dim 4、idx_n_head 2、idx_block 2、n_bid 3（4 个块）、n_kv 9（含尾巴一段）。
int test_qsa_index_cuda() {
    constexpr std::uint64_t kIdxDim = 4, kIdxNh = 2, kIdxBlock = 2, kNBid = 3, kNKv = 9;
    std::vector<float> pooled(static_cast<std::size_t>((kNBid + 1) * kIdxDim)),
        q_idx(static_cast<std::size_t>(kIdxNh * kIdxDim));
    PrepLcg g;
    for (float& t : pooled) t = g.next();
    for (float& t : q_idx) t = g.next();
    std::vector<float> bias(static_cast<std::size_t>(kNBid + 1));
    for (float& t : bias) t = 0.1f * g.next();
    std::vector<float> host(static_cast<std::size_t>(kNKv)), dev(static_cast<std::size_t>(kNKv));
    // bias 变体 + 无 bias 变体（尾巴 1e9 必须无条件加）。
    double worst = 0.0;
    for (int with_bias = 0; with_bias < 2; ++with_bias) {
        std::string err;
        CHECK(qinfer::dense::qsa_index(pooled.data(), kNBid, kIdxDim, kIdxNh, q_idx.data(),
                                       with_bias ? bias.data() : nullptr, kIdxBlock, kNKv,
                                       host.data(), err));
        CHECK(err.empty());
        CHECK(qinfer::dense::qsa_index_cuda(pooled.data(), kNBid, kIdxDim, kIdxNh, q_idx.data(),
                                            with_bias ? bias.data() : nullptr, kIdxBlock, kNKv,
                                            dev.data(), nullptr, err));
        CHECK(err.empty());
        for (std::size_t i = 0; i < host.size(); ++i) {
            const double d = std::fabs(static_cast<double>(host[i]) - static_cast<double>(dev[i]));
            worst = std::max(worst,
                             d / (std::fabs(host[i]) > 1e-30 ? std::fabs(host[i]) : 1e-30));
        }
    }
    // 尾巴必须加 1e9（差异会被 1e9 放大，验证非零）。
    std::printf("  QSA 索引器（bias ±、尾巴 1e9）设备 vs 主机最差相对 L1 %.3e\n", worst);
    return worst < 1e-5 ? 0 : 1;
}

// attend + gate 对照：QsaShapes{idx_top_k, idx_block, head_dim, n_head, n_head_kv}。
int test_qsa_attend_cuda() {
    constexpr std::uint64_t kHd = 8, kNh = 4, kNkv = 2, kNIds = 6;
    const qinfer::dense::QsaShapes s{2048, 4, kHd, kNh, kNkv};
    std::vector<float> q(static_cast<std::size_t>(kNh * kHd)), q_full(static_cast<std::size_t>(kNh * 2 * kHd));
    PrepLcg g;
    for (float& t : q) t = g.next();
    for (float& t : q_full) t = g.next();
    std::vector<std::uint16_t> k(static_cast<std::size_t>(kNIds * kNkv * kHd)),
        v(static_cast<std::size_t>(kNIds * kNkv * kHd));
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = qinfer::kernels::f32_to_bf16_bits(0.5f * g.next());  // fp16 位型（用 bf16 舍入近似无妨）
        v[i] = qinfer::kernels::f32_to_bf16_bits(0.5f * g.next());
    }
    std::vector<float> attn_host(static_cast<std::size_t>(kNh * kHd)),
        attn_dev(static_cast<std::size_t>(kNh * kHd));
    std::vector<float> weights_host(static_cast<std::size_t>(kNh * kNIds)),
        weights_dev(static_cast<std::size_t>(kNh * kNIds));
    std::string err;
    // 主机：attend 未过门 → gate_apply 过门（与设备 qsa_attend_cuda 语义一致：attend_cuda 内含 gate）。
    CHECK(qinfer::dense::qsa_attend(q.data(), k.data(), v.data(), kNIds, s, attn_host.data(),
                                    weights_host.data(), err));
    CHECK(err.empty());
    CHECK(qinfer::dense::qsa_gate_apply_f32(attn_host.data(), q_full.data(), s, attn_host.data(),
                                            err));
    CHECK(err.empty());
    // 设备：attend + gate 一步。
    CHECK(qinfer::dense::qsa_attend_cuda(q.data(), k.data(), v.data(), kNIds, s, q_full.data(),
                                         attn_dev.data(), weights_dev.data(), nullptr, err));
    CHECK(err.empty());
    double worst_a = 0.0, worst_w = 0.0;
    for (std::size_t i = 0; i < attn_host.size(); ++i) {
        const double d = std::fabs(static_cast<double>(attn_host[i]) - static_cast<double>(attn_dev[i]));
        worst_a = std::max(worst_a, d / (std::fabs(attn_host[i]) > 1e-30 ? std::fabs(attn_host[i]) : 1e-30));
    }
    for (std::size_t i = 0; i < weights_host.size(); ++i) {
        const double d = std::fabs(static_cast<double>(weights_host[i]) - static_cast<double>(weights_dev[i]));
        worst_w = std::max(worst_w, d / (std::fabs(weights_host[i]) > 1e-30 ? std::fabs(weights_host[i]) : 1e-30));
    }
    std::printf("  QSA attend+gate 设备 vs 主机：attn %.3e、weights %.3e\n", worst_a, worst_w);
    return worst_a < 1e-4 && worst_w < 1e-4 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    test_oracle_fixtures_load();
    int rc = test_dense_proj_formats();
    rc += test_gdn_preprocess_cuda();
    rc += test_gdn_recurrence_cuda();
    rc += test_gdn_closing_cuda();
    rc += test_qsa_qk_prep_cuda();
    rc += test_qsa_index_cuda();
    rc += test_qsa_attend_cuda();
    // --model <GGUF> [--layer N]：真权重整层对照（环境2 用；无参数时合成用例已覆盖各段）。
    int layer = 1;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--model") {
        }
    }
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        bool qsa = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--layer" && i + 1 < argc) {
                layer = std::atoi(argv[++i]);
            } else if (std::string(argv[i]) == "--qsa") {
                qsa = true;
            }
        }
        rc += qsa ? manual_qsa_layer_entry(argv[2], layer)
                  : manual_gdn_layer_entry(argv[2], layer);
    }
    if (rc == 0) {
        std::puts("gdn_qsa_cuda: 稠密投影 + GDN 前段/递推/收尾/整层设备对照全部通过");
    }
    return rc;
}