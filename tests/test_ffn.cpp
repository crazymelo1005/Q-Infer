// 专家 FFN 三矩阵组装的回归。不引第三方框架。
//
// 五路证据：
//   1. 解析可算的一组：权重全 1、输入全 1 时，gate/up 的每个输出必须恰好等于 2560
//      （10 个 IQ2_S 块逐个精确：权重 1.0、激活量化后也恰好是 1.0）。这一步把「激活量化 + 逐块
//      定点点积 + 10 块串联」整条链钉成精确值。
//   2. 与双精度参考实现交叉核对：同一套（变化的）量化权重与输入，参考实现按同样规则量化激活、
//      逐元素反量化后点乘，两边差值须在相对容差内。这条覆盖 SwiGLU、down 的输入是 h、
//      以及同一组激活被 gate 与 up 共用这三件事。
//   3. 中间量可见：组装过程把 gate/up/h 留在 scratch 里，可直接核对（silu 的值、h 的长度）。
//   4. 不可用的档必须失败：没有内核的档、几何不成立的档，都要返回 false 并说明原因，不能猜。
//   5. moe_ffn 与 expert_ffn 自洽：两个专家的加权和等于各自结果按权重的线性组合。
//
// 真模型的手工入口：`./build/test_ffn --model <分片1.gguf> --layer L --expert E`，
// 从文件里按逐专家偏移取出三个矩阵，跑一次前馈并与双精度参考核对。
#include "experts/ffn.hpp"

#include "artifact/gguf_table.hpp"
#include "check.hpp"
#include "kernels/fp16.hpp"
#include "kernels/iq2s.hpp"
#include "kernels/iq4nl.hpp"
#include "kernels/q2_0.hpp"
#include "kernels/q8k.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kHidden = 2560;
constexpr std::uint64_t kFfn = 640;

struct Matrix {
    std::vector<std::uint8_t> bytes;
};

std::uint64_t blocks_of(experts::Format f, std::uint64_t cols) {
    return cols / static_cast<std::uint64_t>(experts::block_elems(f));
}

// 每个块都一样：d 给定、其余字节给定。
Matrix make_matrix(experts::Format f, std::uint64_t rows, std::uint64_t cols, std::uint16_t d_bits,
                   const std::vector<std::uint8_t>& body) {
    const int bb = experts::block_bytes(f);
    const std::uint64_t nb = blocks_of(f, cols);
    Matrix m;
    m.bytes.resize(static_cast<std::size_t>(rows * nb * static_cast<std::uint64_t>(bb)));
    std::size_t at = 0;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t b = 0; b < nb; ++b) {
            m.bytes[at++] = static_cast<std::uint8_t>(d_bits & 0xFF);
            m.bytes[at++] = static_cast<std::uint8_t>(d_bits >> 8);
            for (std::size_t k = 0; k < body.size(); ++k) m.bytes[at++] = body[k];
        }
    }
    return m;
}

// 每块字节都不同（供交叉核对用）：用 (salt, r, b, k) 的线性组合，避免出现对称到测不出错的常量。
// salt 必须让 gate 与 up 不同——否则 swiglu 两个操作数对调这类错误测不出来（这是第一版的教训）。
Matrix make_matrix_varied(experts::Format f, std::uint64_t rows, std::uint64_t cols,
                          std::uint32_t salt) {
    const int bb = experts::block_bytes(f);
    const std::uint64_t nb = blocks_of(f, cols);
    Matrix m;
    m.bytes.resize(static_cast<std::size_t>(rows * nb * static_cast<std::uint64_t>(bb)));
    std::size_t at = 0;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t b = 0; b < nb; ++b) {
            const std::uint32_t seed = salt * 1000003u + static_cast<std::uint32_t>(r * 131 + b * 17 + 7);
            // 尺度用正值的小 fp16（0x3800 = 0.5 附近），其余字节按 LCG 变化。
            m.bytes[at++] = static_cast<std::uint8_t>(0x00);
            m.bytes[at++] = static_cast<std::uint8_t>(0x38);
            std::uint32_t s = seed;
            for (int k = 2; k < bb; ++k) {
                s = s * 1664525u + 1013904223u;
                m.bytes[at++] = static_cast<std::uint8_t>(s >> 24);
            }
        }
    }
    return m;
}

experts::LayerSpec make_spec(experts::Format gate, experts::Format up, experts::Format down) {
    experts::LayerSpec s;
    s.layer = 0;
    s.gate.format = gate;
    s.gate.cols = kHidden;
    s.gate.rows = kFfn;
    s.gate.experts = 1;
    s.up = s.gate;
    s.up.format = up;
    s.down.format = down;
    s.down.cols = kFfn;
    s.down.rows = kHidden;
    s.down.experts = 1;
    return s;
}

// 按格式反量化一行（长度 cols）。
void dequant_row(experts::Format f, const std::uint8_t* row, std::uint64_t cols, float* out) {
    const int bb = experts::block_bytes(f);
    const std::uint64_t elems = static_cast<std::uint64_t>(experts::block_elems(f));
    const std::uint64_t nb = blocks_of(f, cols);
    float tmp[256];
    for (std::uint64_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = row + b * static_cast<std::uint64_t>(bb);
        switch (f) {
            case experts::Format::kIq2S:
                dequant_iq2s_block(blk, tmp);
                break;
            case experts::Format::kQ2_0:
                dequant_q2_0_block(blk, tmp);
                break;
            case experts::Format::kIq4Nl:
                dequant_iq4nl_block(blk, tmp);
                break;
            default:
                CHECK(false && "参考实现只覆盖已实现的档");
        }
        const std::uint64_t n = cols - b * elems;
        const std::uint64_t take = n < elems ? n : elems;
        for (std::uint64_t j = 0; j < take; ++j) out[b * elems + j] = tmp[j];
    }
}

// 双精度参考实现：激活按同规则量化后逐元素反量化，再与反量化的权重点乘。
void ref_gemv(const experts::MatrixSpec& m, const Matrix& w, const float* x_quantized_dequant,
              std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(m.rows), 0.0);
    std::vector<float> row(static_cast<std::size_t>(m.cols));
    const std::uint64_t row_bytes = m.row_bytes();
    for (std::uint64_t r = 0; r < m.rows; ++r) {
        dequant_row(m.format, w.bytes.data() + r * row_bytes, m.cols, row.data());
        double acc = 0.0;
        for (std::uint64_t j = 0; j < m.cols; ++j) {
            acc += static_cast<double>(row[j]) * static_cast<double>(x_quantized_dequant[j]);
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
}

// 把输入按档量化再反量化（参考实现与主实现看到的是同一组激活）。
std::vector<float> quantize_dequant(const experts::Format f, const float* x, std::uint64_t cols) {
    std::vector<float> out(static_cast<std::size_t>(cols));
    const experts::ActFormat act = experts::activation_format(f);
    if (act == experts::ActFormat::kQ8K) {
        const int nb = static_cast<int>(cols / 256);
        std::vector<Q8kBlock> b(static_cast<std::size_t>(nb));
        q8k_quantize_row(x, b.data(), nb);
        float tmp[256];
        for (int i = 0; i < nb; ++i) {
            q8k_dequant_block(b[static_cast<std::size_t>(i)], tmp);
            for (int j = 0; j < 256; ++j) out[static_cast<std::size_t>(i) * 256 + j] = tmp[j];
        }
    } else {
        const int nb = static_cast<int>(cols / kQ80BlockElems);
        std::vector<Q80Block> b(static_cast<std::size_t>(nb));
        q8_0_quantize_row(x, b.data(), nb);
        float tmp[kQ80BlockElems];
        for (int i = 0; i < nb; ++i) {
            q8_0_dequant_block(b[static_cast<std::size_t>(i)], tmp);
            for (int j = 0; j < kQ80BlockElems; ++j) {
                out[static_cast<std::size_t>(i) * kQ80BlockElems + j] = tmp[j];
            }
        }
    }
    return out;
}

void ref_ffn(const experts::LayerSpec& spec, const Matrix& g, const Matrix& u, const Matrix& d,
             const float* x, std::vector<double>& out) {
    const std::vector<float> xq = quantize_dequant(spec.gate.format, x, kHidden);
    std::vector<double> gate, up;
    ref_gemv(spec.gate, g, xq.data(), gate);
    ref_gemv(spec.up, u, xq.data(), up);
    std::vector<float> h(static_cast<std::size_t>(kFfn));
    for (std::size_t j = 0; j < kFfn; ++j) {
        const double gv = gate[j];
        h[j] = static_cast<float>((gv / (1.0 + std::exp(-gv))) * up[j]);
    }
    const std::vector<float> hq = quantize_dequant(spec.down.format, h.data(), kFfn);
    ref_gemv(spec.down, d, hq.data(), out);
}

int compare(const std::vector<double>& got, const std::vector<double>& want, const char* what,
            double tol) {
    double mag = 0.0;
    for (double v : want) mag += std::fabs(v);
    if (mag == 0.0) mag = 1.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double rel = std::fabs(got[i] - want[i]) / mag;
        if (!(rel < tol)) {
            std::printf("FAIL %s[%zu]: got %.9g want %.9g rel %.3g\n", what, i, got[i], want[i], rel);
            return 1;
        }
    }
    return 0;
}

void test_analytic_uniform() {
    // IQ2_S 权重全 1：d = 1.0、格点索引 0（全 8）、符号 0、尺度 0 -> 值 = 0.125 · 8 = 1.0。
    const std::vector<std::uint8_t> iq2s_body(80, 0);  // qs(64) + qh(8) + scales(8)，全 0
    const Matrix g = make_matrix(experts::Format::kIq2S, kFfn, kHidden, 0x3C00, iq2s_body);
    const Matrix u = g;
    // Q2_0 权重全 1：d = 1.0、每个码字节 0xAA -> 4 个码都是 2 -> 值 = (2-1) · 1.0 = 1.0。
    const Matrix d = make_matrix(experts::Format::kQ2_0, kHidden, kFfn, 0x3C00,
                                 std::vector<std::uint8_t>(16, 0xAA));
    const experts::LayerSpec spec =
        make_spec(experts::Format::kIq2S, experts::Format::kIq2S, experts::Format::kQ2_0);

    std::vector<float> x(static_cast<std::size_t>(kHidden), 1.0f);
    experts::ExpertWeights w{g.bytes.data(), u.bytes.data(), d.bytes.data()};
    experts::FfnScratch scratch;
    std::vector<float> out(static_cast<std::size_t>(kHidden), 0.0f);
    std::string err;
    CHECK(experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));

    // gate/up：权重 1.0、激活量化后恰好 1.0，10 个块 -> 每个输出恰好 2560。
    for (std::size_t j = 0; j < kFfn; ++j) {
        if (scratch.gate[j] != 2560.0f || scratch.up[j] != 2560.0f) {
            std::printf("FAIL analytic gate/up[%zu] = %g / %g\n", j,
                        static_cast<double>(scratch.gate[j]), static_cast<double>(scratch.up[j]));
            CHECK(false);
        }
    }
    // swiglu：silu(2560) 在 float 下就是 1.0，故 h = 2560 · 2560。
    for (std::size_t j = 0; j < kFfn; ++j) {
        if (scratch.h[j] != 6553600.0f) {
            std::printf("FAIL analytic h[%zu] = %g\n", j, static_cast<double>(scratch.h[j]));
            CHECK(false);
        }
    }
    // down：h 被量化成 Q8_0，故输出等于「量化后 h 之和」（每行权重都是 1）；每行都一样。
    const std::vector<float> hq = quantize_dequant(experts::Format::kQ2_0, scratch.h.data(), kFfn);
    double sum_hq = 0.0;
    for (float v : hq) sum_hq += static_cast<double>(v);
    for (std::size_t i = 0; i < kHidden; ++i) {
        const double rel = std::fabs(static_cast<double>(out[i]) - sum_hq) /
                           (std::fabs(sum_hq) > 0 ? std::fabs(sum_hq) : 1.0);
        if (!(rel < 1e-6)) {
            std::printf("FAIL analytic out[%zu] = %.9g want %.9g\n", i,
                        static_cast<double>(out[i]), sum_hq);
            CHECK(false);
        }
    }
}

void test_matches_reference() {
    const Matrix g = make_matrix_varied(experts::Format::kIq2S, kFfn, kHidden, 1);
    const Matrix u = make_matrix_varied(experts::Format::kIq2S, kFfn, kHidden, 2);
    const Matrix d = make_matrix_varied(experts::Format::kQ2_0, kHidden, kFfn, 3);
    const experts::LayerSpec spec =
        make_spec(experts::Format::kIq2S, experts::Format::kIq2S, experts::Format::kQ2_0);

    std::vector<float> x(static_cast<std::size_t>(kHidden));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 37) - 18) * 0.03125);
    }

    experts::ExpertWeights w{g.bytes.data(), u.bytes.data(), d.bytes.data()};
    experts::FfnScratch scratch;
    std::vector<float> out(static_cast<std::size_t>(kHidden), 0.0f);
    std::string err;
    CHECK(experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));

    std::vector<double> ref;
    ref_ffn(spec, g, u, d, x.data(), ref);
    std::vector<double> got(out.begin(), out.end());
    CHECK(compare(got, ref, "ffn vs 参考", 1e-5) == 0);

    // moe_ffn 与 expert_ffn 自洽：把同一个专家按权重 1 累加进零缓冲，应得到同一结果。
    std::vector<float> acc(static_cast<std::size_t>(kHidden), 0.0f);
    const float wt = 1.0f;
    CHECK(experts::moe_ffn(spec, &w, &wt, 1, x.data(), acc.data(), scratch, err));
    for (std::size_t i = 0; i < acc.size(); ++i) CHECK(acc[i] == out[i]);

    // 两个专家各自的加权和。
    Matrix g2 = make_matrix_varied(experts::Format::kIq2S, kFfn, kHidden, 4);
    // 让第二个专家的权重与前一个不同：把尺度字节整体 +1。
    for (std::size_t k = 0; k + 1 < g2.bytes.size(); k += 18) g2.bytes[k + 1] = 0x39;
    experts::ExpertWeights both[2] = {w, {g2.bytes.data(), u.bytes.data(), d.bytes.data()}};
    const float wts[2] = {0.25f, 0.75f};
    std::vector<float> mix(static_cast<std::size_t>(kHidden), 0.0f);
    CHECK(experts::moe_ffn(spec, both, wts, 2, x.data(), mix.data(), scratch, err));
    std::vector<float> only2(static_cast<std::size_t>(kHidden), 0.0f);
    CHECK(experts::expert_ffn(spec, both[1], x.data(), scratch, only2.data(), err));
    for (std::size_t i = 0; i < mix.size(); ++i) {
        const float want = 0.25f * out[i] + 0.75f * only2[i];
        const float got2 = mix[i];
        if (std::fabs(got2 - want) > 1e-2f * (std::fabs(want) + 1.0f)) {
            std::printf("FAIL moe mix[%zu]: got %g want %g\n", i, static_cast<double>(got2),
                        static_cast<double>(want));
            CHECK(false);
        }
    }
}

void test_unusable_specs_fail() {
    experts::FfnScratch scratch;
    std::string err;
    std::vector<float> x(static_cast<std::size_t>(kHidden), 0.1f);
    std::vector<float> out(static_cast<std::size_t>(kHidden), 0.0f);

    // gate 用没有内核的档（Q6_K）：必须在组装前就失败。
    {
        const experts::LayerSpec spec =
            make_spec(experts::Format::kQ6K, experts::Format::kQ6K, experts::Format::kQ2_0);
        std::vector<std::uint8_t> g(256, 0);
        experts::ExpertWeights w{g.data(), g.data(), g.data()};
        err.clear();
        CHECK(!experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));
        CHECK(!err.empty());
    }
    // down 用 256 值块的档（几何不成立：640 不能被 256 整除）。
    {
        const experts::LayerSpec spec =
            make_spec(experts::Format::kIq2S, experts::Format::kIq2S, experts::Format::kIq2S);
        std::vector<std::uint8_t> g(256, 0);
        experts::ExpertWeights w{g.data(), g.data(), g.data()};
        err.clear();
        CHECK(!experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));
        CHECK(!err.empty());
    }
    // gate 与 up 共用同一组激活，故激活档必须一致；不一致要失败而不是各算各的。
    // up 用 Q2_0（激活档 Q8_0）而 gate 用 IQ2_S（激活档 Q8_K）——几何上都成立，所以只有这条检查拦得住。
    {
        const experts::LayerSpec spec =
            make_spec(experts::Format::kIq2S, experts::Format::kQ2_0, experts::Format::kQ2_0);
        CHECK(spec.usable());  // 几何与内核都没问题，问题只在激活档不一致
        std::vector<std::uint8_t> g(256, 0);
        experts::ExpertWeights w{g.data(), g.data(), g.data()};
        err.clear();
        CHECK(!experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));
        CHECK(!err.empty());
    }
    // 缺矩阵指针。
    {
        const experts::LayerSpec spec =
            make_spec(experts::Format::kIq2S, experts::Format::kIq2S, experts::Format::kQ2_0);
        std::vector<std::uint8_t> g(256, 0);
        experts::ExpertWeights w{g.data(), nullptr, g.data()};
        err.clear();
        CHECK(!experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err));
        CHECK(!err.empty());
    }
}

// ---- 真模型手工入口 ----

int manual_model_entry(const std::string& path, int layer, int expert) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(path, err)) {
        std::printf("打不开：%s\n", err.c_str());
        return 1;
    }
    const experts::Table table = experts::build_table(g, layer, layer);
    if (table.layers.empty()) {
        std::printf("没有这一层\n");
        return 1;
    }
    const experts::LayerSpec& spec = table.layers[0];
    if (!spec.usable()) {
        std::printf("层 %d 不可用：", layer);
        for (const auto& p : table.problems) std::printf("%s（%s） ", p.tensor.c_str(), p.reason.c_str());
        std::printf("\n");
        return 1;
    }
    // 逐专家偏移：三维张量 [cols, rows, experts]，专家步长 = rows × row_bytes。
    std::vector<std::uint8_t> bg(spec.gate.rows * spec.gate.row_bytes());
    std::vector<std::uint8_t> bu(spec.up.rows * spec.up.row_bytes());
    std::vector<std::uint8_t> bd(spec.down.rows * spec.down.row_bytes());
    auto load = [&](const std::string& suffix, const experts::MatrixSpec& m,
                    std::vector<std::uint8_t>& buf) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + suffix;
        const artifact::GgufTensorInfo* t = g.find(name);
        if (t == nullptr) return false;
        return g.read_at(t->offset + static_cast<std::uint64_t>(expert) * m.rows * m.row_bytes(),
                         buf.data(), buf.size());
    };
    if (!load("gate_exps.weight", spec.gate, bg) || !load("up_exps.weight", spec.up, bu) ||
        !load("down_exps.weight", spec.down, bd)) {
        std::printf("读矩阵失败\n");
        return 1;
    }

    experts::ExpertWeights w{bg.data(), bu.data(), bd.data()};
    experts::FfnScratch scratch;
    std::vector<float> x(static_cast<std::size_t>(spec.gate.cols));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.03125);
    }
    std::vector<float> out(static_cast<std::size_t>(spec.down.rows), 0.0f);
    if (!experts::expert_ffn(spec, w, x.data(), scratch, out.data(), err)) {
        std::printf("前馈失败：%s\n", err.c_str());
        return 1;
    }
    Matrix G{bg}, U{bu}, D{bd};
    std::vector<double> ref;
    ref_ffn(spec, G, U, D, x.data(), ref);
    std::vector<double> got(out.begin(), out.end());
    const int bad = compare(got, ref, "真模型 ffn vs 参考", 1e-5);
    double mag = 0.0, ssum = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        mag += std::fabs(ref[i]);
        ssum += got[i];
    }
    std::printf("层 %d 专家 %d：gate %s / up %s / down %s；out 前 3 值 %.6g %.6g %.6g；sum %.6g\n",
                layer, expert, experts::format_name(spec.gate.format),
                experts::format_name(spec.up.format), experts::format_name(spec.down.format),
                got[0], got[1], got[2], ssum);
    std::printf("与双精度参考比对：%s（|参考| 合计 %.6g）\n", bad ? "不一致" : "一致", mag);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--model") {
        int layer = 0, expert = 0;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--expert") expert = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer, expert);
    }
    test_analytic_uniform();
    test_matches_reference();
    test_unusable_specs_fail();
    std::puts("ffn: analytic uniform case, double-precision reference and unusable-spec failures hold");
    return 0;
}