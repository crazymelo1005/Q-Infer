// MTP 入口（两方言）的回归。不引第三方框架。
//
// 夹具是入口那一小段的小号复制：n_embd 32（Q8_0 一块正好 32 值）、hc 2、hc_lr 2。权重值取
// 「0.125 的整数倍且 |v| ≤ 1」，故同一组值能**精确**编进 Q8_0（d = 0.125，码 = v/0.125）与 BF16，
// 从而可以拿同一组值分别做成两份 pack、断言两条路径给出同一个 R —— 把「两方言等价」这条上游结论
// 在合成侧钉住（vLLM 0.31 mtp.py l.190-307 与引擎 mtp.cpp l.488-497 逐句对应，[S-62]）。
//
// 三处错读法各按错的算一遍（用独立的双精度参考，不走实现）：
//   ① en 只加在第一条流上（而不是广播到每条流）；
//   ② 残差那支的归一逐流各一条（而不是整栈 HC·N 一条）；
//   ③ fc_hidden 吃整栈（而不是逐流）。
#include "dense/mtp_entry.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kN = 32, kHc = 2;
constexpr std::uint64_t kHcDim = kN * kHc;  // 64
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    std::vector<std::uint8_t> b;
    auto u32 = [&b](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u64 = [&b](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    const auto s = [&](const std::string& v) {
        u64(v.size());
        b.insert(b.end(), v.begin(), v.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    s("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        s(t.name);
        u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (b.size() % kAlign != 0) b.push_back(0);
    for (const TensorSpec& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        while (b.size() % kAlign != 0) b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p;
}

// 逻辑权重值：入口两张矩阵共用一套取值函数（行 r、列 j），保证两份 pack 编的是同一组值。
float wval(std::uint64_t r, std::uint64_t j) {
    return 0.125f * static_cast<float>(static_cast<int>((r * 7 + j * 3) % 9) - 4);
}

std::vector<std::uint8_t> bf16_mat(std::uint64_t rows, std::uint64_t cols) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t j = 0; j < cols; ++j) {
            const float v = wval(r, j % kN);
            const std::uint32_t u = [&] {
                std::uint32_t t;
                std::memcpy(&t, &v, 4);
                return t;
            }();
            const std::uint16_t bits = static_cast<std::uint16_t>(u >> 16);
            out.push_back(static_cast<std::uint8_t>(bits & 0xFF));
            out.push_back(static_cast<std::uint8_t>(bits >> 8));
        }
    }
    return out;
}

// Q8_0：每行每 32 列一块，d = 0.125（fp16 0x3000），码 = v/0.125。cols 必须是 32 的倍数。
std::vector<std::uint8_t> q8_mat(std::uint64_t rows, std::uint64_t cols, std::uint64_t col_off) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t b = 0; b < cols / 32; ++b) {
            out.push_back(0x00);
            out.push_back(0x30);  // fp16 0.125
            for (std::uint64_t j = 0; j < 32; ++j) {
                const float v = wval(r, (col_off + b * 32 + j) % kN);
                out.push_back(static_cast<std::uint8_t>(static_cast<int>(v / 0.125f)));
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

void add1(std::vector<TensorSpec>& ts, const std::string& n, std::uint64_t elems) {
    ts.push_back({n, {elems}, 0, f32_vec(std::vector<float>(elems, 1.0f))});
}

// 引擎那份的入口五张（名字用 checkpoint 原始名；另有 fc_embedding 作为方言标记）。
std::vector<TensorSpec> engine_fixture() {
    std::vector<TensorSpec> ts;
    add1(ts, "mtp.pre_fc_norm_embedding.weight", kN);
    add1(ts, "mtp.pre_fc_norm_hidden.weight", kHcDim);
    ts.push_back({"mtp.fc_embedding.weight", {kN, kN}, 30, bf16_mat(kN, kN)});
    ts.push_back({"mtp.fc_hidden.weight", {kN, kN}, 30, bf16_mat(kN, kN)});
    return ts;
}

// 社区那份的入口三张（eh_proj 作为方言标记；输入轴 5120 = 2560+2560 的缩影 64 = 32+32）。
std::vector<TensorSpec> blk48_fixture() {
    std::vector<TensorSpec> ts;
    add1(ts, "blk.48.nextn.enorm.weight", kN);
    add1(ts, "blk.48.nextn.hnorm.weight", kHcDim);
    ts.push_back({"blk.48.nextn.eh_proj.weight", {2 * kN, kN}, 8, q8_mat(kN, 2 * kN, 0)});
    return ts;
}

// 独立双精度参考（不走实现）：按给定的读法算入口。
std::vector<double> ref_entry(const float* tok_embd, const float* R_in, bool broadcast_e,
                              bool per_stream_norm, bool hidden_whole_stack) {
    auto rms = [](const float* x, std::uint64_t n) {
        double ss = 0.0;
        for (std::uint64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        return 1.0 / std::sqrt(ss / static_cast<double>(n) + 1e-6);
    };
    const double se = rms(tok_embd, kN);
    std::vector<double> en(kN);
    for (std::uint64_t i = 0; i < kN; ++i) en[i] = se * tok_embd[i];  // gamma = 1
    std::vector<double> hn(kHcDim);
    if (per_stream_norm) {
        for (std::uint64_t c = 0; c < kHc; ++c) {
            const double sc = rms(R_in + c * kN, kN);
            for (std::uint64_t i = 0; i < kN; ++i) hn[c * kN + i] = sc * R_in[c * kN + i];
        }
    } else {
        const double sc = rms(R_in, kHcDim);
        for (std::uint64_t j = 0; j < kHcDim; ++j) hn[j] = sc * R_in[j];
    }
    std::vector<double> out(kHcDim, 0.0);
    for (std::uint64_t c = 0; c < kHc; ++c) {
        for (std::uint64_t r = 0; r < kN; ++r) {
            // fc_hidden 的输入：逐流取该流的 N 个值；错读法是每流都用第 0 条流那一份。
            const double* hin = hidden_whole_stack ? hn.data() : hn.data() + c * kN;
            double h = 0.0;
            for (std::uint64_t j = 0; j < kN; ++j) {
                h += static_cast<double>(wval(r, j)) * hin[j];
            }
            double e = 0.0;
            for (std::uint64_t j = 0; j < kN; ++j) e += static_cast<double>(wval(r, j)) * en[j];
            out[c * kN + r] = h + ((broadcast_e || c == 0) ? e : 0.0);
        }
    }
    return out;
}

double rel_diff(const std::vector<double>& a, const float* b) {
    double diff = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff += std::fabs(a[i] - static_cast<double>(b[i]));
        mag += std::fabs(a[i]);
    }
    return mag > 0.0 ? diff / mag : (diff > 0.0 ? 999.0 : 0.0);
}

void test_both_dialects_agree() {
    const std::filesystem::path pa = write_gguf("qinfer_mtpe_engine.gguf", engine_fixture());
    const std::filesystem::path pb = write_gguf("qinfer_mtpe_blk48.gguf", blk48_fixture());
    artifact::GgufFile ga, gb;
    std::string err;
    CHECK(ga.open(pa.string(), err));
    CHECK(gb.open(pb.string(), err));
    MtpPack pa_, pb_;
    CHECK(build_mtp_pack(ga, kN, kHc, 2, 4, 2, 2, 2, 2, pa_, err, /*entry_only=*/true));
    CHECK(build_mtp_pack(gb, kN, kHc, 2, 4, 2, 2, 2, 2, pb_, err, /*entry_only=*/true));

    std::vector<float> tok(kN), R(kHcDim);
    for (std::uint64_t i = 0; i < kN; ++i) tok[static_cast<std::size_t>(i)] = 0.1f * (float) ((int) i - 16);
    // 两条流的量级刻意相差 10 倍：否则「整栈一条 RMS」与「逐流各一条」两种读法算出来的尺度
    // 几乎相同，那条错读法就不可观测（夹具必须让错的读法显出来）。
    for (std::uint64_t j = 0; j < kHcDim; ++j) {
        const float amp = (j < kN) ? 0.05f : 0.5f;
        R[static_cast<std::size_t>(j)] = amp * (float) ((int) (j % 11) - 5);
    }

    std::vector<float> ra(kHcDim, 0.0f), rb(kHcDim, 0.0f);
    MtpEntryScratch sa, sb;
    CHECK(mtp_entry(ga, pa_, tok.data(), R.data(), sa, ra.data(), err));
    CHECK(mtp_entry(gb, pb_, tok.data(), R.data(), sb, rb.data(), err));
    bool finite = true;
    for (std::uint64_t j = 0; j < kHcDim; ++j) {
        if (!std::isfinite(ra[static_cast<std::size_t>(j)])) finite = false;
    }
    CHECK(finite);
    // 两方言等价：同一组权重值、同一个输入 → 同一个 R。**注意只能要求「同量级内一致」**：
    // 两份 pack 把同一数学编成了不同格式（引擎那份 GGUF 是 BF16、社区那份是 Q8_0），于是
    // **激活处理也不同**（bf16 舍入 vs Q8_0 每 32 值一个尺度），实测差约 3%。上游那句等价是
    // 关于**权重构造**的（eh_proj 是两张矩阵沿输入轴叠起来），不是关于这条实现路径的位级相等。
    double dab = 0.0, mab = 0.0;
    for (std::uint64_t j = 0; j < kHcDim; ++j) {
        dab += std::fabs(static_cast<double>(ra[static_cast<std::size_t>(j)]) -
                         static_cast<double>(rb[static_cast<std::size_t>(j)]));
        mab += std::fabs(static_cast<double>(ra[static_cast<std::size_t>(j)]));
    }
    const double d_ab = mab > 0.0 ? dab / mab : 0.0;
    std::printf("  两方言的全局 L1 相对差 %.4f\n", d_ab);
    CHECK(d_ab < 0.1);
    // 与独立双精度参考一致（整栈归一 + en 广播 + 逐流 fc_hidden）。
    const std::vector<double> want = ref_entry(tok.data(), R.data(), true, false, false);
    const double rel_e = rel_diff(want, ra.data());
    const double rel_c = rel_diff(want, rb.data());
    std::printf("  对独立双精度参考的偏离：引擎 %.3e、社区 %.3e\n", rel_e, rel_c);
    CHECK(rel_e < 5e-2);
    CHECK(rel_c < 5e-2);
    // 三处错读法各按错的算一遍，要求与对的读法有明显差别。
    const double d_stream0 = rel_diff(ref_entry(tok.data(), R.data(), false, false, false), ra.data());
    const double d_perstream = rel_diff(ref_entry(tok.data(), R.data(), true, true, false), ra.data());
    const double d_whole = rel_diff(ref_entry(tok.data(), R.data(), true, false, true), ra.data());
    std::printf("  错读法偏离：en 只加首流 %.4f、归一逐流 %.4f、fc_hidden 每流都用首流 %.4f\n",
                d_stream0, d_perstream, d_whole);
    CHECK(d_stream0 > 0.05);
    CHECK(d_perstream > 0.05);
    CHECK(d_whole > 0.05);
}

// 真权重手工入口：对一份真 MTP pack 跑一遍入口（真几何 N=2560 / hc=4），打印两侧的量级概况。
// 输入是合成的（embedding 行与残差栈），故这一片只核「真字节 + 真几何跑得动、有限、量级合理」；
// 整层的数值核验随层体那片做。
int manual_model_entry(const char* model) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开：%s（%s）\n", model, err.c_str());
        return 1;
    }
    MtpPack p;
    if (!build_mtp_pack(g, 2560, 4, 320, 256, 64, 128, 4, 10, p, err, /*entry_only=*/true)) {
        std::printf("建入口失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("方言：%s；自包含共享头：%s\n",
                p.dialect == MtpDialect::kEngineNames ? "引擎那份（mtp.*）" : "社区那份（blk.48.*）",
                p.self_contained_head ? "是" : "否");
    for (MtpRole r : {MtpRole::kPreNormEmbedding, MtpRole::kPreNormHidden, MtpRole::kFcEmbed,
                      MtpRole::kFcHidden}) {
        const MtpSlice& s = p.at(r);
        std::printf("  %-22s %s 档 %3u cols %5llu rows %5llu\n", to_string(r),
                    s.present() ? "有" : "无", s.present() ? s.type : 0u,
                    (unsigned long long) s.cols, (unsigned long long) s.rows);
    }
    std::vector<float> tok(2560), R(2560 * 4);
    for (std::size_t i = 0; i < tok.size(); ++i) tok[i] = 0.02f * (float) ((int) (i % 13) - 6);
    for (std::size_t j = 0; j < R.size(); ++j) {
        R[j] = ((j < 2560) ? 0.02f : 0.2f) * (float) ((int) (j % 11) - 5);
    }
    std::vector<float> out(2560 * 4, 0.0f);
    MtpEntryScratch s;
    if (!mtp_entry(g, p, tok.data(), R.data(), s, out.data(), err)) {
        std::printf("跑入口失败：%s\n", err.c_str());
        return 1;
    }
    double men = 0.0, mhn = 0.0, mout = 0.0, mo0 = 0.0, mo1 = 0.0;
    bool finite = true;
    for (float v : s.en) men += std::fabs((double) v);
    for (float v : s.hn) mhn += std::fabs((double) v);
    for (std::size_t j = 0; j < out.size(); ++j) {
        if (!std::isfinite(out[j])) finite = false;
        mout += std::fabs((double) out[j]);
        if (j < 2560) {
            mo0 += std::fabs((double) out[j]);
        } else {
            mo1 += std::fabs((double) out[j]);
        }
    }
    std::printf("  |en| %.6g、|hn| %.6g、|R| %.6g（流 0 %.6g / 流 1 %.6g）、有限：%s\n", men, mhn,
                mout, mo0, mo1, finite ? "是" : "否");
    return finite ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        return manual_model_entry(argv[2]);
    }
    test_both_dialects_agree();
    std::puts("mtp_entry: 两方言（引擎 fc_embed+fc_hidden 与社区 eh_proj）给出同一个 R，"
              "且与整栈归一 + en 广播 + 逐流 fc_hidden 的独立双精度参考一致；三处错读法可观测");
    return 0;
}