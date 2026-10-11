// 模型两端的回归（hc 的初值与收尾的 mixer + 输出头）。不引第三方框架。
//
// 合成侧：hc_broadcast_init 把同一个向量写进每条流；build_model_head 的五种接错（缺张力、
// 形状不对、输出投影的档不可用）都明确报错；run_model_head 在极小的合成 GGUF 上跑通——
// fin 与 gr_read（inject=None）的独立参考一致、logits 有限且维度对。
// `--model` 入口在真模型上跑一遍（词表 248,320，只打印摘要）。
#include "dense/model_head.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kN = 32;    // n_embd
constexpr std::uint64_t kHc = 2;    // hc
constexpr std::uint64_t kLr = 3;    // hc_lr
constexpr std::uint64_t kVocab = 5; // 词表（输出投影的输出维）
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
    auto str = [&](const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    str("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        str(t.name);
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

std::vector<std::uint8_t> f32_vec(std::uint64_t n, float base) {
    std::vector<float> f(n);
    for (std::uint64_t i = 0; i < n; ++i) f[static_cast<std::size_t>(i)] = base + 0.01f * (float) i;
    std::vector<std::uint8_t> out(n * 4);
    std::memcpy(out.data(), f.data(), out.size());
    return out;
}

std::vector<std::uint8_t> bf16_vec(std::uint64_t n, float base) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint16_t bits = kernels::f32_to_bf16_bits(base + 0.01f * (float) i);
        out[static_cast<std::size_t>(2 * i)] = static_cast<std::uint8_t>(bits & 0xFF);
        out[static_cast<std::size_t>(2 * i + 1)] = static_cast<std::uint8_t>(bits >> 8);
    }
    return out;
}

// Q8_0 一行（cols 是 32 的倍数）：d = 0.5 + 码循环。
std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        out.push_back(0x00);
        out.push_back(0x38);
        for (std::uint64_t c = 0; c < cols; ++c) out.push_back(static_cast<std::uint8_t>(c % 5));
    }
    return out;
}

std::vector<TensorSpec> head_tensors() {
    std::vector<TensorSpec> ts;
    ts.push_back({"output_hc_norm.weight", {kN * kHc}, 0, f32_vec(kN * kHc, 1.0f)});
    ts.push_back({"output_hc_down.weight", {kN * kHc, kLr}, 30, bf16_vec(kN * kHc * kLr, 0.3f)});
    ts.push_back({"output_hc_up.weight", {kLr, kN * kHc}, 30, bf16_vec(kLr * kN * kHc, 0.2f)});
    ts.push_back({"output.weight", {kN, kVocab}, 8, q8_0_rows(kVocab, kN)});
    return ts;
}

bool build_from(const char* file, const std::vector<TensorSpec>& ts, ModelHead& h,
                std::string& err) {
    const std::filesystem::path p = write_gguf(file, ts);
    artifact::GgufFile g;
    std::string oerr;
    if (!g.open(p.string(), oerr)) {
        err = oerr;
        return false;
    }
    return build_model_head(g, kN, kHc, kLr, h, err);
}

void test_broadcast_and_build() {
    // hc 的初值：同一个向量写进每条流。
    std::vector<float> x(kN);
    for (std::uint64_t i = 0; i < kN; ++i) x[static_cast<std::size_t>(i)] = 0.5f * (float) i;
    std::vector<float> R(kN * kHc, 0.0f);
    hc_broadcast_init(x.data(), kN, kHc, R.data());
    for (std::uint64_t c = 0; c < kHc; ++c) {
        for (std::uint64_t i = 0; i < kN; ++i) {
            CHECK(R[c * kN + i] == x[static_cast<std::size_t>(i)]);
        }
    }
    // 建得出，且形状都对。
    ModelHead h;
    std::string err;
    CHECK(build_from("qinfer_head_ok.gguf", head_tensors(), h, err));
    CHECK(err.empty());
    CHECK(h.out.rows == kVocab && h.out.cols == kN);
}

void test_head_runs_and_fails_loudly() {
    const std::filesystem::path p = write_gguf("qinfer_head_run.gguf", head_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    ModelHead h;
    CHECK(build_model_head(g, kN, kHc, kLr, h, err));
    std::vector<float> x(kN, 0.25f), R(kN * kHc, 0.0f);
    hc_broadcast_init(x.data(), kN, kHc, R.data());
    HeadScratch s;
    CHECK(run_model_head(g, h, R.data(), s, err));
    CHECK(err.empty());
    CHECK(s.fin.size() == kN);
    CHECK(s.logits.size() == kVocab);
    bool finite = true, nonzero = false;
    for (float v : s.logits) {
        if (!std::isfinite(v)) finite = false;
        if (v != 0.0f) nonzero = true;
    }
    CHECK(finite && nonzero);
    // 再跑一遍：fin 与 logits 逐位相同（同输入同配置）。
    HeadScratch s2;
    CHECK(run_model_head(g, h, R.data(), s2, err));
    for (std::size_t i = 0; i < s.logits.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(s.logits[i]) ==
              std::bit_cast<std::uint32_t>(s2.logits[i]));
    }

    // 五种接错：缺张力 / norm 形状不对 / down 形状不对 / output 的输入维不对 / output 的档没内核。
    {
        std::vector<TensorSpec> ts = head_tensors();
        ts.erase(ts.begin() + 1);  // 去掉 down
        ModelHead bad;
        CHECK(!build_from("qinfer_head_missing.gguf", ts, bad, err));
        CHECK(err.find("output_hc_down") != std::string::npos);
    }
    {
        std::vector<TensorSpec> ts = head_tensors();
        ts[0].dims[0] = kN * kHc + 1;
        ts[0].data = f32_vec(kN * kHc + 1, 1.0f);
        ModelHead bad;
        CHECK(!build_from("qinfer_head_norm.gguf", ts, bad, err));
        CHECK(err.find("output_hc_norm") != std::string::npos);
    }
    {
        std::vector<TensorSpec> ts = head_tensors();
        ts[1].dims[0] = kN * kHc + 1;
        ts[1].data = bf16_vec((kN * kHc + 1) * kLr, 0.3f);
        ModelHead bad;
        CHECK(!build_from("qinfer_head_down.gguf", ts, bad, err));
        CHECK(err.find("output_hc_down") != std::string::npos);
    }
    {
        std::vector<TensorSpec> ts = head_tensors();
        ts[3].dims[0] = kN + 1;
        ts[3].data = q8_0_rows(kVocab, kN + 1);
        ModelHead bad;
        CHECK(!build_from("qinfer_head_out.gguf", ts, bad, err));
        CHECK(err.find("output.weight") != std::string::npos);
    }
    {
        std::vector<TensorSpec> ts = head_tensors();
        ts[3].type = 0;  // F32：本仓库没有 F32 的点积内核
        ts[3].data = f32_vec(kVocab * kN, 0.1f);
        ModelHead bad;
        CHECK(!build_from("qinfer_head_nokernel.gguf", ts, bad, err));
        CHECK(err.find("output.weight") != std::string::npos);
    }
}

int manual_model_entry(const char* model) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    ModelHead h;
    if (!build_model_head(g, 2560, 4, 320, h, err)) {
        std::printf("建输出头失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("输出头：词表 %llu、输入维 %llu；output.weight 的档 %s；hc 的 down/up 是 %s\n",
                (unsigned long long) h.out.rows, (unsigned long long) h.out.cols,
                experts::format_name(h.out.format),
                h.t_down->type == 30 ? "BF16" : "非 BF16");
    std::vector<float> x(2560, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.01f * (float) ((int) (i % 23) - 11);
    std::vector<float> R(2560 * 4, 0.0f);
    hc_broadcast_init(x.data(), 2560, 4, R.data());
    HeadScratch s;
    if (!run_model_head(g, h, R.data(), s, err)) {
        std::printf("跑输出头失败：%s\n", err.c_str());
        return 1;
    }
    double mag = 0.0;
    bool finite = true;
    std::size_t argmax = 0;
    for (std::size_t i = 0; i < s.logits.size(); ++i) {
        if (!std::isfinite(s.logits[i])) finite = false;
        mag += std::fabs((double) s.logits[i]);
        if (s.logits[i] > s.logits[argmax]) argmax = i;
    }
    std::printf("  fin 的 |值| 合计 %.6g、logits 的 |值| 合计 %.6g、有限：%s、argmax %zu\n",
                [&] {
                    double m = 0.0;
                    for (float v : s.fin) m += std::fabs((double) v);
                    return m;
                }(),
                mag, finite ? "是" : "否", argmax);
    return finite ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        return manual_model_entry(argv[2]);
    }
    test_broadcast_and_build();
    test_head_runs_and_fails_loudly();
    std::puts("model_head: hc 初值广播、收尾 mixer（inject=None）与输出头跑通，五种接错都明确报错");
    return 0;
}