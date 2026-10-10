// 一整层 GDN 的回归。不引第三方框架。
//
// 合成夹具用极小的几何（n_embd 32 / 头宽 32 / h_k 1 / h_v 1 / 卷积核长 2 → conv_dim 96、v_dim 32），
// 三个投影都用 Q8_0（它的块最简单：2 字节 fp16 尺度 + 32 个 int8 码），其余张力是 F32/BF16。
// 合成侧钉的是**接线与几何校验**：跑通、输出有限且非零、两次同输入逐位相同、卷积状态真的在动，
// 以及三种「接错就必须报错」的情形（不是 GDN 层、conv1d 形状不对、投影没有内核）。
// 真权重上的数值在 --model 入口跑（环境2）。
#include "dense/gdn_layer.hpp"

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

constexpr std::uint64_t kN = 32;      // n_embd
// 头宽取 32 是**被 Q8_0 逼的**：投影的输入维必须能被块元素数整除，而 ssm_out 的输入维就是 v_dim。
constexpr std::uint64_t kHd = 32;     // 头宽
constexpr std::uint64_t kHk = 1;      // q/k 头数
constexpr std::uint64_t kHv = 1;      // v 头数
constexpr std::uint64_t kDc = 2;      // 卷积核长
constexpr std::uint64_t kQk = kHk * kHd;          // 2
constexpr std::uint64_t kVd = kHv * kHd;          // 4
constexpr std::uint64_t kCd = 2 * kQk + kVd;      // 8
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

struct Builder {
    std::vector<std::uint8_t> b;
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    }
    void str(const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    Builder w;
    w.b.insert(w.b.end(), {'G', 'G', 'U', 'F'});
    w.u32(3);
    w.u64(ts.size());
    w.u64(1);
    w.str("general.alignment");
    w.u32(4);
    w.u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        w.str(t.name);
        w.u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) w.u64(d);
        w.u32(t.type);
        w.u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (w.b.size() % kAlign != 0) w.b.push_back(0);
    for (const TensorSpec& t : ts) {
        w.b.insert(w.b.end(), t.data.begin(), t.data.end());
        while (w.b.size() % kAlign != 0) w.b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(w.b.data()), static_cast<std::streamsize>(w.b.size()));
    f.close();
    return p;
}

// ---- 数据构造 ----

// Q8_0 一行（cols 必须是 32 的倍数，这里正好一块）：d = 0.5、码取 1..N 的循环。
std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols, int seed) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        out.push_back(0x00);  // fp16 0.5
        out.push_back(0x38);
        for (std::uint64_t c = 0; c < cols; ++c) {
            out.push_back(static_cast<std::uint8_t>(((c + r + seed) % 7) - 3));
        }
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

std::vector<std::uint8_t> bf16_vec(std::uint64_t n, float base) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint16_t bits = kernels::f32_to_bf16_bits(base + 0.01f * static_cast<float>(i));
        out[static_cast<std::size_t>(2 * i)] = static_cast<std::uint8_t>(bits & 0xFF);
        out[static_cast<std::size_t>(2 * i + 1)] = static_cast<std::uint8_t>(bits >> 8);
    }
    return out;
}

std::vector<TensorSpec> gdn_tensors(const std::string& b, bool bad_conv = false) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_qkv.weight", {kN, kCd}, 8, q8_0_rows(kCd, kN, 1)});
    ts.push_back({b + "attn_gate.weight", {kN, kVd}, 8, q8_0_rows(kVd, kN, 2)});
    ts.push_back({b + "ssm_out.weight", {kVd, kN}, 8, q8_0_rows(kN, kVd, 3)});
    ts.push_back({b + "ssm_conv1d.weight", {bad_conv ? kDc + 1 : kDc, kCd}, 0,
                  f32_vec(std::vector<float>(static_cast<std::size_t>((bad_conv ? kDc + 1 : kDc) * kCd), 0.25f))});
    ts.push_back({b + "ssm_alpha.weight", {kN, kHv}, 30, bf16_vec(kN * kHv, 1.0f)});
    ts.push_back({b + "ssm_beta.weight", {kN, kHv}, 30, bf16_vec(kN * kHv, 0.9f)});
    ts.push_back({b + "ssm_a", {kHv}, 0, f32_vec(std::vector<float>{-0.5f})});
    ts.push_back({b + "ssm_dt.bias", {kHv}, 0, f32_vec(std::vector<float>{0.5f})});
    ts.push_back({b + "ssm_norm.weight", {kHd}, 0, f32_vec(std::vector<float>(kHd, 1.0f))});
    return ts;
}

std::vector<float> make_x() {
    std::vector<float> x(kN);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.1f * static_cast<float>(static_cast<int>(i % 9) - 4);
    }
    return x;
}

// 粘贴一个「不是 GDN 层」的层：只有 QSA 的那几个名字。
std::vector<TensorSpec> qsa_tensors(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_q.weight", {kN, kCd}, 8, q8_0_rows(kCd, kN, 1)});
    ts.push_back({b + "attn_k.weight", {kN, kQk}, 8, q8_0_rows(kQk, kN, 2)});
    ts.push_back({b + "attn_v.weight", {kN, kQk}, 8, q8_0_rows(kQk, kN, 3)});
    return ts;
}

int manual_model_entry(const char* model, int layer) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    GdnLayer gl;
    if (!build_gdn_layer(g, layer, 2560, 128, 4, gl, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::printf("层 %d：conv_dim %llu、value_dim %llu、q/k 各 %llu 头、v %llu 头、头宽 %llu\n", layer,
                static_cast<unsigned long long>(gl.qkv.rows),
                static_cast<unsigned long long>(gl.gate.rows),
                static_cast<unsigned long long>(gl.pre.n_q_head),
                static_cast<unsigned long long>(gl.pre.n_v_head),
                static_cast<unsigned long long>(gl.pre.head_dim));
    std::printf("  档：attn_qkv %s、attn_gate %s、ssm_out %s\n",
                experts::format_name(gl.qkv.format), experts::format_name(gl.gate.format),
                experts::format_name(gl.out.format));
    const std::vector<float> x = make_x();
    std::vector<float> out(static_cast<std::size_t>(gl.pre.n_embd), 0.0f), out2(gl.pre.n_embd, 0.0f);
    GdnScratch s1, s2;
    if (!run_gdn_layer(g, gl, x.data(), out.data(), s1, err)) {
        std::printf("跑层 %d 失败：%s\n", layer, err.c_str());
        return 1;
    }
    double sum = 0.0, mag = 0.0;
    bool finite = true;
    for (const float v : out) {
        if (!std::isfinite(v)) finite = false;
        sum += static_cast<double>(v);
        mag += std::fabs(static_cast<double>(v));
    }
    std::printf("  |out| 合计 %.6g、有限：%s\n", mag, finite ? "是" : "否");
    // 同一个输入、同一个 scratch 结构再跑一次 → 状态走过两步，输出应当不同（状态真的在动）；
    // 而用**新 scratch**再跑一次 → 与第一次逐位相同（同配置可复现）。
    std::vector<float> first = out;  // 留住第一次的结果：下一次会把 out 覆盖成「状态前进后」的
    if (!run_gdn_layer(g, gl, x.data(), out.data(), s1, err)) return 1;
    GdnScratch s3;
    if (!run_gdn_layer(g, gl, x.data(), out2.data(), s3, err)) return 1;
    bool state_moves = false, reproducible = true;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(out[i]) != std::bit_cast<std::uint32_t>(first[i])) {
            state_moves = true;
        }
        if (std::bit_cast<std::uint32_t>(out2[i]) != std::bit_cast<std::uint32_t>(first[i])) {
            reproducible = false;
        }
    }
    std::printf("  状态在动：%s；新 scratch 与第一次逐位相同：%s\n", state_moves ? "是" : "否",
                reproducible ? "是" : "否");
    return (finite && state_moves && reproducible) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 0;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer);
    }

    // 合成：跑通、有限且非零、状态在动、可复现。
    {
        const std::filesystem::path p = write_gguf("qinfer_gdn_ok.gguf", gdn_tensors("blk.0."));
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        GdnLayer gl;
        CHECK(build_gdn_layer(g, 0, kN, kHd, kDc, gl, err));
        CHECK(err.empty());
        CHECK(gl.pre.conv_dim() == kCd && gl.pre.v_dim() == kVd && gl.pre.n_q_head == kHk);

        const std::vector<float> x = make_x();
        std::vector<float> out(kN, 0.0f), again(kN, 0.0f), fresh(kN, 0.0f);
        GdnScratch s, s2;
        CHECK(run_gdn_layer(g, gl, x.data(), out.data(), s, err));
        CHECK(err.empty());
        bool nonzero = false, finite = true;
        for (const float v : out) {
            if (v != 0.0f) nonzero = true;
            if (!std::isfinite(v)) finite = false;
        }
        CHECK(nonzero && finite);

        // 状态走过一步之后再跑同一输入 → 输出变了（卷积状态与循环状态都在动）。
        CHECK(run_gdn_layer(g, gl, x.data(), again.data(), s, err));
        bool differs = false;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(again[i]) != std::bit_cast<std::uint32_t>(out[i])) {
                differs = true;
            }
        }
        CHECK(differs);

        // 新 scratch（状态清零）跑一次 → 与第一次逐位相同。
        CHECK(run_gdn_layer(g, gl, x.data(), fresh.data(), s2, err));
        for (std::size_t i = 0; i < out.size(); ++i) {
            CHECK(std::bit_cast<std::uint32_t>(fresh[i]) == std::bit_cast<std::uint32_t>(out[i]));
        }
    }

    // 三种「接错必须报错」。
    {
        const std::filesystem::path p = write_gguf("qinfer_gdn_qsa.gguf", qsa_tensors("blk.3."));
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        GdnLayer gl;
        CHECK(!build_gdn_layer(g, 3, kN, kHd, kDc, gl, err));
        CHECK(err.find("不是 GDN 层") != std::string::npos);
    }
    {
        const std::filesystem::path p =
            write_gguf("qinfer_gdn_badconv.gguf", gdn_tensors("blk.0.", /*bad_conv=*/true));
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        GdnLayer gl;
        CHECK(!build_gdn_layer(g, 0, kN, kHd, kDc, gl, err));
        CHECK(err.find("ssm_conv1d") != std::string::npos);
    }
    {
        // 把一个投影换成 F32（没有内核）→ 明确失败。
        std::vector<TensorSpec> ts = gdn_tensors("blk.0.");
        ts[0].type = 0;
        ts[0].data = f32_vec(std::vector<float>(kCd * kN, 0.5f));
        const std::filesystem::path p = write_gguf("qinfer_gdn_nokernel.gguf", ts);
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        GdnLayer gl;
        CHECK(!build_gdn_layer(g, 0, kN, kHd, kDc, gl, err));
        CHECK(err.find("attn_qkv") != std::string::npos);
    }

    std::puts("gdn_layer: 三层投影 + 前段 + 递推 + 收尾串成一整层跑通（状态在动、同配置可复现），"
              "三种接错都明确报错");
    return 0;
}