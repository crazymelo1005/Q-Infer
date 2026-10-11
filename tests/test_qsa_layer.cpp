// QSA 一层结构侧（`build_qsa_layer`）的回归。不引第三方框架。
//
// 合成夹具：n_embd 32、head_dim 8、n_head 4（故 attn_q 输出 64 = 2·4·8）、n_head_kv 2（attn_v 输出 16）、
// idx_dim 4、idx_n_head 2（indexer.q_proj 输出 8）、n_rot 4。投影用 Q8_0（cols 都是 32 的倍数）、
// 四个 norm 用 F32 一维。四条：建得起来且推出来的几何对；四种接错各自明确报错（不是 QSA 层、
// 缺一张张力、head_dim 与形状不自洽、norm 长度不对）。
#include "dense/qsa_layer.hpp"

#include "check.hpp"
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
constexpr std::uint64_t kHd = 8;      // head_dim
constexpr std::uint64_t kNh = 4;      // n_head
constexpr std::uint64_t kNkv = 2;     // n_head_kv
constexpr std::uint64_t kIdx = 4;     // idx_dim
constexpr std::uint64_t kIdxH = 2;    // idx_n_head
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

// Q8_0 一行（cols 是 32 的倍数）：d = 0.5、码取 1..N 的循环。
std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        out.push_back(0x00);
        out.push_back(0x38);
        for (std::uint64_t c = 0; c < cols; ++c) out.push_back(static_cast<std::uint8_t>(c % 5));
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(std::uint64_t n, float v) {
    std::vector<float> f(n, v);
    std::vector<std::uint8_t> out(n * 4);
    std::memcpy(out.data(), f.data(), out.size());
    return out;
}

std::vector<TensorSpec> qsa_tensors(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_q.weight", {kN, 2 * kNh * kHd}, 8, q8_0_rows(2 * kNh * kHd, kN)});
    ts.push_back({b + "attn_k.weight", {kN, kNkv * kHd}, 8, q8_0_rows(kNkv * kHd, kN)});
    ts.push_back({b + "attn_v.weight", {kN, kNkv * kHd}, 8, q8_0_rows(kNkv * kHd, kN)});
    ts.push_back({b + "attn_output.weight", {kNh * kHd, kN}, 8, q8_0_rows(kN, kNh * kHd)});
    ts.push_back({b + "attn_q_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f)});
    ts.push_back({b + "attn_k_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f)});
    ts.push_back({b + "indexer.q_proj.weight", {kN, kIdxH * kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdxH * kIdx * 2, 0)});
    ts.push_back({b + "indexer.k_proj.weight", {kN, kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdx * 2, 0)});
    ts.push_back({b + "indexer.q_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f)});
    ts.push_back({b + "indexer.k_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f)});
    return ts;
}

bool build_from(const char* name, const std::vector<TensorSpec>& ts, QsaLayer& out,
                std::string& err) {
    const std::filesystem::path p = write_gguf(name, ts);
    artifact::GgufFile g;
    std::string oerr;
    if (!g.open(p.string(), oerr)) {
        err = oerr;
        return false;
    }
    return build_qsa_layer(g, 3, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4, out,
                           err);
}

void test_builds_and_derives_the_geometry() {
    QsaLayer l;
    std::string err;
    const bool ok = build_from("qinfer_qsa_ok.gguf", qsa_tensors("blk.3."), l, err);
    if (!ok) std::printf("DEBUG 建层失败：%s\n", err.c_str());
    CHECK(ok);
    CHECK(err.empty());
    CHECK(l.sh.n_head == kNh);
    CHECK(l.sh.n_head_kv == kNkv);
    CHECK(l.sh.head_dim == kHd);
    CHECK(l.idx_n_head == kIdxH);
    CHECK(l.idx_dim == kIdx);
    CHECK(l.n_rot == 4);
    CHECK(l.sh.idx_top_k == 2048 && l.sh.idx_block == 4);
    // attn_q 的输出是 2·n_head·head_dim。
    CHECK(l.q.rows == 2 * kNh * kHd);
}

void test_failures_are_loud() {
    QsaLayer l;
    std::string err;
    // 不是 QSA 层：带 attn_qkv 的是 GDN 层。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts.push_back({"blk.3.attn_qkv.weight", {kN, 8}, 8, q8_0_rows(8, kN)});
        CHECK(!build_from("qinfer_qsa_gdn.gguf", ts, l, err));
        CHECK(err.find("不是 QSA 层") != std::string::npos);
    }
    // 缺一张张力。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts.erase(ts.begin() + 6);  // indexer.q_proj
        CHECK(!build_from("qinfer_qsa_missing.gguf", ts, l, err));
        CHECK(err.find("indexer.q_proj") != std::string::npos);
    }
    // head_dim 与形状不自洽：attn_q 的输出不是 2·head_dim 的整数倍。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[0].dims[1] = 2 * kNh * kHd + 1;  // 65
        ts[0].data = q8_0_rows(2 * kNh * kHd + 1, kN);
        CHECK(!build_from("qinfer_qsa_geo.gguf", ts, l, err));
        CHECK(!err.empty());
    }
    // norm 的长度不对。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[4].dims[0] = kHd + 1;
        ts[4].data = f32_vec(kHd + 1, 1.0f);
        CHECK(!build_from("qinfer_qsa_norm.gguf", ts, l, err));
        CHECK(err.find("attn_q_norm") != std::string::npos);
    }
    // 档没有内核：把 attn_k 换成 F32。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[1].type = 0;
        ts[1].data = f32_vec(kNkv * kHd * kN, 0.5f);
        CHECK(!build_from("qinfer_qsa_nokernel.gguf", ts, l, err));
        CHECK(err.find("attn_k") != std::string::npos);
    }
}

}  // namespace

int main() {
    test_builds_and_derives_the_geometry();
    test_failures_are_loud();
    std::puts("qsa_layer: 从张力名与形状建起一层并推出几何（n_head/n_head_kv/idx_n_head），"
              "五种接错都明确报错");
    return 0;
}