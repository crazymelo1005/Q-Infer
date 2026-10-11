// 层表的回归。不引第三方框架。
//
// 合成侧三条：GDN/QSA 交替的 6 层能各判对；两套张力同时存在、或都不存在时明确报错并指出是第几层；
// 层数为 0 或负数失败。`--model` 入口扫真模型的 48 层，打印分布（真模型是 36 GDN + 12 QSA、QSA 在
// 3/7/11/…/47）并核「每一层都判得出来」——**分布本身不写死**，因为换一份 pack 可能不同（S-36 的教训）。
#include "dense/layer_map.hpp"

#include "check.hpp"
#include <algorithm>
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
    u32(32);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        str(t.name);
        u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(at);
        at += (t.data.size() + 31) / 32 * 32;
    }
    while (b.size() % 32 != 0) b.push_back(0);
    for (const TensorSpec& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        while (b.size() % 32 != 0) b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p;
}

// 只放一张 1 维小张力即可（判型只看名字在不在）。
TensorSpec marker(const std::string& name) {
    return TensorSpec{name, {4}, 0, std::vector<std::uint8_t>(16, 0)};
}

bool open_and_scan(const char* file, const std::vector<TensorSpec>& ts, int n_layers,
                   std::vector<LayerMapRow>& rows, std::string& err) {
    const std::filesystem::path p = write_gguf(file, ts);
    artifact::GgufFile g;
    std::string oerr;
    if (!g.open(p.string(), oerr)) {
        err = oerr;
        return false;
    }
    return scan_layers(g, n_layers, rows, err);
}

void test_alternating_layers() {
    std::vector<TensorSpec> ts;
    for (int l = 0; l < 6; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        // 偶数层 GDN（有 attn_qkv）、奇数层 QSA（有 attn_q）。
        ts.push_back(marker(b + (l % 2 == 0 ? "attn_qkv.weight" : "attn_q.weight")));
    }
    std::vector<LayerMapRow> rows;
    std::string err;
    CHECK(open_and_scan("qinfer_layermap_alt.gguf", ts, 6, rows, err));
    CHECK(err.empty());
    CHECK(rows.size() == 6);
    for (int l = 0; l < 6; ++l) {
        CHECK(rows[static_cast<std::size_t>(l)].kind ==
              (l % 2 == 0 ? LayerKind::kGdn : LayerKind::kQsa));
    }
}

void test_bad_layers_are_reported_by_number() {
    std::vector<TensorSpec> ts;
    ts.push_back(marker("blk.0.attn_qkv.weight"));  // 0：GDN
    ts.push_back(marker("blk.1.attn_qkv.weight"));  // 1：两种都有 -> 不行
    ts.push_back(marker("blk.1.attn_q.weight"));
    ts.push_back(marker("blk.2.mlp.weight"));  // 2：两种都没有 -> 不行
    std::vector<LayerMapRow> rows;
    std::string err;
    CHECK(!open_and_scan("qinfer_layermap_bad.gguf", ts, 3, rows, err));
    CHECK(err.find("层 1") != std::string::npos);
    CHECK(err.find("层 2") != std::string::npos);
    CHECK(rows.size() == 3);
    CHECK(rows[0].kind == LayerKind::kGdn);
    CHECK(rows[1].kind == LayerKind::kUnknown && rows[2].kind == LayerKind::kUnknown);
    // 层数为 0 失败。
    err.clear();
    CHECK(!open_and_scan("qinfer_layermap_zero.gguf", ts, 0, rows, err));
    CHECK(!err.empty());
}

int manual_model_entry(const char* model, int n_layers) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    std::vector<LayerMapRow> rows;
    if (!scan_layers(g, n_layers, rows, err)) {
        std::printf("扫层失败：%s\n", err.c_str());
        return 1;
    }
    std::size_t gdn = 0, qsa = 0;
    std::vector<int> qsa_at;
    for (const LayerMapRow& r : rows) {
        if (r.kind == LayerKind::kGdn) ++gdn;
        if (r.kind == LayerKind::kQsa) {
            ++qsa;
            qsa_at.push_back(r.layer);
        }
    }
    std::printf("共 %zu 层：GDN %zu、QSA %zu\n", rows.size(), gdn, qsa);
    std::printf("  QSA 在：");
    for (int l : qsa_at) std::printf(" %d", l);
    std::printf("\n");
    return (gdn + qsa == rows.size()) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--model") {
        int n_layers = 48;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layers") n_layers = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], n_layers);
    }
    test_alternating_layers();
    test_bad_layers_are_reported_by_number();
    std::puts("layer_map: 逐层从名字判 GDN/QSA，两类张力同缺或同存都按层号报出来");
    return 0;
}