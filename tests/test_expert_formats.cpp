// 专家格式分派表的回归。不引第三方框架。
//
// 四路证据：
//   1. 几何与内核表：块元素数/块字节数按 gguf-py 的 GGML_QUANT_SIZES 核对（S-42），
//      激活搭档按各自 generic 点积核（S-37 / S-40 / S-41）。抽若干档逐项断言。
//   2. 合成夹具的分派：三层的混合格式，含「缺张力」「类型码不认识」「几何不成立」「没有内核」
//      四种情况，断言各自都进了 problems 而不是被静默接受。
//   3. 两档真实模型的逐层映射：把 S-36 与 S-38 记录的 48 层逐层格式写成两串字符，用合成夹具
//      复现同样的类型码，断言分派表还原出同一串；并断言可用的层数（部署那份 45/48，另一档 29/48）。
//   4. 数据区随机读：按张量登记的偏移读回夹具写进去的字节。
//
// 真模型的手工入口：`./build/test_expert_formats --model <分片1.gguf>`，打印逐层映射串。
#include "experts/expert_formats.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::experts;

namespace {

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
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

constexpr std::uint32_t kAlign = 32;

std::filesystem::path write_fixture(const char* name, const std::vector<TensorSpec>& ts) {
    Builder w;
    w.b.insert(w.b.end(), {'G', 'G', 'U', 'F'});
    w.u32(3);                                  // version
    w.u64(ts.size());                          // tensor_count
    w.u64(1);                                  // kv_count
    w.str("general.alignment");                // 键
    w.u32(4);                                  // 类型 UINT32
    w.u32(kAlign);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        w.str(ts[i].name);
        w.u32(static_cast<std::uint32_t>(ts[i].dims.size()));
        for (std::uint64_t d : ts[i].dims) w.u64(d);
        w.u32(ts[i].type);
        w.u64(static_cast<std::uint64_t>(i) * kAlign);  // 数据区偏移
    }
    while (w.b.size() % kAlign != 0) w.b.push_back(0);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        for (std::uint32_t k = 0; k < kAlign; ++k) {
            w.b.push_back(static_cast<std::uint8_t>(i + 1));  // 便于核对 read_at
        }
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(w.b.data()), static_cast<std::streamsize>(w.b.size()));
    f.close();
    return p;
}

std::vector<TensorSpec> expert_tensors(const std::string& gate_map, const std::string& down_map) {
    auto code = [](char c) -> std::uint32_t {
        switch (c) {
            case '0': return 42;   // Q2_0
            case 'S': return 22;   // IQ2_S
            case 'X': return 17;   // IQ2_XS
            case 'x': return 16;   // IQ2_XXS
            case '1': return 29;   // IQ1_M
            case 'L': return 20;   // IQ4_NL
            case '#': return 18;   // IQ3_XXS
            case '3': return 21;   // IQ3_S
            default: CHECK(false && "未知短码");
        }
        return 0;
    };
    std::vector<TensorSpec> out;
    for (std::size_t l = 0; l < gate_map.size(); ++l) {
        const std::string base = "blk." + std::to_string(l) + ".ffn_";
        const std::uint32_t g = code(gate_map[l]);
        // gate/up 的输入维是 hidden、输出维是 ffn；down 反过来。
        out.push_back({base + "gate_exps.weight", {2560, 640, 512}, g});
        out.push_back({base + "up_exps.weight", {2560, 640, 512}, g});
        out.push_back({base + "down_exps.weight", {640, 2560, 512}, code(down_map[l])});
    }
    return out;
}

char short_of(Format f) {
    switch (f) {
        case Format::kQ2_0: return '0';
        case Format::kIq2S: return 'S';
        case Format::kIq2Xs: return 'X';
        case Format::kIq2Xxs: return 'x';
        case Format::kIq1M: return '1';
        case Format::kIq4Nl: return 'L';
        case Format::kIq3Xxs: return '#';
        case Format::kIq3S: return '3';
        case Format::kQ8_0: return '8';
        default: return '?';
    }
}

std::string map_string(const Table& t, bool gate) {
    std::string s;
    for (const auto& l : t.layers) s.push_back(short_of(gate ? l.gate.format : l.down.format));
    return s;
}

void test_geometry_and_kernels() {
    // 块元素数 / 块字节数 / 是否有内核 / 激活搭档。
    struct Row {
        Format f;
        int elems, bytes;
        bool kernel;
        ActFormat act;
    };
    const Row rows[] = {
        {Format::kQ2_0, 64, 18, true, ActFormat::kQ8_0},
        {Format::kIq4Nl, 32, 18, true, ActFormat::kQ8_0},
        {Format::kIq2Xxs, 256, 66, true, ActFormat::kQ8K},
        {Format::kIq2Xs, 256, 74, true, ActFormat::kQ8K},
        {Format::kIq2S, 256, 82, true, ActFormat::kQ8K},
        {Format::kIq3S, 256, 110, false, ActFormat::kQ8K},
        {Format::kIq1M, 256, 56, false, ActFormat::kQ8K},
        {Format::kQ6K, 256, 210, false, ActFormat::kQ8K},
        {Format::kBf16, 1, 2, false, ActFormat::kNone},
    };
    for (const Row& r : rows) {
        if (block_elems(r.f) != r.elems || block_bytes(r.f) != r.bytes ||
            has_kernel(r.f) != r.kernel || activation_format(r.f) != r.act) {
            std::printf("FAIL geometry %s: elems=%d bytes=%d kernel=%d act=%d\n", format_name(r.f),
                        block_elems(r.f), block_bytes(r.f), static_cast<int>(has_kernel(r.f)),
                        static_cast<int>(activation_format(r.f)));
            CHECK(false);
        }
    }
    // 类型码翻译：认得 42（Q2_0）与 99（不认得）。
    CHECK(format_from_type_code(42) == Format::kQ2_0);
    CHECK(format_from_type_code(22) == Format::kIq2S);
    CHECK(format_from_type_code(99) == Format::kUnknown);

    // 几何：640 不能被 256 整除，所以 256 值块的档做不了 down 的输入维。
    MatrixSpec m;
    m.format = Format::kIq2S;
    m.cols = 640;
    m.rows = 2560;
    CHECK(!m.geometry_ok());
    m.format = Format::kQ2_0;
    CHECK(m.geometry_ok());
    CHECK(m.row_blocks() == 10);                      // 640 / 64
    CHECK(m.row_bytes() == 10 * 18);
    m.format = Format::kIq2Xs;
    m.cols = 2560;                                    // gate/up 的输入维，能被 256 整除
    CHECK(m.geometry_ok());
    CHECK(m.row_blocks() == 10 && m.row_bytes() == 10 * 74);
}

void test_synthetic_problems() {
    // 三层：0 层正常；1 层 down 用 IQ2_S（几何不成立）+ gate 用 IQ1_M（无内核）；2 层缺 down。
    std::vector<TensorSpec> ts = {
        {"blk.0.ffn_gate_exps.weight", {2560, 640, 512}, 22},
        {"blk.0.ffn_up_exps.weight", {2560, 640, 512}, 22},
        {"blk.0.ffn_down_exps.weight", {640, 2560, 512}, 42},
        {"blk.1.ffn_gate_exps.weight", {2560, 640, 512}, 29},
        {"blk.1.ffn_up_exps.weight", {2560, 640, 512}, 29},
        {"blk.1.ffn_down_exps.weight", {640, 2560, 512}, 22},
        {"blk.2.ffn_gate_exps.weight", {2560, 640, 512}, 99},  // 类型码不认识
        {"blk.2.ffn_up_exps.weight", {2560, 640, 512}, 22},
        // 名字里第 5 个字符起是数字加点的张力：没有前缀检查时会被当成「层 1」，
    // 从而把层数算错。这种名字在真实 GGUF 里不会出现，但这条守卫要能被测出来。
        {"aaaa9.weight", {16}, 30},
    };
    const std::filesystem::path p = write_fixture("qinfer_expertfmt.gguf", ts);
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    CHECK(layer_count(g) == 3);

    const Table t = build_table(g);
    CHECK(t.layers.size() == 3u);
    CHECK(t.layers[0].usable());
    CHECK(!t.layers[1].usable());
    CHECK(!t.layers[2].usable());
    CHECK(t.usable_layers() == 1u);
    // 五个问题：1 层 gate 与 up 各一个「尚无内核」、down 一个「几何不成立」；
    // 2 层 gate 一个「类型码不认识」、down 一个「张力不存在」。
    CHECK(t.problems.size() == 5u);
    std::string all;
    for (const auto& pr : t.problems) all += pr.tensor + "|" + pr.reason + "\n";
    const bool has_geometry = all.find("整除不了") != std::string::npos;
    const bool has_kernel = all.find("尚无内核") != std::string::npos;
    const bool has_missing = all.find("张力不存在") != std::string::npos;
    const bool has_unknown = all.find("不认识") != std::string::npos;
    if (!(has_geometry && has_kernel && has_missing && has_unknown)) {
        std::printf("FAIL problems:\n%s", all.c_str());
        CHECK(false);
    }

    // 数据区随机读：第 i 个张量的数据区全是 i+1。
    std::uint8_t buf[kAlign];
    CHECK(g.read_at(0, buf, kAlign));
    CHECK(buf[0] == 1 && buf[kAlign - 1] == 1);
    CHECK(g.read_at(2 * kAlign, buf, kAlign));
    CHECK(buf[0] == 3);
    CHECK(!g.read_at(g.file_size(), buf, kAlign));  // 越界
}

void test_real_model_patterns() {
    // 部署那份（S-36）：gate/up 用 IQ2_S 34 层 / IQ2_XXS 11 层 / IQ1_M 3 层，down 全 Q2_0。
    const std::string gate2 = "SxSSxSSS1xxxS1xSSSxxSSSSSSSxSxxSSSSSS1SSSSSSSSSS";
    const std::string down2 = std::string(48, '0');
    {
        const std::filesystem::path p =
            write_fixture("qinfer_expertfmt_iq2.gguf", expert_tensors(gate2, down2));
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        const Table t = build_table(g);
        CHECK(t.layers.size() == 48u);
        CHECK(layer_count(g) == 48);
        if (map_string(t, true) != gate2 || map_string(t, false) != down2) {
            std::printf("FAIL iq2 map: %s / %s\n", map_string(t, true).c_str(),
                        map_string(t, false).c_str());
            CHECK(false);
        }
        // IQ1_M 那 3 层没有内核，其余 45 层可用。
        CHECK(t.usable_layers() == 45u);
        CHECK(t.problems.size() == 6u);  // 3 层 × (gate + up)
    }
    // 另一档（S-38）：gate/up 为 IQ3_S 13 / IQ2_XS 10 / IQ2_S 10 / IQ2_XXS 9 / IQ3_XXS 6，
    // down 为 Q2_0 30 / IQ4_NL 18。
    const std::string gate3 = "XxXxXXxx3XXxSxXSxS3X3S333XSx3S3x333##3SXS#S3##S#";
    const std::string down3 = "L0L0L00000000000000000000000LLLLL0LL0LLL00LLL0LL";
    {
        const std::filesystem::path p =
            write_fixture("qinfer_expertfmt_iq3.gguf", expert_tensors(gate3, down3));
        artifact::GgufFile g;
        std::string err;
        CHECK(g.open(p.string(), err));
        const Table t = build_table(g);
        CHECK(t.layers.size() == 48u);
        if (map_string(t, true) != gate3 || map_string(t, false) != down3) {
            std::printf("FAIL iq3 map: %s / %s\n", map_string(t, true).c_str(),
                        map_string(t, false).c_str());
            CHECK(false);
        }
        // 有内核的 gate 档只有 IQ2_XS 10 + IQ2_S 10 + IQ2_XXS 9 = 29 层。
        CHECK(t.usable_layers() == 29u);
        CHECK(t.problems.size() == 2u * (13 + 6));  // IQ3_S 13 层 + IQ3_XXS 6 层，各 gate/up
    }
}

void manual_model_entry(const std::string& path) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(path, err)) {
        std::printf("打不开：%s\n", err.c_str());
        return;
    }
    const Table t = build_table(g);
    std::printf("层数 %d，可用 %llu\n", layer_count(g),
                static_cast<unsigned long long>(t.usable_layers()));
    std::printf("gate %s\n", map_string(t, true).c_str());
    std::printf("down %s\n", map_string(t, false).c_str());
    for (const auto& pr : t.problems) {
        std::printf("  [层 %d] %s：%s\n", pr.layer, pr.tensor.c_str(), pr.reason.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        manual_model_entry(argv[2]);
        return 0;
    }
    test_geometry_and_kernels();
    test_synthetic_problems();
    test_real_model_patterns();
    std::puts("expert_formats: geometry, dispatch problems and both real per-layer maps hold");
    return 0;
}