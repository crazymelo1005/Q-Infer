// GGUF 表行读取器的回归。CI 用现场生成的最小合成 GGUF（不把模型文件带进仓库），
// 另可对真实模型手工验收：./test_gguf_table_reader --model <shard2.gguf> [--row N]
// 它打印出的前 16 值与 L1 和，应与 measure/ple_row_oracle.py 的输出一致。
#include "artifact/gguf_table.hpp"
#include "check.hpp"
#include "kernels/iq4nl.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;

namespace {

constexpr int kHeadDim = 160;
constexpr int kRowBytes = 90;

struct Builder {
    std::vector<std::uint8_t> b;
    void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
    void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
    void raw(const char* s, std::size_t n) { b.insert(b.end(), s, s + n); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void kv_u32(const std::string& k, std::uint32_t v) { str(k); u32(4); u32(v); }        // 4 = UINT32
    void kv_str(const std::string& k, const std::string& v) { str(k); u32(8); str(v); }  // 8 = STRING
};

// 第 r 行 = 5 块 ×（fp16 尺度 0x2000|(r&0x3FF) + 16 个零半字节）→ 一行的 160 个值都等于 -127·d。
std::vector<std::uint8_t> synth_row(std::uint32_t r) {
    std::vector<std::uint8_t> row(kRowBytes, 0);
    const std::uint16_t sb = static_cast<std::uint16_t>(0x2000 | (r & 0x3FF));
    for (int blk = 0; blk < 5; ++blk) {
        row[blk * 18] = static_cast<std::uint8_t>(sb & 0xFF);
        row[blk * 18 + 1] = static_cast<std::uint8_t>(sb >> 8);
    }
    return row;
}

float row_value(std::uint32_t r) {
    const std::uint16_t sb = static_cast<std::uint16_t>(0x2000 | (r & 0x3FF));
    return -127.0f * kernels::f16_bits_to_f32(sb);
}

std::filesystem::path fixture_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// 写一个最小但结构完整的 GGUF：头 + 2 个键值 + 1 个张力信息 + 对齐填充 + 行数据。
std::filesystem::path write_fixture(const char* name, std::uint64_t rows, std::uint64_t head_dim = kHeadDim,
                                    std::uint32_t type_code = 20, bool truncate = false,
                                    const std::string& tensor_name = "per_layer_token_embd.weight") {
    Builder x;
    x.raw("GGUF", 4);
    x.u32(3);            // 版本
    x.u64(1);            // 张力数
    x.u64(2);            // 键值数
    x.kv_u32("general.alignment", 32);
    x.kv_str("general.architecture", "qwen4exp");
    x.str(tensor_name);
    x.u32(2);            // 维度数
    x.u64(head_dim);
    x.u64(rows);
    x.u32(type_code);
    x.u64(0);            // 数据区内的偏移
    while (x.b.size() % 32 != 0) x.b.push_back(0);
    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::vector<std::uint8_t> row = synth_row(static_cast<std::uint32_t>(r));
        x.b.insert(x.b.end(), row.begin(), row.end());
    }
    if (truncate && x.b.size() > 40) x.b.resize(x.b.size() - 40);

    const std::filesystem::path p = fixture_path(name);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(x.b.data()), static_cast<std::streamsize>(x.b.size()));
    return p;
}

void test_reads_rows_from_a_synthetic_gguf() {
    const std::filesystem::path p = write_fixture("qinfer_fixture_ok.gguf", 3);
    artifact::GgufTable table;
    std::string err;
    CHECK(table.open(p.string(), err));
    CHECK(table.rows() == 3);
    CHECK(table.row_bytes() == kRowBytes);
    CHECK(table.tensor_name() == "per_layer_token_embd.weight");

    std::uint8_t row[kRowBytes];
    for (std::uint32_t r = 0; r < 3; ++r) {
        CHECK(table.read_row(r, row));
        const std::vector<std::uint8_t> want = synth_row(r);
        CHECK(std::memcmp(row, want.data(), kRowBytes) == 0);
        float v[kHeadDim];
        kernels::dequant_iq4nl_row(row, v);
        for (int k = 0; k < kHeadDim; ++k) CHECK(std::fabs(v[k] - row_value(r)) <= 1e-6f);
    }
    std::filesystem::remove(p);
}

void test_rejects_out_of_range_row() {
    const std::filesystem::path p = write_fixture("qinfer_fixture_range.gguf", 2);
    artifact::GgufTable table;
    std::string err;
    CHECK(table.open(p.string(), err));
    std::uint8_t row[kRowBytes];
    CHECK(!table.read_row(2, row));          // 行号 == rows：越界
    CHECK(table.read_row(1, row));
    std::filesystem::remove(p);
}

void test_rejects_truncated_file() {
    const std::filesystem::path p = write_fixture("qinfer_fixture_short.gguf", 3, kHeadDim, 20, /*truncate=*/true);
    artifact::GgufTable table;
    std::string err;
    CHECK(!table.open(p.string(), err));     // 张力放不进文件
    CHECK(!err.empty());
    std::filesystem::remove(p);
}

void test_rejects_wrong_shape_or_type() {
    {   // 行宽不是 160
        const std::filesystem::path p = write_fixture("qinfer_fixture_shape.gguf", 2, /*head_dim=*/128);
        artifact::GgufTable table;
        std::string err;
        CHECK(!table.open(p.string(), err));
        std::filesystem::remove(p);
    }
    {   // 类型不是 IQ4_NL（20）
        const std::filesystem::path p = write_fixture("qinfer_fixture_type.gguf", 2, kHeadDim, /*type=*/8);
        artifact::GgufTable table;
        std::string err;
        CHECK(!table.open(p.string(), err));
        std::filesystem::remove(p);
    }
}

void test_reports_missing_tensor() {
    const std::filesystem::path p = write_fixture("qinfer_fixture_other.gguf", 2, kHeadDim, 20, false, "other.weight");
    artifact::GgufTable table;
    std::string err;
    CHECK(!table.open(p.string(), err));
    CHECK(err.find("per_layer_token_embd.weight") != std::string::npos);
    std::filesystem::remove(p);
}

void test_rejects_non_gguf() {
    const std::filesystem::path p = fixture_path("qinfer_fixture_notgguf.bin");
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    const char junk[64] = "NOTAGGUFFILE";
    f.write(junk, 64);
    f.close();
    artifact::GgufTable table;
    std::string err;
    CHECK(!table.open(p.string(), err));
    std::filesystem::remove(p);
}

// 手工验收：对真实模型跑一遍，打印第 N 行的前 16 值与 L1 和（与 ple_row_oracle.py 对照）。
int manual_against_real_model(const std::string& model, std::uint32_t row) {
    artifact::GgufTable table;
    std::string err;
    if (!table.open(model, err)) {
        std::printf("open 失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("张力 %s  行数 %llu  行宽 %zu\n", table.tensor_name().c_str(),
                static_cast<unsigned long long>(table.rows()), table.row_bytes());
    std::vector<std::uint8_t> bytes(table.row_bytes());
    if (!table.read_row(row, bytes.data())) {
        std::printf("读行 %u 失败\n", row);
        return 1;
    }
    float v[kHeadDim];
    kernels::dequant_iq4nl_row(bytes.data(), v);
    double l1 = 0;
    for (int k = 0; k < kHeadDim; ++k) l1 += std::fabs(static_cast<double>(v[k]));
    std::printf("行 %u:\n  first16 ", row);
    for (int k = 0; k < 16; ++k) std::printf("%.8g ", v[k]);
    std::printf("\n  last16  ");
    for (int k = 144; k < 160; ++k) std::printf("%.8g ", v[k]);
    std::printf("\n  L1      %.6f\n", l1);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            std::uint32_t row = 0;
            for (int j = i + 1; j < argc; ++j) {
                if (std::strcmp(argv[j], "--row") == 0 && j + 1 < argc) row = static_cast<std::uint32_t>(std::atoi(argv[j + 1]));
            }
            return manual_against_real_model(argv[i + 1], row);
        }
    }
    test_reads_rows_from_a_synthetic_gguf();
    test_rejects_out_of_range_row();
    test_rejects_truncated_file();
    test_rejects_wrong_shape_or_type();
    test_reports_missing_tensor();
    test_rejects_non_gguf();
    std::puts("gguf_table: synthetic fixture + rejections hold");
    return 0;
}
