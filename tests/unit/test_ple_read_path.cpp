// PLE 表行读取路径的回归：token 序列 → 哈希出行索引 → 过预算仲裁 → 命中缓存或读表 → 反量化 →
// 按 head-slowest 拼成 2560 个 float。装配本身现在在 src/ple/row_path（生产代码），本测试只提供
// 一个确定性替身表与断言。
//
// 替身表：第 r 行的字节 = fp16 尺度(0x2000 | (r & 0x3FF)) + 16 个零半字节。零半字节让一行的 160 个值
// 全等于 d·cb[0] = -127·d，于是「哪个头拼到了哪一段」与「有没有走缓存」都能被直接断言；反量化本身的
// 正确性由 test_iq4nl 负责。
//
// 真模型手工入口：`./build/test_ple_read_path --model <分片2.gguf> --tokens a,b,c`，
// 用真实的 per_layer_token_embd.weight 跑同一条路径，打印每个头的行号与取值，便于与
// measure/ple_row_oracle.py 对照（那是独立的 Python 实现）。
#include "ple/row_path.hpp"

#include "artifact/gguf_table.hpp"
#include "check.hpp"
#include "kernels/iq4nl.hpp"
#include "kernels/ngram.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer;

namespace {

std::uint16_t scale_bits(std::uint32_t row) {
    return static_cast<std::uint16_t>(0x2000 | (row & 0x3FF));
}

float row_value(std::uint32_t row) { return -127.0f * kernels::f16_bits_to_f32(scale_bits(row)); }

// 确定性替身表：行字节只由行号决定。行数取真表的行数——哈希常量是按那个几何转录的，
// 故行号必然落在 [0, 320001536) 内；声明少了就会被边界检查正确地拦下。
struct SynthTable : ple::RowSource {
    std::uint64_t rows() const override { return 320001536ull; }
    bool read_row(std::uint32_t row, std::uint8_t* out) const override {
        const std::uint16_t sb = scale_bits(row);
        std::memset(out, 0, kernels::kIq4nlRowBytes);
        for (int blk = 0; blk < 5; ++blk) {  // 5 个块都设尺度，否则后 4 块是 0
            out[blk * 18] = static_cast<std::uint8_t>(sb & 0xFF);
            out[blk * 18 + 1] = static_cast<std::uint8_t>(sb >> 8);
        }
        return true;
    }
};

std::vector<std::uint32_t> rows_of(const std::vector<std::int32_t>& seq) {
    const std::vector<std::int32_t> prev = ple::build_prev(seq.data(), static_cast<int>(seq.size()));
    std::vector<std::uint32_t> rows(seq.size() * ple::kNHeads);
    kernels::ngram_rows(seq.data(), prev.data(), static_cast<int>(seq.size()),
                        kernels::ple_artifact_consts(), rows.data());
    return rows;
}

ple::PathStats run(const SynthTable& table, const std::vector<std::int32_t>& seq,
                   storage::RowCache& cache, std::uint64_t budget, std::vector<float>& out) {
    out.assign(seq.size() * ple::kVectorDim, 0.0f);
    ple::PathStats stats;
    std::string err;
    CHECK(ple::ple_vectors_for_tokens(table, seq.data(), static_cast<int>(seq.size()), cache, budget,
                                      out.data(), stats, err));
    CHECK(err.empty());
    return stats;
}

void test_single_request_read_path() {
    const SynthTable table;
    const std::vector<std::int32_t> seq{92989, 249574, 135030, 248044, 0};
    const std::vector<std::uint32_t> rows = rows_of(seq);

    storage::RowCache cache(1u << 20);  // 足够大：这一遍全冷
    std::vector<float> vec;
    const ple::PathStats p = run(table, seq, cache, 1u << 20, vec);

    CHECK(p.admitted == seq.size() * ple::kNHeads);
    CHECK(p.dropped == 0);
    CHECK(p.overspend_bytes == 0);                       // 预算充足：不超支
    CHECK(p.table_reads == seq.size() * ple::kNHeads);   // 全冷：每行读一次表
    CHECK(p.cache_hits == 0);

    // 每个头拼到自己的那一段，值是 d·cb[0]（零半字节），且各头不同。
    for (std::size_t t = 0; t < seq.size(); ++t) {
        for (int h = 0; h < ple::kNHeads; ++h) {
            const float want = row_value(rows[t * ple::kNHeads + h]);
            for (int k = 0; k < ple::kHeadDim; ++k) {
                const float got = vec[t * ple::kVectorDim + h * ple::kHeadDim + k];
                CHECK(std::fabs(got - want) <= 1e-6f);
            }
        }
    }
}

void test_second_pass_is_all_cache_hits() {
    const SynthTable table;
    const std::vector<std::int32_t> seq{11, 22, 33, 44, 55, 66, 77, 88};
    storage::RowCache cache(1u << 20);

    std::vector<float> first_vec, second_vec;
    const ple::PathStats first = run(table, seq, cache, 1u << 20, first_vec);
    const ple::PathStats second = run(table, seq, cache, 1u << 20, second_vec);

    CHECK(first.table_reads == seq.size() * ple::kNHeads && first.cache_hits == 0);
    CHECK(first.overspend_bytes == 0);
    CHECK(second.table_reads == 0);                      // 第二遍全命中
    CHECK(second.cache_hits == seq.size() * ple::kNHeads);
    CHECK(second_vec == first_vec);                      // 缓存不改变结果
}

void test_tight_budget_still_admits_every_required_row() {
    // 预算小到一个 token 的 16 行装不下：必需项仍全部准入（记 overspend），不得丢表行。
    const SynthTable table;
    const bool w = true;
    (void)w;
    const std::vector<std::int32_t> seq{7, 8, 9};
    const std::vector<std::int32_t> prev = ple::build_prev(seq.data(), static_cast<int>(seq.size()));
    std::vector<std::uint32_t> rows(seq.size() * ple::kNHeads);
    kernels::ngram_rows(seq.data(), prev.data(), static_cast<int>(seq.size()),
                        kernels::ple_artifact_consts(), rows.data());
    (void)rows;

    storage::RowCache c1(64), c2(64);
    std::vector<float> v1, v2;
    const ple::PathStats s1 = run(table, seq, c1, 1, v1);
    const ple::PathStats s2 = run(table, seq, c2, 1u << 20, v2);
    CHECK(s1.dropped == 0);
    CHECK(v1 == v2);                                        // 紧预算不改变结果
    // 每个 token 的 16 行各自过一轮仲裁，超支按轮累计：一轮 16×90 字节对上 1 字节预算 -> 超 1439。
    CHECK(s1.overspend_bytes == static_cast<std::uint64_t>(seq.size()) *
                                    (ple::kNHeads * kernels::kIq4nlRowBytes - 1));
}

void test_build_prev_layout() {
    // 哈希要的是「前两个 token」，顺序是 [i-2, i-1]；越界写 kTokenNull。
    // 这条必须直接断言：测试里的 rows_of 也用模块的 build_prev，若它写错两边会一起错、测不出来。
    const std::int32_t seq[4] = {10, 20, 30, 40};
    const std::vector<std::int32_t> prev = ple::build_prev(seq, 4);
    CHECK(prev.size() == 8u);
    CHECK(prev[0] == kernels::kTokenNull && prev[1] == kernels::kTokenNull);
    CHECK(prev[2] == kernels::kTokenNull && prev[3] == 10);
    CHECK(prev[4] == 10 && prev[5] == 20);
    CHECK(prev[6] == 20 && prev[7] == 30);
}

void test_row_out_of_range_fails() {
    // 行号越界必须失败，而不是静默读到别的行。
    struct TinyTable : ple::RowSource {
        std::uint64_t rows() const override { return 4; }   // 只有 4 行，而哈希会给出行号远大于 4
        bool read_row(std::uint32_t, std::uint8_t* out) const override {
            std::memset(out, 0, kernels::kIq4nlRowBytes);
            return true;
        }
    } tiny;
    const std::vector<std::int32_t> seq{92989, 249574};
    storage::RowCache cache(64);
    std::vector<float> vec(seq.size() * ple::kVectorDim, 0.0f);
    ple::PathStats stats;
    std::string err;
    CHECK(!ple::ple_vectors_for_tokens(tiny, seq.data(), static_cast<int>(seq.size()), cache,
                                       1u << 20, vec.data(), stats, err));
    CHECK(err.find("超出表范围") != std::string::npos);
}

// ---- 真模型手工入口 ----

int manual_model_entry(const std::string& path, const std::string& tokens_arg) {
    artifact::GgufTable table;
    std::string err;
    if (!table.open(path, err)) {
        std::printf("打不开：%s\n", err.c_str());
        return 1;
    }
    std::printf("表：%s 行 %llu，行宽 %zu\n", table.tensor_name().c_str(),
                static_cast<unsigned long long>(table.rows()), table.row_bytes());

    std::vector<std::int32_t> seq;
    std::size_t at = 0;
    while (at < tokens_arg.size()) {
        const std::size_t comma = tokens_arg.find(',', at);
        seq.push_back(static_cast<std::int32_t>(
            std::strtol(tokens_arg.substr(at, comma - at).c_str(), nullptr, 10)));
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    if (seq.empty()) {
        std::printf("没有 token\n");
        return 1;
    }

    ple::GgufTableRowSource src(table);
    storage::RowCache cache(1u << 16);
    std::vector<float> vec(seq.size() * ple::kVectorDim, 0.0f);
    ple::PathStats stats;
    if (!ple::ple_vectors_for_tokens(src, seq.data(), static_cast<int>(seq.size()), cache, 1u << 20,
                                     vec.data(), stats, err)) {
        std::printf("路径失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("token %zu 个，全冷：读表 %llu 次、命中 %llu 次、越界 %llu\n", seq.size(),
                static_cast<unsigned long long>(stats.table_reads),
                static_cast<unsigned long long>(stats.cache_hits),
                static_cast<unsigned long long>(stats.dropped));

    const std::vector<std::uint32_t> rows = rows_of(seq);
    for (std::size_t t = 0; t < seq.size(); ++t) {
        std::printf("token %d 的 16 个行号：", seq[t]);
        for (int h = 0; h < ple::kNHeads; ++h) std::printf(" %u", rows[t * ple::kNHeads + h]);
        std::printf("\n");
    }
    // 打印第 0 个 token 每个头的前 16 个值与 L1，便于与 ple_row_oracle.py 的输出逐行对照
    // （那个脚本对同一行也输出 first16 与 L1，是独立的 Python 实现）。
    for (int h = 0; h < ple::kNHeads; ++h) {
        const float* seg = vec.data() + static_cast<std::size_t>(h) * ple::kHeadDim;
        double l1 = 0.0;
        for (int k = 0; k < ple::kHeadDim; ++k) l1 += std::fabs(static_cast<double>(seg[k]));
        std::printf("  head %2d row %-10u L1 %.6f  first16", h, rows[h], l1);
        for (int k = 0; k < 16; ++k) std::printf(" %.9g", static_cast<double>(seg[k]));
        std::printf("\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--model") {
        std::string tokens = "92989,249574,135030,248044,0";
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--tokens") tokens = argv[i + 1];
        }
        return manual_model_entry(argv[2], tokens);
    }
    test_single_request_read_path();
    test_second_pass_is_all_cache_hits();
    test_tight_budget_still_admits_every_required_row();
    test_build_prev_layout();
    test_row_out_of_range_fails();
    std::puts("ple_read_path: token -> row -> cache -> dequant -> 2560 floats");
    return 0;
}