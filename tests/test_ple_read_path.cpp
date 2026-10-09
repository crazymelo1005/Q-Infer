// 单请求的表行读取路径端到端（主机侧）：token 序列 → 哈希出行索引 → 过预算仲裁 → 命中缓存或
// 读表 → 反量化 → 按 head-slowest 拼成 2560 个 float。
//
// 表用一个确定性合成替身：第 r 行的字节 = fp16 尺度(0x2000 | (r & 0x3FF)) + 16 个零半字节。
// 零半字节让一行的 160 个值全等于 d·cb[0] = -127·d，于是「哪个头拼到了哪一段」与「有没有走缓存」
// 都能被直接断言，而反量化本身的正确性由 test_iq4nl 负责。
#include "kernels/iq4nl.hpp"
#include "kernels/ngram.hpp"
#include "kernels/ple_gather.hpp"
#include "scheduling/budget_arbiter.hpp"
#include "storage/row_cache.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer;

namespace {

constexpr int kNHeads = kernels::kPleNHeads;
constexpr int kHeadDim = 160;
constexpr int kVector = kNHeads * kHeadDim;   // 2560

std::uint16_t scale_bits(std::uint32_t row) { return static_cast<std::uint16_t>(0x2000 | (row & 0x3FF)); }

float row_value(std::uint32_t row) { return -127.0f * kernels::f16_bits_to_f32(scale_bits(row)); }

std::vector<std::uint8_t> synth_row(std::uint32_t row) {
    std::vector<std::uint8_t> b(kernels::kIq4nlRowBytes, 0);
    const std::uint16_t sb = scale_bits(row);
    for (int blk = 0; blk < 5; ++blk) {          // 5 个块都设尺度，否则后 4 块是 0
        b[blk * 18] = static_cast<std::uint8_t>(sb & 0xFF);
        b[blk * 18 + 1] = static_cast<std::uint8_t>(sb >> 8);
    }
    return b;   // 半字节全 0 → 每块 32 个值都等于 d·cb[0]
}

std::vector<std::int32_t> build_prev(const std::vector<std::int32_t>& seq) {
    std::vector<std::int32_t> prev(seq.size() * 2);
    for (std::size_t i = 0; i < seq.size(); ++i) {
        prev[i * 2] = i >= 2 ? seq[i - 2] : kernels::kTokenNull;
        prev[i * 2 + 1] = i >= 1 ? seq[i - 1] : kernels::kTokenNull;
    }
    return prev;
}

// 读取一段 token 的 PLE 向量，并统计走了多少次缓存 / 读了多少次表。
struct Pass {
    std::vector<float> vector;        // token-major，每 token 2560
    std::uint64_t cache_hits = 0;
    std::uint64_t table_reads = 0;
    std::uint64_t admitted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t overspend = 0;
};

Pass read_pass(const std::vector<std::int32_t>& seq, storage::RowCache& cache, std::uint64_t budget_bytes) {
    const std::vector<std::int32_t> prev = build_prev(seq);
    std::vector<std::uint32_t> rows(seq.size() * kNHeads);
    kernels::ngram_rows(seq.data(), prev.data(), static_cast<int>(seq.size()),
                        kernels::ple_artifact_consts(), rows.data());

    Pass out;
    out.vector.assign(seq.size() * kVector, 0.0f);
    std::vector<std::uint8_t> row_buf(kernels::kIq4nlRowBytes);

    for (std::size_t t = 0; t < seq.size(); ++t) {
        // 一个 token 的 16 行 = 16 个必需的表行请求，竞争同步的 PCIe 字节预算。
        std::vector<scheduling::Request> reqs;
        reqs.reserve(kNHeads);
        for (int h = 0; h < kNHeads; ++h) {
            scheduling::Request r;
            r.cls = scheduling::FlowClass::kTableRow;
            r.bytes = kernels::kIq4nlRowBytes;
            r.consume_at_step = static_cast<std::int64_t>(t);
            r.must = true;
            r.tag = static_cast<std::uint32_t>(h);
            reqs.push_back(r);
        }
        const scheduling::Result res = scheduling::arbitrate(budget_bytes, reqs);
        out.admitted += res.admit_order.size();
        out.dropped += res.dropped;
        out.overspend += res.overspend_bytes;
        float dequant[kNHeads][kHeadDim];
        for (int h = 0; h < kNHeads; ++h) {
            const std::uint32_t row = rows[t * kNHeads + h];
            const std::uint8_t* p = nullptr;
            if (cache.find(row, &p)) {
                ++out.cache_hits;
            } else {
                row_buf = synth_row(row);      // 替身表：这里本应是一次真实读表
                cache.insert(row, row_buf.data());
                ++out.table_reads;
                const bool now_resident = cache.find(row, &p);
                CHECK(now_resident);
            }
            kernels::dequant_iq4nl_row(p, dequant[h]);
        }
        kernels::assemble_ple_vector(dequant, out.vector.data() + t * kVector);
    }
    return out;
}

void test_single_request_read_path() {
    const std::vector<std::int32_t> seq{92989, 249574, 135030, 248044, 0};
    const std::vector<std::int32_t> prev = build_prev(seq);
    std::vector<std::uint32_t> rows(seq.size() * kNHeads);
    kernels::ngram_rows(seq.data(), prev.data(), static_cast<int>(seq.size()),
                        kernels::ple_artifact_consts(), rows.data());

    storage::RowCache cache(1u << 20);        // 足够大：这一遍全冷
    const Pass p = read_pass(seq, cache, /*budget_bytes=*/1u << 20);

    CHECK(p.admitted == seq.size() * kNHeads);
    CHECK(p.dropped == 0);
    CHECK(p.overspend == 0);                         // 预算充足：不超支
    CHECK(p.table_reads == seq.size() * kNHeads);    // 全冷：每行读一次表
    CHECK(p.cache_hits == 0);

    // 每个头拼到自己的那一段，值是 d·cb[0]（零半字节），且各头不同。
    for (std::size_t t = 0; t < seq.size(); ++t) {
        for (int h = 0; h < kNHeads; ++h) {
            const float want = row_value(rows[t * kNHeads + h]);
            for (int k = 0; k < kHeadDim; ++k) {
                const float got = p.vector[t * kVector + h * kHeadDim + k];
                CHECK(std::fabs(got - want) <= 1e-6f);
            }
        }
    }
}

void test_second_pass_is_all_cache_hits() {
    const std::vector<std::int32_t> seq{11, 22, 33, 44, 55, 66, 77, 88};
    storage::RowCache cache(1u << 20);

    const Pass first = read_pass(seq, cache, 1u << 20);
    const Pass second = read_pass(seq, cache, 1u << 20);

    CHECK(first.table_reads == seq.size() * kNHeads && first.cache_hits == 0);
    CHECK(first.overspend == 0);
    CHECK(second.table_reads == 0);                       // 第二遍全命中
    CHECK(second.cache_hits == seq.size() * kNHeads);
    CHECK(second.vector == first.vector);                 // 缓存不改变结果
}

void test_tight_budget_still_admits_every_required_row() {
    // 预算小到一个 token 的 16 行装不下：必需项仍全部准入（记 overspend），不得丢表行。
    const std::vector<std::int32_t> seq{7, 8, 9};
    storage::RowCache cache(64);
    const std::vector<std::int32_t> prev = build_prev(seq);
    std::vector<std::uint32_t> rows(seq.size() * kNHeads);
    kernels::ngram_rows(seq.data(), prev.data(), static_cast<int>(seq.size()),
                        kernels::ple_artifact_consts(), rows.data());

    std::vector<scheduling::Request> reqs;
    for (int h = 0; h < kNHeads; ++h) {
        scheduling::Request r;
        r.cls = scheduling::FlowClass::kTableRow;
        r.bytes = kernels::kIq4nlRowBytes;
        r.must = true;
        r.tag = static_cast<std::uint32_t>(h);
        reqs.push_back(r);
    }
    const scheduling::Result res = scheduling::arbitrate(1 /*字节*/, reqs);
    CHECK(res.dropped == 0);
    CHECK(res.admit_order.size() == static_cast<std::size_t>(kNHeads));
    CHECK(res.overspend_bytes == kNHeads * kernels::kIq4nlRowBytes - 1);

    // 用这个紧预算跑一遍整条路径，结果必须与充足预算一致。
    storage::RowCache c1(64), c2(64);
    CHECK(read_pass(seq, c1, 1).vector == read_pass(seq, c2, 1u << 20).vector);
}

}  // namespace

int main() {
    test_single_request_read_path();
    test_second_pass_is_all_cache_hits();
    test_tight_budget_still_admits_every_required_row();
    std::puts("ple_read_path: token -> row -> cache -> dequant -> 2560 floats");
    return 0;
}
