#include "ple/row_path.hpp"

#include "kernels/ple_gather.hpp"
#include "scheduling/budget_arbiter.hpp"

namespace qinfer::ple {

std::vector<std::int32_t> build_prev(const std::int32_t* tokens, int n_tokens) {
    std::vector<std::int32_t> prev(static_cast<std::size_t>(n_tokens) * 2);
    for (int i = 0; i < n_tokens; ++i) {
        prev[static_cast<std::size_t>(i) * 2] = i >= 2 ? tokens[i - 2] : kernels::kTokenNull;
        prev[static_cast<std::size_t>(i) * 2 + 1] = i >= 1 ? tokens[i - 1] : kernels::kTokenNull;
    }
    return prev;
}

bool ple_vector_with_prev(const RowSource& table, std::int32_t token, std::int32_t prev0,
                          std::int32_t prev1, storage::RowCache& cache,
                          std::uint64_t budget_bytes, float* out, PathStats& stats,
                          std::string& err) {
    stats = PathStats{};
    if (out == nullptr) {
        err = "参数为空";
        return false;
    }
    const std::int32_t toks[1] = {token};
    const std::int32_t prev[2] = {prev0, prev1};
    std::uint32_t rows[kNHeads];
    kernels::ngram_rows(toks, prev, 1, kernels::ple_artifact_consts(), rows);

    // 一个 token 的 16 行 = 16 个必需请求，竞争这一批同步的 PCIe 字节预算。
    std::vector<scheduling::Request> reqs;
    reqs.reserve(kNHeads);
    for (int h = 0; h < kNHeads; ++h) {
        scheduling::Request r;
        r.cls = scheduling::FlowClass::kTableRow;
        r.bytes = kernels::kIq4nlRowBytes;
        r.consume_at_step = 0;
        r.must = true;
        r.tag = static_cast<std::uint32_t>(h);
        reqs.push_back(r);
    }
    const scheduling::Result res = scheduling::arbitrate(budget_bytes, reqs);
    stats.admitted += res.admit_order.size();
    stats.dropped += res.dropped;
    stats.overspend_bytes += res.overspend_bytes;

    std::vector<std::uint8_t> row_buf(kernels::kIq4nlRowBytes);
    float dequant[kNHeads][kHeadDim];
    for (int h = 0; h < kNHeads; ++h) {
        const std::uint32_t row = rows[h];
        const std::uint8_t* p = nullptr;
        if (cache.find(row, &p)) {
            ++stats.cache_hits;
        } else {
            // 行号越界是与文件数据的边界，必须挡住：否则会读到别的行或越界。
            if (row >= table.rows()) {
                err = "行号 " + std::to_string(row) + " 超出表范围（" + std::to_string(table.rows()) +
                      " 行）";
                return false;
            }
            if (!table.read_row(row, row_buf.data())) {
                err = "读表行 " + std::to_string(row) + " 失败";
                return false;
            }
            cache.insert(row, row_buf.data());
            ++stats.table_reads;
            if (!cache.find(row, &p)) {
                err = "插入缓存后仍取不到行 " + std::to_string(row);
                return false;
            }
        }
        kernels::dequant_iq4nl_row(p, dequant[h]);
    }
    kernels::assemble_ple_vector(dequant, out);
    return true;
}

bool ple_vectors_for_tokens(const RowSource& table, const std::int32_t* tokens, int n_tokens,
                            storage::RowCache& cache, std::uint64_t budget_bytes, float* out,
                            PathStats& stats, std::string& err) {
    stats = PathStats{};
    if (tokens == nullptr || out == nullptr || n_tokens <= 0) {
        err = "参数为空";
        return false;
    }
    // 逐 token 走同一条核（`ple_vector_with_prev`），前驱由序列自己推：这样「整段」与「逐 token 喂」
    // 是同一份算术，两者可以互相对照。
    const std::vector<std::int32_t> prev = build_prev(tokens, n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        PathStats one;
        if (!ple_vector_with_prev(table, tokens[t], prev[static_cast<std::size_t>(t) * 2],
                                  prev[static_cast<std::size_t>(t) * 2 + 1], cache, budget_bytes,
                                  out + static_cast<std::size_t>(t) * kVectorDim, one, err)) {
            err = "第 " + std::to_string(t) + " 个 token：" + err;
            return false;
        }
        stats.cache_hits += one.cache_hits;
        stats.table_reads += one.table_reads;
        stats.admitted += one.admitted;
        stats.dropped += one.dropped;
        stats.overspend_bytes += one.overspend_bytes;
    }
    return true;
}

}  // namespace qinfer::ple