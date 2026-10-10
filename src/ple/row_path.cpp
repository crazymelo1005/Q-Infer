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

bool ple_vectors_for_tokens(const RowSource& table, const std::int32_t* tokens, int n_tokens,
                            storage::RowCache& cache, std::uint64_t budget_bytes, float* out,
                            PathStats& stats, std::string& err) {
    stats = PathStats{};
    if (tokens == nullptr || out == nullptr || n_tokens <= 0) {
        err = "参数为空";
        return false;
    }

    const std::vector<std::int32_t> prev = build_prev(tokens, n_tokens);
    std::vector<std::uint32_t> rows(static_cast<std::size_t>(n_tokens) * kNHeads);
    kernels::ngram_rows(tokens, prev.data(), n_tokens, kernels::ple_artifact_consts(), rows.data());

    std::vector<std::uint8_t> row_buf(kernels::kIq4nlRowBytes);
    for (int t = 0; t < n_tokens; ++t) {
        // 一个 token 的 16 行 = 16 个必需请求，竞争这一批同步的 PCIe 字节预算。
        std::vector<scheduling::Request> reqs;
        reqs.reserve(kNHeads);
        for (int h = 0; h < kNHeads; ++h) {
            scheduling::Request r;
            r.cls = scheduling::FlowClass::kTableRow;
            r.bytes = kernels::kIq4nlRowBytes;
            r.consume_at_step = t;
            r.must = true;
            r.tag = static_cast<std::uint32_t>(h);
            reqs.push_back(r);
        }
        const scheduling::Result res = scheduling::arbitrate(budget_bytes, reqs);
        stats.admitted += res.admit_order.size();
        stats.dropped += res.dropped;
        stats.overspend_bytes += res.overspend_bytes;

        float dequant[kNHeads][kHeadDim];
        for (int h = 0; h < kNHeads; ++h) {
            const std::uint32_t row = rows[static_cast<std::size_t>(t) * kNHeads + h];
            const std::uint8_t* p = nullptr;
            if (cache.find(row, &p)) {
                ++stats.cache_hits;
            } else {
                // 行号越界是与文件数据的边界，必须挡住：否则会读到别的行或越界。
                if (row >= table.rows()) {
                    err = "第 " + std::to_string(t) + " 个 token 的第 " + std::to_string(h) +
                          " 个行号 " + std::to_string(row) + " 超出表范围（" +
                          std::to_string(table.rows()) + " 行）";
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
        kernels::assemble_ple_vector(dequant, out + static_cast<std::size_t>(t) * kVectorDim);
    }
    return true;
}

}  // namespace qinfer::ple