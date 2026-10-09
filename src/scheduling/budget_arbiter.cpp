#include "scheduling/budget_arbiter.hpp"

#include <algorithm>

namespace qinfer::scheduling {

const char* to_string(FlowClass c) {
    switch (c) {
        case FlowClass::kKv: return "kv";
        case FlowClass::kTableRow: return "table-row";
        case FlowClass::kExpert: return "expert";
    }
    return "?";
}

namespace {

int must_rank(FlowClass c) {
    switch (c) {
        case FlowClass::kKv: return 0;
        case FlowClass::kTableRow: return 1;
        case FlowClass::kExpert: return 2;
    }
    return 3;
}

}  // namespace

Result arbitrate(std::uint64_t budget_bytes, const std::vector<Request>& requests) {
    Result res;
    res.decisions.resize(requests.size());
    std::vector<std::size_t> must_idx, pred_idx;
    must_idx.reserve(requests.size());
    pred_idx.reserve(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        res.decisions[i].tag = requests[i].tag;
        res.decisions[i].bytes = requests[i].bytes;
        res.decisions[i].verdict = Verdict::kDropped;
        (requests[i].must ? must_idx : pred_idx).push_back(i);
    }

    // 必需项：类别优先（KV → 表行 → 专家），同类按输入顺序。一律准入；超出预算的部分记 overspend。
    std::stable_sort(must_idx.begin(), must_idx.end(), [&](std::size_t a, std::size_t b) {
        return must_rank(requests[a].cls) < must_rank(requests[b].cls);
    });
    std::uint64_t remaining = budget_bytes;
    std::uint64_t must_bytes = 0;
    for (std::size_t i : must_idx) {
        const Request& r = requests[i];
        res.decisions[i].verdict = Verdict::kAdmitted;
        res.admit_order.push_back(r.tag);
        res.admitted_bytes += r.bytes;
        must_bytes += r.bytes;
        remaining = r.bytes > remaining ? 0 : remaining - r.bytes;
    }
    res.overspend_bytes = must_bytes > budget_bytes ? must_bytes - budget_bytes : 0;

    // 预测性：收益 ÷ 字节降序（同值按输入顺序）。放不下就弃，但继续尝试后面的。
    std::stable_sort(pred_idx.begin(), pred_idx.end(), [&](std::size_t a, std::size_t b) {
        return requests[a].benefit_per_byte > requests[b].benefit_per_byte;
    });
    for (std::size_t i : pred_idx) {
        const Request& r = requests[i];
        if (r.bytes <= remaining) {
            res.decisions[i].verdict = Verdict::kAdmitted;
            res.admit_order.push_back(r.tag);
            res.admitted_bytes += r.bytes;
            remaining -= r.bytes;
        } else {
            ++res.dropped;
            res.dropped_bytes += r.bytes;
        }
    }
    return res;
}

}  // namespace qinfer::scheduling
