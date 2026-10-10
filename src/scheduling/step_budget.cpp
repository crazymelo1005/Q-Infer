#include "scheduling/step_budget.hpp"

#include <cmath>

namespace qinfer::scheduling {

namespace {

// 画像把 GB 当 1024 MiB 换算（formulas 字段的 ×1024），故这里用 GiB 的字节数：1 GiB = 1024^3。
constexpr double kBytesPerGibi = 1024.0 * 1024.0 * 1024.0;

}  // namespace

std::uint64_t step_budget_bytes(const ProfileInputs& in) {
    if (!(in.pcie_h2d_gbps > 0.0) || !(in.step_ms > 0.0) || !(in.io_share > 0.0)) return 0;
    const double gib = in.pcie_h2d_gbps * (in.step_ms / 1000.0) * in.io_share;
    return static_cast<std::uint64_t>(std::llround(gib * kBytesPerGibi));
}

std::uint64_t table_demand_bytes(const TableDemand& d) {
    if (d.rows_per_token == 0 || d.row_bytes == 0 || !(d.positions_per_token > 0.0)) return 0;
    const double bytes = static_cast<double>(d.rows_per_token) * static_cast<double>(d.row_bytes) *
                         d.positions_per_token;
    return static_cast<std::uint64_t>(std::llround(bytes));
}

std::vector<Request> build_requests(const StepDemand& d) {
    std::vector<Request> reqs;
    reqs.reserve(2 + d.experts.size());

    if (d.kv_bytes > 0) {
        Request r;
        r.cls = FlowClass::kKv;
        r.bytes = d.kv_bytes;
        r.consume_at_step = d.step;
        r.must = true;
        r.tag = 0;
        reqs.push_back(r);
    }

    const std::uint64_t table_bytes = table_demand_bytes(d.table) * d.table_tokens;
    if (table_bytes > 0) {
        Request r;
        r.cls = FlowClass::kTableRow;
        r.bytes = table_bytes;
        r.consume_at_step = d.step;
        r.must = true;
        r.tag = 0;
        reqs.push_back(r);
    }

    for (const ExpertCandidate& e : d.experts) {
        if (e.bytes == 0) continue;  // 字节为 0 的候选没有意义，也不该占一个决策位
        Request r;
        r.cls = FlowClass::kExpert;
        r.bytes = e.bytes;
        r.consume_at_step = d.step;
        r.must = false;
        r.benefit_per_byte = e.probability / static_cast<double>(e.bytes);
        r.tag = e.id;
        reqs.push_back(r);
    }
    return reqs;
}

StepObservation observe(const Result& r, std::uint64_t budget_bytes, std::int64_t step) {
    StepObservation o;
    o.step = step;
    o.budget_bytes = budget_bytes;
    o.admitted_bytes = r.admitted_bytes;
    o.overspend_bytes = r.overspend_bytes;
    o.late_requests = r.dropped;
    o.late_bytes = r.dropped_bytes;
    for (const Decision& d : r.decisions) {
        if (d.verdict != Verdict::kAdmitted) continue;
        switch (d.cls) {
            case FlowClass::kKv: ++o.kv_admitted; break;
            case FlowClass::kTableRow: ++o.table_rows_admitted; break;
            case FlowClass::kExpert: ++o.experts_admitted; break;
        }
    }
    o.budget_used = budget_bytes > 0 ? static_cast<double>(o.admitted_bytes) /
                                           static_cast<double>(budget_bytes)
                                     : 0.0;
    return o;
}

}  // namespace qinfer::scheduling