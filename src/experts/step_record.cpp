#include "experts/step_record.hpp"

#include <cstdio>

namespace qinfer::experts {

namespace {

std::uint64_t delta(std::uint64_t now, std::uint64_t prev) { return now >= prev ? now - prev : 0; }

}  // namespace

CacheSnapshot CacheSnapshot::of(const ExpertSource& src) {
    const storage::CacheStats& st = src.stats();
    CacheSnapshot s;
    s.hits = st.hits;
    s.misses = st.misses;
    s.compulsory = st.compulsory;
    s.replaced = st.replaced;
    s.blocked = st.blocked;
    s.transient_reads = src.transient_reads();
    s.residents = src.residents();
    s.bytes_resident = src.bytes_resident();
    s.page_entries = src.page_entries();
    return s;
}

StepRecord make_step_record(std::int64_t step, const scheduling::StepObservation& budget,
                            const CacheSnapshot& prev, const CacheSnapshot& now) {
    StepRecord r;
    r.step = step;
    r.budget_bytes = budget.budget_bytes;
    r.admitted_bytes = budget.admitted_bytes;
    r.overspend_bytes = budget.overspend_bytes;
    r.late_requests = budget.late_requests;
    r.late_bytes = budget.late_bytes;
    r.budget_used = budget.budget_used;

    r.cache_hits = delta(now.hits, prev.hits);
    r.cache_misses = delta(now.misses, prev.misses);
    r.cache_evictions = delta(now.replaced, prev.replaced);
    r.cache_blocked = delta(now.blocked, prev.blocked);
    r.cache_transient = delta(now.transient_reads, prev.transient_reads);
    r.residents = now.residents;
    r.bytes_resident = now.bytes_resident;
    r.page_entries = now.page_entries;

    const std::uint64_t accesses = r.cache_hits + r.cache_misses;
    r.hit_rate = accesses == 0 ? 0.0 : static_cast<double>(r.cache_hits) / static_cast<double>(accesses);
    return r;
}

std::string format_step_record(const StepRecord& r) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "step=%lld budget_bytes=%llu admitted_bytes=%llu overspend_bytes=%llu "
                  "late_requests=%u late_bytes=%llu budget_used=%.4f "
                  "cache_hits=%llu cache_misses=%llu hit_rate=%.4f evictions=%llu blocked=%llu "
                  "transient=%llu residents=%llu bytes_resident=%llu page_entries=%zu",
                  static_cast<long long>(r.step),
                  static_cast<unsigned long long>(r.budget_bytes),
                  static_cast<unsigned long long>(r.admitted_bytes),
                  static_cast<unsigned long long>(r.overspend_bytes), r.late_requests,
                  static_cast<unsigned long long>(r.late_bytes), r.budget_used,
                  static_cast<unsigned long long>(r.cache_hits),
                  static_cast<unsigned long long>(r.cache_misses), r.hit_rate,
                  static_cast<unsigned long long>(r.cache_evictions),
                  static_cast<unsigned long long>(r.cache_blocked),
                  static_cast<unsigned long long>(r.cache_transient),
                  static_cast<unsigned long long>(r.residents),
                  static_cast<unsigned long long>(r.bytes_resident), r.page_entries);
    return std::string(buf);
}

}  // namespace qinfer::experts