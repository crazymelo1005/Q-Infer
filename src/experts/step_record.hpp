// 每步观测的一行记录：把预算侧与缓存侧合到一起（interfaces.md 第 7 节）。
//
// 为什么两半要合：§7 说「每步至少记录：命中率、PCIe 实际用量、预取命中数与迟到数、各层占用、两个
// KV 池的余量、每步耗时」，并说这些观测是「调参与复核的唯一依据」。预算那一半在
// `scheduling/step_budget`（现已实现：预算用量、超支、迟到），缓存那一半在 `storage/expert_cache`
// 的累计计数里——累计量不能直接当每步的观测，故这里做两件事：把累计计数折成**本步增量**，再与预算
// 观测拼成一条记录。
//
// 命中率是**本步**的比率，不是累计比率：累计命中率会随运行时间平滑掉波动，看它看不出「这一步为什么
// 慢」。这一步没有访问时记 0，不记 NaN。
#pragma once

#include <cstdint>
#include <string>

#include "experts/expert_source.hpp"
#include "scheduling/step_budget.hpp"

namespace qinfer::experts {

// 缓存侧的一次快照（累计量）。每步开始记一次，结束时再记一次，差就是本步的增量。
struct CacheSnapshot {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t compulsory = 0;
    std::uint64_t replaced = 0;
    std::uint64_t blocked = 0;
    std::uint64_t transient_reads = 0;
    std::uint64_t residents = 0;
    std::uint64_t bytes_resident = 0;
    std::size_t page_entries = 0;

    static CacheSnapshot of(const ExpertSource& src);
};

struct StepRecord {
    std::int64_t step = 0;

    // 预算侧（直接来自 scheduling 的观测）。全 0 表示这一步没有走预算仲裁——例如只核缓存
    // 接线的手工入口，那里不做仲裁。
    std::uint64_t budget_bytes = 0;
    std::uint64_t admitted_bytes = 0;
    std::uint64_t overspend_bytes = 0;
    std::uint32_t late_requests = 0;
    std::uint64_t late_bytes = 0;
    double budget_used = 0.0;

    // 缓存侧：本步增量 + 步末占用
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_evictions = 0;
    std::uint64_t cache_blocked = 0;
    std::uint64_t cache_transient = 0;
    std::uint64_t residents = 0;
    std::uint64_t bytes_resident = 0;
    std::size_t page_entries = 0;
    double hit_rate = 0.0;  // 本步 hits ÷ (hits + misses)；本步无访问时记 0
};

// now 与 prev 都是累计量；本步增量 = now − prev（若某个计数反而变小，按 0 记——那是换了缓存实例）。
StepRecord make_step_record(std::int64_t step, const scheduling::StepObservation& budget,
                            const CacheSnapshot& prev, const CacheSnapshot& now);

// 一行、字段名=值、无空格歧义，便于落盘与事后解析。
std::string format_step_record(const StepRecord& r);

}  // namespace qinfer::experts