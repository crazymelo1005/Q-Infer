// 每步观测的合成（interfaces.md 第 7 节）。不引第三方框架。
//
// 要点：缓存侧取的是**本步增量**而不是累计量（累计命中率会把波动平滑掉，看不出这一步为什么慢）；
// 本步没有访问时命中率记 0 而不是 NaN；容器换了实例（计数倒退）时按 0 记而不是负数。
#include "experts/step_record.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace qinfer;
using namespace qinfer::experts;

namespace {

scheduling::StepObservation budget_obs(std::uint64_t budget, std::uint64_t admitted,
                                       std::uint64_t overspend, std::uint32_t late,
                                       std::uint64_t late_bytes) {
    scheduling::StepObservation o;
    o.budget_bytes = budget;
    o.admitted_bytes = admitted;
    o.overspend_bytes = overspend;
    o.late_requests = late;
    o.late_bytes = late_bytes;
    o.budget_used = budget > 0 ? static_cast<double>(admitted) / static_cast<double>(budget) : 0.0;
    return o;
}

void test_deltas_are_per_step() {
    CacheSnapshot prev;
    prev.hits = 100;
    prev.misses = 20;
    prev.replaced = 5;
    prev.blocked = 1;
    prev.transient_reads = 2;
    prev.residents = 7;
    prev.bytes_resident = 7000;
    prev.page_entries = 7;

    CacheSnapshot now = prev;
    now.hits = 140;          // 本步 40 次命中
    now.misses = 24;         // 本步 4 次未命中
    now.replaced = 8;        // 本步 3 次换出
    now.blocked = 2;         // 本步 1 次被拒
    now.transient_reads = 5; // 本步 3 次临时读
    now.residents = 9;
    now.bytes_resident = 9000;
    now.page_entries = 9;

    const StepRecord r = make_step_record(12, budget_obs(1000, 900, 0, 2, 50), prev, now);
    CHECK(r.step == 12);
    CHECK(r.cache_hits == 40 && r.cache_misses == 4);
    CHECK(r.cache_evictions == 3 && r.cache_blocked == 1 && r.cache_transient == 3);
    CHECK(r.residents == 9 && r.bytes_resident == 9000 && r.page_entries == 9);
    // 命中率是**本步**的 40/44，不是累计的 140/164。
    CHECK(std::fabs(r.hit_rate - 40.0 / 44.0) < 1e-12);
    CHECK(std::fabs(r.hit_rate - 140.0 / 164.0) > 1e-3);  // 与累计口径确实不同
    // 预算侧原样带过来。
    CHECK(r.budget_bytes == 1000 && r.admitted_bytes == 900 && r.overspend_bytes == 0);
    CHECK(r.late_requests == 2 && r.late_bytes == 50);
    CHECK(std::fabs(r.budget_used - 0.9) < 1e-12);

    // 一行格式：字段名都在，且能看出是本步口径的那些数。
    const std::string line = format_step_record(r);
    CHECK(line.find("step=12") != std::string::npos);
    CHECK(line.find("cache_hits=40") != std::string::npos);
    CHECK(line.find("hit_rate=") != std::string::npos);
    CHECK(line.find("residents=9") != std::string::npos);
    CHECK(line.find(' ') != std::string::npos);
}

void test_idle_step_records_zero_not_nan() {
    CacheSnapshot snap;
    snap.hits = 33;
    snap.misses = 7;
    const StepRecord r = make_step_record(3, budget_obs(0, 0, 0, 0, 0), snap, snap);
    CHECK(r.cache_hits == 0 && r.cache_misses == 0);
    CHECK(r.hit_rate == 0.0);                 // 不是 NaN
    CHECK(r.hit_rate == r.hit_rate);          // 顺带钉住非 NaN
    CHECK(r.budget_used == 0.0);              // 预算为 0 也记 0
    CHECK(std::isnan(r.hit_rate) == false);
}

void test_counters_going_backwards_clamp_to_zero() {
    // 换了缓存实例（或计数被重置）时，增量按 0 记——负数会让下游统计失真。
    CacheSnapshot prev;
    prev.hits = 500;
    prev.misses = 100;
    prev.replaced = 50;
    CacheSnapshot now;  // 全 0
    const StepRecord r = make_step_record(1, budget_obs(100, 10, 0, 0, 0), prev, now);
    CHECK(r.cache_hits == 0 && r.cache_misses == 0 && r.cache_evictions == 0);
    CHECK(r.hit_rate == 0.0);
}

}  // namespace

int main() {
    test_deltas_are_per_step();
    test_idle_step_records_zero_not_nan();
    test_counters_going_backwards_clamp_to_zero();
    std::puts("step_record: 本步增量、本步命中率、空闲步记 0、计数倒退按 0 记");
    return 0;
}