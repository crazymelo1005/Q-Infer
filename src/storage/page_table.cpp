#include "storage/page_table.hpp"

#include <algorithm>
#include <limits>

namespace qinfer::storage {

const char* to_string(BlockClass c) {
    switch (c) {
        case BlockClass::kExpert: return "expert";
        case BlockClass::kTableRow: return "table-row";
        case BlockClass::kKv: return "kv";
    }
    return "?";
}

const char* to_string(Tier t) {
    switch (t) {
        case Tier::kVram: return "vram";
        case Tier::kHostPinned: return "host-pinned";
        case Tier::kNvme: return "nvme";
    }
    return "?";
}

const char* to_string(PrefetchState s) {
    switch (s) {
        case PrefetchState::kIdle: return "idle";
        case PrefetchState::kInflight: return "inflight";
        case PrefetchState::kReady: return "ready";
    }
    return "?";
}

bool PageTable::admit(BlockId id, BlockClass cls, Tier tier, std::uint64_t bytes, std::uint64_t offset) {
    if (bytes == 0) return false;
    Entry e;
    e.id = id;
    e.cls = cls;
    e.tier = tier;
    e.bytes = bytes;
    e.offset = offset;
    e.generation = generation_;
    return entries_.emplace(id, e).second;   // 已存在即 false：同一逻辑块只有一份条目
}

bool PageTable::acquire(BlockId id, std::int64_t deadline_steps, bool must) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    Entry& e = it->second;
    ++e.refcount;
    e.deadline_steps = deadline_steps;
    e.must = must;
    if (e.prefetch == PrefetchState::kInflight) e.prefetch = PrefetchState::kReady;
    return true;
}

bool PageTable::release(BlockId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    if (it->second.refcount == 0) return false;   // 不允许静默下溢
    --it->second.refcount;
    return true;
}

void PageTable::advance_generation() { ++generation_; }

const Entry* PageTable::query(BlockId id) const {
    auto it = entries_.find(id);
    return it == entries_.end() ? nullptr : &it->second;
}

bool PageTable::set_tier(BlockId id, Tier tier, std::uint64_t offset) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    it->second.tier = tier;
    it->second.offset = offset;
    return true;
}

bool PageTable::set_prefetch(BlockId id, PrefetchState state) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    it->second.prefetch = state;
    return true;
}

bool PageTable::set_heat(BlockId id, std::uint32_t heat) {
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    it->second.heat = heat;
    return true;
}

bool PageTable::evictable(BlockId id) const {
    const Entry* e = query(id);
    if (e == nullptr) return false;
    return e->refcount == 0 && e->prefetch != PrefetchState::kInflight && e->generation != generation_;
}

std::optional<BlockId> PageTable::evict_one() {
    const BlockId* best = nullptr;
    const Entry* best_e = nullptr;
    for (const auto& [id, e] : entries_) {
        if (e.refcount != 0) continue;                    // 不变量 2
        if (e.prefetch == PrefetchState::kInflight) continue;  // 不变量 1
        if (e.generation == generation_) continue;        // 本世代刚搬进来的不动
        if (best_e == nullptr || e.heat < best_e->heat ||
            (e.heat == best_e->heat &&
             (e.generation < best_e->generation ||
              (e.generation == best_e->generation && id.value < best->value)))) {
            best = &id;
            best_e = &e;
        }
    }
    if (best == nullptr) return std::nullopt;
    BlockId victim = *best;
    entries_.erase(victim);
    return victim;
}

std::uint64_t PageTable::bytes_in(Tier tier) const {
    std::uint64_t total = 0;
    for (const auto& [id, e] : entries_) {
        if (e.tier == tier) total += e.bytes;
    }
    return total;
}

}  // namespace qinfer::storage
