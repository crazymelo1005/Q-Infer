#include "storage/expert_cache.hpp"

#include <algorithm>
#include <numeric>

namespace qinfer::storage {

namespace {

// 键的稳定序：层在前、专家在后。用于所有并列的确定性打破。
bool key_less(ExpertKey a, ExpertKey b) {
    return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
}

bool in_set(const ExpertKey* keys, int n, ExpertKey k) {
    for (int i = 0; i < n; ++i) {
        if (keys[i] == k) return true;
    }
    return false;
}

}  // namespace

int FifoPolicy::rank(const SlotState* slots, int ways, const ExpertKey&, const PolicyContext&,
                     int* out_order) const {
    std::iota(out_order, out_order + ways, 0);
    std::stable_sort(out_order, out_order + ways, [&](int a, int b) {
        if (slots[a].inserted != slots[b].inserted) return slots[a].inserted < slots[b].inserted;
        return key_less(slots[a].key, slots[b].key);
    });
    return ways;
}

int LruPolicy::rank(const SlotState* slots, int ways, const ExpertKey&, const PolicyContext&,
                    int* out_order) const {
    std::iota(out_order, out_order + ways, 0);
    std::stable_sort(out_order, out_order + ways, [&](int a, int b) {
        if (slots[a].last_used != slots[b].last_used) return slots[a].last_used < slots[b].last_used;
        if (slots[a].inserted != slots[b].inserted) return slots[a].inserted < slots[b].inserted;
        return key_less(slots[a].key, slots[b].key);
    });
    return ways;
}

int CooccurrenceAwarePolicy::rank(const SlotState* slots, int ways, const ExpertKey&,
                                  const PolicyContext& ctx, int* out_order) const {
    const Cooccurrence* cooc = ctx.cooc;
    std::iota(out_order, out_order + ways, 0);
    std::stable_sort(out_order, out_order + ways, [&](int a, int b) {
        // 本步正在读的键排在最后：它们现在就要用，换掉等于自伤。
        const bool na = in_set(ctx.step_keys, ctx.step_keys_n, slots[a].key);
        const bool nb = in_set(ctx.step_keys, ctx.step_keys_n, slots[b].key);
        if (na != nb) return !na;
        // 共现最弱的先换。
        const std::uint64_t sa = cooc != nullptr
                                     ? cooc->score_with(slots[a].key, ctx.step_keys, ctx.step_keys_n)
                                     : 0;
        const std::uint64_t sb = cooc != nullptr
                                     ? cooc->score_with(slots[b].key, ctx.step_keys, ctx.step_keys_n)
                                     : 0;
        if (sa != sb) return sa < sb;
        // 并列退回 LRU，再并列取键更小的。
        if (slots[a].last_used != slots[b].last_used) return slots[a].last_used < slots[b].last_used;
        return key_less(slots[a].key, slots[b].key);
    });
    return ways;
}

std::uint64_t ExpertCache::mix(ExpertKey k) {
    const std::uint64_t x = static_cast<std::uint64_t>(key_code(k)) * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
}

ExpertCache::ExpertCache(std::size_t capacity_slots, int ways, const EvictionPolicy& policy,
                         std::size_t max_cooc_pairs)
    : ways_(ways < 1 ? 1 : ways), policy_(&policy), cooc_(max_cooc_pairs) {
    sets_ = capacity_slots / static_cast<std::size_t>(ways_);  // 不足一组的尾巴丢掉
    slots_.assign(sets_ * static_cast<std::size_t>(ways_), SlotState{});
}

const char* to_string(PolicyKind k) {
    switch (k) {
        case PolicyKind::kFifo: return "fifo";
        case PolicyKind::kLru: return "lru";
        case PolicyKind::kCooccurrence: return "cooccurrence-aware";
    }
    return "?";
}

ExpertCache::ExpertCache(std::size_t capacity_slots, int ways, PolicyKind kind,
                         std::size_t max_cooc_pairs)
    : ways_(ways < 1 ? 1 : ways), cooc_(max_cooc_pairs) {
    switch (kind) {
        case PolicyKind::kFifo: owned_policy_ = std::make_unique<FifoPolicy>(); break;
        case PolicyKind::kLru: owned_policy_ = std::make_unique<LruPolicy>(); break;
        case PolicyKind::kCooccurrence:
            owned_policy_ = std::make_unique<CooccurrenceAwarePolicy>();
            break;
    }
    policy_ = owned_policy_.get();
    sets_ = capacity_slots / static_cast<std::size_t>(ways_);  // 不足一组的尾巴丢掉
    slots_.assign(sets_ * static_cast<std::size_t>(ways_), SlotState{});
}

int ExpertCache::probe(const SlotState* set, const ExpertKey& k) const {
    for (int w = 0; w < ways_; ++w) {
        if (set[w].occupied && set[w].key == k) return w;
    }
    return -1;
}

ExpertCache::Outcome ExpertCache::access(const ExpertKey& k, std::int64_t step,
                                         EvictableFn evictable, void* ctx, ExpertKey* victim_out) {
    if (sets_ == 0) {
        ++stats_.misses;
        return Outcome::kMissBlocked;  // 缓存关闭：一律不占槽位
    }
    SlotState* set = &slots_[static_cast<std::size_t>(mix(k) % sets_) * static_cast<std::size_t>(ways_)];

    const int hit = probe(set, k);
    if (hit >= 0) {
        set[hit].last_used = step;
        ++stats_.hits;
        return Outcome::kHit;
    }

    ++stats_.misses;
    for (int w = 0; w < ways_; ++w) {
        if (!set[w].occupied) {
            set[w] = SlotState{k, step, step, true};
            ++used_;
            ++stats_.compulsory;
            return Outcome::kMissFree;
        }
    }

    // 组满了：按策略给的顺序找第一个当前可驱逐的槽位。
    std::vector<int> order(static_cast<std::size_t>(ways_));
    PolicyContext pc;
    pc.step = step;
    pc.step_keys = step_keys_.empty() ? nullptr : step_keys_.data();
    pc.step_keys_n = static_cast<int>(step_keys_.size());
    pc.cooc = &cooc_;
    policy_->rank(set, ways_, k, pc, order.data());
    for (int idx = 0; idx < ways_; ++idx) {
        const int w = order[static_cast<std::size_t>(idx)];
        if (evictable != nullptr && !evictable(set[w].key, ctx)) continue;
        if (victim_out != nullptr) *victim_out = set[w].key;
        set[w] = SlotState{k, step, step, true};
        ++stats_.replaced;
        return Outcome::kMissReplaced;
    }
    ++stats_.blocked;
    return Outcome::kMissBlocked;
}

void ExpertCache::observe_step(const ExpertKey* keys, int n) {
    step_keys_.assign(keys, keys + (n > 0 ? n : 0));
    cooc_.observe(keys, n);
}

bool ExpertCache::preload(const ExpertKey& k) {
    if (sets_ == 0) return false;
    SlotState* set = &slots_[static_cast<std::size_t>(mix(k) % sets_) * static_cast<std::size_t>(ways_)];
    if (probe(set, k) >= 0) return true;  // 已在槽里
    for (int w = 0; w < ways_; ++w) {
        if (!set[w].occupied) {
            set[w] = SlotState{k, 0, 0, true};
            ++used_;
            return true;
        }
    }
    return false;  // 组满：预载不驱逐
}

}  // namespace qinfer::storage