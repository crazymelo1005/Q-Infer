// 专家槽位缓存：组相联、键是 (层, 专家)、替换策略可插拔。ADR-001 的三层存储里「显存剩余全部作为
// 专家缓存」指的就是这些槽位；engine §5 第 3 条要求策略可插拔，第 1 条要求共现图作策略输入。
//
// 职责边界：本类只管**谁占着哪个槽位**（策略与观测），不管槽位里的字节——字节在介质层，按同一个
// ExpertKey 登记进 `storage/page_table`。因此驱逐前必须经过页表那两条不变量（引用计数归零、预取不在
// 途），本类通过调用方给的 evictable 回调来遵守：access 在需要换出时按策略给出的顺序挑第一个
// 「当前可驱逐」的槽位，全都不可驱逐就返回 kMissBlocked 而不是硬换（engine §3.1）。
//
// 命中率按 engine §5 最后一段只作观测量：引用它的结论必须同时给出 PCIe 侧占比与空闲显存。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "storage/cooccurrence.hpp"

namespace qinfer::storage {

// 一个槽位的策略可见状态。last_used / inserted 由缓存无条件维护，策略不必自己记。
struct SlotState {
    ExpertKey key{};
    std::int64_t last_used = 0;
    std::int64_t inserted = 0;
    bool occupied = false;
};

struct PolicyContext {
    std::int64_t step = 0;
    const ExpertKey* step_keys = nullptr;  // 本步正在读的键（共现感知策略的查询集）
    int step_keys_n = 0;
    const Cooccurrence* cooc = nullptr;
};

// 替换策略只回答一件事：把这一组的 ways 个槽位按「先换谁」的顺序排出来。
// 排成顺序而不是只挑一个，是为了让调用方能跳过当前不可驱逐的那些（页表不变量）而不必重排。
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    // 填 out_order（长度 ways）并返回写入的个数（= ways）。out_order[0] 是最该被换掉的。
    virtual int rank(const SlotState* slots, int ways, const ExpertKey& incoming,
                     const PolicyContext& ctx, int* out_order) const = 0;
    virtual const char* name() const = 0;
};

// 组内轮转的朴素基线：按插入先后换（最早进来的先出去）。这与参考引擎的行缓存、本仓库
// `storage/row_cache` 的组内轮转指针同口径——那两者在命中时都不重排，故行为就是 FIFO。
class FifoPolicy : public EvictionPolicy {
public:
    int rank(const SlotState* slots, int ways, const ExpertKey& incoming, const PolicyContext& ctx,
             int* out_order) const override;
    const char* name() const override { return "fifo"; }
};

// 最久未用；并列取插入更早的，再并列取键更小的（确定性）。
class LruPolicy : public EvictionPolicy {
public:
    int rank(const SlotState* slots, int ways, const ExpertKey& incoming, const PolicyContext& ctx,
             int* out_order) const override;
    const char* name() const override { return "lru"; }
};

// 共现感知（engine §5 第 1 条）：换掉「与本步正在读的键共现最弱」的那一个；并列退回 LRU。
// 本步正在读的键一律排到最后——它们现在就要用，换掉等于自伤。
class CooccurrenceAwarePolicy : public EvictionPolicy {
public:
    int rank(const SlotState* slots, int ways, const ExpertKey& incoming, const PolicyContext& ctx,
             int* out_order) const override;
    const char* name() const override { return "cooccurrence-aware"; }
};

struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t compulsory = 0;  // 空槽位直接占用的次数（冷启动的那些）
    std::uint64_t replaced = 0;    // 换出后占用的次数
    std::uint64_t blocked = 0;     // 组内全都不可驱逐、这次没换成的次数
};

class ExpertCache {
public:
    // capacity_slots = 槽位数（0 表示关闭）；ways = 相联度（1 表示直接映射，= capacity 表示全相联）。
    // 容量按 ways 向下取整到整数个组。策略按引用保存，调用方保证它在缓存存活期内有效。
    // max_cooc_pairs = 共现图的种类上限（engine §3.4 要求显式给出这个密度上限）。
    ExpertCache(std::size_t capacity_slots, int ways, const EvictionPolicy& policy,
                std::size_t max_cooc_pairs = 1u << 18);

    // 调用方提供的「当前可驱逐」判定（页表那两条不变量）；nullptr 表示全都可驱逐。
    using EvictableFn = bool (*)(const ExpertKey& key, void* ctx);

    enum class Outcome : std::uint8_t {
        kHit = 0,
        kMissFree,      // 组内有空位，直接占用
        kMissReplaced,  // 换出了 victim_out
        kMissBlocked,   // 组内全不可驱逐，没有换成（调用方应放弃这次准入）
    };

    Outcome access(const ExpertKey& k, std::int64_t step, EvictableFn evictable, void* ctx,
                   ExpertKey* victim_out);

    // 把本步正在读的键记进共现图（engine §5 第 1 条）；也用于策略的查询集。
    void observe_step(const ExpertKey* keys, int n);

    // 预载一个键（engine §5 第 2 条：会话开始按簇预载，而不是等它被读到）。只占空槽，槽满返回 false——
    // 预载不驱逐，因为驱逐的判据是策略与页表不变量，预载阶段这两者都还没建立。不计入命中/未命中。
    bool preload(const ExpertKey& k);

    std::size_t capacity() const { return slots_.size(); }
    std::size_t sets() const { return sets_; }
    int ways() const { return ways_; }
    std::size_t size() const { return used_; }
    const CacheStats& stats() const { return stats_; }
    const Cooccurrence& cooccurrence() const { return cooc_; }

private:
    static std::uint64_t mix(ExpertKey k);
    int probe(const SlotState* set, const ExpertKey& k) const;

    std::size_t sets_ = 0;
    int ways_ = 0;
    std::vector<SlotState> slots_;
    std::size_t used_ = 0;
    CacheStats stats_;
    const EvictionPolicy* policy_ = nullptr;
    Cooccurrence cooc_;
    std::vector<ExpertKey> step_keys_;
};

}  // namespace qinfer::storage