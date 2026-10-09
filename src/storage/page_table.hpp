// 统一页表：把逻辑块映射到介质，并守住 interfaces.md 第 2 节的三条不变量。
//
//   1. 预取一旦发出即锁定到消费完成，中途不得被驱逐。
//   2. 驱逐前引用计数必须归零。
//   3. 跨卡与跨实例共享的块由同一份页表管理，不重复复制。
//
// 第 3 条由「调用方对同一逻辑块只调一次 admit」保证：admit 对已存在的 id 返回 false，
// 而不是新建第二份条目。
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qinfer::storage {

enum class BlockClass : std::uint8_t { kExpert, kTableRow, kKv };
enum class Tier : std::uint8_t { kVram, kHostPinned, kNvme };
enum class PrefetchState : std::uint8_t { kIdle, kInflight, kReady };

const char* to_string(BlockClass c);
const char* to_string(Tier t);
const char* to_string(PrefetchState s);

struct BlockId {
    std::uint64_t value = 0;
    friend bool operator==(BlockId a, BlockId b) { return a.value == b.value; }
};

struct BlockIdHash {
    std::size_t operator()(BlockId id) const { return std::hash<std::uint64_t>{}(id.value); }
};

struct Entry {
    BlockId id{};
    BlockClass cls = BlockClass::kExpert;
    Tier tier = Tier::kNvme;
    std::uint64_t bytes = 0;      // 块字节数，用于分层记账
    std::uint64_t offset = 0;     // 在所属层中的物理偏移
    std::uint32_t refcount = 0;
    std::uint64_t generation = 0; // 进入本表时的世代号
    std::uint32_t heat = 0;
    PrefetchState prefetch = PrefetchState::kIdle;
    std::int64_t deadline_steps = 0;  // 消费点（以 decode 步为单位）
    bool must = false;                 // 必需请求；预算不足时不得丢弃
};

class PageTable {
public:
    // 登记一个逻辑块。已被登记过的 id 返回 false（不变量 3：不重复复制）。
    bool admit(BlockId id, BlockClass cls, Tier tier, std::uint64_t bytes, std::uint64_t offset = 0);

    // 引用一次：引用计数加一，并记录本次的消费点与是否必需。未知 id 返回 false。
    // 若该块正在预取途中，本次引用同时把它标为就绪（消费点到达）。
    bool acquire(BlockId id, std::int64_t deadline_steps, bool must);

    // 解除一次引用。计数已为 0 或未知 id 返回 false（不允许静默下溢）。
    bool release(BlockId id);

    // 递增世代号。本世代登记的块在驱逐时受保护，避免把当前步刚搬进来的东西换出去。
    void advance_generation();

    // 查询。未知 id 返回 nullptr。
    const Entry* query(BlockId id) const;

    bool set_tier(BlockId id, Tier tier, std::uint64_t offset);
    bool set_prefetch(BlockId id, PrefetchState state);
    bool set_heat(BlockId id, std::uint32_t heat);

    // 试探某块当前能否被驱逐：引用计数归零、且预取不在途。
    bool evictable(BlockId id) const;

    // 驱逐一个块并返回它的 id。选法：热度最低、其次世代号最旧、再次 id 最小（确定性）。
    // 只考虑本世代之前登记、引用计数归零、预取不在途的块；没有候选时返回 nullopt。
    std::optional<BlockId> evict_one();

    std::size_t size() const { return entries_.size(); }
    std::uint64_t bytes_in(Tier tier) const;
    std::uint64_t generation() const { return generation_; }

private:
    std::unordered_map<BlockId, Entry, BlockIdHash> entries_;
    std::uint64_t generation_ = 1;
};

}  // namespace qinfer::storage
