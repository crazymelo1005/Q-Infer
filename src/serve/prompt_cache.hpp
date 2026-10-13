// 服务层 token 前缀缓存：/metrics 的 conversation_cache 数据源（spec §6 的占位在此填实）。
// 键 = 渲染后的完整 prompt 字符串，值 = encode 得到的 token ids。
// 只做全串精确匹配：BPE 合并可跨任意字节边界，把 prompt 从某处切开只编码尾巴，结果与整段
// 编码的对应段不一致，前缀级 token 复用不成立（前缀级 KV 复用属引擎侧能力，排引擎里程碑）。
// 容量以条数计（LRU），单条超过 kMaxTokensPerEntry 不缓存（防巨提示词吃满内存）；capacity=0
// 禁用。统计（hits/misses/entries/tokens）由 /metrics 实读，全是真实计数。
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace qinfer::serve {

class PromptCache {
public:
    // 单条 token 数上限：超限的提示词不缓存（65536 × 4B = 256 KiB/条，64 条上限约 16 MiB）。
    static constexpr std::size_t kMaxTokensPerEntry = 65536;

    struct Stats {
        std::uint64_t hits = 0, misses = 0;
        std::size_t entries = 0;
        std::uint64_t tokens = 0;   // 缓存内全部条目 token 数之和
    };

    explicit PromptCache(std::size_t capacity);   // 0 = 禁用（不缓存也不计数）

    bool enabled() const { return capacity_ > 0; }
    // 命中 → true 并拷出 ids（该条目提到最前）；未命中 → false 且不碰 ids。上锁。
    bool lookup(const std::string& key, std::vector<std::int32_t>& ids);
    // 存入：容量 0 或超单条上限则跳过；同键覆写（先扣旧 token 数）。上锁。
    void store(const std::string& key, const std::vector<std::int32_t>& ids);
    Stats stats() const;   // 上锁取快照；禁用态返回全零

private:
    std::size_t capacity_;
    mutable std::mutex mu_;
    using Entry = std::pair<std::string, std::vector<std::int32_t>>;
    std::list<Entry> lru_;   // 头 = 最近使用
    std::unordered_map<std::string, std::list<Entry>::iterator> map_;
    std::uint64_t hits_ = 0, misses_ = 0;
    std::uint64_t tokens_ = 0;
};

}  // namespace qinfer::serve