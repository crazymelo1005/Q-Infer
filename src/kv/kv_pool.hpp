// 两个 KV 池的记账与准入（interfaces.md 第 5 节）。
//
// 两块 KV 必须分开记账：GDN 循环状态随并发序列数增长、与上下文长度无关；QSA KV 与索引器 KV 随
// 上下文长度乘以并发增长。故两个池有独立配额、独立限额、独立驱逐，而调度器必须**同时**看两个余量，
// 任一不足就拒绝新序列——这条约束是本类存在的理由。
//
// 与「常驻热窗」的关系（§7 第 2 条）：QSA 只常驻最近一段 token，其余分块流式读。所以一个序列在
// QSA 侧占的是**常驻窗口**那部分字节，它可以被独立驱逐到 0（该序列改全流式读）而序列本身还活着；
// GDN 侧则是全有或全无，要腾出它只能把整个序列赶走。两种驱逐因此分开实现。
//
// 已知缺口（画像自己记的）：GDN 侧每序列字节需要 SSM 几何，环境2 上还没取到，故那条配置只能由
// 调用方给；QSA 侧可以用画像的两条数字反推（见测试）。
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qinfer::kv {

enum class Pool : std::uint8_t { kGdnState = 0, kQsaKv };
const char* to_string(Pool p);

struct PoolConfig {
    // 两个池各自的配额上限（字节）；0 表示这个池不启用，则任何非零需求都会被拒。
    std::uint64_t gdn_limit_bytes = 0;
    std::uint64_t qsa_limit_bytes = 0;
    // GDN：一个序列占多少（与上下文无关）。
    std::uint64_t gdn_bytes_per_sequence = 0;
    // QSA：一个 token 的 KV 在一个 QSA 层占多少字节。
    std::uint64_t qsa_bytes_per_cell = 0;
    // QSA 常驻热窗的上限：一个序列最多常驻多少个 cell（超出部分流式读）。
    std::uint32_t qsa_resident_cell_cap = 0;
};

struct SequenceEntry {
    std::uint64_t id = 0;
    std::uint64_t gdn_bytes = 0;
    std::uint64_t qsa_resident_bytes = 0;  // 常驻窗口那部分；可被驱逐到 0
    std::uint64_t qsa_total_bytes = 0;     // 该序列在 QSA 侧的全部 KV（常驻 + 流式）
};

class Pools {
public:
    explicit Pools(const PoolConfig& cfg) : cfg_(cfg) {}

    // 一个序列在给定上下文长度下，QSA 侧会常驻多少字节（受常驻窗口上限约束）。
    std::uint64_t resident_bytes_for(std::uint64_t context_tokens) const;
    // 一个序列在 QSA 侧的全部 KV 字节。
    std::uint64_t total_bytes_for(std::uint64_t context_tokens) const;

    // 准入：两个池都要有余量。任一不足则拒绝，并写明是哪个池（why 可为空）。
    bool admit(std::uint64_t sequence_id, std::uint64_t context_tokens, std::string* why = nullptr);

    // 驱逐一个池：QSA 侧把常驻窗口最大的那个序列的常驻字节清零（该序列改全流式读、序列仍在）；
    // GDN 侧把占用最大的那个序列整个赶走（两个池的字节一起释放）。返回释放的字节数，没有可驱逐的
    // 东西时返回 0。选谁是确定性的（占用最大、并列取 id 最小）；策略要换就换这一处。
    std::uint64_t evict(Pool pool);

    // 释放一个序列（两个池一起）。
    bool release(std::uint64_t sequence_id);

    // §5 的 usage 返回**余量**（该节原文如此），故这里按余量实现；used 另外给，便于观测。
    std::uint64_t usage(Pool pool) const { return remaining(pool); }
    std::uint64_t remaining(Pool pool) const;
    std::uint64_t used(Pool pool) const;
    const SequenceEntry* query(std::uint64_t sequence_id) const;
    std::size_t sequences() const { return seqs_.size(); }
    const PoolConfig& config() const { return cfg_; }

private:
    PoolConfig cfg_;
    std::unordered_map<std::uint64_t, SequenceEntry> seqs_;
};

}  // namespace qinfer::kv