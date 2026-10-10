// 专家字节的来源：带缓存的读取路径。把三件东西串起来——
//
//   storage/expert_cache   谁占着槽位（策略可插拔，engine §5）
//   storage/page_table     驱逐前那两条不变量（引用计数归零、预取不在途，interfaces §2）
//   artifact/gguf_table    真的把字节读出来
//
// 职责分工写在类型上：缓存只管占位决策，字节存在本类的常驻区里、按引用来归还（ADR-001 的三层存储
// 里「显存剩余全部作为专家缓存」指的就是这些槽位）。这样专家缓存的策略与页表的不变量都能独立测试，
// 而这条路径只负责把它们接对。
//
// 三条语义，都写进了测试：
//   1. 驱逐只发生在策略给出的顺序里、且只挑页表判定可驱逐的键；全不可驱逐时这次不准入，改走临时缓冲。
//   2. 字节预算是硬上限：用尽后不再准入，改用临时缓冲——缓存不装不下就该退化，不能让正确性受影响。
//   3. pin/unpin 走页表的引用计数，被 pin 的专家在期间不会被换出。
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "experts/ffn.hpp"
#include "storage/expert_cache.hpp"
#include "storage/page_table.hpp"

namespace qinfer::experts {

// 一个专家的三矩阵字节。指针指向常驻区内的副本；走临时缓冲时指向本类内部的临时区，下次调用即失效。
struct ExpertBytes {
    const std::uint8_t* gate = nullptr;
    const std::uint8_t* up = nullptr;
    const std::uint8_t* down = nullptr;
    bool resident = false;  // 假表示这次没进缓存（预算用尽或组内不可驱逐）
};

class ExpertSource {
public:
    // slots / ways / policy 交给专家缓存；byte_budget 是本条路径自己的硬上限（0 表示不限）。
    ExpertSource(const artifact::GgufFile& gguf, const std::string& tensor_prefix,
                 const LayerSpec& spec, const storage::EvictionPolicy& policy,
                 std::size_t slots, int ways, std::uint64_t byte_budget);

    // 同上，但策略由缓存自己持有（只给内置种类）——给不想管策略生命周期的调用方用。
    ExpertSource(const artifact::GgufFile& gguf, const std::string& tensor_prefix,
                 const LayerSpec& spec, storage::PolicyKind kind, std::size_t slots, int ways,
                 std::uint64_t byte_budget);

    // 这一步要读的专家（同一 token 内一起激活，喂给共现图）。步号每调用一次 +1，供 LRU 用。
    void begin_token();
    void observe_token(const std::uint32_t* experts, int n);

    // 取一个专家的三矩阵。返回假只可能是读表失败或专家越界——缓存装不下不会让它失败。
    bool get(std::uint64_t expert, ExpertBytes& out, std::string& err);

    // 引用计数：被 pin 的专家在 unpin 之前不会被换出。
    bool pin(std::uint64_t expert, std::string& err);
    bool unpin(std::uint64_t expert, std::string& err);

    const storage::CacheStats& stats() const { return cache_.stats(); }
    const storage::Cooccurrence& cooccurrence() const { return cache_.cooccurrence(); }
    std::uint64_t residents() const { return static_cast<std::uint64_t>(bytes_.size()); }
    std::uint64_t bytes_resident() const { return bytes_resident_; }
    std::uint64_t transient_reads() const { return transient_reads_; }
    std::uint64_t blocked_admissions() const { return blocked_; }
    std::size_t page_entries() const { return page_.size(); }
    std::int64_t step() const { return step_; }

private:
    void init_common(const std::string& tensor_prefix, const LayerSpec& spec);
    storage::BlockId id_of(std::uint64_t expert) const;
    bool read_into(std::uint64_t expert, std::vector<std::uint8_t>& dst, std::string& err) const;

    const artifact::GgufFile& gguf_;
    const artifact::GgufTensorInfo* gate_ = nullptr;
    const artifact::GgufTensorInfo* up_ = nullptr;
    const artifact::GgufTensorInfo* down_ = nullptr;
    std::uint32_t layer_ = 0;
    std::uint64_t gate_bytes_ = 0;   // 一个专家的 gate 矩阵字节数
    std::uint64_t up_bytes_ = 0;
    std::uint64_t down_bytes_ = 0;
    std::uint64_t expert_bytes_ = 0;  // 三矩阵之和

    storage::ExpertCache cache_;
    storage::PageTable page_;
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> bytes_;
    std::vector<std::uint8_t> transient_;
    std::uint64_t bytes_resident_ = 0;
    std::uint64_t budget_ = 0;
    std::uint64_t transient_reads_ = 0;
    std::uint64_t blocked_ = 0;
    std::int64_t step_ = 0;
};

}  // namespace qinfer::experts