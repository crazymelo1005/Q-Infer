// KV 的常驻窗口：逻辑页 ↔ 物理页的映射、字节配额、LRU 驱逐与钉住。
//
// 这是「KV 搬运」在主机侧参考路径上的落点。分页 KV（`kv_page_cache`，[S-57]）只认「逻辑页 → 物理页」
// 这张表与 `-1 = 不在显存`；本窗口负责在那张表被用之前把需要的页搬进来：未常驻就准入（必要时按 LRU
// 驱逐一个没被钉住的页），窗口满且全被钉住则**明确失败**（不静默改写、也不假装常驻）。
//
// 三条不变量（每条都在测试里按错的读法算一遍）：
//   1. 常驻页数不超过配额（`quota_bytes / page_bytes`）；
//   2. 被钉住的页在本次使用期间不被驱逐——**本步刚写入的那一页也在其中**（刚写的 K/V 不能被换出）；
//   3. `ensure` 成功才去动 `table[...]`，失败时表保持原样（`-1` 就是「不在显存」，`kv_append`/`kv_gather`
//      见到它会明确报错，这正是设计里流式换入的信号）。
//
// 主机侧没有设备/主机之分，故「搬入」只是记账（字节本就在内存里）；真机上它对应一次 H2D。这一点在
// 接口上是同一件事，故这一层可以直接复用。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dense/kv_page_cache.hpp"
#include "dense/qsa_layer.hpp"
#include "dense/kv_store.hpp"

namespace qinfer::dense {

struct KvWindow {
    std::uint64_t page_bytes = 0;   // 一页的字节数（k+v：2 × n_head_kv × page_size × head_dim × 2）
    std::uint64_t quota_bytes = 0;  // 常驻上限；0 = 不限

    // 统计（每次 decode 步之间可读、可清零）：命中、未命中、驱逐、被拒。
    std::uint64_t hits = 0, misses = 0, evictions = 0, refusals = 0;

    // 权威副本（可选）：给了就在**准入时**把该 block 的字节从副本拷进新槽位（引擎 `kv_stream_resolve`
    // 的第 3 步）。不给就只是记账——主机侧那点「字节本来就都在内存里」的旧行为，与加之前逐位一致。
    KvHostStore* store = nullptr;
    // 槽位所在（`attach_kv_window` 挂上）。准入时往这里拷；没挂槽位却挂了副本会明确报错。
    KvPageCache* pool = nullptr;

    struct Entry {
        std::uint64_t logical = 0;
        std::uint64_t physical = 0;
        int pins = 0;
        std::uint64_t last_use = 0;
    };
    std::vector<Entry> entries;
    std::uint64_t clock = 0;
    std::uint64_t next_physical = 0;

    // 逻辑页表（`kv_page_cache` 的那张），由 `attach_kv_window` 挂上。窗口自己维护它：准入写物理页号、
    // **驱逐写回 -1**。让窗口维护而不是让调用方维护，是因为「驱逐了却忘了把表标成不在显存」在主机侧
    // 恰好仍然读得到对的字节（同一块内存），到设备侧才会读到别的物理槽——这种「主机上静默正确」的错
    // 是最难发现的一类，故不给调用方留下这个自由度。
    std::vector<std::int32_t>* table = nullptr;

    // 配额允许的常驻页数（`quota_bytes == 0` 时返回 SIZE_MAX）。page_bytes 为 0 时返回 0（调用方错）。
    std::uint64_t capacity_pages() const;

    bool resident(std::uint64_t logical) const;
    const Entry* find(std::uint64_t logical) const;

    // 确保逻辑页常驻：已在则记命中并刷新使用序；不在则准入（必要时驱逐一个未钉住的页）并把物理页号
    // 写进 physical（同时写进挂上的逻辑页表）。窗口满且全部被钉住时记一次被拒并返回假（err 说明原因），
    // 表不动。驱逐会把被驱逐页在表里写成 -1。挂了权威副本时，**准入**这一步还要把该 block 的字节从副本
    // 拷进新槽位（`admitted` 为真即这次是准入）——驱逐不回写，因为副本才是权威的那一份。
    bool ensure(std::uint64_t logical, std::uint64_t& physical, std::string& err,
                bool* admitted = nullptr);

    // 钉住/松开一个**已常驻**的页（本次使用期间不许被驱逐）。未常驻或计数下溢都返回假。
    bool pin(std::uint64_t logical, std::string& err);
    bool unpin(std::uint64_t logical, std::string& err);

    void reset_stats() { hits = misses = evictions = refusals = 0; }
};

// 给一层 QSA 挂上常驻窗口：把它的逻辑页表整表标成「不在显存」，再由窗口按需搬入。
// 必须同时清表——否则表里预填的恒等映射会让窗口形同虚设（这是本接口最常见的错接线）。
//
// `logical_blocks` 是逻辑页表的长度（会话上下文能到多少 cell / page_size），**不是槽位数**——引擎那边
// 这两者也是分开的（`page_table` 是 (n_blocks,)、`slot_block` 是 (n_slots,)）。给 0 表示按权威副本来
// （它就是这个会话的逻辑容量），没有副本时退回池的大小（恒等映射那种用法）。
void attach_kv_window(QsaScratch& s, KvWindow& w, std::uint64_t logical_blocks = 0);

}  // namespace qinfer::dense
