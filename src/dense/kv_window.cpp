#include "dense/kv_window.hpp"

#include <cstddef>
#include <limits>

namespace qinfer::dense {

std::uint64_t KvWindow::capacity_pages() const {
    if (page_bytes == 0) return 0;
    if (quota_bytes == 0) return std::numeric_limits<std::uint64_t>::max();
    return quota_bytes / page_bytes;
}

const KvWindow::Entry* KvWindow::find(std::uint64_t logical) const {
    for (const Entry& e : entries) {
        if (e.logical == logical) return &e;
    }
    return nullptr;
}

bool KvWindow::resident(std::uint64_t logical) const { return find(logical) != nullptr; }

bool KvWindow::ensure(std::uint64_t logical, std::uint64_t& physical, std::string& err,
                      bool* admitted) {
    err.clear();
    if (admitted != nullptr) *admitted = false;
    if (page_bytes == 0) {
        err = "常驻窗口没给页字节数";
        return false;
    }
    const std::uint64_t cap = capacity_pages();
    if (cap == 0) {
        err = "配额 " + std::to_string(quota_bytes) + " 字节装不下一页 " + std::to_string(page_bytes);
        refusals++;
        return false;
    }
    if (const Entry* e = find(logical)) {
        // 命中：刷新使用序（用 const_cast 是因为 find 为了 const 表查询而返回 const 指针）。
        const_cast<Entry*>(e)->last_use = clock++;
        hits++;
        physical = e->physical;
        return true;
    }
    // 未命中：窗口满了就先驱逐一个**没被钉住**、最久未用的页，并**复用它的槽位**。
    // （不复用就会一直要新槽位——真流式跑几步就越过池的边；引擎那侧也是「老 block -> -1、新 block -> slot」。）
    misses++;
    std::uint64_t slot = 0;
    if (entries.size() >= cap) {
        std::size_t victim = entries.size();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].pins != 0) continue;
            if (victim == entries.size() || entries[i].last_use < entries[victim].last_use) victim = i;
        }
        if (victim == entries.size()) {
            err = "常驻窗口已满（" + std::to_string(entries.size()) + " 页）且全部被钉住，无法为逻辑页 " +
                  std::to_string(logical) + " 腾位";
            refusals++;
            return false;
        }
        const std::uint64_t victim_logical = entries[victim].logical;
        slot = entries[victim].physical;
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(victim));
        if (table != nullptr) (*table)[static_cast<std::size_t>(victim_logical)] = -1;  // 被换出 → 不在显存
        evictions++;
    } else {
        slot = next_physical++;
    }
    Entry e;
    e.logical = logical;
    e.physical = slot;
    e.pins = 0;
    e.last_use = clock++;
    entries.push_back(e);
    if (table != nullptr) (*table)[static_cast<std::size_t>(logical)] = static_cast<std::int32_t>(e.physical);
    physical = e.physical;
    // 准入：把这一页的字节从权威副本拷进新槽位（引擎 resolve 的第 3 步）。驱逐那一步不回写——副本才是
    // 权威的那一份，故槽位永远不会比副本新，也就不需要失效重填。
    if (store != nullptr) {
        if (pool == nullptr) {
            err = "挂了 KV 权威副本但没挂槽位缓冲（先 attach_kv_window）";
            return false;
        }
        if (store->page_size != pool->page_size || store->n_head_kv != pool->n_head_kv ||
            store->head_dim != pool->head_dim) {
            err = "KV 权威副本的几何与槽位池不一致";
            return false;
        }
        const std::uint64_t elems = store->block_elems();
        if (e.physical >= pool->n_pages) {
            err = "槽位号 " + std::to_string(e.physical) + " 超出池的 " + std::to_string(pool->n_pages) +
                  " 页（窗口的配额与池对不上）";
            return false;
        }
        std::uint16_t* k_dst = pool->k_pool.data() + static_cast<std::size_t>(e.physical * elems);
        std::uint16_t* v_dst = pool->v_pool.data() + static_cast<std::size_t>(e.physical * elems);
        if (!store->read_block(logical, k_dst, v_dst, err)) return false;
    }
    if (admitted != nullptr) *admitted = true;
    return true;
}

bool KvWindow::pin(std::uint64_t logical, std::string& err) {
    err.clear();
    for (Entry& e : entries) {
        if (e.logical == logical) {
            ++e.pins;
            return true;
        }
    }
    err = "钉住失败：逻辑页 " + std::to_string(logical) + " 不在常驻窗口里";
    return false;
}

bool KvWindow::unpin(std::uint64_t logical, std::string& err) {
    err.clear();
    for (Entry& e : entries) {
        if (e.logical == logical) {
            if (e.pins <= 0) {
                err = "松开失败：逻辑页 " + std::to_string(logical) + " 的钉住计数已是 0";
                return false;
            }
            --e.pins;
            return true;
        }
    }
    err = "松开失败：逻辑页 " + std::to_string(logical) + " 不在常驻窗口里";
    return false;
}

void attach_kv_window(QsaScratch& s, KvWindow& w, std::uint64_t logical_blocks) {
    s.window = &w;
    w.table = &s.kv.table;
    w.pool = &s.kv;
    // 挂了权威副本就同时挂到这一层上（写 cell 时要写副本那一份）。
    s.store = w.store;
    // 逻辑页表的长度按**逻辑页数**，不是槽位数：引擎那边也是分开的（`page_table` 是 (n_blocks,)、
    // `slot_block` 是 (n_slots,)）。给了就用给的，没给就按权威副本的容量，再没有就按池的大小。
    std::uint64_t blocks = logical_blocks;
    if (blocks == 0) blocks = w.store != nullptr ? w.store->n_blocks : s.kv.n_pages;
    s.kv.table.assign(static_cast<std::size_t>(blocks), -1);
}

}  // namespace qinfer::dense
