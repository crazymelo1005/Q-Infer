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

bool KvWindow::ensure(std::uint64_t logical, std::uint64_t& physical, std::string& err) {
    err.clear();
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
    // 未命中：窗口满了就先驱逐一个**没被钉住**、最久未用的页。
    misses++;
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
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(victim));
        if (table != nullptr) (*table)[static_cast<std::size_t>(victim_logical)] = -1;  // 被换出 → 不在显存
        evictions++;
    }
    Entry e;
    e.logical = logical;
    e.physical = next_physical++;
    e.pins = 0;
    e.last_use = clock++;
    entries.push_back(e);
    if (table != nullptr) (*table)[static_cast<std::size_t>(logical)] = static_cast<std::int32_t>(e.physical);
    physical = e.physical;
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

void attach_kv_window(QsaScratch& s, KvWindow& w) {
    s.window = &w;
    w.table = &s.kv.table;
    // 整表标成「不在显存」，之后的常驻与否由窗口决定（窗口负责准入时写、驱逐时写回 -1）。
    s.kv.table.assign(static_cast<std::size_t>(s.kv.n_pages), -1);
}

}  // namespace qinfer::dense
