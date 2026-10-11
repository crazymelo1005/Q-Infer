#include "dense/kv_page_cache.hpp"

#include <cstring>

namespace qinfer::dense {

namespace {

// 一个 cell 的槽位数（按 KV 头 × 头宽）。
inline std::uint64_t cell_stride(const KvPageCache& c) { return c.n_head_kv * c.head_dim; }

// 逻辑 cell t、头 h 在某一物理页里的行号：页内 **head-major**（一个头的连续 cell 连续）。
inline std::uint64_t row_in_page(const KvPageCache& c, std::uint64_t t, std::uint64_t h) {
    return (h * c.page_size + (t % c.page_size)) * c.head_dim;
}

// 解析出 cell t 所在物理页的行基址（表里 -1 直接失败）。
bool page_row(const KvPageCache& c, std::uint64_t t, std::int32_t& page, std::string& err) {
    const std::uint64_t logical = t / c.page_size;
    if (logical >= c.table.size()) {
        err = "cell " + std::to_string(t) + " 的逻辑页 " + std::to_string(logical) + " 超出页表";
        return false;
    }
    page = c.table[static_cast<std::size_t>(logical)];
    if (page < 0) {
        err = "逻辑页 " + std::to_string(logical) + " 不在显存（表里为 -1）——这一页要先流式换入";
        return false;
    }
    if (static_cast<std::uint64_t>(page) >= c.n_pages) {
        err = "页表给出的物理页 " + std::to_string(page) + " 超出 " + std::to_string(c.n_pages);
        return false;
    }
    return true;
}

}  // namespace

void KvPageCache::reset() {
    const std::uint64_t per = n_pages * n_head_kv * page_size * head_dim;
    k_pool.assign(static_cast<std::size_t>(per), 0);
    v_pool.assign(static_cast<std::size_t>(per), 0);
    table.assign(static_cast<std::size_t>(n_pages), -1);
}

bool kv_append(KvPageCache& c, std::uint64_t t, const KvCell& cell, std::string& err) {
    err.clear();
    if (c.page_size == 0 || c.n_head_kv == 0 || c.head_dim == 0 || c.n_pages == 0) {
        err = "分页 KV 的几何不成立";
        return false;
    }
    if (cell.k == nullptr || cell.v == nullptr) {
        err = "kv_append 的 K/V 指针为空";
        return false;
    }
    std::int32_t page = 0;
    if (!page_row(c, t, page, err)) return false;
    const std::uint64_t base = static_cast<std::uint64_t>(page) * cell_stride(c) * c.page_size;
    for (std::uint64_t h = 0; h < c.n_head_kv; ++h) {
        const std::uint64_t at = base + row_in_page(c, t, h);
        std::memcpy(&c.k_pool[static_cast<std::size_t>(at)], cell.k + h * c.head_dim,
                    static_cast<std::size_t>(c.head_dim * 2));
        std::memcpy(&c.v_pool[static_cast<std::size_t>(at)], cell.v + h * c.head_dim,
                    static_cast<std::size_t>(c.head_dim * 2));
    }
    return true;
}

bool kv_gather(const KvPageCache& c, const std::uint64_t* ids, std::uint64_t n_ids,
               std::uint16_t* k_scratch, std::uint16_t* v_scratch, std::string& err) {
    err.clear();
    if (n_ids == 0) return true;  // 没选中任何 cell 是合法的（此时也不该要求指针非空）
    if (ids == nullptr || k_scratch == nullptr || v_scratch == nullptr) {
        err = "kv_gather 的指针不完整";
        return false;
    }
    const std::uint64_t stride = cell_stride(c);
    for (std::uint64_t i = 0; i < n_ids; ++i) {
        const std::uint64_t t = ids[i];
        std::int32_t page = 0;
        if (!page_row(c, t, page, err)) return false;
        const std::uint64_t base = static_cast<std::uint64_t>(page) * stride * c.page_size;
        for (std::uint64_t h = 0; h < c.n_head_kv; ++h) {
            const std::uint64_t from = base + row_in_page(c, t, h);
            // 第 i 个 id 的结果落在第 i 个槽位（按给定顺序，即升序）。
            const std::uint64_t to = (i * c.n_head_kv + h) * c.head_dim;
            std::memcpy(&k_scratch[static_cast<std::size_t>(to)],
                        &c.k_pool[static_cast<std::size_t>(from)],
                        static_cast<std::size_t>(c.head_dim * 2));
            std::memcpy(&v_scratch[static_cast<std::size_t>(to)],
                        &c.v_pool[static_cast<std::size_t>(from)],
                        static_cast<std::size_t>(c.head_dim * 2));
        }
    }
    return true;
}

}  // namespace qinfer::dense