// 有界行缓存：按行号哈希定组、组内轮转替换。这是 ADR-007 退守后保留的那一件——G-04 证明表行的
// 顺序性为零、近邻复用为零，能吃下的只有长距离复现，而它正是缓存能利用的东西。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qinfer::storage {

inline constexpr std::size_t kTableRowBytes = 90;

class RowCache {
public:
    // 容量按 8 向下取整到 8 的倍数（与参考引擎的 sets = rows / 8 同口径）；容量 0 表示关闭。
    explicit RowCache(std::size_t capacity_rows);

    // 命中则把指向缓存内字节的指针写进 out 并返回 true；未命中返回 false。两次调用都计数。
    bool find(std::uint32_t row, const std::uint8_t** out);

    // 已在缓存里的行不重复写入；否则按组内轮转替换一个槽位。
    void insert(std::uint32_t row, const std::uint8_t* bytes);

    std::size_t capacity() const { return sets_ * kWays; }
    std::size_t size() const { return used_; }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }

private:
    static constexpr std::uint32_t kWays = 8;
    static constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;
    static std::uint64_t mix(std::uint32_t row);
    const std::uint8_t* probe(std::uint32_t row) const;   // 不计数，供 find 与 insert 共用

    std::size_t sets_ = 0;
    std::vector<std::uint32_t> keys_;
    std::vector<std::uint8_t> data_;
    std::vector<std::uint8_t> next_;
    std::size_t used_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
};

}  // namespace qinfer::storage
