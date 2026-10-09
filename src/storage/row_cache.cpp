#include "storage/row_cache.hpp"

#include <cstring>

namespace qinfer::storage {

std::uint64_t RowCache::mix(std::uint32_t row) {
    const std::uint64_t x = static_cast<std::uint64_t>(row) * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
}

RowCache::RowCache(std::size_t capacity_rows) : sets_(capacity_rows / kWays) {
    keys_.assign(sets_ * kWays, kEmpty);
    data_.assign(sets_ * kWays * kTableRowBytes, 0);
    next_.assign(sets_, 0);
}

const std::uint8_t* RowCache::probe(std::uint32_t row) const {
    const std::size_t s = static_cast<std::size_t>(mix(row) % sets_);
    for (std::uint32_t w = 0; w < kWays; ++w) {
        const std::size_t slot = s * kWays + w;
        if (keys_[slot] == row) return &data_[slot * kTableRowBytes];
    }
    return nullptr;
}

bool RowCache::find(std::uint32_t row, const std::uint8_t** out) {
    if (sets_ == 0) {
        ++misses_;
        return false;
    }
    const std::uint8_t* p = probe(row);
    if (p == nullptr) {
        ++misses_;
        return false;
    }
    ++hits_;
    *out = p;
    return true;
}

void RowCache::insert(std::uint32_t row, const std::uint8_t* bytes) {
    if (sets_ == 0) return;
    if (probe(row) != nullptr) return;   // 已在缓存里，不重复写入
    const std::size_t s = static_cast<std::size_t>(mix(row) % sets_);
    const std::uint32_t w = next_[s];
    next_[s] = static_cast<std::uint8_t>((w + 1) % kWays);
    const std::size_t slot = s * kWays + w;
    if (keys_[slot] == kEmpty) ++used_;
    keys_[slot] = row;
    std::memcpy(&data_[slot * kTableRowBytes], bytes, kTableRowBytes);
}

}  // namespace qinfer::storage
