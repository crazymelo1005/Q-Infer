// 有界行缓存的行为回归。不引第三方框架。
#include "storage/row_cache.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer::storage;

namespace {

std::vector<std::uint8_t> pattern(std::uint32_t row) {
    std::vector<std::uint8_t> b(kTableRowBytes);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint8_t>((row * 31 + i * 17) & 0xFF);
    return b;
}

void test_hit_and_payload() {
    RowCache cache(16);
    const std::vector<std::uint8_t> want = pattern(4242);

    const std::uint8_t* p = nullptr;
    CHECK(!cache.find(4242, &p));            // 冷：未命中
    CHECK(cache.misses() == 1 && cache.hits() == 0);

    cache.insert(4242, want.data());
    CHECK(cache.size() == 1);
    CHECK(cache.find(4242, &p));             // 命中，且拿回来的字节与写进去的一致
    CHECK(cache.hits() == 1 && cache.misses() == 1);
    for (std::size_t i = 0; i < want.size(); ++i) CHECK(p[i] == want[i]);
}

void test_capacity_is_a_hard_bound() {
    RowCache cache(16);
    CHECK(cache.capacity() == 16);

    // 容量是硬上界：插入远多于容量的行，占用不涨。
    // 注意 size() 不会恰好等于容量——组是按行号哈希分的，各行未必把每个槽位都填到，
    // 轮转替换也不会增加占用。这里断言的是「有界」，不是「填满」。
    for (std::uint32_t r = 0; r < 64; ++r) cache.insert(r, pattern(r).data());
    CHECK(cache.size() <= cache.capacity());
    CHECK(cache.size() > 0);

    for (std::uint32_t r = 64; r < 256; ++r) cache.insert(r, pattern(r).data());
    CHECK(cache.size() <= cache.capacity());
}

void test_eviction_is_deterministic() {
    // 同一串插入顺序 → 两次运行留下的行集合与命中行为完全一致（替换策略不能带随机性）。
    auto run = [] {
        RowCache c(16);
        for (std::uint32_t r = 0; r < 40; ++r) c.insert(r, pattern(r).data());
        std::vector<bool> present(40, false);
        const std::uint8_t* p = nullptr;
        for (std::uint32_t r = 0; r < 40; ++r) present[r] = c.find(r, &p);
        return present;
    };
    CHECK(run() == run());
}

void test_reinsert_does_not_grow() {
    RowCache cache(16);
    const std::vector<std::uint8_t> b = pattern(7);
    for (int i = 0; i < 10; ++i) cache.insert(7, b.data());
    CHECK(cache.size() == 1);
    CHECK(cache.capacity() == 16);
}

void test_zero_capacity_degrades_safely() {
    RowCache cache(0);
    const std::vector<std::uint8_t> b = pattern(1);
    cache.insert(1, b.data());
    const std::uint8_t* p = nullptr;
    CHECK(!cache.find(1, &p));
    CHECK(cache.size() == 0);
}

}  // namespace

int main() {
    test_hit_and_payload();
    test_capacity_is_a_hard_bound();
    test_eviction_is_deterministic();
    test_reinsert_does_not_grow();
    test_zero_capacity_degrades_safely();
    std::puts("row_cache: bounded, deterministic, payload intact");
    return 0;
}
