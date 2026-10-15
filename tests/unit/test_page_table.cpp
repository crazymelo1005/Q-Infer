// 页表的不变量回归。不引第三方框架：assert 失败即中止，ctest 据此判成败。
#include "storage/page_table.hpp"

#include "check.hpp"
#include <cstdio>

using namespace qinfer::storage;

namespace {

constexpr BlockId kA{1}, kB{2}, kC{3};

void admit_three(PageTable& t) {
    CHECK(t.admit(kA, BlockClass::kExpert, Tier::kVram, 1'000));
    CHECK(t.admit(kB, BlockClass::kExpert, Tier::kHostPinned, 2'000));
    CHECK(t.admit(kC, BlockClass::kTableRow, Tier::kNvme, 90));
}

void test_admit_and_query() {
    PageTable t;
    admit_three(t);
    CHECK(t.size() == 3);

    const Entry* a = t.query(kA);
    CHECK(a != nullptr && a->cls == BlockClass::kExpert && a->tier == Tier::kVram);
    CHECK(a->bytes == 1'000 && a->refcount == 0 && a->prefetch == PrefetchState::kIdle);
    CHECK(t.query(BlockId{99}) == nullptr);

    // 不变量 3：同一逻辑块不重复登记（跨卡共享的块只该有一份条目）
    CHECK(!t.admit(kA, BlockClass::kKv, Tier::kNvme, 5));
    CHECK(t.size() == 3);
    CHECK(!t.admit(BlockId{7}, BlockClass::kKv, Tier::kNvme, 0));  // 零字节的块没有意义
}

void test_refcount_and_evictable() {
    PageTable t;
    admit_three(t);
    t.advance_generation();                      // 让这三个块离开「本世代」

    CHECK(t.evictable(kA));
    CHECK(t.acquire(kA, 3, /*must=*/true));
    CHECK(!t.evictable(kA));                    // 不变量 2：引用未归零
    CHECK(t.acquire(kA, 3, false));
    CHECK(t.query(kA)->refcount == 2);
    CHECK(t.release(kA));
    CHECK(!t.evictable(kA));
    CHECK(t.release(kA));
    CHECK(t.evictable(kA));

    CHECK(!t.release(kA));                      // 不允许静默下溢
    CHECK(!t.release(BlockId{42}));             // 未知 id
    CHECK(!t.acquire(BlockId{42}, 1, false));   // 未登记过的 id 不能引用
}

void test_inflight_prefetch_is_locked() {
    PageTable t;
    admit_three(t);
    t.advance_generation();

    CHECK(t.set_prefetch(kB, PrefetchState::kInflight));
    CHECK(!t.evictable(kB));                    // 不变量 1：预取在途不得被驱逐

    // 消费点到达：acquire 把在途的块标为就绪
    CHECK(t.acquire(kB, 5, false));
    CHECK(t.query(kB)->prefetch == PrefetchState::kReady);
    CHECK(!t.evictable(kB));                    // 仍被引用着
    CHECK(t.release(kB));
    CHECK(t.evictable(kB));
}

void test_eviction_prefers_cold_and_old() {
    PageTable t;
    CHECK(t.admit(BlockId{10}, BlockClass::kKv, Tier::kVram, 512));
    t.advance_generation();
    CHECK(t.set_heat(BlockId{10}, 7));

    CHECK(t.admit(BlockId{11}, BlockClass::kKv, Tier::kVram, 512));
    t.advance_generation();
    CHECK(t.set_heat(BlockId{11}, 3));

    CHECK(t.admit(BlockId{12}, BlockClass::kKv, Tier::kVram, 512));
    t.advance_generation();
    CHECK(t.set_heat(BlockId{12}, 3));

    // 热度最低者优先；同热度时世代号更旧者优先（12 与 11 同热度，11 更旧）
    auto victim = t.evict_one();
    CHECK(victim.has_value() && victim->value == 11);
    victim = t.evict_one();
    CHECK(victim.has_value() && victim->value == 12);
    victim = t.evict_one();
    CHECK(victim.has_value() && victim->value == 10);
    CHECK(!t.evict_one().has_value());          // 表空了
}

void test_current_generation_is_protected() {
    PageTable t;
    CHECK(t.admit(BlockId{20}, BlockClass::kExpert, Tier::kVram, 1'000));
    CHECK(!t.evict_one().has_value());          // 本世代刚搬进来的不动
    t.advance_generation();
    CHECK(t.evict_one().has_value());           // 换代后可驱逐
}

void test_tier_accounting() {
    PageTable t;
    admit_three(t);
    CHECK(t.bytes_in(Tier::kVram) == 1'000);
    CHECK(t.bytes_in(Tier::kHostPinned) == 2'000);
    CHECK(t.bytes_in(Tier::kNvme) == 90);

    CHECK(t.set_tier(kA, Tier::kHostPinned, 0));
    CHECK(t.bytes_in(Tier::kVram) == 0);
    CHECK(t.bytes_in(Tier::kHostPinned) == 3'000);
    CHECK(!t.set_tier(BlockId{99}, Tier::kVram, 0));
}

}  // namespace

void test_remove_guards_the_invariants() {
    PageTable t;
    admit_three(t);
    t.advance_generation();

    // 未登记过的不存在可摘。
    CHECK(!t.remove(BlockId{99}));

    // 不变量 2：引用计数不为零时不许摘。
    CHECK(t.acquire(kA, 3, /*must=*/true));
    CHECK(!t.remove(kA));
    CHECK(t.size() == 3);                       // 拒绝之后条目还在
    CHECK(t.release(kA));
    CHECK(t.remove(kA));                        // 归零之后可以
    CHECK(t.query(kA) == nullptr && t.size() == 2);

    // 不变量 1：预取在途时不许摘（摘了会让预取的回填写到别人的位置上）。
    CHECK(t.set_prefetch(kB, PrefetchState::kInflight));
    CHECK(!t.remove(kB));
    CHECK(t.query(kB) != nullptr);
    CHECK(t.set_prefetch(kB, PrefetchState::kIdle));
    CHECK(t.remove(kB));
    CHECK(t.size() == 1);

    // 摘掉之后同一个块可以重新登记（缓存换出又换回来是常态）。
    CHECK(t.admit(kA, BlockClass::kExpert, Tier::kVram, 1'000));
    CHECK(t.query(kA) != nullptr);
}

int main() {
    test_admit_and_query();
    test_refcount_and_evictable();
    test_inflight_prefetch_is_locked();
    test_eviction_prefers_cold_and_old();
    test_current_generation_is_protected();
    test_tier_accounting();
    test_remove_guards_the_invariants();
    std::puts("page_table: all invariants hold");
    return 0;
}
