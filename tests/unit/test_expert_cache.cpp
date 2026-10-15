// 专家槽位缓存与共现图的回归。不引第三方框架。
//
// 五路证据：
//   1. 几何与键：容量按相联度取整、容量 0 即关闭、键是 (层, 专家) 而不是单独的专家。
//   2. 空位优先、组内替换：新键只在它哈希到的那个组里换，组内全满才换。
//   3. FIFO 与 LRU 的可观测差异（构造 trace，逐步手算）：LRU 在这条 trace 上必须严格优于 FIFO。
//   4. 共现感知策略的**机制**（直接测 rank）：共现弱的先换、本步在用的排最后、并列退回 LRU；
//      并与 LRU 的选择对照，证明两条策略给出的顺序确实不同。
//   5. 页表不变量：不可驱逐的槽位不被换出；组内全不可驱逐时返回 kMissBlocked 而不是硬换。
//   6. 共现图的有界性：种类上限满了之后丢当时最弱的一对，且计数不越界。
#include "storage/expert_cache.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer::storage;

namespace {

ExpertKey K(std::uint32_t layer, std::uint32_t expert) { return ExpertKey{layer, expert}; }

void put(ExpertCache& c, ExpertKey k, std::int64_t step) {
    ExpertKey victim;
    c.access(k, step, nullptr, nullptr, &victim);
}

void test_geometry_and_key() {
    LruPolicy lru;
    ExpertCache off(0, 8, lru);
    CHECK(off.capacity() == 0 && off.sets() == 0);
    ExpertKey v;
    CHECK(off.access(K(0, 1), 0, nullptr, nullptr, &v) == ExpertCache::Outcome::kMissBlocked);

    // 容量 100 / 相联度 8 -> 12 组 96 槽（尾巴 4 个丢掉）。
    ExpertCache c(100, 8, lru);
    CHECK(c.capacity() == 96 && c.sets() == 12 && c.ways() == 8);
    // 相联度 = 容量 -> 全相联（1 组）。
    ExpertCache fa(64, 64, lru);
    CHECK(fa.sets() == 1 && fa.capacity() == 64);

    // 键含层：同一个专家号在不同层是不同的键。
    ExpertCache small(4, 4, lru);  // 1 组 4 槽
    put(small, K(0, 7), 0);
    ExpertKey v1;
    CHECK(small.access(K(0, 7), 1, nullptr, nullptr, &v1) == ExpertCache::Outcome::kHit);
    ExpertKey v2;
    CHECK(small.access(K(1, 7), 1, nullptr, nullptr, &v2) == ExpertCache::Outcome::kMissFree);
    CHECK(small.size() == 2);  // 层不同 => 各占一个槽
}

void test_free_way_then_replace_within_set() {
    FifoPolicy fifo;
    ExpertCache c(4, 4, fifo);  // 1 组 4 槽
    // 头 4 个是冷启动（空位优先），第 5 个才触发替换。
    for (int i = 0; i < 4; ++i) {
        ExpertKey v;
        CHECK(c.access(K(0, static_cast<std::uint32_t>(i)), i, nullptr, nullptr, &v) ==
              ExpertCache::Outcome::kMissFree);
    }
    CHECK(c.stats().compulsory == 4 && c.stats().replaced == 0);
    ExpertKey victim;
    CHECK(c.access(K(0, 9), 5, nullptr, nullptr, &victim) == ExpertCache::Outcome::kMissReplaced);
    CHECK(c.stats().replaced == 1);
    CHECK(victim == K(0, 0));  // FIFO 换最早进来的那个
    CHECK(c.size() == 4);      // 占用数不变

    // 命中不改变占用数，也不计入 compulsory / replaced。
    ExpertKey v2;
    CHECK(c.access(K(0, 9), 6, nullptr, nullptr, &v2) == ExpertCache::Outcome::kHit);
    CHECK(c.size() == 4 && c.stats().compulsory == 4 && c.stats().replaced == 1);
}

std::uint64_t drive(ExpertCache& c, const std::vector<std::uint32_t>& trace) {
    for (std::size_t i = 0; i < trace.size(); ++i) {
        ExpertKey v;
        c.access(K(0, trace[i]), static_cast<std::int64_t>(i), nullptr, nullptr, &v);
    }
    return c.stats().hits;
}

void test_lru_beats_fifo_on_a_designed_trace() {
    // trace：1 2 1 3 1 2 1 3 …… 两路一组。手算：LRU 命中 7 次（第 2、4、6、8、10、12、14 次访问），
    // FIFO 命中 4 次（第 2、6、10、14 次）——FIFO 会把刚要用的键按插入序换掉。
    std::vector<std::uint32_t> trace;
    for (int r = 0; r < 4; ++r) {
        trace.push_back(1);
        trace.push_back(2);
        trace.push_back(1);
        trace.push_back(3);
    }
    FifoPolicy fifo;
    LruPolicy lru;
    ExpertCache c_fifo(2, 2, fifo);
    ExpertCache c_lru(2, 2, lru);
    const std::uint64_t h_fifo = drive(c_fifo, trace);
    const std::uint64_t h_lru = drive(c_lru, trace);
    if (h_fifo != 4 || h_lru != 7) {
        std::printf("FAIL 手算命中数：fifo=%llu（应 4） lru=%llu（应 7）\n",
                    static_cast<unsigned long long>(h_fifo),
                    static_cast<unsigned long long>(h_lru));
        CHECK(false);
    }
    CHECK(h_lru > h_fifo);
    CHECK(c_fifo.stats().hits + c_fifo.stats().misses == trace.size());
    CHECK(c_lru.stats().hits + c_lru.stats().misses == trace.size());
}

void test_cooccurrence_policy_mechanism() {
    // 先在共现图里造出不对称：(A,B) 记 5 次、(A,C) 记 1 次。
    Cooccurrence cooc(1024);
    const ExpertKey ab[2] = {K(0, 1), K(0, 2)};
    const ExpertKey ac[2] = {K(0, 1), K(0, 3)};
    for (int i = 0; i < 5; ++i) cooc.observe(ab, 2);
    cooc.observe(ac, 2);
    CHECK(cooc.score(K(0, 1), K(0, 2)) == 5);
    CHECK(cooc.score(K(0, 2), K(0, 1)) == 5);  // 无向
    CHECK(cooc.score(K(0, 1), K(0, 3)) == 1);
    CHECK(cooc.score(K(0, 2), K(0, 3)) == 0);

    // 两个槽位：B（更早用过）、C（更近用过）。本步在用的是 A。
    SlotState slots[2];
    slots[0] = SlotState{K(0, 2), /*last_used=*/10, /*inserted=*/1, true};  // B
    slots[1] = SlotState{K(0, 3), /*last_used=*/11, /*inserted=*/2, true};  // C
    PolicyContext ctx;
    ctx.step = 12;
    ctx.cooc = &cooc;
    const ExpertKey step[1] = {K(0, 1)};  // 本步在读 A
    ctx.step_keys = step;
    ctx.step_keys_n = 1;

    CooccurrenceAwarePolicy co;
    int order[2] = {-1, -1};
    CHECK(co.rank(slots, 2, K(0, 4), ctx, order) == 2);
    CHECK(order[0] == 1 && order[1] == 0);  // 先换 C（与 A 只共现 1 次），B 留着（共现 5 次）

    LruPolicy lru;
    int lru_order[2] = {-1, -1};
    CHECK(lru.rank(slots, 2, K(0, 4), ctx, lru_order) == 2);
    CHECK(lru_order[0] == 0 && lru_order[1] == 1);  // LRU 先换更早用过的 B
    CHECK(order[0] != lru_order[0]);               // 两条策略确实给出不同顺序

    // 本步在用的键一律排最后：A 更老，但 LRU 会换它，共现感知不会。
    SlotState with_a[2];
    with_a[0] = SlotState{K(0, 1), /*last_used=*/5, /*inserted=*/1, true};   // A（更老）
    with_a[1] = SlotState{K(0, 9), /*last_used=*/11, /*inserted=*/9, true};  // 无关键
    int order2[2] = {-1, -1};
    CHECK(co.rank(with_a, 2, K(0, 4), ctx, order2) == 2);
    CHECK(order2[0] == 1);  // 先换无关键
    CHECK(order2[1] == 0);  // A 留到最后
}

void test_evictable_callback_and_blocked() {
    LruPolicy lru;
    ExpertCache c(2, 2, lru);  // 1 组 2 槽
    put(c, K(0, 1), 0);
    put(c, K(0, 2), 1);

    // 让 LRU 的第一人选（K(0,1)）不可驱逐：应换到第二人选 K(0,2)。
    struct Ctx {
        ExpertKey protected_key;
    };
    Ctx ctx{K(0, 1)};
    auto evictable = [](const ExpertKey& key, void* p) -> bool {
        return !(key == static_cast<Ctx*>(p)->protected_key);
    };
    ExpertKey victim;
    CHECK(c.access(K(0, 3), 2, evictable, &ctx, &victim) == ExpertCache::Outcome::kMissReplaced);
    CHECK(victim == K(0, 2));
    // K(0,1) 仍在（被保护），K(0,2) 被换出。
    ExpertKey v;
    CHECK(c.access(K(0, 1), 3, nullptr, nullptr, &v) == ExpertCache::Outcome::kHit);
    CHECK(c.access(K(0, 2), 4, nullptr, nullptr, &v) == ExpertCache::Outcome::kMissReplaced);

    // 两个都不可驱逐：返回 kMissBlocked，且不动任何槽位。
    Ctx all_protected{K(0, 3)};
    auto never = [](const ExpertKey&, void*) -> bool { return false; };
    ExpertKey v2;
    CHECK(c.access(K(0, 5), 5, never, &all_protected, &v2) == ExpertCache::Outcome::kMissBlocked);
    CHECK(c.stats().blocked == 1);
    CHECK(c.size() == 2);
}

void test_cooccurrence_bounds() {
    // 上限 3 对、不衰减：满之后拒收新对，已有的对照常计数（不做全表扫描找最弱，见头文件）。
    Cooccurrence cooc(3, /*aging_interval=*/0);
    const ExpertKey p1[2] = {K(0, 1), K(0, 2)};
    const ExpertKey p2[2] = {K(0, 3), K(0, 4)};
    const ExpertKey p3[2] = {K(0, 5), K(0, 6)};
    cooc.observe(p1, 2);
    cooc.observe(p2, 2);
    cooc.observe(p2, 2);
    cooc.observe(p3, 2);
    cooc.observe(p3, 2);
    cooc.observe(p3, 2);
    CHECK(cooc.pairs() == 3);
    CHECK(cooc.score(K(0, 1), K(0, 2)) == 1);
    const ExpertKey p4[2] = {K(0, 7), K(0, 8)};
    cooc.observe(p4, 2);
    cooc.observe(p4, 2);
    CHECK(cooc.pairs() == 3);              // 不越界
    CHECK(cooc.dropped_pairs() == 2);      // 两次都被拒收
    CHECK(cooc.score(K(0, 7), K(0, 8)) == 0);
    CHECK(cooc.score(K(0, 3), K(0, 4)) == 2);  // 已有的对照样计数
    CHECK(cooc.score(K(0, 5), K(0, 6)) == 3);

    // 上限 0 = 不记录。
    Cooccurrence none(0);
    none.observe(p1, 2);
    CHECK(none.pairs() == 0 && none.score(K(0, 1), K(0, 2)) == 0);

    // 单元素组不产生对。
    Cooccurrence c1(64);
    const ExpertKey one[1] = {K(0, 1)};
    c1.observe(one, 1);
    CHECK(c1.pairs() == 0);
}

void test_cooccurrence_aging_frees_room() {
    // 每 4 次观察衰减一次：计数 1 的对会归零退场，于是新对又能进来。
    Cooccurrence cooc(2, /*aging_interval=*/4);
    const ExpertKey a[2] = {K(0, 1), K(0, 2)};
    const ExpertKey b[2] = {K(0, 3), K(0, 4)};
    const ExpertKey c[2] = {K(0, 5), K(0, 6)};
    cooc.observe(a, 2);  // 第 1 次观察：(A)=1
    cooc.observe(b, 2);  // 第 2 次观察：(A)=1 (B)=1
    CHECK(cooc.pairs() == 2 && cooc.dropped_pairs() == 0);
    cooc.observe(c, 2);  // 第 3 次观察：满了，C 被拒
    CHECK(cooc.dropped_pairs() == 1 && cooc.score(K(0, 5), K(0, 6)) == 0);
    cooc.observe(c, 2);  // 第 4 次观察后触发衰减：A、B 计数减半归零 -> 表空出来
    CHECK(cooc.agings() == 1);
    cooc.observe(c, 2);  // 表已空，C 这次能进来了
    CHECK(cooc.score(K(0, 5), K(0, 6)) == 1);
    CHECK(cooc.pairs() == 1);
}

void test_observe_step_feeds_the_policy() {
    // 端到端：让缓存在一条「两个簇交替」的 trace 上自己积累共现，确认 observe_step 真的喂进去了。
    CooccurrenceAwarePolicy co;
    ExpertCache c(4, 4, co);  // 1 组 4 槽
    const ExpertKey tok[3] = {K(0, 1), K(0, 2), K(0, 3)};
    c.observe_step(tok, 3);
    CHECK(c.cooccurrence().pairs() == 3);  // C(3,2) = 3 对
    CHECK(c.cooccurrence().score(K(0, 1), K(0, 2)) == 1);
}

void test_cooccurrence_aging_halves_counts() {
    // 衰减是「减半」而不是别的：计数 4 的那一对在衰减后必须是 2（不是 1，也不是 0）。
    Cooccurrence cooc(8, /*aging_interval=*/4);
    const ExpertKey a[2] = {K(0, 1), K(0, 2)};
    cooc.observe(a, 2);  // 1
    cooc.observe(a, 2);  // 2
    cooc.observe(a, 2);  // 3
    CHECK(cooc.score(K(0, 1), K(0, 2)) == 3);
    cooc.observe(a, 2);  // 4 -> 触发衰减：4 >> 1 = 2
    CHECK(cooc.agings() == 1);
    CHECK(cooc.score(K(0, 1), K(0, 2)) == 2);
}

void test_centrality_carries_no_information_beyond_frequency() {
    // 固定 top-k 的路由下，一次激活恰好带来 (k−1) 个伙伴对，故「中心度 == (k−1) × 激活频次」——
    // 中心度不含超出边缘频次的信息，用它排序与用频次排序同序。这是本轮实测「共现图作预载/替换输入
    // 无收益」的结构性原因，用不变量钉住，免得日后有人凭直觉再试一遍。
    constexpr int k = 3;
    // 一批 k=3 的 token，成员不同但规模固定。
    const std::vector<std::vector<std::uint32_t>> tokens = {
        {1, 2, 3}, {1, 2, 3}, {2, 3, 4}, {1, 4, 5}, {1, 2, 5},
        {3, 4, 5}, {1, 3, 5}, {2, 4, 5}, {1, 2, 4}, {1, 2, 3},
    };
    Cooccurrence cooc(1u << 12, /*aging_interval=*/0);
    std::vector<ExpertKey> keys;
    std::vector<std::uint64_t> freq(8, 0);
    for (const auto& tk : tokens) {
        keys.clear();
        for (std::uint32_t e : tk) {
            keys.push_back(K(0, e));
            ++freq[e];
        }
        cooc.observe(keys.data(), static_cast<int>(keys.size()));
    }
    for (std::uint32_t e = 1; e <= 5; ++e) {
        const std::uint64_t want = static_cast<std::uint64_t>(k - 1) * freq[e];
        const std::uint64_t got = cooc.degree_weight(K(0, e));
        if (got != want) {
            std::printf("FAIL 中心度 e=%u：得 %llu，应 (k−1)×频次 = %llu\n", e,
                        static_cast<unsigned long long>(got),
                        static_cast<unsigned long long>(want));
            CHECK(false);
        }
    }
    // 没有出现过的键中心度为 0。
    CHECK(cooc.degree_weight(K(0, 7)) == 0);

    // 排序同序：频次降序与中心度降序给出同一批键（并列都取小键号）。
    std::vector<ExpertKey> top;
    cooc.centrality_ranking(top, 5);
    std::vector<std::uint32_t> got;
    for (const ExpertKey& kk : top) got.push_back(kk.expert);
    // 频次：1 与 2 各 7，3 是 6，4 与 5 各 5 -> 中心度同序，并列取小键号。
    CHECK(freq[1] == 7 && freq[2] == 7 && freq[3] == 6 && freq[4] == 5 && freq[5] == 5);
    CHECK(got.size() == 5u);
    for (std::size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == static_cast<std::uint32_t>(i + 1));
    }
}

}  // namespace

int main() {
    test_geometry_and_key();
    test_free_way_then_replace_within_set();
    test_lru_beats_fifo_on_a_designed_trace();
    test_cooccurrence_policy_mechanism();
    test_evictable_callback_and_blocked();
    test_cooccurrence_bounds();
    test_cooccurrence_aging_frees_room();
    test_cooccurrence_aging_halves_counts();
    test_observe_step_feeds_the_policy();
    test_centrality_carries_no_information_beyond_frequency();
    std::puts("expert_cache: geometry, policies, page-table invariants and bounded cooccurrence hold");
    return 0;
}