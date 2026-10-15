// 两个 KV 池的记账与准入（interfaces.md 第 5 节）。不引第三方框架。
//
// 五条：两块 KV 独立记账（填满一个不动另一个）；任一池不足即拒绝且指得出是哪个池；QSA 侧驱逐只清
// 常驻窗口、序列还在（改全流式读），GDN 侧驱逐整个序列；usage 按该节原文返回余量；最后一条是画像
// 互校——用画像的两条独立数字（kv_full_mib 与 kv_resident_mib）反推同一个 cell 字节数，必须一致。
#include "kv/kv_pool.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace qinfer::kv;

namespace {

constexpr std::uint64_t kMib = 1024ull * 1024ull;

PoolConfig make_cfg() {
    PoolConfig c;
    c.gdn_limit_bytes = 64 * kMib;
    c.qsa_limit_bytes = 256 * kMib;
    c.gdn_bytes_per_sequence = 8 * kMib;   // 与长度无关
    c.qsa_bytes_per_cell = 1024;           // 一个 token 在一个 QSA 层占 1 KiB（测试用的圆整值）
    c.qsa_resident_cell_cap = 4096;        // 常驻最多 4096 个 cell
    return c;
}

void test_two_pools_are_independent() {
    Pools p(make_cfg());
    CHECK(p.used(Pool::kGdnState) == 0 && p.used(Pool::kQsaKv) == 0);
    CHECK(p.remaining(Pool::kGdnState) == 64 * kMib);
    CHECK(p.remaining(Pool::kQsaKv) == 256 * kMib);

    // 8 个序列 × 8 MiB = 64 MiB：刚好把 GDN 池占满，QSA 侧只动了其中一部分。
    for (std::uint64_t s = 1; s <= 8; ++s) {
        CHECK(p.admit(s, /*context_tokens=*/1024));
    }
    CHECK(p.remaining(Pool::kGdnState) == 0);
    CHECK(p.used(Pool::kGdnState) == 64 * kMib);
    CHECK(p.used(Pool::kQsaKv) == 8 * 1024 * 1024);  // 8 个序列各 1024 cell × 1 KiB
    CHECK(p.remaining(Pool::kQsaKv) == 256 * kMib - 8 * kMib);

    // GDN 满了：第九个序列被拒，且指得出是哪个池。
    std::string why;
    CHECK(!p.admit(9, 1024, &why));
    CHECK(why.find("GDN") != std::string::npos);

    // 释放一个序列后又能进；两个池的账都跟着走。
    CHECK(p.release(1));
    CHECK(p.remaining(Pool::kGdnState) == 8 * kMib);
    CHECK(p.used(Pool::kQsaKv) == 7 * kMib);
    CHECK(p.admit(9, 1024));
    CHECK(!p.release(1));  // 已经不在池里
}

void test_qsa_limit_and_resident_cap() {
    PoolConfig c = make_cfg();
    c.gdn_limit_bytes = 1024 * kMib;  // GDN 很宽，专门压 QSA
    Pools p(c);
    // 常驻窗口上限 4096 cell：上下文再长，常驻也只有 4096 × 1 KiB = 4 MiB。
    CHECK(p.resident_bytes_for(100000) == 4096 * 1024);
    CHECK(p.total_bytes_for(100000) == 100000 * 1024);
    CHECK(p.resident_bytes_for(100) == 100 * 1024);

    // 256 MiB / 4 MiB = 64 个序列刚好占满 QSA 池。
    for (std::uint64_t s = 1; s <= 64; ++s) CHECK(p.admit(s, 100000));
    CHECK(p.remaining(Pool::kQsaKv) == 0);
    std::string why;
    CHECK(!p.admit(65, 100000, &why));
    CHECK(why.find("QSA") != std::string::npos);

    // QSA 驱逐：只清常驻窗口，序列仍在（且这条序列还能继续被查到）。
    const std::uint64_t freed = p.evict(Pool::kQsaKv);
    CHECK(freed == 4096 * 1024);
    CHECK(p.remaining(Pool::kQsaKv) == freed);
    CHECK(p.sequences() == 64);
    const SequenceEntry* e = p.query(1);
    CHECK(e != nullptr && e->qsa_resident_bytes == 0);
    CHECK(e->qsa_total_bytes == 100000 * 1024);  // 全量 KV 还在，只是不再常驻
    // 再驱逐一次：第二个序列（并列取 id 最小）的常驻被清零。
    CHECK(p.evict(Pool::kQsaKv) == 4096 * 1024);
    CHECK(p.query(2) != nullptr && p.query(2)->qsa_resident_bytes == 0);
}

void test_gdn_evict_removes_the_whole_sequence() {
    Pools p(make_cfg());
    for (std::uint64_t s = 1; s <= 4; ++s) CHECK(p.admit(s, 512));
    CHECK(p.sequences() == 4);
    const std::uint64_t before_qsa = p.used(Pool::kQsaKv);
    const std::uint64_t freed = p.evict(Pool::kGdnState);
    CHECK(freed == 8 * kMib);                 // GDN 侧全有或全无：一个序列的整份
    CHECK(p.sequences() == 3);                // 序列被赶走
    CHECK(p.query(1) == nullptr);             // 占用并列时取 id 最小的
    CHECK(p.used(Pool::kQsaKv) == before_qsa - 512 * 1024);  // QSA 侧那份也一起释放
    // 空池上没有可驱逐的。
    Pools empty(make_cfg());
    CHECK(empty.evict(Pool::kGdnState) == 0);
    CHECK(empty.evict(Pool::kQsaKv) == 0);
}

void test_usage_returns_remaining() {
    // §5 原文写作「usage(pool) 返回余量」，故这里必须与 remaining 一致（而不是与 used 一致）。
    Pools p(make_cfg());
    CHECK(p.usage(Pool::kQsaKv) == p.remaining(Pool::kQsaKv));
    CHECK(p.admit(1, 1024));
    CHECK(p.usage(Pool::kQsaKv) == p.remaining(Pool::kQsaKv));
    CHECK(p.usage(Pool::kQsaKv) == 256 * kMib - kMib);
    CHECK(p.used(Pool::kQsaKv) == kMib);
}

void test_cell_bytes_cross_checks_the_profile() {
    // 画像两条互相独立的数字：kv_full_mib = max_context × cell × n_qsa = 3168；
    //                          kv_resident_mib = resident_cells × cell × n_qsa = 396。
    // 由第一条解出 cell，再用它复现第二条——两条必须一致，否则画像内部就不自洽。
    constexpr std::uint64_t kMaxContext = 262144;
    constexpr std::uint64_t kNQsa = 12;
    constexpr std::uint64_t kResidentCells = 32768;
    constexpr double kKvFullMib = 3168.0;
    constexpr double kKvResidentMib = 396.0;

    const double cell = kKvFullMib * static_cast<double>(kMib) /
                        (static_cast<double>(kMaxContext) * static_cast<double>(kNQsa));
    CHECK(std::fabs(cell - 1056.0) < 1e-9);  // 画像反推出的一 cell 一层的字节数

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(std::llround(cell));
    const double resident_mib =
        static_cast<double>(kResidentCells * cell_bytes * kNQsa) / static_cast<double>(kMib);
    CHECK(std::fabs(resident_mib - kKvResidentMib) < 1e-9);  // 用解出的 cell 复现第二条
    // 顺带：用画像参数建的两个池，常驻窗口一满就是画像那条 396 MiB。
    PoolConfig c;
    c.qsa_bytes_per_cell = cell_bytes;
    c.qsa_resident_cell_cap = static_cast<std::uint32_t>(kResidentCells);
    c.qsa_limit_bytes = kResidentCells * cell_bytes;  // 上限就设成常驻窗口那一份
    c.gdn_limit_bytes = 1;
    c.gdn_bytes_per_sequence = 0;  // GDN 侧不参与这条
    Pools p(c);
    CHECK(p.used(Pool::kQsaKv) == 0);
    CHECK(p.admit(1, kMaxContext));
    CHECK(p.used(Pool::kQsaKv) == kResidentCells * cell_bytes);
    CHECK(p.remaining(Pool::kQsaKv) == 0);
}

void test_used_never_exceeds_the_limit() {
    // 不变量：admit 先查余量，故无论怎么混合 admit / evict / release，used 都不超过 limit
    // （remaining 的实现正依赖这一条，故它不该再防守一次）。
    Pools p(make_cfg());
    std::uint64_t id = 1;
    for (int round = 0; round < 400; ++round) {
        p.admit(id++, 700 + static_cast<std::uint64_t>(round % 97));
        if (round % 3 == 0) p.evict(Pool::kQsaKv);
        if (round % 7 == 0) p.evict(Pool::kGdnState);
        if (round % 11 == 0) p.release(id / 2);
        CHECK(p.used(Pool::kGdnState) <= p.config().gdn_limit_bytes);
        CHECK(p.used(Pool::kQsaKv) <= p.config().qsa_limit_bytes);
        CHECK(p.remaining(Pool::kGdnState) == p.config().gdn_limit_bytes - p.used(Pool::kGdnState));
        CHECK(p.remaining(Pool::kQsaKv) == p.config().qsa_limit_bytes - p.used(Pool::kQsaKv));
    }
}

}  // namespace

int main() {
    test_two_pools_are_independent();
    test_qsa_limit_and_resident_cap();
    test_gdn_evict_removes_the_whole_sequence();
    test_usage_returns_remaining();
    test_cell_bytes_cross_checks_the_profile();
    test_used_never_exceeds_the_limit();
    std::puts("kv_pool: 两池独立记账、任一不足即拒、两类驱逐、usage 按余量、画像互校通过");
    return 0;
}