// 内存池配平的回归（interfaces §8 那条待定项）。不引第三方框架。
//
// 五路证据：
//   1. 装得下就给目标；装不下时**先削专家、后削 KV**——顺序不许破（KV 在被削之前一直不低于它的硬地板，
//      而专家已经削到 0）；
//   2. 连 KV 的硬地板都装不下时**拒绝**（返回假 + err 说清），不静默配一个装不下的数；
//   3. 池变大时两侧都不减（单调），且任何一份计划的总量都不超池；
//   4. 硬地板的公式与上一片的实测对得上：`kv_window` 的合成夹具（页大小 2、选择宽度 5）算出 4 个槽位，
//      而那里「4 个槽位刚好、1 个槽位被拒」——规则算出来的正是那条边界；
//   5. 上下文变长时 KV 目标变大、固定池下专家槽位随之变小（这就是「弹性」本身）。
//
// `--g11` 入口：按真几何把 G-11 那张主机内存表算出来（纯算术，不碰文件）。
#include "scheduling/memory_plan.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace qinfer::scheduling;

namespace {

constexpr std::uint64_t kMiB = 1024 * 1024;

// 真几何（部署那份，[S-54]/[S-57] 与画像）：12 个 QSA 层、每 token 每层 1056 字节、页大小 4、
// 选择宽度上界 2048 + 4 - 1 = 2051。
PoolGeometry real_geometry(std::uint64_t pool_bytes, std::uint64_t context,
                           std::uint64_t concurrency) {
    PoolGeometry g;
    g.pool_bytes = pool_bytes;
    g.kv_layers = 12;
    g.kv_bytes_per_cell_per_layer = 1056;
    g.page_size = 4;
    g.selection_width = 2051;
    g.context_tokens = context;
    g.concurrency = concurrency;
    // 一个专家槽位跨 48 层三矩阵之和（部署那份一个专家约 1.5 MB，见 engine §6 的实测口径）。
    g.expert_bytes_per_slot = 48ull * 1536 * 1024;
    g.expert_target_slots = 32;
    return g;
}

// 错读法：把顺序反过来（先削 KV、后削专家）。只在两刀都要动的场景里给得出不同答案。
void plan_reversed_order(const PoolGeometry& g, PoolPlan& out) {
    out = PoolPlan{};
    out.kv_floor_cells = 0;
    // 直接把 KV 削到「装得下专家目标」再均值分配，是「先动 KV」的写法。
    const std::uint64_t expert_bytes = g.expert_target_slots * g.expert_bytes_per_slot;
    const std::uint64_t room = g.pool_bytes > expert_bytes ? g.pool_bytes - expert_bytes : 0;
    const std::uint64_t per_cell = g.kv_bytes_per_cell_per_layer * g.kv_layers * g.concurrency;
    out.kv_window_cells = per_cell == 0 ? 0 : room / per_cell;
    out.expert_slots = g.expert_target_slots;
    out.expert_bytes = expert_bytes;
    out.kv_bytes = out.kv_window_cells * per_cell;
    out.leftover_bytes = g.pool_bytes - out.expert_bytes - out.kv_bytes;
}

void test_fits_then_cuts_experts_then_kv() {
    std::string err;
    // 池给得很宽：两侧都拿目标。
    PoolGeometry g = real_geometry(64ull * 1024 * kMiB, 4096, 1);
    PoolPlan p;
    CHECK(plan_pool(g, p, err));
    CHECK(p.degrade == PoolDegrade::kNone);
    CHECK(p.expert_slots == g.expert_target_slots);
    CHECK(p.kv_window_cells == g.context_tokens);  // 上下文比常驻上限小 → 全常驻
    CHECK(p.total_bytes() <= g.pool_bytes);
    std::printf("  池 64 GiB、上下文 4096：两侧都够（专家 %llu 槽、KV %llu cell）\n",
                (unsigned long long) p.expert_slots, (unsigned long long) p.kv_window_cells);

    // 池只够 KV 目标 + 一部分专家：削专家，KV 不动。
    // KV 目标 = 4096 cell × 1056 × 12 = 51.9 MB；专家一槽 = 48 × 1.5 MB = 72 MB。
    const std::uint64_t kv_target = 4096ull * 1056 * 12;
    PoolGeometry g2 = real_geometry(kv_target + 10 * 72 * 1024 * 1024, 4096, 1);
    PoolPlan p2;
    CHECK(plan_pool(g2, p2, err));
    CHECK(p2.degrade == PoolDegrade::kExpertSlotsCut);
    CHECK(p2.kv_window_cells == p2.kv_target_cells);  // KV 仍在目标上
    CHECK(p2.expert_slots < g2.expert_target_slots && p2.expert_slots > 0);
    CHECK(p2.kv_window_cells >= p2.kv_floor_cells);
    CHECK(p2.total_bytes() <= g2.pool_bytes);
    std::printf("  池只够 KV 目标 + 10 个专家：只削专家（%llu 槽），KV 仍 %llu cell（地板 %llu）\n",
                (unsigned long long) p2.expert_slots, (unsigned long long) p2.kv_window_cells,
                (unsigned long long) p2.kv_floor_cells);

    // 池连 KV 目标都装不下（但高于地板）：这时专家必须已经到 0，KV 才往下削（顺序不许破）。
    PoolGeometry g3 = real_geometry(kv_target * 2 / 3, 4096, 1);
    PoolPlan p3;
    CHECK(plan_pool(g3, p3, err));
    CHECK(p3.degrade == PoolDegrade::kKvWindowCut);
    CHECK(p3.expert_slots == 0);
    CHECK(p3.kv_window_cells < p3.kv_target_cells);
    CHECK(p3.kv_window_cells >= p3.kv_floor_cells);
    CHECK(p3.total_bytes() <= g3.pool_bytes);
    std::printf("  池只有 KV 目标的三分之二：专家到 0、KV 削到 %llu cell（目标 %llu、地板 %llu）\n",
                (unsigned long long) p3.kv_window_cells, (unsigned long long) p3.kv_target_cells,
                (unsigned long long) p3.kv_floor_cells);
}

void test_refuses_below_the_floor() {
    std::string err;
    PoolGeometry g = real_geometry(1, 4096, 1);  // 1 字节
    PoolPlan p;
    CHECK(!plan_pool(g, p, err));
    CHECK(err.find("硬地板") != std::string::npos);
    CHECK(p.degrade == PoolDegrade::kRefused);
    std::printf("  池连地板都装不下：拒绝（%s）\n", err.c_str());

    // 几何不成立也要明确失败。
    PoolGeometry bad = real_geometry(64 * kMiB, 4096, 1);
    bad.page_size = 0;
    CHECK(!plan_pool(bad, p, err));
    CHECK(err.find("几何不成立") != std::string::npos);
}

void test_monotone_and_within_pool() {
    std::string err;
    std::uint64_t prev_slots = 0, prev_cells = 0;
    // 起点要在硬地板之上（这个几何的地板 = 2056 cell × 1056 B × 12 层 × 4 序列 ≈ 104 MB）。
    for (std::uint64_t mib = 128; mib <= 8192; mib += 128) {
        PoolGeometry g = real_geometry(mib * kMiB, 32768, 4);
        PoolPlan p;
        CHECK(plan_pool(g, p, err));
        CHECK(p.total_bytes() <= g.pool_bytes);
        CHECK(p.kv_window_cells >= p.kv_floor_cells);
        CHECK(p.expert_slots >= prev_slots);
        CHECK(p.kv_window_cells >= prev_cells);
        prev_slots = p.expert_slots;
        prev_cells = p.kv_window_cells;
    }
    std::printf("  单调：池 128 MiB 到 8 GiB 扫一遍，两侧都不减、总量不超池\n");
}

void test_floor_matches_the_kv_window_fixture() {
    // 上一片 `kv_window` 的合成夹具：页大小 2、选择宽度 = idx_top_k 2 + idx_block 4 - 1 = 5、
    // 池 4 个槽位刚好、1 个槽位被拒。这里的公式必须算出同一条边界。
    PoolGeometry g;
    g.pool_bytes = 1ull << 40;
    g.kv_layers = 1;
    g.kv_bytes_per_cell_per_layer = 128;  // 夹具里一页 128 字节 / 2 个 cell
    g.page_size = 2;
    g.selection_width = 5;
    g.context_tokens = 64;
    g.concurrency = 1;
    PoolPlan p;
    std::string err;
    CHECK(plan_pool(g, p, err));
    CHECK(p.kv_floor_cells == 8);          // 1 + ceil(5/2) = 4 页 × 2 = 8 个 cell
    CHECK(p.kv_floor_cells / g.page_size == 4);  // = 夹具里那 4 个槽位
    std::printf("  地板公式：页大小 2、宽度 5 → %llu cell = %llu 个槽位（夹具里正是 4 个槽位刚好、"
                "再少一页被拒）\n",
                (unsigned long long) p.kv_floor_cells,
                (unsigned long long) (p.kv_floor_cells / g.page_size));
}

void test_elasticity_across_context() {
    std::string err;
    const std::uint64_t pool = 2ull * 1024 * kMiB;
    std::printf("  弹性（池固定 2 GiB、单序列）：\n");
    std::uint64_t prev_slots = 0;
    for (std::uint64_t ctx : {std::uint64_t{1024}, std::uint64_t{4096}, std::uint64_t{32768},
                              std::uint64_t{262144}}) {
        PoolGeometry g = real_geometry(pool, ctx, 1);
        PoolPlan p;
        CHECK(plan_pool(g, p, err));
        std::printf("    上下文 %7llu：KV 目标 %7llu cell（%6llu MiB）、专家 %2llu 槽、%s\n",
                    (unsigned long long) ctx, (unsigned long long) p.kv_target_cells,
                    (unsigned long long) (p.kv_bytes / kMiB), (unsigned long long) p.expert_slots,
                    to_string(p.degrade));
        CHECK(p.total_bytes() <= pool);
        if (ctx > 1024) CHECK(p.expert_slots <= prev_slots);  // 上下文越长，留给专家的越少
        prev_slots = p.expert_slots;
    }
}

void test_wrong_order_is_observable() {
    const std::uint64_t kv_target = 4096ull * 1056 * 12;
    PoolGeometry g = real_geometry(kv_target * 2 / 3, 4096, 1);
    PoolPlan right, wrong;
    std::string err;
    CHECK(plan_pool(g, right, err));
    plan_reversed_order(g, wrong);
    const bool differs = right.expert_slots != wrong.expert_slots ||
                         right.kv_window_cells != wrong.kv_window_cells;
    std::printf("  错读法「先削 KV 后削专家」：专家 %llu 对 %llu、KV %llu 对 %llu cell（可观测：%s）\n",
                (unsigned long long) right.expert_slots, (unsigned long long) wrong.expert_slots,
                (unsigned long long) right.kv_window_cells, (unsigned long long) wrong.kv_window_cells,
                differs ? "是" : "否");
    CHECK(differs);
    CHECK(wrong.expert_slots > right.expert_slots);  // 错读法把专家保住了，代价是 KV 掉到地板以下
    CHECK(wrong.kv_window_cells < right.kv_floor_cells);
}

// ---- G-11：按真几何把主机内存表算出来 ----

int g11_entry() {
    struct Row { const char* name; std::uint64_t bytes; const char* note; };
    std::printf("G-11 主机内存表（按真几何算；可支配余量 6 至 16 GiB，见 engine §3.4）\n");
    for (std::uint64_t ctx : {std::uint64_t{4096}, std::uint64_t{32768}, std::uint64_t{262144}}) {
        for (std::uint64_t conc : {std::uint64_t{1}, std::uint64_t{8}}) {
            const std::uint64_t kv = ctx * 1056 * 12 * conc;
            const Row rows[] = {
                {"记忆表行缓存", 1048576ull * 90, "默认 1,048,576 行 × 90 B"},
                {"共现图", 0, "按会话级 top-K 稀疏格式（条目随会话数增长，不在本条口径内）"},
                {"记忆表中频行内存层", 0, "ADR-007 已砍（表留 NVMe，按需读）"},
                {"专家 pinned 区", 0, "本设计不 pin：mmap 直读走页缓存（engine §6）"},
                {"KV 权威副本（主机侧）", kv, "上下文 × 1056 B × 12 层 × 并发"},
                {"机器画像", 256 * 1024, "小于 1 MiB"},
                {"视觉编码器主机张量", 0, "约 1.4 GB，按需载入载出、不常驻"},
            };
            std::uint64_t total = 0;
            std::printf("  上下文 %llu、并发 %llu：\n", (unsigned long long) ctx,
                        (unsigned long long) conc);
            for (const Row& r : rows) {
                total += r.bytes;
                if (r.bytes > 0) {
                    std::printf("    %-22s %8.1f MiB  %s\n", r.name,
                                static_cast<double>(r.bytes) / static_cast<double>(kMiB), r.note);
                } else {
                    std::printf("    %-22s %8s      %s\n", r.name, "0", r.note);
                }
            }
            std::printf("    合计 %.2f GiB（余量 6 至 16 GiB：%s）\n",
                        static_cast<double>(total) / static_cast<double>(1024 * kMiB),
                        total <= 6ull * 1024 * kMiB ? "通过" : "超（按 §3.4 的顺序降级）");
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--g11") return g11_entry();
    test_fits_then_cuts_experts_then_kv();
    test_refuses_below_the_floor();
    test_monotone_and_within_pool();
    test_floor_matches_the_kv_window_fixture();
    test_elasticity_across_context();
    test_wrong_order_is_observable();
    std::puts("memory_plan: 先削专家后削 KV（顺序不许破）、地板装不下则拒绝、池变大两侧不减、"
              "地板公式与 kv_window 夹具的边界一致、上下文越长专家越少");
    return 0;
}