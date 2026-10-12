// 见 memory_plan.hpp 的口径说明（为什么只做「专家 → KV」这一步、为什么 KV 的地板由几何算）。
#include "scheduling/memory_plan.hpp"

#include <cstddef>

namespace qinfer::scheduling {

namespace {

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

// 一次 decode 步的 KV 工作集：本步刚写的页 + 本步要读的页。选择宽度 `min(上下文, 上界)` 个 cell 最多
// 落在 `ceil(宽度 / 页大小)` 个页上，再加当前这一页。取整到整页（窗口是按页常驻的）。
std::uint64_t kv_floor_cells(const PoolGeometry& g) {
    const std::uint64_t width = g.context_tokens < g.selection_width ? g.context_tokens : g.selection_width;
    const std::uint64_t pages = 1 + ceil_div(width, g.page_size);
    const std::uint64_t cells = pages * g.page_size;
    return cells < g.context_tokens ? cells : g.context_tokens;
}

std::uint64_t kv_target_cells(const PoolGeometry& g) {
    std::uint64_t cap = g.resident_cap_cells == 0 ? g.context_tokens : g.resident_cap_cells;
    if (cap > g.context_tokens) cap = g.context_tokens;
    return cap;
}

// 一个序列常驻 n 个 cell 要多少字节（跨全部 KV 层、乘上并发序列数）。
std::uint64_t kv_bytes_for(const PoolGeometry& g, std::uint64_t cells) {
    return cells * g.kv_bytes_per_cell_per_layer * g.kv_layers * g.concurrency;
}

std::uint64_t expert_bytes_for(const PoolGeometry& g, std::uint64_t slots) {
    return slots * g.expert_bytes_per_slot;
}

}  // namespace

const char* to_string(PoolDegrade d) {
    switch (d) {
        case PoolDegrade::kNone: return "两侧都够";
        case PoolDegrade::kExpertSlotsCut: return "只削了专家槽位";
        case PoolDegrade::kKvWindowCut: return "专家已到 0，再削 KV 常驻窗";
        case PoolDegrade::kRefused: return "连 KV 硬地板都装不下";
    }
    return "未知";
}

bool plan_pool(const PoolGeometry& g, PoolPlan& out, std::string& err) {
    err.clear();
    out = PoolPlan{};
    if (g.pool_bytes == 0) {
        err = "池配额是 0";
        return false;
    }
    if (g.page_size == 0 || g.kv_layers == 0 || g.kv_bytes_per_cell_per_layer == 0 ||
        g.context_tokens == 0) {
        err = "KV 的几何不成立（页大小 / 层数 / 每 cell 字节 / 上下文有零）";
        return false;
    }
    if (g.concurrency == 0) {
        err = "并发序列数是 0";
        return false;
    }

    out.kv_floor_cells = kv_floor_cells(g);
    out.kv_target_cells = kv_target_cells(g);
    const std::uint64_t floor_bytes = kv_bytes_for(g, out.kv_floor_cells);
    if (floor_bytes > g.pool_bytes) {
        // 硬地板都装不下：拒绝，不静默配一个装不下的数（interfaces §5 的「任一不足即拒绝新序列」）。
        out.degrade = PoolDegrade::kRefused;
        out.kv_bytes = floor_bytes;
        err = "KV 的硬地板（" + std::to_string(out.kv_floor_cells) + " cell / 序列、共 " +
              std::to_string(floor_bytes) + " 字节）超过池配额 " + std::to_string(g.pool_bytes) +
              " 字节";
        return false;
    }

    // 先削专家：KV 保持目标，专家拿剩下的（降到 0 为止）。
    const std::uint64_t kv_target_bytes = kv_bytes_for(g, out.kv_target_cells);
    if (kv_target_bytes <= g.pool_bytes) {
        const std::uint64_t room = g.pool_bytes - kv_target_bytes;
        const std::uint64_t by_room = g.expert_bytes_per_slot == 0 ? g.expert_target_slots
                                                                   : room / g.expert_bytes_per_slot;
        out.expert_slots = by_room < g.expert_target_slots ? by_room : g.expert_target_slots;
        out.kv_window_cells = out.kv_target_cells;
        out.expert_bytes = expert_bytes_for(g, out.expert_slots);
        out.kv_bytes = kv_target_bytes;
        out.degrade = out.expert_slots < g.expert_target_slots ? PoolDegrade::kExpertSlotsCut
                                                              : PoolDegrade::kNone;
        out.leftover_bytes = g.pool_bytes - out.total_bytes();
        return true;
    }

    // 连 KV 的目标都装不下（专家已无处可削）：这时才动 KV，从目标往下削、以硬地板为限。
    const std::uint64_t per_cell = kv_bytes_for(g, 1);
    std::uint64_t kv_cells = per_cell == 0 ? 0 : g.pool_bytes / per_cell;
    if (kv_cells > out.kv_target_cells) kv_cells = out.kv_target_cells;
    if (kv_cells < out.kv_floor_cells) kv_cells = out.kv_floor_cells;  // 上面已保证地板装得下
    out.expert_slots = 0;
    out.expert_bytes = 0;
    out.kv_window_cells = kv_cells;
    out.kv_bytes = kv_bytes_for(g, kv_cells);
    out.degrade = PoolDegrade::kKvWindowCut;
    out.leftover_bytes = g.pool_bytes - out.total_bytes();
    return true;
}

}  // namespace qinfer::scheduling