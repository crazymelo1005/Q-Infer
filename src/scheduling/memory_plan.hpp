// 内存池配平：把一组的内存池配额在「专家槽位」与「KV 常驻窗」之间分掉。
//
// 依据：interfaces §8 最后一段把这条待定项写成「槽位数在 KV 常驻窗口与专家数量之间配平，落到接口上就是
// 组的内存池配额」；§3.4 给了**固定的降级顺序**（表行缓存 → 共现图 → 表内存层 → 专家 pinned 区 → KV 池），
// 故这里只做最后两项之间的那一步：**先削专家、后削 KV**。本模块是纯函数——只出计划，不动任何池；真正
// 按计划动态重配（运行中改槽位数）属于设备侧那一半。
//
// KV 的硬地板不是拍的：一次 decode 步要同时装下「本步刚写的页」与「本步要读的页」，即
// `1 + ceil(选择宽度 / page_size)` 个页——低于它只剩「被拒」这一条路（`kv_window` 的合成夹具里 4 个
// 槽位正好等于这个数，再少一页就报错）。故地板由几何算出来，不由调用方给。
#pragma once

#include <cstdint>
#include <string>

namespace qinfer::scheduling {

// 一侧（或两侧）都按「字节 / 每单位」给几何，好让规则里只有除法与比较。
struct PoolGeometry {
    std::uint64_t pool_bytes = 0;  // 这一组的内存池配额（显存侧或主机侧，取决于这组挂在哪）

    // KV（QSA）侧
    std::uint64_t kv_layers = 0;                    // 有 KV 的层数（真模型 12）
    std::uint64_t kv_bytes_per_cell_per_layer = 0;  // 一个 token 在一层占的 K+V 字节（真模型 1056）
    std::uint64_t page_size = 0;                    // 一个 KV 页几个 cell（真模型 4）
    std::uint64_t selection_width = 0;              // 选择宽度 `idx_top_k + idx_block - 1` 的上界（[S-57]）
    std::uint64_t context_tokens = 0;               // 该序列的上下文
    std::uint64_t concurrency = 1;                  // 同时几条序列常驻
    std::uint64_t resident_cap_cells = 0;           // 常驻窗上限（0 = 不限，取上下文）

    // 专家侧
    std::uint64_t expert_bytes_per_slot = 0;  // 多一个槽位、跨全部专家层要多少字节（由调用方逐层求和）
    std::uint64_t expert_target_slots = 0;    // 由命中率目标反推的目标槽位数
};

enum class PoolDegrade : std::uint8_t {
    kNone = 0,         // 两侧都拿到目标
    kExpertSlotsCut,   // 只削了专家（KV 仍在目标上）
    kKvWindowCut,      // 专家已削到 0，只好再削 KV
    kRefused,          // 连 KV 的硬地板都装不下：拒绝，不静默配一个装不下的数
};

const char* to_string(PoolDegrade d);

struct PoolPlan {
    std::uint64_t expert_slots = 0;     // 计划给的专家槽位数（0 = 专家全在 CPU 路径上，§8.5）
    std::uint64_t kv_window_cells = 0;  // 计划给每个序列的 KV 常驻窗（cell 数）
    std::uint64_t kv_floor_cells = 0;   // 硬地板（= 本步工作集；供观测与断言）
    std::uint64_t kv_target_cells = 0;  // 未削之前的期望（= min(上下文, 常驻上限)）
    std::uint64_t expert_bytes = 0;
    std::uint64_t kv_bytes = 0;
    std::uint64_t leftover_bytes = 0;
    PoolDegrade degrade = PoolDegrade::kNone;

    std::uint64_t total_bytes() const { return expert_bytes + kv_bytes; }
};

// 算一份计划。几何不成立（池为 0、页大小或层数为 0 等）或连 KV 地板都装不下时返回假并写 err。
bool plan_pool(const PoolGeometry& g, PoolPlan& out, std::string& err);

}  // namespace qinfer::scheduling