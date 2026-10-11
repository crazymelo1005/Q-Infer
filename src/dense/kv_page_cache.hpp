// QSA 的 KV 分页缓存：把新 token 的 K/V 写进分页缓存，以及按页表把选中的 cell 收进连续缓冲。
//
// 布局与物理行公式取自参考引擎 `include/strata/kernels/qsa.hpp`（[S-57]、[S-59]）：
//
//   K/V 缓存：fp16、分页，布局 `[page][kv_head][page_size][head_dim]`，**head_dim 最快**；
//   逻辑 cell t、头 h 的物理行是
//       page = table[t / page_size];  (page * n_head_kv + h) * page_size + (t % page_size)
//   即**一个头的连续若干 cell 在页内连续**——这正是注意力要的性质（另一种排布是把一个 cell 的几个头
//   放一起，形状同样正确但访问模式相反）。
//
// 四处「都产出形状正确的结果」的地方，每处都在测试里按错的算一遍要求它明显偏离：
//   1. 页内排布是 head-major（一个头的连续 cell 连续），不是 cell-major；
//   2. 页表的下标是 `t / page_size`（逻辑页号），不是 cell 号；
//   3. 表里 `-1` 表示**该页不在显存**（流式档）——必须明确报错，不能当成页号 0 静默读；
//   4. 聚集按给定的 id 顺序（升序）产出连续的 n_ids 个 cell，不是按原 cell 号散着写。
//
// 一处范围：本文件只管「一页对一页」的搬运与寻址，不做驱逐也不做页的换入换出——那些属于存储层
// （[interfaces.md] 的页表与预算仲裁），接进来时再串。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 分页 KV 缓存的几何与物理缓冲。pool 是 fp16 位型（uint16），长度 = n_pages × n_head_kv × page_size × head_dim。
struct KvPageCache {
    std::uint64_t page_size = 0;   // 每页几个 cell（真模型 4）
    std::uint64_t n_head_kv = 0;   // KV 头数（真模型 2）
    std::uint64_t head_dim = 0;    // 头宽（真模型 256）
    std::uint64_t n_pages = 0;     // 物理页数
    std::vector<std::uint16_t> k_pool;
    std::vector<std::uint16_t> v_pool;
    std::vector<std::int32_t> table;  // 逻辑页 → 物理页；-1 = 不在显存（流式）

    void reset();
};

// 一个 token（一个 cell）的 K/V：各 n_head_kv × head_dim 个 fp16 值。
struct KvCell {
    const std::uint16_t* k = nullptr;
    const std::uint16_t* v = nullptr;
};

// 把 cell 号 t 的 K/V 写进缓存。t 的逻辑页必须在表里（`-1` 直接失败：那说明这一页要流式换入）。
bool kv_append(KvPageCache& c, std::uint64_t t, const KvCell& cell, std::string& err);

// 按 id 列表把选中的 cell 收进连续缓冲（第 i 个 id 的结果落在第 i 个 cell 槽位）。
// k_scratch/v_scratch 长 n_ids × n_head_kv × head_dim。
bool kv_gather(const KvPageCache& c, const std::uint64_t* ids, std::uint64_t n_ids,
               std::uint16_t* k_scratch, std::uint16_t* v_scratch, std::string& err);

}  // namespace qinfer::dense