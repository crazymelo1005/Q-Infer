// 一层的一批未命中专家：把字节取齐 → 一批交给工人 → 按专家 id 全序归并。
//
// 这是 interfaces §8 那条缝的落地：`experts/expert_worker` 定了「作业」与「工人」的形状，
// `experts/expert_source` 定了「字节从哪来」，本类把它们接起来，并负责归并顺序。
//
// 三件事各有各的坑，故分开写：
//   1. 字节的存活期。`ExpertSource::get` 对常驻专家给的是缓存内的指针，对走临时缓冲的给的是
//      **下一次 get 就会被覆写**的那个共享缓冲。一批要同时持有 n 个专家的字节，故常驻的当场用页表
//      引用计数钉住（pin，被钉住的不会被选作牺牲者）、非常驻的拷进批内缓冲。只做一半就会读到别人的
//      权重——这是静默错，故测试里有一条「预算极小、全走临时缓冲」的用例专门抓它。
//   2. 一个汇合点。工人的 wait() 在这里调用，且只在这里调用（interfaces §8.3）；本函数不含第二条
//      同步路径。批量粒度是一层一批。
//   3. 归并顺序。结果按**专家 id 升序**并入 out，与 `run_moe_layer` 同序——三路计算（显存侧命中、
//      CPU 侧未命中、核显分担）必须按同一顺序并入，否则同配置下无法逐位复现（engine §16）。
//
// 本函数只算路由专家那一批，不含共享专家：共享专家在显存侧算（engine §6），由调用方用
// `add_shared_expert` 另加。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "experts/expert_source.hpp"
#include "experts/expert_worker.hpp"
#include "experts/moe_layer.hpp"

namespace qinfer::experts {

// 一批的中间缓冲。三项都对得上上面那三个坑：批内字节副本、钉住的专家、工人写结果的区。
struct LayerDispatchScratch {
    std::vector<int> order;                        // 该批按 id 升序的排列（归并顺序的唯一依据）
    std::vector<ExpertWeights> weights;            // 与 order 同序的字节指针
    std::vector<std::vector<std::uint8_t>> owned;  // 非常驻专家的字节副本，按 order 同序
    std::vector<std::uint64_t> pinned;             // 本批钉住的专家，算完逐个 unpin
    std::vector<float> batch_out;                  // n × hidden，工人写这里
};

// ids / route_weight 各 n 项，顺序任意（重复的 id 由调用方保证不出现——路由取前 k 保证互异）。
// out 长 hidden：本函数只往里累加，初值由调用方负责（路由侧按权重、共享侧另加）。
// primary 不可用或失败时改派 fallback（interfaces §8.4），重派次数记进 rs。
bool dispatch_layer_batch(const MoeLayer& layer, ExpertSource& src, ExpertWorker& primary,
                          ExpertWorker& fallback, const int* ids, const float* route_weight, int n,
                          const float* x, float* out, LayerDispatchScratch& s, RerouteStats& rs,
                          std::string& err);

}  // namespace qinfer::experts