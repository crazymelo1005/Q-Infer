// 一层的一批未命中专家：把字节取齐 → 选一组交出去 → 按专家 id 全序归并。
//
// 这是 interfaces §8 那条缝的落地：`experts/compute_group` 定了「组」与选组，`experts/expert_worker`
// 定了「作业」与「工人」（组的提交面），`experts/expert_source` 定了「字节从哪来」，本类把它们接起来，
// 并负责归并顺序与份额的退回。
//
// 四件事各有各的坑，故分开写：
//   1. 字节的存活期。`ExpertSource::get` 对常驻专家给的是缓存内的指针，对走临时缓冲的给的是
//      **下一次 get 就会被覆写**的那个共享缓冲。一批要同时持有 n 个专家的字节，故常驻的当场用页表
//      引用计数钉住（pin，被钉住的不会被选作牺牲者）、非常驻的拷进批内缓冲。只做一半就会读到别人的
//      权重——这是静默错，故测试里有一条「预算极小、全走临时缓冲」的用例专门抓它。
//   2. 一个汇合点。工人的 wait() 在这里调用，且只在这里调用（interfaces §8.4）；本函数不含第二条
//      同步路径。批量粒度是一层一批。
//   3. 归并顺序。结果按**专家 id 升序**并入 out，与 `run_moe_layer` 同序——三路计算（显存侧命中、
//      CPU 侧未命中、核显分担）必须按同一顺序并入，否则同配置下无法逐位复现（engine §16）。
//   4. 份额的退回。持份额限流器的主机侧组在选组时就把这一批的字节记进了本步份额（interfaces §8.3）；
//      作业被改派或被拒时那次搬运没有发生，故要原额退回，否则份额会被白吃的尝试耗掉。
//
// 本函数只算路由专家那一批，不含共享专家：共享专家在显存侧算（engine §6），由调用方用
// `add_shared_expert` 另加。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "experts/compute_group.hpp"
#include "experts/expert_source.hpp"
#include "experts/moe_layer.hpp"

namespace qinfer::experts {

// 一批的中间缓冲。三项都对得上上面那几个坑：批内字节副本、钉住的专家、工人写结果的区。
struct LayerDispatchScratch {
    std::vector<int> order;                        // 该批按 id 升序的排列（归并顺序的唯一依据）
    std::vector<ExpertWeights> weights;            // 与 order 同序的字节指针
    std::vector<std::vector<std::uint8_t>> owned;  // 非常驻专家的字节副本，按 order 同序
    std::vector<std::uint64_t> pinned;             // 本批钉住的专家，算完逐个 unpin
    std::vector<float> batch_out;                  // n × hidden，工人写这里
};

// 一批的分段耗时（只作观测，给了才计时）。三段互斥且穷尽，把「交出去」这一件事拆成：
//   fetch  从字节来源取齐本批的字节（常驻的按住引用计数、非常驻的拷进批内缓冲）
//   worker 选中组的 submit + 汇合点 wait（设备侧的内核与同步都在这里面）
//   merge  按专家 id 升序并入 out
// 拆它的理由：设备侧整步比主机池子慢一倍多，而「设备慢」与「取数/归并慢」的修法完全不同。
struct DispatchPhases {
    double fetch_ms = 0.0;
    double worker_ms = 0.0;
    double merge_ms = 0.0;
};

// ids / route_weight 各 n 项，顺序任意（重复的 id 由调用方保证不出现——路由取前 k 保证互异）。
// out 长 hidden：本函数只往里累加，初值由调用方负责（路由侧按权重、共享侧另加）。
// 候选里的非兜底组比兜底组（CPU）先试（interfaces §8.1）；选中的组失败时改派兜底组
// （interfaces §8.5），重派次数记进 rs。
bool dispatch_layer_batch(const MoeLayer& layer, ExpertSource& src,
                          const std::vector<ComputeGroup>& candidates, const int* ids,
                          const float* route_weight, int n, const float* x, float* out,
                          LayerDispatchScratch& s, RerouteStats& rs, std::string& err,
                          DispatchPhases* phases = nullptr);

}  // namespace qinfer::experts
