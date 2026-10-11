// 设备侧的专家工人：一块卡 + 按层索引的权重常驻区（interfaces §8.2 的提交面、ADR-011 的计算组）。
//
// 一个工人持有一块卡上若干层的常驻区（一层一批的作业带着层号，故按层号索引）。`submit` 收一层一批：
// 先把还没常驻的专家搬进空槽（只搬一次），再让 `ExpertFfnDevice` 在同一个流上把这一批算完；
// `wait` 是幂等的空操作（设备侧那批在 submit 内已经同步完），故「每步一个汇合点」不变。
//
// 能力表只是**必要**条件：能不能算还取决于角色（gate/up 与 down 的激活搭档不同），由工人自己的
// init 判定；判不过就返回假，由调用方按 §8.5 改派给兜底组。
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "experts/compute_group.hpp"
#include "experts/cuda/expert_ffn_cuda.hpp"

namespace qinfer::experts {

class CudaExpertWorker : public ExpertWorker {
public:
    // slots_per_layer：一层最多常驻多少个专家。不够就明确失败——驱逐是专家缓存那一层的事
    // （interfaces §5），不是工人的事。
    CudaExpertWorker(int device, int slots_per_layer);
    ~CudaExpertWorker() override;
    CudaExpertWorker(const CudaExpertWorker&) = delete;
    CudaExpertWorker& operator=(const CudaExpertWorker&) = delete;

    const char* name() const override { return "cuda"; }
    bool available() const override;
    bool submit(const ExpertJob& job, std::string& err) override;
    bool wait(std::string& err) override;
    const Stats& stats() const override { return stats_; }

    int device() const { return device_; }
    int slots_per_layer() const { return slots_per_layer_; }
    std::uint64_t uploaded_bytes() const;  // 观测：常驻之后不该再增长
    std::uint64_t resident_bytes() const;

private:
    struct LayerState;
    int device_ = -1;
    int slots_per_layer_ = 0;
    std::map<int, std::unique_ptr<LayerState>> layers_;
    Stats stats_;
};

// 设备侧内核覆盖的档（位掩码）：gate/up 的 IQ2_S / IQ2_XXS / IQ1_M 与 down 的 Q2_0。
std::uint32_t device_format_mask();

// 按 [ADR-004](../../docs/design/adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md) 的层切分把
// 层连续地分给各卡：`n_layers` 层切成 `device_count` 段，前 `n_layers % device_count` 段各多一层。
// 连续切分而不是轮流（layer % device_count）：层间只传残差栈，切口越少越好。`layer` 越界时按最后一段算。
int cuda_owner_of(int device_count, int n_layers, int layer);

// 一台机器上的设备组集合：每张卡一个工人（各自那块卡上的权重常驻区）与一个组；每张卡各有一份候选表
// 「本卡的组在前、CPU 兜底组在后」。它不管字节来源——调用方按层建 `MoeOffload` 时把候选表指进来。
class CudaGroupFleet {
public:
    // devices 是要用的卡（如 {0,1}）；任何一张不可用都返回假并写 err。cpu_fallback 由调用方持有。
    bool init(const std::vector<int>& devices, int slots_per_layer, ExpertWorker& cpu_fallback,
              std::string& err);

    int card_count() const { return static_cast<int>(workers_.size()); }
    int owner_of(int n_layers, int layer) const;
    // 第 owner 张卡的候选表（地址在 init 之后稳定，可直接给 MoeOffload.candidates）。
    const std::vector<ComputeGroup>& candidates_for(int owner) const;
    // 观测：第 owner 张卡的工人。
    ExpertWorker& worker(int owner) { return *workers_[static_cast<std::size_t>(owner)]; }
    int device_of(int owner) const;

private:
    std::vector<std::unique_ptr<CudaExpertWorker>> workers_;
    std::vector<std::vector<ComputeGroup>> candidates_;  // 一张卡一份
};

// 一个 CUDA 组：显存侧组，不持总线份额限流器（它不是主机侧的组），非兜底。
ComputeGroup cuda_group(ExpertWorker& worker);

}  // namespace qinfer::experts
