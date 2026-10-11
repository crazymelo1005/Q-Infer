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

// 一个 CUDA 组：显存侧组，不持总线份额限流器（它不是主机侧的组），非兜底。
ComputeGroup cuda_group(ExpertWorker& worker);

}  // namespace qinfer::experts
