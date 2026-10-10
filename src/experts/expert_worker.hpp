// 专家分担的作业与工人（interfaces.md 第 8 节）。
//
// 语义边界：worker 的语义是「算这批专家的人」，CPU 与核显同形，调度器按算力强度与带宽强度分派
// （proposals.md 的 P-04）。这里不引入设备概念，也不引入第二条同步路径——wait() 由步循环在**同一步
// 的汇合点**调用（interfaces §8.3）。
//
// 粒度是一层一批：每步至多 48 次提交，队列开销可忽略；不再切细，因为每步只有一个汇合点。
#pragma once

#include <cstdint>
#include <string>

#include "experts/ffn.hpp"

namespace qinfer::experts {

// 一层的一批未命中专家。out 为 n × hidden、专家主序（第 e 个专家的结果在 out + e*hidden）。
struct ExpertJob {
    const LayerSpec* spec = nullptr;         // 该层三矩阵的档与块几何
    const ExpertWeights* experts = nullptr;  // n 个专家的三矩阵字节区
    int n = 0;                               // 专家数
    const float* x = nullptr;                // 激活（长度 hidden，已按档由 worker 内部量化）
    std::uint64_t hidden = 0;
    float* out = nullptr;
};

class ExpertWorker {
public:
    struct Stats {
        std::uint64_t jobs = 0;
        std::uint64_t experts = 0;
        std::uint64_t failed = 0;
    };
    virtual ~ExpertWorker() = default;

    virtual const char* name() const = 0;

    // 能否用：无运行时、无可用设备、设备节点打不开都在这里返回假（interfaces §8.4）。
    virtual bool available() const = 0;

    // 提交一批。返回假时 err 写明原因；此时 out 的内容未定义，调用方应改派。
    virtual bool submit(const ExpertJob& job, std::string& err) = 0;

    // 在汇合点等这批算完。没有待等的作业时返回真（幂等）。
    virtual bool wait(std::string& err) = 0;

    virtual const Stats& stats() const = 0;
};

// 纯 CPU 实现：同步跑完（线程池随步循环一起引入，接口本身不假设异步）。
class CpuExpertWorker : public ExpertWorker {
public:
    const char* name() const override { return "cpu"; }
    bool available() const override { return true; }
    bool submit(const ExpertJob& job, std::string& err) override;
    bool wait(std::string& err) override;
    const Stats& stats() const override { return stats_; }

private:
    FfnScratch scratch_;
    Stats stats_;
    bool pending_ = false;
};

// 核显占位：本仓库还没有核显内核，故 available() 恒假，err 里带实测原因。它的存在是为了让接口
// 与失败路径现在就可测，而不是等到有内核时才发现接口不对。
class IgpuExpertWorker : public ExpertWorker {
public:
    explicit IgpuExpertWorker(bool device_usable) : device_usable_(device_usable) {}
    const char* name() const override { return "igpu"; }
    bool available() const override { return device_usable_; }
    bool submit(const ExpertJob& job, std::string& err) override;
    bool wait(std::string& err) override;
    const Stats& stats() const override { return stats_; }

private:
    bool device_usable_ = false;
    Stats stats_;
};

// interfaces §8.4 的失败重派：先试 worker，失败则用 fallback 跑同一批并把这次计入 rerouted。
struct RerouteStats {
    std::uint64_t routed = 0;
    std::uint64_t rerouted = 0;
};
bool run_job_with_fallback(ExpertWorker& worker, ExpertWorker& fallback, const ExpertJob& job,
                           RerouteStats& stats, std::string& err);

}  // namespace qinfer::experts