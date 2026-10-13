#include "experts/expert_worker.hpp"

#include "experts/igpu/igpu_expert.hpp"

namespace qinfer::experts {

namespace {

bool job_sane(const ExpertJob& job, std::string& err) {
    if (job.spec == nullptr || job.experts == nullptr || job.x == nullptr || job.out == nullptr) {
        err = "作业的指针不完整";
        return false;
    }
    if (job.n <= 0) {
        err = "作业里没有专家";
        return false;
    }
    if (job.hidden == 0) {
        err = "作业的 hidden 为 0";
        return false;
    }
    return true;
}

}  // namespace

bool CpuExpertWorker::submit(const ExpertJob& job, std::string& err) {
    if (!job_sane(job, err)) {
        ++stats_.failed;
        return false;
    }
    for (int e = 0; e < job.n; ++e) {
        if (!expert_ffn(*job.spec, job.experts[e], job.x, scratch_,
                        job.out + static_cast<std::size_t>(e) * job.hidden, err)) {
            ++stats_.failed;
            return false;
        }
    }
    ++stats_.jobs;
    stats_.experts += static_cast<std::uint64_t>(job.n);
    pending_ = true;
    return true;
}

bool CpuExpertWorker::wait(std::string& err) {
    (void) err;
    pending_ = false;  // 同步实现：submit 返回时已经算完
    return true;
}

// Stage C: the real iGPU worker. Device probing is lazy (first available() or
// submit()); the batch runs synchronously, matching the CPU worker's contract.
IgpuExpertWorker::IgpuExpertWorker(bool force_unavailable)
    : force_unavailable_(force_unavailable) {}

IgpuExpertWorker::~IgpuExpertWorker() = default;

bool IgpuExpertWorker::available() const {
    if (force_unavailable_) {
        reason_ = "forced unavailable (test switch)";
        return false;
    }
    if (dev_ == nullptr) {
        dev_ = std::make_unique<igpu::IgpuExpertDevice>();
    }
    if (dev_->ready()) return true;
    std::string err;
    if (!dev_->init(err)) {
        reason_ = err.empty() ? "iGPU unavailable (no reason reported)" : err;
        return false;
    }
    reason_.clear();
    return true;
}

bool IgpuExpertWorker::submit(const ExpertJob& job, std::string& err) {
    err.clear();
    if (!available()) {
        ++stats_.failed;
        err = "iGPU unavailable: " + reason_;
        return false;
    }
    if (!job_sane(job, err)) {
        ++stats_.failed;
        return false;
    }
    if (!dev_->run_batch(*job.spec, job.experts, job.n, job.x, job.hidden, job.out, err)) {
        ++stats_.failed;
        return false;
    }
    ++stats_.jobs;
    stats_.experts += static_cast<std::uint64_t>(job.n);
    return true;
}

bool IgpuExpertWorker::wait(std::string& err) {
    err.clear();
    return true;  // synchronous implementation: submit has already finished
}

bool run_job_with_fallback(ExpertWorker& worker, ExpertWorker& fallback, const ExpertJob& job,
                           RerouteStats& stats, std::string& err) {
    err.clear();
    if (worker.available()) {
        std::string why;
        if (worker.submit(job, why)) {
            if (!worker.wait(err)) {
                // 提交成功但汇合失败：同样重派，因为 out 的内容不可信。
                std::string ferr;
                if (!fallback.submit(job, ferr) || !fallback.wait(ferr)) {
                    err = "重派也失败：" + ferr;
                    return false;
                }
                ++stats.rerouted;
                return true;
            }
            ++stats.routed;
            return true;
        }
        err = why;
    }
    // 不可用或提交失败：改派给 fallback（interfaces §8.4 的默认路径，不算错误）。
    std::string ferr;
    if (!fallback.submit(job, ferr) || !fallback.wait(ferr)) {
        err = err.empty() ? "重派失败：" + ferr : err + "；重派也失败：" + ferr;
        return false;
    }
    ++stats.rerouted;
    err.clear();
    return true;
}

}  // namespace qinfer::experts