// 见 cuda_worker.hpp 的口径说明。
#include "experts/cuda/cuda_worker.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace qinfer::experts {

struct CudaExpertWorker::LayerState {
    ExpertFfnDevice dev;
    std::vector<int> slot_of_id;  // 专家 id → 槽位（-1 表示没常驻），按需长到见过的最大 id
    int used_slots = 0;
};

CudaExpertWorker::CudaExpertWorker(int device, int slots_per_layer)
    : device_(device), slots_per_layer_(slots_per_layer) {}

CudaExpertWorker::~CudaExpertWorker() = default;

bool CudaExpertWorker::available() const {
    if (device_ < 0 || slots_per_layer_ <= 0) return false;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) return false;
    return device_ < devices;
}

bool CudaExpertWorker::wait(std::string& err) {
    err.clear();
    return true;  // 设备侧那批在 submit 内已经同步完，这里没有待等的作业（幂等）
}

bool CudaExpertWorker::submit(const ExpertJob& job, std::string& err) {
    err.clear();
    if (job.spec == nullptr || job.experts == nullptr || job.ids == nullptr || job.x == nullptr ||
        job.out == nullptr) {
        err = "作业的指针不完整（设备侧的作业还要专家 id 列表）";
        ++stats_.failed;
        return false;
    }
    if (job.n <= 0 || job.hidden == 0) {
        err = "作业里没有专家或 hidden 为 0";
        ++stats_.failed;
        return false;
    }
    if (job.n > slots_per_layer_) {
        err = "这一批的专家数超过一层的常驻槽位数";
        ++stats_.failed;
        return false;
    }
    const int layer = job.spec->layer;
    if (layer < 0) {
        err = "作业的层号无效";
        ++stats_.failed;
        return false;
    }

    auto it = layers_.find(layer);
    if (it == layers_.end()) {
        auto st = std::make_unique<LayerState>();
        if (!st->dev.init(device_, *job.spec, slots_per_layer_, err)) {
            ++stats_.failed;
            return false;
        }
        it = layers_.emplace(layer, std::move(st)).first;
    }
    LayerState& st = *it->second;

    std::vector<int> slots(static_cast<std::size_t>(job.n), -1);
    for (int e = 0; e < job.n; ++e) {
        const int id = job.ids[e];
        if (id < 0) {
            err = "专家 id 为负";
            ++stats_.failed;
            return false;
        }
        if (static_cast<std::size_t>(id) >= st.slot_of_id.size()) {
            st.slot_of_id.resize(static_cast<std::size_t>(id) + 1, -1);
        }
        int slot = st.slot_of_id[static_cast<std::size_t>(id)];
        if (slot < 0) {
            if (st.used_slots >= slots_per_layer_) {
                err = "设备侧这一层的常驻区满了（要 " + std::to_string(st.used_slots + 1) +
                      " 个槽，给的是 " + std::to_string(slots_per_layer_) + "）";
                ++stats_.failed;
                return false;
            }
            slot = st.used_slots++;
            if (!st.dev.upload(slot, job.experts[e], err)) {
                ++stats_.failed;
                return false;
            }
            st.slot_of_id[static_cast<std::size_t>(id)] = slot;
        }
        slots[static_cast<std::size_t>(e)] = slot;
    }

    if (!st.dev.run(slots.data(), job.n, job.x, job.out, err)) {
        ++stats_.failed;
        return false;
    }
    ++stats_.jobs;
    stats_.experts += static_cast<std::uint64_t>(job.n);
    return true;
}

std::uint64_t CudaExpertWorker::uploaded_bytes() const {
    std::uint64_t total = 0;
    for (const auto& kv : layers_) total += kv.second->dev.uploaded_bytes();
    return total;
}

std::uint64_t CudaExpertWorker::resident_bytes() const {
    std::uint64_t total = 0;
    for (const auto& kv : layers_) total += kv.second->dev.resident_bytes();
    return total;
}

std::uint32_t device_format_mask() {
    return format_bit(Format::kIq2S) | format_bit(Format::kIq2Xxs) | format_bit(Format::kIq1M) |
           format_bit(Format::kQ2_0);
}

ComputeGroup cuda_group(ExpertWorker& worker) {
    ComputeGroup g;
    g.kind = GroupKind::kCuda;
    g.tier = MemoryTier::kVram;
    g.caps.formats = device_format_mask();
    g.caps.integer_dot = true;  // sm_120 有整数点积类指令；本仓库的核还没用它，这一项先只作声明
    g.worker = &worker;
    g.bus = nullptr;  // 显存侧的组不持份额限流器：权重是经 PCIe 搬运子系统进来的（第 3 节的仲裁器）
    g.fallback = false;
    return g;
}

}  // namespace qinfer::experts
