// 计算组：后端接入的最小单位（interfaces.md 第 8 节，决策见 ADR-011）。
//
// 一组 = 设备 + 队列 + 内存池 + 能力表 +（主机侧组才有的）DDR 总线份额限流器。CPU、核显、每张
// CUDA 卡各是一组，CUDA 卡按卡成实例（ADR-004 的按层切分）。
//
// 选组只读声明，不读组的名字或类型。谁读哪一项：
//   - 选择器读 caps.formats（筛掉算不了这一批的档）与 bus（本步份额还装不装得下这一批）。
//   - caps.integer_dot 与 tier 由组自己的内核与放置代码读：核显在 192 项扩展里没有整数点积指令，
//     故那一项为假——它影响快慢，不影响能不能算，所以选择器不读它。
//
// 兜底组（CPU）恒可用，但排在非兜底组之后：两遍扫描，先试非兜底组，都不接才用它（interfaces §8.1）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "experts/expert_formats.hpp"
#include "experts/expert_worker.hpp"
#include "scheduling/bus_share.hpp"

namespace qinfer::experts {

// 设备类型。用于观测与失败原因；分派不按它分支。
enum class GroupKind : std::uint8_t { kCpu, kIgpu, kCuda };

// 组从哪一层取权重与激活。
enum class MemoryTier : std::uint8_t { kVram, kHostPinned, kSharedDdr };

// 能力表。取值是数据而不是类型，故核显那组按实测填即可，不必写第二套调度分支。
struct GroupCaps {
    // 有可用内核的权重量化档，位号 = Format 的值。空集表示这一组什么也接不了。
    std::uint32_t formats = 0;
    // 有无整数点积类指令（DP4a 等同效）。组自己的内核读它选实现；选择器不读。
    bool integer_dot = false;
};

std::uint32_t format_bit(Format f);
bool has_format(const GroupCaps& caps, Format f);
// 本仓库已有内核的全部档（位掩码）。CPU 组的能力表就是它。
std::uint32_t supported_format_mask();

// 一组。字段都是声明，没有行为；worker 与 bus 都是借来的，生命周期由调用方持有。
struct ComputeGroup {
    GroupKind kind = GroupKind::kCpu;
    MemoryTier tier = MemoryTier::kHostPinned;
    GroupCaps caps{};
    ExpertWorker* worker = nullptr;
    // 只有争用共享 DDR 的主机侧组才有：一个 decode 步里允许它占用的总线字节数，份额 0 即不启用
    // （该组不进候选，interfaces §8.5）。显存侧的组不需要它——权重是经 PCIe 搬运子系统进来的，
    // 那笔账记在第 3 节的仲裁器上，不记在组里。
    scheduling::BusShareLimiter* bus = nullptr;
    // 兜底组：非兜底组都不接时才用它；重派也落到它身上（interfaces §8.5）。
    bool fallback = false;

    // §8.5：不可用即不进候选。理由写进 why。
    bool available(std::string& why) const;
    const char* name() const;
};

// 插好的一组。group 为 nullptr 表示没有组能接这一批，逐组的理由拼在 why 里。
// 组本身只读（改的是 worker 与份额限流器的状态，不是组的声明），故指针是 const。
struct GroupPick {
    const ComputeGroup* group = nullptr;
    // 已在选中组的份额限流器里记账的字节。重派时那次搬运没有发生，调用方要按它退回。
    std::uint64_t booked_bytes = 0;
};

// 一批的权重字节（三矩阵逐专家之和）。几何不成立或没有专家时返回假。
bool job_weight_bytes(const ExpertJob& job, std::uint64_t& out);

// 挑一组来接这批。三道筛子：不可用不选；能力表里没有这一批全部档的不选；持份额限流器的组，
// 本步份额装不下这一批的不选。非兜底组扫一遍，都不接再扫兜底组。
GroupPick select_group(const std::vector<ComputeGroup>& groups, const ExpertJob& job, std::string& why);

// CPU 组：能力表 = 本仓库全部有内核的档；不持份额限流器（那条总线上的余量是它的）；恒可用、兜底。
ComputeGroup cpu_group(ExpertWorker& worker);
// 核显组：与 CPU 共用 DDR，故持份额限流器等调用方给份额；能力表按已测事实填——还没有核显内核
// （空集）、没有整数点积指令。
ComputeGroup igpu_group(ExpertWorker& worker, scheduling::BusShareLimiter& bus);

}  // namespace qinfer::experts
