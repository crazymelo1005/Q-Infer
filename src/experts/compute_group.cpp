#include "experts/compute_group.hpp"

namespace qinfer::experts {

namespace {

void append_reason(std::string& acc, const std::string& one) {
    if (!acc.empty()) acc += "；";
    acc += one;
}

// 这一批的三矩阵的档必须都在组的能力表里，否则这一组算不了它。只查档，不查字节。
bool formats_covered(const ComputeGroup& g, const LayerSpec& spec, std::string& why) {
    const Format fs[3] = {spec.gate.format, spec.up.format, spec.down.format};
    for (const Format f : fs) {
        if (!has_format(g.caps, f)) {
            why = std::string(g.name()) + " 没有 " + format_name(f) + " 的内核";
            return false;
        }
    }
    return true;
}

}  // namespace

std::uint32_t format_bit(Format f) {
    const unsigned int v = static_cast<unsigned int>(f);
    return v == 0 ? 0u : (1u << v);  // kUnknown 不进任何能力表
}

bool has_format(const GroupCaps& caps, Format f) { return (caps.formats & format_bit(f)) != 0; }

std::uint32_t supported_format_mask() {
    std::uint32_t mask = 0;
    // 从 kUnknown 之后扫到枚举的最后一个值；新档追加在末尾即自动纳入。
    for (int i = 1; i <= static_cast<int>(Format::kIq4Xs); ++i) {
        const Format f = static_cast<Format>(i);
        if (has_kernel(f)) mask |= format_bit(f);
    }
    return mask;
}

const char* ComputeGroup::name() const {
    switch (kind) {
        case GroupKind::kCpu:
            return "cpu";
        case GroupKind::kIgpu:
            return "igpu";
        case GroupKind::kCuda:
            return "cuda";
    }
    return "group";
}

bool ComputeGroup::available(std::string& why) const {
    why.clear();
    if (worker == nullptr) {
        why = "没有提交面";
        return false;
    }
    if (!worker->available()) {
        why = std::string(name()) + " 不可用（原因见它的失败信息）";
        return false;
    }
    if (bus != nullptr && !bus->enabled()) {
        why = std::string(name()) + " 的份额为 0（不启用）";
        return false;
    }
    return true;
}

bool job_weight_bytes(const ExpertJob& job, std::uint64_t& out) {
    out = 0;
    if (job.spec == nullptr || job.n <= 0) return false;
    const MatrixSpec* ms[3] = {&job.spec->gate, &job.spec->up, &job.spec->down};
    std::uint64_t per = 0;
    for (const MatrixSpec* m : ms) {
        if (!m->geometry_ok()) return false;
        per += m->rows * m->row_bytes();
    }
    if (per == 0) return false;
    out = per * static_cast<std::uint64_t>(job.n);
    return true;
}

GroupPick select_group(const std::vector<ComputeGroup>& groups, const ExpertJob& job, std::string& why) {
    why.clear();
    GroupPick pick;
    if (job.spec == nullptr || job.n <= 0) {
        why = "作业没有层规格或没有专家";
        return pick;
    }
    std::uint64_t bytes = 0;
    const bool bytes_known = job_weight_bytes(job, bytes);

    for (int pass = 0; pass < 2 && pick.group == nullptr; ++pass) {
        const bool want_fallback = pass == 1;
        for (const ComputeGroup& g : groups) {
            if (g.fallback != want_fallback) continue;
            std::string gwhy;
            if (!g.available(gwhy)) {
                append_reason(why, gwhy);
                continue;
            }
            if (!formats_covered(g, *job.spec, gwhy)) {
                append_reason(why, gwhy);
                continue;
            }
            if (g.bus != nullptr) {
                if (!bytes_known) {
                    append_reason(why, std::string(g.name()) + " 这一批的字节算不出来");
                    continue;
                }
                if (!g.bus->try_acquire(bytes)) {
                    append_reason(why, std::string(g.name()) + " 本步份额装不下这一批");
                    continue;
                }
                pick.booked_bytes = bytes;
            }
            pick.group = &g;
            break;
        }
    }
    return pick;
}

ComputeGroup cpu_group(ExpertWorker& worker) {
    ComputeGroup g;
    g.kind = GroupKind::kCpu;
    g.tier = MemoryTier::kHostPinned;
    g.caps.formats = supported_format_mask();
    g.caps.integer_dot = true;  // AVX2 的整数乘加；这一项只作声明，核显那组才为假
    g.worker = &worker;
    g.bus = nullptr;
    g.fallback = true;
    return g;
}

ComputeGroup igpu_group(ExpertWorker& worker, scheduling::BusShareLimiter& bus) {
    ComputeGroup g;
    g.kind = GroupKind::kIgpu;
    g.tier = MemoryTier::kSharedDdr;
    g.caps.formats = 0;         // 核显内核尚未接入（interfaces §8.7）
    g.caps.integer_dot = false;  // 192 项扩展里没有 VK_KHR_shader_integer_dot_product
    g.worker = &worker;
    g.bus = &bus;
    g.fallback = false;
    return g;
}

}  // namespace qinfer::experts
