// DDR 总线份额限流器：一个 decode 步里允许核显占用的总线字节数。
//
// 为什么它与 `budget_arbiter` 是两件事（interfaces.md 第 8.2 节）：仲裁器管的是过 PCIe 的字节
// （权重换入显存、KV 换入、必需表行）；核显争的是 DDR 总线，专家在 CPU 就地计算也走这条总线。
// 把 DDR 侧的字节记进 PCIe 预算会算错账，故这里单开一个限制点。
//
// 份额的默认值是 0，即不启用核显：这不是「暂时没配」而是设计默认（核显是可选协处理器，
// ADR-005）。份额上限必须实测（risks.md 的 R-08）；环境2 的带宽型争用实测见 engine §13。
#pragma once

#include <cstdint>

namespace qinfer::scheduling {

class BusShareLimiter {
public:
    // bytes_per_step = 0 表示关闭（核显一律不放过）。
    explicit BusShareLimiter(std::uint64_t bytes_per_step) : share_(bytes_per_step) {}

    bool enabled() const { return share_ != 0; }
    std::uint64_t share() const { return share_; }
    std::uint64_t used() const { return used_; }
    std::uint64_t remaining() const { return share_ > used_ ? share_ - used_ : 0; }
    std::uint64_t refusals() const { return refusals_; }

    // 每步开始时清零本步用量。步循环在汇合点之后调用。
    void start_step() { used_ = 0; }

    // 本步能否再让核显搬 bytes：可以则记账并返回 true。
    // 三条规则：关闭时一律不受理（含 0 字节请求）；0 字节请求不占份额、也不计入拒收；单次请求超过
    // 整个份额时直接拒（不做部分准入——半途中断的搬运对调用方没有意义，反而会诱使它去做「补一半」
    // 这种额外的同步）。
    bool try_acquire(std::uint64_t bytes) {
        if (share_ == 0) {
            ++refusals_;
            return false;
        }
        if (bytes == 0) return true;
        if (bytes > share_ - used_) {
            ++refusals_;
            return false;
        }
        used_ += bytes;
        return true;
    }

private:
    std::uint64_t share_ = 0;
    std::uint64_t used_ = 0;
    std::uint64_t refusals_ = 0;
};

}  // namespace qinfer::scheduling