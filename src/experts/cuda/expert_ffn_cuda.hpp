// 设备侧的一层专家前馈：权重常驻 + 一次算一批专家（interfaces.md §8 的设备侧那一半）。
//
// 一条前馈在这条路径上是同一个流里的四个 kernel，**中间不与主机同步**（§8.4：每步只有一个显存与
// 主机的汇合点）：
//   gate GEMV → up GEMV → silu·up 并就地量化成 Q8_0 → down GEMV
// 激活在主机侧量化（复用 `experts/ffn` 的 `quantize_input`，不另写一份量化算术），只把块搬上去；
// h 的量化在设备上做（它只存在于设备），照主机 `kernels/q8_0` 的量化器逐句对应。
//
// 支持范围（不满足就在 init 里明确失败，不猜）：
//   - 三个矩阵的档都要在设备侧有内核（gate/up：IQ2_S / IQ2_XXS / IQ1_M；down：Q2_0）；
//   - down 的激活搭档必须是 Q8_0（设备侧只做了这一种 h 量化器）。
// 这两条正好覆盖部署那份模型：down 全 48 层是 Q2_0，gate/up 是那三档。
//
// 权重常驻：`upload` 把一个专家的三矩阵放进一个槽，只搬一次；之后每次 `run` 只搬激活（几十 KB 级）
// 与结果。`uploaded_bytes` 是这条的观测量——常驻之后它不该再增长。
#pragma once

#include <cstdint>
#include <string>

#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"

namespace qinfer::experts {

class ExpertFfnDevice {
public:
    ExpertFfnDevice() = default;
    ~ExpertFfnDevice();
    ExpertFfnDevice(const ExpertFfnDevice&) = delete;
    ExpertFfnDevice& operator=(const ExpertFfnDevice&) = delete;

    // 在 device 上给这一层开 slots 个专家的常驻区。档不在支持范围里就失败。
    bool init(int device, const LayerSpec& spec, int slots, std::string& err);
    // 把一个专家的三矩阵放进 slot（同一槽重复放是覆盖写）。
    bool upload(int slot, const ExpertWeights& w, std::string& err);
    // 算一批：slots[e] 指定第 e 个专家用哪个槽，out 为 n × hidden（专家主序，与工人那套一致）。
    // n 不能超过 slots——结果的设备缓冲是按槽位数开的（解码步内不分配）；要更大的批就给更大的槽位数。
    bool run(const int* slots, int n, const float* x, float* out, std::string& err);

    bool ready() const { return d_w_ != nullptr; }
    int device() const { return device_; }
    int slots() const { return slots_; }
    std::uint64_t resident_bytes() const { return w_bytes_; }
    std::uint64_t uploaded_bytes() const { return uploaded_; }

private:
    bool free_all(std::string& err);

    int device_ = -1;
    LayerSpec spec_{};
    int slots_ = 0;
    std::uint64_t gate_bytes_ = 0;  // 一个专家的三矩阵各自字节数
    std::uint64_t up_bytes_ = 0;
    std::uint64_t down_bytes_ = 0;
    std::uint64_t per_expert_ = 0;
    std::uint64_t w_bytes_ = 0;
    std::uint64_t uploaded_ = 0;
    std::uint64_t x_blocks_bytes_ = 0;   // 激活块（gate/up 用）
    std::uint64_t h_blocks_bytes_ = 0;   // h 的 Q8_0 块（down 用）
    void* d_w_ = nullptr;                // slots × per_expert_
    void* d_x_ = nullptr;                // 激活块（主机量化后搬上来）
    void* d_gate_ = nullptr;             // ffn 个 float
    void* d_up_ = nullptr;               // ffn 个 float
    void* d_h_ = nullptr;                // down 的 Q8_0 块
    void* d_out_ = nullptr;              // slots × hidden 个 float
    FfnScratch scratch_;                 // 主机侧：复用主机的量化器
};

}  // namespace qinfer::experts
