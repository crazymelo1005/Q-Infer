// 遥测：每秒采样一次硬件读数，保留最近 60 点历史（spec §6 的 hardware 字段集与 10 系列清单，
// 逐键对齐 Strata serve/telemetry.py 的 now/history/static 字典）。
// 零依赖（ADR-008 在遥测上同样成立）：NVIDIA 走 dlopen("libnvidia-ml.so.1")（Windows 为
// LoadLibrary("nvml.dll")）动态取符号；AMD 走 Linux sysfs（KFD 拓扑找卡 + amdgpu 驱动读值，
// 路径与 hop 判定照抄 Strata 的 _Amd）；CPU/RAM 走 /proc/stat 差值法与 /proc/meminfo
// （Windows 为 GetSystemTimes 与 GlobalMemoryStatusEx）；磁盘速率解析 /proc/diskstats 按整盘聚合
// （Windows 在 v0 为 null，对齐 Strata 无 psutil 时的行为）。
// 任何字段读不到都是 null；任何异常都不允许拖垮服务——所有读取都带兜底，extra 回调同样兜底。
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace qinfer::serve {

// 解析 /proc/cpuinfo 的 "model name" 行，取冒号后的值并去首尾空白；无冒号或值全空白 → 空串。
// 独立成函数以便单测钉死（曾用 line.substr(11) 直切——"model name" 恰 10 字符，index 11 是冒号
// 本身，cpu_name 被读成 ": Intel(R)…"）。
std::string parse_cpu_model_name(const std::string& line);

// 单卡读器（NVML 卡或 AMD sysfs 卡）与 NVML 动态符号表：都是不透明类型，定义在 telemetry.cpp。
struct GpuCtx;
struct NvmlApi;

// 引擎侧指标：由服务层每次生成完成后刷新（server.cpp 的 telemetry_extra），采样线程每秒读一份。
struct TelemetryExtra {
    double tok_s = 0.0;               // v0：最近一次完成请求的解码速度（tok/s）
    double tok_s_mean = 0.0;          // 全程均值 tok/s（累计生成 token ÷ 累计解码耗时）
    double prefill_tok_s_mean = 0.0;  // v0 近似：提示词 token ÷ 租约后到首 token 耗时
};

// 单卡明细（多卡时 /metrics 的 gpus[] 数据源；v1_status 只认卡 0 的名义读数是 Strata 口径）。
struct GpuReading {
    int index = 0;
    std::string name;                  // 空 → null
    std::optional<double> util;        // %（0-100）
    std::optional<double> mem_used;    // 字节
    std::optional<double> mem_total;
    std::optional<double> temp;        // °C
    std::optional<double> power;       // W
};

// 静态信息（启动时读一次；读不到 → 空串 / 0，JSON 里是 null）。
struct TelemetryStatic {
    std::string gpu_name;   // 多卡名 " + " 连接；无卡 → 空
    int gpu_count = 0;
    std::string gpu_note;   // v0 恒空（AMD on Windows 的说明；本仓库 v0 无该用例）
    std::string cpu_name;   // /proc/cpuinfo "model name"
    int cores = 0;          // physical core count from /proc/cpuinfo (unique core id entries)
    int threads = 0;        // std::thread::hardware_concurrency()
    // Stage C: the integrated GPU (Vulkan compute for unhit experts). name from
    // the runtime probe, note carries its reason when unavailable; freq series
    // come from the i915/xe sysfs (gt/gt0/rps_*_freq_mhz).
    std::string igpu_name;
    bool igpu_available = false;
    std::string igpu_note;
};

// 快照：snapshot() 签出拷贝，读线程安全。now 的键集固定为 Strata telemetry.py 的 now 字典
// （gpu_util、gpu_mem_used、gpu_mem_total、gpu_temp、gpu_power、gpu_power_limit、gpu_pcie_gen、
// gpu_pcie_gen_max、gpu_pcie_width、gpu_pcie_rx_mb、gpu_pcie_tx_mb、cpu、ram_used、ram_total、
// disk_read_mb、disk_write_mb、tok_s、tok_s_mean、prefill_tok_s_mean）；读不到 = nullopt。
struct TelemetrySnapshot {
    std::map<std::string, std::optional<double>> now;
    std::vector<GpuReading> gpus;      // 多卡时才非空
    TelemetryStatic stat;
    std::map<std::string, std::vector<std::optional<double>>> history;  // 10 系列 × 最近 ≤60 点
};

class Telemetry {
public:
    using ExtraFn = std::function<TelemetryExtra()>;

    // gpu_filter: when non-empty, only monitor these GPU indices (NVML numbering).
    // Used by the Strata backend to restrict monitoring to the GPUs the engine actually uses
    // (parsed from --engine-arg --gpu). Empty = monitor all visible GPUs (default behavior).
    explicit Telemetry(ExtraFn extra = nullptr, std::vector<int> gpu_filter = {});
    ~Telemetry();
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    void start();   // 起采样线程（每秒一次；幂等；stop 后可再 start）
    void stop();    // 停线程并 join（幂等）
    void sample_now();              // 同步采一次（测试与线程共用）
    TelemetrySnapshot snapshot() const;  // 线程安全拷贝

private:
    void collect_cpu_ram_disk(std::map<std::string, std::optional<double>>& now);
    void push_history();

    ExtraFn extra_;
    std::vector<GpuCtx*> gpus_;      // 找到的卡（NVML 优先；全失败 → 空）
    NvmlApi* nvml_ = nullptr;        // NVML 装载态（nullptr = 未装载）
    std::vector<GpuReading> gpus_read_;
    TelemetryStatic stat_;

    mutable std::mutex mu_;
    std::map<std::string, std::optional<double>> now_;
    std::map<std::string, std::vector<std::optional<double>>> hist_;

    bool cpu_prev_valid_ = false;    // /proc/stat 差值法的前一时刻
    std::uint64_t cpu_prev_idle_ = 0, cpu_prev_total_ = 0;
    bool disk_prev_valid_ = false;   // /proc/diskstats 差值法的前一时刻
    double disk_prev_t_ = 0.0;
    std::uint64_t disk_prev_r_ = 0, disk_prev_w_ = 0;
    std::string igpu_freq_path_;     // gt/gt0/rps_act_freq_mhz (i915) when found
    std::string igpu_freq_max_path_;

    std::thread thread_;
    std::mutex cv_mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace qinfer::serve