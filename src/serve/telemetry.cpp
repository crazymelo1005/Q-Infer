// 实现要点（对齐 Strata serve/telemetry.py，逐条照抄其语义）：
// - NVML：dlopen 成功后从 index 0 逐个 nvmlDeviceGetHandleByIndex_v2 取句柄直到失败（等价
//   Strata 的 gpu_reader 列表）；符号表 12 个，缺任一关键读取符号 → 整包弃用。PowerUsage 是
//   mW → /1000；温度传感器类型 0（NVML_TEMPERATURE_GPU）；PcieThroughput 是 KB/s / 1024 → MB/s
//   （counter 1 = RX、0 = TX）。
// - AMD：KFD 拓扑 nodes 的数字子目录（升序）读 properties，gfx_target_version 与 simd_count 非零者
//   即 GPU 节点，按 drm_render_minor 拼 /sys/class/drm/renderD<minor>/device；hwmon 取第一个目录
//   （temp1_input 是 m°C → /1000，power1_average 或 power1_input 是 µW → /1e6，power1_cap 同理）；
//   link() 走 realpath 上的 BDF 段（长度 12、格式 0000:00:00.0），读每段 max_link_speed /
//   max_link_width，全段可读 → 取最小（Strata 的 hop 瓶颈判定），否则退回卡自身的
//   current_link_speed / current_link_width；速率表 {2.5:1, 5.0:2, 8.0:3, 16.0:4, 32.0:5, 64.0:6}。
// - CPU：/proc/stat 第一行全字段，idle = f[3]+f[4]（idle+iowait）、total = sum(f)，差值法算占用率
//   clamp 到 [0,100]；首个采样无前一时刻 → null。Windows 走 GetSystemTimes（kernel 含 idle）。
// - RAM：/proc/meminfo 的 MemTotal/MemAvailable（kB）→ 字节，used = total - avail；Windows 走
//   GlobalMemoryStatusEx。
// - 磁盘：/proc/diskstats 只取「整盘」行（minor == 0；分区行 minor 非 0），聚合 sectors 计数差
//   × 512 / Δt / 2^20 → MB/s；首个采样 null。Windows 在 v0 为 null（Strata 无 psutil 时同口径）。
// - 多卡聚合（NVML 多卡或 AMD 多节点）：mem/power/pcie 流量取总和、util 取均值、temp 取最高、
//   pcie_gen/width 取第一张卡（Strata 同款：聚合键表里没有 gen/width）。
// - history：10 系列（键表见 push_history），每样一次各推一个值（读不到推 nullopt），封顶 60。
// - gpu_note/cores 在 v0 恒空/恒 0（见头文件注释），但键位保留，T14 照常序列化。
#include "serve/telemetry.hpp"

#include "experts/igpu/vulkan_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qinfer::serve {

namespace {

// ---- 小工具：读 sysfs /proc 文本文件（读不到 → nullopt / 空串）----
std::optional<long long> read_int(const std::string& path) {
    std::ifstream f(path);
    long long v = 0;
    return (f >> v) ? std::optional<long long>(v) : std::nullopt;
}

std::string read_str(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

// "16.0 GT/s PCIe" → 4（Strata _gen 的显式速率表；v/2.5 不是 gen）。
std::optional<int> link_gen(const std::string& text) {
    double gt = 0.0;
    if (std::sscanf(text.c_str(), "%lf", &gt) != 1) return std::nullopt;
    static const std::pair<double, int> kGen[] = {{2.5, 1}, {5.0, 2}, {8.0, 3},
                                                  {16.0, 4}, {32.0, 5}, {64.0, 6}};
    for (const auto& [rate, gen] : kGen)
        if (std::abs(gt - rate) < 1e-9) return gen;
    return std::nullopt;
}

// 系统路径里是否为 PCI 设备名（"0000:03:00.0"）。
bool is_bdf(const std::string& s) {
    if (s.size() != 12 || s[4] != ':' || s[7] != ':' || s[10] != '.') return false;
    for (int i = 0; i < 12; ++i) {
        if (i == 4 || i == 7 || i == 10) continue;
        const char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// NVML 返回结构（nvmlUtilization_t / nvmlMemory_t 的内存布局）。
struct NvmlUtilization { unsigned gpu = 0, mem = 0; };
struct NvmlMemory { unsigned long long total = 0, free = 0, used = 0; };

}  // namespace

// ---- 不透明类型定义（telemetry.hpp 里只有前置声明）----

// 单卡读器基类（NVML 卡或 AMD sysfs 卡），统一出 Strata 的 g 字典键（无 "gpu_" 前缀）。
struct GpuCtx {
    virtual ~GpuCtx() = default;
    virtual std::string name() = 0;   // 空 = 读不到
    virtual std::map<std::string, std::optional<double>> read() = 0;
};

// NVML 动态符号表（符号名 → 函数指针；返回码 0 = NVML_SUCCESS）。
struct NvmlApi {
    void* lib = nullptr;   // dlopen / LoadLibrary 句柄
    int (*init)() = nullptr;
    int (*get_handle)(unsigned int, void**) = nullptr;
    int (*get_name)(void*, char*, unsigned int) = nullptr;
    int (*get_utilization)(void*, NvmlUtilization*) = nullptr;
    int (*get_memory)(void*, NvmlMemory*) = nullptr;
    int (*get_temperature)(void*, unsigned int, unsigned int*) = nullptr;
    int (*get_power)(void*, unsigned int*) = nullptr;
    int (*get_power_limit)(void*, unsigned int*) = nullptr;
    int (*get_curr_gen)(void*, unsigned int*) = nullptr;
    int (*get_max_gen)(void*, unsigned int*) = nullptr;
    int (*get_curr_width)(void*, unsigned int*) = nullptr;
    int (*get_pcie_tp)(void*, unsigned int, unsigned int*) = nullptr;

    ~NvmlApi() {
        if (!lib) return;
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(lib));
#else
        dlclose(lib);
#endif
    }
};

namespace {

// 装载 NVML；失败返回 nullptr（Windows 走 LoadLibraryA + GetProcAddress）。
NvmlApi* load_nvml() {
    auto* api = new NvmlApi();
#if defined(_WIN32)
    api->lib = static_cast<void*>(LoadLibraryA("nvml.dll"));
#else
    api->lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!api->lib) api->lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
#endif
    if (!api->lib) { delete api; return nullptr; }
#if defined(_WIN32)
    auto sym = [&](const char* n) {
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(api->lib), n));
    };
#else
    auto sym = [&](const char* n) { return dlsym(api->lib, n); };
#endif
    api->init = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
    if (!api->init || api->init() != 0) { delete api; return nullptr; }
    api->get_handle = reinterpret_cast<int (*)(unsigned int, void**)>(
        sym("nvmlDeviceGetHandleByIndex_v2"));
    if (!api->get_handle) api->get_handle =
        reinterpret_cast<int (*)(unsigned int, void**)>(sym("nvmlDeviceGetHandleByIndex"));
    api->get_name = reinterpret_cast<int (*)(void*, char*, unsigned int)>(sym("nvmlDeviceGetName"));
    api->get_utilization = reinterpret_cast<int (*)(void*, NvmlUtilization*)>(
        sym("nvmlDeviceGetUtilizationRates"));
    api->get_memory = reinterpret_cast<int (*)(void*, NvmlMemory*)>(sym("nvmlDeviceGetMemoryInfo"));
    api->get_temperature = reinterpret_cast<int (*)(void*, unsigned int, unsigned int*)>(
        sym("nvmlDeviceGetTemperature"));
    api->get_power = reinterpret_cast<int (*)(void*, unsigned int*)>(sym("nvmlDeviceGetPowerUsage"));
    api->get_power_limit = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetEnforcedPowerLimit"));
    api->get_curr_gen = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetCurrPcieLinkGeneration"));
    api->get_max_gen = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetMaxPcieLinkGeneration"));
    api->get_curr_width = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetCurrPcieLinkWidth"));
    api->get_pcie_tp = reinterpret_cast<int (*)(void*, unsigned int, unsigned int*)>(
        sym("nvmlDeviceGetPcieThroughput"));
    if (!api->get_handle || !api->get_name || !api->get_memory) { delete api; return nullptr; }
    return api;
}

struct NvmlCard : GpuCtx {
    NvmlApi* api = nullptr;
    void* dev = nullptr;
    int index = 0;

    std::string name() override {
        char buf[96] = {0};
        if (!api || !api->get_name || api->get_name(dev, buf, sizeof(buf)) != 0) return "";
        return buf;
    }
    std::map<std::string, std::optional<double>> read() override {
        std::map<std::string, std::optional<double>> o;
        if (!api) return o;
        NvmlUtilization u;
        if (api->get_utilization && api->get_utilization(dev, &u) == 0) o["util"] = u.gpu;
        NvmlMemory m;
        if (api->get_memory && api->get_memory(dev, &m) == 0) {
            o["mem_used"] = static_cast<double>(m.used);
            o["mem_total"] = static_cast<double>(m.total);
        }
        unsigned tmp = 0;
        if (api->get_temperature && api->get_temperature(dev, 0, &tmp) == 0) o["temp"] = tmp;
        if (api->get_power && api->get_power(dev, &tmp) == 0) o["power"] = tmp / 1000.0;  // mW → W
        if (api->get_power_limit && api->get_power_limit(dev, &tmp) == 0)
            o["power_limit"] = tmp / 1000.0;
        unsigned rx = 0, tx = 0;
        if (api->get_pcie_tp && api->get_pcie_tp(dev, 1, &rx) == 0) o["pcie_rx_mb"] = rx / 1024.0;
        if (api->get_pcie_tp && api->get_pcie_tp(dev, 0, &tx) == 0) o["pcie_tx_mb"] = tx / 1024.0;
        return o;
    }
};

struct AmdCard : GpuCtx {
    std::string dev_;    // /sys/class/drm/renderD<N>/device
    std::string hwmon_;  // 第一个 hwmon 子目录（可为空）
    std::vector<std::string> hops_;  // realpath 上的 BDF 段目录（Strata _hops）

    std::string name() override {
        const std::string n = read_str(dev_ + "/product_name");
        return n.empty() ? "AMD Radeon" : n;
    }
    std::map<std::string, std::optional<double>> read() override {
        std::map<std::string, std::optional<double>> o;
        if (dev_.empty()) return o;
        if (auto v = read_int(dev_ + "/gpu_busy_percent")) o["util"] = static_cast<double>(*v);
        if (auto v = read_int(dev_ + "/mem_info_vram_used")) o["mem_used"] = static_cast<double>(*v);
        if (auto v = read_int(dev_ + "/mem_info_vram_total")) o["mem_total"] = static_cast<double>(*v);
        if (!hwmon_.empty()) {
            if (auto t = read_int(hwmon_ + "/temp1_input")) o["temp"] = *t / 1000.0;  // m°C → °C
            auto p = read_int(hwmon_ + "/power1_average");
            if (!p) p = read_int(hwmon_ + "/power1_input");
            if (p) o["power"] = *p / 1e6;  // µW → W
            if (auto c = read_int(hwmon_ + "/power1_cap")) o["power_limit"] = *c / 1e6;
        }
        // link()：全 hop 可读 → 最小 gen/width（hop 瓶颈判定）；否则退回卡自身 current_link_*。
        bool whole = !hops_.empty();
        std::vector<int> gens, widths;
        for (const auto& h : hops_) {
            auto g = link_gen(read_str(h + "/max_link_speed"));
            auto w = read_int(h + "/max_link_width");
            if (!g && !w) whole = false;
            if (g) gens.push_back(*g);
            if (w) widths.push_back(static_cast<int>(*w));
        }
        const auto own_gen = link_gen(read_str(dev_ + "/current_link_speed"));
        const auto own_width = read_int(dev_ + "/current_link_width");
        if (whole && !gens.empty()) {
            o["pcie_gen"] = *std::min_element(gens.begin(), gens.end());
            o["pcie_gen_max"] = o["pcie_gen"];
        } else if (own_gen) {
            o["pcie_gen"] = *own_gen;
            o["pcie_gen_max"] = *own_gen;
        }
        if (whole && !widths.empty()) {
            o["pcie_width"] = *std::min_element(widths.begin(), widths.end());
        } else if (own_width) {
            o["pcie_width"] = static_cast<double>(*own_width);
        }
        return o;
    }
};

// AMD 卡发现：KFD 拓扑节点 → properties 过滤 → renderD 设备目录 + hwmon + BDF 段
// （Strata amd_device_dir + _Amd.__init__ + _hops）。Windows 无 sysfs → nullptr。
AmdCard* find_amd_card(int index) {
#if defined(_WIN32)
    (void)index;
    return nullptr;
#else
    using namespace std::filesystem;
    static const std::string kBase = "/sys/class/kfd/kfd/topology/nodes";
    std::vector<int> nodes;
    if (is_directory(kBase)) {
        for (const auto& e : directory_iterator(kBase)) {
            const std::string n = e.path().filename().string();
            if (!n.empty() && std::all_of(n.begin(), n.end(),
                                          [](char c) { return c >= '0' && c <= '9'; }))
                nodes.push_back(std::stoi(n));
        }
    }
    std::sort(nodes.begin(), nodes.end());
    std::vector<std::string> devs;
    for (int n : nodes) {
        // properties：每行 "key value"；只认 GPU 节点（gfx_target_version != 0 且 simd_count != 0）。
        long long target = 0, simd = 0;
        std::string minor;
        std::ifstream f(kBase + "/" + std::to_string(n) + "/properties");
        std::string line;
        while (std::getline(f, line)) {
            const auto sp = line.find(' ');
            const std::string key = line.substr(0, sp);
            const std::string val = sp == std::string::npos ? "" : line.substr(sp + 1);
            if (key == "gfx_target_version") target = std::stoll(val, nullptr, 16);
            else if (key == "simd_count") simd = std::stoll(val);
            else if (key == "drm_render_minor") minor = val;
        }
        if (target == 0 || simd == 0 || minor.empty()) continue;
        const std::string dev = "/sys/class/drm/renderD" + minor + "/device";
        if (is_directory(dev)) devs.push_back(dev);
    }
    if (index < 0 || index >= static_cast<int>(devs.size())) return nullptr;
    auto* card = new AmdCard();
    card->dev_ = devs[index];
    try {  // hwmon 第一个目录（Strata sorted 之后取 [0]）
        const std::string hdir = card->dev_ + "/hwmon";
        if (is_directory(hdir))
            for (const auto& e : directory_iterator(hdir)) { card->hwmon_ = e.path().string(); break; }
    } catch (...) {}
    try {  // realpath 上的 BDF 段（Strata _hops：路径里每个 BDF 名逐段累积）
        std::string acc;
        for (const auto& part : canonical(card->dev_)) {
            acc += "/" + part.string();
            if (is_bdf(part.string())) card->hops_.push_back(acc);
        }
    } catch (...) {}
    return card;
#endif
}

}  // namespace

std::string parse_cpu_model_name(const std::string& line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) return {};
    const std::size_t first = line.find_first_not_of(" \t", colon + 1);
    if (first == std::string::npos) return {};
    const std::size_t last = line.find_last_not_of(" \t");
    return line.substr(first, last - first + 1);
}

Telemetry::Telemetry(ExtraFn extra, std::vector<int> gpu_filter) : extra_(std::move(extra)) {
    // 卡发现：NVML 优先（index 0..64 逐个取句柄到失败），全失败 → AMD sysfs 卡 0。
    // When gpu_filter is non-empty, only discover GPUs whose index is in the filter list.
    // This lets the Strata backend restrict monitoring to the GPUs the engine actually uses
    // (parsed from --engine-arg --gpu), matching Strata's own monitoring convention.
    nvml_ = load_nvml();
    if (nvml_) {
        for (int i = 0; i < 64; ++i) {
            void* dev = nullptr;
            if (nvml_->get_handle(static_cast<unsigned>(i), &dev) != 0) break;
            if (!gpu_filter.empty()) {
                bool found = false;
                for (int fi : gpu_filter) {
                    if (fi == i) { found = true; break; }
                }
                if (!found) continue;
            }
            auto* card = new NvmlCard();
            card->api = nvml_;
            card->dev = dev;
            card->index = i;
            gpus_.push_back(card);
        }
    }
    if (gpus_.empty()) {
        if (AmdCard* c = find_amd_card(0)) gpus_.push_back(c);
    }

    // 静态信息（Strata self.static 的键，读不到 → 空串 / 0）。
    std::string names;
    for (GpuCtx* g : gpus_) {
        if (!names.empty()) names += " + ";
        names += g->name().empty() ? "?" : g->name();
    }
    stat_.gpu_name = gpus_.empty() ? "" : names;
    stat_.gpu_count = static_cast<int>(gpus_.size());
    stat_.gpu_note = "";
    std::ifstream ci("/proc/cpuinfo");
    std::string line;
    while (std::getline(ci, line)) {
        if (line.rfind("model name", 0) == 0) {
            stat_.cpu_name = parse_cpu_model_name(line);
            break;
        }
    }
    stat_.threads = static_cast<int>(std::thread::hardware_concurrency());

    // Stage C: the integrated GPU. Availability and the device name come from
    // the Vulkan runtime probe (the same one the expert worker uses); the load
    // proxy is the i915/xe sysfs frequency pair — this kernel exposes no
    // busyness counter that user-space can read.
    {
        experts::igpu::VulkanRuntime& rt = experts::igpu::igpu_runtime();
        std::string ig_err;
        if (rt.load(ig_err) && rt.init_device(ig_err) && rt.available()) {
            stat_.igpu_available = true;
            stat_.igpu_name = rt.device_name();
        } else {
            stat_.igpu_note = rt.reason().empty() ? ig_err : rt.reason();
        }
        for (int i = 0; i < 8; ++i) {
            const std::string card = "/sys/class/drm/card" + std::to_string(i);
            std::error_code e;
            const auto drv = std::filesystem::read_symlink(card + "/device/driver", e);
            if (e) continue;
            const std::string dn = drv.filename().string();
            if (dn != "i915" && dn != "xe") continue;
            const std::string base = card + "/gt/gt0/";
            if (std::filesystem::exists(base + "rps_act_freq_mhz", e)) {
                igpu_freq_path_ = base + "rps_act_freq_mhz";
                igpu_freq_max_path_ = base + "rps_max_freq_mhz";
            }
            break;
        }
    }
}

Telemetry::~Telemetry() {
    stop();
    for (GpuCtx* g : gpus_) delete g;
    delete nvml_;
}

void Telemetry::start() {
    std::lock_guard<std::mutex> lk(cv_mu_);
    if (thread_.joinable()) return;
    stop_ = false;
    thread_ = std::thread([this] {
        for (;;) {
            sample_now();                 // 启动即采（历史第一点不等 1s）
            std::unique_lock<std::mutex> w(cv_mu_);
            cv_.wait_for(w, std::chrono::seconds(1), [this] { return stop_; });
            if (stop_) return;
        }
    });
}

void Telemetry::stop() {
    {
        std::lock_guard<std::mutex> lk(cv_mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Telemetry::sample_now() {
    std::map<std::string, std::optional<double>> now;
    std::vector<GpuReading> gpus_tmp;
    if (!gpus_.empty()) {
        std::vector<std::map<std::string, std::optional<double>>> reads;
        for (GpuCtx* g : gpus_) {
            auto r = g->read();
            GpuReading gr;
            gr.index = static_cast<int>(reads.size());
            gr.name = g->name();
            if (r.count("util")) gr.util = r["util"];
            if (r.count("mem_used")) gr.mem_used = r["mem_used"];
            if (r.count("mem_total")) gr.mem_total = r["mem_total"];
            if (r.count("temp")) gr.temp = r["temp"];
            if (r.count("power")) gr.power = r["power"];
            reads.push_back(std::move(r));
            gpus_tmp.push_back(std::move(gr));
        }
        // Strata 聚合口径：mem/power/pcie 总和、util 均值、temp 最高；gen/width 取第一张卡。
        const auto sums = [&reads](const char* key) -> std::optional<double> {
            double s = 0.0;
            bool any = false;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key)) { s += *r.at(key); any = true; }
            return any ? std::optional<double>(s) : std::nullopt;
        };
        const auto mean = [&reads](const char* key) -> std::optional<double> {
            double s = 0.0;
            int n = 0;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key)) { s += *r.at(key); ++n; }
            return n > 0 ? std::optional<double>(s / n) : std::nullopt;
        };
        const auto maxv = [&reads](const char* key) -> std::optional<double> {
            std::optional<double> m;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key))
                    m = m ? std::max(*m, *r.at(key)) : r.at(key);
            return m;
        };
        now["gpu_mem_used"] = sums("mem_used");
        now["gpu_mem_total"] = sums("mem_total");
        now["gpu_power"] = sums("power");
        now["gpu_power_limit"] = sums("power_limit");
        now["gpu_pcie_rx_mb"] = sums("pcie_rx_mb");
        now["gpu_pcie_tx_mb"] = sums("pcie_tx_mb");
        now["gpu_util"] = mean("util");
        now["gpu_temp"] = maxv("temp");
        const auto& first = reads.front();
        for (const char* k : {"pcie_gen", "pcie_gen_max", "pcie_width"})
            if (first.count(k)) now[std::string("gpu_") + k] = first.at(k);
    }

    // 引擎侧指标（extra 回调兜底：异常不拖垮遥测；0.0 = 尚无数据）。
    TelemetryExtra ex;
    if (extra_) {
        try { ex = extra_(); } catch (...) {}
    }
    now["tok_s"] = ex.tok_s;
    now["tok_s_mean"] = ex.tok_s_mean;
    now["prefill_tok_s_mean"] = ex.prefill_tok_s_mean;

    // iGPU frequency pair (stage C load proxy); keys stay present (null when
    // there is no i915/xe card), matching this file's fixed-key convention.
    {
        const auto read_num = [](const std::string& p) -> std::optional<double> {
            if (p.empty()) return std::nullopt;
            std::ifstream f(p);
            double v = 0.0;
            if (!(f >> v)) return std::nullopt;
            return v;
        };
        now["igpu_freq_mhz"] = read_num(igpu_freq_path_);
        now["igpu_freq_max_mhz"] = read_num(igpu_freq_max_path_);
    }

    collect_cpu_ram_disk(now);
    {
        std::lock_guard<std::mutex> lk(mu_);   // now_ 与 gpus_read_ 同锁换出，snapshot 无竞态
        now_.swap(now);
        gpus_read_.swap(gpus_tmp);
    }
    push_history();
}

void Telemetry::collect_cpu_ram_disk(std::map<std::string, std::optional<double>>& now) {
#if defined(_WIN32)
    FILETIME idle_t{}, kern_t{}, user_t{};
    if (GetSystemTimes(&idle_t, &kern_t, &user_t)) {
        const auto t100 = [](const FILETIME& f) {
            return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
        };
        const std::uint64_t idle = t100(idle_t);
        const std::uint64_t total = t100(kern_t) + t100(user_t);   // kernel 含 idle（Strata 注释）
        if (cpu_prev_valid_ && total > cpu_prev_total_) {
            const double p = 100.0 * (1.0 - static_cast<double>(idle - cpu_prev_idle_) /
                                                  static_cast<double>(total - cpu_prev_total_));
            now["cpu"] = std::max(0.0, std::min(100.0, p));
        }
        cpu_prev_idle_ = idle;
        cpu_prev_total_ = total;
        cpu_prev_valid_ = true;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        now["ram_used"] = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys);
        now["ram_total"] = static_cast<double>(ms.ullTotalPhys);
    }
    // 磁盘：Windows 在 v0 为 null（Strata 无 psutil 时同样缺磁盘读数）。
#else
    {
        std::ifstream f("/proc/stat");
        std::string tag;
        f >> tag;
        std::uint64_t v[8] = {0};
        for (int i = 0; i < 8 && f; ++i) f >> v[i];
        const std::uint64_t idle = v[3] + v[4];    // Strata：f[3]+f[4] = idle+iowait
        std::uint64_t total = 0;
        for (std::uint64_t x : v) total += x;
        if (cpu_prev_valid_ && total > cpu_prev_total_) {
            const double p = 100.0 * (1.0 - static_cast<double>(idle - cpu_prev_idle_) /
                                                  static_cast<double>(total - cpu_prev_total_));
            now["cpu"] = std::max(0.0, std::min(100.0, p));
        }
        cpu_prev_idle_ = idle;
        cpu_prev_total_ = total;
        cpu_prev_valid_ = true;
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string key;
        long long total_kb = 0, avail_kb = 0;
        while (f >> key) {
            long long v = 0;
            if (key == "MemTotal:") { f >> v; total_kb = v; }
            else if (key == "MemAvailable:") { f >> v; avail_kb = v; }
            else f.ignore(4096, '\n');
        }
        if (total_kb > 0) {
            now["ram_total"] = static_cast<double>(total_kb) * 1024.0;
            now["ram_used"] = static_cast<double>(total_kb - avail_kb) * 1024.0;
        }
    }
    {
        // /proc/diskstats：整盘行（minor == 0）的 sectors_* 计数聚合；差值 × 512 / Δt / 2^20 → MB/s。
        std::uint64_t r = 0, w = 0;
        std::ifstream f("/proc/diskstats");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            long long major = 0, minor = 0;
            std::string name;
            std::uint64_t a = 0, b = 0, rs = 0, c = 0, d = 0, e = 0, ws = 0;
            ss >> major >> minor >> name >> a >> b >> rs >> c >> d >> e >> ws;
            (void)major; (void)a; (void)b; (void)c; (void)d; (void)e;
            if (minor != 0) continue;   // 分区行
            r += rs;
            w += ws;
        }
        const double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (disk_prev_valid_ && t > disk_prev_t_) {
            const double dt = t - disk_prev_t_;
            now["disk_read_mb"] = static_cast<double>(r - disk_prev_r_) * 512.0 / dt / (1 << 20);
            now["disk_write_mb"] = static_cast<double>(w - disk_prev_w_) * 512.0 / dt / (1 << 20);
        }
        disk_prev_t_ = t;
        disk_prev_r_ = r;
        disk_prev_w_ = w;
        disk_prev_valid_ = true;
    }
#endif
}

// 10 系列键表（spec §6）：每样一次各推一值（读不到推 nullopt），封顶 60；浮点四舍五入两位
// （Strata: round(v, 2)）。
void Telemetry::push_history() {
    static const char* const kSeries[] = {
        "gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb",
        "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"};
    std::lock_guard<std::mutex> lk(mu_);
    for (const char* key : kSeries) {
        auto& vec = hist_[key];
        if (now_.count(key) && now_.at(key)) {
            const double v = *now_.at(key);
            vec.push_back(std::round(v * 100.0) / 100.0);
        } else {
            vec.push_back(std::nullopt);
        }
        if (vec.size() > 60) vec.erase(vec.begin());
    }
}

TelemetrySnapshot Telemetry::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    TelemetrySnapshot out;
    out.now = now_;
    out.gpus = gpus_read_;
    out.stat = stat_;
    out.history = hist_;
    return out;
}

}  // namespace qinfer::serve