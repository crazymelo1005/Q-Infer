// /metrics 的 Prometheus text 端口（M2 监控）：vLLM 家族名 + qinfer: 前缀家族，与 JSON 同一份快照。
// 契约对照 Strata serve/prometheus.py：Accept 含 text/plain|openmetrics 或 ?format=prometheus → text；
// 指标名前半取 vLLM 命名（Grafana 面板免改直连），JSON 自有事实走 qinfer: 前缀；模型名单卡标在
// 全部样本上（反斜杠/引号转义）；延迟三直方图（ttft/itl/e2e，vLLM 22 桶）由完成时 observe。
#pragma once

#include "serve/json.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace qinfer::serve {

inline constexpr std::size_t kBucketCount = 22;

inline const std::array<double, kBucketCount>& vllm_buckets() {
    static const std::array<double, kBucketCount> bs = {
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
    return bs;
}

// 内容协商谓词：hold 时 /metrics 输出 text/plain；否则保持 JSON（前端与无头 curl 不破）。
bool wants_prometheus(const std::string& accept, const std::string& query);

// 一条直方图的快照：counts 为累计桶计数（单调不减），n 为样本数，sum 为数值和。
struct HistStats {
    std::array<std::uint64_t, kBucketCount> counts{};
    double sum = 0.0;
    std::uint64_t n = 0;
};

struct LatencySnapshot {
    HistStats ttft;
    HistStats itl;
    HistStats e2e;
};

// 请求延迟三直方图：worker 在 record_request 落完成时刻时 observe，/metrics 线程 snapshot。
// 自有锁，与 Json 渲染解耦——JSON 与 text 都是同一份快照的两种投影。
class Latencies {
public:
    // ttft_s：入队到首 token；output_tokens：生成 token 数（>1 才有间隙）；decode_s：首 token 到
    // 完成；e2e_s：入队到完成。非法（≤0 或非有限）输入整条跳过，不污染桶。
    void observe(double ttft_s, std::uint64_t output_tokens, double decode_s, double e2e_s);
    LatencySnapshot snapshot() const;

private:
    struct Hist {
        void observe(double v, std::uint64_t times);
        HistStats s_;
    };
    Hist ttft_, itl_, e2e_;
    mutable std::mutex mu_;
};

// 把 /metrics 构建的那一份 Json 快照渲染成 Prometheus text。JSON 中 null/缺键的字段 → 家族无样本
// （v0 的进行中视图占位符不会伪装成 0 出现）；vllm:num_preemptions_total 恒 0（本服务不抢占，
// 保 vLLM 面板兼容）；kv_cache_usage_perc 在无进行中数据时为 0。
std::string render_prometheus(const Json& m, const LatencySnapshot& lat, const std::string& model_name);

}  // namespace qinfer::serve