// render 与直方图实现。渲染只读 /metrics 已构建好的 Json（单一事实源），任何字段缺失都不抛：
// 缺键/null → 家族无样本（抽样器端所见即 JSON 所见，不发明读数）。
#include "serve/prometheus.hpp"

#include <charconv>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace qinfer::serve {

bool wants_prometheus(const std::string& accept, const std::string& query) {
    return query.find("format=prometheus") != std::string::npos ||
           accept.find("openmetrics") != std::string::npos ||
           accept.find("text/plain") != std::string::npos;
}

// 数值文本：整数走整数十进制（8e9 字节不写成科学计数）；小数走最短往返表示（0.3 → "0.3"）。
static std::string fmt(double v) {
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 9.2e18)
        return std::to_string(static_cast<long long>(v));
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

static void hist_observe(HistStats& s, double v, std::uint64_t times) {
    if (!std::isfinite(v) || times == 0) return;
    const auto& bs = vllm_buckets();
    for (std::size_t i = 0; i < bs.size(); ++i)
        if (v <= bs[i]) s.counts[i] += times;
    s.sum += v * static_cast<double>(times);
    s.n += times;
}

void Latencies::Hist::observe(double v, std::uint64_t times) { hist_observe(s_, v, times); }

void Latencies::observe(double ttft_s, std::uint64_t output_tokens, double decode_s, double e2e_s) {
    std::lock_guard<std::mutex> lk(mu_);
    if (std::isfinite(ttft_s) && ttft_s > 0) ttft_.observe(ttft_s, 1);
    // 间隙口径与 Strata 相同：decode 耗时 ÷ 间隙数（每间隙一次，times = 间隙数）。
    if (output_tokens > 1 && std::isfinite(decode_s) && decode_s > 0)
        itl_.observe(decode_s / static_cast<double>(output_tokens - 1), output_tokens - 1);
    if (std::isfinite(e2e_s) && e2e_s > 0) e2e_.observe(e2e_s, 1);
}

LatencySnapshot Latencies::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return LatencySnapshot{ttft_.s_, itl_.s_, e2e_.s_};
}

namespace {

// 从对象取数值字段：缺键/非数值 → nullopt（JSON null 与缺失同义）。
std::optional<double> numv(const Json& o, const char* key) {
    const Json* p = o.find(key);
    if (p == nullptr || !p->is_num()) return std::nullopt;
    return p->num;
}

std::string esc_label(const std::string& v) {
    std::string out;
    out.reserve(v.size());
    for (const char c : v) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

// 输出缓冲：每家族第一条样本前写 HELP/TYPE 各一次（families 语义要求）。
struct Out {
    std::vector<std::string> lines;
    std::vector<std::string> typed;

    void metric(const std::string& name, const std::string& kind, const std::string& help,
                const std::string& labels, std::optional<double> v) {
        if (!v.has_value()) return;
        bool first = true;
        for (const std::string& t : typed)
            if (t == name) first = false;
        if (first) {
            typed.push_back(name);
            lines.push_back("# HELP " + name + " " + help);
            lines.push_back("# TYPE " + name + " " + kind);
        }
        lines.push_back(name + "{" + labels + "} " + fmt(*v));
    }

    void histogram(const std::string& name, const std::string& help, const HistStats& s,
                   const std::string& labels) {
        lines.push_back("# HELP " + name + " " + help);
        lines.push_back("# TYPE " + name + " histogram");
        const auto& bs = vllm_buckets();
        for (std::size_t i = 0; i < bs.size(); ++i)
            lines.push_back(name + "_bucket{" + labels + ",le=\"" + fmt(bs[i]) + "\"} " +
                            std::to_string(s.counts[i]));
        lines.push_back(name + "_bucket{" + labels + ",le=\"+Inf\"} " + std::to_string(s.n));
        lines.push_back(name + "_sum{" + labels + "} " + fmt(s.sum));
        lines.push_back(name + "_count{" + labels + "} " + std::to_string(s.n));
    }
};

// GPU 样本：多卡走 hardware.gpus[]（逐卡 index 标签）；单卡走 now 键（gpu="0"），与 Strata 同构。
void gpu_samples(Out& out, const Json& hw, const std::string& lab) {
    const Json* gl = hw.find("gpus");
    if (gl != nullptr && gl->is_arr() && !gl->arr.empty()) {
        for (const Json& g : gl->arr) {
            const std::string glab = lab + ",gpu=\"" + fmt(numv(g, "index").value_or(0.0)) + "\"";
            out.metric("qinfer:gpu_util", "gauge", "GPU busy, percent (hardware.gpus[].util).", glab, numv(g, "util"));
            out.metric("qinfer:gpu_mem_used_bytes", "gauge", "GPU memory in use (hardware.gpus[].mem_used).",
                       glab, numv(g, "mem_used"));
            out.metric("qinfer:gpu_mem_total_bytes", "gauge", "GPU memory (hardware.gpus[].mem_total).",
                       glab, numv(g, "mem_total"));
            out.metric("qinfer:gpu_temp_celsius", "gauge", "GPU temperature (hardware.gpus[].temp).",
                       glab, numv(g, "temp"));
            out.metric("qinfer:gpu_power_watts", "gauge", "GPU power draw (hardware.gpus[].power).",
                       glab, numv(g, "power"));
        }
        return;
    }
    if (numv(hw, "gpu_util").has_value()) {
        const std::string glab = lab + ",gpu=\"0\"";
        out.metric("qinfer:gpu_util", "gauge", "GPU busy, percent (hardware.gpu_util).", glab, numv(hw, "gpu_util"));
        out.metric("qinfer:gpu_mem_used_bytes", "gauge", "GPU memory in use (hardware.gpu_mem_used).",
                   glab, numv(hw, "gpu_mem_used"));
        out.metric("qinfer:gpu_mem_total_bytes", "gauge", "GPU memory (hardware.gpu_mem_total).",
                   glab, numv(hw, "gpu_mem_total"));
        out.metric("qinfer:gpu_temp_celsius", "gauge", "GPU temperature (hardware.gpu_temp).",
                   glab, numv(hw, "gpu_temp"));
        out.metric("qinfer:gpu_power_watts", "gauge", "GPU power draw (hardware.gpu_power).",
                   glab, numv(hw, "gpu_power"));
    }
}

}  // namespace

std::string render_prometheus(const Json& m, const LatencySnapshot& lat, const std::string& model_name) {
    const std::string lab = "model_name=\"" + esc_label(model_name) + "\"";
    Out out;
    const Json* eng = m.find("engine");
    const Json* live = m.find("live");
    const Json* totals = m.find("totals");
    const Json* cc = m.find("conversation_cache");

    const bool busy = live != nullptr && live->find("state") != nullptr &&
                      live->find("state")->is_str() && live->find("state")->str == "generating";

    // ---- vLLM 家族：面板兼容名 ----
    if (live != nullptr) {
        out.metric("vllm:num_requests_running", "gauge", "Requests generating.", lab, numv(*live, "running"));
        out.metric("vllm:num_requests_waiting", "gauge", "Requests waiting for a slot.", lab, numv(*live, "waiting"));
    }
    // 进行中请求占上下文的份额（1 = 满）。v0 无逐项视图，prompt_tokens/generated 恒 null → 0。
    const double ctx = eng != nullptr ? numv(*eng, "max_context").value_or(0.0) : 0.0;
    double used = 0.0;
    if (busy && live != nullptr)
        used = numv(*live, "prompt_tokens").value_or(0.0) + numv(*live, "generated").value_or(0.0);
    out.metric("vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).",
               lab, ctx > 0.0 ? std::optional<double>(std::min(1.0, used / ctx)) : std::optional<double>(0.0));
    if (totals != nullptr) {
        out.metric("vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.", lab,
                   numv(*totals, "prompt_tokens"));
        out.metric("vllm:generation_tokens_total", "counter", "Generated tokens.", lab,
                   numv(*totals, "completion_tokens"));
        out.metric("vllm:request_success_total", "counter", "Finished requests.", lab, numv(*totals, "requests"));
    }
    // 前缀缓存按查询次数计（vLLM 同名指标语义）：queries = hits + misses，全串命中即免编码。
    if (cc != nullptr) {
        const double queries = numv(*cc, "hits").value_or(0.0) + numv(*cc, "misses").value_or(0.0);
        out.metric("vllm:prefix_cache_queries_total", "counter", "Prompt-cache lookups.", lab,
                   std::optional<double>(queries));
        out.metric("vllm:prefix_cache_hits_total", "counter", "Prompt-cache lookups already cached.", lab,
                   numv(*cc, "hits"));
    }
    out.metric("vllm:num_preemptions_total", "counter", "Preempted requests (this service does not preempt).", lab,
               std::optional<double>(0.0));

    out.histogram("vllm:time_to_first_token_seconds", "Time to the first token.", lat.ttft, lab);
    out.histogram("vllm:inter_token_latency_seconds", "Time between two tokens.", lat.itl, lab);
    out.histogram("vllm:e2e_request_latency_seconds", "A request from enqueue to finish.", lat.e2e, lab);

    // ---- qinfer: 家族：JSON 自有事实 ----
    if (live != nullptr) {
        const Json* state = live->find("state");
        const std::string stnow = state != nullptr && state->is_str() ? state->str : "";
        for (const char* st : {"idle", "generating"})
            out.metric("qinfer:live_state", "gauge", "1 for what the engine is doing (live.state).",
                       lab + ",state=\"" + st + "\"", std::optional<double>(stnow == st ? 1.0 : 0.0));
        out.metric("qinfer:live_queued", "gauge", "Requests queued (live.queued).", lab, numv(*live, "queued"));
        out.metric("qinfer:live_parallel", "gauge", "Slot count (live.parallel).", lab, numv(*live, "parallel"));
        out.metric("qinfer:live_tok_s", "gauge", "Decode rate of the last request (live.tok_s).", lab,
                   numv(*live, "tok_s"));
        out.metric("qinfer:live_prefill_tok_s_mean", "gauge",
                   "Prompt reading rate of the last request (live.prefill_tok_s_mean).", lab,
                   numv(*live, "prefill_tok_s_mean"));
        out.metric("qinfer:live_tok_s_mean", "gauge", "Mean decode rate (live.tok_s_mean).", lab,
                   numv(*live, "tok_s_mean"));
    }
    out.metric("qinfer:engine_max_context", "gauge", "The engine's context (engine.max_context).", lab,
               eng != nullptr ? numv(*eng, "max_context") : std::optional<double>());
    if (totals != nullptr) {
        out.metric("qinfer:totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).",
                   lab, std::optional<double>(numv(*totals, "prompt_ms").value_or(0.0) / 1000.0));
        out.metric("qinfer:totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).",
                   lab, std::optional<double>(numv(*totals, "decode_ms").value_or(0.0) / 1000.0));
        out.metric("qinfer:totals_reused_total", "counter", "Prompt tokens the token cache already held.",
                   lab, numv(*totals, "reused"));
    }
    if (cc != nullptr) {
        out.metric("qinfer:conversation_cache_hits_total", "counter", "Prompt-cache lookups that hit (cached).",
                   lab, numv(*cc, "hits"));
        out.metric("qinfer:conversation_cache_misses_total", "counter", "Prompt-cache lookups that missed.",
                   lab, numv(*cc, "misses"));
        out.metric("qinfer:conversation_cache_entries", "gauge", "Prompt-cache entries held.", lab,
                   numv(*cc, "entries"));
        out.metric("qinfer:conversation_cache_capacity", "gauge", "Prompt-cache capacity (0 = disabled).", lab,
                   numv(*cc, "capacity"));
    }

    const Json* hw = m.find("hardware");
    if (hw != nullptr) {
        out.metric("qinfer:cpu", "gauge", "CPU busy, percent (hardware.cpu).", lab, numv(*hw, "cpu"));
        out.metric("qinfer:ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", lab, numv(*hw, "ram_used"));
        out.metric("qinfer:ram_total_bytes", "gauge", "RAM (hardware.ram_total).", lab, numv(*hw, "ram_total"));
        gpu_samples(out, *hw, lab);
    }

    std::string out_text;
    for (const std::string& l : out.lines) {
        out_text += l;
        out_text.push_back('\n');
    }
    return out_text;
}

}  // namespace qinfer::serve