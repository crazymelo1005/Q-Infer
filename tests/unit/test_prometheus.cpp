// /metrics 的 Prometheus text 端口（M2 监控）：
// 内容协商谓词（Accept/format → text 或 JSON）、Latencies 三直方图（ttft/itl/e2e，
// vLLM 22 桶、累计计数）、render_prometheus 映射（vLLM 家族名 + qinfer: 前缀家族，
// 与 JSON 同一快照——/metrics 构建一份 Json 后按协商渲染）。
// 契约对照 Strata serve/prometheus.py；strict 校验：每家族恰好一条 # TYPE。
#include "serve/json.hpp"
#include "serve/prometheus.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include "check.hpp"

using namespace qinfer::serve;

namespace {

// ---------------- 测试内宽松解析：样本 → {(名, 标签串): 值}，并校验 TYPE 唯一性 ----------------
struct Parsed {
    std::map<std::string, std::map<std::string, double>> samples;  // 名 → {标签串 → 值}
    std::set<std::string> typed;                                   // 已见 # TYPE 家族
};

std::string family_of(const std::string& name) {
    for (const char* sfx : {"_bucket", "_sum", "_count"}) {
        if (name.size() > std::string(sfx).size() && name.ends_with(sfx))
            return name.substr(0, name.size() - std::string(sfx).size());
    }
    return name;
}

Parsed parse(const std::string& text) {
    Parsed pg;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("# TYPE ", 0) == 0) {
                std::istringstream ls(line);
                std::string h, t, fam;
                ls >> h >> t >> fam;
                CHECK(pg.typed.insert(fam).second);   // 同家族只允许一条 TYPE
            }
            continue;
        }
        const std::size_t br = line.find('{');
        const std::size_t hx = line.find("} ");
        CHECK(br != std::string::npos && hx != std::string::npos && hx > br);
        const std::string name = line.substr(0, br);
        const std::string labels = line.substr(br + 1, hx - br - 1);
        const std::string val = line.substr(hx + 2);
        const std::string fam = family_of(name);
        CHECK(pg.typed.count(fam) || pg.typed.count(name));   // 样本行必须属某个已声明 TYPE 家族
        pg.samples[name][labels] = std::stod(val);
    }
    return pg;
}

double get(const Parsed& pg, const std::string& name, const std::string& labels) {
    const auto fam = pg.samples.find(name);
    CHECK(fam != pg.samples.end());
    const auto it = fam->second.find(labels);
    CHECK(it != fam->second.end());
    return it->second;
}

bool has(const Parsed& pg, const std::string& name) { return pg.samples.count(name) > 0; }

// ---------------- 1. 内容协商 ----------------
void test_wants_prometheus() {
    CHECK(wants_prometheus("text/plain;version=0.0.4", ""));
    CHECK(wants_prometheus("application/openmetrics-text;version=1.0.0", ""));
    CHECK(wants_prometheus("", "format=prometheus"));
    CHECK(wants_prometheus("*/*", "format=prometheus"));
    CHECK(!wants_prometheus("application/json", ""));
    CHECK(!wants_prometheus("*/*", ""));
}

// ---------------- 2. Latencies：桶累计、间隙口径 ----------------
std::size_t bucket_index(double v) {
    const auto& bs = vllm_buckets();
    for (std::size_t i = 0; i < bs.size(); ++i)
        if (bs[i] == v) return i;
    CHECK(false && "桶不存在");
    return 0;
}

void test_latencies() {
    CHECK(vllm_buckets().size() == 22);
    CHECK(vllm_buckets()[0] == 0.001);
    CHECK(vllm_buckets()[21] == 2560.0);

    Latencies lat;
    // 一次 11 token 的完整请求：ttft 0.3s、10 个间隙共 1.0s、全程 1.5s。
    lat.observe(0.3, 11, 1.0, 1.5);
    // 一次 1 token 的失败请求：无首 token（ttft 跳过）、无间隙（itl 跳过）、全程 0.2s。
    lat.observe(0.0, 1, 0.0, 0.2);
    const LatencySnapshot s = lat.snapshot();

    CHECK(s.ttft.n == 1);
    CHECK(s.ttft.sum == 0.3);
    CHECK(s.ttft.counts[bucket_index(0.5)] == 1);    // 0.3 首个含它的桶是 0.5
    CHECK(s.ttft.counts[bucket_index(0.25)] == 0);   // 0.3 > 0.25 → 不入更小桶

    CHECK(s.itl.n == 10);
    CHECK(std::fabs(s.itl.sum - 1.0) < 1e-9);
    CHECK(s.itl.counts[bucket_index(0.1)] == 10);
    CHECK(s.itl.counts[bucket_index(0.08)] == 0);   // 0.1 > 0.08 → 不入更小桶
    for (std::size_t i = 1; i < s.itl.counts.size(); ++i)
        CHECK(s.itl.counts[i] >= s.itl.counts[i - 1]);   // 累计桶单调不减

    CHECK(s.e2e.n == 2);
    CHECK(s.e2e.counts[bucket_index(2.5)] == 2);     // 1.5 与 0.2 都 ≤ 2.5
    CHECK(s.e2e.counts[bucket_index(0.25)] == 1);    // 只有 0.2 入 0.25 桶
    CHECK(s.e2e.sum == 1.7);
}

// ---------------- 3. render 映射（vLLM + qinfer 家族） ----------------
void test_render_maps_vllm_and_qinfer() {
    Json m = Json::objv();
    Json eng = Json::objv();
    eng.set("model", Json::strv("a \"b\""));
    eng.set("max_context", Json::numv(1000.0));
    m.set("engine", std::move(eng));

    Json live = Json::objv();
    live.set("state", Json::strv("generating"));
    live.set("queued", Json::numv(2.0));
    live.set("running", Json::numv(1.0));
    live.set("waiting", Json::numv(2.0));
    live.set("parallel", Json::numv(4.0));
    live.set("prompt_tokens", Json::numv(400.0));
    live.set("generated", Json::numv(100.0));
    live.set("tok_s", Json::nullv());   // v0 无进行中速率 → qinfer:live_tok_s 不出现
    m.set("live", std::move(live));

    Json cc = Json::objv();
    cc.set("enabled", Json::boolv(true));
    cc.set("capacity", Json::numv(64.0));
    cc.set("hits", Json::numv(40.0));
    cc.set("misses", Json::numv(10.0));
    cc.set("entries", Json::numv(3.0));
    m.set("conversation_cache", std::move(cc));

    Json total = Json::objv();
    total.set("requests", Json::numv(5.0));
    total.set("prompt_tokens", Json::numv(900.0));
    total.set("reused", Json::numv(600.0));
    total.set("completion_tokens", Json::numv(50.0));
    total.set("prompt_ms", Json::numv(1500.0));
    total.set("decode_ms", Json::numv(2000.0));
    m.set("totals", std::move(total));

    Json hw = Json::objv();
    hw.set("cpu", Json::numv(12.5));
    hw.set("ram_used", Json::numv(static_cast<double>(8LL << 30)));
    hw.set("ram_total", Json::numv(static_cast<double>(64LL << 30)));
    hw.set("gpu_util", Json::numv(40.0));   // 单卡：走 now 键，样本 gpu="0"
    m.set("hardware", std::move(hw));

    Latencies lat;
    lat.observe(0.3, 11, 1.0, 1.5);
    lat.observe(0.0, 1, 0.0, 0.2);
    const std::string text = render_prometheus(m, lat.snapshot(), "a \"b\"");
    const Parsed pg = parse(text);
    const std::string lab = "model_name=\"a \\\"b\\\"\"";   // 标签转义

    // ---- vLLM 家族 ----
    CHECK(get(pg, "vllm:num_requests_running", lab) == 1.0);
    CHECK(get(pg, "vllm:num_requests_waiting", lab) == 2.0);
    CHECK(get(pg, "vllm:kv_cache_usage_perc", lab) == 0.5);
    CHECK(get(pg, "vllm:prompt_tokens_total", lab) == 900.0);
    CHECK(get(pg, "vllm:generation_tokens_total", lab) == 50.0);
    CHECK(get(pg, "vllm:request_success_total", lab) == 5.0);
    CHECK(get(pg, "vllm:prefix_cache_queries_total", lab) == 50.0);   // hits + misses
    CHECK(get(pg, "vllm:prefix_cache_hits_total", lab) == 40.0);
    CHECK(get(pg, "vllm:num_preemptions_total", lab) == 0.0);

    // ---- 直方图家族：桶、+Inf、sum、count ----
    const std::string l01 = lab + ",le=\"0.1\"";
    const std::string l008 = lab + ",le=\"0.08\"";
    const std::string linf = lab + ",le=\"+Inf\"";
    CHECK(get(pg, "vllm:inter_token_latency_seconds_bucket", l01) == 10.0);
    CHECK(get(pg, "vllm:inter_token_latency_seconds_bucket", l008) == 0.0);
    CHECK(get(pg, "vllm:inter_token_latency_seconds_bucket", linf) == 10.0);
    CHECK(get(pg, "vllm:inter_token_latency_seconds_sum", lab) == 1.0);
    CHECK(get(pg, "vllm:inter_token_latency_seconds_count", lab) == 10.0);
    CHECK(get(pg, "vllm:time_to_first_token_seconds_count", lab) == 1.0);   // 第二个请求无首 token
    CHECK(get(pg, "vllm:e2e_request_latency_seconds_count", lab) == 2.0);

    // ---- qinfer: 家族（JSON 自有事实） ----
    CHECK(get(pg, "qinfer:live_state", lab + ",state=\"generating\"") == 1.0);
    CHECK(get(pg, "qinfer:live_state", lab + ",state=\"idle\"") == 0.0);
    CHECK(get(pg, "qinfer:live_queued", lab) == 2.0);
    CHECK(get(pg, "qinfer:live_parallel", lab) == 4.0);
    CHECK(!has(pg, "qinfer:live_tok_s"));   // JSON null → 无样本
    CHECK(get(pg, "qinfer:engine_max_context", lab) == 1000.0);
    CHECK(get(pg, "qinfer:totals_prompt_seconds_total", lab) == 1.5);
    CHECK(get(pg, "qinfer:totals_decode_seconds_total", lab) == 2.0);
    CHECK(get(pg, "qinfer:totals_reused_total", lab) == 600.0);
    CHECK(get(pg, "qinfer:conversation_cache_hits_total", lab) == 40.0);
    CHECK(get(pg, "qinfer:conversation_cache_misses_total", lab) == 10.0);
    CHECK(get(pg, "qinfer:conversation_cache_entries", lab) == 3.0);
    CHECK(get(pg, "qinfer:conversation_cache_capacity", lab) == 64.0);
    CHECK(get(pg, "qinfer:cpu", lab) == 12.5);
    CHECK(get(pg, "qinfer:ram_used_bytes", lab) == static_cast<double>(8LL << 30));
    CHECK(get(pg, "qinfer:ram_total_bytes", lab) == static_cast<double>(64LL << 30));
    CHECK(get(pg, "qinfer:gpu_util", lab + ",gpu=\"0\"") == 40.0);
}

// ---------------- 4. 多卡 gpus[]：逐卡样本；未测到的不出样本 ----------------
void test_render_multi_gpu() {
    Json m = Json::objv();
    Json eng = Json::objv();
    eng.set("model", Json::strv("m"));
    m.set("engine", std::move(eng));
    m.set("live", Json::objv());
    m.set("totals", Json::objv());
    m.set("conversation_cache", Json::objv());

    Json hw = Json::objv();
    Json gpus = Json::arrv();
    Json g0 = Json::objv();
    g0.set("index", Json::numv(0.0));
    g0.set("util", Json::numv(30.0));
    g0.set("mem_used", Json::numv(static_cast<double>(1LL << 30)));
    g0.set("temp", Json::numv(60.0));
    gpus.push(std::move(g0));
    Json g1 = Json::objv();
    g1.set("index", Json::numv(1.0));
    g1.set("util", Json::numv(50.0));
    g1.set("mem_used", Json::numv(static_cast<double>(2LL << 30)));
    g1.set("temp", Json::numv(65.0));
    gpus.push(std::move(g1));
    hw.set("gpus", std::move(gpus));
    m.set("hardware", std::move(hw));

    const std::string text = render_prometheus(m, Latencies().snapshot(), "m");
    const Parsed pg = parse(text);
    const std::string lab = "model_name=\"m\"";
    CHECK(get(pg, "qinfer:gpu_util", lab + ",gpu=\"0\"") == 30.0);
    CHECK(get(pg, "qinfer:gpu_util", lab + ",gpu=\"1\"") == 50.0);
    CHECK(get(pg, "qinfer:gpu_mem_used_bytes", lab + ",gpu=\"0\"") == static_cast<double>(1LL << 30));
    CHECK(get(pg, "qinfer:gpu_temp_celsius", lab + ",gpu=\"1\"") == 65.0);
    CHECK(!has(pg, "qinfer:gpu_power_watts"));   // power 未测 → 没有样本
}

}  // namespace

int main() {
    test_wants_prometheus();
    test_latencies();
    test_render_maps_vllm_and_qinfer();
    test_render_multi_gpu();
    std::printf("prometheus 全部通过\n");
    return 0;
}