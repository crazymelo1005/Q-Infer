// 专家缓存的离线重放：把参考引擎 `--dump-routing` 的轨迹灌进三种替换策略，比较命中率。
//
// 轨迹格式（两个独立读者一致：引擎侧 `src/program/generate.cpp` 的写出与 `tools/make_profile.py`
// 的 `read_trace`）是逐条定长记录：
//   layer(i32) k(i32)  k×专家id(i32)  k×权重(f32)
// 一条记录 = 一层在一个位置上选的 k 个专家。解码路径才写轨迹，批量预填不写（见 S-34）。
//
// 这个工具**不是**在做性能结论：命中率按 engine §5 只作观测量，引用它必须同时给出 PCIe 侧占比与
// 空闲显存。它做的是同一批轨迹下三条策略的相对比较，以及槽位数、相联度这两个旋钮的效果。
//
// 用法：expert_cache_replay --trace <trace.bin> [--slots N] [--ways W] [--json]
#include "storage/expert_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace qinfer::storage;

namespace {

constexpr std::uint32_t kMaxExpert = 4096;   // 专家号上界（本模型 512），越界的记录整条跳过
constexpr std::uint32_t kMaxLayer = 512;     // 层数上界（本模型 48）

struct Record {
    std::uint32_t layer;
    std::vector<std::uint32_t> experts;
};

bool read_trace(const std::string& path, std::vector<Record>& out, std::uint64_t& bad) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "打不开 %s\n", path.c_str());
        return false;
    }
    out.clear();
    bad = 0;
    std::int32_t head[2];
    while (std::fread(head, sizeof(std::int32_t), 2, f) == 2) {
        const std::int32_t layer = head[0];
        const std::int32_t k = head[1];
        if (k < 0 || k > kMaxExpert) {
            // 记录长度不可信，无法安全跨过：停在这里并记账。
            ++bad;
            break;
        }
        std::vector<std::int32_t> ids(static_cast<std::size_t>(k));
        if (k > 0 && std::fread(ids.data(), sizeof(std::int32_t), static_cast<std::size_t>(k), f) !=
                          static_cast<std::size_t>(k)) {
            ++bad;
            break;
        }
        std::vector<float> weights(static_cast<std::size_t>(k));
        if (k > 0 && std::fread(weights.data(), sizeof(float), static_cast<std::size_t>(k), f) !=
                          static_cast<std::size_t>(k)) {
            ++bad;
            break;
        }
        if (layer < 0 || layer >= static_cast<std::int32_t>(kMaxLayer)) {
            ++bad;
            continue;
        }
        Record rec;
        rec.layer = static_cast<std::uint32_t>(layer);
        for (std::int32_t e : ids) {
            if (e < 0 || e >= static_cast<std::int32_t>(kMaxExpert)) {
                ++bad;
                rec.experts.clear();
                break;
            }
            rec.experts.push_back(static_cast<std::uint32_t>(e));
        }
        if (!rec.experts.empty()) out.push_back(std::move(rec));
    }
    std::fclose(f);
    return true;
}

struct RunResult {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t compulsory = 0;
    std::uint64_t replaced = 0;
    std::uint64_t blocked = 0;
    std::uint64_t pair_kinds = 0;
    std::uint64_t dropped_pairs = 0;
};

RunResult run(const std::vector<Record>& trace, std::size_t slots, int ways,
              const EvictionPolicy& policy) {
    ExpertCache cache(slots, ways, policy);
    std::vector<ExpertKey> keys;
    for (std::size_t i = 0; i < trace.size(); ++i) {
        const Record& rec = trace[i];
        keys.clear();
        for (std::uint32_t e : rec.experts) keys.push_back(ExpertKey{rec.layer, e});
        cache.observe_step(keys.data(), static_cast<int>(keys.size()));
        for (const ExpertKey& k : keys) {
            ExpertKey victim;
            cache.access(k, static_cast<std::int64_t>(i), nullptr, nullptr, &victim);
        }
    }
    const CacheStats& s = cache.stats();
    RunResult r;
    r.hits = s.hits;
    r.misses = s.misses;
    r.compulsory = s.compulsory;
    r.replaced = s.replaced;
    r.blocked = s.blocked;
    r.pair_kinds = cache.cooccurrence().pairs();
    r.dropped_pairs = cache.cooccurrence().dropped_pairs();
    return r;
}

double hit_rate(const RunResult& r) {
    const std::uint64_t total = r.hits + r.misses;
    return total == 0 ? 0.0 : static_cast<double>(r.hits) / static_cast<double>(total);
}

}  // namespace

int main(int argc, char** argv) {
    std::string trace_path;
    std::string env = "未标注";
    std::size_t slots = 8106;  // 默认取 G-09 定的部署槽位口径
    int ways = 8;
    bool json = false;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0) trace_path = argv[i + 1];
        else if (std::strcmp(argv[i], "--slots") == 0) slots = static_cast<std::size_t>(std::atol(argv[i + 1]));
        else if (std::strcmp(argv[i], "--ways") == 0) ways = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--env") == 0) env = argv[i + 1];
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) json = true;
    }
    if (trace_path.empty()) {
        std::fprintf(stderr, "用法：expert_cache_replay --trace <trace.bin> [--slots N] [--ways W] [--json]\n");
        return 2;
    }

    std::vector<Record> trace;
    std::uint64_t bad = 0;
    if (!read_trace(trace_path, trace, bad)) return 1;

    FifoPolicy fifo;
    LruPolicy lru;
    CooccurrenceAwarePolicy cooc;
    const RunResult r_fifo = run(trace, slots, ways, fifo);
    const RunResult r_lru = run(trace, slots, ways, lru);
    const RunResult r_cooc = run(trace, slots, ways, cooc);

    if (json) {
        std::printf("{\n");
        char stamp[32] = {0};
        const std::time_t now = std::time(nullptr);
        std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
        std::printf("  \"measured_at\": \"%s\",\n", stamp);
        std::printf("  \"measure\": \"序 4 专家缓存的替换策略离线重放\",\n");
        std::printf("  \"env\": \"%s\",\n", env.c_str());
        std::printf("  \"method\": \"读参考引擎 --dump-routing 的轨迹（layer i32, k i32, k 专家 i32, "
                    "k 权重 f32），逐条喂给三种替换策略；命中率只作观测量\",\n");
        std::printf("  \"trace\": \"%s\",\n", trace_path.c_str());
        std::printf("  \"records\": %llu,\n", static_cast<unsigned long long>(trace.size()));
        std::printf("  \"bad_records\": %llu,\n", static_cast<unsigned long long>(bad));
        std::printf("  \"slots\": %llu,\n", static_cast<unsigned long long>(slots));
        std::printf("  \"ways\": %d,\n", ways);
        std::printf("  \"policies\": {\n");
        const char* names[3] = {"fifo", "lru", "cooccurrence-aware"};
        const RunResult* rs[3] = {&r_fifo, &r_lru, &r_cooc};
        for (int i = 0; i < 3; ++i) {
            std::printf("    \"%s\": {\"hits\": %llu, \"misses\": %llu, \"hit_rate\": %.6f, "
                        "\"compulsory\": %llu, \"replaced\": %llu, \"blocked\": %llu}%s\n",
                        names[i], static_cast<unsigned long long>(rs[i]->hits),
                        static_cast<unsigned long long>(rs[i]->misses), hit_rate(*rs[i]),
                        static_cast<unsigned long long>(rs[i]->compulsory),
                        static_cast<unsigned long long>(rs[i]->replaced),
                        static_cast<unsigned long long>(rs[i]->blocked),
                        i == 2 ? "" : ",");
        }
        std::printf("  },\n");
        std::printf("  \"cooccurrence\": {\"pair_kinds\": %llu, \"dropped_pairs\": %llu}\n",
                    static_cast<unsigned long long>(r_cooc.pair_kinds),
                    static_cast<unsigned long long>(r_cooc.dropped_pairs));
        std::printf("}\n");
    } else {
        std::printf("轨迹 %s：%llu 条记录（坏 %llu）槽位 %llu 相联度 %d\n", trace_path.c_str(),
                    static_cast<unsigned long long>(trace.size()),
                    static_cast<unsigned long long>(bad),
                    static_cast<unsigned long long>(slots), ways);
        std::printf("  fifo               命中率 %.6f（命中 %llu / 未命中 %llu）\n", hit_rate(r_fifo),
                    static_cast<unsigned long long>(r_fifo.hits),
                    static_cast<unsigned long long>(r_fifo.misses));
        std::printf("  lru                命中率 %.6f\n", hit_rate(r_lru));
        std::printf("  cooccurrence-aware 命中率 %.6f（共现对 %llu，被上限丢弃 %llu）\n",
                    hit_rate(r_cooc), static_cast<unsigned long long>(r_cooc.pair_kinds),
                    static_cast<unsigned long long>(r_cooc.dropped_pairs));
    }
    return 0;
}