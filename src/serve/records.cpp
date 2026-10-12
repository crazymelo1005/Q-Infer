#include "serve/records.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace qinfer::serve {

// Ruling 4：rng_ 用 random_device 播种，让 hpp 里的注释为真（缺省种子 5489u 的 mt19937_64
// 首值超 2^48，压不住 12 位 hex；播种后 id 仍是随机分布，测试不依赖具体值）。
Records::Records() : rng_(std::random_device{}()) {}

namespace {

// steady 当前秒数（未结束记录的实时 wallclock_s 用）。
double steady_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

std::string Records::next_id() {
    char buf[13];
    // 掩 48 位：%012llx 打满 64 位会超 12 位 hex；掩码后恒 12 位，前导零也合法（Ruling 4）。
    std::snprintf(buf, sizeof(buf), "%012llx",
                  static_cast<unsigned long long>(rng_() & 0xFFFFFFFFFFFFull));
    return buf;
}

std::string Records::begin(const RequestBegin& b) {
    RequestRecord r;
    r.id = next_id();
    r.path = b.path;
    r.model = b.model;
    r.stream = b.stream;
    r.response_format = b.response_format;
    r.created = b.created;
    r.input = b.input;
    truncate_inplace(r.input, r.input_truncated);
    r.clock_ = steady_s();
    const std::string id = r.id;   // 先留 id：move 进环后 r 已被搬空
    std::lock_guard<std::mutex> lk(mu_);
    ring_.push_back(std::move(r));
    if (ring_.size() > kCap) ring_.erase(ring_.begin());
    return id;
}

void Records::finish(const std::string& id, const RequestFinish& f) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& r : ring_) {
        if (r.id != id) continue;
        r.state = f.error.empty() ? "completed" : "error";
        r.output = f.text;
        truncate_inplace(r.output, r.output_truncated);
        r.queue_s = f.queue_s;
        r.first_token_s = f.first_token_s;
        r.wallclock_s = f.wallclock_s;
        r.decode_tok_s = f.decode_tok_s;
        r.prompt_tokens = f.prompt_tokens;
        r.output_tokens = f.output_tokens;
        r.finished = true;
        break;
    }
}

void Records::truncate_inplace(std::string& s, bool& flag) const {
    if (s.size() > kTruncBytes) {
        s.resize(kTruncBytes);   // 字节级截断（Strata 是字符级；v0 按字节，注释为证）
        flag = true;
    }
}

// 未结束的 wallclock_s 补实时（Strata 的 list 视图同款）。
static void fill_live(RequestRecord& r) {
    if (!r.finished) r.wallclock_s = steady_s() - r.clock_;
}

bool Records::detail(const std::string& id, RequestRecord& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& r : ring_) {
        if (r.id == id) {
            out = r;
            fill_live(out);
            return true;
        }
    }
    return false;
}

std::vector<RequestRecord> Records::list() const {
    std::vector<RequestRecord> out;
    {
        std::lock_guard<std::mutex> lk(mu_);
        out = ring_;
    }
    for (auto& r : out) fill_live(r);
    std::reverse(out.begin(), out.end());   // 最新在前
    return out;
}

std::size_t Records::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ring_.size();
}

}  // namespace qinfer::serve