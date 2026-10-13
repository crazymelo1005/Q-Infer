#include "serve/prompt_cache.hpp"

namespace qinfer::serve {

PromptCache::PromptCache(std::size_t capacity) : capacity_(capacity) {}

bool PromptCache::lookup(const std::string& key, std::vector<std::int32_t>& ids) {
    if (capacity_ == 0) return false;   // 禁用态不计数（/metrics 的 enabled=false 由此而来）
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = map_.find(key);
    if (it == map_.end()) {
        ++misses_;
        return false;
    }
    ids = it->second->second;                        // 拷出（条目本身保留）
    lru_.splice(lru_.begin(), lru_, it->second);     // 提到头
    ++hits_;
    return true;
}

void PromptCache::store(const std::string& key, const std::vector<std::int32_t>& ids) {
    if (capacity_ == 0 || ids.size() > kMaxTokensPerEntry) return;
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = map_.find(key);
    if (it != map_.end()) {                          // 同键覆写：先扣旧值再整体替换
        tokens_ -= it->second->second.size();
        lru_.erase(it->second);
        map_.erase(it);
    }
    lru_.emplace_front(key, ids);
    map_[key] = lru_.begin();
    tokens_ += ids.size();
    while (lru_.size() > capacity_) {                // 满容量丢最老（LRU）
        tokens_ -= lru_.back().second.size();
        map_.erase(lru_.back().first);
        lru_.pop_back();
    }
}

PromptCache::Stats PromptCache::stats() const {
    if (capacity_ == 0) return {};                   // 禁用态：零快照
    std::lock_guard<std::mutex> lock(mu_);
    Stats s;
    s.hits = hits_;
    s.misses = misses_;
    s.entries = lru_.size();
    s.tokens = tokens_;
    return s;
}

}  // namespace qinfer::serve