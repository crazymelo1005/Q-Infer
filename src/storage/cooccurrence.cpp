#include "storage/cooccurrence.hpp"

#include <limits>

namespace qinfer::storage {

std::uint32_t key_code(ExpertKey k) {
    if (k.layer > 0xFFFFu || k.expert > 0xFFFFu) return 0xFFFFFFFFu;
    return (k.layer << 16) | k.expert;
}

namespace {

// 无向对的编码：小的在前，保证 (a,b) 与 (b,a) 同键。
std::uint64_t pair_code(std::uint32_t a, std::uint32_t b) {
    return a <= b ? (static_cast<std::uint64_t>(a) << 32) | b
                  : (static_cast<std::uint64_t>(b) << 32) | a;
}

}  // namespace

void Cooccurrence::observe(const ExpertKey* keys, int n) {
    if (max_pairs_ == 0 || keys == nullptr || n < 2) return;
    for (int i = 0; i < n; ++i) {
        const std::uint32_t ca = key_code(keys[i]);
        if (ca == 0xFFFFFFFFu) continue;
        for (int j = i + 1; j < n; ++j) {
            const std::uint32_t cb = key_code(keys[j]);
            if (cb == 0xFFFFFFFFu) continue;
            const std::uint64_t code = pair_code(ca, cb);
            auto it = counts_.find(code);
            if (it != counts_.end()) {
                ++it->second;
            } else if (counts_.size() < max_pairs_) {
                counts_.emplace(code, 1);
            } else {
                ++dropped_;  // 满了：拒收新对（见头文件里为什么不做「丢最弱」的全表扫描）
            }
            ++observations_;
            if (aging_interval_ != 0 && observations_ % aging_interval_ == 0) age();
        }
    }
}

void Cooccurrence::age() {
    ++agings_;
    for (auto it = counts_.begin(); it != counts_.end();) {
        it->second >>= 1;
        if (it->second == 0) {
            it = counts_.erase(it);
        } else {
            ++it;
        }
    }
}

std::uint32_t Cooccurrence::score(ExpertKey a, ExpertKey b) const {
    const std::uint32_t ca = key_code(a);
    const std::uint32_t cb = key_code(b);
    if (ca == 0xFFFFFFFFu || cb == 0xFFFFFFFFu || ca == cb) return 0;
    auto it = counts_.find(pair_code(ca, cb));
    return it == counts_.end() ? 0 : it->second;
}

std::uint64_t Cooccurrence::score_with(ExpertKey k, const ExpertKey* query, int n) const {
    if (query == nullptr || n <= 0) return 0;
    std::uint64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += score(k, query[i]);
    return sum;
}

}  // namespace qinfer::storage