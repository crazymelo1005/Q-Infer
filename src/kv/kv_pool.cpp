#include "kv/kv_pool.hpp"

#include <algorithm>

namespace qinfer::kv {

const char* to_string(Pool p) {
    switch (p) {
        case Pool::kGdnState: return "gdn-state";
        case Pool::kQsaKv: return "qsa-kv";
    }
    return "?";
}

std::uint64_t Pools::resident_bytes_for(std::uint64_t context_tokens) const {
    if (cfg_.qsa_bytes_per_cell == 0 || cfg_.qsa_resident_cell_cap == 0 || context_tokens == 0) return 0;
    const std::uint64_t cells =
        std::min<std::uint64_t>(context_tokens, cfg_.qsa_resident_cell_cap);
    return cells * cfg_.qsa_bytes_per_cell;
}

std::uint64_t Pools::total_bytes_for(std::uint64_t context_tokens) const {
    return context_tokens * cfg_.qsa_bytes_per_cell;
}

std::uint64_t Pools::used(Pool pool) const {
    std::uint64_t sum = 0;
    for (const auto& kv : seqs_) {
        sum += pool == Pool::kGdnState ? kv.second.gdn_bytes : kv.second.qsa_resident_bytes;
    }
    return sum;
}

std::uint64_t Pools::remaining(Pool pool) const {
    // admit 先查余量再记账，evict/release 只会减，故 used <= limit 是不变量（有测试钉住），
    // 这里不需要再防守一次。
    const std::uint64_t limit = pool == Pool::kGdnState ? cfg_.gdn_limit_bytes : cfg_.qsa_limit_bytes;
    return limit - used(pool);
}

const SequenceEntry* Pools::query(std::uint64_t sequence_id) const {
    auto it = seqs_.find(sequence_id);
    return it == seqs_.end() ? nullptr : &it->second;
}

bool Pools::admit(std::uint64_t sequence_id, std::uint64_t context_tokens, std::string* why) {
    if (seqs_.count(sequence_id) != 0) {
        if (why != nullptr) *why = "序列已在池里";
        return false;
    }
    const std::uint64_t gdn = cfg_.gdn_bytes_per_sequence;
    const std::uint64_t qsa = resident_bytes_for(context_tokens);
    // 约束：两个池都要有余量，任一不足即拒绝。
    if (gdn > remaining(Pool::kGdnState)) {
        if (why != nullptr) *why = "GDN 状态池余量不足";
        return false;
    }
    if (qsa > remaining(Pool::kQsaKv)) {
        if (why != nullptr) *why = "QSA KV 池余量不足";
        return false;
    }
    SequenceEntry e;
    e.id = sequence_id;
    e.gdn_bytes = gdn;
    e.qsa_resident_bytes = qsa;
    e.qsa_total_bytes = total_bytes_for(context_tokens);
    seqs_.emplace(sequence_id, e);
    return true;
}

bool Pools::release(std::uint64_t sequence_id) {
    auto it = seqs_.find(sequence_id);
    if (it == seqs_.end()) return false;
    seqs_.erase(it);
    return true;
}

std::uint64_t Pools::evict(Pool pool) {
    if (seqs_.empty()) return 0;
    if (pool == Pool::kQsaKv) {
        // 常驻窗口最大的那个序列：清零它的常驻字节（改全流式读），序列保留。
        auto best = seqs_.begin();
        for (auto it = seqs_.begin(); it != seqs_.end(); ++it) {
            const std::uint64_t a = it->second.qsa_resident_bytes;
            const std::uint64_t b = best->second.qsa_resident_bytes;
            if (a > b || (a == b && it->first < best->first)) best = it;
        }
        const std::uint64_t freed = best->second.qsa_resident_bytes;
        best->second.qsa_resident_bytes = 0;
        return freed;
    }
    // GDN 侧全有或全无：把占用最大的那个序列整个赶走，两个池的字节一起释放。
    auto best = seqs_.begin();
    for (auto it = seqs_.begin(); it != seqs_.end(); ++it) {
        const std::uint64_t a = it->second.gdn_bytes;
        const std::uint64_t b = best->second.gdn_bytes;
        if (a > b || (a == b && it->first < best->first)) best = it;
    }
    const std::uint64_t freed = best->second.gdn_bytes;
    seqs_.erase(best);
    return freed;
}

}  // namespace qinfer::kv