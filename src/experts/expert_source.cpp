#include "experts/expert_source.hpp"

namespace qinfer::experts {

namespace {

// 专家字节按 (层, 专家) 压成一个 64 位块号；页表与缓存看到的是同一个编号。
std::uint64_t pack_key(std::uint32_t layer, std::uint64_t expert) {
    return (static_cast<std::uint64_t>(layer) << 32) | (expert & 0xFFFFFFFFull);
}

// 页表的可驱逐判定：不在表里（没登记）也算可驱逐；在表里则要求引用计数归零且预取不在途。
bool evictable_fn(const storage::ExpertKey& key, void* ctx) {
    const storage::PageTable* pt = static_cast<const storage::PageTable*>(ctx);
    const storage::BlockId id{pack_key(key.layer, key.expert)};
    if (pt->query(id) == nullptr) return true;
    return pt->evictable(id);
}

}  // namespace

ExpertSource::ExpertSource(const artifact::GgufFile& gguf, const std::string& tensor_prefix,
                           const LayerSpec& spec, const storage::EvictionPolicy& policy,
                           std::size_t slots, int ways, std::uint64_t byte_budget)
    : gguf_(gguf),
      cache_(slots, ways, policy),
      budget_(byte_budget) {
    gate_ = gguf.find(tensor_prefix + "gate_exps.weight");
    up_ = gguf.find(tensor_prefix + "up_exps.weight");
    down_ = gguf.find(tensor_prefix + "down_exps.weight");
    layer_ = static_cast<std::uint32_t>(spec.layer < 0 ? 0 : spec.layer);
    gate_bytes_ = spec.gate.rows * spec.gate.row_bytes();
    up_bytes_ = spec.up.rows * spec.up.row_bytes();
    down_bytes_ = spec.down.rows * spec.down.row_bytes();
    expert_bytes_ = gate_bytes_ + up_bytes_ + down_bytes_;
}

storage::BlockId ExpertSource::id_of(std::uint64_t expert) const {
    return storage::BlockId{pack_key(layer_, expert)};
}

bool ExpertSource::read_into(std::uint64_t expert, std::vector<std::uint8_t>& dst,
                             std::string& err) const {
    if (gate_ == nullptr || up_ == nullptr || down_ == nullptr) {
        err = "专家矩阵张力缺失";
        return false;
    }
    dst.resize(static_cast<std::size_t>(expert_bytes_));
    if (!gguf_.read_at(gate_->offset + expert * gate_bytes_, dst.data(),
                       static_cast<std::size_t>(gate_bytes_)) ||
        !gguf_.read_at(up_->offset + expert * up_bytes_, dst.data() + gate_bytes_,
                       static_cast<std::size_t>(up_bytes_)) ||
        !gguf_.read_at(down_->offset + expert * down_bytes_, dst.data() + gate_bytes_ + up_bytes_,
                       static_cast<std::size_t>(down_bytes_))) {
        err = "读专家 " + std::to_string(expert) + " 的矩阵失败";
        return false;
    }
    return true;
}

void ExpertSource::begin_token() {
    ++step_;
    // 每步推进世代：页表的 evictable 要求「本世代刚搬进来的不动」（interfaces §2 的不变量之外那条
    // 世代保护），不推进的话本步搬进来的东西下一步也换不掉，缓存就只能填满空位、再也换不动。
    page_.advance_generation();
}

void ExpertSource::observe_token(const std::uint32_t* experts, int n) {
    if (experts == nullptr || n <= 0) return;
    std::vector<storage::ExpertKey> keys(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        keys[static_cast<std::size_t>(i)] = storage::ExpertKey{layer_, experts[i]};
    }
    cache_.observe_step(keys.data(), n);
}

bool ExpertSource::get(std::uint64_t expert, ExpertBytes& out, std::string& err) {
    err.clear();
    const storage::ExpertKey key{layer_, static_cast<std::uint32_t>(expert)};

    // 预算用尽就地走临时缓冲：不再准入（缓存装不下该退化，不能让正确性受影响）。
    const bool budget_full =
        budget_ != 0 && bytes_resident_ + expert_bytes_ > budget_ && bytes_.find(expert) == bytes_.end();
    if (budget_full) {
        ++transient_reads_;
        if (!read_into(expert, transient_, err)) return false;
        out.gate = transient_.data();
        out.up = transient_.data() + gate_bytes_;
        out.down = transient_.data() + gate_bytes_ + up_bytes_;
        out.resident = false;
        return true;
    }

    storage::ExpertKey victim{};
    const storage::ExpertCache::Outcome verdict =
        cache_.access(key, step_, &evictable_fn, &page_, &victim);
    if (verdict == storage::ExpertCache::Outcome::kMissBlocked) {
        // 组内全不可驱逐：这次不换，改走临时缓冲（推迟到下一次，而不是硬换）。
        ++blocked_;
        ++transient_reads_;
        if (!read_into(expert, transient_, err)) return false;
        out.gate = transient_.data();
        out.up = transient_.data() + gate_bytes_;
        out.down = transient_.data() + gate_bytes_ + up_bytes_;
        out.resident = false;
        return true;
    }

    if (verdict == storage::ExpertCache::Outcome::kHit) {
        auto it = bytes_.find(expert);
        if (it == bytes_.end()) {
            err = "缓存报告命中但常驻区里没有这个专家（状态不一致）";
            return false;
        }
        out.gate = it->second.data();
        out.up = it->second.data() + gate_bytes_;
        out.down = it->second.data() + gate_bytes_ + up_bytes_;
        out.resident = true;
        return true;
    }

    // 未命中且占了槽位：先把被换出的那个从常驻区与页表里退掉，再读入新的。
    if (verdict == storage::ExpertCache::Outcome::kMissReplaced) {
        const std::uint64_t victim_expert = victim.expert;
        auto it = bytes_.find(victim_expert);
        if (it != bytes_.end()) {
            bytes_resident_ -= it->second.size();
            bytes_.erase(it);
        }
        page_.remove(storage::BlockId{pack_key(victim.layer, victim_expert)});
    }

    page_.admit(id_of(expert), storage::BlockClass::kExpert, storage::Tier::kVram, expert_bytes_);
    std::vector<std::uint8_t> fresh;
    if (!read_into(expert, fresh, err)) return false;
    bytes_resident_ += fresh.size();
    auto ins = bytes_.emplace(expert, std::move(fresh));
    out.gate = ins.first->second.data();
    out.up = ins.first->second.data() + gate_bytes_;
    out.down = ins.first->second.data() + gate_bytes_ + up_bytes_;
    out.resident = true;
    return true;
}

bool ExpertSource::pin(std::uint64_t expert, std::string& err) {
    const storage::BlockId id = id_of(expert);
    if (page_.query(id) == nullptr) {
        err = "pin 一个不在常驻区的专家";
        return false;
    }
    return page_.acquire(id, step_, /*must=*/true);
}

bool ExpertSource::unpin(std::uint64_t expert, std::string& err) {
    const storage::BlockId id = id_of(expert);
    if (!page_.release(id)) {
        err = "unpin 一个引用计数已为 0 的专家";
        return false;
    }
    return true;
}

}  // namespace qinfer::experts