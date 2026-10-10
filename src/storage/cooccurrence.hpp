// 共现图：同一 token（或同一会话）内被一起激活的 (层, 专家) 对，作为替换策略的输入。
//
// engine §5 第 1 条要它当策略输入；engine §3.4 的内存预算把「共现图」列为要给出稀疏密度上限的一项，
// 故这里的容量是**显式参数**。两条规则：
//
//   1. 计数衰减：每 aging_interval 次观察把全表计数减半、归零的条目丢掉。这样「新出现的对」有机会进来，
//      而陈旧的弱对会自己退场；代价是每 aging_interval 次观察做一次 O(表容量) 的扫描，摊还下来是 O(1)。
//      计数因此是「带衰减的频次」，不是原始频次——它只当排序的输入用，不当统计量引用。
//   2. 满则拒收：衰减之后仍然满，就不再接纳新对（已有的对照常计数），并记进 dropped_pairs。
//
// 为什么不做「满时丢最弱的那一对」：那需要全表扫描，实测在 28,080 条记录的真轨迹上把一次重放从
// 十几毫秒拖到 14 分钟（上限 262,144）。拒绝新对 + 衰减得到同样的有界性与「留强汰弱」效果，且是 O(1)。
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qinfer::storage {

// (层, 专家)。两者都小于 2^16（本模型层数 48、专家数 512），故能压进一个 u32 当键。
struct ExpertKey {
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
};

inline bool operator==(ExpertKey a, ExpertKey b) {
    return a.layer == b.layer && a.expert == b.expert;
}
inline bool operator<(ExpertKey a, ExpertKey b) {
    return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
}

// 键的 32 位编码；层或专家超出 16 位时返回 0xFFFFFFFF（调用方不该有这种键）。
std::uint32_t key_code(ExpertKey k);

class Cooccurrence {
public:
    // max_pairs = 0 表示不记录（退化成「无共现」）；aging_interval = 0 表示不衰减。
    explicit Cooccurrence(std::size_t max_pairs, std::uint64_t aging_interval = 1u << 20)
        : max_pairs_(max_pairs), aging_interval_(aging_interval) {}

    // 记一次「这一组键在同一个 token 里被一起激活」：组内两两 +1（无向，i<j 只记一次）。
    void observe(const ExpertKey* keys, int n);

    // a 与 b 的共现计数（带衰减）；没记过返回 0。
    std::uint32_t score(ExpertKey a, ExpertKey b) const;

    // k 与一组键的共现计数之和，策略用它排序。
    std::uint64_t score_with(ExpertKey k, const ExpertKey* query, int n) const;

    std::size_t pairs() const { return counts_.size(); }
    std::uint64_t dropped_pairs() const { return dropped_; }  // 因满而被拒收的新对数
    std::uint64_t agings() const { return agings_; }
    std::size_t max_pairs() const { return max_pairs_; }

private:
    void age();

    std::unordered_map<std::uint64_t, std::uint32_t> counts_;
    std::size_t max_pairs_ = 0;
    std::uint64_t aging_interval_ = 0;
    std::uint64_t observations_ = 0;  // 记过的对次（含重复），衰减的节拍
    std::uint64_t dropped_ = 0;
    std::uint64_t agings_ = 0;
};

}  // namespace qinfer::storage