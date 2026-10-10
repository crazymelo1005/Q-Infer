#include "dense/qsa_select.hpp"

#include <algorithm>
#include <numeric>

namespace qinfer::dense {

std::uint64_t qsa_selection_width(std::uint64_t n_kv, const QsaShapes& s) {
    const std::uint64_t blk = s.idx_block == 0 ? 1 : s.idx_block;
    const std::uint64_t w = s.idx_top_k + blk - 1;
    return n_kv < w ? n_kv : w;
}

bool qsa_select(const float* scores, std::uint64_t n_kv, std::uint64_t width,
                std::vector<std::uint64_t>& out, std::string& err) {
    err.clear();
    out.clear();
    if (scores == nullptr && n_kv != 0) {
        err = "qsa_select 的分数指针为空";
        return false;
    }
    if (width > n_kv) {
        err = "选择宽度 " + std::to_string(width) + " 超过 cell 数 " + std::to_string(n_kv);
        return false;
    }
    if (width == 0) return true;  // 什么都不选是合法的（上下文短到不必选）

    // 排序键：分数降序、并列取索引升序。用下标全序排，规则就一条、不依赖排序是否稳定。
    std::vector<std::uint64_t> idx(n_kv);
    std::iota(idx.begin(), idx.end(), std::uint64_t{0});
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(width), idx.end(),
                      [scores](std::uint64_t a, std::uint64_t b) {
                          const float sa = scores[a], sb = scores[b];
                          if (sa != sb) return sa > sb;  // 分数大的在前
                          return a < b;                  // 并列取索引小的
                      });
    out.assign(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(width));
    // 返回索引升序（下游按页聚集要的是有序的 cell 列表，不是分数序）。
    std::sort(out.begin(), out.end());
    return true;
}

}  // namespace qinfer::dense