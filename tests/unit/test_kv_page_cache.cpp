// QSA 分页 KV 的回归：写进去、按 id 收回来。不引第三方框架。
//
// 四条：写—收回来的**往返**一致（对任意选出的 id 子集都成立，且是全 bit 一致而不是近似）；
// 页内是 head-major（一个头的连续 cell 连续）——把同页两个 cell 的不同头写进去，能看出排布；
// 表里 `-1`（不在显存）必须**明确报错**而不是当成页 0 静默读；聚集按给定顺序落槽位。
// 另把三处错读法各按错的算一遍要求明显偏离：页内改成 cell-major、页表下标用 cell 号而不是逻辑页、
// 忽略 -1。
#include "dense/kv_page_cache.hpp"

#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kPage = 4;    // page_size（真模型 4）
constexpr std::uint64_t kNkv = 2;     // kv 头数（真模型 2）
constexpr std::uint64_t kDim = 4;     // 头宽（真模型 256，这里取小）
constexpr std::uint64_t kNPages = 3;  // 物理页数
constexpr std::uint64_t kCells = 10;  // 要写的 cell 数（跨 3 个逻辑页）

std::vector<std::uint16_t> make_cell(std::uint64_t t, bool is_k) {
    std::vector<std::uint16_t> v(kNkv * kDim);
    for (std::uint64_t i = 0; i < v.size(); ++i) {
        v[static_cast<std::size_t>(i)] =
            static_cast<std::uint16_t>(1000 * (is_k ? 1 : 2) + 10 * t + i);
    }
    return v;
}

KvPageCache make_cache(const std::vector<std::int32_t>& table) {
    KvPageCache c;
    c.page_size = kPage;
    c.n_head_kv = kNkv;
    c.head_dim = kDim;
    c.n_pages = kNPages;
    c.reset();
    c.table = table;
    return c;
}

void test_round_trip() {
    // 3 个逻辑页映射到 3 个物理页（顺序打乱：逻辑页 0→物理页 2、1→0、2→1）。
    KvPageCache c = make_cache({2, 0, 1});
    std::vector<std::vector<std::uint16_t>> ks, vs;
    std::string err;
    for (std::uint64_t t = 0; t < kCells; ++t) {
        const std::vector<std::uint16_t> k = make_cell(t, true), v = make_cell(t, false);
        ks.push_back(k);
        vs.push_back(v);
        CHECK(kv_append(c, t, KvCell{k.data(), v.data()}, err));
        CHECK(err.empty());
    }
    // 选一部分 cell（乱序给出也不行——按给定顺序落槽位，故这里按升序给）。
    const std::vector<std::uint64_t> ids{0, 3, 4, 8, 9};
    std::vector<std::uint16_t> kscr(ids.size() * kNkv * kDim, 0), vscr(kscr.size(), 0);
    CHECK(kv_gather(c, ids.data(), ids.size(), kscr.data(), vscr.data(), err));
    CHECK(err.empty());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::uint64_t t = ids[i];
        for (std::uint64_t j = 0; j < kNkv * kDim; ++j) {
            CHECK(kscr[i * kNkv * kDim + j] == ks[static_cast<std::size_t>(t)][j]);
            CHECK(vscr[i * kNkv * kDim + j] == vs[static_cast<std::size_t>(t)][j]);
        }
    }
}

// 页内 head-major：同页两个 cell 的同一个头必须相邻（相隔 head_dim），而不是同一个 cell 的两个头相邻。
void test_page_layout_is_head_major() {
    KvPageCache c = make_cache({0, 1, 2});
    std::string err;
    const std::vector<std::uint16_t> k0 = make_cell(0, true), k1 = make_cell(1, true);
    CHECK(kv_append(c, 0, KvCell{k0.data(), k0.data()}, err));
    CHECK(kv_append(c, 1, KvCell{k1.data(), k1.data()}, err));
    // 物理页 0 上的排布：头 0 的 cell0、cell1 在前（0 与 head_dim 处），头 1 的从 page_size*head_dim 起。
    const std::uint64_t h0c0 = 0;
    const std::uint64_t h0c1 = kDim;
    const std::uint64_t h1c0 = kPage * kDim;
    CHECK(c.k_pool[h0c0] == k0[0]);
    CHECK(c.k_pool[h0c1] == k1[0]);
    CHECK(c.k_pool[h1c0] == k0[kDim]);
    // 错的读法（cell-major）会把 (cell0 的 head1) 放在 head_dim 处——这里断言那里不是它。
    CHECK(c.k_pool[kDim] != k0[kDim]);
}

void test_missing_page_fails_loudly() {
    KvPageCache c = make_cache({0, -1, 1});  // 逻辑页 1 不在显存
    std::string err;
    const std::vector<std::uint16_t> k = make_cell(0, true);
    CHECK(kv_append(c, 0, KvCell{k.data(), k.data()}, err));  // 逻辑页 0 在
    CHECK(err.empty());
    err.clear();
    CHECK(!kv_append(c, 4, KvCell{k.data(), k.data()}, err));  // 逻辑页 1 不在 -> 明确失败
    CHECK(err.find("不在显存") != std::string::npos);
    err.clear();
    std::vector<std::uint16_t> kscr(kNkv * kDim, 0), vscr(kscr.size(), 0);
    const std::vector<std::uint64_t> ids{5};
    CHECK(!kv_gather(c, ids.data(), 1, kscr.data(), vscr.data(), err));
    CHECK(err.find("不在显存") != std::string::npos);
    // 页表下标越大越界也要报。
    err.clear();
    CHECK(!kv_append(c, 100, KvCell{k.data(), k.data()}, err));
    CHECK(!err.empty());
}

void test_rival_layout_is_rejected() {
    // 错的读法：页内 cell-major（`(t % page) * n_head_kv + h`）。
    KvPageCache c = make_cache({0, 1, 2});
    std::string err;
    for (std::uint64_t t = 0; t < kCells; ++t) {
        const std::vector<std::uint16_t> k = make_cell(t, true);
        CHECK(kv_append(c, t, KvCell{k.data(), k.data()}, err));
    }
    std::vector<std::uint16_t> kscr(2 * kNkv * kDim, 0), vscr(kscr.size(), 0);
    const std::vector<std::uint64_t> ids{1, 2};
    CHECK(kv_gather(c, ids.data(), 2, kscr.data(), vscr.data(), err));
    // 按 cell-major 重新解读同一块显存会得到不同的值。
    std::vector<std::uint16_t> wrong(kscr.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::uint64_t t = ids[i];
        const std::uint64_t page = 0 + t / kPage;  // 页表是恒等
        const std::uint64_t base = page * kNkv * kPage * kDim;
        for (std::uint64_t h = 0; h < kNkv; ++h) {
            const std::uint64_t from = base + ((t % kPage) * kNkv + h) * kDim;
            for (std::uint64_t d = 0; d < kDim; ++d) {
                wrong[(i * kNkv + h) * kDim + d] = c.k_pool[from + d];
            }
        }
    }
    bool differs = false;
    for (std::size_t i = 0; i < kscr.size(); ++i) {
        if (kscr[i] != wrong[i]) differs = true;
    }
    std::printf("cell-major 读法与正确的读法是否不同：%s\n", differs ? "是" : "否");
    CHECK(differs);
}

void test_edge_cases() {
    KvPageCache c = make_cache({0, 1, 2});
    std::string err;
    // 空 id 列表合法。
    CHECK(kv_gather(c, nullptr, 0, nullptr, nullptr, err));
    CHECK(err.empty());
    // 指针不完整要报。
    err.clear();
    std::vector<std::uint16_t> buf(kNkv * kDim, 0);
    const std::vector<std::uint64_t> ids{0};
    CHECK(!kv_gather(c, ids.data(), 1, nullptr, buf.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!kv_append(c, 0, KvCell{nullptr, buf.data()}, err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_round_trip();
    test_page_layout_is_head_major();
    test_missing_page_fails_loudly();
    test_rival_layout_is_rejected();
    test_edge_cases();
    std::puts("kv_page_cache: 写—收往返逐位一致、页内 head-major、不在显存的页明确报错，"
              "cell-major 的错读法被拒");
    return 0;
}