// KV 常驻窗口（`KvWindow`）与它在 QSA 一层里的接线的回归。不引第三方框架。
//
// 合成夹具是一层 QSA（n_embd 32 / head_dim 8 / n_head 4 / n_head_kv 2 / idx_dim 4 / n_rot 4），
// 分页 KV 用 page_size 2（故 4 个 token 跨 2 页）。页字节 = 2(k+v) × n_head_kv 2 × page_size 2 ×
// head_dim 8 × 2 = 128。
//
// 钉子：
//   1. 窗口本体：配额、LRU 驱逐（驱逐最久未用的、不是最新用的）、钉住后拒绝驱逐（记一次被拒并返回假）、
//      松开后能再准入、钉住计数下溢报错；
//   2. `attach_kv_window` 会把逻辑页表整表标成「不在显存」——否则预填的恒等映射会让窗口形同虚设；
//   3. 挂窗口跑出来的输出与**不挂窗口**（恒等页表）逐位一致：常驻与否只是记账，不改数值；
//   4. 配额小到装不下「本步刚写的页 + 本步要读的页」时**明确报错**，不静默读错（错读法：不钉住刚写的页）。
#include "dense/kv_window.hpp"

#include "dense/rope.hpp"
#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kN = 32, kHd = 8, kNh = 4, kNkv = 2, kIdx = 4, kIdxH = 2;
constexpr std::uint64_t kQk = kNkv * kHd;
constexpr std::uint64_t kPageSize = 2;
constexpr std::uint64_t kNPages = 4;
constexpr std::uint64_t kPageBytes = 2 * kNkv * kPageSize * kHd * 2;  // 128
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    std::vector<std::uint8_t> b;
    auto u32 = [&b](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u64 = [&b](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    const auto put_str = [&](const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    put_str("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        put_str(t.name);
        u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (b.size() % kAlign != 0) b.push_back(0);
    for (const TensorSpec& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        while (b.size() % kAlign != 0) b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p;
}

std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        out.push_back(0x00);
        out.push_back(0x38);
        for (std::uint64_t c = 0; c < cols; ++c) out.push_back(static_cast<std::uint8_t>(c % 5));
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(std::uint64_t n, float v) {
    std::vector<float> f(n, v);
    std::vector<std::uint8_t> out(n * 4);
    std::memcpy(out.data(), f.data(), out.size());
    return out;
}

std::vector<TensorSpec> qsa_tensors() {
    const std::string b = "blk.3.";
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_q.weight", {kN, 2 * kNh * kHd}, 8, q8_0_rows(2 * kNh * kHd, kN)});
    ts.push_back({b + "attn_k.weight", {kN, kQk}, 8, q8_0_rows(kQk, kN)});
    ts.push_back({b + "attn_v.weight", {kN, kQk}, 8, q8_0_rows(kQk, kN)});
    ts.push_back({b + "attn_output.weight", {kNh * kHd, kN}, 8, q8_0_rows(kN, kNh * kHd)});
    ts.push_back({b + "attn_q_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f)});
    ts.push_back({b + "attn_k_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f)});
    ts.push_back({b + "indexer.q_proj.weight", {kN, kIdxH * kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdxH * kIdx * 2, 0)});
    ts.push_back({b + "indexer.k_proj.weight", {kN, kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdx * 2, 0)});
    ts.push_back({b + "indexer.q_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f)});
    ts.push_back({b + "indexer.k_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f)});
    return ts;
}

// 配好分页 KV（page_size 2、恒等页表）与 rope 表。
void configure(QsaScratch& s) {
    s.kv.page_size = kPageSize;
    s.kv.n_head_kv = kNkv;
    s.kv.head_dim = kHd;
    s.kv.n_pages = kNPages;
    s.kv.reset();
    s.kv.table.assign(kNPages, 0);
    for (std::uint64_t p = 0; p < kNPages; ++p) s.kv.table[p] = static_cast<std::int32_t>(p);
    std::string err;
    CHECK(build_rope_table(/*n_rot=*/4, 1.0e7, 64, s.cos_tab, s.sin_tab, err));
    CHECK(err.empty());
}

std::vector<float> token_x(std::uint64_t i) {
    std::vector<float> x(kN);
    for (std::uint64_t j = 0; j < kN; ++j) {
        x[j] = 0.05f * static_cast<float>(static_cast<int>((i * 7 + j) % 11) - 5);
    }
    return x;
}

// ---- 窗口本体 ----

void test_window_unit() {
    KvWindow w;
    w.page_bytes = 100;
    w.quota_bytes = 200;
    std::vector<std::int32_t> tbl(4, -1);
    w.table = &tbl;  // 挂上表：准入写、驱逐写回 -1
    CHECK(w.capacity_pages() == 2);
    std::string err;
    std::uint64_t p = 999;
    CHECK(w.ensure(0, p, err) && p == 0);
    CHECK(w.ensure(1, p, err) && p == 1);
    CHECK(w.misses == 2 && w.hits == 0);
    CHECK(tbl[0] == 0 && tbl[1] == 1);  // 准入写进了表
    CHECK(w.ensure(0, p, err) && p == 0);  // 命中，刷新使用序
    CHECK(w.hits == 1);
    // 再进第 3 页 → 驱逐最久未用的（页 1，因为页 0 刚被刷新过），并把表写回 -1。
    CHECK(w.ensure(2, p, err));
    CHECK(w.evictions == 1);
    CHECK(w.resident(0) && !w.resident(1) && w.resident(2));
    CHECK(tbl[1] == -1);  // 被驱逐 → 不在显存
    CHECK(tbl[0] == 0 && tbl[2] == 2);

    // 钉住后拒绝驱逐。
    KvWindow w2;
    w2.page_bytes = 100;
    w2.quota_bytes = 200;
    std::uint64_t q = 0;
    CHECK(w2.ensure(0, q, err) && w2.pin(0, err));
    CHECK(w2.ensure(1, q, err) && w2.pin(1, err));
    err.clear();
    CHECK(!w2.ensure(2, q, err));  // 满且全被钉住
    CHECK(w2.refusals == 1);
    CHECK(err.find("钉住") != std::string::npos);
    CHECK(w2.entries.size() == 2 && w2.evictions == 0);
    // 松开一页之后就能准入。
    CHECK(w2.unpin(0, err));
    CHECK(w2.ensure(2, q, err));
    CHECK(w2.evictions == 1 && !w2.resident(0));

    // 计数下溢与未常驻都报错。
    CHECK(!w2.unpin(0, err));                       // 已被驱逐
    CHECK(!w2.pin(99, err));                        // 从未在窗口里
    CHECK(w2.pin(2, err) && w2.unpin(2, err));
    CHECK(!w2.unpin(2, err));                       // 计数已是 0

    // 配额为 0 表示不限；页字节为 0 是调用方错。
    KvWindow w3;
    w3.page_bytes = 100;
    w3.quota_bytes = 0;
    CHECK(w3.capacity_pages() > 1000000u);
    KvWindow w4;  // 没给页字节数：调用方错，明确报错（不算「被拒」）
    w4.quota_bytes = 100;
    std::uint64_t r = 0;
    CHECK(!w4.ensure(0, r, err));
    CHECK(err.find("页字节数") != std::string::npos);
    // 配额装不下一页。
    KvWindow w5;
    w5.page_bytes = 100;
    w5.quota_bytes = 50;
    CHECK(!w5.ensure(0, r, err) && w5.refusals == 1);
    CHECK(err.find("装不下一页") != std::string::npos);
}

void test_attach_clears_table() {
    QsaScratch s;
    configure(s);
    for (std::int32_t v : s.kv.table) CHECK(v >= 0);  // 配完是恒等映射
    KvWindow w;
    w.page_bytes = kPageBytes;
    w.quota_bytes = kNPages * kPageBytes;
    attach_kv_window(s, w);
    CHECK(s.window == &w);
    for (std::int32_t v : s.kv.table) CHECK(v == -1);  // 整表标成「不在显存」
}

// ---- 接进 QSA 一层 ----

// 跑 n 个 token，返回拼起来的输出；window 为空则用恒等页表。
std::vector<float> run_qsa(const char* file, int n, KvWindow* win, double& out_misses) {
    const std::filesystem::path p = write_gguf(file, qsa_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    QsaLayer l;
    CHECK(build_qsa_layer(g, 3, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4, l,
                          err));
    QsaScratch s;
    configure(s);
    if (win != nullptr) attach_kv_window(s, *win);
    std::vector<float> all;
    for (int t = 0; t < n; ++t) {
        const std::vector<float> x = token_x(static_cast<std::uint64_t>(t));
        std::vector<float> out(kN, 0.0f);
        CHECK(run_qsa_layer(g, l, x.data(), out.data(), s, err));
        CHECK(err.empty());
        all.insert(all.end(), out.begin(), out.end());
    }
    out_misses = win != nullptr ? static_cast<double>(win->misses) : 0.0;
    return all;
}

void test_window_matches_identity() {
    // 不挂窗口（恒等页表）。
    double m0 = 0.0;
    const std::vector<float> a = run_qsa("qinfer_kvwin_plain.gguf", 4, nullptr, m0);
    // 挂窗口：配额 4 页（够装 2 页、不驱逐）。
    KvWindow w;
    w.page_bytes = kPageBytes;
    w.quota_bytes = kNPages * kPageBytes;
    double m1 = 0.0;
    const std::vector<float> b = run_qsa("qinfer_kvwin_win.gguf", 4, &w, m1);
    if (a.size() != b.size()) CHECK(false);
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i]));
    }
    CHECK(m1 > 0.0);                 // 页确实被搬进来过
    CHECK(w.refusals == 0 && w.evictions == 0);
    CHECK(w.entries.size() == 2u);   // 4 个 token / page_size 2 = 2 页

    // 配额恰好 1 页：本步刚写的页被钉住，本步要读的旧页就进不来 → 明确报错（不静默读错）。
    const std::filesystem::path p = write_gguf("qinfer_kvwin_tight.gguf", qsa_tensors());
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    QsaLayer l;
    CHECK(build_qsa_layer(g, 3, kN, kHd, 4, kIdx, 2048, 4, l, err));
    QsaScratch s;
    configure(s);
    KvWindow tight;
    tight.page_bytes = kPageBytes;
    tight.quota_bytes = kPageBytes;  // 只装得下一页
    attach_kv_window(s, tight);
    bool failed = false;
    for (int t = 0; t < 4; ++t) {
        const std::vector<float> x = token_x(static_cast<std::uint64_t>(t));
        std::vector<float> out(kN, 0.0f);
        if (!run_qsa_layer(g, l, x.data(), out.data(), s, err)) {
            failed = true;
            break;
        }
    }
    CHECK(failed);
    CHECK(tight.refusals >= 1);
    CHECK(err.find("钉住") != std::string::npos);
    // 不变式：**本步刚写入的那一页没有被换出**。t=2 写的是页 1（cell 2、3 在页 1），
    // 它被钉住，故失败时仍在窗口里（错读法——不钉住刚写的页——会让它先被换出，这一条就是判据）。
    CHECK(tight.resident(1));
}

}  // namespace

int main() {
    test_window_unit();
    test_attach_clears_table();
    test_window_matches_identity();
    std::puts("kv_window: 配额/LRU 驱逐/钉住拒绝都可观测；挂窗口与恒等页表逐位一致；"
              "配额装不下本步工作集时明确报错");
    return 0;
}
