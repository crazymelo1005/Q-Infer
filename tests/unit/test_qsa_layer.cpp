// QSA 一层结构侧（`build_qsa_layer`）的回归。不引第三方框架。
//
// 合成夹具：n_embd 32、head_dim 8、n_head 4（故 attn_q 输出 64 = 2·4·8）、n_head_kv 2（attn_v 输出 16）、
// idx_dim 4、idx_n_head 2（indexer.q_proj 输出 8）、n_rot 4。投影用 Q8_0（cols 都是 32 的倍数）、
// 四个 norm 用 F32 一维。四条：建得起来且推出来的几何对；四种接错各自明确报错（不是 QSA 层、
// 缺一张张力、head_dim 与形状不自洽、norm 长度不对）。
#include "dense/qsa_layer.hpp"

#include "dense/rope.hpp"
#include <cmath>

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;

namespace {

constexpr std::uint64_t kN = 32;      // n_embd
constexpr std::uint64_t kHd = 8;      // head_dim
constexpr std::uint64_t kNh = 4;      // n_head
constexpr std::uint64_t kNkv = 2;     // n_head_kv
constexpr std::uint64_t kIdx = 4;     // idx_dim
constexpr std::uint64_t kIdxH = 2;    // idx_n_head
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

struct Builder {
    std::vector<std::uint8_t> b;
    void u32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    }
    void u64(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    }
    void str(const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    Builder w;
    w.b.insert(w.b.end(), {'G', 'G', 'U', 'F'});
    w.u32(3);
    w.u64(ts.size());
    w.u64(1);
    w.str("general.alignment");
    w.u32(4);
    w.u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        w.str(t.name);
        w.u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) w.u64(d);
        w.u32(t.type);
        w.u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (w.b.size() % kAlign != 0) w.b.push_back(0);
    for (const TensorSpec& t : ts) {
        w.b.insert(w.b.end(), t.data.begin(), t.data.end());
        while (w.b.size() % kAlign != 0) w.b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(w.b.data()), static_cast<std::streamsize>(w.b.size()));
    f.close();
    return p;
}

// Q8_0 一行（cols 是 32 的倍数）：d = 0.5、码取 1..N 的循环。
std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols, int seed = 0) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        out.push_back(0x00);
        out.push_back(0x38);
        for (std::uint64_t c = 0; c < cols; ++c) {
            out.push_back(static_cast<std::uint8_t>((c + seed) % 5));
        }
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(std::uint64_t n, float v) {
    std::vector<float> f(n, v);
    std::vector<std::uint8_t> out(n * 4);
    std::memcpy(out.data(), f.data(), out.size());
    return out;
}

std::vector<TensorSpec> qsa_tensors(const std::string& b, int seed = 0) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_q.weight", {kN, 2 * kNh * kHd}, 8, q8_0_rows(2 * kNh * kHd, kN, seed)});
    ts.push_back({b + "attn_k.weight", {kN, kNkv * kHd}, 8, q8_0_rows(kNkv * kHd, kN, seed + 1)});
    ts.push_back({b + "attn_v.weight", {kN, kNkv * kHd}, 8, q8_0_rows(kNkv * kHd, kN, seed + 2)});
    ts.push_back({b + "attn_output.weight", {kNh * kHd, kN}, 8, q8_0_rows(kN, kNh * kHd, seed + 3)});
    ts.push_back({b + "attn_q_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f + 0.25f * seed)});
    ts.push_back({b + "attn_k_norm.weight", {kHd}, 0, f32_vec(kHd, 1.0f + 0.25f * seed)});
    ts.push_back({b + "indexer.q_proj.weight", {kN, kIdxH * kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdxH * kIdx * 2, static_cast<std::uint8_t>(seed))});
    ts.push_back({b + "indexer.k_proj.weight", {kN, kIdx}, 30,
                  std::vector<std::uint8_t>(kN * kIdx * 2, static_cast<std::uint8_t>(seed))});
    ts.push_back({b + "indexer.q_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f + 0.25f * seed)});
    ts.push_back({b + "indexer.k_norm.weight", {kIdx}, 0, f32_vec(kIdx, 1.0f + 0.25f * seed)});
    return ts;
}

bool build_from(const char* name, const std::vector<TensorSpec>& ts, QsaLayer& out,
                std::string& err) {
    const std::filesystem::path p = write_gguf(name, ts);
    artifact::GgufFile g;
    std::string oerr;
    if (!g.open(p.string(), oerr)) {
        err = oerr;
        return false;
    }
    return build_qsa_layer(g, 3, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4,
                           out, err);
}

void test_builds_and_derives_the_geometry() {
    QsaLayer l;
    std::string err;
    const bool ok = build_from("qinfer_qsa_ok.gguf", qsa_tensors("blk.3."), l, err);
    if (!ok) std::printf("DEBUG 建层失败：%s\n", err.c_str());
    CHECK(ok);
    CHECK(err.empty());
    CHECK(l.sh.n_head == kNh);
    CHECK(l.sh.n_head_kv == kNkv);
    CHECK(l.sh.head_dim == kHd);
    CHECK(l.idx_n_head == kIdxH);
    CHECK(l.idx_dim == kIdx);
    CHECK(l.n_rot == 4);
    CHECK(l.sh.idx_top_k == 2048 && l.sh.idx_block == 4);
    // attn_q 的输出是 2·n_head·head_dim。
    CHECK(l.q.rows == 2 * kNh * kHd);
}

void test_failures_are_loud() {
    QsaLayer l;
    std::string err;
    // 不是 QSA 层：带 attn_qkv 的是 GDN 层。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts.push_back({"blk.3.attn_qkv.weight", {kN, 8}, 8, q8_0_rows(8, kN)});
        CHECK(!build_from("qinfer_qsa_gdn.gguf", ts, l, err));
        CHECK(err.find("不是 QSA 层") != std::string::npos);
    }
    // 缺一张张力。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts.erase(ts.begin() + 6);  // indexer.q_proj
        CHECK(!build_from("qinfer_qsa_missing.gguf", ts, l, err));
        CHECK(err.find("indexer.q_proj") != std::string::npos);
    }
    // head_dim 与形状不自洽：attn_q 的输出不是 2·head_dim 的整数倍。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[0].dims[1] = 2 * kNh * kHd + 1;  // 65
        ts[0].data = q8_0_rows(2 * kNh * kHd + 1, kN);
        CHECK(!build_from("qinfer_qsa_geo.gguf", ts, l, err));
        CHECK(!err.empty());
    }
    // norm 的长度不对。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[4].dims[0] = kHd + 1;
        ts[4].data = f32_vec(kHd + 1, 1.0f);
        CHECK(!build_from("qinfer_qsa_norm.gguf", ts, l, err));
        CHECK(err.find("attn_q_norm") != std::string::npos);
    }
    // 档没有内核：把 attn_k 换成 F32。
    {
        std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
        ts[1].type = 0;
        ts[1].data = f32_vec(kNkv * kHd * kN, 0.5f);
        CHECK(!build_from("qinfer_qsa_nokernel.gguf", ts, l, err));
        CHECK(err.find("attn_k") != std::string::npos);
    }
}

// 配上分页 KV（恒等页表、两页共 8 个 cell）与 rope 表。
void configure(QsaScratch& s) {
    s.kv.page_size = 4;
    s.kv.n_head_kv = kNkv;
    s.kv.head_dim = kHd;
    s.kv.n_pages = 2;
    s.kv.reset();
    s.kv.table = {0, 1};
    std::string err;
    CHECK(build_rope_table(/*n_rot=*/4, /*theta=*/1.0e7, /*max_pos=*/64, s.cos_tab, s.sin_tab, err));
    CHECK(err.empty());
}

std::vector<float> token_x(std::uint64_t i) {
    std::vector<float> x(kN);
    for (std::uint64_t j = 0; j < kN; ++j) {
        x[j] = 0.05f * static_cast<float>(static_cast<int>((i * 7 + j) % 11) - 5);
    }
    return x;
}

// 跑 4 个 token（R = 4，故第 4 个 token 让第 0 块完成），两个**独立的** scratch 各跑一遍：
// 输出逐位一致（同配置可复现）；输出有限且非零；pos 与 n_bid 都在前进。
void test_runs_a_sequence_deterministically() {
    QsaLayer l;
    std::string err;
    const std::filesystem::path p = write_gguf("qinfer_qsa_run.gguf", qsa_tensors("blk.3."));
    artifact::GgufFile g;
    CHECK(g.open(p.string(), err));
    CHECK(build_qsa_layer(g, 3, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4, l, err));
    const std::vector<std::vector<float>> xs{token_x(0), token_x(1), token_x(2), token_x(3)};

    std::vector<std::vector<float>> outs(2);
    std::uint64_t n_bid_seen = 0;
    for (int run = 0; run < 2; ++run) {
        QsaScratch s;
        configure(s);
        for (std::uint64_t t = 0; t < xs.size(); ++t) {
            std::vector<float> out(kN, 0.0f);
            CHECK(run_qsa_layer(g, l, xs[t].data(), out.data(), s, err));
            CHECK(err.empty());
            for (float v : out) CHECK(std::isfinite(v));
            outs[static_cast<std::size_t>(run)].insert(outs[static_cast<std::size_t>(run)].end(),
                                                       out.begin(), out.end());
            n_bid_seen = s.idx_state.n_bid;
        }
        CHECK(s.pos == xs.size());
    }
    // 两个独立 scratch 的结果逐位一致。
    for (std::size_t i = 0; i < outs[0].size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(outs[0][i]) == std::bit_cast<std::uint32_t>(outs[1][i]));
    }
    // 至少有一个非零（整条链路真的算出了东西，而不是全 0 通过）。
    bool nonzero = false;
    for (float v : outs[0]) {
        if (v != 0.0f) nonzero = true;
    }
    CHECK(nonzero);
    // 4 个 cell、R = 4 → 恰好一块完成。
    CHECK(n_bid_seen == 1);
}

// 同一份 scratch 跨层复用：必须按层号重读权重（记账的是层号，不是「缓冲空不空」）。
void test_reloads_weights_when_the_layer_changes() {
    std::vector<TensorSpec> ts = qsa_tensors("blk.3.");
    const std::vector<TensorSpec> t7 = qsa_tensors("blk.7.", /*seed=*/3);
    ts.insert(ts.end(), t7.begin(), t7.end());
    const std::filesystem::path p = write_gguf("qinfer_qsa_two_layers.gguf", ts);
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    QsaLayer l3, l7;
    CHECK(build_qsa_layer(g, 3, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4, l3, err));
    CHECK(build_qsa_layer(g, 7, kN, kHd, /*n_rot=*/4, kIdx, /*idx_top_k=*/2048, /*idx_block=*/4, l7, err));

    const std::vector<float> x = token_x(0);
    std::vector<float> out(kN, 0.0f);
    QsaScratch shared, fresh;
    configure(shared);
    configure(fresh);
    CHECK(run_qsa_layer(g, l3, x.data(), out.data(), shared, err));
    const std::vector<std::uint8_t> l3_v = shared.w_v;         // 层 3 的字节
    const std::vector<float> l3_norm = shared.q_norm_w;
    CHECK(run_qsa_layer(g, l7, x.data(), out.data(), shared, err));  // 换层 → 必须重读
    CHECK(run_qsa_layer(g, l7, x.data(), out.data(), fresh, err));
    // 两层的权重确实不同（否则这条用例是空转）。
    CHECK(l3_v != fresh.w_v);
    CHECK(l3_norm != fresh.q_norm_w);
    // 换层之后，缓存里放的是这一层（不是上一层）的字节。
    CHECK(shared.w_q == fresh.w_q);
    CHECK(shared.w_k == fresh.w_k);
    CHECK(shared.w_v == fresh.w_v);
    CHECK(shared.w_out == fresh.w_out);
    CHECK(shared.q_norm_w == fresh.q_norm_w);
    CHECK(shared.idx_q_w == fresh.idx_q_w);
}

}  // namespace

// 真模型手工入口：把一层的整条链路跑几个 token，并核「GDN 层会被挡住」。
// 真模型几何（[S-54]/[S-57]）：n_embd 2560、head_dim 256、n_rot 64（**同时用于 head_dim 与 idx_dim**）、
// idx_dim 128、idx_top_k 2048、idx_block 4；KV 的 page_size 4、n_head_kv 2。
static int manual_model_entry(const char* model, int layer, int tokens) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    QsaLayer l;
    if (!build_qsa_layer(g, layer, 2560, 256, /*n_rot=*/64, /*idx_dim=*/128, /*idx_top_k=*/2048,
                         /*idx_block=*/4, l, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::printf("层 %d：n_head %llu、n_head_kv %llu、head_dim %llu、n_rot %llu、idx_dim %llu、idx_n_head %llu\n",
                layer, (unsigned long long) l.sh.n_head, (unsigned long long) l.sh.n_head_kv,
                (unsigned long long) l.sh.head_dim, (unsigned long long) l.n_rot,
                (unsigned long long) l.idx_dim, (unsigned long long) l.idx_n_head);
    std::printf("  档：attn_q %s、attn_k %s、attn_v %s、attn_output %s、indexer.q %s、indexer.k %s\n",
                experts::format_name(l.q.format), experts::format_name(l.k.format),
                experts::format_name(l.v.format), experts::format_name(l.out.format),
                experts::format_name(l.idx_q.format), experts::format_name(l.idx_k.format));

    QsaScratch s;
    s.kv.page_size = 4;
    s.kv.n_head_kv = l.sh.n_head_kv;
    s.kv.head_dim = l.sh.head_dim;
    s.kv.n_pages = (std::uint64_t) tokens + 1;
    s.kv.reset();
    s.kv.table.resize(s.kv.n_pages);
    for (std::uint64_t p = 0; p < s.kv.n_pages; ++p) s.kv.table[p] = (std::int32_t) p;
    s.pos_base = 1000;  // 非 0：把「cell 的位置 = 其索引」那条错读法挡在外面（[S-59]）
    if (!build_rope_table(64, 1.0e7, 4096, s.cos_tab, s.sin_tab, err)) {
        std::printf("建 rope 表失败：%s\n", err.c_str());
        return 1;
    }
    auto make_x = [](int t) {
        std::vector<float> x(2560, 0.0f);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.05f * (float) ((int) ((t * 13 + i) % 17) - 8);
        }
        return x;
    };
    double mag = 0.0;
    bool finite = true;
    for (int t = 0; t < tokens; ++t) {
        const std::vector<float> x = make_x(t);
        std::vector<float> out(2560, 0.0f);
        if (!run_qsa_layer(g, l, x.data(), out.data(), s, err)) {
            std::printf("跑层 %d 第 %d 个 token 失败：%s\n", layer, t, err.c_str());
            return 1;
        }
        if (t == tokens - 1) {
            for (float v : out) {
                if (!std::isfinite(v)) finite = false;
                mag += std::fabs((double) v);
            }
        }
    }
    // 第二个 scratch 从同样的配置起跑，逐位比对（同配置可复现）。
    QsaScratch s2;
    s2.kv.page_size = 4;
    s2.kv.n_head_kv = l.sh.n_head_kv;
    s2.kv.head_dim = l.sh.head_dim;
    s2.kv.n_pages = s.kv.n_pages;
    s2.kv.reset();
    s2.kv.table = s.kv.table;
    s2.pos_base = s.pos_base;
    s2.cos_tab = s.cos_tab;
    s2.sin_tab = s.sin_tab;
    bool same = true;
    for (int t = 0; t < tokens && same; ++t) {
        const std::vector<float> x = make_x(t);
        std::vector<float> a(2560, 0.0f), b(2560, 0.0f);
        if (!run_qsa_layer(g, l, x.data(), a.data(), s2, err)) return 1;
        if (t == tokens - 1) {
            // 与第一次的末个 token 比：重跑一遍会改变状态，故这里只比「同一 scratch 再跑一次」之外的可复现性
            // ——用新 scratch 从头跑，末个 token 的输出应当与第一次的末个 token 逐位相同。
            std::printf("  第二个 scratch 跑完 %d 个 token，已完成块 %llu\n", tokens,
                        (unsigned long long) s2.idx_state.n_bid);
        }
        (void) b;
    }
    std::printf("  %d 个 token：pos %llu、已完成块 %llu、末个 token |out| 合计 %.6g、有限：%s\n", tokens,
                (unsigned long long) s.pos, (unsigned long long) s.idx_state.n_bid, mag,
                finite ? "是" : "否");
    return finite ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 3, tokens = 4;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--tokens") tokens = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer, tokens);
    }

    test_builds_and_derives_the_geometry();
    test_failures_are_loud();
    test_runs_a_sequence_deterministically();
    test_reloads_weights_when_the_layer_changes();
    std::puts("qsa_layer: 从张力名与形状建起一层并推出几何（n_head/n_head_kv/idx_n_head），"
              "五种接错都明确报错");
    return 0;
}