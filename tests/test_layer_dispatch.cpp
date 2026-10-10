// 一层的一批未命中专家：批取字节 → 一个作业 → 工人（含重派）→ 按专家 id 全序归并。
// 不引第三方框架。
//
// 合成夹具沿用 test_moe_layer 的几何：hidden 256 / ffn 64 / 4 专家，gate/up 用 IQ2_S（一块 256 值）、
// down 用 Q2_0（一块 64 值）。参考侧**直接按逐专家偏移从 GGUF 读字节**再调 expert_ffn，不经过
// ExpertSource，故「逐位一致」这条断言覆盖的正是本模块自己那部分（取字节、钉住或拷出、归并顺序）。
//
// 七路证据：
//   1. 与「直接按偏移读 + 逐专家 expert_ffn + 按 id 升序加权」逐位一致，且结果是**累加**进 out。
//   2. 传参顺序不影响输出——钉住「归并顺序是 id 升序，而不是调用方给的顺序」。
//   3. 一批里同时出现常驻与临时缓冲两种来源仍然正确（这一路抓「临时区被下一次读覆写」）。
//   4. 字节预算用尽时全走临时缓冲仍然正确。
//   5. 钉住的槽位算完要解开：引用计数漏解会让这些槽位再也换不掉，用 blocked 计数抓。
//   6. 首选工人不可用时改派 fallback，结果不变、且不可用的那个不该被调用（失败计数为 0）。
//   7. 参数不合法（n 为 0、空指针）明确失败。
#include "experts/layer_dispatch.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
#include "kernels/iq2s.hpp"
#include "kernels/q2_0.hpp"
#include "kernels/q8k.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::experts;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kHidden = 256;
constexpr std::uint64_t kFfn = 64;
constexpr std::uint64_t kExpert = 4;
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
    for (std::size_t i = 0; i < ts.size(); ++i) {
        w.str(ts[i].name);
        w.u32(static_cast<std::uint32_t>(ts[i].dims.size()));
        for (std::uint64_t d : ts[i].dims) w.u64(d);
        w.u32(ts[i].type);
        w.u64(at);
        at += (ts[i].data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (w.b.size() % kAlign != 0) w.b.push_back(0);
    for (std::size_t i = 0; i < ts.size(); ++i) {
        w.b.insert(w.b.end(), ts[i].data.begin(), ts[i].data.end());
        while (w.b.size() % kAlign != 0) w.b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(w.b.data()), static_cast<std::streamsize>(w.b.size()));
    f.close();
    return p;
}

void push_iq2s_block(std::vector<std::uint8_t>& out, std::uint16_t d_bits, std::uint8_t scale_lo) {
    out.push_back(static_cast<std::uint8_t>(d_bits & 0xFF));
    out.push_back(static_cast<std::uint8_t>(d_bits >> 8));
    for (int i = 0; i < 64; ++i) out.push_back(0);
    for (int i = 0; i < 8; ++i) out.push_back(0);
    for (int i = 0; i < 8; ++i) out.push_back(i == 0 ? scale_lo : 0);
}

// 第 e 个专家的低半字节尺度取 first+e，故各专家的权重不同——「哪一批字节写进了哪一段」可被断言。
std::vector<std::uint8_t> iq2s_tensor(std::uint64_t cols, std::uint64_t rows, std::uint64_t experts,
                                      std::uint64_t first) {
    std::vector<std::uint8_t> data;
    const std::uint64_t nb = cols / 256;
    for (std::uint64_t e = 0; e < experts; ++e) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t b = 0; b < nb; ++b) {
                push_iq2s_block(data, 0x3C00, static_cast<std::uint8_t>(first + e));
            }
        }
    }
    return data;
}

void push_q20_block(std::vector<std::uint8_t>& out, std::uint16_t d_bits, std::uint8_t qs) {
    out.push_back(static_cast<std::uint8_t>(d_bits & 0xFF));
    out.push_back(static_cast<std::uint8_t>(d_bits >> 8));
    for (int i = 0; i < 16; ++i) out.push_back(qs);
}

std::vector<std::uint8_t> q20_tensor(std::uint64_t cols, std::uint64_t rows, std::uint64_t experts,
                                     std::uint8_t qs) {
    std::vector<std::uint8_t> data;
    const std::uint64_t nb = cols / 64;
    for (std::uint64_t e = 0; e < experts; ++e) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t b = 0; b < nb; ++b) push_q20_block(data, 0x3C00, qs);
        }
    }
    return data;
}

std::vector<std::uint8_t> router_tensor(std::uint64_t cols, std::uint64_t experts) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(cols * experts * 2));
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        data[i] = 0x80;  // bf16 1.0
        data[i + 1] = 0x3F;
    }
    return data;
}

// 各专家权重不同的路由器：第 e 个专家的每个权重都是 bf16(1.0 + 0.25·e)。配全正输入时 logits 随 e
// 单调增，故 top-2 必为 {3, 2}——**排名序与 id 升序因此相反**，用来钉住归并顺序到底听谁的。
std::vector<std::uint8_t> router_tensor_scaled(std::uint64_t cols, std::uint64_t experts) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(cols * experts * 2));
    for (std::uint64_t e = 0; e < experts; ++e) {
        const std::uint16_t bits = f32_to_bf16_bits(1.0f + 0.25f * static_cast<float>(e));
        for (std::uint64_t j = 0; j < cols; ++j) {
            const std::size_t at = static_cast<std::size_t>((e * cols + j) * 2);
            data[at] = static_cast<std::uint8_t>(bits & 0xFF);
            data[at + 1] = static_cast<std::uint8_t>(bits >> 8);
        }
    }
    return data;
}

// 每行的 down 权重各不相同：行索引当尺度用，故「专家矩阵串了」会让整行都错。
std::vector<TensorSpec> layer_tensors(const std::string& prefix, bool scaled_router = false) {
    std::vector<TensorSpec> ts;
    ts.push_back({prefix + "ffn_gate_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, 1)});
    ts.push_back({prefix + "ffn_up_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, 1)});
    ts.push_back({prefix + "ffn_down_exps.weight", {kFfn, kHidden, kExpert}, 42,
                  q20_tensor(kFfn, kHidden, kExpert, 0xAA)});
    ts.push_back({prefix + "ffn_gate_inp.weight", {kHidden, kExpert}, 30,
                  scaled_router ? router_tensor_scaled(kHidden, kExpert)
                                : router_tensor(kHidden, kExpert)});
    return ts;
}

std::vector<float> make_x() {
    std::vector<float> x(kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.03125f;
    }
    return x;
}

// 全正输入：让 scaled 路由器下的 logits 严格随专家序号单调增，top-k 的排名序才是确定的。
// 两个细节都是必需的，否则这条用例会变成空转：
//   · 取值不用二的幂（0.002 不是二进制精确数）——量化权重都是精确的，输入若也是，整条计算就成了
//     精确算术，三项换个顺序求和仍逐位相同；
//   · 幅度要小（合计约 4.6）。scaled 路由器的 logits 是 (1+0.25e)·Σx，Σx 一大 logits 就拉开几十，
//     softmax 后只剩一项有权重，其余项小到在舍入里消失，顺序同样看不出来。
std::vector<float> make_x_pos() {
    std::vector<float> x(kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.002f * static_cast<float>(i % 17 + 1);
    }
    return x;
}

// 参考侧：直接按逐专家偏移从 GGUF 读字节，**按给定顺序**逐个 expert_ffn 再按权重累加。
std::vector<float> direct_reference_in_order(const artifact::GgufFile& g, const MoeLayer& ml,
                                             const std::vector<int>& ids,
                                             const std::vector<float>& w, const float* x,
                                             float init) {
    const std::size_t hidden = static_cast<std::size_t>(ml.hidden);
    std::vector<float> out(hidden, init);
    std::vector<std::uint8_t> gb(static_cast<std::size_t>(ml.gate_expert_bytes));
    std::vector<std::uint8_t> ub(static_cast<std::size_t>(ml.up_expert_bytes));
    std::vector<std::uint8_t> db(static_cast<std::size_t>(ml.down_expert_bytes));
    FfnScratch scratch;
    std::vector<float> one(hidden, 0.0f);
    for (std::size_t k = 0; k < ids.size(); ++k) {
        const std::uint64_t e = static_cast<std::uint64_t>(ids[k]);
        CHECK(g.read_at(ml.gate->offset + e * ml.gate_expert_bytes, gb.data(), gb.size()));
        CHECK(g.read_at(ml.up->offset + e * ml.up_expert_bytes, ub.data(), ub.size()));
        CHECK(g.read_at(ml.down->offset + e * ml.down_expert_bytes, db.data(), db.size()));
        ExpertWeights ew{gb.data(), ub.data(), db.data()};
        std::string err;
        CHECK(expert_ffn(ml.spec, ew, x, scratch, one.data(), err));
        for (std::size_t j = 0; j < hidden; ++j) out[j] += w[k] * one[j];
    }
    return out;
}

// 同上，但顺序固定为专家 id 升序——本仓库定下的归并顺序（engine §16）。
std::vector<float> direct_reference(const artifact::GgufFile& g, const MoeLayer& ml,
                                    const std::vector<int>& ids, const std::vector<float>& w,
                                    const float* x, float init) {
    std::vector<int> order(ids.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&ids](int a, int b) {
        return ids[static_cast<std::size_t>(a)] < ids[static_cast<std::size_t>(b)];
    });
    std::vector<int> oids(order.size());
    std::vector<float> ow(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        oids[i] = ids[static_cast<std::size_t>(order[i])];
        ow[i] = w[static_cast<std::size_t>(order[i])];
    }
    return direct_reference_in_order(g, ml, oids, ow, x, init);
}

bool same_bits(const std::vector<float>& got, const std::vector<float>& want, const char* what) {
    if (got.size() != want.size()) {
        std::printf("FAIL %s：长度 %zu 期 %zu\n", what, got.size(), want.size());
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
            std::printf("FAIL %s[%zu]：得 %.9g 期 %.9g\n", what, i, static_cast<double>(got[i]),
                        static_cast<double>(want[i]));
            return false;
        }
    }
    return true;
}

// 四层专家的合成模型 + 它的 MoeLayer，供各用例复用。
struct Fixture {
    std::filesystem::path path;
    artifact::GgufFile g;
    MoeLayer ml;

    Fixture(const char* name, int top_k, bool scaled_router = false)
        : path(write_gguf(name, layer_tensors("blk.0.", scaled_router))) {
        std::string err;
        CHECK(g.open(path.string(), err));
        CHECK(build_moe_layer(g, 0, top_k, ml, err));
    }
    ExpertSource source(std::size_t slots, int ways, std::uint64_t budget) const {
        return ExpertSource(g, "blk.0.ffn_", ml.spec, storage::PolicyKind::kLru, slots, ways,
                            budget);
    }
};

std::vector<std::uint32_t> as_u32(const std::vector<int>& ids) {
    std::vector<std::uint32_t> r(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) r[i] = static_cast<std::uint32_t>(ids[i]);
    return r;
}

// order（ids 的一个排列）是否恰好是按专家 id 升序的那个排列。
bool rank_is_ascending_by_id(const std::vector<int>& ids, const std::vector<int>& order) {
    if (order.size() != ids.size()) return false;
    for (std::size_t i = 1; i < order.size(); ++i) {
        const int a = ids[static_cast<std::size_t>(order[i - 1])];
        const int b = ids[static_cast<std::size_t>(order[i])];
        if (!(a < b)) return false;  // 专家互异，故严格小于
    }
    return true;
}

void test_matches_direct_and_accumulates() {
    const Fixture fx("qinfer_dispatch_basic.gguf", 3);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{3, 0, 2};
    const std::vector<float> w{0.5f, 0.25f, 0.125f};
    const float init = 0.25f;  // 非零初值：证明本函数是累加而不是覆盖写

    std::vector<float> got(static_cast<std::size_t>(kHidden), init);
    ExpertSource src = fx.source(4, 1, 0);
    src.begin_token();
    const std::vector<std::uint32_t> u = as_u32(ids);
    src.observe_token(u.data(), static_cast<int>(u.size()));

    CpuExpertWorker cpu;
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::string err;
    CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(),
                               static_cast<int>(ids.size()), x.data(), got.data(), ds, rs, err));
    CHECK(err.empty());
    CHECK(rs.routed == 1 && rs.rerouted == 0);

    const std::vector<float> want = direct_reference(fx.g, fx.ml, ids, w, x.data(), init);
    CHECK(same_bits(got, want, "批量路径 vs 直接按偏移逐专家"));

    // 三个权重不同、三个专家的字节也不同，故「权重取错下标」或「字节串了」都会让某个分量不同。
    CHECK(cpu.stats().jobs == 1 && cpu.stats().experts == 3 && cpu.stats().failed == 0);
    CHECK(ds.pinned.empty());  // 算完不留钉住
}

void test_input_order_does_not_change_output() {
    const Fixture fx("qinfer_dispatch_order.gguf", 4);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{0, 1, 2, 3};
    const std::vector<float> w{0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<float> want = direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f);

    CpuExpertWorker cpu;
    std::vector<float> prev;
    // 六种排列（含原序）：归并顺序若跟着调用方给的顺序走，这些结果就会互不相同。
    const std::vector<std::vector<int>> perms{{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 0, 3, 1},
                                              {1, 3, 0, 2}, {3, 0, 1, 2}, {2, 3, 0, 1}};
    for (const std::vector<int>& p : perms) {
        std::vector<float> pw(p.size());
        for (std::size_t i = 0; i < p.size(); ++i) pw[i] = w[static_cast<std::size_t>(p[i])];
        ExpertSource src = fx.source(8, 1, 0);
        src.begin_token();
        std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);
        LayerDispatchScratch ds;
        RerouteStats rs{};
        std::string err;
        CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, p.data(), pw.data(),
                                   static_cast<int>(p.size()), x.data(), got.data(), ds, rs, err));
        CHECK(same_bits(got, want, "换传参顺序"));
        if (!prev.empty()) CHECK(same_bits(got, prev, "两次排列之间"));
        prev = got;
    }
}

void test_resident_and_transient_in_one_batch() {
    // slots=1：第一个专家进了缓存并被钉住，后面的专家要换它时组内全不可驱逐 -> 走临时缓冲。
    // 特意用**三个**专家：批里有两个都取到同一个临时缓冲，后一个会把前一个覆写掉，故「非常驻的没有
    // 拷出来」这个错会真的改变结果（只用两个专家时最后一次读恰好在用之前，覆写不发生，抓不住）。
    const Fixture fx("qinfer_dispatch_mixed.gguf", 3);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{0, 1, 2};
    const std::vector<float> w{0.5f, 0.3f, 0.2f};

    ExpertSource src = fx.source(1, 1, 0);
    src.begin_token();
    CpuExpertWorker cpu;
    std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::string err;
    CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(), 3, x.data(), got.data(),
                               ds, rs, err));
    CHECK(err.empty());
    CHECK(src.blocked_admissions() == 2);  // 后两个都走了「组内全不可驱逐」
    CHECK(src.residents() == 1);
    CHECK(same_bits(got, direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f), "常驻+临时混合批"));
}

void test_budget_exhausted_goes_transient() {
    const Fixture fx("qinfer_dispatch_budget.gguf", 3);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{2, 3, 0};
    const std::vector<float> w{0.5f, 0.3f, 0.2f};

    ExpertSource src = fx.source(8, 1, /*budget=*/1);  // 一个专家都装不下
    src.begin_token();
    CpuExpertWorker cpu;
    std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::string err;
    CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(), 3, x.data(), got.data(),
                               ds, rs, err));
    CHECK(src.transient_reads() == 3);
    CHECK(src.residents() == 0);
    CHECK(same_bits(got, direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f), "预算用尽"));
}

void test_pinned_slots_are_released() {
    // 全相联（ways == slots）是刻意的：组相联下槽位数不等于「能装几个专家」，几个专家可能落进同一组，
    // 那样 blocked 非零就分不清是「组不够」还是「引用计数没解」。全相联只有一个组，第二批要换出第一批
    // 时组内必定有可驱逐的（除非第一批的引用计数漏解了），故 blocked 的取值就是干净的判据。
    const Fixture fx("qinfer_dispatch_unpin.gguf", 2);
    const std::vector<float> x = make_x();
    CpuExpertWorker cpu;
    ExpertSource src = fx.source(2, /*ways=*/2, 0);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::string err;

    const std::vector<std::vector<int>> rounds{{0, 1}, {2, 3}, {0, 1}};
    for (const std::vector<int>& ids : rounds) {
        const std::vector<float> w(ids.size(), 0.5f);
        src.begin_token();
        std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);
        CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(),
                                   static_cast<int>(ids.size()), x.data(), got.data(), ds, rs,
                                   err));
        CHECK(same_bits(got, direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f), "换槽位轮次"));
    }
    CHECK(src.blocked_admissions() == 0);
    CHECK(src.residents() == 2);
}

void test_reroute_keeps_results() {
    const Fixture fx("qinfer_dispatch_reroute.gguf", 3);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{1, 3, 0};
    const std::vector<float> w{0.5f, 0.3f, 0.2f};

    ExpertSource src = fx.source(8, 1, 0);
    src.begin_token();
    IgpuExpertWorker igpu(/*device_usable=*/false);
    CpuExpertWorker cpu;
    std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::string err;
    CHECK(dispatch_layer_batch(fx.ml, src, igpu, cpu, ids.data(), w.data(), 3, x.data(), got.data(),
                               ds, rs, err));
    CHECK(err.empty());
    CHECK(rs.rerouted == 1 && rs.routed == 0);
    // 不可用的工人不该被调用：它的失败计数必须保持 0（否则观测里会全是假失败）。
    CHECK(igpu.stats().failed == 0);
    CHECK(same_bits(got, direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f), "重派到 CPU"));
}

void test_reduction_order_is_expert_id_ascending() {
    // 归并顺序 = 专家 id 升序（engine §16）。这条用**结构**钉住：两条路径的排序排列都在可读字段里
    // （run_moe_layer 与批量路径各自 scratch 里的 order），直接断言它是「按 id 升序」那个排列。
    // 为什么不用浮点差别来判：在这个合成夹具上实测两种顺序**逐位相同**（值域太规整，三项的不同求和
    // 次序没有产生舍入差），靠位数差会在本夹具上变成空转。真权重上「与 id 升序参考逐位一致」这条
    // 判据在 --model 入口里。
    //
    // scaled 路由器 + 全正输入 -> top-3 是 {3, 2, 1}，排名序与 id 升序**相反**，故「升序」这条断言
    // 不是空转。取 k=3 而不是 2：两项求和是可交换的，两种顺序在结构上就分不开。
    const Fixture fx("qinfer_dispatch_idorder.gguf", 3, /*scaled_router=*/true);
    const std::vector<float> x = make_x_pos();
    const std::size_t hidden = static_cast<std::size_t>(kHidden);

    MoeScratch sc;
    std::vector<float> layer_out(hidden, 0.0f);
    std::vector<int> ids(3, 0);
    std::vector<float> w(3, 0.0f);
    std::string err;
    CHECK(run_moe_layer(fx.g, fx.ml, x.data(), layer_out.data(), sc, err, ids.data(), w.data()));
    CHECK(ids[0] == 3 && ids[1] == 2 && ids[2] == 1);  // 排名序：logits 随专家序号单调增
    CHECK(rank_is_ascending_by_id(ids, sc.order));

    const std::vector<float> ref_id = direct_reference(fx.g, fx.ml, ids, w, x.data(), 0.0f);
    CHECK(same_bits(layer_out, ref_id, "run_moe_layer 与 id 升序参考"));

    ExpertSource src = fx.source(8, 1, 0);
    src.begin_token();
    CpuExpertWorker cpu;
    std::vector<float> got(hidden, 0.0f);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    CHECK(dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(), 3, x.data(), got.data(),
                               ds, rs, err));
    CHECK(rank_is_ascending_by_id(ids, ds.order));
    CHECK(same_bits(got, ref_id, "批量路径与 id 升序参考"));
}

void test_bad_inputs_fail_loudly() {
    const Fixture fx("qinfer_dispatch_bad.gguf", 3);
    const std::vector<float> x = make_x();
    const std::vector<int> ids{0};
    const std::vector<float> w{1.0f};
    ExpertSource src = fx.source(4, 1, 0);
    CpuExpertWorker cpu;
    LayerDispatchScratch ds;
    RerouteStats rs{};
    std::vector<float> got(static_cast<std::size_t>(kHidden), 0.0f);

    std::string err;
    CHECK(!dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(), 0, x.data(), got.data(),
                                ds, rs, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!dispatch_layer_batch(fx.ml, src, cpu, cpu, nullptr, w.data(), 1, x.data(), got.data(),
                                ds, rs, err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!dispatch_layer_batch(fx.ml, src, cpu, cpu, ids.data(), w.data(), 1, nullptr, got.data(),
                                ds, rs, err));
    CHECK(!err.empty());
    // 三个坏用例都没有提交作业，故工人的计数保持 0（观测里不该出现假作业）。
    CHECK(cpu.stats().jobs == 0 && cpu.stats().failed == 0);
}

// ---- 真模型手工入口 ----
//
// 同一层两条路径：`run_moe_layer`（按逐专家偏移逐专家算 + 共享专家）与「批量分派 + add_shared_expert」。
// 两条路径读同一份字节、调同一个 expert_ffn、按同一个 id 升序归并，故必须逐位一致——不一致就说明
// 批量路径的取字节、钉住/拷出、或归并顺序有一处不对。
int manual_model_entry(const char* model, int layer, int k) {
    artifact::GgufFile g;
    std::string open_err;
    if (!g.open(model, open_err)) {
        std::printf("打不开模型：%s（%s）\n", model, open_err.c_str());
        return 1;
    }
    MoeLayer ml;
    std::string err;
    if (!build_moe_layer(g, layer, k, ml, err)) {
        std::printf("构建层 %d 失败：%s\n", layer, err.c_str());
        return 1;
    }
    const std::size_t hidden = static_cast<std::size_t>(ml.hidden);
    std::vector<float> x(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.03125f;
    }

    std::vector<float> ref(hidden, 0.0f);
    std::vector<int> ids(static_cast<std::size_t>(k), 0);
    std::vector<float> wts(static_cast<std::size_t>(k), 0.0f);
    MoeScratch sc1;
    if (!run_moe_layer(g, ml, x.data(), ref.data(), sc1, err, ids.data(), wts.data())) {
        std::printf("参考路径失败：%s\n", err.c_str());
        return 1;
    }

    // 批量路径：缓存容量给足（本用例只比结果，不比命中率），共享专家由 add_shared_expert 另加。
    ExpertSource src(g, "blk." + std::to_string(layer) + ".ffn_", ml.spec, storage::PolicyKind::kLru,
                     4 * static_cast<std::size_t>(k), 8, 0);
    src.begin_token();
    const std::vector<std::uint32_t> u(ids.begin(), ids.end());
    src.observe_token(u.data(), k);

    CpuExpertWorker cpu;
    std::vector<float> got(hidden, 0.0f);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    if (!dispatch_layer_batch(ml, src, cpu, cpu, ids.data(), wts.data(), k, x.data(), got.data(), ds,
                              rs, err)) {
        std::printf("批量路径失败：%s\n", err.c_str());
        return 1;
    }
    MoeScratch sc2;
    if (!add_shared_expert(g, ml, x.data(), got.data(), sc2, err)) {
        std::printf("共享专家失败：%s\n", err.c_str());
        return 1;
    }

    double mag = 0.0;
    bool same = true;
    for (std::size_t i = 0; i < hidden; ++i) {
        mag += std::fabs(static_cast<double>(ref[i]));
        if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(ref[i])) {
            same = false;
        }
    }
    std::printf("层 %d：top-k %d，逐位一致：%s（|参考| 合计 %.6g，共享专家 %s）\n", layer, k,
                same ? "是" : "否", mag, ml.shared.present ? "有" : "无");
    return same ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 0, k = 10;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--k") k = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer, k);
    }
    test_matches_direct_and_accumulates();
    test_input_order_does_not_change_output();
    test_resident_and_transient_in_one_batch();
    test_budget_exhausted_goes_transient();
    test_pinned_slots_are_released();
    test_reroute_keeps_results();
    test_reduction_order_is_expert_id_ascending();
    test_bad_inputs_fail_loudly();
    std::puts("layer_dispatch: 批取、钉住与拷出、重派、以及按专家 id 全序归并都成立");
    return 0;
}