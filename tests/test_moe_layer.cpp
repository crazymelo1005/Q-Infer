// MoE 一层的回归：路由 → 逐专家前馈 → 加权累加，含逐专家偏移的核对。不引第三方框架。
//
// 合成夹具用小几何（hidden 256 / ffn 64 / 4 专家 / top_k 2），块格式与真实档一致：
// gate/up 用 IQ2_S（一块 256 值）、down 用 Q2_0（一块 64 值）、路由器用 BF16。
//
// 五路证据：
//   1. 解析可算：路由器权重全为 bf16 1.0 时各专家 logits 相同 -> softmax 均匀 -> top-2 的 id 必为
//      0、1 且权重各 0.5；再让四个专家的矩阵完全相同，则 MoE 输出恰等于单个专家的前馈输出。
//   2. 逐专家偏移：让第 e 个专家的 gate 尺度随 e 变化，参考侧**直接按 e 单独生成**该专家的矩阵
//      （不走偏移算术），核对 MoE 输出 = Σ w_e · 单专家前馈(e)。偏移算错就会读到别的专家。
//   3. 不可用必须失败：IQ1_M 的层、缺路由器、top_k 超过专家数、维度不自洽。
//   4. 路由结果回填与直接调用 router 模块的结果一致。
//   5. 真模型手工入口：对部署那份的可用层跑通，并与测试侧「显式按偏移读三矩阵 + expert_ffn + 加权」核对。
#include "experts/moe_layer.hpp"

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
constexpr int kTopK = 2;
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

// ---- 块数据构造 ----

void push_iq2s_block(std::vector<std::uint8_t>& out, std::uint16_t d_bits, std::uint8_t scale_lo) {
    out.push_back(static_cast<std::uint8_t>(d_bits & 0xFF));
    out.push_back(static_cast<std::uint8_t>(d_bits >> 8));
    for (int i = 0; i < 64; ++i) out.push_back(0);  // qs：格点索引 0
    for (int i = 0; i < 8; ++i) out.push_back(0);   // qh
    for (int i = 0; i < 8; ++i) out.push_back(i == 0 ? scale_lo : 0);  // scales
}

// [cols, rows, experts] 的 IQ2_S 数据。vary=false 时所有专家的尺度都是 0（四个专家逐字节相同）；
// vary=true 时第 first+e 个专家的低半字节尺度取 e，用来把专家区分开。
std::vector<std::uint8_t> iq2s_tensor(std::uint64_t cols, std::uint64_t rows, std::uint64_t experts,
                                      std::uint64_t first, bool vary) {
    std::vector<std::uint8_t> data;
    const std::uint64_t nb = cols / 256;
    for (std::uint64_t e = 0; e < experts; ++e) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t b = 0; b < nb; ++b) {
                const std::uint8_t scale = vary ? static_cast<std::uint8_t>(first + e) : 0;
                push_iq2s_block(data, 0x3C00, scale);
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

std::vector<std::uint8_t> q20_tensor(std::uint64_t cols, std::uint64_t rows, std::uint64_t experts) {
    std::vector<std::uint8_t> data;
    const std::uint64_t nb = cols / 64;
    for (std::uint64_t e = 0; e < experts; ++e) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t b = 0; b < nb; ++b) push_q20_block(data, 0x3C00, 0xAA);
        }
    }
    return data;
}

std::vector<std::uint8_t> router_tensor(std::uint64_t cols, std::uint64_t experts) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(cols * experts * 2));
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        data[i] = 0x80;      // bf16 1.0
        data[i + 1] = 0x3F;
    }
    return data;
}

// 各专家权重不同的路由器：第 e 个专家的每个权重都是 bf16(1.0 + 0.25·e)。配全正输入时
// logits 随 e 单调增，故 top-2 必为 {3, 2}，且两个权重不同——用来抓「累加时权重取错下标」。
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

std::vector<TensorSpec> layer_tensors(const std::string& prefix, bool vary_gate,
                                      std::uint64_t first_expert) {
    std::vector<TensorSpec> ts;
    ts.push_back({prefix + "ffn_gate_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, first_expert, vary_gate)});
    ts.push_back({prefix + "ffn_up_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, 0, false)});
    ts.push_back({prefix + "ffn_down_exps.weight", {kFfn, kHidden, kExpert}, 42,
                  q20_tensor(kFfn, kHidden, kExpert)});
    ts.push_back({prefix + "ffn_gate_inp.weight", {kHidden, kExpert}, 30,
                  router_tensor(kHidden, kExpert)});
    return ts;
}

// ---- 参考侧：直接按构造规则算单个专家的前馈（不走偏移算术） ----

float dq_iq2s(const std::vector<std::uint8_t>& t, std::size_t row, std::size_t col) {
    const std::size_t at = row * 82;  // 每行 1 块
    float tmp[256];
    dequant_iq2s_block(t.data() + at, tmp);
    return tmp[col];
}

float dq_q20(const std::vector<std::uint8_t>& t, std::size_t row, std::size_t col) {
    const std::size_t at = row * 18;  // 每行 1 块（cols=64）
    float tmp[64];
    dequant_q2_0_block(t.data() + at, tmp);
    return tmp[col];
}

// 单专家前馈的双精度参考：激活量化规则与主实现一致，但点乘在 double 里逐元素做。
std::vector<double> ref_expert_ffn(const std::vector<std::uint8_t>& g,
                                   const std::vector<std::uint8_t>& u,
                                   const std::vector<std::uint8_t>& d, const float* x) {
    Q8kBlock xb{};
    q8k_quantize_row(x, &xb, 1);
    float xq[256];
    q8k_dequant_block(xb, xq);

    std::vector<double> gate(kFfn, 0.0), up(kFfn, 0.0);
    for (std::size_t r = 0; r < kFfn; ++r) {
        for (std::size_t j = 0; j < kHidden; ++j) {
            gate[r] += static_cast<double>(dq_iq2s(g, r, j)) * static_cast<double>(xq[j]);
            up[r] += static_cast<double>(dq_iq2s(u, r, j)) * static_cast<double>(xq[j]);
        }
    }
    float h[kFfn];
    for (std::size_t j = 0; j < kFfn; ++j) {
        const double gv = gate[j];
        h[j] = static_cast<float>((gv / (1.0 + std::exp(-gv))) * up[j]);
    }
    Q80Block hb[2];
    q8_0_quantize_row(h, hb, 2);
    float hq[kFfn];
    for (int b = 0; b < 2; ++b) q8_0_dequant_block(hb[b], hq + b * kQ80BlockElems);

    std::vector<double> out(kHidden, 0.0);
    for (std::size_t i = 0; i < kHidden; ++i) {
        for (std::size_t j = 0; j < kFfn; ++j) {
            out[i] += static_cast<double>(dq_q20(d, i, j)) * static_cast<double>(hq[j]);
        }
    }
    return out;
}

int near_vec(const std::vector<float>& got, const std::vector<double>& want, const char* what) {
    double mag = 0.0;
    for (double v : want) mag += std::fabs(v);
    if (mag == 0.0) mag = 1.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double rel = std::fabs(static_cast<double>(got[i]) - want[i]) / mag;
        if (!(rel < 1e-5)) {
            std::printf("FAIL %s[%zu]: got %.9g want %.9g rel %.3g\n", what, i,
                        static_cast<double>(got[i]), want[i], rel);
            return 1;
        }
    }
    return 0;
}

std::vector<float> make_x() {
    std::vector<float> x(kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.03125f;
    }
    return x;
}

void test_identical_experts() {
    // 四个专家的矩阵完全相同、路由器权重全 1.0 -> logits 相同 -> top-2 = {0,1}，权重各 0.5
    // -> MoE 输出恰等于单个专家的前馈输出。
    const auto ts = layer_tensors("blk.0.", false, 0);
    const std::filesystem::path p = write_gguf("qinfer_moe_same.gguf", ts);
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    MoeLayer layer;
    CHECK(build_moe_layer(g, 0, kTopK, layer, err));
    CHECK(layer.hidden == kHidden && layer.n_expert == kExpert);

    const std::vector<float> x = make_x();
    MoeScratch s;
    std::vector<float> out(kHidden, 0.0f);
    int ids[kTopK];
    float weights[kTopK];
    CHECK(run_moe_layer(g, layer, x.data(), out.data(), s, err, ids, weights));
    CHECK(ids[0] == 0 && ids[1] == 1);            // 同值取小索引
    CHECK(std::fabs(weights[0] - 0.5f) < 1e-6f && std::fabs(weights[1] - 0.5f) < 1e-6f);

    const std::vector<double> ref =
        ref_expert_ffn(ts[0].data, ts[1].data, ts[2].data, x.data());
    CHECK(near_vec(out, ref, "identical experts") == 0);
}

void test_expert_offsets() {
    // gate 的尺度随专家变化：参考侧按 e 单独生成该专家的矩阵，核对加权和。
    const auto ts = layer_tensors("blk.0.", true, 0);
    const std::filesystem::path p = write_gguf("qinfer_moe_off.gguf", ts);
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    MoeLayer layer;
    CHECK(build_moe_layer(g, 0, kTopK, layer, err));

    const std::vector<float> x = make_x();
    MoeScratch s;
    std::vector<float> out(kHidden, 0.0f);
    int ids[kTopK];
    float weights[kTopK];
    CHECK(run_moe_layer(g, layer, x.data(), out.data(), s, err, ids, weights));
    CHECK(ids[0] == 0 && ids[1] == 1);

    // 参考：专家 0 与 1 的 gate 各自按构造规则单独生成（first=0 / first=1），不走偏移算术。
    const auto g0 = iq2s_tensor(kHidden, kFfn, 1, 0, true);
    const auto g1 = iq2s_tensor(kHidden, kFfn, 1, 1, true);
    const auto u0 = iq2s_tensor(kHidden, kFfn, 1, 0, false);
    const auto d0 = q20_tensor(kFfn, kHidden, 1);
    const std::vector<double> r0 = ref_expert_ffn(g0, u0, d0, x.data());
    const std::vector<double> r1 = ref_expert_ffn(g1, u0, d0, x.data());
    std::vector<double> want(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) {
        want[i] = static_cast<double>(weights[0]) * r0[i] + static_cast<double>(weights[1]) * r1[i];
    }
    CHECK(near_vec(out, want, "expert offsets") == 0);

    // 两个专家确实不同（否则上面这条等于自证）：r0 与 r1 必须有可观差异。
    double diff = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < kHidden; ++i) {
        diff += std::fabs(r0[i] - r1[i]);
        mag += std::fabs(r0[i]);
    }
    CHECK(diff > 0.01 * mag);
}

void test_nonuniform_router_weights() {
    // 专家互不相同 + 各专家路由权重不同 + 输入全正：logits 随专家序号单调增，故 top-2 必为 {3, 2}，
    // 且两个权重不相等。这条专抓「累加时把权重下标写死」——权重相同时那种错误看不出来。
    std::vector<TensorSpec> ts;
    ts.push_back({"blk.0.ffn_gate_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, 0, true)});
    ts.push_back({"blk.0.ffn_up_exps.weight", {kHidden, kFfn, kExpert}, 22,
                  iq2s_tensor(kHidden, kFfn, kExpert, 0, false)});
    ts.push_back({"blk.0.ffn_down_exps.weight", {kFfn, kHidden, kExpert}, 42,
                  q20_tensor(kFfn, kHidden, kExpert)});
    ts.push_back({"blk.0.ffn_gate_inp.weight", {kHidden, kExpert}, 30,
                  router_tensor_scaled(kHidden, kExpert)});
    const std::filesystem::path p = write_gguf("qinfer_moe_scaled.gguf", ts);
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    MoeLayer layer;
    CHECK(build_moe_layer(g, 0, kTopK, layer, err));

    std::vector<float> x(kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.03125f * static_cast<float>(1 + static_cast<int>(i % 7));  // 全正
    }
    MoeScratch s;
    std::vector<float> out(kHidden, 0.0f);
    int ids[kTopK];
    float w[kTopK];
    CHECK(run_moe_layer(g, layer, x.data(), out.data(), s, err, ids, w));
    CHECK(ids[0] == 3 && ids[1] == 2);
    CHECK(std::fabs(w[0] - w[1]) > 1e-3f);          // 两个权重必须不同
    CHECK(w[0] > w[1]);                              // 且按降序

    // 参考：专家 3 与 2 的 gate 各自按 e 单独生成，权重取自 router 模块的直接调用。
    RouterScratch rs;
    std::vector<std::uint16_t> rw(static_cast<std::size_t>(kHidden * kExpert));
    const auto* rt = g.find("blk.0.ffn_gate_inp.weight");
    CHECK(g.read_at(rt->offset, reinterpret_cast<std::uint8_t*>(rw.data()), rw.size() * 2));
    int rids[kTopK];
    float rweights[kTopK];
    CHECK(route(rw.data(), static_cast<int>(kExpert), static_cast<int>(kHidden), kTopK, x.data(),
                rs, rids, rweights, err));
    CHECK(rids[0] == 3 && rids[1] == 2);
    CHECK(rweights[0] == w[0] && rweights[1] == w[1]);

    const auto g3 = iq2s_tensor(kHidden, kFfn, 1, 3, true);
    const auto g2 = iq2s_tensor(kHidden, kFfn, 1, 2, true);
    const auto u0 = iq2s_tensor(kHidden, kFfn, 1, 0, false);
    const auto d0 = q20_tensor(kFfn, kHidden, 1);
    const std::vector<double> r3 = ref_expert_ffn(g3, u0, d0, x.data());
    const std::vector<double> r2 = ref_expert_ffn(g2, u0, d0, x.data());
    std::vector<double> want(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) {
        want[i] = static_cast<double>(w[0]) * r3[i] + static_cast<double>(w[1]) * r2[i];
    }
    CHECK(near_vec(out, want, "nonuniform weights") == 0);
}

void test_failures() {
    std::string err;
    // top_k 超过专家数。
    {
        const auto ts = layer_tensors("blk.0.", false, 0);
        const std::filesystem::path p = write_gguf("qinfer_moe_k.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        MoeLayer layer;
        err.clear();
        CHECK(!build_moe_layer(g, 0, static_cast<int>(kExpert) + 1, layer, err));
        CHECK(!err.empty());
    }
    // 缺路由器。
    {
        auto ts = layer_tensors("blk.0.", false, 0);
        ts.pop_back();
        const std::filesystem::path p = write_gguf("qinfer_moe_norouter.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        MoeLayer layer;
        err.clear();
        CHECK(!build_moe_layer(g, 0, kTopK, layer, err));
        CHECK(!err.empty());
    }
    // 不可用的档（IQ3_S）：把 gate/up 的类型码改成 21。
    {
        auto ts = layer_tensors("blk.0.", false, 0);
        ts[0].type = 21;
        ts[1].type = 21;
        const std::filesystem::path p = write_gguf("qinfer_moe_nokernel.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        MoeLayer layer;
        err.clear();
        CHECK(!build_moe_layer(g, 0, kTopK, layer, err));
        CHECK(err.find("IQ3_S") != std::string::npos);
    }
}

// ---- 真模型手工入口 ----

int manual_model_entry(const std::string& path, int layer, int k) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(path, err)) {
        std::printf("打不开：%s\n", err.c_str());
        return 1;
    }
    MoeLayer ml;
    if (!build_moe_layer(g, layer, k, ml, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::vector<float> x(static_cast<std::size_t>(ml.hidden));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.03125f;
    }
    MoeScratch s;
    std::vector<float> out(static_cast<std::size_t>(ml.hidden), 0.0f);
    std::vector<int> ids(static_cast<std::size_t>(k));
    std::vector<float> w(static_cast<std::size_t>(k));
    if (!run_moe_layer(g, ml, x.data(), out.data(), s, err, ids.data(), w.data())) {
        std::printf("跑失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("层 %d：hidden=%llu n_expert=%llu top_k=%d gate=%s up=%s down=%s\n", layer,
                static_cast<unsigned long long>(ml.hidden),
                static_cast<unsigned long long>(ml.n_expert), k,
                format_name(ml.spec.gate.format), format_name(ml.spec.up.format),
                format_name(ml.spec.down.format));
    std::printf("  ids    ");
    for (int i = 0; i < k; ++i) std::printf(" %d", ids[i]);
    std::printf("\n  weights");
    for (int i = 0; i < k; ++i) std::printf(" %.6f", w[i]);
    double osum = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) osum += out[i];
    std::printf("\n  out 前 3 值 %.6g %.6g %.6g；sum %.6g\n", out[0], out[1], out[2], osum);

    // 独立参考：测试侧按「张力起点 + e × rows × row_bytes」自己读三矩阵，再走 expert_ffn 与加权。
    std::vector<std::uint8_t> bg(static_cast<std::size_t>(ml.gate_expert_bytes));
    std::vector<std::uint8_t> bu(static_cast<std::size_t>(ml.up_expert_bytes));
    std::vector<std::uint8_t> bd(static_cast<std::size_t>(ml.down_expert_bytes));
    std::vector<float> ref(static_cast<std::size_t>(ml.hidden), 0.0f);
    FfnScratch fs;
    std::vector<float> one(static_cast<std::size_t>(ml.hidden), 0.0f);
    for (int i = 0; i < k; ++i) {
        const std::uint64_t e = static_cast<std::uint64_t>(ids[static_cast<std::size_t>(i)]);
        if (!g.read_at(ml.gate->offset + e * ml.gate_expert_bytes, bg.data(), bg.size()) ||
            !g.read_at(ml.up->offset + e * ml.up_expert_bytes, bu.data(), bu.size()) ||
            !g.read_at(ml.down->offset + e * ml.down_expert_bytes, bd.data(), bd.size())) {
            std::printf("参考侧读矩阵失败\n");
            return 1;
        }
        ExpertWeights ew{bg.data(), bu.data(), bd.data()};
        if (!expert_ffn(ml.spec, ew, x.data(), fs, one.data(), err)) {
            std::printf("参考侧前馈失败：%s\n", err.c_str());
            return 1;
        }
        for (std::size_t j = 0; j < ref.size(); ++j) {
            ref[j] += w[static_cast<std::size_t>(i)] * one[j];
        }
    }
    double diff = 0.0, mag = 0.0;
    for (std::size_t j = 0; j < ref.size(); ++j) {
        diff += std::fabs(static_cast<double>(out[j] - ref[j]));
        mag += std::fabs(static_cast<double>(ref[j]));
    }
    const bool ok = diff <= 1e-5 * (mag > 0.0 ? mag : 1.0);
    std::printf("与「显式按偏移读 + 逐专家加权」的参考比对：%s（|参考| 合计 %.6g）\n",
                ok ? "一致" : "不一致", mag);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--model") {
        int layer = 0, k = 10;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--k") k = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer, k);
    }
    test_identical_experts();
    test_expert_offsets();
    test_nonuniform_router_weights();
    test_failures();
    std::puts("moe_layer: routing, per-expert offsets and weighted accumulation hold");
    return 0;
}