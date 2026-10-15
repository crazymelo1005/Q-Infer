// 一层的组合（`block_layer`）的回归。不引第三方框架。
//
// 合成夹具：n_embd 256（MoE 的 hidden 与它同宽）、hc 2、hc_lr 3、MoE ffn 64 / 4 专家 / top-2
// （gate/up 用 IQ2_S、down 用 Q2_0、路由器 BF16）、GDN 走 q/k/v 各 1 头、头宽 32（卷积维 96），
// QSA 走 4 头 / kv 2 头、头宽 32、idx_dim 4 / idx 头 2、n_rot 4。两组 hc_* 的 norm 是 F32、down/up/inject
// 是 BF16（[S-53]），且两组的 norm 取值明显不同（一组 ~1、一组 ~2）。
//
// 钉子：组装顺序（pre = hc_attn_*→注意力、post = hc_ffn_*→MoE）；inject 必须真的读进来；单 token 的两段
// 组合与「按子模块显式拼一遍」的独立参考逐位一致（GDN 与 QSA 都核）；接错必须报错。真权重走 --model。
#include "dense/block_layer.hpp"

#include "dense/gated_residual.hpp"
#include "dense/gdn_layer.hpp"
#include "dense/model_head.hpp"
#include "dense/qsa_layer.hpp"
#include "dense/rope.hpp"
#include "experts/moe_layer.hpp"

#include "check.hpp"
#include "kernels/bf16.hpp"
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
using namespace qinfer::dense;
using namespace qinfer::experts;

namespace {

constexpr std::uint64_t kN = 256;
constexpr std::uint64_t kHc = 2;
constexpr std::uint64_t kLr = 3;
constexpr std::uint64_t kFfn = 64;
constexpr std::uint64_t kExpert = 4;
constexpr int kTopK = 2;
constexpr std::uint64_t kGdnHd = 32;  // q/k/v 各 1 头 → 卷积维 2·32 + 32 = 96
constexpr std::uint64_t kQk = kGdnHd, kVd = kGdnHd, kCd = kQk + kQk + kVd;
constexpr std::uint64_t kDc = 4;
constexpr std::uint64_t kQsaHd = 32, kQsaNh = 4, kQsaNkv = 2, kIdx = 4, kIdxH = 2, kNrot = 4;
constexpr std::uint64_t kIdxTopK = 2048, kIdxBlock = 4;
constexpr float kEps = 1e-6f;
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
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    const std::string align = "general.alignment";
    u64(align.size());
    b.insert(b.end(), align.begin(), align.end());
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        u64(t.name.size());
        b.insert(b.end(), t.name.begin(), t.name.end());
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

// Q8_0 的一批行（cols 是 32 的倍数 → 每行 cols/32 个块）。d 由调用方给（默认 0.125）；码取循环，
// seed 让不同投影/不同专家的取值不同。
std::vector<std::uint8_t> q8_0_rows(std::uint64_t rows, std::uint64_t cols, int seed,
                                    std::uint16_t d = 0x3000u) {
    std::vector<std::uint8_t> out;
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t b = 0; b < cols / 32; ++b) {
            out.push_back(static_cast<std::uint8_t>(d & 0xFF));
            out.push_back(static_cast<std::uint8_t>(d >> 8));
            for (std::uint64_t c = 0; c < 32; ++c) {
                out.push_back(static_cast<std::uint8_t>(((r + b * 32 + c + seed) % 7) - 3));
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> f32_vec(const std::vector<float>& v) {
    std::vector<std::uint8_t> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

std::vector<std::uint8_t> f32_lin(std::uint64_t n, float base) {
    std::vector<float> f(n);
    for (std::uint64_t i = 0; i < n; ++i) f[static_cast<std::size_t>(i)] = base + 0.01f * (float) i;
    return f32_vec(f);
}

// bf16 一组对称取值 amp·((i%7)-3)/3：避免净和爆掉，也让 inject 的门不饱和。
std::vector<std::uint8_t> bf16_sym(std::uint64_t n, float amp) {
    std::vector<std::uint8_t> out(n * 2);
    for (std::uint64_t i = 0; i < n; ++i) {
        const float v = amp * (static_cast<float>(static_cast<int>(i % 7) - 3) / 3.0f);
        const std::uint16_t bits = kernels::f32_to_bf16_bits(v);
        out[static_cast<std::size_t>(2 * i)] = static_cast<std::uint8_t>(bits & 0xFF);
        out[static_cast<std::size_t>(2 * i + 1)] = static_cast<std::uint8_t>(bits >> 8);
    }
    return out;
}

std::vector<std::uint8_t> router_tensor(std::uint64_t cols, std::uint64_t experts) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(cols * experts * 2));
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        data[i] = 0x80;  // bf16 1.0
        data[i + 1] = 0x3F;
    }
    return data;
}

// 一层的 MoE 四张张力（hidden = kN）。三个投影都用 Q8_0：尺度好控、专家之间差别明显，也不会像
// 「全 +1 的 Q2_0 下投影」那样退化成 rank-1（输出是 h 的和，会相消到 0，让「顺序」不可观测）。
// d 取 0.03125 让 MoE 的输出与注意力同一个量级。
std::vector<TensorSpec> moe_tensors(const std::string& b) {
    const std::uint16_t d = 0x2800u;
    std::vector<TensorSpec> ts;
    ts.push_back({b + "ffn_gate_exps.weight", {kN, kFfn, kExpert}, 8,
                  q8_0_rows(kFfn * kExpert, kN, 1, d)});
    ts.push_back({b + "ffn_up_exps.weight", {kN, kFfn, kExpert}, 8,
                  q8_0_rows(kFfn * kExpert, kN, 2, d)});
    ts.push_back({b + "ffn_down_exps.weight", {kFfn, kN, kExpert}, 8,
                  q8_0_rows(kN * kExpert, kFfn, 3, d)});
    ts.push_back({b + "ffn_gate_inp.weight", {kN, kExpert}, 30, router_tensor(kN, kExpert)});
    return ts;
}

// 一组 hc_*（norm F32 一维、down/up/inject BF16 二维）。norm 基值不同才能把「两段用了同一组」暴露出来。
std::vector<TensorSpec> gr_group(const std::string& p, float norm_base, float d_amp, float u_amp,
                                 float inj_amp) {
    std::vector<TensorSpec> ts;
    ts.push_back({p + "norm.weight", {kN * kHc}, 0, f32_lin(kN * kHc, norm_base)});
    ts.push_back({p + "down.weight", {kN * kHc, kLr}, 30, bf16_sym(kN * kHc * kLr, d_amp)});
    ts.push_back({p + "up.weight", {kLr, kN * kHc}, 30, bf16_sym(kLr * kN * kHc, u_amp)});
    ts.push_back({p + "inject.weight", {kN * kHc, kHc}, 30, bf16_sym(kN * kHc * kHc, inj_amp)});
    return ts;
}

std::vector<TensorSpec> gdn_attn(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back({b + "attn_qkv.weight", {kN, kCd}, 8, q8_0_rows(kCd, kN, 1)});
    ts.push_back({b + "attn_gate.weight", {kN, kVd}, 8, q8_0_rows(kVd, kN, 2)});
    ts.push_back({b + "ssm_out.weight", {kVd, kN}, 8, q8_0_rows(kN, kVd, 3)});
    ts.push_back({b + "ssm_conv1d.weight", {kDc, kCd}, 0,
                  f32_vec(std::vector<float>(static_cast<std::size_t>(kDc * kCd), 0.25f))});
    ts.push_back({b + "ssm_alpha.weight", {kN, 1}, 30, bf16_sym(kN, 1.0f)});
    ts.push_back({b + "ssm_beta.weight", {kN, 1}, 30, bf16_sym(kN, 0.9f)});
    ts.push_back({b + "ssm_a", {1}, 0, f32_vec(std::vector<float>{-0.5f})});
    ts.push_back({b + "ssm_dt.bias", {1}, 0, f32_vec(std::vector<float>{0.5f})});
    ts.push_back({b + "ssm_norm.weight", {kGdnHd}, 0, f32_vec(std::vector<float>(kGdnHd, 1.0f))});
    return ts;
}

std::vector<TensorSpec> qsa_attn(const std::string& b) {
    std::vector<TensorSpec> ts;
    ts.push_back(
        {b + "attn_q.weight", {kN, 2 * kQsaNh * kQsaHd}, 8, q8_0_rows(2 * kQsaNh * kQsaHd, kN, 1)});
    ts.push_back(
        {b + "attn_k.weight", {kN, kQsaNkv * kQsaHd}, 8, q8_0_rows(kQsaNkv * kQsaHd, kN, 2)});
    ts.push_back(
        {b + "attn_v.weight", {kN, kQsaNkv * kQsaHd}, 8, q8_0_rows(kQsaNkv * kQsaHd, kN, 3)});
    ts.push_back({b + "attn_output.weight", {kQsaNh * kQsaHd, kN}, 8,
                  q8_0_rows(kN, kQsaNh * kQsaHd, 4)});
    ts.push_back({b + "attn_q_norm.weight", {kQsaHd}, 0, f32_vec(std::vector<float>(kQsaHd, 1.0f))});
    ts.push_back({b + "attn_k_norm.weight", {kQsaHd}, 0, f32_vec(std::vector<float>(kQsaHd, 1.0f))});
    ts.push_back({b + "indexer.q_proj.weight", {kN, kIdxH * kIdx}, 30,
                  bf16_sym(kN * kIdxH * kIdx, 0.1f)});
    ts.push_back({b + "indexer.k_proj.weight", {kN, kIdx}, 30, bf16_sym(kN * kIdx, 0.1f)});
    ts.push_back({b + "indexer.q_norm.weight", {kIdx}, 0, f32_vec(std::vector<float>(kIdx, 1.0f))});
    ts.push_back({b + "indexer.k_norm.weight", {kIdx}, 0, f32_vec(std::vector<float>(kIdx, 1.0f))});
    return ts;
}

void append(std::vector<TensorSpec>& base, std::vector<TensorSpec> more) {
    for (auto& t : more) base.push_back(std::move(t));
}

// 两组的 norm 基值取 1 / 2：两段确实不同，且各自的 RMS 尺度也就不同。
std::vector<TensorSpec> block_tensors(bool gdn) {
    std::vector<TensorSpec> ts = gr_group("blk.0.hc_attn_", 1.0f, 0.05f, 0.5f, 0.4f);
    append(ts, gr_group("blk.0.hc_ffn_", 2.0f, 0.05f, 0.5f, -0.4f));
    append(ts, gdn ? gdn_attn("blk.0.") : qsa_attn("blk.0."));
    append(ts, moe_tensors("blk.0."));
    return ts;
}

std::vector<float> make_R() {
    std::vector<float> R(kN * kHc);
    for (std::size_t i = 0; i < R.size(); ++i) {
        R[i] = 0.5f * (static_cast<float>(static_cast<int>(i % 13) - 6) / 6.0f);
    }
    return R;
}

struct GrLoaded {
    std::vector<float> an, fn;
    std::vector<std::uint16_t> ad, au, ai, fd, fu, fi;
    GrWeights aw, fw;
};

void load_gr(const artifact::GgufFile& g, const BlockLayer& bl, GrLoaded& L) {
    auto rd_f32 = [&](const artifact::GgufTensorInfo* t, std::vector<float>& v) {
        v.resize(static_cast<std::size_t>(t->elements()));
        CHECK(g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(v.data()), v.size() * 4));
    };
    auto rd_bf = [&](const artifact::GgufTensorInfo* t, std::vector<std::uint16_t>& v) {
        v.resize(static_cast<std::size_t>(t->elements()));
        CHECK(g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(v.data()), v.size() * 2));
    };
    rd_f32(bl.a_norm, L.an);
    rd_bf(bl.a_down, L.ad);
    rd_bf(bl.a_up, L.au);
    rd_bf(bl.a_inject, L.ai);
    rd_f32(bl.f_norm, L.fn);
    rd_bf(bl.f_down, L.fd);
    rd_bf(bl.f_up, L.fu);
    rd_bf(bl.f_inject, L.fi);
    L.aw = GrWeights{GrTensor{L.an.data(), GrPrecision::kF32},
                     GrTensor{L.ad.data(), GrPrecision::kBf16},
                     GrTensor{L.au.data(), GrPrecision::kBf16},
                     GrTensor{L.ai.data(), GrPrecision::kBf16}};
    L.fw = GrWeights{GrTensor{L.fn.data(), GrPrecision::kF32},
                     GrTensor{L.fd.data(), GrPrecision::kBf16},
                     GrTensor{L.fu.data(), GrPrecision::kBf16},
                     GrTensor{L.fi.data(), GrPrecision::kBf16}};
}

void configure_qsa(QsaScratch& s, const QsaLayer& l) {
    s.kv.page_size = 4;
    s.kv.n_head_kv = l.sh.n_head_kv;
    s.kv.head_dim = l.sh.head_dim;
    s.kv.n_pages = 4;
    s.kv.reset();
    s.kv.table.resize(4);
    for (std::uint64_t p = 0; p < 4; ++p) s.kv.table[p] = static_cast<std::int32_t>(p);
    std::string err;
    CHECK(build_rope_table(l.n_rot, 1.0e7, 64, s.cos_tab, s.sin_tab, err));
    CHECK(err.empty());
}

// 两段组合：Rin → R1 → Rout。开关用来按错的读法算。每段 = gr_read(权重) → mixer → gr_write；
// mixer 由 use_moe 决定（pre 段是注意力、post 段是 MoE）。
void ref_compose(const artifact::GgufFile& g, const BlockLayer& bl, const GrWeights& aw,
                 const GrWeights& fw, const float* Rin, float* Rout, bool ffn_first,
                 bool same_group, bool drop_inject) {
    GrWeights a = aw, f = fw;
    if (drop_inject) {
        a.w_inject = GrTensor{};
        f.w_inject = GrTensor{};
    }
    if (same_group) f = a;
    const std::uint64_t N = bl.n_embd, hc = bl.gr.hc;
    auto segment = [&](const GrWeights& w, bool use_moe, const float* R, float* R_out) {
        GrScratch gr;
        std::vector<float> mixed(static_cast<std::size_t>(N), 0.0f);
        std::vector<float> inject(static_cast<std::size_t>(hc), 0.0f);
        std::vector<float> bo(static_cast<std::size_t>(N), 0.0f);
        std::string err;
        CHECK(gr_read(bl.gr, w, R, kEps, mixed.data(), inject.data(), gr, err));
        if (use_moe) {
            MoeScratch s;
            CHECK(run_moe_layer(g, bl.moe, mixed.data(), bo.data(), s, err));
        } else if (bl.kind == LayerKind::kGdn) {
            GdnScratch s;
            CHECK(run_gdn_layer(g, bl.gdn, mixed.data(), bo.data(), s, err));
        } else {
            QsaScratch s;
            configure_qsa(s, bl.qsa);
            CHECK(run_qsa_layer(g, bl.qsa, mixed.data(), bo.data(), s, err));
        }
        CHECK(gr_write(bl.gr, R, bo.data(), inject.data(), R_out, err));
    };
    std::vector<float> R1(static_cast<std::size_t>(N * hc), 0.0f);
    if (!ffn_first) {
        segment(a, /*use_moe=*/false, Rin, R1.data());
        segment(f, /*use_moe=*/true, R1.data(), Rout);
    } else {
        segment(f, /*use_moe=*/true, Rin, R1.data());
        segment(a, /*use_moe=*/false, R1.data(), Rout);
    }
}

bool same_bits(const std::vector<float>& a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
    }
    return true;
}

double rel_diff(const std::vector<float>& right, const std::vector<float>& wrong) {
    double diff = 0.0, mag = 0.0;
    for (std::size_t i = 0; i < right.size(); ++i) {
        diff += std::fabs(static_cast<double>(right[i]) - static_cast<double>(wrong[i]));
        mag += std::fabs(static_cast<double>(right[i]));
    }
    return mag > 0.0 ? diff / mag : (diff > 0.0 ? 999.0 : 0.0);
}

void test_gdn_block() {
    const std::filesystem::path p = write_gguf("qinfer_block_gdn.gguf", block_tensors(true));
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    BlockLayer bl;
    CHECK(build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock, bl,
                            err));
    CHECK(err.empty());
    CHECK(bl.kind == LayerKind::kGdn);
    CHECK(bl.gdn.pre.conv_dim() == kCd && bl.gdn.pre.v_dim() == kVd);

    GrLoaded L;
    load_gr(g, bl, L);
    const std::vector<float> R0 = make_R();
    std::vector<float> R = R0;
    BlockScratch s;
    CHECK(run_block_layer(g, bl, R.data(), s, err));
    CHECK(err.empty());
    const std::vector<float>& block_out = R;
    CHECK(block_out != R0);  // 两段 gr_write 就地改写过

    // 参考：两段组合与块输出逐位一致。
    std::vector<float> R2 = R0;
    std::vector<float> ref(kN * kHc, 0.0f);
    ref_compose(g, bl, L.aw, L.fw, R2.data(), ref.data(), false, false, false);
    if (!same_bits(ref, block_out.data(), block_out.size())) {
        std::printf("FAIL GDN 两段组合与参考不一致\n");
        CHECK(false);
    }

    // 错的读法必须可观测。
    std::vector<float> wrong(kN * kHc, 0.0f);
    std::vector<float> Rt = R0;
    ref_compose(g, bl, L.aw, L.fw, Rt.data(), wrong.data(), /*ffn_first=*/true, false, false);
    const double d_order = rel_diff(ref, wrong);
    Rt = R0;
    ref_compose(g, bl, L.aw, L.fw, Rt.data(), wrong.data(), false, /*same_group=*/true, false);
    const double d_same = rel_diff(ref, wrong);
    Rt = R0;
    ref_compose(g, bl, L.aw, L.fw, Rt.data(), wrong.data(), false, false, /*drop_inject=*/true);
    const double d_noinj = rel_diff(ref, wrong);
    std::printf("  GDN：对调两段 %.4f、两段共用一组 %.4f、inject 缺席 %.4f\n", d_order, d_same,
                d_noinj);
    CHECK(d_order > 0.05);
    CHECK(d_same > 0.05);
    CHECK(d_noinj > 0.05);

    bool finite = true, nonzero = false;
    for (float v : block_out) {
        if (!std::isfinite(v)) finite = false;
        if (v != 0.0f) nonzero = true;
    }
    CHECK(finite && nonzero);
}

void test_qsa_block() {
    const std::filesystem::path p = write_gguf("qinfer_block_qsa.gguf", block_tensors(false));
    artifact::GgufFile g;
    std::string err;
    CHECK(g.open(p.string(), err));
    BlockLayer bl;
    CHECK(build_block_layer(g, 0, kN, kHc, kLr, kTopK, kQsaHd, kNrot, kIdx, kIdxTopK, kIdxBlock, bl,
                            err));
    CHECK(bl.kind == LayerKind::kQsa);

    // 单 token：QSA 分支的接线与独立参考逐位一致。
    const std::vector<float> R0 = make_R();
    {
        std::vector<float> R = R0;
        BlockScratch s;
        configure_qsa(s.qsa, bl.qsa);
        CHECK(run_block_layer(g, bl, R.data(), s, err));
        const std::vector<float>& one = R;
        GrLoaded L;
        load_gr(g, bl, L);
        std::vector<float> R2 = R0;
        std::vector<float> ref(kN * kHc, 0.0f);
        ref_compose(g, bl, L.aw, L.fw, R2.data(), ref.data(), false, false, false);
        if (!same_bits(ref, one.data(), one.size())) {
            std::printf("FAIL QSA 两段组合与参考不一致\n");
            CHECK(false);
        }
    }

    // 4 个 token：两个独立 scratch 逐位一致、有限、非零。
    auto make_x = [](std::uint64_t t) {
        std::vector<float> x(kN);
        for (std::uint64_t i = 0; i < kN; ++i) {
            x[static_cast<std::size_t>(i)] =
                0.05f * static_cast<float>(static_cast<int>((t * 7 + i) % 11) - 5);
        }
        return x;
    };
    std::vector<std::vector<float>> outs(2);
    for (int run = 0; run < 2; ++run) {
        BlockScratch s;
        configure_qsa(s.qsa, bl.qsa);
        std::vector<float> R = R0;
        for (std::uint64_t t = 0; t < 4; ++t) {
            const std::vector<float> x = make_x(t);
            CHECK(run_block_layer(g, bl, R.data(), s, err));
            for (float v : R) CHECK(std::isfinite(v));
            outs[static_cast<std::size_t>(run)].insert(outs[static_cast<std::size_t>(run)].end(),
                                                       R.begin(), R.end());
        }
        CHECK(s.qsa.pos == 4);
    }
    for (std::size_t i = 0; i < outs[0].size(); ++i) {
        CHECK(std::bit_cast<std::uint32_t>(outs[0][i]) == std::bit_cast<std::uint32_t>(outs[1][i]));
    }
    bool nonzero = false;
    for (float v : outs[0]) {
        if (v != 0.0f) nonzero = true;
    }
    CHECK(nonzero);
}

// 构建期/运行期的接错都必须报错。
void test_failures() {
    std::string err;
    // 缺 inject。
    {
        std::vector<TensorSpec> ts = gr_group("blk.0.hc_attn_", 1.0f, 0.05f, 0.5f, 0.4f);
        ts.pop_back();
        append(ts, gr_group("blk.0.hc_ffn_", 2.0f, 0.05f, 0.5f, -0.4f));
        append(ts, gdn_attn("blk.0."));
        append(ts, moe_tensors("blk.0."));
        const std::filesystem::path p = write_gguf("qinfer_block_noinj.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(!build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                 bl, err));
        CHECK(err.find("hc_attn_inject") != std::string::npos);
    }
    // hc 组形状与几何不自洽：hc_ffn_up（索引 6）的行宽多 1。
    {
        std::vector<TensorSpec> ts = block_tensors(true);
        ts[6].dims[1] = kN * kHc + 1;
        ts[6].data = bf16_sym(kLr * (kN * kHc + 1), 0.5f);
        const std::filesystem::path p = write_gguf("qinfer_block_badshape.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(!build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                 bl, err));
        CHECK(err.find("不自洽") != std::string::npos);
    }
    // inject 的形状不自洽：hc_attn_inject（索引 3）的第二维不是 hc。
    {
        std::vector<TensorSpec> ts = block_tensors(true);
        ts[3].dims[1] = kHc + 1;
        ts[3].data = bf16_sym((kN * kHc) * (kHc + 1), 0.4f);
        const std::filesystem::path p = write_gguf("qinfer_block_badinject.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(!build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                 bl, err));
        CHECK(err.find("不自洽") != std::string::npos);
    }
    // 判不出层型：注意力张力一个都没有。
    {
        std::vector<TensorSpec> ts = gr_group("blk.0.hc_attn_", 1.0f, 0.05f, 0.5f, 0.4f);
        append(ts, gr_group("blk.0.hc_ffn_", 2.0f, 0.05f, 0.5f, -0.4f));
        append(ts, moe_tensors("blk.0."));
        const std::filesystem::path p = write_gguf("qinfer_block_nokind.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(!build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                 bl, err));
        CHECK(err.find("判不出") != std::string::npos);
    }
    // MoE 建不起来：缺路由器。
    {
        std::vector<TensorSpec> ts = block_tensors(true);
        for (auto it = ts.begin(); it != ts.end(); ++it) {
            if (it->name == "blk.0.ffn_gate_inp.weight") {
                ts.erase(it);
                break;
            }
        }
        const std::filesystem::path p = write_gguf("qinfer_block_norouter.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(!build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                 bl, err));
        CHECK(!err.empty());
    }
    // 运行期：hc_attn_norm 不是 F32（BF16）→ 明确报错（构建不管精度，跑的时候才核）。
    {
        std::vector<TensorSpec> ts = block_tensors(true);
        ts[0].type = 30;
        ts[0].data = bf16_sym(kN * kHc, 1.0f);
        const std::filesystem::path p = write_gguf("qinfer_block_normbf16.gguf", ts);
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        BlockLayer bl;
        CHECK(build_block_layer(g, 0, kN, kHc, kLr, kTopK, kGdnHd, kNrot, kIdx, kIdxTopK, kIdxBlock,
                                bl, err));
        std::vector<float> R = make_R();
        BlockScratch s;
        err.clear();
        CHECK(!run_block_layer(g, bl, R.data(), s, err));
        CHECK(err.find("F32") != std::string::npos);
    }
}

// 真模型手工入口：建一层、跑一个 token、与「显式拼一遍」的参考核对。
int manual_model_entry(const char* model, int layer, int tokens) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    // 先判型以决定 head_dim（GDN 是 128、QSA 是 256；[S-54]/[S-57]）。
    const bool gdn = g.find("blk." + std::to_string(layer) + ".attn_qkv.weight") != nullptr;
    const std::uint64_t head_dim = gdn ? 128 : 256;
    BlockLayer bl;
    if (!build_block_layer(g, layer, 2560, 4, 320, 10, head_dim, /*n_rot=*/64, /*idx_dim=*/128,
                           /*idx_top_k=*/2048, /*idx_block=*/4, bl, err)) {
        std::printf("层 %d 建不起来：%s\n", layer, err.c_str());
        return 1;
    }
    std::printf("层 %d：%s；hc 几何 n_embd %llu / hc %llu / hc_lr %llu\n", layer,
                bl.kind == LayerKind::kGdn ? "GDN" : "QSA", (unsigned long long) bl.n_embd,
                (unsigned long long) bl.gr.hc, (unsigned long long) bl.gr.hc_lr);
    if (bl.kind == LayerKind::kQsa) {
        std::printf("  QSA：n_head %llu、n_head_kv %llu、head_dim %llu、idx_dim %llu\n",
                    (unsigned long long) bl.qsa.sh.n_head, (unsigned long long) bl.qsa.sh.n_head_kv,
                    (unsigned long long) bl.qsa.sh.head_dim, (unsigned long long) bl.qsa.idx_dim);
    }

    GrLoaded L;
    load_gr(g, bl, L);
    std::vector<float> x(2560);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.05f * (float) ((int) (i % 17) - 8);
    }
    std::vector<float> R0(2560 * 4, 0.0f);
    hc_broadcast_init(x.data(), 2560, 4, R0.data());

    BlockScratch s;
    if (bl.kind == LayerKind::kQsa) {
        configure_qsa(s.qsa, bl.qsa);
        s.qsa.kv.n_pages = (std::uint64_t) tokens + 1;
        s.qsa.kv.reset();
        s.qsa.kv.table.resize(s.qsa.kv.n_pages);
        for (std::uint64_t p = 0; p < s.qsa.kv.n_pages; ++p) s.qsa.kv.table[p] = (std::int32_t) p;
        std::string e2;
        if (!build_rope_table(64, 1.0e7, 4096, s.qsa.cos_tab, s.qsa.sin_tab, e2)) return 1;
    }
    double mag = 0.0;
    bool finite = true;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> R = R0;
        if (!run_block_layer(g, bl, R.data(), s, err)) {
            std::printf("跑第 %d 个 token 失败：%s\n", t, err.c_str());
            return 1;
        }
        if (t == tokens - 1) {
            for (float v : R) {
                if (!std::isfinite(v)) finite = false;
                mag += std::fabs((double) v);
            }
            // 与「显式拼一遍」的参考核对（末个 token，用同一份 R0 与同配置的 scratch）。QSA 的
            // 参考要起一个只有 1 个 token 的序列，故只在 GDN 层核对（QSA 的接线由合成侧覆盖）。
            if (bl.kind == LayerKind::kGdn) {
                std::vector<float> R2 = R0;
                std::vector<float> ref(2560 * 4, 0.0f);
                ref_compose(g, bl, L.aw, L.fw, R2.data(), ref.data(), false, false, false);
                double diff = 0.0, m = 0.0;
                for (std::size_t i = 0; i < ref.size(); ++i) {
                    diff += std::fabs((double) ref[i] - (double) R[i]);
                    m += std::fabs((double) ref[i]);
                }
                std::printf("  与显式拼一遍的参考：%s（|参考| %.6g）\n",
                            diff <= 1e-4 * (m > 0 ? m : 1.0) ? "一致" : "不一致", m);
            }
        }
    }
    std::printf("  %d 个 token：末个 token |out| 合计 %.6g、有限：%s\n", tokens, mag,
                finite ? "是" : "否");
    return finite ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 0, tokens = 1;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--tokens") tokens = std::atoi(argv[i + 1]);
        }
        return manual_model_entry(argv[2], layer, tokens);
    }
    test_gdn_block();
    test_qsa_block();
    test_failures();
    std::puts("block_layer: 两段门控残差接注意力与 MoE 跑通（GDN 与 QSA 都核），"
              "对调两段 / 两段共用一组 / inject 缺席三条错读法都可观测，五种接错都明确报错");
    return 0;
}
