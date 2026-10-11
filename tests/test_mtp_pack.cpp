// MTP 权重加载（两方言）的回归。不引第三方框架。
//
// 夹具是**同一层的小号复制**，两份方言各一份：n_embd 8 / hc 2 / hc_lr 2 / head_dim 4（n_head 2、
// n_head_kv 1）/ idx_dim 2（idx_n_head 2）/ 每专家前馈 4 / 3 个专家 / top-2。逐张力的档也照真文件的
// 差别给：引擎那份路由是 BF16、社区那份是 F32；共享专家的标量门引擎是 [8×1] 二维、社区是 [8] 一维。
//
// 钉子：
//   1. 两份方言都能建，判型对，且推出同一组几何（n_head / n_head_kv / idx_n_head / 每专家前馈 / 专家数）；
//   2. 融合切片对得上：引擎那份 gate/up 出自同一张张力、up.row_off = gate.rows；索引器同理，且 q 的
//      行数 = idx_n_head·idx_dim 由调用方给的 idx_n_head 决定（融合后拆不出比例）；
//   3. 同一角色的精度与维数可以随方言变（路由 BF16 对 F32；标量门二维对一维）；
//   4. 共享头：社区那份自包含，引擎那份没有（要 bind 主模型）；
//   5. 接错必须报错：两种方言都没有 / 同时出现 / 缺一张角色张力 / 调用方给的 idx_n_head 与形状不符。
#include "dense/mtp_pack.hpp"

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

constexpr std::uint64_t kN = 8, kHc = 2, kLr = 2;
constexpr std::uint64_t kHcDim = kN * kHc;   // 16
constexpr std::uint64_t kHd = 4, kNh = 2, kNkv = 1;
constexpr std::uint64_t kQk = kNh * kHd;     // 8
constexpr std::uint64_t kKv = kNkv * kHd;    // 4
constexpr std::uint64_t kIdx = 2, kIdxH = 2;
constexpr std::uint64_t kFfn = 4, kExpert = 3;
constexpr std::uint64_t kTopK = 2;
constexpr std::uint32_t kAlign = 32;
constexpr std::uint32_t kF32 = 0, kQ8 = 8, kBF16 = 30, kQ20 = 42;

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
    const auto s = [&](const std::string& v) {
        u64(v.size());
        b.insert(b.end(), v.begin(), v.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    s("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        s(t.name);
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

// 只给表项：数据内容本模块不读，故填 4 字节占位。
std::vector<std::uint8_t> dummy() { return std::vector<std::uint8_t>(4, 0); }

void add1(std::vector<TensorSpec>& ts, const std::string& n, std::uint64_t elems, std::uint32_t type) {
    ts.push_back({n, {elems}, type, dummy()});
}
void add2(std::vector<TensorSpec>& ts, const std::string& n, std::uint64_t cols, std::uint64_t rows,
          std::uint32_t type) {
    ts.push_back({n, {cols, rows}, type, dummy()});
}
void add3(std::vector<TensorSpec>& ts, const std::string& n, std::uint64_t cols, std::uint64_t rows,
          std::uint64_t experts, std::uint32_t type) {
    ts.push_back({n, {cols, rows, experts}, type, dummy()});
}

// 引擎那份（31 张、checkpoint 原始名、融合、不含共享头）。
std::vector<TensorSpec> engine_fixture() {
    std::vector<TensorSpec> ts;
    add1(ts, "mtp.pre_fc_norm_embedding.weight", kN, kF32);
    add1(ts, "mtp.pre_fc_norm_hidden.weight", kHcDim, kF32);
    add2(ts, "mtp.fc_embedding.weight", kN, kN, kBF16);
    add2(ts, "mtp.fc_hidden.weight", kN, kN, kBF16);
    add1(ts, "mtp.hyper_connection_mixer.hc_norm.weight", kHcDim, kF32);
    add2(ts, "mtp.hyper_connection_mixer.input_mix_weight_down.weight", kHcDim, kLr, kBF16);
    add2(ts, "mtp.hyper_connection_mixer.input_mix_weight_up.weight", kLr, kHcDim, kBF16);
    const char* hc[2] = {"mtp.layers.0.attn_hyper_connection.", "mtp.layers.0.mlp_hyper_connection."};
    for (const char* p : hc) {
        add1(ts, std::string(p) + "hc_norm.weight", kHcDim, kF32);
        add2(ts, std::string(p) + "input_mix_weight_down.weight", kHcDim, kLr, kBF16);
        add2(ts, std::string(p) + "input_mix_weight_up.weight", kLr, kHcDim, kBF16);
        add2(ts, std::string(p) + "block_inject_weight.weight", kHcDim, kHc, kBF16);
    }
    const char* sa = "mtp.layers.0.self_attn.";
    add2(ts, std::string(sa) + "q_proj.weight", kN, 2 * kQk, kBF16);
    add2(ts, std::string(sa) + "k_proj.weight", kN, kKv, kBF16);
    add2(ts, std::string(sa) + "v_proj.weight", kN, kKv, kBF16);
    add2(ts, std::string(sa) + "o_proj.weight", kQk, kN, kBF16);
    add1(ts, std::string(sa) + "q_norm.weight", kHd, kF32);
    add1(ts, std::string(sa) + "k_norm.weight", kHd, kF32);
    add2(ts, std::string(sa) + "indexer.index_qk_proj.weight", kN, (kIdxH + 1) * kIdx, kBF16);
    add1(ts, std::string(sa) + "indexer.q_layernorm.weight", kIdx, kF32);
    add1(ts, std::string(sa) + "indexer.k_layernorm.weight", kIdx, kF32);
    const char* mlp = "mtp.layers.0.mlp.";
    add2(ts, std::string(mlp) + "gate.weight", kN, kExpert, kBF16);
    add3(ts, std::string(mlp) + "experts.gate_up_proj", kN, 2 * kFfn, kExpert, kQ20);
    add3(ts, std::string(mlp) + "experts.down_proj", kFfn, kN, kExpert, kQ20);
    add2(ts, std::string(mlp) + "shared_expert.gate_proj.weight", kN, kFfn, kBF16);
    add2(ts, std::string(mlp) + "shared_expert.up_proj.weight", kN, kFfn, kBF16);
    add2(ts, std::string(mlp) + "shared_expert.down_proj.weight", kFfn, kN, kBF16);
    add2(ts, std::string(mlp) + "shared_expert_gate.weight", kN, 1, kBF16);  // 二维
    return ts;
}

// 社区那份（35 张、blk.48.*、拆开、自包含）。
std::vector<TensorSpec> blk48_fixture() {
    std::vector<TensorSpec> ts;
    add1(ts, "blk.48.nextn.enorm.weight", kN, kF32);
    add1(ts, "blk.48.nextn.hnorm.weight", kHcDim, kF32);
    add2(ts, "blk.48.nextn.eh_proj.weight", 2 * kN, kN, kQ8);  // 5120 = 2560+2560 那一维的缩影
    add1(ts, "blk.48.nextn.shared_head_norm.weight", kHcDim, kF32);
    add1(ts, "blk.48.hc_attn_norm.weight", kHcDim, kF32);
    add2(ts, "blk.48.hc_attn_down.weight", kHcDim, kLr, kQ8);
    add2(ts, "blk.48.hc_attn_up.weight", kLr, kHcDim, kQ8);
    add2(ts, "blk.48.hc_attn_inject.weight", kHcDim, kHc, kQ8);
    add1(ts, "blk.48.hc_ffn_norm.weight", kHcDim, kF32);
    add2(ts, "blk.48.hc_ffn_down.weight", kHcDim, kLr, kQ8);
    add2(ts, "blk.48.hc_ffn_up.weight", kLr, kHcDim, kQ8);
    add2(ts, "blk.48.hc_ffn_inject.weight", kHcDim, kHc, kQ8);
    add2(ts, "blk.48.attn_q.weight", kN, 2 * kQk, kQ8);
    add2(ts, "blk.48.attn_k.weight", kN, kKv, kQ8);
    add2(ts, "blk.48.attn_v.weight", kN, kKv, kQ8);
    add2(ts, "blk.48.attn_output.weight", kQk, kN, kQ8);
    add1(ts, "blk.48.attn_q_norm.weight", kHd, kF32);
    add1(ts, "blk.48.attn_k_norm.weight", kHd, kF32);
    add2(ts, "blk.48.indexer.q_proj.weight", kN, kIdxH * kIdx, kBF16);
    add2(ts, "blk.48.indexer.k_proj.weight", kN, kIdx, kBF16);
    add1(ts, "blk.48.indexer.q_norm.weight", kIdx, kF32);
    add1(ts, "blk.48.indexer.k_norm.weight", kIdx, kF32);
    add2(ts, "blk.48.ffn_gate_inp.weight", kN, kExpert, kF32);  // 路由：F32（引擎那份是 BF16）
    add3(ts, "blk.48.ffn_gate_exps.weight", kN, kFfn, kExpert, kQ8);
    add3(ts, "blk.48.ffn_up_exps.weight", kN, kFfn, kExpert, kQ8);
    add3(ts, "blk.48.ffn_down_exps.weight", kFfn, kN, kExpert, kQ8);
    add2(ts, "blk.48.ffn_gate_shexp.weight", kN, kFfn, kQ8);
    add2(ts, "blk.48.ffn_up_shexp.weight", kN, kFfn, kQ8);
    add2(ts, "blk.48.ffn_down_shexp.weight", kFfn, kN, kQ8);
    add1(ts, "blk.48.ffn_gate_inp_shexp.weight", kN, kF32);  // 一维
    add2(ts, "token_embd.weight", kN, 4, kQ8);               // 词表随便
    add2(ts, "output.weight", kN, 4, kQ8);
    add1(ts, "output_hc_norm.weight", kHcDim, kF32);
    add2(ts, "output_hc_down.weight", kHcDim, kLr, kF32);
    add2(ts, "output_hc_up.weight", kLr, kHcDim, kF32);
    return ts;
}

bool build_from(const char* file, const std::vector<TensorSpec>& ts, std::uint64_t idx_n_head,
                MtpPack& out, std::string& err) {
    const std::filesystem::path p = write_gguf(file, ts);
    artifact::GgufFile g;
    if (!g.open(p.string(), err)) return false;
    return build_mtp_pack(g, kN, kHc, kLr, kHd, /*n_rot=*/2, kIdx, idx_n_head, kTopK, out, err);
}

void test_both_dialects() {
    MtpPack a;
    std::string err;
    CHECK(build_from("qinfer_mtp_engine.gguf", engine_fixture(), kIdxH, a, err));
    CHECK(err.empty());
    CHECK(a.dialect == MtpDialect::kEngineNames);
    CHECK(a.n_head == kNh && a.n_head_kv == kNkv && a.idx_n_head == kIdxH);
    CHECK(a.n_expert == kExpert && a.expert_ffn == kFfn && a.qk_dim == kQk);
    CHECK(!a.self_contained_head);
    // 融合切片：gate/up 同一张张力、up 在每专家块的后半；非融合的角色 row_off = 0、block_rows == rows。
    const MtpSlice& ge = a.at(MtpRole::kGateExps);
    const MtpSlice& ue = a.at(MtpRole::kUpExps);
    CHECK(ge.fused && ue.fused && ge.t == ue.t);
    CHECK(ge.rows == kFfn && ue.row_off == kFfn && ge.block_rows == 2 * kFfn);
    CHECK(a.at(MtpRole::kDownExps).row_off == 0 && !a.at(MtpRole::kDownExps).fused);
    // 融合索引器：q 占前 idx_n_head·idx_dim 行、k 占后 idx_dim 行。
    const MtpSlice& iq = a.at(MtpRole::kIdxQ);
    const MtpSlice& ik = a.at(MtpRole::kIdxK);
    CHECK(iq.t == ik.t && iq.rows == kIdxH * kIdx && ik.row_off == iq.rows && ik.rows == kIdx);
    CHECK(iq.block_rows == (kIdxH + 1) * kIdx);
    // fc_hidden 在引擎那份里是独立一张；fc_embed 的输入维是 n_embd。
    CHECK(a.at(MtpRole::kFcHidden).present() && a.at(MtpRole::kFcHidden).cols == kN);
    CHECK(a.at(MtpRole::kFcEmbed).cols == kN);
    CHECK(a.at(MtpRole::kAttnHcInject).rows == kHc);
    CHECK(a.at(MtpRole::kSGateInp).rows == 1);  // [n_embd × 1] 二维

    MtpPack b;
    CHECK(build_from("qinfer_mtp_blk48.gguf", blk48_fixture(), kIdxH, b, err));
    CHECK(err.empty());
    CHECK(b.dialect == MtpDialect::kBlock48Names);
    CHECK(b.n_head == kNh && b.n_head_kv == kNkv && b.idx_n_head == kIdxH);
    CHECK(b.n_expert == kExpert && b.expert_ffn == kFfn);
    CHECK(b.self_contained_head);
    CHECK(b.t_token_embd != nullptr && b.t_output != nullptr && b.t_out_hc_norm != nullptr);
    // 拆开的那份：没有融合、没有独立的 fc_hidden、eh_proj 吃 concat。
    CHECK(!b.at(MtpRole::kGateExps).fused && b.at(MtpRole::kGateExps).rows == kFfn);
    CHECK(!b.at(MtpRole::kIdxQ).fused && b.at(MtpRole::kIdxQ).rows == kIdxH * kIdx);
    CHECK(!b.at(MtpRole::kFcHidden).present());
    CHECK(b.at(MtpRole::kFcEmbed).cols == 2 * kN);
    // 同一角色的精度与维数随方言变。
    CHECK(a.at(MtpRole::kRouter).type == kBF16 && b.at(MtpRole::kRouter).type == kF32);
    CHECK(b.at(MtpRole::kSGateInp).rows == kN);  // 一维：rows 记元素数
    // 按档算行字节：attn_q 是 BF16（引擎）/ Q8_0（社区），列数 8 都能整除各自的块。
    CHECK(a.at(MtpRole::kAttnQ).row_bytes() == kN * 2);
}

void test_failures() {
    std::string err;
    MtpPack out;
    // 两种方言都没有。
    {
        std::vector<TensorSpec> ts;
        add1(ts, "blk.0.hc_attn_norm.weight", kHcDim, kF32);
        CHECK(!build_from("qinfer_mtp_none.gguf", ts, kIdxH, out, err));
        CHECK(err.find("判不出方言") != std::string::npos);
    }
    // 两种方言同时出现。
    {
        std::vector<TensorSpec> ts = engine_fixture();
        add2(ts, "blk.48.nextn.eh_proj.weight", 2 * kN, kN, kQ8);
        CHECK(!build_from("qinfer_mtp_both.gguf", ts, kIdxH, out, err));
        CHECK(err.find("同时出现") != std::string::npos);
    }
    // 缺一张角色张力（引擎那份的 attn_v）。
    {
        std::vector<TensorSpec> ts = engine_fixture();
        for (std::size_t i = 0; i < ts.size(); ++i) {
            if (ts[i].name == "mtp.layers.0.self_attn.v_proj.weight") {
                ts.erase(ts.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        CHECK(!build_from("qinfer_mtp_missing.gguf", ts, kIdxH, out, err));
        CHECK(err.find("attn_v") != std::string::npos);
    }
    // 调用方给的 idx_n_head 与形状不符（引擎那份的融合块只有 (2+1)·2 = 6 行）。
    {
        CHECK(!build_from("qinfer_mtp_idxh.gguf", engine_fixture(), /*idx_n_head=*/3, out, err));
        CHECK(err.find("index_qk_proj") != std::string::npos);
    }
    // 社区那份：拆开时 q 头数与调用方给的 idx_n_head 不符。
    {
        CHECK(!build_from("qinfer_mtp_idxh2.gguf", blk48_fixture(), /*idx_n_head=*/5, out, err));
        CHECK(err.find("idx_n_head") != std::string::npos);
    }
    // top_k 超过专家数。
    {
        const std::filesystem::path p = write_gguf("qinfer_mtp_topk.gguf", blk48_fixture());
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        MtpPack pk;
        CHECK(!build_mtp_pack(g, kN, kHc, kLr, kHd, 2, kIdx, kIdxH, /*top_k=*/kExpert + 1, pk, err));
        CHECK(err.find("top_k") != std::string::npos);
    }
}

// 真模型手工入口：对一份真 MTP pack 建一遍，打印方言、反推出来的几何与几张关键角色的切片。
int manual_model_entry(const char* model) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开：%s（%s）\n", model, err.c_str());
        return 1;
    }
    // 真几何（[S-62]）：n_embd 2560 / hc 4 / hc_lr 320 / head_dim 256 / n_rot 64 / idx_dim 128 /
    // idx_n_head 4 / top_k 10。
    MtpPack p;
    if (!build_mtp_pack(g, 2560, 4, 320, 256, 64, 128, 4, 10, p, err)) {
        std::printf("建 MTP pack 失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("方言：%s；自包含共享头：%s\n",
                p.dialect == MtpDialect::kEngineNames ? "引擎那份（mtp.*）" : "社区那份（blk.48.*）",
                p.self_contained_head ? "是" : "否（要 bind 主模型）");
    std::printf("  反推几何：n_head %llu、n_head_kv %llu、idx_n_head %llu、专家 %llu、每专家前馈 %llu\n",
                (unsigned long long) p.n_head, (unsigned long long) p.n_head_kv,
                (unsigned long long) p.idx_n_head, (unsigned long long) p.n_expert,
                (unsigned long long) p.expert_ffn);
    const char* roles[] = {"GateExps", "UpExps", "DownExps", "IdxQ", "IdxK", "SGateInp", "Router"};
    const MtpRole rs[] = {MtpRole::kGateExps, MtpRole::kUpExps, MtpRole::kDownExps, MtpRole::kIdxQ,
                          MtpRole::kIdxK, MtpRole::kSGateInp, MtpRole::kRouter};
    for (int i = 0; i < 7; ++i) {
        const MtpSlice& s = p.at(rs[i]);
        if (!s.present()) {
            std::printf("  %-9s 缺席\n", roles[i]);
            continue;
        }
        std::printf("  %-9s 档 %3u、cols %5llu、rows %5llu、row_off %5llu、块行 %5llu、专家 %llu%s\n",
                    roles[i], s.type, (unsigned long long) s.cols, (unsigned long long) s.rows,
                    (unsigned long long) s.row_off, (unsigned long long) s.block_rows,
                    (unsigned long long) s.experts, s.fused ? "（融合）" : "");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        return manual_model_entry(argv[2]);
    }
    test_both_dialects();
    test_failures();
    std::puts("mtp_pack: 两份方言各自解析到同一组角色（含融合切片与逐张力精度），"
              "n_head/n_head_kv/idx_n_head/每专家前馈维由形状反推互核，六种接错都明确报错");
    return 0;
}