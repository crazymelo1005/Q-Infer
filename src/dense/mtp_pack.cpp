#include "dense/mtp_pack.hpp"

#include "experts/expert_formats.hpp"

#include <vector>

namespace qinfer::dense {

const char* to_string(MtpRole r) {
    switch (r) {
        case MtpRole::kPreNormEmbedding: return "入口的 embedding 归一";
        case MtpRole::kPreNormHidden: return "入口的残差归一";
        case MtpRole::kFcEmbed: return "入口的 embedding 投影";
        case MtpRole::kFcHidden: return "入口的残差投影";
        case MtpRole::kFinalHcNorm: return "收尾 mixer 的归一";
        case MtpRole::kFinalHcDown: return "收尾 mixer 的 down";
        case MtpRole::kFinalHcUp: return "收尾 mixer 的 up";
        case MtpRole::kAttnHcNorm: return "注意力超连接的归一";
        case MtpRole::kAttnHcDown: return "注意力超连接的 down";
        case MtpRole::kAttnHcUp: return "注意力超连接的 up";
        case MtpRole::kAttnHcInject: return "注意力超连接的 inject";
        case MtpRole::kFfnHcNorm: return "MLP 超连接的归一";
        case MtpRole::kFfnHcDown: return "MLP 超连接的 down";
        case MtpRole::kFfnHcUp: return "MLP 超连接的 up";
        case MtpRole::kFfnHcInject: return "MLP 超连接的 inject";
        case MtpRole::kAttnQ: return "attn_q";
        case MtpRole::kAttnK: return "attn_k";
        case MtpRole::kAttnV: return "attn_v";
        case MtpRole::kAttnOut: return "attn_output";
        case MtpRole::kAttnQNorm: return "attn_q_norm";
        case MtpRole::kAttnKNorm: return "attn_k_norm";
        case MtpRole::kIdxQ: return "索引器 q";
        case MtpRole::kIdxK: return "索引器 k";
        case MtpRole::kIdxQNorm: return "索引器 q 归一";
        case MtpRole::kIdxKNorm: return "索引器 k 归一";
        case MtpRole::kRouter: return "路由";
        case MtpRole::kGateExps: return "专家 gate";
        case MtpRole::kUpExps: return "专家 up";
        case MtpRole::kDownExps: return "专家 down";
        case MtpRole::kSGate: return "共享专家 gate";
        case MtpRole::kSUp: return "共享专家 up";
        case MtpRole::kSDown: return "共享专家 down";
        case MtpRole::kSGateInp: return "共享专家的标量门";
        case MtpRole::kCount: break;
    }
    return "?";
}

namespace {

// 两套方言的名字。`nullptr` 表示该方言下这个角色由另一个张力承担（eh_proj 融合）。
struct Names {
    const char* engine;  // `mtp.*`
    const char* blk48;   // `blk.48.*`
};

const Names kNames[static_cast<std::size_t>(MtpRole::kCount)] = {
    {"mtp.pre_fc_norm_embedding.weight", "blk.48.nextn.enorm.weight"},
    {"mtp.pre_fc_norm_hidden.weight", "blk.48.nextn.hnorm.weight"},
    {"mtp.fc_embedding.weight", "blk.48.nextn.eh_proj.weight"},
    {"mtp.fc_hidden.weight", nullptr},
    {"mtp.hyper_connection_mixer.hc_norm.weight", "blk.48.nextn.shared_head_norm.weight"},
    {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "blk.48.hc_ffn_down.weight"},
    {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "blk.48.hc_ffn_up.weight"},
    {"mtp.layers.0.attn_hyper_connection.hc_norm.weight", "blk.48.hc_attn_norm.weight"},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "blk.48.hc_attn_down.weight"},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "blk.48.hc_attn_up.weight"},
    {"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", "blk.48.hc_attn_inject.weight"},
    {"mtp.layers.0.mlp_hyper_connection.hc_norm.weight", "blk.48.hc_ffn_norm.weight"},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "blk.48.hc_ffn_down.weight"},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "blk.48.hc_ffn_up.weight"},
    {"mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", "blk.48.hc_ffn_inject.weight"},
    {"mtp.layers.0.self_attn.q_proj.weight", "blk.48.attn_q.weight"},
    {"mtp.layers.0.self_attn.k_proj.weight", "blk.48.attn_k.weight"},
    {"mtp.layers.0.self_attn.v_proj.weight", "blk.48.attn_v.weight"},
    {"mtp.layers.0.self_attn.o_proj.weight", "blk.48.attn_output.weight"},
    {"mtp.layers.0.self_attn.q_norm.weight", "blk.48.attn_q_norm.weight"},
    {"mtp.layers.0.self_attn.k_norm.weight", "blk.48.attn_k_norm.weight"},
    {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "blk.48.indexer.q_proj.weight"},
    {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight", "blk.48.indexer.k_proj.weight"},
    {"mtp.layers.0.self_attn.indexer.q_layernorm.weight", "blk.48.indexer.q_norm.weight"},
    {"mtp.layers.0.self_attn.indexer.k_layernorm.weight", "blk.48.indexer.k_norm.weight"},
    {"mtp.layers.0.mlp.gate.weight", "blk.48.ffn_gate_inp.weight"},
    {"mtp.layers.0.mlp.experts.gate_up_proj", "blk.48.ffn_gate_exps.weight"},
    {"mtp.layers.0.mlp.experts.gate_up_proj", "blk.48.ffn_up_exps.weight"},
    {"mtp.layers.0.mlp.experts.down_proj", "blk.48.ffn_down_exps.weight"},
    {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", "blk.48.ffn_gate_shexp.weight"},
    {"mtp.layers.0.mlp.shared_expert.up_proj.weight", "blk.48.ffn_up_shexp.weight"},
    {"mtp.layers.0.mlp.shared_expert.down_proj.weight", "blk.48.ffn_down_shexp.weight"},
    {"mtp.layers.0.mlp.shared_expert_gate.weight", "blk.48.ffn_gate_inp_shexp.weight"},
};

// 期望形状：rank 必须精确匹配；cols / rows / elems 为 0 表示不核（由形状反推或由别的角色交叉核）。
enum class Fuse : std::uint8_t { kNone, kFirst, kSecond };

struct Expect {
    int rank = 2;
    std::uint64_t cols = 0;
    std::uint64_t rows = 0;
    std::uint64_t elems = 0;  // 一维张量的元素数
    Fuse fuse = Fuse::kNone;  // 融合张力的前半 / 后半（两半等分）
};

const artifact::GgufTensorInfo* find(const artifact::GgufFile& g, const char* name) {
    return name == nullptr ? nullptr : g.find(name);
}

bool fill(const artifact::GgufFile& g, const char* name, const Expect& e, MtpSlice& out,
          std::string& err) {
    const artifact::GgufTensorInfo* t = find(g, name);
    if (t == nullptr) {
        err = std::string(name) + " 不存在";
        return false;
    }
    if (t->dims.size() != static_cast<std::size_t>(e.rank)) {
        err = std::string(name) + " 是 " + std::to_string(t->dims.size()) + " 维，期望 " +
              std::to_string(e.rank) + " 维";
        return false;
    }
    out.t = t;
    out.type = t->type;
    out.cols = t->dims[0];
    out.experts = e.rank == 3 ? t->dims[2] : 1;
    out.block_rows = e.rank >= 2 ? t->dims[1] : t->elements();
    if (e.rank == 1) {
        out.rows = t->elements();
        if (e.elems != 0 && t->elements() != e.elems) {
            err = std::string(name) + " 的元素数 " + std::to_string(t->elements()) + " 期望 " +
                  std::to_string(e.elems);
            return false;
        }
    }
    // 融合张力的两半（等分）：每半的行数 = 每专家块总行数的一半。
    if (e.fuse != Fuse::kNone) {
        if (e.rank < 2 || out.block_rows % 2 != 0) {
            err = std::string(name) + " 的每专家块行数 " + std::to_string(out.block_rows) +
                  " 不是偶数，装不下两半";
            return false;
        }
        out.rows = out.block_rows / 2;
        out.row_off = e.fuse == Fuse::kSecond ? out.rows : 0;
        out.fused = true;
    } else {
        if (e.rank >= 2) out.rows = out.block_rows;
        out.row_off = 0;
    }
    if (e.cols != 0 && out.cols != e.cols) {
        err = std::string(name) + " 的输入维 " + std::to_string(out.cols) + " 期望 " +
              std::to_string(e.cols);
        return false;
    }
    if (e.rows != 0 && out.rows != e.rows) {
        err = std::string(name) + " 的行数 " + std::to_string(out.rows) + " 期望 " +
              std::to_string(e.rows);
        return false;
    }
    return true;
}

bool require_equal(std::uint64_t got, std::uint64_t want, const char* what, std::string& err) {
    if (got != want) {
        err = std::string(what) + " 对不上：" + std::to_string(got) + " vs " + std::to_string(want);
        return false;
    }
    return true;
}

}  // namespace

std::uint64_t MtpSlice::row_bytes() const {
    if (t == nullptr) return 0;
    experts::MatrixSpec s;
    s.format = experts::format_from_type_code(type);
    s.experts = 1;
    s.cols = cols;
    s.rows = 1;
    if (!s.geometry_ok()) return 0;
    return s.row_bytes();
}

bool build_mtp_pack(const artifact::GgufFile& gguf, std::uint64_t n_embd, std::uint64_t hc,
                    std::uint64_t hc_lr, std::uint64_t head_dim, std::uint64_t n_rot,
                    std::uint64_t idx_dim, std::uint64_t idx_n_head, std::uint64_t top_k,
                    MtpPack& out, std::string& err) {
    err.clear();
    out = MtpPack{};
    out.n_embd = n_embd;
    out.hc = hc;
    out.hc_lr = hc_lr;
    out.head_dim = head_dim;
    out.n_rot = n_rot;
    out.idx_dim = idx_dim;
    out.top_k = top_k;
    if (n_embd == 0 || hc == 0 || hc_lr == 0 || head_dim == 0 || idx_dim == 0 || idx_n_head == 0) {
        err = "几何参数有零";
        return false;
    }
    const std::uint64_t hc_dim = n_embd * hc;

    // 判方言：引擎那份有 `mtp.fc_embedding.weight`，社区那份有 `blk.48.nextn.eh_proj.weight`。
    const bool has_engine = gguf.find("mtp.fc_embedding.weight") != nullptr;
    const bool has_blk48 = gguf.find("blk.48.nextn.eh_proj.weight") != nullptr;
    if (has_engine == has_blk48) {
        err = has_engine ? "两种方言同时出现（既有 mtp.fc_embedding 又有 blk.48.nextn.eh_proj）"
                         : "判不出方言：两种名字都没有";
        return false;
    }
    out.dialect = has_engine ? MtpDialect::kEngineNames : MtpDialect::kBlock48Names;
    const bool engine = has_engine;

    struct Job {
        MtpRole role;
        Expect e;
    };
    const std::vector<Job> jobs = {
        {MtpRole::kPreNormEmbedding, {1, 0, 0, n_embd, Fuse::kNone}},
        {MtpRole::kPreNormHidden, {1, 0, 0, hc_dim, Fuse::kNone}},
        {MtpRole::kFcEmbed, {2, engine ? n_embd : 2 * n_embd, n_embd, 0, Fuse::kNone}},
        {MtpRole::kFcHidden, {2, hc_dim, n_embd, 0, Fuse::kNone}},
        {MtpRole::kFinalHcNorm, {1, 0, 0, hc_dim, Fuse::kNone}},
        {MtpRole::kFinalHcDown, {2, hc_dim, hc_lr, 0, Fuse::kNone}},
        {MtpRole::kFinalHcUp, {2, hc_lr, hc_dim, 0, Fuse::kNone}},
        {MtpRole::kAttnHcNorm, {1, 0, 0, hc_dim, Fuse::kNone}},
        {MtpRole::kAttnHcDown, {2, hc_dim, hc_lr, 0, Fuse::kNone}},
        {MtpRole::kAttnHcUp, {2, hc_lr, hc_dim, 0, Fuse::kNone}},
        {MtpRole::kAttnHcInject, {2, hc_dim, hc, 0, Fuse::kNone}},
        {MtpRole::kFfnHcNorm, {1, 0, 0, hc_dim, Fuse::kNone}},
        {MtpRole::kFfnHcDown, {2, hc_dim, hc_lr, 0, Fuse::kNone}},
        {MtpRole::kFfnHcUp, {2, hc_lr, hc_dim, 0, Fuse::kNone}},
        {MtpRole::kFfnHcInject, {2, hc_dim, hc, 0, Fuse::kNone}},
        {MtpRole::kAttnQ, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kAttnK, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kAttnV, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kAttnOut, {2, 0, n_embd, 0, Fuse::kNone}},
        {MtpRole::kAttnQNorm, {1, 0, 0, head_dim, Fuse::kNone}},
        {MtpRole::kAttnKNorm, {1, 0, 0, head_dim, Fuse::kNone}},
        {MtpRole::kIdxQ, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kIdxK, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kIdxQNorm, {1, 0, 0, idx_dim, Fuse::kNone}},
        {MtpRole::kIdxKNorm, {1, 0, 0, idx_dim, Fuse::kNone}},
        {MtpRole::kRouter, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kGateExps, {3, n_embd, 0, 0, engine ? Fuse::kFirst : Fuse::kNone}},
        {MtpRole::kUpExps, {3, n_embd, 0, 0, engine ? Fuse::kSecond : Fuse::kNone}},
        {MtpRole::kDownExps, {3, 0, 0, 0, Fuse::kNone}},
        {MtpRole::kSGate, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kSUp, {2, n_embd, 0, 0, Fuse::kNone}},
        {MtpRole::kSDown, {2, 0, 0, 0, Fuse::kNone}},
        {MtpRole::kSGateInp, {engine ? 2 : 1, n_embd, engine ? 1 : 0, n_embd, Fuse::kNone}},
    };
    for (const Job& j : jobs) {
        const std::size_t idx = static_cast<std::size_t>(j.role);
        const char* name = engine ? kNames[idx].engine : kNames[idx].blk48;
        if (name == nullptr) {
            // 社区那份没有单独的 fc_hidden：eh_proj 的输入轴前半是 e、后半是 h。
            if (j.role == MtpRole::kFcHidden) continue;
            err = std::string(to_string(j.role)) + " 在该方言里没有对应张力";
            return false;
        }
        if (!fill(gguf, name, j.e, out.role[idx], err)) {
            err = std::string("解析 ") + to_string(j.role) + " 失败：" + err;
            return false;
        }
    }
    // 社区那份没有独立的 fc_hidden：入口投影是一张 eh_proj 吃 concat(e, h)，故 kFcHidden 缺席
    // （由调用方按 eh_proj 的输入轴两半理解，而不是把它当成两张独立的 fc）。

    // 由形状反推 n_head / n_head_kv / idx_n_head / 每专家前馈维，并互相核对。
    const MtpSlice& q = out.at(MtpRole::kAttnQ);
    const MtpSlice& k = out.at(MtpRole::kAttnK);
    const MtpSlice& v = out.at(MtpRole::kAttnV);
    const MtpSlice& o = out.at(MtpRole::kAttnOut);
    if (q.rows % (2 * head_dim) != 0) {
        err = "attn_q 的行数 " + std::to_string(q.rows) + " 不是 2·head_dim 的整数倍";
        return false;
    }
    out.n_head = q.rows / (2 * head_dim);
    out.qk_dim = out.n_head * head_dim;
    out.n_head_kv = k.rows / head_dim;
    if (!require_equal(k.rows, out.n_head_kv * head_dim, "attn_k 的行数", err)) return false;
    if (!require_equal(v.rows, k.rows, "attn_v 的行数与 attn_k", err)) return false;
    if (!require_equal(o.cols, out.qk_dim, "attn_output 的输入维与 n_head·head_dim", err)) return false;
    if (out.n_head % out.n_head_kv != 0) {
        err = "n_head 不是 n_head_kv 的整数倍";
        return false;
    }
    MtpSlice& iq = out.role[static_cast<std::size_t>(MtpRole::kIdxQ)];
    MtpSlice& ik = out.role[static_cast<std::size_t>(MtpRole::kIdxK)];
    if (engine) {
        // 融合的 index_qk_proj：q 占前 idx_n_head·idx_dim 行、k 占后 idx_dim 行；整块 = (idx_n_head+1)·idx_dim。
        if (iq.t == nullptr || iq.t != ik.t) {
            err = "融合的 index_qk_proj 两半没指向同一张张力";
            return false;
        }
        iq.rows = idx_n_head * idx_dim;
        iq.row_off = 0;
        iq.fused = true;
        ik.rows = idx_dim;
        ik.row_off = iq.rows;
        ik.fused = true;
        if (!require_equal(iq.block_rows, (idx_n_head + 1) * idx_dim, "index_qk_proj 的行数", err)) {
            return false;
        }
    } else {
        if (iq.rows % idx_dim != 0) {
            err = "索引器 q 的行数 " + std::to_string(iq.rows) + " 不是 idx_dim 的整数倍";
            return false;
        }
        if (!require_equal(iq.rows / idx_dim, idx_n_head, "索引器 q 头数与调用方给的 idx_n_head",
                           err)) {
            return false;
        }
        if (!require_equal(ik.rows, idx_dim, "索引器 k 的行数", err)) return false;
    }
    out.idx_n_head = idx_n_head;

    // 专家：gate 的每专家块行数 = 每专家前馈维（引擎那份融合两半，故取半）。
    const MtpSlice& gate = out.at(MtpRole::kGateExps);
    const MtpSlice& up = out.at(MtpRole::kUpExps);
    const MtpSlice& down = out.at(MtpRole::kDownExps);
    out.n_expert = gate.experts;
    if (out.n_expert == 0) {
        err = "专家数为 0";
        return false;
    }
    if (engine) {
        if (gate.t != up.t || up.row_off != gate.rows) {
            err = "融合的 gate_up_proj 两半没对上";
            return false;
        }
        if (gate.block_rows % 2 != 0) {
            err = "gate_up_proj 的每专家块行数不是偶数";
            return false;
        }
    }
    out.expert_ffn = gate.rows;
    if (!require_equal(up.rows, out.expert_ffn, "专家 up 的行数与 gate", err)) return false;
    if (!require_equal(down.cols, out.expert_ffn, "专家 down 的输入维与每专家前馈维", err)) return false;
    if (!require_equal(down.rows, n_embd, "专家 down 的输出行数", err)) return false;
    if (!require_equal(up.experts, out.n_expert, "专家 up 的专家数", err)) return false;
    if (!require_equal(down.experts, out.n_expert, "专家 down 的专家数", err)) return false;
    if (static_cast<std::uint64_t>(top_k) > out.n_expert) {
        err = "top_k 超过专家数";
        return false;
    }
    // 路由器输出维 = 专家数。
    if (!require_equal(out.at(MtpRole::kRouter).rows, out.n_expert, "路由的行数与专家数", err)) {
        return false;
    }
    // 共享专家：无专家维，形状与路由专家一一对应。
    if (!require_equal(out.at(MtpRole::kSGate).rows, out.expert_ffn, "共享专家 gate 的行数", err)) {
        return false;
    }
    if (!require_equal(out.at(MtpRole::kSUp).rows, out.expert_ffn, "共享专家 up 的行数", err)) {
        return false;
    }
    if (!require_equal(out.at(MtpRole::kSDown).cols, out.expert_ffn, "共享专家 down 的输入维", err)) {
        return false;
    }
    if (!require_equal(out.at(MtpRole::kSDown).rows, n_embd, "共享专家 down 的行数", err)) return false;

    // 共享的 embedding 与头。
    out.self_contained_head = !engine;
    if (out.self_contained_head) {
        out.t_token_embd = find(gguf, "token_embd.weight");
        out.t_output = find(gguf, "output.weight");
        out.t_out_hc_norm = find(gguf, "output_hc_norm.weight");
        out.t_out_hc_down = find(gguf, "output_hc_down.weight");
        out.t_out_hc_up = find(gguf, "output_hc_up.weight");
        if (out.t_token_embd == nullptr || out.t_output == nullptr || out.t_out_hc_norm == nullptr ||
            out.t_out_hc_down == nullptr || out.t_out_hc_up == nullptr) {
            err = "社区那份应自包含 token_embd / output / output_hc_*，但有一张缺失";
            return false;
        }
        if (!require_equal(out.t_out_hc_norm->elements(), hc_dim, "output_hc_norm 的元素数", err)) {
            return false;
        }
    }
    return true;
}

}  // namespace qinfer::dense