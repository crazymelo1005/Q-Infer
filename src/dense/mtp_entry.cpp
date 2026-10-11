#include "dense/mtp_entry.hpp"

#include "dense/gated_residual.hpp"
#include "experts/expert_formats.hpp"
#include "kernels/bf16.hpp"

#include <cstring>

namespace qinfer::dense {

namespace {

constexpr float kEps = 1e-6f;

bool load_one(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
              std::vector<std::uint8_t>& dst, std::string& err) {
    std::uint64_t bytes = 0;
    if (t.type == 0) {
        bytes = t.elements() * 4;
    } else if (t.type == 30) {
        bytes = t.elements() * 2;
    } else {
        experts::MatrixSpec s;
        s.format = experts::format_from_type_code(t.type);
        s.experts = 1;
        s.cols = t.dims[0];
        s.rows = t.dims[1];
        if (!s.usable()) {
            err = t.name + " 的档没有内核（或几何不成立）";
            return false;
        }
        bytes = s.rows * s.row_bytes();
    }
    dst.resize(static_cast<std::size_t>(bytes));
    if (!g.read_at(t.offset, dst.data(), dst.size())) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

// 一维 F32 gamma 读成 float。
bool load_gamma(const artifact::GgufFile& g, const artifact::GgufTensorInfo* t,
                std::vector<float>& dst, std::string& err) {
    if (t == nullptr) {
        err = "gamma 张力缺席";
        return false;
    }
    if (t->type != 0) {
        err = t->name + " 不是 F32";
        return false;
    }
    dst.resize(static_cast<std::size_t>(t->elements()));
    if (!g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(dst.data()), dst.size() * 4)) {
        err = t->name + " 读失败";
        return false;
    }
    return true;
}

// 一个切片上的 GEMV：x 长 cols，输出 rows。权重按切片自己声明的档走「档 → 内核 + 激活搭档」那张表。
bool slice_gemv(const MtpSlice& sl, const std::vector<std::uint8_t>& bytes, const float* x,
                experts::FfnScratch& sc, float* out, std::string& err) {
    experts::MatrixSpec spec;
    spec.format = experts::format_from_type_code(sl.type);
    spec.experts = 1;
    spec.cols = sl.cols;
    spec.rows = sl.rows;
    // BF16 的矩阵**没有**「档 → 内核」那条路（路由器与索引器也走 bf16 点积那条），单独处理。
    if (spec.format == experts::Format::kBf16) {
        if (sl.cols == 0) {
            err = "MTP 入口的 BF16 投影列数为 0";
            return false;
        }
        const std::uint64_t row_bytes = sl.cols * 2;
        const std::uint64_t base = sl.row_off * row_bytes;
        if (base + sl.rows * row_bytes > bytes.size()) {
            err = "MTP 入口的切片越界（融合张力的行区间对不上）";
            return false;
        }
        kernels::bf16_gemv(reinterpret_cast<const std::uint16_t*>(bytes.data() + base),
                           sl.rows, sl.cols, x, out);
        return true;
    }
    if (!spec.usable()) {
        err = "MTP 入口的投影档不可用";
        return false;
    }
    const std::uint64_t base = sl.row_off * spec.row_bytes();
    if (base + spec.rows * spec.row_bytes() > bytes.size()) {
        err = "MTP 入口的切片越界（融合张力的行区间对不上）";
        return false;
    }
    if (!experts::quantize_input(spec, x, sc, err)) return false;
    return experts::run_gemv(spec, bytes.data() + base, sc, out, err);
}

}  // namespace

bool mtp_entry(const artifact::GgufFile& gguf, const MtpPack& pack, const float* tok_embd,
               const float* R_in, MtpEntryScratch& s, float* R_out, std::string& err) {
    err.clear();
    if (tok_embd == nullptr || R_in == nullptr || R_out == nullptr) {
        err = "mtp_entry 的指针为空";
        return false;
    }
    const std::uint64_t N = pack.n_embd;
    const std::uint64_t hc = pack.hc;
    const std::uint64_t hc_dim = N * hc;
    if (N == 0 || hc == 0) {
        err = "mtp_entry 的几何有零";
        return false;
    }

    // 入口两支的 gamma（都是 F32 一维）。
    if (!load_gamma(gguf, pack.at(MtpRole::kPreNormEmbedding).t, s.gamma_embed, err) ||
        !load_gamma(gguf, pack.at(MtpRole::kPreNormHidden).t, s.gamma_hidden, err)) {
        return false;
    }
    if (s.gamma_embed.size() != N || s.gamma_hidden.size() != hc_dim) {
        err = "入口 gamma 的长度与几何不符";
        return false;
    }
    const MtpSlice& fc_embed = pack.at(MtpRole::kFcEmbed);
    const MtpSlice& fc_hidden = pack.at(MtpRole::kFcHidden);
    if (!load_one(gguf, *fc_embed.t, s.w_fc_embed, err)) return false;
    const bool engine = pack.dialect == MtpDialect::kEngineNames;
    if (engine) {
        if (!fc_hidden.present()) {
            err = "引擎那份缺 fc_hidden";
            return false;
        }
        if (!load_one(gguf, *fc_hidden.t, s.w_fc_hidden, err)) return false;
    } else {
        if (!load_one(gguf, *fc_embed.t, s.w_eh_proj, err)) return false;
    }

    // en = rms_n(tok_embd) ⊙ gamma_embed。
    s.en.assign(static_cast<std::size_t>(N), 0.0f);
    {
        const float sc = rms_scale(tok_embd, N, kEps);
        for (std::uint64_t i = 0; i < N; ++i) {
            s.en[static_cast<std::size_t>(i)] =
                sc * tok_embd[i] * s.gamma_embed[static_cast<std::size_t>(i)];
        }
    }
    // hn = rms_{hc_dim}(R_in) ⊙ gamma_hidden —— 整栈一条 RMS（不是逐流）。
    s.hn.assign(static_cast<std::size_t>(hc_dim), 0.0f);
    {
        const float sc = rms_scale(R_in, hc_dim, kEps);
        for (std::uint64_t j = 0; j < hc_dim; ++j) {
            s.hn[static_cast<std::size_t>(j)] =
                sc * R_in[j] * s.gamma_hidden[static_cast<std::size_t>(j)];
        }
    }

    if (engine) {
        // 引擎那份：e2 = fc_embedding(en)（**先投影再广播**，不是把 en 直接加）；h2[c] = fc_hidden(第 c 条流)。
        s.tmp.assign(static_cast<std::size_t>(N), 0.0f);
        if (!slice_gemv(fc_embed, s.w_fc_embed, s.en.data(), s.gemv, s.tmp.data(), err)) return false;
        s.h2.assign(static_cast<std::size_t>(N), 0.0f);
        for (std::uint64_t c = 0; c < hc; ++c) {
            if (!slice_gemv(fc_hidden, s.w_fc_hidden, s.hn.data() + c * N, s.gemv, s.h2.data(),
                            err)) {
                return false;
            }
            for (std::uint64_t i = 0; i < N; ++i) {
                R_out[c * N + i] =
                    s.h2[static_cast<std::size_t>(i)] + s.tmp[static_cast<std::size_t>(i)];
            }
        }
        return true;
    }

    // 社区那份：R[c] = eh_proj(concat(en, hn 的第 c 条流))——两张矩阵已融进这一张。
    s.tmp.assign(static_cast<std::size_t>(2 * N), 0.0f);
    for (std::uint64_t c = 0; c < hc; ++c) {
        std::memcpy(s.tmp.data(), s.en.data(), static_cast<std::size_t>(N) * sizeof(float));
        std::memcpy(s.tmp.data() + N, s.hn.data() + c * N,
                    static_cast<std::size_t>(N) * sizeof(float));
        if (!slice_gemv(fc_embed, s.w_eh_proj, s.tmp.data(), s.gemv, R_out + c * N, err)) {
            return false;
        }
    }
    return true;
}

}  // namespace qinfer::dense