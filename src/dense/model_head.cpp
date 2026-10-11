#include "dense/model_head.hpp"

#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"

#include <cstring>

namespace qinfer::dense {

namespace {

const artifact::GgufTensorInfo* need(const artifact::GgufFile& g, const std::string& name,
                                     std::string& err) {
    const artifact::GgufTensorInfo* t = g.find(name);
    if (t == nullptr) err = name + " 不存在";
    return t;
}

// 一维 F32 的 gamma。
bool load_f32(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
              std::vector<float>& dst, std::string& err) {
    dst.resize(static_cast<std::size_t>(t.elements()));
    if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(dst.data()), dst.size() * 4)) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

}  // namespace

void hc_broadcast_init(const float* x, std::uint64_t n_embd, std::uint64_t hc, float* R) {
    for (std::uint64_t c = 0; c < hc; ++c) {
        std::memcpy(R + c * n_embd, x, static_cast<std::size_t>(n_embd) * sizeof(float));
    }
}

bool build_model_head(const artifact::GgufFile& gguf, std::uint64_t n_embd, std::uint64_t hc,
                      std::uint64_t hc_lr, ModelHead& out, std::string& err) {
    err.clear();
    out = ModelHead{};
    out.n_embd = n_embd;
    out.gr = GrShapes{n_embd, hc, hc_lr};
    if (!out.gr.sane()) {
        err = "输出的几何不成立（n_embd / hc / hc_lr 有零）";
        return false;
    }
    out.t_norm = need(gguf, "output_hc_norm.weight", err);
    if (out.t_norm == nullptr) return false;
    out.t_down = need(gguf, "output_hc_down.weight", err);
    if (out.t_down == nullptr) return false;
    out.t_up = need(gguf, "output_hc_up.weight", err);
    if (out.t_up == nullptr) return false;
    out.t_out = need(gguf, "output.weight", err);
    if (out.t_out == nullptr) return false;

    // 形状核对（[S-53] 已实测过的布局）：norm 一维 hc·n_embd、down {hc·n_embd, hc_lr}、
    // up {hc_lr, hc·n_embd}；输出投影的输入维必须是 n_embd。
    if (out.t_norm->elements() != n_embd * hc || out.t_norm->dims.size() != 1) {
        err = "output_hc_norm 的形状不是 [hc·n_embd]";
        return false;
    }
    if (out.t_down->dims.size() != 2 || out.t_down->dims[0] != n_embd * hc ||
        out.t_down->dims[1] != hc_lr) {
        err = "output_hc_down 的形状不是 [hc·n_embd, hc_lr]";
        return false;
    }
    if (out.t_up->dims.size() != 2 || out.t_up->dims[0] != hc_lr ||
        out.t_up->dims[1] != n_embd * hc) {
        err = "output_hc_up 的形状不是 [hc_lr, hc·n_embd]";
        return false;
    }
    if (out.t_out->dims.size() != 2 || out.t_out->dims[0] != n_embd) {
        err = "output.weight 的输入维不是 n_embd";
        return false;
    }
    out.out.format = experts::format_from_type_code(out.t_out->type);
    out.out.experts = 1;
    out.out.cols = out.t_out->dims[0];
    out.out.rows = out.t_out->dims[1];
    if (!out.out.geometry_ok() || !experts::has_kernel(out.out.format)) {
        err = std::string("output.weight 的档 ") + experts::format_name(out.out.format) +
              " 不可用（几何或内核）";
        return false;
    }
    return true;
}

bool run_model_head(const artifact::GgufFile& gguf, const ModelHead& h, const float* R,
                    HeadScratch& s, std::string& err) {
    err.clear();
    if (R == nullptr) {
        err = "run_model_head 的 R 为空";
        return false;
    }
    const std::uint64_t hc_dim = h.n_embd * h.gr.hc;
    // w_norm 是 F32（[S-53] 实测：hc_*_norm 是 F32、三个投影是 BF16）——按声明的类型读，不写死。
    if (s.w_norm_f32.empty()) {
        if (!load_f32(gguf, *h.t_norm, s.w_norm_f32, err)) return false;
        const std::uint32_t bf16 = 30;
        const auto load_bf16 = [&](const artifact::GgufTensorInfo* t,
                                   std::vector<std::uint16_t>& dst) {
            dst.resize(static_cast<std::size_t>(t->elements()));
            return gguf.read_at(t->offset, reinterpret_cast<std::uint8_t*>(dst.data()),
                                dst.size() * 2);
        };
        if (h.t_down->type != bf16 || h.t_up->type != bf16) {
            err = "output_hc_down / up 不是 BF16（本路径按 BF16 读）";
            return false;
        }
        if (!load_bf16(h.t_down, s.w_down_bf16) || !load_bf16(h.t_up, s.w_up_bf16)) {
            err = "读 output_hc_down / up 失败";
            return false;
        }
    }
    if (s.w_norm_f32.size() != hc_dim) {
        err = "w_norm 的长度与几何不符";
        return false;
    }

    // fin = gr_read(R, output_hc_* , inject=None)：**没有 inject**，故 inject 缓冲不动。
    // w_norm 是 F32，而 gr_read 的 GrTensor 支持逐张力声明精度。
    const GrWeights w{GrTensor{s.w_norm_f32.data(), GrPrecision::kF32},
                      GrTensor{s.w_down_bf16.data(), GrPrecision::kBf16},
                      GrTensor{s.w_up_bf16.data(), GrPrecision::kBf16},
                      GrTensor{}};
    s.fin.assign(static_cast<std::size_t>(h.n_embd), 0.0f);
    std::vector<float> unused_inject(static_cast<std::size_t>(h.gr.hc), 0.0f);
    if (!gr_read(h.gr, w, R, h.rms_eps, s.fin.data(), unused_inject.data(), s.gr, err)) return false;

    // logits = output @ fin（按 output.weight 自己的档分派，与专家/注意力同一份表）。
    s.w_out.resize(static_cast<std::size_t>(h.out.rows * h.out.row_bytes()));
    if (!gguf.read_at(h.t_out->offset, s.w_out.data(), s.w_out.size())) {
        err = "读 output.weight 失败";
        return false;
    }
    s.logits.assign(static_cast<std::size_t>(h.out.rows), 0.0f);
    if (!experts::quantize_input(h.out, s.fin.data(), s.gemv, err)) return false;
    return experts::run_gemv(h.out, s.w_out.data(), s.gemv, s.logits.data(), err);
}

}  // namespace qinfer::dense