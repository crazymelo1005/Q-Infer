#include "experts/ffn.hpp"

#include "scheduling/thread_pool.hpp"

#include "kernels/bf16.hpp"
#include "kernels/iq1m.hpp"
#include "kernels/iq2s.hpp"
#include "kernels/iq2xs.hpp"
#include "kernels/iq2xs_dot.hpp"
#include "kernels/iq2xxs.hpp"
#include "kernels/iq3s.hpp"
#include "kernels/iq3xxs.hpp"
#include "kernels/iq4nl.hpp"
#include "kernels/iq4xs.hpp"
#include "kernels/q2_0.hpp"
#include "kernels/q6k.hpp"
#include "kernels/q8_0.hpp"

#include <cmath>

namespace qinfer::experts {

namespace {

// 按矩阵的档选激活量化方式并量化 x。返回实际使用的那一组块数（供核验）。
}  // namespace

bool quantize_input(const MatrixSpec& m, const float* x, FfnScratch& s, std::string& err) {
    const ActFormat act = activation_format(m.format);
    const std::uint64_t blocks = m.row_blocks();
    if (act == ActFormat::kQ8K) {
        s.act_k.resize(static_cast<std::size_t>(blocks));
        kernels::q8k_quantize_row(x, s.act_k.data(), static_cast<int>(blocks));
        return true;
    }
    if (act == ActFormat::kQ8_0) {
        // Q8_0 的块是 32 个元素，与 32/64 值块的权重档对齐（一个 Q2_0 块配两个 Q8_0 块）。
        const std::uint64_t q80_blocks = m.cols / static_cast<std::uint64_t>(kernels::kQ80BlockElems);
        s.act_32.resize(static_cast<std::size_t>(q80_blocks));
        kernels::q8_0_quantize_row(x, s.act_32.data(), static_cast<int>(q80_blocks));
        return true;
    }
    err = std::string("格式 ") + format_name(m.format) + " 没有激活搭档，无法做定点点积";
    return false;
}

namespace {

// 某个档的那一次调用：只看「从 w 这块字节、out 这块输出的第 0 行起算 rows 行」，故按行切块时把 w
// 与 out 各偏移 r0 行即可——每一行算的东西完全没变（这是逐位一致的前提）。
bool gemv_rows(Format f, const std::uint8_t* w, int rows, int blocks, const FfnScratch& s,
               float* out, std::string& err) {
    switch (f) {
        case Format::kIq2S:
            kernels::iq2s_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq2Xs:
            kernels::iq2xs_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq2Xxs:
            kernels::iq2xxs_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq1M:
            kernels::iq1m_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kQ6K:
            kernels::q6k_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq3S:
            kernels::iq3s_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq4Xs:
            kernels::iq4xs_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kIq3Xxs:
            kernels::iq3xxs_gemv_q8k(w, rows, blocks, s.act_k.data(), out);
            return true;
        case Format::kQ2_0:
            kernels::q2_0_gemv_q8_0(w, rows, blocks, s.act_32.data(), out);
            return true;
        case Format::kIq4Nl:
            kernels::iq4nl_gemv_q8_0(w, rows, blocks, s.act_32.data(), out);
            return true;
        case Format::kQ8_0:
            kernels::q8_0_gemv_q8_0(w, rows, blocks, s.act_32.data(), out);
            return true;
        default:
            err = std::string("格式 ") + format_name(f) + " 尚无内核，不能参与组装";
            return false;
    }
}

// 有没有可分派的档。先判一次，是为了让并行那侧永远不会去写 err（那里多个线程同时写一个 string
// 就是数据竞争）。
bool dispatchable(Format f) {
    switch (f) {
        case Format::kIq2S:
        case Format::kIq2Xs:
        case Format::kIq2Xxs:
        case Format::kIq1M:
        case Format::kQ6K:
        case Format::kIq3S:
        case Format::kIq4Xs:
        case Format::kIq3Xxs:
        case Format::kQ2_0:
        case Format::kIq4Nl:
        case Format::kQ8_0:
            return true;
        default:
            return false;
    }
}

}  // namespace

bool run_gemv(const MatrixSpec& m, const std::uint8_t* w, FfnScratch& s, float* out,
              std::string& err) {
    const int rows = static_cast<int>(m.rows);
    const int blocks = static_cast<int>(m.row_blocks());
    if (!dispatchable(m.format)) {
        err = std::string("格式 ") + format_name(m.format) + " 尚无内核，不能参与组装";
        return false;
    }
    // 按行并行：只切行、不切归约维，故与单线程逐位相同（engine §16 那条口径）。池子默认关。
    scheduling::ThreadPool* p = scheduling::pool();
    if (p == nullptr || rows <= 1) return gemv_rows(m.format, w, rows, blocks, s, out, err);
    const std::uint64_t row_bytes = m.row_bytes();
    p->run_rows(static_cast<std::uint64_t>(rows), /*grain=*/8,
                [&](std::uint64_t b, std::uint64_t e) {
                    gemv_rows(m.format, w + b * row_bytes, static_cast<int>(e - b), blocks, s,
                              out + b, err);
                });
    return true;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

bool run_bf16_gemv(const std::uint16_t* w, std::uint64_t rows, std::uint64_t cols, const float* x,
                   float* out, std::string& err) {
    if (w == nullptr || x == nullptr || out == nullptr) {
        err = "bf16 投影的参数为空";
        return false;
    }
    // 激活舍一遍就够（与行数无关，为切行服务；单线程上并不比逐行各算一遍快，编译器本来就提），
    // 再按行并行。两处都不动任何元素的归约顺序 → 与单线程逐位相同。
    std::vector<std::uint16_t> xb(static_cast<std::size_t>(cols));
    kernels::bf16_round_row(x, cols, xb.data());
    scheduling::ThreadPool* p = scheduling::pool();
    if (p == nullptr || rows <= 1) {
        kernels::bf16_gemv_rows(w, rows, cols, xb.data(), out);
        return true;
    }
    p->run_rows(rows, /*grain=*/16, [&](std::uint64_t b, std::uint64_t e) {
        kernels::bf16_gemv_rows(w + b * cols, e - b, cols, xb.data(), out + b);
    });
    return true;
}

bool expert_ffn(const LayerSpec& spec, const ExpertWeights& w, const float* x, FfnScratch& scratch,
                float* out, std::string& err) {
    if (!spec.usable()) {
        err = "层的矩阵里有不可用项（几何不成立或没有内核）";
        return false;
    }
    if (!w.complete() || x == nullptr || out == nullptr) {
        err = "参数为空";
        return false;
    }
    // gate 与 up 共用同一组激活，故两者的激活档必须一致；不一致说明分派表或调用方有问题。
    if (activation_format(spec.gate.format) != activation_format(spec.up.format)) {
        err = std::string("gate 的激活档是 ") +
              std::to_string(static_cast<int>(activation_format(spec.gate.format))) + "，up 是 " +
              std::to_string(static_cast<int>(activation_format(spec.up.format))) + "，不一致";
        return false;
    }

    const std::size_t ffn = static_cast<std::size_t>(spec.gate.rows);
    scratch.gate.resize(ffn);
    scratch.up.resize(ffn);
    scratch.h.resize(ffn);

    if (!quantize_input(spec.gate, x, scratch, err)) return false;
    if (!run_gemv(spec.gate, w.gate, scratch, scratch.gate.data(), err)) return false;
    if (!run_gemv(spec.up, w.up, scratch, scratch.up.data(), err)) return false;

    for (std::size_t j = 0; j < ffn; ++j) {
        scratch.h[j] = silu(scratch.gate[j]) * scratch.up[j];
    }

    if (!quantize_input(spec.down, scratch.h.data(), scratch, err)) return false;
    if (!run_gemv(spec.down, w.down, scratch, out, err)) return false;
    return true;
}

float shared_scalar_gate(const std::uint16_t* w_bf16, const float* x, std::uint64_t n) {
    float acc = 0.0f;
    for (std::uint64_t j = 0; j < n; ++j) {
        acc += kernels::bf16_bits_to_f32(w_bf16[j]) *
               kernels::bf16_bits_to_f32(kernels::f32_to_bf16_bits(x[j]));
    }
    return 1.0f / (1.0f + std::exp(-acc));
}

// 权重与激活都不舍 bf16（F32×F32），累加在 double 上——与 bf16 那条的差别只在这一点（ADR-010）。
float shared_scalar_gate_f32(const float* w_f32, const float* x, std::uint64_t n) {
    double dot = 0.0;
    for (std::uint64_t j = 0; j < n; ++j) {
        dot += static_cast<double>(w_f32[j]) * static_cast<double>(x[j]);
    }
    return static_cast<float>(1.0 / (1.0 + std::exp(-dot)));
}

bool shared_expert_ffn(const SharedSpec& spec, const ExpertWeights& w,
                       const void* gate_inp, const float* x, float* out,
                       FfnScratch& scratch, std::string& err) {
    if (!spec.present) {
        err = "这一层没有共享专家";
        return false;
    }
    if (!spec.gate_inp || gate_inp == nullptr) {
        err = "共享专家的标量门缺失（或不是 1 维 bf16 / f32）";
        return false;
    }
    // 三矩阵复用 expert_ffn：它已经按 gate/up 的档各自量化激活、在 gate 上做 SwiGLU、并把中间量
    // 按 down 的档量化。这里只把共享专家的三段拼成一个 LayerSpec 视图。
    LayerSpec view;
    view.layer = -1;
    view.gate = spec.gate;
    view.up = spec.up;
    view.down = spec.down;
    if (!view.usable()) {
        err = "共享专家的三矩阵有档不可用（几何不成立或没有内核）";
        return false;
    }
    if (!expert_ffn(view, w, x, scratch, out, err)) return false;
    // 档随张力声明走（ADR-010）：bf16 那条「两个操作数都舍 bf16」，f32 那条 F32×F32。
    const float g = spec.gate_inp_type == 30
                        ? shared_scalar_gate(static_cast<const std::uint16_t*>(gate_inp), x,
                                             spec.gate.cols)
                        : shared_scalar_gate_f32(static_cast<const float*>(gate_inp), x,
                                                 spec.gate.cols);
    const std::size_t hidden = static_cast<std::size_t>(spec.down.rows);
    for (std::size_t i = 0; i < hidden; ++i) out[i] *= g;
    return true;
}

bool moe_ffn(const LayerSpec& spec, const ExpertWeights* ws, const float* route_weight, int n,
             const float* x, float* out, FfnScratch& scratch, std::string& err) {
    if (n <= 0 || ws == nullptr || route_weight == nullptr) {
        err = "没有选中任何专家";
        return false;
    }
    const std::size_t hidden = static_cast<std::size_t>(spec.down.rows);
    scratch.out_tmp.resize(hidden);
    for (int e = 0; e < n; ++e) {
        if (!expert_ffn(spec, ws[e], x, scratch, scratch.out_tmp.data(), err)) return false;
        for (std::size_t i = 0; i < hidden; ++i) {
            out[i] += route_weight[e] * scratch.out_tmp[i];
        }
    }
    return true;
}

}  // namespace qinfer::experts