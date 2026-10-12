// 见 block.hpp 的口径说明（公式、三处错读法、与 dense/norm.hpp 的两处差别都记在那里）。
#include "ple/block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels/bf16.hpp"
#include "kernels/fp16.hpp"

namespace qinfer::ple {

namespace {

// ggml 的 silu：`x / (1 + e^{-x})`。**不是** `x·sigmoid(x)`——那一条是本仓库 GR/GDN 两处转录的写法，
// 两者数学同值、末位不同；引擎的 PLE 主机参考逐字写的就是这一个（ple.cu 的 `silu_f`）。
inline float silu_ggml(float x) { return x / (1.0f + std::exp(-x)); }

// 逐流的 RMS 归一（引擎的 `grouped_norm`：一条流一个块，约化轴是 n_embd，之后逐维乘 gamma）。
// 两处不能改的算术，都是引擎注释点名的：
//   * 平方和逐项 `(double)(x·x)`——乘积先在 f32 里舍一遍再放宽到 double（先放宽再相乘是另一个值）；
//   * 尺度是 f32 的 `1/sqrtf(mean + eps)`，eps 加在均值上。
// `x` 与 `y` 可以同址（先全读一遍再写）。
void grouped_norm(const float* x, const float* gamma, std::uint64_t n_embd, std::uint64_t hc,
                  float* y) {
    for (std::uint64_t c = 0; c < hc; ++c) {
        const float* xc = x + c * n_embd;
        const float* gc = gamma + c * n_embd;
        float* yc = y + c * n_embd;
        double sum = 0.0;
        for (std::uint64_t d = 0; d < n_embd; ++d) {
            const float xv = xc[d];
            sum += static_cast<double>(xv * xv);
        }
        const float mean = static_cast<float>(sum / static_cast<double>(n_embd));
        const float scale = 1.0f / std::sqrt(mean + kPleRmsEps);
        for (std::uint64_t d = 0; d < n_embd; ++d) yc[d] = xc[d] * scale * gc[d];
    }
}

// 门：逐流的点积 → 带符号的平方根 → sigmoid。三处细节：累加是 `(double)(k·q)`（乘积先舍 f32）、
// 除以 `sqrt(n_embd)` 而不是乘预先算好的倒数、`sign(0) = 0`（于是零点那一路的门正好 0.5）。
void gate_of(const float* key, const float* query, std::uint64_t n_embd, std::uint64_t hc,
             float* gate) {
    for (std::uint64_t c = 0; c < hc; ++c) {
        const float* kc = key + c * n_embd;
        const float* qc = query + c * n_embd;
        double acc = 0.0;
        for (std::uint64_t d = 0; d < n_embd; ++d) acc += static_cast<double>(kc[d] * qc[d]);
        const float s = static_cast<float>(acc) / std::sqrt(static_cast<float>(n_embd));
        const float mag = std::sqrt(std::fmax(std::fabs(s), 1e-6f));
        const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
        gate[c] = 1.0f / (1.0f + std::exp(-(sgn * mag)));
    }
}

// 4 抽头膨胀卷积 + silu。tap k 读历史里第 `kHistRows - (kConvKernel-1-k)·ngram_size` 行：k=0 读最远
// 那一行、k=3 读当前这一条 normalized（它不在历史里，是唯一一个边界）。核是 ggml 原生的 `k + 4·c`。
void conv_of(const float* normalized, const float* hist, const std::uint16_t* conv1d_f16,
             std::uint64_t hc_dim, float* conv) {
    for (std::uint64_t c = 0; c < hc_dim; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < kConvKernel; ++k) {
            const int row = kHistRows - (kConvKernel - 1 - k) * kernels::kNgramSize;
            const float v =
                (row == kHistRows)
                    ? normalized[c]
                    : hist[static_cast<std::size_t>(row) + static_cast<std::size_t>(kHistRows) * c];
            const std::size_t ki = static_cast<std::size_t>(k) + static_cast<std::size_t>(kConvKernel) * c;
            acc += kernels::f16_bits_to_f32(conv1d_f16[ki]) * v;
        }
        conv[c] = silu_ggml(acc);
    }
}

// 两条入口共用的那几项：几何、权重是否齐、以及残差有没有去处（它必须可写）。
bool geometry_ok(const BlockOut& out, const BlockWeights& w, const BlockShapes& g, std::string& err) {
    if (out.result == nullptr) {
        err = "PLE 块没给 result（残差的新值没有去处）";
        return false;
    }
    if (!g.sane()) {
        err = "PLE 的几何不成立（n_embd 或 hc 是零）";
        return false;
    }
    if (!w.complete()) {
        err = "PLE 的六张权重不全";
        return false;
    }
    return true;
}

}  // namespace

bool ple_block_projected(float* key_raw, const float* value_raw, float* hidden, const float* hist,
                         const BlockWeights& w, const BlockShapes& g, BlockOut& out,
                         BlockScratch& s, std::string& err) {
    err.clear();
    if (key_raw == nullptr || value_raw == nullptr || hidden == nullptr || hist == nullptr) {
        err = "PLE 块的输入指针不完整";
        return false;
    }
    if (!geometry_ok(out, w, g, err)) return false;
    if (!s.resize(g)) {
        err = "PLE 的 scratch 配不出来";
        return false;
    }
    const std::size_t n_embd = static_cast<std::size_t>(g.n_embd);
    const std::size_t hc_dim = static_cast<std::size_t>(g.hc_dim());

    grouped_norm(key_raw, w.norm_key, g.n_embd, g.hc, key_raw);  // 就地，与引擎的主机参考一致
    grouped_norm(hidden, w.norm_query, g.n_embd, g.hc, s.query.data());
    gate_of(key_raw, s.query.data(), g.n_embd, g.hc, s.gate.data());
    for (std::size_t i = 0; i < hc_dim; ++i) {
        s.gated[i] = value_raw[i % n_embd] * s.gate[i / n_embd];
    }
    grouped_norm(s.gated.data(), w.norm_conv, g.n_embd, g.hc, s.normalized.data());
    conv_of(s.normalized.data(), hist, w.conv1d_f16, g.hc_dim(), s.conv.data());
    for (std::size_t i = 0; i < hc_dim; ++i) {
        out.result[i] = hidden[i] + s.gated[i] + s.conv[i];
    }

    auto put = [](float* dst, const float* src, std::size_t len) {
        if (dst != nullptr) std::copy(src, src + len, dst);
    };
    put(out.key, key_raw, hc_dim);
    put(out.value, value_raw, n_embd);
    put(out.gate, s.gate.data(), static_cast<std::size_t>(g.hc));
    put(out.gated, s.gated.data(), hc_dim);
    put(out.normalized, s.normalized.data(), hc_dim);
    put(out.conv, s.conv.data(), hc_dim);
    return true;
}

bool ple_block(const float* emb, float* hidden, const float* hist, const BlockWeights& w,
               const BlockShapes& g, BlockOut& out, BlockScratch& s, std::string& err) {
    err.clear();
    if (emb == nullptr || hidden == nullptr || hist == nullptr) {
        err = "PLE 块没给 emb（表行拼出的那 2560）或残差/历史";
        return false;
    }
    if (!geometry_ok(out, w, g, err)) return false;
    if (!s.resize(g)) {
        err = "PLE 的 scratch 配不出来";
        return false;
    }
    // 两个投影都是 BF16 的张力：**激活也按 bf16 舍入**（权重是 bf16 时 ggml 会把激活转到权重的
    // vec_dot_type，[S-44]），累加在 float 上——`kernels/bf16.hpp` 的 bf16_gemv 就是这个口径。
    // key 的投影直接落在 `s.key` 上，随后就地归一（省一份 hc_dim 的缓冲）。
    kernels::bf16_gemv(w.key_bf16, g.hc_dim(), g.n_embd, emb, s.key.data());
    kernels::bf16_gemv(w.value_bf16, g.n_embd, g.n_embd, emb, s.value.data());
    return ple_block_projected(s.key.data(), s.value.data(), hidden, hist, w, g, out, s, err);
}

void ple_history_advance(float* hist, const float* normalized, const BlockShapes& g) {
    if (hist == nullptr || normalized == nullptr) return;
    for (std::size_t c = 0; c < static_cast<std::size_t>(g.hc_dim()); ++c) {
        float* column = hist + c * kHistRows;
        for (int r = 0; r + 1 < kHistRows; ++r) column[r] = column[r + 1];
        column[kHistRows - 1] = normalized[c];
    }
}

}  // namespace qinfer::ple