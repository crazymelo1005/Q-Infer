// 门控残差（超连接）：不是「一条残差加一个门」，而是 hc 条并行的残差流。
//
// 依据（三处同源，逐字对过）：llama.cpp `src/models/qwen4exp.cpp` 的 `build_hc_mix` / `build_hc_combine`
// （提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，MIT，Copyright (c) 2023-2026 The ggml authors；
// 见 S-53）与参考引擎 `include/strata/kernels/gr.hpp` 的契约头、以及它 `src/kernels/gr_parity.cpp` 里
// 带开关的主机参考实现。两处对同一件事的措辞一致，故这次的转写有一条独立的对照。
//
//   gr_read   xn     = rms_norm(R) * w_norm          （逐流一条 RMS，化简在 n_embd 上——不是整栈一条）
//             lo     = silu( (bf16(xn) · w_downᵀ) / hc )   （除 hc 在 silu **里面**）
//             mixed[d] = (1/hc) Σ_c xn[c,d] · sigmoid( bf16(lo) · w_up[c,d]ᵀ )
//             inject[c] = bf16(xn)[c] · w_inject[c]ᵀ        （w_inject 为 null 时 inject 不动）
//
//   gr_write  w[c] = 2 · sigmoid(inject[c] / hc)
//             R_out[c,d] = R[c,d] + block_out[d] · w[c]      （block_out 对每条流是同一份，只有权重逐流）
//
// 六处「读过去很顺但会给出可信的错值」的地方，每一处都在测试里按错的读法算一遍并要求它与对的读法有
// 明显差别（与上游 parity 测试同样的做法）：逐流的 RMS 而不是整栈一条；除 hc 在 silu 里；lo 上取
// silu 而门上取 sigmoid（两个都在同一段里，写反是常见笔误）；按流求**平均**而不是求和；激活必须按
// bf16 舍入（这是参考实现的默认契约，不是「因为权重是 bf16」）；以及 2·sigmoid 把门心定在 1，使零注入
// 退化成普通残差相加。
//
// 权重在包里是 bf16（f32 的高 16 位），按清单原样存，不做置换：w_down 行主序 (hc_lr, hc·n_embd)、
// w_up 行主序 (hc·n_embd, hc_lr)、w_inject 行主序 (hc, hc·n_embd)。w_norm 已被转换器折叠成 (1 + w)，
// 故直接乘、不再加一。真模型几何：n_embd = 2560、hc = 4、hc_lr = 320。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::dense {

// 超连接的几何。hc 条流各 n_embd 宽，down/up 的瓶颈秩是 hc_lr。
struct GrShapes {
    std::uint64_t n_embd = 0;
    std::uint64_t hc = 0;
    std::uint64_t hc_lr = 0;

    bool sane() const { return n_embd != 0 && hc != 0 && hc_lr != 0; }
};

// 一次 gr_read 的四组权重（bf16 位型）。w_inject 为 null 表示这是最后那个 mixer：它没有回写门。
struct GrWeights {
    const std::uint16_t* w_norm = nullptr;    // hc·n_embd
    const std::uint16_t* w_down = nullptr;    // hc_lr × hc·n_embd
    const std::uint16_t* w_up = nullptr;      // hc·n_embd × hc_lr
    const std::uint16_t* w_inject = nullptr;  // hc × hc·n_embd，可为 null
};

// 复用的中间缓冲，避免每次调用分配。
struct GrScratch {
    std::vector<float> xn;   // hc·n_embd，未舍入的归一化流（gated 平均要的是未舍入值）
    std::vector<float> act;  // hc·n_embd，bf16(xn)
    std::vector<float> lo;   // hc_lr
    std::vector<float> lq;   // hc_lr，bf16(lo)
};

// 一条流的 RMS。w_norm 存的已经是 (1 + w)。
float rms_scale(const float* row, std::uint64_t n, float eps);

// R 是 (hc, n_embd)，n_embd 最快（第 c 条流从 c·n_embd 开始）。mixed 与 inject 由调用方给缓冲
// （mixed 长 n_embd、inject 长 hc）。w_inject 为 null 时 inject 原样不动。
bool gr_read(const GrShapes& s, const GrWeights& w, const float* R, float eps, float* mixed,
             float* inject, GrScratch& scratch, std::string& err);

// R_out[c,d] = R[c,d] + block_out[d] · 2·sigmoid(inject[c] / hc)。block_out 逐条流都是同一份。
bool gr_write(const GrShapes& s, const float* R, const float* block_out, const float* inject,
              float* R_out, std::string& err);

}  // namespace qinfer::dense