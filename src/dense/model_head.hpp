// 模型的两端：残差栈的初值，以及收尾那个 mixer 加输出头（logits）。
//
// 依据 [S-61]（两处都是引擎源码里的原话）：
//
//   hc 的初值：残差是**同一个向量在 hc 条流上的副本**——「a real sequence's first token is the embedding
//   broadcast to every stream」。故 `R[c·n_embd + :] = x`。
//
//   收尾：`fin = GR.gr_read(R, output_hc_*, inject=None, rms_eps); logits = output @ fin`。
//   注意 **inject=None**（`output_hc_*` 没有 inject 张力）——`gr_read` 在缺 inject 时不动那个缓冲，
//   这正是门控残差那一块已经支持的情形（[S-53]）。这条也顺带说明：**收尾那条归一就是 gr_read 里的
//   逐流 RMSNorm×gamma 加门控平均**，循环之外没有第二处最终 norm。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/gated_residual.hpp"
#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"

namespace qinfer::dense {

// 把 x 广播到 hc 条流上（残差的初值）。R 长 hc·n_embd。
void hc_broadcast_init(const float* x, std::uint64_t n_embd, std::uint64_t hc, float* R);

// 收尾那一段需要的东西：三张 output_hc_* 张力 + 输出投影。
struct ModelHead {
    GrShapes gr;                    // n_embd / hc / hc_lr
    std::uint64_t n_embd = 0;
    experts::MatrixSpec out;        // output.weight：输入维 = n_embd、输出维 = 词表
    const artifact::GgufTensorInfo* t_norm = nullptr;
    const artifact::GgufTensorInfo* t_down = nullptr;
    const artifact::GgufTensorInfo* t_up = nullptr;
    const artifact::GgufTensorInfo* t_out = nullptr;
    float rms_eps = 1e-6f;
};

bool build_model_head(const artifact::GgufFile& gguf, std::uint64_t n_embd, std::uint64_t hc,
                      std::uint64_t hc_lr, ModelHead& out, std::string& err);

struct HeadScratch {
    std::vector<std::uint8_t> w_norm_raw, w_down, w_up, w_out;
    std::vector<std::uint16_t> w_norm, w_down_bf16, w_up_bf16;  // 按声明精度读，这里存 bf16 视图
    std::vector<float> w_norm_f32;
    GrScratch gr;
    experts::FfnScratch gemv;
    std::vector<float> fin;
    std::vector<float> logits;
};

// R 是 (hc, n_embd)：先 gr_read（inject=None）得 fin，再 `logits = output @ fin`。
// 词表很大时 logits 也很大（真模型 248,320），调用方可以只关心前若干个。
bool run_model_head(const artifact::GgufFile& gguf, const ModelHead& h, const float* R,
                    HeadScratch& s, std::string& err);

}  // namespace qinfer::dense