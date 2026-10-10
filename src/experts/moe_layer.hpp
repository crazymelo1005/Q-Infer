// 一层的 MoE：路由出 top-k 专家 → 逐个前馈 → 按路由权重累加。
//
// 这一步把已有的两块接起来：`experts/router`（选专家与权重）与 `experts/ffn`（三矩阵组装）。
// 它同时负责从 GGUF 里按「逐专家偏移」取字节——三维张量 [cols, rows, experts] 的专家步长是
// `rows × row_bytes()`，三个矩阵各自如此；偏移算错会静默读到别的专家的权重，故测试里专门有一条。
//
// 本实现是主机侧参考路径：每次调用按需从文件读路由矩阵与选中专家的三矩阵，不做缓存。
// 专家缓存（槽位/替换）是独立子系统，接在 `storage/page_table` 之上，不在这里。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"
#include "experts/router.hpp"

namespace qinfer::experts {

// 一层 MoE 的静态描述：分派表 + 张力登记 + 逐专家字节步长。构建一次，逐 token 复用。
// 里面的指针指向 GgufFile 内部的张力表，故 GgufFile 在 MoeLayer 存活期间不得重新 open。
struct MoeLayer {
    int layer = -1;
    int top_k = 0;
    LayerSpec spec;
    SharedSpec shared;   // 这一层的共享专家；present=false 表示没有
    RouterSpec router_spec;
    const artifact::GgufTensorInfo* gate = nullptr;
    const artifact::GgufTensorInfo* up = nullptr;
    const artifact::GgufTensorInfo* down = nullptr;
    const artifact::GgufTensorInfo* router = nullptr;
    const artifact::GgufTensorInfo* s_gate = nullptr;
    const artifact::GgufTensorInfo* s_up = nullptr;
    const artifact::GgufTensorInfo* s_down = nullptr;
    const artifact::GgufTensorInfo* s_gate_inp = nullptr;  // 1 维 bf16 标量门，长度 hidden
    std::uint64_t gate_expert_bytes = 0;
    std::uint64_t up_expert_bytes = 0;
    std::uint64_t down_expert_bytes = 0;
    std::uint64_t hidden = 0;
    std::uint64_t n_expert = 0;
};

// 校验一层的三个专家矩阵与路由器：几何成立、有内核、路由是 BF16、维度自洽（gate.cols == hidden == down.rows、
// gate.rows == ffn == down.cols）。任何一条不成立都返回 false 并写 err。
bool build_moe_layer(const artifact::GgufFile& gguf, int layer, int top_k, MoeLayer& out,
                     std::string& err);

struct MoeScratch {
    RouterScratch router;
    FfnScratch ffn;
    std::vector<std::uint16_t> router_w;  // 路由矩阵（bf16）
    std::vector<std::uint8_t> gate, up, down;
    std::vector<float> expert_out;
    std::vector<int> ids;
    std::vector<float> weights;
    // 共享专家：标量门的 bf16 权重、三矩阵的字节区、以及它自己的中间缓冲。
    std::vector<std::uint16_t> s_gate_inp;
    std::vector<std::uint8_t> s_gate, s_up, s_down;
    FfnScratch s_ffn;
    std::vector<float> shared_out;
};

// 跑一层：读路由矩阵 → 路由出 top-k → 逐个读该专家的三矩阵并前馈 → 按权重累加 → 有共享专家则再加。
// 组合口径是 `y = Σ wᵢ·expertᵢ + shared`：路由侧按权重、共享侧**不加权**直接相加（[S-50]）。
// out 长 hidden，覆盖写（MoE 输出本身）。若 ids 与 weights 非空则回填路由结果，便于核对。
bool run_moe_layer(const artifact::GgufFile& gguf, const MoeLayer& layer, const float* x, float* out,
                   MoeScratch& scratch, std::string& err, int* ids_out = nullptr,
                   float* weights_out = nullptr);

}  // namespace qinfer::experts