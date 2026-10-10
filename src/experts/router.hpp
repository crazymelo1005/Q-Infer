// 路由门控：logits = ffn_gate_inp · bf16(x)，然后「全专家 softmax → 稳定降序取前 k → 再归一化」。
//
// 口径取自参考引擎自身的主机参考实现（`src/kernels/router_top10_parity.cpp` 的 `reference()` 与
// `src/core/layer.cpp` 的 routing 注释；登记见 S-44）。三处必须照做、不能凭常识改：
//   1. softmax 是「全专家」的、且用双精度算（减最大值后再 exp）；
//   2. 排序是「稳定降序」，同值保持索引序（即同值取小索引）；
//   3. 归一化的分母是前 k 个概率之和，且**下限钳在 2^-14**（本模型的几何下钳不到，但口径要写对）。
// 另有一处容易吃暗亏：路由器的激活要转 bf16，不是 fp16——引擎自注改用 fp16 会引入 8.100e-03 的
// 误差并翻掉选择。
//
// 注意口径的边界：引擎那份参考是「按同一份规格另写的」，它自己声明「不是与 llama.cpp 对拍」。
// 因此与它一致只能说明两边同规格，不能推出与 llama.cpp 一致。这条差距照写在这里，不隐含。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::experts {

// 引擎的归一化分母下限（2^-14）。本模型的几何下取不到，写在这里是为了口径完整。
inline constexpr double kRenormClamp = 6.103515625e-05;

struct RouterSpec {
    std::uint64_t hidden = 0;
    std::uint64_t n_expert = 0;
    std::uint32_t type = 0;
    bool ok = false;
};

// 从 GGUF 取某一层的路由器矩阵形状。要求存在、是二维、且是 BF16（30）。
bool read_router_spec(const artifact::GgufFile& gguf, int layer, RouterSpec& spec, std::string& err);

// logits = W · bf16(x)：权重与激活都按 bf16 读，累加在 float 上（x 先逐元素转 bf16）。
void router_logits(const std::uint16_t* w_bf16, int n_expert, int hidden, const float* x,
                   float* logits);

struct RouterScratch {
    std::vector<double> p;
    std::vector<int> idx;
    std::vector<std::uint16_t> x_bf16;
    std::vector<float> logits;
};

// 全专家 softmax（双精度）→ 稳定降序（同值小索引在前）→ 取前 k → 按前 k 之和归一化（下限 2^-14）。
void router_topk(const float* logits, int n_expert, int k, RouterScratch& scratch, int* ids,
                 float* weights);

// 一步到位：投影 + 门控。weights 之和为 1，ids 互不相同且按权重降序。
bool route(const std::uint16_t* w_bf16, int n_expert, int hidden, int k, const float* x,
           RouterScratch& scratch, int* ids, float* weights, std::string& err);

}  // namespace qinfer::experts