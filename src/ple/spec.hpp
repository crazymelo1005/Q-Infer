// 第 1 层那六张 PLE 张力的登记与加载（引擎 `PleWeights` 的对应物）。
//
// 名字、形状与类型取自两份真 pack 的读数（[S-63]）：
//   blk.<L>.ple_key.weight        BF16 [n_embd, hc_dim]（ne0 = n_embd 快，故行距 n_embd）
//   blk.<L>.ple_value.weight      BF16 [n_embd, n_embd]
//   blk.<L>.ple_norm_key.weight   F32  [hc_dim]
//   blk.<L>.ple_norm_query.weight F32  [hc_dim]
//   blk.<L>.ple_norm_conv.weight  F32  [hc_dim]
//   blk.<L>.ple_conv1d.weight     F16  [kConvKernel, hc_dim]（ne0 = 抽头快，即 ggml 原生的 `k + 4·c`）
//
// 六张一张都不在 = 这一层没有 PLE（引擎的说法：`PleRun` 为 null 时 `block_layer_pre` 整块跳过）；
// 在了就要六张都齐、形状与类型都对，否则失败——不静默跳过，也不静默按别的类型读。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "ple/block.hpp"

namespace qinfer::ple {

struct PleSpec {
    int layer = -1;
    bool present = false;
    const artifact::GgufTensorInfo* key = nullptr;
    const artifact::GgufTensorInfo* value = nullptr;
    const artifact::GgufTensorInfo* norm_key = nullptr;
    const artifact::GgufTensorInfo* norm_query = nullptr;
    const artifact::GgufTensorInfo* norm_conv = nullptr;
    const artifact::GgufTensorInfo* conv1d = nullptr;
    bool usable() const {
        return present && key != nullptr && value != nullptr && norm_key != nullptr &&
               norm_query != nullptr && norm_conv != nullptr && conv1d != nullptr;
    }
};

// 读出某一层的 PLE 登记。`n_embd` 与 `hc` 由调用方给（与 [S-54]/[S-57] 同一条口径：几何解不出唯一值）。
bool read_ple_spec(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd, std::uint64_t hc,
                   PleSpec& out, std::string& err);

// 六张张力解成可以直接喂 `ple_block` 的形态。两张投影与卷积核保持位型不展开——`kernels/bf16.hpp` 的
// `bf16_gemv` 与卷积要的就是位型（展开成 f32 会让激活那边的舍入口径无处可挂）。
struct PleBuffers {
    std::vector<std::uint16_t> key_bf16;     // hc_dim × n_embd
    std::vector<std::uint16_t> value_bf16;   // n_embd × n_embd
    std::vector<float> norm_key;             // hc_dim
    std::vector<float> norm_query;           // hc_dim
    std::vector<float> norm_conv;            // hc_dim
    std::vector<std::uint16_t> conv1d_f16;   // kConvKernel × hc_dim
    BlockWeights view() const {
        return BlockWeights{key_bf16.data(), value_bf16.data(), norm_key.data(), norm_query.data(),
                            norm_conv.data(), conv1d_f16.data()};
    }
};

bool load_ple_weights(const artifact::GgufFile& gguf, const PleSpec& spec, PleBuffers& out,
                      std::string& err);

}  // namespace qinfer::ple