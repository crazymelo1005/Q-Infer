// 全文的层表：48 层里哪 36 层是 GDN、哪 12 层是 QSA，以及每层能不能按已有两套构建器建起来。
//
// 逐层**从文件读**判型，不写死分布：QSA 层有 `attn_q`、没有 `attn_qkv`；GDN 层反之（[S-54]）。
// 真模型实测是 QSA 落在 3 / 7 / 11 / … / 47 共 12 层、其余 36 层 GDN——但那是**这份文件的分布**，
// 换一份 pack 就可能不同（档位与层型都随 pack 变，S-36 的教训），故这里只核「每层都能被判成一类」。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::dense {

enum class LayerKind : std::uint8_t { kGdn, kQsa, kUnknown };
const char* to_string(LayerKind k);

// 一层的一行：层号 + 判型 + 判错的理由（判不出来时）。
struct LayerMapRow {
    int layer = -1;
    LayerKind kind = LayerKind::kUnknown;
    std::string why;  // 判不出来时写清「两种层的张力都缺/都有」之类
};

// 扫描 0..n_layers-1。任一层的判型失败会记进该行并让整体返回假（不静默跳过）。
bool scan_layers(const artifact::GgufFile& gguf, int n_layers, std::vector<LayerMapRow>& out,
                 std::string& err);

}  // namespace qinfer::dense