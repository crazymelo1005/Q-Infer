#include "dense/layer_map.hpp"

namespace qinfer::dense {

const char* to_string(LayerKind k) {
    switch (k) {
        case LayerKind::kGdn: return "GDN";
        case LayerKind::kQsa: return "QSA";
        case LayerKind::kUnknown: break;
    }
    return "未知";
}

bool scan_layers(const artifact::GgufFile& gguf, int n_layers, std::vector<LayerMapRow>& out,
                 std::string& err) {
    err.clear();
    out.clear();
    if (n_layers <= 0) {
        err = "层数必须为正";
        return false;
    }
    bool ok = true;
    for (int l = 0; l < n_layers; ++l) {
        LayerMapRow row;
        row.layer = l;
        const std::string b = "blk." + std::to_string(l) + ".";
        // 判型只看**注意力那一段**的张力名：GDN 有 attn_qkv、QSA 有 attn_q（两者互斥）。
        const bool has_qkv = gguf.find(b + "attn_qkv.weight") != nullptr;
        const bool has_q = gguf.find(b + "attn_q.weight") != nullptr;
        if (has_qkv && has_q) {
            row.why = "两种注意力张力同时存在（attn_qkv 与 attn_q）";
            ok = false;
        } else if (has_qkv) {
            row.kind = LayerKind::kGdn;
        } else if (has_q) {
            row.kind = LayerKind::kQsa;
        } else {
            row.why = "既没有 attn_qkv 也没有 attn_q——这一层的注意力张力两种都不是";
            ok = false;
        }
        out.push_back(std::move(row));
    }
    if (!ok) {
        err = "有层判不出类型：";
        for (const LayerMapRow& r : out) {
            if (r.kind == LayerKind::kUnknown) {
                err += "层 " + std::to_string(r.layer) + "（" + r.why + "） ";
            }
        }
    }
    return ok;
}

}  // namespace qinfer::dense