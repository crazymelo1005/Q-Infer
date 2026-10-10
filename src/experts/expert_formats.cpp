// 块几何转录自 llama.cpp 检出内 gguf-py 的 GGML_QUANT_SIZES（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d；
// 来源登记见 S-42）——即「每块多少元素、每块多少字节」，从中可推出每个档的 bpw：
//   Q2_0 (64, 18) 2.25 · Q4_0 (32, 18) 4.5 · Q5_0 (32, 22) 5.5 · Q8_0 (32, 34) 8.5
//   Q2_K (256, 84) 2.625 · Q3_K (256, 110) 3.4375 · Q4_K (256, 144) 4.5 · Q5_K (256, 176) 5.5
//   Q6_K (256, 210) 6.5625 · IQ1_S (256, 50) 1.5625 · IQ1_M (256, 56) 1.75
//   IQ2_XXS (256, 66) 2.0625 · IQ2_XS (256, 74) 2.3125 · IQ2_S (256, 82) 2.5625
//   IQ3_XXS (256, 98) 3.0625 · IQ3_S (256, 110) 3.4375 · IQ4_NL (32, 18) 4.5 · IQ4_XS (256, 136) 4.25
// 激活搭档的对应关系来自各自 generic 点积核（QK4_NL == QK8_0，QK2_0 = 64 配两个 Q8_0，
// 256 值块的 K-quant 与 IQ 档配 Q8_K）：S-37、S-40、S-41。
#include "experts/expert_formats.hpp"

#include <cctype>

namespace qinfer::experts {

namespace {

struct Geometry {
    int elems;
    int bytes;
    bool kernel;      // 本仓库是否已实现该档的权重内核
    ActFormat act;
};

Geometry geometry(Format f) {
    switch (f) {
        // 非块状：元素即单位。
        case Format::kF32:  return {1, 4, false, ActFormat::kNone};
        case Format::kF16:  return {1, 2, false, ActFormat::kNone};
        case Format::kBf16: return {1, 2, false, ActFormat::kNone};
        // 32 值块：配 Q8_0。
        case Format::kQ2_0:  return {64, 18, true, ActFormat::kQ8_0};
        case Format::kQ4_0:  return {32, 18, false, ActFormat::kQ8_0};
        case Format::kQ5_0:  return {32, 22, false, ActFormat::kQ8_0};
        case Format::kQ8_0:  return {32, 34, false, ActFormat::kQ8_0};  // 本仓库只把它当激活档
        case Format::kIq4Nl: return {32, 18, true, ActFormat::kQ8_0};
        // 256 值块：配 Q8_K。
        case Format::kQ2K:    return {256, 84, false, ActFormat::kQ8K};
        case Format::kQ3K:    return {256, 110, false, ActFormat::kQ8K};
        case Format::kQ4K:    return {256, 144, false, ActFormat::kQ8K};
        case Format::kQ5K:    return {256, 176, false, ActFormat::kQ8K};
        case Format::kQ6K:    return {256, 210, true, ActFormat::kQ8K};
        case Format::kIq1S:   return {256, 50, false, ActFormat::kQ8K};
        case Format::kIq1M:   return {256, 56, true, ActFormat::kQ8K};
        case Format::kIq2Xxs: return {256, 66, true, ActFormat::kQ8K};
        case Format::kIq2Xs:  return {256, 74, true, ActFormat::kQ8K};
        case Format::kIq2S:   return {256, 82, true, ActFormat::kQ8K};
        case Format::kIq3Xxs: return {256, 98, false, ActFormat::kQ8K};
        case Format::kIq3S:   return {256, 110, false, ActFormat::kQ8K};
        case Format::kIq4Xs:  return {256, 136, false, ActFormat::kQ8K};
        case Format::kUnknown: break;
    }
    return {0, 0, false, ActFormat::kNone};
}

// 从 "blk.<digits>.<rest>" 里取 <digits>；不是这个形状就返回 -1。
int layer_of(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    std::size_t i = 4;
    int n = 0;
    bool any = false;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) {
        n = n * 10 + (name[i] - '0');
        any = true;
        ++i;
    }
    if (!any || i >= name.size() || name[i] != '.') return -1;
    return n;
}

MatrixSpec make_spec(const artifact::GgufTensorInfo& t, std::string& why) {
    MatrixSpec s;
    s.format = format_from_type_code(t.type);
    if (s.format == Format::kUnknown) {
        why = "类型码 " + std::to_string(t.type) + " 不认识";
        return s;
    }
    if (t.dims.size() < 2 || t.dims.size() > 3) {
        why = "维度数 " + std::to_string(t.dims.size()) + " 不是 2 或 3";
        return s;
    }
    s.cols = t.dims[0];
    s.rows = t.dims[1];
    s.experts = (t.dims.size() == 3) ? t.dims[2] : 1;
    if (!s.geometry_ok()) {
        why = std::string(format_name(s.format)) + " 的块元素数 " +
              std::to_string(block_elems(s.format)) + " 整除不了输入维 " + std::to_string(s.cols);
    } else if (!has_kernel(s.format)) {
        why = std::string(format_name(s.format)) + " 尚无内核";
    }
    return s;
}

}  // namespace

Format format_from_type_code(std::uint32_t code) {
    switch (code) {
        case 0: return Format::kF32;
        case 1: return Format::kF16;
        case 2: return Format::kQ4_0;
        case 6: return Format::kQ5_0;
        case 8: return Format::kQ8_0;
        case 10: return Format::kQ2K;
        case 11: return Format::kQ3K;
        case 12: return Format::kQ4K;
        case 13: return Format::kQ5K;
        case 14: return Format::kQ6K;
        case 16: return Format::kIq2Xxs;
        case 17: return Format::kIq2Xs;
        case 18: return Format::kIq3Xxs;
        case 19: return Format::kIq1S;
        case 20: return Format::kIq4Nl;
        case 21: return Format::kIq3S;
        case 22: return Format::kIq2S;
        case 23: return Format::kIq4Xs;
        case 29: return Format::kIq1M;
        case 30: return Format::kBf16;
        case 42: return Format::kQ2_0;
        default: return Format::kUnknown;
    }
}

const char* format_name(Format f) {
    switch (f) {
        case Format::kF32: return "F32";
        case Format::kF16: return "F16";
        case Format::kBf16: return "BF16";
        case Format::kQ2_0: return "Q2_0";
        case Format::kQ4_0: return "Q4_0";
        case Format::kQ5_0: return "Q5_0";
        case Format::kQ8_0: return "Q8_0";
        case Format::kQ2K: return "Q2_K";
        case Format::kQ3K: return "Q3_K";
        case Format::kQ4K: return "Q4_K";
        case Format::kQ5K: return "Q5_K";
        case Format::kQ6K: return "Q6_K";
        case Format::kIq1S: return "IQ1_S";
        case Format::kIq1M: return "IQ1_M";
        case Format::kIq2Xxs: return "IQ2_XXS";
        case Format::kIq2Xs: return "IQ2_XS";
        case Format::kIq2S: return "IQ2_S";
        case Format::kIq3Xxs: return "IQ3_XXS";
        case Format::kIq3S: return "IQ3_S";
        case Format::kIq4Nl: return "IQ4_NL";
        case Format::kIq4Xs: return "IQ4_XS";
        case Format::kUnknown: break;
    }
    return "未知";
}

int block_elems(Format f) { return geometry(f).elems; }
int block_bytes(Format f) { return geometry(f).bytes; }
bool has_kernel(Format f) { return geometry(f).kernel; }
ActFormat activation_format(Format f) { return geometry(f).act; }

bool MatrixSpec::geometry_ok() const {
    const int e = block_elems(format);
    return e > 0 && cols > 0 && rows > 0 && cols % static_cast<std::uint64_t>(e) == 0;
}

std::uint64_t MatrixSpec::row_blocks() const {
    const int e = block_elems(format);
    return e > 0 ? cols / static_cast<std::uint64_t>(e) : 0;
}

std::uint64_t MatrixSpec::row_bytes() const {
    return row_blocks() * static_cast<std::uint64_t>(block_bytes(format));
}

bool MatrixSpec::usable() const { return geometry_ok() && has_kernel(format); }

std::uint64_t Table::usable_layers() const {
    std::uint64_t n = 0;
    for (const auto& l : layers) {
        if (l.usable()) ++n;
    }
    return n;
}

int layer_count(const artifact::GgufFile& gguf) {
    int max_layer = -1;
    for (const auto& t : gguf.tensors()) {
        const int l = layer_of(t.name);
        if (l > max_layer) max_layer = l;
    }
    return max_layer + 1;
}

Table build_table(const artifact::GgufFile& gguf, int first, int last) {
    Table table;
    for (int l = first; l <= last; ++l) {
        LayerSpec spec;
        spec.layer = l;
        const std::string base = "blk." + std::to_string(l) + ".ffn_";
        struct Slot {
            const char* suffix;
            MatrixSpec* out;
        };
        const Slot slots[3] = {
            {"gate_exps.weight", &spec.gate},
            {"up_exps.weight", &spec.up},
            {"down_exps.weight", &spec.down},
        };
        for (const Slot& s : slots) {
            const std::string name = base + s.suffix;
            const artifact::GgufTensorInfo* t = gguf.find(name);
            if (t == nullptr) {
                table.problems.push_back({l, name, "张力不存在"});
                continue;
            }
            std::string why;
            MatrixSpec m = make_spec(*t, why);
            *s.out = m;
            if (!why.empty()) table.problems.push_back({l, name, why});
        }
        table.layers.push_back(spec);
    }
    return table;
}

Table build_table(const artifact::GgufFile& gguf) {
    const int n = layer_count(gguf);
    if (n <= 0) {
        Table empty;
        empty.problems.push_back({-1, "", "文件里没有 blk.<N>. 形状的张力，认不出层"});
        return empty;
    }
    return build_table(gguf, 0, n - 1);
}

}  // namespace qinfer::experts