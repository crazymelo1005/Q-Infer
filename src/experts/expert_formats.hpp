// 专家权重的格式分派：把 GGUF 的逐张量类型码翻译成「档 + 块几何 + 激活搭档」，再按层组装成一张表。
//
// 这张表必须逐文件从 GGUF 读出来，不能写死：环境2 上现有两档同款模型的逐层映射互不相同，
// 同名档位覆盖的层数也不同（见 engine §6、S-36、S-38）。块几何按 gguf-py 的 GGML_QUANT_SIZES
// 核对（S-42），不是凭记忆。
//
// 「没有内核」与「读不出来」是两件事：前者记进 problems 而不是静默跳过——接口里悄悄用错格式
// 比直接失败危险得多。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::experts {

// 专家权重可能出现的量化档。kUnknown 表示类型码不认识。
enum class Format : std::uint8_t {
    kUnknown = 0,
    kF32,
    kF16,
    kBf16,
    kQ2_0,
    kQ4_0,
    kQ5_0,
    kQ8_0,
    kQ2K,
    kQ3K,
    kQ4K,
    kQ5K,
    kQ6K,
    kIq1S,
    kIq1M,
    kIq2Xxs,
    kIq2Xs,
    kIq2S,
    kIq3Xxs,
    kIq3S,
    kIq4Nl,
    kIq4Xs,
};

// 激活侧量化档。权重块与激活块的元素数必须一致才能做定点点积。
enum class ActFormat : std::uint8_t { kNone = 0, kQ8_0, kQ8K };

Format format_from_type_code(std::uint32_t code);
const char* format_name(Format f);

// 块几何。非块状档（F32/F16/BF16）的 block_elems 记为 1。
int block_elems(Format f);
int block_bytes(Format f);

// 该档在本仓库里有没有可用的权重内核（false 表示认得但还不能算）。
bool has_kernel(Format f);

// 该档做定点点积时的激活搭档；kNone 表示非块状或无内核。
ActFormat activation_format(Format f);

// 一个专家矩阵（gate / up / down）的形状与格式。
struct MatrixSpec {
    Format format = Format::kUnknown;
    std::uint64_t experts = 0;  // 第三维（专家数），二维张量时为 1
    std::uint64_t rows = 0;     // 输出维：gate/up 是 ffn，down 是 hidden
    std::uint64_t cols = 0;     // 输入维，也是定点点积的 k

    // cols 能被块元素数整除（这是「640 不是 256 的整数倍，故 down 不能用 QK=256 的档」的来源）。
    bool geometry_ok() const;
    std::uint64_t row_blocks() const;  // cols / block_elems
    std::uint64_t row_bytes() const;
    bool usable() const;               // 几何成立且有内核
};

struct LayerSpec {
    int layer = -1;
    MatrixSpec gate;
    MatrixSpec up;
    MatrixSpec down;
    bool usable() const { return gate.usable() && up.usable() && down.usable(); }
};

// 分派过程中发现的问题。不可用就记下来，不静默跳过。
struct Problem {
    int layer = -1;
    std::string tensor;
    std::string reason;
};

struct Table {
    std::vector<LayerSpec> layers;
    std::vector<Problem> problems;
    std::uint64_t usable_layers() const;
};

// 层数由张力名里的 blk.<N>. 前缀推出，不依赖元数据键。
int layer_count(const artifact::GgufFile& gguf);

// 逐层读张力类型组装分派表。缺张力、形状不符、几何不成立、没有内核，都进 problems。
Table build_table(const artifact::GgufFile& gguf);
// 只处理 [first, last] 这段层（含两端），便于用合成夹具做小范围测试。
Table build_table(const artifact::GgufFile& gguf, int first, int last);

}  // namespace qinfer::experts