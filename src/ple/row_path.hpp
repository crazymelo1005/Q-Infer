// PLE 表行的读取路径：token 序列 → ngram 哈希出 16 个行索引 → 过 PCIe 字节预算仲裁 →
// 命中行缓存或读表 → IQ4_NL 反量化 → 按 head-slowest 拼成 2560 个 float。
//
// 这段装配此前只存在于 tests/test_ple_read_path.cpp 里（那里用合成替身表）。这里把它提成生产代码，
// 并把「表行从哪来」抽成一个可注入的来源，使同一段装配既能被合成表测，也能接到真模型的分片2。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "kernels/iq4nl.hpp"
#include "kernels/ngram.hpp"
#include "kernels/ple_gather.hpp"
#include "storage/row_cache.hpp"

namespace qinfer::ple {

inline constexpr int kNHeads = kernels::kPleNHeads;      // 16
inline constexpr int kHeadDim = kernels::kPleHeadDim;    // 160
inline constexpr int kVectorDim = kernels::kPleVectorDim;  // 2560

// 表行来源：把第 row 行的 90 字节读进 out。真模型上是分片2 的 per_layer_token_embd.weight。
class RowSource {
public:
    virtual ~RowSource() = default;
    virtual std::uint64_t rows() const = 0;
    virtual bool read_row(std::uint32_t row, std::uint8_t* out) const = 0;
};

// 用 GGUF 表读器做来源（它已经校验过形状 [160, N] 与类型 IQ4_NL）。
class GgufTableRowSource : public RowSource {
public:
    explicit GgufTableRowSource(artifact::GgufTable& table) : table_(table) {}
    std::uint64_t rows() const override { return table_.rows(); }
    bool read_row(std::uint32_t row, std::uint8_t* out) const override { return table_.read_row(row, out); }

private:
    artifact::GgufTable& table_;
};

struct PathStats {
    std::uint64_t cache_hits = 0;
    std::uint64_t table_reads = 0;
    std::uint64_t admitted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t overspend_bytes = 0;
};

// 由 token 序列算出每个 token 的「前两个 token」数组（prev[i*2] = 第 i-2 个、prev[i*2+1] = 第 i-1 个；
// 越界写 kTokenNull）。哈希需要它，故与路径放在一起，避免各处各写一遍。
std::vector<std::int32_t> build_prev(const std::int32_t* tokens, int n_tokens);

// 算出 n_tokens 个 token 的 PLE 向量，token-major 写进 out（长 n_tokens × 2560）。
// budget_bytes 是这批请求共享的同步 PCIe 字节预算：必需的表行即使装不下也全部准入（记为 overspend），
// 一条都不许丢——表行是精确输入，不走近似。行号越出表范围即失败（这是与文件数据的边界）。
bool ple_vectors_for_tokens(const RowSource& table, const std::int32_t* tokens, int n_tokens,
                            storage::RowCache& cache, std::uint64_t budget_bytes, float* out,
                            PathStats& stats, std::string& err);

}  // namespace qinfer::ple