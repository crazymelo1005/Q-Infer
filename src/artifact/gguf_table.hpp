// GGUF 表行读取：定位记忆表张力并读出指定行的原始字节。
//
// 只做这一件事：解析 GGUF 头与张力信息、按对齐算出数据区起点、校验张力形状与类型、按行号做随机读。
// 不做量化、不做缓存、不做 mmap——那些分别由 kernels/iq4nl 与 storage/row_cache 负责。
//
// 头部与张力信息的布局按参考引擎的 tools/gguf_reader.py 核对（S-34）。两个容易错的地方：
// kv_count 是 u64（不是 u32），张力的 offset 相对数据区起点而不是文件起点。
#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace qinfer::artifact {

class GgufTable {
public:
    GgufTable() = default;
    GgufTable(const GgufTable&) = delete;
    GgufTable& operator=(const GgufTable&) = delete;

    // 解析头并校验记忆表张力。失败时把原因写进 err（含张力名）。
    bool open(const std::string& path, std::string& err);
    void close();

    std::uint64_t rows() const { return rows_; }
    std::size_t row_bytes() const { return row_bytes_; }
    const std::string& tensor_name() const { return name_; }

    // 读一行原始字节到 out（长度须为 row_bytes()）。行号越界或读失败返回 false。
    bool read_row(std::uint32_t row, std::uint8_t* out);

private:
    std::ifstream file_;
    std::string path_;
    std::string name_;
    std::uint64_t rows_ = 0;
    std::uint64_t data_start_ = 0;
    std::uint64_t tensor_offset_ = 0;
    std::size_t row_bytes_ = 0;
    std::uint64_t file_size_ = 0;
};

}  // namespace qinfer::artifact
