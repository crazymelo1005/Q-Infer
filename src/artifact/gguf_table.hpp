// GGUF 读取：解析头与张力信息、按对齐算出数据区起点。两种取用方式：
//   * GgufTable —— 记忆表（per_layer_token_embd.weight）专用：校验形状与类型，按行号随机读。
//   * GgufFile  —— 通用：枚举全部张力的名字/维度/类型/偏移，按数据区偏移随机读。
// 不做量化、不做缓存、不做 mmap——那些分别由 kernels/* 与 storage/row_cache 负责。
//
// 头部与张力信息的布局按参考引擎的 tools/gguf_reader.py 核对（S-34）。两个容易错的地方：
// kv_count 是 u64（不是 u32），张力的 offset 相对数据区起点而不是文件起点。
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace qinfer::artifact {

// 一个张力在文件里的登记项。offset 相对数据区起点。
struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type = 0;  // ggml_type 编号
    std::uint64_t offset = 0;

    std::uint64_t elements() const {
        std::uint64_t n = 1;
        for (std::uint64_t d : dims) n *= d;
        return dims.empty() ? 0 : n;
    }
};

class GgufFile {
public:
    GgufFile() = default;
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // 解析头与全部张力信息。失败时把原因写进 err。
    bool open(const std::string& path, std::string& err);
    void close();

    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
    const GgufTensorInfo* find(const std::string& name) const;

    std::uint64_t data_start() const { return data_start_; }
    std::uint64_t file_size() const { return file_size_; }

    // 从数据区内的偏移处读 n 字节。越界或读失败返回 false。
    // const：读操作不改对象的逻辑状态（只移动流位置，故流本身声明为 mutable）。
    bool read_at(std::uint64_t offset_in_data, std::uint8_t* out, std::size_t n) const;

private:
    mutable std::ifstream file_;
    std::vector<GgufTensorInfo> tensors_;
    std::uint64_t data_start_ = 0;
    std::uint64_t file_size_ = 0;
    std::string path_;
};

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