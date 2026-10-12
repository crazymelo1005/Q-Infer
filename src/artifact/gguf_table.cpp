#include "artifact/gguf_table.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

namespace qinfer::artifact {

namespace {

// GGUF 元数据类型编号
enum : std::uint32_t {
    kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4, kInt32 = 5,
    kFloat32 = 6, kBool = 7, kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11, kFloat64 = 12,
};

constexpr std::uint32_t kIq4nlTypeCode = 20;      // ggml_type 20 = IQ4_NL
constexpr std::uint64_t kDefaultAlignment = 32;
constexpr std::size_t kRowBytes = 90;
constexpr std::uint64_t kHeadDim = 160;

bool read_exact(std::ifstream& f, void* dst, std::size_t n) {
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(f.gcount()) == n;
}

bool read_u32(std::ifstream& f, std::uint32_t& v) { return read_exact(f, &v, 4); }
bool read_u64(std::ifstream& f, std::uint64_t& v) { return read_exact(f, &v, 8); }

bool read_string(std::ifstream& f, std::string& out) {
    std::uint64_t n = 0;
    if (!read_u64(f, n)) return false;
    if (n > (1ull << 20)) return false;           // 元数据里的字符串不该这么大
    out.resize(static_cast<std::size_t>(n));
    return n == 0 || read_exact(f, out.data(), static_cast<std::size_t>(n));
}

// 读一个值（按类型）。数组要递归，否则后面的张力信息会错位。认不出的类型返回假——不猜。
bool read_value(std::ifstream& f, std::uint32_t type, GgufMeta& out) {
    out.type = type;
    switch (type) {
        case kUint8: { std::uint8_t v = 0; if (!read_exact(f, &v, 1)) return false; out.num = v; return true; }
        case kInt8: { std::int8_t v = 0; if (!read_exact(f, &v, 1)) return false; out.num = static_cast<std::uint64_t>(v); return true; }
        case kUint16: { std::uint16_t v = 0; if (!read_exact(f, &v, 2)) return false; out.num = v; return true; }
        case kInt16: { std::int16_t v = 0; if (!read_exact(f, &v, 2)) return false; out.num = static_cast<std::uint64_t>(v); return true; }
        case kUint32: { std::uint32_t v = 0; if (!read_exact(f, &v, 4)) return false; out.num = v; return true; }
        case kInt32: { std::int32_t v = 0; if (!read_exact(f, &v, 4)) return false; out.num = static_cast<std::uint64_t>(v); return true; }
        case kUint64: { if (!read_u64(f, out.num)) return false; return true; }
        case kInt64: { std::int64_t v = 0; if (!read_exact(f, &v, 8)) return false; out.num = static_cast<std::uint64_t>(v); return true; }
        case kFloat32: { std::uint32_t v = 0; if (!read_exact(f, &v, 4)) return false; float fv = 0; std::memcpy(&fv, &v, 4); out.real = fv; return true; }
        case kFloat64: { std::uint64_t v = 0; if (!read_u64(f, v)) return false; double dv = 0; std::memcpy(&dv, &v, 8); out.real = dv; return true; }
        case kBool: { std::uint8_t v = 0; if (!read_exact(f, &v, 1)) return false; out.num = v ? 1 : 0; return true; }
        case kString: return read_string(f, out.str);
        case kArray: {
            std::uint32_t elem = 0;
            std::uint64_t count = 0;
            if (!read_u32(f, elem) || !read_u64(f, count)) return false;
            if (count > (1ull << 32)) return false;  // 元数据里的数组不该这么大
            for (std::uint64_t i = 0; i < count; ++i) {
                GgufMeta one;
                if (!read_value(f, elem, one)) return false;
                switch (elem) {
                    case kString: out.strs.push_back(std::move(one.str)); break;
                    case kFloat32: case kFloat64: out.reals.push_back(one.real); break;
                    default: out.nums.push_back(one.num); break;  // 整数族与布尔
                }
            }
            return true;
        }
        default: return false;
    }
}

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) {
    return a == 0 ? v : (v + a - 1) / a * a;
}

// 解析结果：张力表 + 元数据 + 对齐 + 数据区起点。两个读取类共用这一份，避免两处各写一遍头部解析。
struct Parsed {
    std::vector<GgufTensorInfo> tensors;
    std::unordered_map<std::string, GgufMeta> meta;
    std::uint64_t alignment = kDefaultAlignment;
    std::uint64_t data_start = 0;
    std::uint64_t file_size = 0;
};

bool parse_gguf(std::ifstream& f, const std::string& path, Parsed& out, std::string& err) {
    f.seekg(0, std::ios::end);
    out.file_size = static_cast<std::uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    char magic[4];
    if (!read_exact(f, magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        err = path + " 不是 GGUF";
        return false;
    }
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0, kv_count = 0;
    if (!read_u32(f, version) || !read_u64(f, tensor_count) || !read_u64(f, kv_count)) {
        err = path + " 头部截断";
        return false;
    }
    if (version != 3) {
        err = path + " 是 GGUF v" + std::to_string(version) + "，本读取器只认 v3";
        return false;
    }

    for (std::uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        std::uint32_t type = 0;
        if (!read_string(f, key) || !read_u32(f, type)) {
            err = path + " 键值区截断";
            return false;
        }
        GgufMeta value;
        if (!read_value(f, type, value)) {
            err = path + " 键 " + key + " 的值读不出来（类型码 " + std::to_string(type) + "）";
            return false;
        }
        if (key == "general.alignment" && type == kUint32 && value.num != 0) {
            out.alignment = value.num;
        }
        out.meta.insert_or_assign(std::move(key), std::move(value));
    }

    out.tensors.reserve(static_cast<std::size_t>(tensor_count > 4096 ? 4096 : tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo t;
        std::uint32_t n_dims = 0;
        if (!read_string(f, t.name) || !read_u32(f, n_dims)) {
            err = path + " 张力区截断";
            return false;
        }
        if (n_dims > 8) { err = path + " 张力 " + t.name + " 的维度数不合理"; return false; }
        t.dims.resize(n_dims);
        for (std::uint32_t d = 0; d < n_dims; ++d)
            if (!read_u64(f, t.dims[d])) { err = path + " 张力维度截断"; return false; }
        if (!read_u32(f, t.type) || !read_u64(f, t.offset)) {
            err = path + " 张力信息截断";
            return false;
        }
        out.tensors.push_back(std::move(t));
    }

    // 数据区起点 = 张力信息之后按对齐上取整。这里必须自己算：张力的 offset 是相对数据区的。
    out.data_start = align_up(static_cast<std::uint64_t>(f.tellg()), out.alignment);
    return true;
}

}  // namespace

const GgufTensorInfo* GgufFile::find(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

void GgufFile::close() {
    if (map_ != nullptr) {
        ::munmap(const_cast<std::uint8_t*>(map_), static_cast<std::size_t>(map_len_));
        map_ = nullptr;
        map_len_ = 0;
    }
    map_tried_ = false;
    if (file_.is_open()) file_.close();
    tensors_.clear();
    meta_.clear();
    data_start_ = 0;
    file_size_ = 0;
}

const GgufMeta* GgufFile::meta(const std::string& key) const {
    const auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

const std::vector<std::string>* GgufFile::meta_strings(const std::string& key) const {
    const GgufMeta* m = meta(key);
    return m == nullptr || m->strs.empty() ? nullptr : &m->strs;
}

const std::vector<std::uint64_t>* GgufFile::meta_nums(const std::string& key) const {
    const GgufMeta* m = meta(key);
    return m == nullptr || m->nums.empty() ? nullptr : &m->nums;
}

bool GgufFile::meta_u64(const std::string& key, std::uint64_t& out) const {
    const GgufMeta* m = meta(key);
    if (m == nullptr || m->type == 6 || m->type == 12 || m->type == 8 || m->type == 9) return false;
    out = m->num;
    return true;
}

const std::string* GgufFile::meta_str(const std::string& key) const {
    const GgufMeta* m = meta(key);
    return m == nullptr || m->type != 8 ? nullptr : &m->str;
}

bool GgufFile::ensure_mapped() const {
    if (map_ != nullptr) return true;
    if (map_tried_ || path_.empty() || file_size_ == 0) return false;
    map_tried_ = true;
    const int fd = ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) return false;
    void* p = ::mmap(nullptr, static_cast<std::size_t>(file_size_), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);  // 映射之后 fd 就可以关掉了（映射自己持有一份引用）
    if (p == MAP_FAILED) return false;
    map_ = static_cast<const std::uint8_t*>(p);
    map_len_ = file_size_;
    return true;
}

const std::uint8_t* GgufFile::view_at(std::uint64_t offset_in_data, std::uint64_t n) const {
    if (n == 0 || !ensure_mapped()) return nullptr;
    const std::uint64_t at = data_start_ + offset_in_data;
    if (at > map_len_ || n > map_len_ - at) return nullptr;
    return map_ + at;
}

bool GgufFile::open(const std::string& path, std::string& err) {
    close();
    path_ = path;
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        err = "打不开 " + path;
        return false;
    }
    Parsed p;
    if (!parse_gguf(file_, path, p, err)) {
        close();
        return false;
    }
    tensors_ = std::move(p.tensors);
    meta_ = std::move(p.meta);
    data_start_ = p.data_start;
    file_size_ = p.file_size;
    return true;
}

bool GgufFile::read_at(std::uint64_t offset_in_data, std::uint8_t* out, std::size_t n) const {
    if (!file_.is_open() || n == 0) return false;
    if (offset_in_data > file_size_) return false;
    const std::uint64_t at = data_start_ + offset_in_data;
    if (at > file_size_ || n > file_size_ - at) return false;
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(at), std::ios::beg);
    return read_exact(file_, out, n);
}

void GgufTable::close() {
    if (file_.is_open()) file_.close();
    rows_ = 0;
    row_bytes_ = 0;
    data_start_ = 0;
    tensor_offset_ = 0;
    file_size_ = 0;
    name_.clear();
}

bool GgufTable::open(const std::string& path, std::string& err) {
    close();
    path_ = path;
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        err = "打不开 " + path;
        return false;
    }
    Parsed p;
    if (!parse_gguf(file_, path, p, err)) {
        close();
        return false;
    }
    file_size_ = p.file_size;
    data_start_ = p.data_start;

    const std::string kTable = "per_layer_token_embd.weight";
    const GgufTensorInfo* t = nullptr;
    for (const auto& x : p.tensors) {
        if (x.name == kTable) { t = &x; break; }
    }
    if (t == nullptr) {
        err = path + " 里没有 " + kTable;
        close();
        return false;
    }
    name_ = t->name;
    tensor_offset_ = t->offset;
    if (t->dims.size() != 2 || t->dims[0] != kHeadDim) {
        err = path + " 的 " + kTable + " 形状不是 [160, N]";
        close();
        return false;
    }
    if (t->type != kIq4nlTypeCode) {
        err = path + " 的 " + kTable + " 类型码 " + std::to_string(t->type) +
              "，不是 IQ4_NL（20）";
        close();
        return false;
    }
    rows_ = t->dims[1];
    row_bytes_ = kRowBytes;

    const std::uint64_t need = rows_ * static_cast<std::uint64_t>(row_bytes_);
    if (data_start_ > file_size_ || tensor_offset_ > file_size_ - data_start_ ||
        need > file_size_ - data_start_ - tensor_offset_) {
        err = path + " 装不下这张表：需要 " + std::to_string(need) + " 字节（数据区起点 " +
              std::to_string(data_start_) + "，文件 " + std::to_string(file_size_) + "）";
        close();
        return false;
    }
    return true;
}

bool GgufTable::read_row(std::uint32_t row, std::uint8_t* out) const {
    if (!file_.is_open() || row_bytes_ == 0 || row >= rows_) return false;
    const std::uint64_t at = data_start_ + tensor_offset_ + static_cast<std::uint64_t>(row) * row_bytes_;
    if (at + row_bytes_ > file_size_) return false;
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(at), std::ios::beg);
    return read_exact(file_, out, row_bytes_);
}

}  // namespace qinfer::artifact