#include "ple/spec.hpp"

#include <cstddef>

namespace qinfer::ple {

namespace {

// GGUF 的类型码（规范表在 experts/expert_formats.cpp，S-42 核对过；这里只用得上这三个）。
constexpr std::uint32_t kTypeF32 = 0;
constexpr std::uint32_t kTypeF16 = 1;
constexpr std::uint32_t kTypeBf16 = 30;

std::string prefix_of(int layer) { return "blk." + std::to_string(layer) + "."; }

// 一个张力的形状与类型都要对上：形状按 ne 序（ne0 在前），不是「人类的行列序」。
bool check_tensor(const artifact::GgufTensorInfo& t, std::uint32_t type,
                  const std::vector<std::uint64_t>& dims, std::string& err) {
    if (t.type != type) {
        err = t.name + " 的类型码是 " + std::to_string(t.type) + "，这一路要的是 " +
              std::to_string(type);
        return false;
    }
    if (t.dims != dims) {
        std::string got;
        for (std::uint64_t d : t.dims) got += " " + std::to_string(d);
        std::string want;
        for (std::uint64_t d : dims) want += " " + std::to_string(d);
        err = t.name + " 的形状是 [" + got + " ]，要的是 [" + want + " ]";
        return false;
    }
    return true;
}

bool read_into(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t, void* dst,
               std::size_t bytes, std::string& err) {
    if (!g.read_at(t.offset, static_cast<std::uint8_t*>(dst), bytes)) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

}  // namespace

bool read_ple_spec(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd, std::uint64_t hc,
                   PleSpec& out, std::string& err) {
    err.clear();
    out = PleSpec{};
    out.layer = layer;
    if (n_embd == 0 || hc == 0) {
        err = "PLE 的几何不成立（n_embd 或 hc 是零）";
        return false;
    }
    const std::uint64_t hc_dim = n_embd * hc;
    const std::string p = prefix_of(layer);

    out.key = gguf.find(p + "ple_key.weight");
    out.value = gguf.find(p + "ple_value.weight");
    out.norm_key = gguf.find(p + "ple_norm_key.weight");
    out.norm_query = gguf.find(p + "ple_norm_query.weight");
    out.norm_conv = gguf.find(p + "ple_norm_conv.weight");
    out.conv1d = gguf.find(p + "ple_conv1d.weight");

    const int found = (out.key ? 1 : 0) + (out.value ? 1 : 0) + (out.norm_key ? 1 : 0) +
                      (out.norm_query ? 1 : 0) + (out.norm_conv ? 1 : 0) + (out.conv1d ? 1 : 0);
    if (found == 0) {
        out.present = false;  // 这一层没有 PLE（不是错误）
        return true;
    }
    out.present = true;
    if (found != 6) {
        err = "层 " + std::to_string(layer) + " 的 PLE 张力只找到 " + std::to_string(found) +
              " / 6 张——不全就明确失败，不按剩下的那几张凑";
        return false;
    }
    if (!check_tensor(*out.key, kTypeBf16, {n_embd, hc_dim}, err) ||
        !check_tensor(*out.value, kTypeBf16, {n_embd, n_embd}, err) ||
        !check_tensor(*out.norm_key, kTypeF32, {hc_dim}, err) ||
        !check_tensor(*out.norm_query, kTypeF32, {hc_dim}, err) ||
        !check_tensor(*out.norm_conv, kTypeF32, {hc_dim}, err) ||
        !check_tensor(*out.conv1d, kTypeF16, {static_cast<std::uint64_t>(kConvKernel), hc_dim},
                      err)) {
        err = "层 " + std::to_string(layer) + " 的 PLE：" + err;
        return false;
    }
    return true;
}

bool load_ple_weights(const artifact::GgufFile& gguf, const PleSpec& spec, PleBuffers& out,
                      std::string& err) {
    err.clear();
    if (!spec.usable()) {
        err = "这一层没有可用的 PLE 登记（先 read_ple_spec）";
        return false;
    }
    const std::uint64_t hc_dim = spec.key->dims[1];
    const std::uint64_t n_embd = spec.key->dims[0];

    out.key_bf16.resize(static_cast<std::size_t>(hc_dim * n_embd));
    out.value_bf16.resize(static_cast<std::size_t>(n_embd * n_embd));
    out.norm_key.resize(static_cast<std::size_t>(hc_dim));
    out.norm_query.resize(static_cast<std::size_t>(hc_dim));
    out.norm_conv.resize(static_cast<std::size_t>(hc_dim));
    out.conv1d_f16.resize(static_cast<std::size_t>(kConvKernel) * hc_dim);

    if (!read_into(gguf, *spec.key, out.key_bf16.data(), out.key_bf16.size() * 2, err) ||
        !read_into(gguf, *spec.value, out.value_bf16.data(), out.value_bf16.size() * 2, err) ||
        !read_into(gguf, *spec.norm_key, out.norm_key.data(), out.norm_key.size() * 4, err) ||
        !read_into(gguf, *spec.norm_query, out.norm_query.data(), out.norm_query.size() * 4, err) ||
        !read_into(gguf, *spec.norm_conv, out.norm_conv.data(), out.norm_conv.size() * 4, err) ||
        !read_into(gguf, *spec.conv1d, out.conv1d_f16.data(), out.conv1d_f16.size() * 2, err)) {
        return false;
    }
    return true;
}

}  // namespace qinfer::ple