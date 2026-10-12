#include "dense/kv_store.hpp"

#include <cstddef>
#include <cstring>

namespace qinfer::dense {

namespace {

// 一个 cell 在它那个 block 里的元素起点：block 内是 `[kv_head][page_size][head_dim]`，head_dim 最快。
std::size_t cell_at(std::uint64_t block, std::uint64_t cell_in_block, std::uint64_t n_head_kv,
                    std::uint64_t page_size, std::uint64_t head_dim, std::uint64_t head) {
    return static_cast<std::size_t>(((block * n_head_kv + head) * page_size + cell_in_block) * head_dim);
}

}  // namespace

bool KvHostStore::write_cell(std::uint64_t t, const KvCell& cell, std::string& err) {
    err.clear();
    if (!sane()) {
        err = "KV 权威副本的几何不成立（或缓冲没按 n_blocks 开）";
        return false;
    }
    if (cell.k == nullptr || cell.v == nullptr) {
        err = "cell 的 K 或 V 指针为空";
        return false;
    }
    const std::uint64_t block = t / page_size;
    if (block >= n_blocks) {
        err = "cell " + std::to_string(t) + " 落在第 " + std::to_string(block) + " 个 block 上，副本只开到 " +
              std::to_string(n_blocks) + " 个";
        return false;
    }
    const std::uint64_t inner = t % page_size;
    for (std::uint64_t h = 0; h < n_head_kv; ++h) {
        const std::size_t at = cell_at(block, inner, n_head_kv, page_size, head_dim, h);
        std::memcpy(k.data() + at, cell.k + h * head_dim, static_cast<std::size_t>(head_dim) * 2);
        std::memcpy(v.data() + at, cell.v + h * head_dim, static_cast<std::size_t>(head_dim) * 2);
    }
    return true;
}

bool KvHostStore::read_block(std::uint64_t block, std::uint16_t* k_dst, std::uint16_t* v_dst,
                             std::string& err) {
    err.clear();
    if (k_dst == nullptr || v_dst == nullptr) {
        err = "read_block 的目标为空";
        return false;
    }
    if (!read_block_side(block, /*k_side=*/true, k_dst, err)) return false;
    return read_block_side(block, /*k_side=*/false, v_dst, err);
}

bool KvHostStore::read_block_side(std::uint64_t block, bool k_side, std::uint16_t* dst,
                                  std::string& err) {
    err.clear();
    if (!sane()) {
        err = "KV 权威副本的几何不成立（或缓冲没按 n_blocks 开）";
        return false;
    }
    if (dst == nullptr) {
        err = "read_block 的目标为空";
        return false;
    }
    if (block >= n_blocks) {
        err = "要搬的 block " + std::to_string(block) + " 超出副本的 " + std::to_string(n_blocks) + " 个";
        return false;
    }
    // 一个 block 的所有头在副本里是连续一段（与槽位同形），故搬入是一次 memcpy。
    const std::vector<std::uint16_t>& src = k_side ? k : v;
    const std::size_t off = static_cast<std::size_t>(block * block_elems());
    std::memcpy(dst, src.data() + off, static_cast<std::size_t>(block_elems()) * 2);
    bytes_moved += block_elems() * 2;
    return true;
}

}  // namespace qinfer::dense