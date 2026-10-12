// KV 的权威副本（主机侧）：一层 K/V 的**完整逻辑内容**，恒等布局 `[block][kv_head][page_size][head_dim]`。
//
// 这是「KV 换出/换入」的下面那一层。`kv_page_cache`（[S-57]）只管一页对一页的搬运与寻址、并规定表里的
// `-1` 表示「不在显存」；`kv_window` 管驻留、配额、驱逐与钉住；本文件补上**字节从哪来**。形状取自引擎的
// `include/strata/kernels/kv_stream.hpp`（[S-64]）：权威副本在主机内存里、布局与「全常驻 + 恒等页表」
// 完全一样（故一个 block 的字节在两侧都是连续一段，「搬入」就是一次 memcpy），显存里只有 n_slots 个槽位、
// 页表是驻留映射。
//
// 两条口径，都是引擎明写的：
//   * **写永远写这份副本**（常驻时顺带写槽位）。于是驱逐**不需要回写**，槽位也不会陈旧——拒绝的草稿、
//     会话恢复都不必失效重填。这是这一层最重要的那条简化，也是「槽位与副本不一致」这类静默错的来源，
//     故写副本这一步放在写槽位之前、且不由窗口代劳。
//   * 副本的档（引擎里有 int8 / q4_0 / K8V4）本仓库先只做 fp16：与 `kv_page_cache` 的档一致，压缩档
//     要连内核一起做，属另一片。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dense/kv_page_cache.hpp"

namespace qinfer::dense {

struct KvHostStore {
    std::uint64_t page_size = 0;    // 每页几个 cell（真模型 4）
    std::uint64_t n_head_kv = 0;    // KV 头数（真模型 2）
    std::uint64_t head_dim = 0;     // 头宽（真模型 256）
    std::uint64_t n_blocks = 0;     // 逻辑页数 = 会话上下文能到多少 cell / page_size
    std::vector<std::uint16_t> k;   // 各 n_blocks × n_head_kv × page_size × head_dim，fp16 位型
    std::vector<std::uint16_t> v;

    // 准入时搬进槽位的字节数（观测：常驻与否只是记账，故这是「真搬了字节」的唯一证据）。
    std::uint64_t bytes_moved = 0;

    void init(std::uint64_t ps, std::uint64_t nh, std::uint64_t hd, std::uint64_t blocks) {
        page_size = ps;
        n_head_kv = nh;
        head_dim = hd;
        n_blocks = blocks;
        const std::size_t n = static_cast<std::size_t>(block_elems() * blocks);
        k.assign(n, 0);
        v.assign(n, 0);
        bytes_moved = 0;
    }

    // 一个 block 的元素数（K 与 V 各这么多）。
    std::uint64_t block_elems() const { return n_head_kv * page_size * head_dim; }

    bool sane() const {
        return page_size != 0 && n_head_kv != 0 && head_dim != 0 && n_blocks != 0 &&
               k.size() == static_cast<std::size_t>(block_elems() * n_blocks) && v.size() == k.size();
    }

    // 写一个 cell 的 K/V（逻辑 cell 号 t）。寻址与 `kv_page_cache` 的物理行公式同形（[S-57]）：
    // block = t / page_size，页内第 t % page_size 个 cell，逐头一段 head_dim 连续。
    bool write_cell(std::uint64_t t, const KvCell& cell, std::string& err);

    // 把一个 block（**所有 kv 头 × page_size 个 cell × head_dim**）整段拷进 dst。即引擎 resolve 的第 3 步。
    bool read_block(std::uint64_t block, std::uint16_t* k_dst, std::uint16_t* v_dst, std::string& err);

    // 只读一面（K 或 V），供夹具做「只拷一半」那条错读法的对照。
    bool read_block_side(std::uint64_t block, bool k_side, std::uint16_t* dst, std::string& err);
};

}  // namespace qinfer::dense