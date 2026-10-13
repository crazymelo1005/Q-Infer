// QSA 一层的静态描述：五个投影的档与块几何 + 四个 norm 的登记项。先建结构，数值路径随后。
//
// 张力名（[S-54] 直读 GGUF）：`attn_q` / `attn_k` / `attn_v` / `attn_output`、
// `attn_q_norm` / `attn_k_norm`、`indexer.q_proj` / `indexer.k_proj` / `indexer.q_norm` / `indexer.k_norm`。
// QSA 层**没有** `attn_qkv`（那是 GDN 层的），故这一条也用来把两种层区分开。
//
// 几何由数据推出并互相核对（真模型：head_dim 256、n_head 24、n_head_kv 2，见 [S-54]）：
//   attn_q 的输出 = 2 · n_head · head_dim（q 与门融合，见 [S-57]）
//   attn_v 的输出 = n_head_kv · head_dim；attn_output 的输入 = n_head · head_dim
// 但 `head_dim` 与 `n_rot` 无法从形状唯一解出（6144 可以是 256×24 也可以是 128×48），故由调用方给，
// 本函数负责核。索引器：`indexer.q_proj` 的输出 = idx_n_head · idx_dim、`indexer.k_proj` 的输出 = idx_dim，
// 两者也要与 `indexer.*_norm` 的长度对上。
#pragma once

#include <cstdint>
#include <string>

#include "artifact/gguf_table.hpp"
#include "dense/attn_phases.hpp"
#include "dense/kv_device_cache.hpp"
#include "dense/kv_page_cache.hpp"
#include "dense/qsa_indexer.hpp"
#include "dense/qsa_select.hpp"
#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"  // FfnScratch（量化投影复用的激活像缓冲）

namespace qinfer::dense {

struct KvWindow;  // 常驻窗口（见 dense/kv_window.hpp）：非空时由它管逻辑页的搬入
struct KvHostStore;  // KV 的权威副本（见 dense/kv_store.hpp）：非空时每个新 cell 都要写它一份

struct QsaLayer {
    int layer = -1;
    QsaShapes sh;                  // idx_top_k / idx_block / head_dim / n_head / n_head_kv
    // 部分 rope 的维数。**注意它与注意力和索引器共用一份**——引擎的 QsaShapes 只有一个 n_rot，
    // 注释明写「64 rotated dims of head_dim (and of idx_dim)」，两边都是 64（见 S-57）。
    std::uint64_t n_rot = 0;
    std::uint64_t idx_dim = 0;     // 索引器的头宽（真模型 128）
    std::uint64_t idx_n_head = 0;  // 索引器的 q 头数（真模型 4）
    experts::MatrixSpec q, k, v, out;      // 四个注意力投影
    // 两个索引器投影：这份 artifact 里是 **BF16**（不是量化权重），故不经「按档分派」那张表，
    // 走 bf16 的点积（与路由器同一条路）；这里只登记它们的几何。
    experts::MatrixSpec idx_q, idx_k;
    bool idx_dense_bf16 = false;
    const artifact::GgufTensorInfo* t_q = nullptr;
    const artifact::GgufTensorInfo* t_k = nullptr;
    const artifact::GgufTensorInfo* t_v = nullptr;
    const artifact::GgufTensorInfo* t_out = nullptr;
    const artifact::GgufTensorInfo* t_idx_q = nullptr;
    const artifact::GgufTensorInfo* t_idx_k = nullptr;
    const artifact::GgufTensorInfo* t_q_norm = nullptr;
    const artifact::GgufTensorInfo* t_k_norm = nullptr;
    const artifact::GgufTensorInfo* t_idx_q_norm = nullptr;
    const artifact::GgufTensorInfo* t_idx_k_norm = nullptr;
};

// 建一层。`head_dim` / `n_rot` / `idx_dim` / `idx_top_k` / `idx_block` 由调用方给（形状解不出唯一值），
// `n_head` / `n_head_kv` / `idx_n_head` 从形状推出并互相核对。任一必需张力缺失、档没内核、几何不自洽
// （含「这一层不是 QSA 层」）都返回假并写 err。
bool build_qsa_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                     std::uint64_t head_dim, std::uint64_t n_rot, std::uint64_t idx_dim,
                     std::uint64_t idx_top_k, std::uint64_t idx_block, QsaLayer& out,
                     std::string& err);

// 跨 token 存活的状态 + 复用的中间量。`kv`（分页 KV 与其页表）与 `cos_tab`/`sin_tab`（rope 表）由调用方
// 配好：本函数只管算，不管页的换入换出（那是存储层的事）。`pos` 是本 token 的 cell 号、`pos_base`
// 是序列首个 cell 的位置（[S-59]）。
struct QsaScratch {
    std::uint64_t pos = 0;
    std::uint64_t pos_base = 0;
    KvPageCache kv;
    // 常驻窗口（可空）。非空时本层不再假定「表里预填了恒等映射」：写入与聚集前后会经窗口把需要的
    // 逻辑页搬入、并在本次使用期间钉住（刚写入的那一页也在其中）。见 dense/kv_window.hpp。
    KvWindow* window = nullptr;
    // KV 的权威副本（可空，与窗口一起由 attach_kv_window 挂上）。非空时**每个新 cell 都要写它一份**
    // （写槽位那条只在该页常驻时才写），于是槽位被驱逐再准入时能从副本把字节拿回来。见 dense/kv_store.hpp。
    KvHostStore* store = nullptr;
    // Device KV (stage B): an optional exec + live pool. When both are non-null
    // and the page-cache geometry is set, append/gather/attend run on the
    // device (pool reserved lazily on first entry into this layer, released by
    // StepScratch's destructor). Host builds keep both null -> host path.
    const KvDeviceExec* dev_kv = nullptr;
    KvDevicePool* dev_pool = nullptr;
    // Device pool capacity in LOGICAL pages (the window's logical_blocks, not
    // the physical slot count): the host path's 64 slots and the session's
    // logical length are two different numbers, and the device pool is an
    // identity-mapped mirror of the logical space. Set by attach_kv_windows;
    // 0 falls back to the physical n_pages (and disables the device path when
    // that is 0 too).
    std::uint64_t dev_kv_pages = 0;
    // Device residency cap in tokens (--kv-resident; 0 = all-resident). Only
    // meaningful when dev_kv is active: above the cap the oldest logical page is
    // evicted on append and readmitted from the host store on gather misses.
    std::uint64_t resident_tokens = 0;
    QsaIndexerState idx_state;
    std::vector<float> cos_tab, sin_tab;
    std::vector<float> q_norm_w, k_norm_w, idx_q_norm_w, idx_k_norm_w;
    std::vector<std::uint16_t> idx_q_w, idx_k_w;  // 索引器两个投影的 BF16 权重
    std::vector<std::uint8_t> w_q, w_k, w_v, w_out;
    // 上面那些权重（四个 norm、索引器两投影、四个注意力投影）按层缓存。记账的是**层号**，不是
    // 「缓冲空不空」：同一份 scratch 跨层复用时，层号变了就必须重读，否则会把上一层的大权重当成这一层的。
    int w_layer = -1;
    experts::FfnScratch gemv;
    std::vector<float> q_full, k_raw, v_raw, q_ready, k_ready, gate;
    std::vector<float> idx_q_out, idx_k_out, q_idx;
    std::vector<float> scores, attn, gated;
    std::vector<std::uint64_t> ids;
    std::vector<std::uint16_t> k_f16, v_f16, k_scratch, v_scratch;
};

// 跑一层一个 token（`out` 长 n_embd，覆盖写）。顺序（[S-57]）：四个注意力投影 → q/门拆分与 q/k 归一+rope
// → KV 写入 → 索引器两投影与 q 预处理、键追加 → 逐 cell 打分 → 选择 → 按页聚集 → softmax GQA →
// 乘门 → 输出投影。索引器的两个投影是 BF16，走与路由同一份 bf16 点积（[S-44]）。
bool run_qsa_layer(const artifact::GgufFile& gguf, const QsaLayer& layer, const float* x, float* out,
                   QsaScratch& s, std::string& err, AttnPhases* phases = nullptr);

}  // namespace qinfer::dense