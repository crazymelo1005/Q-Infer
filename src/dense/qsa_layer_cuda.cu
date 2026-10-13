// QSA 整层设备执行（见 qsa_layer_cuda.hpp 的口径说明）。
// 串接顺序与主机 run_qsa_layer 一致；计算走已验收的设备内核（投影/bf16/qk_prep/打分/attend+gate），
// 状态（KV 页表、idx_state、选择、聚集、窗口）留主机。权重按层缓存（w_layer 记账，与主机同语义）。
#include "dense/qsa_layer_cuda.hpp"

#include <cmath>

#include "dense/kv_window.hpp"
#include "dense/qsa_index_attend_cuda.hpp"
#include "dense/qsa_indexer.hpp"
#include "dense/qsa_qk_prep_cuda.hpp"
#include "dense/qsa_select.hpp"
#include "kernels/fp16.hpp"
#include "kernels/cuda/gdn_proj_gemv.hpp"

namespace qinfer::dense {

namespace {

// 读一个量化投影的字节，并按它自己的档设备算一次（主机侧搬运 + gdn_proj_gemv）。
bool project_cuda(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                  const experts::MatrixSpec& spec, const float* x, std::vector<std::uint8_t>& w,
                  bool reload, experts::FfnScratch& sc, float* out,
                  std::string& err) {
    if (reload) {
        w.resize(static_cast<std::size_t>(spec.rows * spec.row_bytes()));
        if (!g.read_at(t.offset, w.data(), w.size())) {
            err = t.name + " 读失败";
            return false;
        }
    }
    if (!experts::quantize_input(spec, x, sc, err)) return false;
    return qinfer::kernels::gdn_proj_gemv(spec, w.data(), sc, out, nullptr, err);
}

// 读一个 F32 一维张力（norm 权重）。
bool load_f32_1d(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                 std::vector<float>& dst, std::string& err) {
    dst.resize(static_cast<std::size_t>(t.elements()));
    if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(dst.data()),
                   static_cast<std::size_t>(t.elements()) * sizeof(float))) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

// 读一个 BF16 二维张力（索引器投影）。
bool load_bf16_2d(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                  std::vector<std::uint16_t>& dst, std::string& err) {
    dst.resize(static_cast<std::size_t>(t.elements()));
    if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(dst.data()),
                   static_cast<std::size_t>(t.elements()) * 2)) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

// 把逻辑页搬入常驻窗口（非空时）——与主机 touch_page 同款。
bool touch_page(QsaScratch& s, std::uint64_t logical, std::string& err) {
    if (s.window == nullptr) return true;
    std::uint64_t phys = 0;
    return s.window->ensure(logical, phys, err);
}

}  // namespace

bool run_qsa_layer_cuda(const artifact::GgufFile& gguf, const QsaLayer& layer, const float* x,
                        float* out, QsaScratch& s, std::string& err) {
    err.clear();
    if (x == nullptr || out == nullptr) {
        err = "run_qsa_layer_cuda 的输入输出指针为空";
        return false;
    }
    const std::uint64_t N = layer.q.cols;
    const std::uint64_t hd = layer.sh.head_dim;
    const std::uint64_t idx_dim = layer.idx_dim;
    const bool reload = s.w_layer != layer.layer;

    // Stage B: lazily reserve the device KV pool on first entry into this layer
    // (geometry comes from the page cache; without a window n_pages is 0 and
    // the device path stays off -> host path). Released by StepScratch.
    if (s.dev_kv != nullptr && s.dev_pool == nullptr && s.kv.n_pages > 0) {
        if (!s.dev_kv->reserve(s.dev_pool, s.kv.n_pages, s.kv.n_head_kv, s.kv.page_size,
                               s.kv.head_dim, err)) {
            err = "QSA layer " + std::to_string(layer.layer) + " device KV reserve: " + err;
            return false;
        }
    }
    const bool dev_kv = s.dev_kv != nullptr && s.dev_pool != nullptr;

    // 0. 一层一次：四个 norm 与索引器两个 BF16 投影读进来（按层号判）。
    if (reload) {
        if (!load_f32_1d(gguf, *layer.t_q_norm, s.q_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_k_norm, s.k_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_idx_q_norm, s.idx_q_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_idx_k_norm, s.idx_k_norm_w, err) ||
            !load_bf16_2d(gguf, *layer.t_idx_q, s.idx_q_w, err) ||
            !load_bf16_2d(gguf, *layer.t_idx_k, s.idx_k_w, err)) {
            return false;
        }
        if (s.cos_tab.empty()) {
            err = "QSA 的 rope 表没配（cos_tab/sin_tab 由调用方建好）";
            return false;
        }
    }

    // 1. 四个注意力投影（设备）。
    s.q_full.assign(static_cast<std::size_t>(layer.q.rows), 0.0f);
    s.k_raw.assign(static_cast<std::size_t>(layer.k.rows), 0.0f);
    s.v_raw.assign(static_cast<std::size_t>(layer.v.rows), 0.0f);
    if (!project_cuda(gguf, *layer.t_q, layer.q, x, s.w_q, reload, s.gemv, s.q_full.data(), err) ||
        !project_cuda(gguf, *layer.t_k, layer.k, x, s.w_k, reload, s.gemv, s.k_raw.data(), err) ||
        !project_cuda(gguf, *layer.t_v, layer.v, x, s.w_v, reload, s.gemv, s.v_raw.data(), err)) {
        return false;
    }

    // 2. q/门拆分 + q/k 归一 + 部分 rope（设备）。
    s.q_ready.assign(static_cast<std::size_t>(layer.sh.n_head * hd), 0.0f);
    s.k_ready.assign(static_cast<std::size_t>(layer.sh.n_head_kv * hd), 0.0f);
    s.gate.assign(s.q_ready.size(), 0.0f);
    const QsaQkWeights qkw{s.q_norm_w.data(), s.k_norm_w.data(), 1e-6f};
    QsaQkPrepCudaScratch qk_scratch{nullptr, nullptr, 0};
    if (!qsa_qk_prep_cuda(layer.sh.n_head, layer.sh.n_head_kv, hd, layer.n_rot, s.pos,
                          s.q_full.data(), s.k_raw.data(), qkw, s.cos_tab, s.sin_tab,
                          s.q_ready.data(), s.k_ready.data(), s.gate.data(), qk_scratch, nullptr,
                          err)) {
        return false;
    }

    // 3. KV 写入（fp16）：K 存归一+rope 之后的那一份，V 原样。权威副本照旧写主机。
    s.k_f16.assign(s.k_ready.size(), 0);
    s.v_f16.assign(s.v_raw.size(), 0);
    for (std::size_t i = 0; i < s.k_ready.size(); ++i) s.k_f16[i] = kernels::f32_to_f16_bits(s.k_ready[i]);
    for (std::size_t i = 0; i < s.v_raw.size(); ++i) s.v_f16[i] = kernels::f32_to_f16_bits(s.v_raw[i]);
    bool append_pinned = false;
    std::uint64_t append_page = 0;
    if (s.window != nullptr) {
        if (s.kv.page_size == 0) {
            err = "挂了常驻窗口但 page_size 是 0";
            return false;
        }
        append_page = s.pos / s.kv.page_size;
        if (!touch_page(s, append_page, err) || !s.window->pin(append_page, err)) return false;
        append_pinned = true;
    }
    if (s.store != nullptr &&
        !s.store->write_cell(s.pos, KvCell{s.k_f16.data(), s.v_f16.data()}, err)) {
        err = "写 KV 权威副本失败：" + err;
        return false;
    }
    if (!kv_append(s.kv, s.pos, KvCell{s.k_f16.data(), s.v_f16.data()}, err)) return false;
    // Device KV (stage B): also write the cell into the device pool. The host
    // store/window writes above stay for now (task 4 switches them off in the
    // all-resident mode); the twin write is what keeps the two paths comparable.
    if (dev_kv &&
        !s.dev_kv->append(s.dev_pool, s.pos, KvCell{s.k_f16.data(), s.v_f16.data()}, err)) {
        err = "QSA layer " + std::to_string(layer.layer) + " device KV append: " + err;
        return false;
    }
    // Stage B streaming cap (--kv-resident N): once the window slides past N
    // tokens, evict the oldest logical page (table -> -1; the host store stays
    // authoritative and the miss path below readmits on demand).
    if (dev_kv && s.resident_tokens > 0 && s.pos >= s.resident_tokens) {
        const std::uint64_t old = (s.pos - s.resident_tokens) / s.kv.page_size;
        const std::uint64_t cur = s.pos / s.kv.page_size;
        if (old < s.kv.n_pages && old != cur) {
            bool res = false;
            if (!s.dev_kv->resident(s.dev_pool, old, res, err) ||
                (res && !s.dev_kv->evict(s.dev_pool, old, err))) {
                err = "QSA layer " + std::to_string(layer.layer) + " device KV evict: " + err;
                return false;
            }
        }
    }

    // 4. 索引器：q/k 走 bf16 投影（设备）→ 主机 qsa_indexer_prep / indexer_key_append。
    s.idx_q_out.assign(static_cast<std::size_t>(layer.idx_q.rows), 0.0f);
    s.idx_k_out.assign(static_cast<std::size_t>(layer.idx_k.rows), 0.0f);
    if (!bf16_gemv_cuda(s.idx_q_w.data(), layer.idx_q.rows, layer.idx_q.cols, x,
                        s.idx_q_out.data(), nullptr, err) ||
        !bf16_gemv_cuda(s.idx_k_w.data(), layer.idx_k.rows, layer.idx_k.cols, x,
                        s.idx_k_out.data(), nullptr, err)) {
        return false;
    }
    s.q_idx.assign(static_cast<std::size_t>(layer.idx_n_head * idx_dim), 0.0f);
    if (!qsa_indexer_prep(layer.idx_n_head, idx_dim, layer.n_rot, s.pos, s.idx_q_out.data(),
                          s.idx_q_norm_w.data(), 1e-6f, s.cos_tab, s.sin_tab, s.q_idx.data(), err)) {
        return false;
    }
    if (s.idx_state.dead.empty()) s.idx_state.reset(layer.sh, idx_dim);
    if (!indexer_key_append(layer.sh, idx_dim, layer.n_rot, s.pos, s.pos_base, s.idx_k_out.data(),
                            s.idx_k_norm_w.data(), 1e-6f, s.cos_tab, s.sin_tab, s.idx_state, err)) {
        return false;
    }

    // 5. 逐 cell 打分（设备）→ 主机选择 → 按页聚集。
    const std::uint64_t n_kv = s.pos + 1;
    s.scores.assign(static_cast<std::size_t>(n_kv), 0.0f);
    if (!qsa_index_cuda(s.idx_state.pooled.data(), s.idx_state.n_bid, idx_dim, layer.idx_n_head,
                        s.q_idx.data(), nullptr, layer.sh.idx_block, n_kv, s.scores.data(),
                        nullptr, err)) {
        return false;
    }
    const std::uint64_t width = qsa_selection_width(n_kv, layer.sh);
    if (!qsa_select(s.scores.data(), n_kv, width, s.ids, err)) return false;
    const std::uint64_t n_ids = s.ids.size();
    std::vector<std::uint64_t> gather_pages;
    if (s.window != nullptr) {
        for (std::uint64_t id : s.ids) {
            const std::uint64_t pg = id / s.kv.page_size;
            bool seen = false;
            for (std::uint64_t q : gather_pages) {
                if (q == pg) seen = true;
            }
            if (seen) continue;
            if (!touch_page(s, pg, err) || !s.window->pin(pg, err)) return false;
            gather_pages.push_back(pg);
        }
    }
    s.attn.assign(static_cast<std::size_t>(layer.sh.n_head * hd), 0.0f);
    if (dev_kv) {
        // Stage B streaming miss path: readmit any selected page the cap
        // evicted, from the authoritative host store (one block copy each).
        if (s.store != nullptr) {
            std::vector<std::uint64_t> need;
            for (std::uint64_t id : s.ids) {
                const std::uint64_t pg = id / s.kv.page_size;
                bool dup = false;
                for (std::uint64_t q : need) {
                    if (q == pg) dup = true;
                }
                if (dup) continue;
                bool res = false;
                if (!s.dev_kv->resident(s.dev_pool, pg, res, err)) return false;
                if (!res) need.push_back(pg);
            }
            for (std::uint64_t pg : need) {
                std::vector<std::uint16_t> kb(static_cast<std::size_t>(s.store->block_elems())),
                    vb(static_cast<std::size_t>(s.store->block_elems()));
                if (!s.store->read_block(pg, kb.data(), vb.data(), err) ||
                    !s.dev_kv->admit_block(s.dev_pool, pg, kb.data(), vb.data(), err)) {
                    err = "QSA layer " + std::to_string(layer.layer) + " device KV admit: " + err;
                    return false;
                }
            }
        }
        // Stage B device path: gather into device scratch, attend on device
        // buffers (no per-step k/v H2D), attn copied back for the projection.
        const std::size_t kvb = static_cast<std::size_t>(n_ids * layer.sh.n_head_kv * hd) *
                                sizeof(std::uint16_t);
        const std::size_t qb = s.q_ready.size() * sizeof(float);
        const std::size_t qfb = s.q_full.size() * sizeof(float);
        const std::size_t ab = s.attn.size() * sizeof(float);
        void* dk = nullptr;
        void* dv = nullptr;
        float* d_q = nullptr;
        float* d_qf = nullptr;
        float* d_attn = nullptr;
        if (cudaMalloc(&dk, kvb) != cudaSuccess || cudaMalloc(&dv, kvb) != cudaSuccess ||
            cudaMalloc(&d_q, qb) != cudaSuccess || cudaMalloc(&d_qf, qfb) != cudaSuccess ||
            cudaMalloc(&d_attn, ab) != cudaSuccess) {
            err = "QSA layer " + std::to_string(layer.layer) + " device KV scratch alloc failed";
            return false;
        }
        bool ok = cudaMemcpy(d_q, s.q_ready.data(), qb, cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaMemcpy(d_qf, s.q_full.data(), qfb, cudaMemcpyHostToDevice) == cudaSuccess &&
                  s.dev_kv->gather(s.dev_pool, s.ids.data(), n_ids,
                                   static_cast<std::uint16_t*>(dk),
                                   static_cast<std::uint16_t*>(dv), err) &&
                  qsa_attend_cuda_dev(d_q, d_qf, static_cast<const std::uint16_t*>(dk),
                                      static_cast<const std::uint16_t*>(dv), n_ids, layer.sh,
                                      d_attn, nullptr, nullptr, err) &&
                  cudaMemcpy(s.attn.data(), d_attn, ab, cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(dk);
        cudaFree(dv);
        cudaFree(d_q);
        cudaFree(d_qf);
        cudaFree(d_attn);
        if (!ok) return false;
    } else {
        s.k_scratch.assign(static_cast<std::size_t>(n_ids * layer.sh.n_head_kv * hd), 0);
        s.v_scratch.assign(s.k_scratch.size(), 0);
        if (!kv_gather(s.kv, s.ids.data(), n_ids, s.k_scratch.data(), s.v_scratch.data(), err)) {
            return false;
        }
        if (!qsa_attend_cuda(s.q_ready.data(), s.k_scratch.data(), s.v_scratch.data(), n_ids,
                             layer.sh, s.q_full.data(), s.attn.data(), nullptr, nullptr, err)) {
            return false;
        }
    }
    for (std::uint64_t pg : gather_pages) {
        if (!s.window->unpin(pg, err)) return false;
    }

    s.gated.assign(s.attn.size(), 0.0f);
    std::swap(s.attn, s.gated);  // attend 已含 gate；s.gated 现在就是过门后的 attn
    if (!project_cuda(gguf, *layer.t_out, layer.out, s.gated.data(), s.w_out, reload, s.gemv, out,
                      err)) {
        return false;
    }
    if (append_pinned && !s.window->unpin(append_page, err)) return false;
    s.w_layer = layer.layer;
    ++s.pos;
    return true;
}

}  // namespace qinfer::dense