// QSA 整层设备执行（阶段 A，Task 8）。
//
// 顺序照主机 `run_qsa_layer`：四个注意力投影（gdn_proj_gemv，真档：q/v IQ4_XS、k Q6_K、out IQ3_S）
// → q/门拆分与归一+rope（qsa_qk_prep_cuda）→ KV 写入（权威副本照旧写主机——阶段 B 才搬 KV）
// → 索引器两投影（bf16_gemv_cuda）+ 主机 qsa_indexer_prep / indexer_key_append（idx_state 在主机）
// → 逐 cell 打分（qsa_index_cuda）→ 主机 qsa_select / kv_gather（页聚集仍走主机）
// → softmax GQA + 门（qsa_attend_cuda，含 gate）→ 输出投影（gdn_proj_gemv）。
//
// 与 GDN 整层同样的「计算上卡、状态留主机」形态：KV 页表、idx_state、窗口、选择结果都在 QsaScratch
// 主机侧维护，每段设备内核各自 H2D/D2H；跨 token 状态（pos、idx_state、KV）经 s 传入传出。
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/qsa_layer.hpp"

namespace qinfer::dense {

// x 长 n_embd、out 长 n_embd（覆盖写）。s 的状态（pos / idx_state / KV / 权重缓存）跨 token 存活。
bool run_qsa_layer_cuda(const artifact::GgufFile& gguf, const QsaLayer& layer, const float* x,
                        float* out, QsaScratch& s, std::string& err);

}  // namespace qinfer::dense