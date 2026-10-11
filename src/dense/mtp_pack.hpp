// MTP 草稿层的权重加载：识别两种 pack 的方言，把它们的张力解析到同一组「角色」上。
//
// 该层在 HF 上不是一个单独文件（官方 checkpoint 里是 31 个 `mtp.*` 张力散在 28 个分片，社区量化档里
// 根本没有），本地因此有两份可互相替代的 pack（[S-62]，逐张对照见 research/02 §9）：
//
//   引擎那份 `mtp-q2_0.gguf`：31 个张力、名字是 checkpoint 原始名（`mtp.*`）、专家 Q2_0、
//     **gate/up 与索引器 q/k 是融合的**、**不含**共享的 embedding 与头（运行时 bind 主模型那三件）。
//   社区那份 `blk.48.*`：35 个张力、llama.cpp 式名字（挂成第 49 层）、专家 Q8_0、
//     gate/up 与索引器 q/k 各自拆开、**自包含** token_embd / output / output_hc_*。
//
// 四条只有对着两份真文件才看得出的差异，本模块都要吃下（[S-62]）：
//   ① 拆合：融合的那两份要用「切片」描述（每专家块内的行区间 + 块总行数），不能当两张独立张力读；
//   ② 同一角色的精度可以不同（路由在引擎那份是 BF16、社区那份是 F32），故逐张力记它声明的类型码；
//   ③ 同一角色的**维数**也可以不同（共享专家的标量门：引擎那份 [n_embd×1] 二维，社区那份 [n_embd] 一维）；
//   ④ 共享头来源：自包含的用 pack 里的，否则由调用方 bind 主模型那三件（`self_contained_head`）。
//
// n_head / n_head_kv / idx_n_head / 每专家前馈维都**从形状反推并互相核对**，不写死。
// 本模块只做加载与校验；前向要复用 `block_layer` 的门控残差 / 注意力 / MoE 三件，另起一片。
#pragma once

#include <cstdint>
#include <string>

#include "artifact/gguf_table.hpp"

namespace qinfer::dense {

enum class MtpDialect : std::uint8_t { kUnknown, kEngineNames, kBlock48Names };

// 该层要读的每一个角色。融合张力（gate_up、index_qk）在两侧各占一个角色。
enum class MtpRole : std::uint8_t {
    kPreNormEmbedding,
    kPreNormHidden,
    kFcEmbed,   // 引擎：fc_embedding [n_embd × n_embd]；社区：eh_proj [2·n_embd × n_embd]，吃 concat(e, h)
    kFcHidden,  // 引擎：fc_hidden [hc_dim × n_embd]；社区那份**缺席**（已合进 eh_proj 的输入轴）
    kFinalHcNorm,
    kFinalHcDown,
    kFinalHcUp,
    kAttnHcNorm,
    kAttnHcDown,
    kAttnHcUp,
    kAttnHcInject,
    kFfnHcNorm,
    kFfnHcDown,
    kFfnHcUp,
    kFfnHcInject,
    kAttnQ,
    kAttnK,
    kAttnV,
    kAttnOut,
    kAttnQNorm,
    kAttnKNorm,
    kIdxQ,
    kIdxK,
    kIdxQNorm,
    kIdxKNorm,
    kRouter,
    kGateExps,
    kUpExps,
    kDownExps,
    kSGate,
    kSUp,
    kSDown,
    kSGateInp,
    kCount
};

const char* to_string(MtpRole r);

// 一个角色对应的字节视图。融合张力时 `rows < block_rows`：第 e 个专家块里要用的行是
// [e·block_rows + row_off, e·block_rows + row_off + rows)。非专家张力时 experts == 1、block_rows == rows。
struct MtpSlice {
    const artifact::GgufTensorInfo* t = nullptr;
    std::uint32_t type = 0;  // 该张力自己声明的 ggml_type 码（逐张力记，不按角色写死）
    std::uint64_t cols = 0;  // 输入维（ggml dims[0]）
    std::uint64_t rows = 0;  // 本角色要用的输出行数
    std::uint64_t row_off = 0;
    std::uint64_t block_rows = 0;
    std::uint64_t experts = 1;
    bool fused = false;
    bool present() const { return t != nullptr; }
    // 一行（一个专家块里、本角色那一段的一行）的字节数，按它自己声明的档算。
    std::uint64_t row_bytes() const;
};

struct MtpPack {
    MtpDialect dialect = MtpDialect::kUnknown;
    std::uint64_t n_embd = 0, hc = 0, hc_lr = 0;
    std::uint64_t head_dim = 0, n_rot = 0, idx_dim = 0;
    std::uint64_t top_k = 0;
    // 由形状反推并互核过的量。
    std::uint64_t n_head = 0, n_head_kv = 0, idx_n_head = 0;
    std::uint64_t n_expert = 0, expert_ffn = 0, qk_dim = 0;
    // 共享的 embedding 与头是否在这个 pack 里（社区那份自包含；引擎那份没有，要 bind 主模型）。
    bool self_contained_head = false;
    const artifact::GgufTensorInfo* t_token_embd = nullptr;
    const artifact::GgufTensorInfo* t_output = nullptr;
    const artifact::GgufTensorInfo* t_out_hc_norm = nullptr;
    const artifact::GgufTensorInfo* t_out_hc_down = nullptr;
    const artifact::GgufTensorInfo* t_out_hc_up = nullptr;

    MtpSlice role[static_cast<std::size_t>(MtpRole::kCount)];
    const MtpSlice& at(MtpRole r) const { return role[static_cast<std::size_t>(r)]; }
};

// 识别方言并解析全部角色。几何与超参由调用方给（head_dim / n_rot / idx_dim / top_k 解不出唯一值），
// n_head / n_head_kv / 每专家前馈维 / 专家数由形状推出并互相核对。**`idx_n_head` 也必须由调用方给**：
// 引擎那份把索引器的 q 与 k 融合成一张 [n_embd × (idx_n_head+1)·idx_dim]，融合后的行数只能反推出
// 「q 的行数 + k 的行数」，拆不出比例（社区那份是拆开的，那份能反推）。
// 任一处缺失、形状不自洽、或两种方言同时/都不出现，都返回假并把原因写进 err（含张力名）。
bool build_mtp_pack(const artifact::GgufFile& gguf, std::uint64_t n_embd, std::uint64_t hc,
                    std::uint64_t hc_lr, std::uint64_t head_dim, std::uint64_t n_rot,
                    std::uint64_t idx_dim, std::uint64_t idx_n_head, std::uint64_t top_k,
                    MtpPack& out, std::string& err);

}  // namespace qinfer::dense