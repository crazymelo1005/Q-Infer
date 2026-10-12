// 一层的 MoE：路由出 top-k 专家 → 逐个前馈 → 按路由权重累加。
//
// 这一步把已有的两块接起来：`experts/router`（选专家与权重）与 `experts/ffn`（三矩阵组装）。
// 它同时负责从 GGUF 里按「逐专家偏移」取字节——三维张量 [cols, rows, experts] 的专家步长是
// `rows × row_bytes()`，三个矩阵各自如此；偏移算错会静默读到别的专家的权重，故测试里专门有一条。
//
// 本实现是主机侧参考路径：每次调用按需从文件读路由矩阵与选中专家的三矩阵，不做缓存。
// 专家缓存（槽位/替换）是独立子系统，接在 `storage/page_table` 之上，不在这里。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "experts/expert_formats.hpp"
#include "experts/expert_source.hpp"
#include "experts/ffn.hpp"
#include "experts/router.hpp"

namespace qinfer::experts {

// 一层 MoE 的静态描述：分派表 + 张力登记 + 逐专家字节步长。构建一次，逐 token 复用。
// 里面的指针指向 GgufFile 内部的张力表，故 GgufFile 在 MoeLayer 存活期间不得重新 open。
struct MoeLayer {
    int layer = -1;
    int top_k = 0;
    LayerSpec spec;
    SharedSpec shared;   // 这一层的共享专家；present=false 表示没有
    RouterSpec router_spec;
    const artifact::GgufTensorInfo* gate = nullptr;
    const artifact::GgufTensorInfo* up = nullptr;
    const artifact::GgufTensorInfo* down = nullptr;
    const artifact::GgufTensorInfo* router = nullptr;
    const artifact::GgufTensorInfo* s_gate = nullptr;
    const artifact::GgufTensorInfo* s_up = nullptr;
    const artifact::GgufTensorInfo* s_down = nullptr;
    const artifact::GgufTensorInfo* s_gate_inp = nullptr;  // 1 维 bf16 标量门，长度 hidden
    std::uint64_t gate_expert_bytes = 0;
    std::uint64_t up_expert_bytes = 0;
    std::uint64_t down_expert_bytes = 0;
    std::uint64_t hidden = 0;
    std::uint64_t n_expert = 0;

    // 专家缓存（engine §5）。cache_slots = 0 表示不走缓存：按需直接读表，与最早的实现一致。
    // 开了之后读取路径走 storage/expert_cache 的策略决策与 storage/page_table 的不变量。
    std::size_t cache_slots = 0;
    int cache_ways = 8;
    std::uint64_t cache_byte_budget = 0;  // 0 表示不限
    storage::PolicyKind cache_policy = storage::PolicyKind::kLru;
};

// 校验一层的三个专家矩阵与路由器：几何成立、有内核、路由是 BF16、维度自洽（gate.cols == hidden == down.rows、
// gate.rows == ffn == down.cols）。任何一条不成立都返回 false 并写 err。
bool build_moe_layer(const artifact::GgufFile& gguf, int layer, int top_k, MoeLayer& out,
                     std::string& err);

struct MoeScratch {
    RouterScratch router;
    FfnScratch ffn;
    std::vector<std::uint16_t> router_w;      // 路由矩阵（声明为 BF16 时）
    std::vector<float> router_w_f32;          // 路由矩阵（声明为 F32 时，ADR-010）
    // 上面那两份缓冲是按**层号**缓存的：路由矩阵每层一份且逐 token 恒定，不缓存的话每步每层都要重读
    // 2.6 MB。记账的是层号，不是「缓冲空不空」（同一份 scratch 跨层复用时后者会拿错层的路由）。
    int router_layer = -1;
    std::vector<std::uint8_t> gate, up, down;
    std::vector<float> expert_out;
    std::vector<int> ids;
    std::vector<float> weights;
    std::vector<int> order;  // 归并顺序：ids 的按专家 id 升序排列（engine §16）
    // 共享专家：标量门的 bf16 权重、三矩阵的字节区、以及它自己的中间缓冲。
    std::vector<std::uint16_t> s_gate_inp;      // 共享专家的标量门（声明为 BF16 时）
    std::vector<float> s_gate_inp_f32;         // 同上（声明为 F32 时，ADR-010）
    std::vector<std::uint8_t> s_gate, s_up, s_down;
    FfnScratch s_ffn;
    std::vector<float> shared_out;
    // 走缓存时的字节来源（按层懒建；见 MoeLayer 的 cache_slots）。
    std::unique_ptr<ExpertSource> source;
    int source_layer = -1;
};

// 前向声明：这几个类型定义在 `experts/layer_dispatch.hpp`，而那个头要用本头的 MoeLayer，故只能
// 前向声明、不能互相包含。
struct ComputeGroup;
struct LayerDispatchScratch;
struct RerouteStats;

// 「把未命中专家这批算力交出去」的入口（interfaces §8）：给了这一项就改走批量分派（候选计算组 +
// 字节来源 + 批量的中间缓冲），不给就还是逐专家的主机路径。两条路的归并顺序相同（按专家 id 升序），
// 故同配置下逐位一样。
struct MoeOffload {
    const std::vector<ComputeGroup>* candidates = nullptr;  // 候选组（按本层的档给）
    ExpertSource* source = nullptr;                         // 字节来源（含该层的缓存）
    LayerDispatchScratch* scratch = nullptr;                // 批量的中间缓冲
    RerouteStats* stats = nullptr;                          // 重派计数（观测用，可为空）
};

// 一层 MoE 的耗时切分（只作观测，给了才计时）。四项互斥且穷尽：
//   io     取数：路由矩阵的读 + 逐专家的字节取数（走缓存时含查找与未命中时的拷入）
//   route  路由计算：bf16/f32 点积 + softmax + top-k + 再归一
//   expert 逐专家前馈 + 加权累加（那 3k 次按档 GEMV 都在这里）
//   shared 共享专家（它常驻显存侧，不在未命中那一批里）
//
// 为什么要切：整步相位图里 MoE 是最大的单段，而「读 / 路由 / 逐专家 / 共享」四者的修法完全不同
// （分别是字节常驻、把路由也按行并行、把提交并大、共享专家单独处理），不切就只能猜。
struct MoePhases {
    double io_ms = 0.0;
    double route_ms = 0.0;
    double expert_ms = 0.0;
    double shared_ms = 0.0;
};

// 跑一层：读路由矩阵 → 路由出 top-k → 逐个读该专家的三矩阵并前馈 → 按权重累加 → 有共享专家则再加。
// 组合口径是 `y = Σ wᵢ·expertᵢ + shared`：路由侧按权重、共享侧**不加权**直接相加（[S-50]）。
// out 长 hidden，覆盖写（MoE 输出本身）。若 ids 与 weights 非空则回填路由结果，便于核对。
//
// 归并顺序固定为**专家 id 升序**（engine §16）：路由给出的排名序只决定「取哪 k 个、各自权重多少」，
// 不决定归并顺序。三路计算（显存侧命中专家、CPU 侧未命中、核显分担）都必须按同一顺序并入，否则同
// 配置下无法逐位复现。`src/experts/layer_dispatch` 的批量路径用的就是同一顺序。
bool run_moe_layer(const artifact::GgufFile& gguf, const MoeLayer& layer, const float* x, float* out,
                   MoeScratch& scratch, std::string& err, int* ids_out = nullptr,
                   float* weights_out = nullptr, const MoeOffload* offload = nullptr,
                   MoePhases* phases = nullptr);

// 加上这一层的共享专家（`y += shared`）。该层没有共享专家时什么也不做、返回真。
// 单独抽出来是因为共享专家不属于「未命中专家」那一批：批量分担走 `layer_dispatch`，共享专家在
// 显存侧算（engine §6），两条路径各自把结果加进同一个 out。
bool add_shared_expert(const artifact::GgufFile& gguf, const MoeLayer& layer, const float* x,
                       float* out, MoeScratch& scratch, std::string& err);

}  // namespace qinfer::experts