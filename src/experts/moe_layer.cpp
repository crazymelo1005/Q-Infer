#include "experts/moe_layer.hpp"

#include <algorithm>

namespace qinfer::experts {

namespace {

const artifact::GgufTensorInfo* need(const artifact::GgufFile& g, const std::string& name,
                                     std::string& err) {
    const artifact::GgufTensorInfo* t = g.find(name);
    if (t == nullptr) err = name + " 不存在";
    return t;
}

}  // namespace

bool build_moe_layer(const artifact::GgufFile& gguf, int layer, int top_k, MoeLayer& out,
                     std::string& err) {
    out = MoeLayer{};
    out.layer = layer;
    out.top_k = top_k;

    // 三个专家矩阵：直接复用分派表那一份口径（几何 + 内核 + 类型码）。
    const Table table = build_table(gguf, layer, layer);
    if (table.layers.empty()) {
        err = "没有这一层";
        return false;
    }
    out.spec = table.layers[0];
    if (!out.spec.usable()) {
        err = "层 " + std::to_string(layer) + " 的专家矩阵不可用：";
        for (const auto& p : table.problems) err += p.tensor + "（" + p.reason + "） ";
        return false;
    }

    const std::string base = "blk." + std::to_string(layer) + ".ffn_";
    out.gate = need(gguf, base + "gate_exps.weight", err);
    out.up = need(gguf, base + "up_exps.weight", err);
    out.down = need(gguf, base + "down_exps.weight", err);
    out.router = need(gguf, base + "gate_inp.weight", err);
    if (out.gate == nullptr || out.up == nullptr || out.down == nullptr || out.router == nullptr) {
        return false;
    }

    if (!read_router_spec(gguf, layer, out.router_spec, err)) return false;
    out.hidden = out.router_spec.hidden;
    out.n_expert = out.router_spec.n_expert;

    // 三个矩阵的专家数与路由器必须一致；几何上 gate/up 的输入维是 hidden、down 的输出维是 hidden。
    const std::uint64_t gate_experts = out.gate->dims.size() == 3 ? out.gate->dims[2] : 1;
    if (gate_experts != out.n_expert) {
        err = "gate 的专家数 " + std::to_string(gate_experts) + " 与路由器 " +
              std::to_string(out.n_expert) + " 不一致";
        return false;
    }
    if (out.spec.gate.cols != out.hidden || out.spec.down.rows != out.hidden) {
        err = "hidden 不自洽：gate.cols=" + std::to_string(out.spec.gate.cols) +
              " down.rows=" + std::to_string(out.spec.down.rows) + " 路由器=" +
              std::to_string(out.hidden);
        return false;
    }
    if (out.spec.gate.rows != out.spec.down.cols) {
        err = "ffn 不自洽：gate.rows=" + std::to_string(out.spec.gate.rows) +
              " down.cols=" + std::to_string(out.spec.down.cols);
        return false;
    }
    if (top_k <= 0 || static_cast<std::uint64_t>(top_k) > out.n_expert) {
        err = "top_k=" + std::to_string(top_k) + " 超过专家数 " + std::to_string(out.n_expert);
        return false;
    }

    out.gate_expert_bytes = out.spec.gate.rows * out.spec.gate.row_bytes();
    out.up_expert_bytes = out.spec.up.rows * out.spec.up.row_bytes();
    out.down_expert_bytes = out.spec.down.rows * out.spec.down.row_bytes();

    // 共享专家是可选的一层：缺它不算错（引擎侧 shared 允许为空），但一旦有一层就要四个都齐、且几何自洽。
    out.shared = table.shared.empty() ? SharedSpec{} : table.shared[0];
    if (out.shared.present) {
        if (!out.shared.usable()) {
            err = "层 " + std::to_string(layer) + " 的共享专家不可用：";
            for (const auto& p : table.problems) err += p.tensor + "（" + p.reason + "） ";
            return false;
        }
        out.s_gate = need(gguf, base + "gate_shexp.weight", err);
        out.s_up = need(gguf, base + "up_shexp.weight", err);
        out.s_down = need(gguf, base + "down_shexp.weight", err);
        out.s_gate_inp = need(gguf, base + "gate_inp_shexp.weight", err);
        if (out.s_gate == nullptr || out.s_up == nullptr || out.s_down == nullptr ||
            out.s_gate_inp == nullptr) {
            return false;
        }
        // 共享专家的 ffn 宽必须与路由专家一致（该模型的共享专家宽 = 640，与路由专家同）。
        if (out.shared.gate.cols != out.hidden || out.shared.down.rows != out.hidden ||
            out.shared.gate.rows != out.spec.gate.rows ||
            out.shared.down.cols != out.spec.down.cols) {
            err = "共享专家的几何与路由专家不一致：shared gate " +
                  std::to_string(out.shared.gate.cols) + "×" + std::to_string(out.shared.gate.rows) +
                  " 与路由 " + std::to_string(out.spec.gate.cols) + "×" +
                  std::to_string(out.spec.gate.rows);
            return false;
        }
    }
    return true;
}

bool run_moe_layer(const artifact::GgufFile& gguf, const MoeLayer& layer, const float* x, float* out,
                   MoeScratch& s, std::string& err, int* ids_out, float* weights_out) {
    const int k = layer.top_k;
    const std::size_t hidden = static_cast<std::size_t>(layer.hidden);

    // 路由矩阵：bf16，2 字节 × hidden × n_expert。
    s.router_w.resize(hidden * static_cast<std::size_t>(layer.n_expert));
    if (!gguf.read_at(layer.router->offset, reinterpret_cast<std::uint8_t*>(s.router_w.data()),
                      s.router_w.size() * 2)) {
        err = "读路由矩阵失败";
        return false;
    }
    s.ids.resize(static_cast<std::size_t>(k));
    s.weights.resize(static_cast<std::size_t>(k));
    if (!route(s.router_w.data(), static_cast<int>(layer.n_expert), static_cast<int>(hidden), k, x,
               s.router, s.ids.data(), s.weights.data(), err)) {
        return false;
    }
    if (ids_out != nullptr) {
        for (int i = 0; i < k; ++i) ids_out[i] = s.ids[static_cast<std::size_t>(i)];
    }
    if (weights_out != nullptr) {
        for (int i = 0; i < k; ++i) weights_out[i] = s.weights[static_cast<std::size_t>(i)];
    }

    s.gate.resize(static_cast<std::size_t>(layer.gate_expert_bytes));
    s.up.resize(static_cast<std::size_t>(layer.up_expert_bytes));
    s.down.resize(static_cast<std::size_t>(layer.down_expert_bytes));
    s.expert_out.assign(hidden, 0.0f);
    for (std::size_t i = 0; i < hidden; ++i) out[i] = 0.0f;

    // 走缓存时按层懒建一次字节来源；这一步也算作一个「token」，供 LRU 与共现图用。
    if (layer.cache_slots > 0) {
        if (s.source == nullptr || s.source_layer != layer.layer) {
            const std::string prefix = "blk." + std::to_string(layer.layer) + ".ffn_";
            s.source = std::make_unique<ExpertSource>(gguf, prefix, layer.spec, layer.cache_policy,
                                                      layer.cache_slots, layer.cache_ways,
                                                      layer.cache_byte_budget);
            s.source_layer = layer.layer;
        }
        s.source->begin_token();
        s.source->observe_token(reinterpret_cast<const std::uint32_t*>(s.ids.data()), k);
    }

    // 归并顺序固定为专家 id 升序（engine §16）。排名序只用来取前 k 与定权重，不用来定归并顺序。
    s.order.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) s.order[static_cast<std::size_t>(i)] = i;
    std::sort(s.order.begin(), s.order.end(),
              [&s](int a, int b) { return s.ids[static_cast<std::size_t>(a)] <
                                           s.ids[static_cast<std::size_t>(b)]; });

    for (int t = 0; t < k; ++t) {
        const std::size_t i = static_cast<std::size_t>(s.order[static_cast<std::size_t>(t)]);
        const std::uint64_t e = static_cast<std::uint64_t>(s.ids[i]);
        ExpertWeights w;
        if (s.source != nullptr) {
            ExpertBytes eb;
            if (!s.source->get(e, eb, err)) return false;
            w = ExpertWeights{eb.gate, eb.up, eb.down};
        } else {
            // 逐专家偏移：专家 e 的矩阵起点 = 张力起点 + e × (rows × row_bytes)。
            if (!gguf.read_at(layer.gate->offset + e * layer.gate_expert_bytes, s.gate.data(),
                              s.gate.size()) ||
                !gguf.read_at(layer.up->offset + e * layer.up_expert_bytes, s.up.data(),
                              s.up.size()) ||
                !gguf.read_at(layer.down->offset + e * layer.down_expert_bytes, s.down.data(),
                              s.down.size())) {
                err = "读专家 " + std::to_string(e) + " 的矩阵失败";
                return false;
            }
            w = ExpertWeights{s.gate.data(), s.up.data(), s.down.data()};
        }
        if (!expert_ffn(layer.spec, w, x, s.ffn, s.expert_out.data(), err)) return false;
        const float wt = s.weights[i];
        for (std::size_t j = 0; j < hidden; ++j) out[j] += wt * s.expert_out[j];
    }

    return add_shared_expert(gguf, layer, x, out, s, err);
}

bool add_shared_expert(const artifact::GgufFile& gguf, const MoeLayer& layer, const float* x,
                       float* out, MoeScratch& s, std::string& err) {
    // 共享专家：路由侧按权重、共享侧不加权，直接加（`y = Σ wᵢ·expertᵢ + shared`，[S-50]）。
    if (!layer.shared.present) return true;
    const std::size_t hidden = static_cast<std::size_t>(layer.hidden);
    s.s_gate_inp.resize(hidden);
    if (!gguf.read_at(layer.s_gate_inp->offset, reinterpret_cast<std::uint8_t*>(s.s_gate_inp.data()),
                      hidden * 2)) {
        err = "读共享专家的标量门失败";
        return false;
    }
    const std::uint64_t s_gate_bytes = layer.shared.gate.rows * layer.shared.gate.row_bytes();
    const std::uint64_t s_up_bytes = layer.shared.up.rows * layer.shared.up.row_bytes();
    const std::uint64_t s_down_bytes = layer.shared.down.rows * layer.shared.down.row_bytes();
    s.s_gate.resize(static_cast<std::size_t>(s_gate_bytes));
    s.s_up.resize(static_cast<std::size_t>(s_up_bytes));
    s.s_down.resize(static_cast<std::size_t>(s_down_bytes));
    // 共享专家没有专家维，故起点就是张力起点，不加逐专家偏移。
    if (!gguf.read_at(layer.s_gate->offset, s.s_gate.data(), s.s_gate.size()) ||
        !gguf.read_at(layer.s_up->offset, s.s_up.data(), s.s_up.size()) ||
        !gguf.read_at(layer.s_down->offset, s.s_down.data(), s.s_down.size())) {
        err = "读共享专家的矩阵失败";
        return false;
    }
    ExpertWeights sw{s.s_gate.data(), s.s_up.data(), s.s_down.data()};
    s.shared_out.assign(hidden, 0.0f);
    if (!shared_expert_ffn(layer.shared, sw, s.s_gate_inp.data(), x, s.shared_out.data(), s.s_ffn,
                           err)) {
        return false;
    }
    for (std::size_t j = 0; j < hidden; ++j) out[j] += s.shared_out[j];
    return true;
}

}  // namespace qinfer::experts