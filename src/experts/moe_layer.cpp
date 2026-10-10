#include "experts/moe_layer.hpp"

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

    for (int i = 0; i < k; ++i) {
        const std::uint64_t e = static_cast<std::uint64_t>(s.ids[static_cast<std::size_t>(i)]);
        // 逐专家偏移：专家 e 的矩阵起点 = 张力起点 + e × (rows × row_bytes)。
        if (!gguf.read_at(layer.gate->offset + e * layer.gate_expert_bytes, s.gate.data(),
                          s.gate.size()) ||
            !gguf.read_at(layer.up->offset + e * layer.up_expert_bytes, s.up.data(), s.up.size()) ||
            !gguf.read_at(layer.down->offset + e * layer.down_expert_bytes, s.down.data(),
                          s.down.size())) {
            err = "读专家 " + std::to_string(e) + " 的矩阵失败";
            return false;
        }
        ExpertWeights w{s.gate.data(), s.up.data(), s.down.data()};
        if (!expert_ffn(layer.spec, w, x, s.ffn, s.expert_out.data(), err)) return false;
        const float wt = s.weights[static_cast<std::size_t>(i)];
        for (std::size_t j = 0; j < hidden; ++j) out[j] += wt * s.expert_out[j];
    }
    return true;
}

}  // namespace qinfer::experts