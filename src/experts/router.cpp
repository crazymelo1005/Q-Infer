#include "experts/router.hpp"

#include "experts/ffn.hpp"  // run_bf16_gemv（激活舍一遍 + 按行并行）
#include "kernels/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace qinfer::experts {

bool read_router_spec(const artifact::GgufFile& gguf, int layer, RouterSpec& spec, std::string& err) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight";
    const artifact::GgufTensorInfo* t = gguf.find(name);
    if (t == nullptr) {
        err = name + " 不存在";
        return false;
    }
    if (t->dims.size() != 2) {
        err = name + " 的维度数不是 2";
        return false;
    }
    // 档随张力声明走（ADR-010）：30 = BF16 走「权重与激活都舍 bf16」那条，0 = F32 走 F32×F32。
    if (t->type != 30 && t->type != 0) {
        err = name + " 的类型码是 " + std::to_string(t->type) +
              "，既不是 BF16（30）也不是 F32（0）";
        return false;
    }
    spec.hidden = t->dims[0];
    spec.n_expert = t->dims[1];
    spec.type = t->type;
    spec.ok = true;
    return true;
}

void router_logits(const std::uint16_t* w_bf16, int n_expert, int hidden, const float* x,
                   float* logits) {
    // 与 QSA 索引器那两个 BF16 投影同一份算术（权重与激活都按 bf16 读、float 累加），故走共用的
    // 那个入口：激活舍一遍 + 按行并行（512 个专家的点积互相独立，只切行不动归约维）。
    std::string err;
    run_bf16_gemv(w_bf16, static_cast<std::uint64_t>(n_expert), static_cast<std::uint64_t>(hidden),
                  x, logits, err);
}

// 权重与激活都不舍 bf16（F32×F32），累加在 double 上——与 bf16 那条的差别只在这一点（ADR-010）。
void router_logits_f32(const float* w_f32, int n_expert, int hidden, const float* x, float* logits) {
    for (int e = 0; e < n_expert; ++e) {
        double acc = 0.0;
        const float* row = w_f32 + static_cast<std::size_t>(e) * static_cast<std::size_t>(hidden);
        for (int j = 0; j < hidden; ++j) {
            acc += static_cast<double>(row[j]) * static_cast<double>(x[j]);
        }
        logits[e] = static_cast<float>(acc);
    }
}

void router_topk(const float* logits, int n_expert, int k, RouterScratch& s, int* ids,
                 float* weights) {
    s.p.resize(static_cast<std::size_t>(n_expert));
    s.idx.resize(static_cast<std::size_t>(n_expert));

    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, static_cast<double>(logits[e]));
    double sum = 0.0;
    for (int e = 0; e < n_expert; ++e) {
        s.p[static_cast<std::size_t>(e)] = std::exp(static_cast<double>(logits[e]) - mx);
        sum += s.p[static_cast<std::size_t>(e)];
    }
    for (int e = 0; e < n_expert; ++e) s.p[static_cast<std::size_t>(e)] /= sum;

    // 稳定降序：严格大于的比较配稳定排序，等值元素保持索引序（即同值取小索引）。
    std::iota(s.idx.begin(), s.idx.end(), 0);
    std::stable_sort(s.idx.begin(), s.idx.end(), [&](int a, int b) {
        return s.p[static_cast<std::size_t>(a)] > s.p[static_cast<std::size_t>(b)];
    });

    double top = 0.0;
    for (int i = 0; i < k; ++i) {
        const int e = s.idx[static_cast<std::size_t>(i)];
        ids[i] = e;
        weights[i] = static_cast<float>(s.p[static_cast<std::size_t>(e)]);
        top += s.p[static_cast<std::size_t>(e)];
    }
    top = std::max(top, kRenormClamp);
    for (int i = 0; i < k; ++i) weights[i] = static_cast<float>(static_cast<double>(weights[i]) / top);
}

bool route(const std::uint16_t* w_bf16, int n_expert, int hidden, int k, const float* x,
           RouterScratch& scratch, int* ids, float* weights, std::string& err) {
    if (w_bf16 == nullptr || x == nullptr || ids == nullptr || weights == nullptr) {
        err = "参数为空";
        return false;
    }
    if (n_expert <= 0 || hidden <= 0 || k <= 0 || k > n_expert) {
        err = "几何不合理：n_expert=" + std::to_string(n_expert) + " hidden=" +
              std::to_string(hidden) + " k=" + std::to_string(k);
        return false;
    }
    scratch.logits.resize(static_cast<std::size_t>(n_expert));
    router_logits(w_bf16, n_expert, hidden, x, scratch.logits.data());
    router_topk(scratch.logits.data(), n_expert, k, scratch, ids, weights);
    return true;
}

bool route_f32(const float* w_f32, int n_expert, int hidden, int k, const float* x,
           RouterScratch& scratch, int* ids, float* weights, std::string& err) {
    if (w_f32 == nullptr || x == nullptr || ids == nullptr || weights == nullptr) {
        err = "参数为空";
        return false;
    }
    if (n_expert <= 0 || hidden <= 0 || k <= 0 || k > n_expert) {
        err = "几何不合理：n_expert=" + std::to_string(n_expert) + " hidden=" +
              std::to_string(hidden) + " k=" + std::to_string(k);
        return false;
    }
    scratch.logits.resize(static_cast<std::size_t>(n_expert));
    router_logits_f32(w_f32, n_expert, hidden, x, scratch.logits.data());
    router_topk(scratch.logits.data(), n_expert, k, scratch, ids, weights);
    return true;
}

}  // namespace qinfer::experts