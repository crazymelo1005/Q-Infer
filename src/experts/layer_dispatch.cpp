#include "experts/layer_dispatch.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace qinfer::experts {

namespace {

// 钉住的专家要在任何返回路径上解开，否则页表引用计数会一直非零、这些槽位再也换不掉。
bool unpin_all(ExpertSource& src, std::vector<std::uint64_t>& pinned, std::string& err) {
    bool ok = true;
    for (const std::uint64_t e : pinned) {
        std::string why;
        if (!src.unpin(e, why)) {
            if (ok) err = "unpin 专家 " + std::to_string(e) + " 失败：" + why;
            ok = false;
        }
    }
    pinned.clear();
    return ok;
}

}  // namespace

bool dispatch_layer_batch(const MoeLayer& layer, ExpertSource& src, ExpertWorker& primary,
                          ExpertWorker& fallback, const int* ids, const float* route_weight, int n,
                          const float* x, float* out, LayerDispatchScratch& s, RerouteStats& rs,
                          std::string& err) {
    err.clear();
    if (ids == nullptr || route_weight == nullptr || x == nullptr || out == nullptr) {
        err = "批量分派的指针不完整";
        return false;
    }
    if (n <= 0) {
        err = "这一批没有专家";
        return false;
    }

    const std::size_t hidden = static_cast<std::size_t>(layer.hidden);
    if (hidden == 0) {
        err = "这一层的 hidden 为 0";
        return false;
    }

    // 1. 归并顺序 = 专家 id 升序。排名序只决定权重，不决定并入顺序。
    s.order.resize(static_cast<std::size_t>(n));
    std::iota(s.order.begin(), s.order.end(), 0);
    std::sort(s.order.begin(), s.order.end(), [ids](int a, int b) { return ids[a] < ids[b]; });

    // 2. 取字节。常驻的钉住（页表引用计数），非常驻的拷出来（共享临时缓冲下一次就被覆写）。
    const std::size_t gb = static_cast<std::size_t>(layer.gate_expert_bytes);
    const std::size_t ub = static_cast<std::size_t>(layer.up_expert_bytes);
    const std::size_t db = static_cast<std::size_t>(layer.down_expert_bytes);
    if (gb == 0 || ub == 0 || db == 0) {
        err = "这一层的专家矩阵字节数为 0";
        return false;
    }
    s.weights.assign(static_cast<std::size_t>(n), ExpertWeights{});
    s.owned.resize(static_cast<std::size_t>(n));  // 内层缓冲一次建好：地址不再随扩容变化
    s.pinned.clear();

    for (int t = 0; t < n; ++t) {
        const std::size_t j = static_cast<std::size_t>(s.order[static_cast<std::size_t>(t)]);
        const std::uint64_t e = static_cast<std::uint64_t>(ids[j]);
        ExpertBytes eb;
        if (!src.get(e, eb, err)) {
            std::string ignore;
            unpin_all(src, s.pinned, ignore);
            return false;
        }
        if (eb.resident) {
            if (!src.pin(e, err)) {
                std::string ignore;
                unpin_all(src, s.pinned, ignore);
                return false;
            }
            s.pinned.push_back(e);
            s.weights[j] = ExpertWeights{eb.gate, eb.up, eb.down};
        } else {
            std::vector<std::uint8_t>& dst = s.owned[j];
            dst.resize(gb + ub + db);
            std::memcpy(dst.data(), eb.gate, gb);
            std::memcpy(dst.data() + gb, eb.up, ub);
            std::memcpy(dst.data() + gb + ub, eb.down, db);
            s.weights[j] = ExpertWeights{dst.data(), dst.data() + gb, dst.data() + gb + ub};
        }
    }

    // 3. 一批交出去，在汇合点等回来。
    s.batch_out.assign(static_cast<std::size_t>(n) * hidden, 0.0f);
    ExpertJob job;
    job.spec = &layer.spec;
    job.experts = s.weights.data();
    job.n = n;
    job.x = x;
    job.hidden = layer.hidden;
    job.out = s.batch_out.data();

    if (!run_job_with_fallback(primary, fallback, job, rs, err)) {
        std::string ignore;
        unpin_all(src, s.pinned, ignore);
        return false;
    }

    // 4. 按同一顺序并入（专家的权重各归各的，求和顺序与 order 一致）。
    for (int t = 0; t < n; ++t) {
        const std::size_t j = static_cast<std::size_t>(s.order[static_cast<std::size_t>(t)]);
        const float w = route_weight[j];
        const float* bo = s.batch_out.data() + j * hidden;
        for (std::size_t i = 0; i < hidden; ++i) out[i] += w * bo[i];
    }

    std::string uerr;
    if (!unpin_all(src, s.pinned, uerr)) {
        err = uerr;
        return false;
    }
    return true;
}

}  // namespace qinfer::experts