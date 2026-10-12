#include "serve/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace qinfer::serve {

namespace {
// 温度缩放的 softmax（不归一化也没关系，top_p 与抽样都是按相对质量走）；返回长 n 的概率向量。
std::vector<double> softmax(const float* logits, std::size_t n, float temperature) {
    const double inv = 1.0 / static_cast<double>(temperature);
    double maxv = logits[0];
    for (std::size_t i = 1; i < n; ++i) maxv = std::max(maxv, static_cast<double>(logits[i]));
    std::vector<double> p(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = std::exp((static_cast<double>(logits[i]) - maxv) * inv);
        sum += p[i];
    }
    for (auto& x : p) x /= sum;
    return p;
}
}  // namespace

std::uint32_t Sampler::sample(const float* logits, std::size_t n, const SamplerParams& p) const {
    if (p.temperature == 0.0f) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (logits[i] > logits[best]) best = i;   // 并列保小索引（与 dense 的贪心同口径）
        }
        return static_cast<std::uint32_t>(best);
    }
    std::vector<double> probs = softmax(logits, n, p.temperature);
    if (p.top_p < 1.0f) {
        // 从高到低累加，累加到 >= top_p 时把后面的质量清零再归一化。
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            acc += probs[order[i]];
            if (i >= 1 && acc >= static_cast<double>(p.top_p)) {
                for (std::size_t j = i + 1; j < n; ++j) probs[order[j]] = 0.0;
                break;
            }
        }
    }
    const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum <= 0.0) return static_cast<std::uint32_t>(n - 1);  // 防御：理论上到不了
    std::uniform_real_distribution<double> u(0.0, sum);
    const double r = u(rng_);
    double c = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        c += probs[i];
        if (r < c) return static_cast<std::uint32_t>(i);   // 累加式抽样，保首项不漏
    }
    return static_cast<std::uint32_t>(n - 1);
}

}  // namespace qinfer::serve