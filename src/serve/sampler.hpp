// 采样器：把 logits 变成 token id。贪心 = temperature 0（并列取小索引，与 dense::greedy_argmax 同口径）；
// 温度与 top_p 采样用标准库 mt19937_64，不接受外部 RNG 依赖（ADR-008 在采样器上同样成立）。
// 默认值对齐 llama.cpp 采样抽屉（Strata 同源）：温度 0.8、top_p 0.95、seed -1 = 随机。
#pragma once

#include <cstdint>
#include <random>

namespace qinfer::serve {

struct SamplerParams {
    float temperature = 0.8f;
    float top_p = 0.95f;
    std::int64_t seed = -1;

    bool valid() const {
        return temperature >= 0.0f && top_p > 0.0f && top_p <= 1.0f;
    }
};

class Sampler {
public:
    explicit Sampler(std::int64_t seed) {
        seed = seed >= 0 ? seed : std::random_device{}();
        rng_.seed(static_cast<std::uint64_t>(seed));
    }

    // 从 logits（长 n）采一个下标。temperature == 0 → 贪心。
    std::uint32_t sample(const float* logits, std::size_t n, const SamplerParams& p) const;

private:
    mutable std::mt19937_64 rng_;
};

}  // namespace qinfer::serve