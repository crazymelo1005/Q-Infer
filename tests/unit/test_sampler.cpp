#include "check.hpp"
#include "serve/sampler.hpp"
#include <cstdio>
#include <vector>

using namespace qinfer::serve;

namespace {

// 一条固定 logits：索引 3 最高，2 次之，其余低且彼此接近。
std::vector<float> logs() {
    std::vector<float> v(16, 0.1f);
    v[2] = 1.0f;
    v[3] = 2.0f;
    return v;
}

}  // namespace

int main() {
    // 贪心：温度 0 → 恒取 argmax，并列取小索引。
    {
        Sampler s(42);
        SamplerParams p;
        p.temperature = 0.0f;
        const auto v = logs();
        bool all3 = true;
        for (int i = 0; i < 8; ++i) {
            if (s.sample(v.data(), v.size(), p) != 3u) all3 = false;
        }
        CHECK(all3);
        std::printf("贪心（温度 0）：恒取 argmax=3\n");
    }
    // seed 可复现：同 seed、同参数、同 logits → 同一串。
    {
        const auto v = logs();
        Sampler a(7), b(7);
        SamplerParams p;
        p.temperature = 0.9f;
        bool same = true;
        for (int i = 0; i < 64; ++i) {
            if (a.sample(v.data(), v.size(), p) != b.sample(v.data(), v.size(), p)) same = false;
        }
        CHECK(same);
        std::printf("seed 可复现：同 seed 同参数 64 次采样一致\n");
    }
    // 不同 seed 大概率不同（高概率断言：256 次里至少 3 次不同）。
    {
        const auto v = logs();
        Sampler a(1), b(2);
        SamplerParams p;
        p.temperature = 0.9f;
        int diff = 0;
        for (int i = 0; i < 256; ++i) {
            if (a.sample(v.data(), v.size(), p) != b.sample(v.data(), v.size(), p)) ++diff;
        }
        CHECK(diff >= 3);
        std::printf("不同 seed：256 次里 %d 次不同\n", diff);
    }
    // 分布支持域：top_p 截断后只从保留集采样，且支配项频率按权重占优。
    {
        std::vector<float> v(16, 0.0f);   // 尾部 0：加 top_p 后必须被整体剪掉
        v[2] = 9.0f;
        v[3] = 10.0f;
        Sampler s(9);
        SamplerParams p;
        p.temperature = 1.0f;
        p.top_p = 0.9f;
        int c2 = 0, c3 = 0, low = 0;
        for (int i = 0; i < 4000; ++i) {
            const std::uint32_t t = s.sample(v.data(), v.size(), p);
            if (t == 2) ++c2;
            else if (t == 3) ++c3;
            else ++low;
        }
        CHECK(low == 0);                   // top_p 把尾部整体剪掉：只会出 2 或 3
        CHECK(c3 > c2);                    // 权重 10 > 9 → 3 出现次数更多（4000 次下稳）
        CHECK(c2 > 0);
        std::printf("top_p=0.9 剪尾：2 出现 %d、3 出现 %d、其余 %d\n", c2, c3, low);
    }
    // 参数校验：温度负、top_p 越界 → invalid。
    {
        SamplerParams ok;
        CHECK(ok.valid());
        SamplerParams bad;
        bad.temperature = -0.1f;
        CHECK(!bad.valid());
        SamplerParams p1;
        p1.top_p = 1.5f;
        CHECK(!p1.valid());
        SamplerParams p0;
        p0.top_p = 0.0f;
        CHECK(!p0.valid());
        std::printf("参数校验：温度负 / top_p 越界均判 invalid\n");
    }
    std::puts("sampler: 通过");
    return 0;
}