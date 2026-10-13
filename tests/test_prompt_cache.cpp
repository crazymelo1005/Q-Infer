// 服务层 token 前缀缓存（/metrics conversation_cache 的数据源）：
// 全串精确匹配（BPE 合并可跨字节边界，前缀切开单独编码不成立）；LRU、容量=0 禁用、
// 超单条 token 上限不缓存；hits/misses/entries/tokens 统计由 /metrics 实读。
#include "serve/prompt_cache.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"

using namespace qinfer::serve;

int main() {
    // 1. 禁用态（capacity=0）：lookup 恒 false，store 不产生条目，不计数。
    {
        PromptCache c(0);
        CHECK(!c.enabled());
        std::vector<std::int32_t> ids = {1, 2, 3};
        c.store("k", ids);
        std::vector<std::int32_t> out;
        CHECK(!c.lookup("k", out));
        const PromptCache::Stats s = c.stats();
        CHECK(s.hits == 0 && s.misses == 0 && s.entries == 0 && s.tokens == 0);
    }

    // 2. 命中：store 后同键 lookup 返回逐值相等的 ids；统计 hits/entries/tokens 如实。
    {
        PromptCache c(4);
        CHECK(c.enabled());
        c.store("k1", {1, 2, 3});
        std::vector<std::int32_t> out;
        CHECK(c.lookup("k1", out));
        CHECK(out.size() == 3 && out[0] == 1 && out[1] == 2 && out[2] == 3);
        const PromptCache::Stats s = c.stats();
        CHECK(s.hits == 1 && s.misses == 0 && s.entries == 1 && s.tokens == 3);
    }

    // 3. 未命中：缺键 lookup 返回 false 且不碰 out，misses 计数。
    {
        PromptCache c(4);
        c.store("k", {9});
        std::vector<std::int32_t> out = {7};
        CHECK(!c.lookup("nope", out));
        CHECK(out.size() == 1 && out[0] == 7);   // 未命中不改写调用侧
        const PromptCache::Stats s = c.stats();
        CHECK(s.hits == 0 && s.misses == 1);
    }

    // 4. LRU 驱逐：满容量后存新键丢最老（先命中过的键被保活）。
    {
        PromptCache c(2);
        c.store("a", {1});
        c.store("b", {2});
        std::vector<std::int32_t> out;
        CHECK(c.lookup("a", out));               // a 提到头
        c.store("c", {3});                       // 驱逐 b
        CHECK(!c.lookup("b", out));
        CHECK(c.lookup("a", out));
        CHECK(c.lookup("c", out));
        const PromptCache::Stats s = c.stats();
        CHECK(s.entries == 2 && s.tokens == 2);  // a 与 c 各 1 个 token
    }

    // 5. 同键覆写：旧值先扣 token 再记新值，条目数不涨。
    {
        PromptCache c(4);
        c.store("k", {1});
        c.store("k", {1, 2, 2});
        std::vector<std::int32_t> out;
        CHECK(c.lookup("k", out));
        CHECK(out.size() == 3);
        const PromptCache::Stats s = c.stats();
        CHECK(s.entries == 1 && s.tokens == 3 && s.hits == 1);
    }

    // 6. 超单条 token 上限的 prompt 不缓存（防巨提示词吃满内存）。
    {
        PromptCache c(4);
        std::vector<std::int32_t> big(PromptCache::kMaxTokensPerEntry + 1, 1);
        c.store("big", big);
        const PromptCache::Stats s = c.stats();
        CHECK(s.entries == 0 && s.tokens == 0);
        std::vector<std::int32_t> out;
        CHECK(!c.lookup("big", out));
    }

    // 7. 命中计数与 miss 计数互不影响，序列行为连读。
    {
        PromptCache c(2);
        c.store("a", {1});
        std::vector<std::int32_t> out;
        CHECK(c.lookup("a", out));
        CHECK(!c.lookup("b", out));
        CHECK(c.lookup("a", out));
        const PromptCache::Stats s = c.stats();
        CHECK(s.hits == 2 && s.misses == 1);
    }

    std::printf("prompt_cache 全部通过\n");
    return 0;
}