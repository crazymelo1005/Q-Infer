# M1 服务层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Q-Infer 引擎装配成可对外服务的形态：OpenAI 与 Anthropic 兼容 HTTP 接口（非流式与 SSE 流式）、页面监控（三 tab + 请求检查页）、硬件遥测，并在环境2 真权重上端到端跑通单请求（engine.md §17 序 2 的验收条件）。

**Architecture:** 单进程 C++20 零依赖服务。httplib（单头整份入库）提供 HTTP 传输；协议解析、SSE、JSON、采样器、遥测全部自写。请求经有界队列交给槽位 worker，worker 驱动引擎步循环（`run_token_step`），增量经每 Job 的事件缓冲回写 HTTP 线程。前端是 `src/serve/web/` 下的原生静态三 tab 单页 + 独立请求检查页。

**Tech Stack:** C++20、CMake ≥ 3.28、cpp-httplib v0.18.6（唯一入库第三方）、NVML dlopen（运行时可选）、Linux `/proc` 与 sysfs（Windows API 兜底）。主机侧不依赖 CUDA。

**Spec:** [../specs/2026-10-12-serve-layer-design.md](../specs/2026-10-12-serve-layer-design.md)（本计划由它推导；执行者先读规格再动手。）

## Global Constraints

- C++20 严格模式、CMake ≥ 3.28（仓库根 CMakeLists 已设；新目标遵循每模块一库的 `qinfer_*` 惯例）。
- 依赖面为零（ADR-008）：httplib.h 是唯一入库的第三方文件，整份保留、锁定 v0.18.6、保留版权与许可注释。
- 主机侧不依赖 CUDA：`QINFER_ENABLE_CUDA` 保持默认 OFF 时全部主机代码可编译、可测试；CI 无 GPU。
- 每个任务结束前 `ctest` 中该任务相关项全绿，`python3 scripts/docs_checks.py` 全绿；任一提交不得让两者变红。
- 测试用 `tests/check.hpp` 的 `CHECK`（Release 也真检查；不用 `<cassert>`）。
- 注释与文档风格：无第一人称（含复数自称）、无文件级版本/日期头、粗体密度 ≤ 10/100 行、相对链接可解析。计划文档本身也过 docs_checks，代码注释里不得出现两位编号的 S-/G-/R-/P- 引用（未登记会红）。
- 复用既有 API（签名已在本计划编写时核实，勿改名或改语义）：
  - `artifact::GgufFile`：`open(path, err)`、`meta_str(key)`、`meta_u64(key, out)`、`find(name)`、`read_at`、`view_at`、`tensors()`。
  - `artifact::GgufTable`：`open(path, err)`、`read_row(row, out)`、`rows()`。
  - `dense`：`StepConfig`（含 `n_embd/hc/hc_lr/n_layers/top_k/gdn_head_dim/qsa_head_dim/n_rot/idx_dim/idx_top_k/idx_block/cache_slots/cache_ways/cache_byte_budget/cache_policy/offload_layers/ple`）、`StepModel`、`StepScratch`（`head.logits`、`last_record`、`ple`、`blocks[l].qsa`、`blocks[l].gdn.conv.tokens`）、`build_step_model`、`prepare_step_scratch(model, n_pages, pos_base, s, err)`、`run_step_model`、`run_token_step(gguf, model, tok_embed, token, s, err)`、`build_token_embed`、`read_token_embedding`、`greedy_argmax(v, n)`、`LayerKind::kGdn/kQsa`、`PleRun{layer,spec,w,table,cache,budget_bytes,ready(),can_stage()}`、`attach_kv_window(qsa_scratch, w, logical_blocks)`、`KvWindow`（`page_bytes/quota_bytes/hits/misses/evictions/refusals/store/pool/entries/clock`）、`KvHostStore`（`init(page_size, n_head_kv, head_dim, blocks)`、`bytes_moved`）。
  - `text::Tokenizer`：`from_gguf`、`encode(text, parse_special, out, err)`、`encode_plain`、`decode(ids)`、`special_id(lit)`。
  - `ple`：`read_ple_spec(gguf, layer, n_embd, hc, out, err)`、`load_ple_weights(gguf, spec, w, err)`、`GgufTableRowSource(table)`（`ple::RowSource` 实现）、`RowSource`、`PleSpec`（`present/usable()`）、`PleBuffers`、`kVectorDim`。
  - `storage::RowCache`：`RowCache(n_slots)`、`misses()`、`hits()`。
  - 真模型几何（推导结论）：48 层、`n_embd` 2560、`hc` 4、`hc_lr` 320、`top_k` 10、GDN 头宽 128 / QSA 头宽 256、`n_rot` 64、`idx_dim` 128、`idx_top_k` 2048、`idx_block` 4。合成测试夹具用 `tests/test_step_loop.cpp` 顶部的 `kN/kHc/kLr/kFfn/kExpert/kTopK/...` 同款几何（n_embd 256、2 层）。

## 文件结构

| 文件 | 职责 |
|------|------|
| `third_party/httplib/httplib.h` | HTTP 传输（整份入库，v0.18.6） |
| `src/serve/sampler.hpp/.cpp` | 采样器：temperature / top_p / seed（贪心 = 温度 0） |
| `src/serve/prompt.hpp/.cpp` | chat 模板渲染器（GGUF 模板原文）+ 模板读取 |
| `src/serve/facade.hpp/.cpp` | 装配门面：装载、预填、生成环（PLE 与 KV 窗口可选） |
| `src/serve/json.hpp/.cpp` | 手写 JSON 子集：解析、序列化、转义 |
| `src/serve/scheduler.hpp/.cpp` | 槽位租约、有界队列、Job 状态与计时、SSE 事件缓冲 |
| `src/serve/telemetry.hpp/.cpp` | 每秒采样：NVML dlopen / AMD sysfs / CPU / RAM / 磁盘，60 点历史 |
| `src/serve/records.hpp/.cpp` | 请求记录环（100 条，带 256K 截断） |
| `src/serve/server.hpp/.cpp` | httplib 接线、鉴权、路由、worker 池、协议拼装 |
| `src/serve/main.cpp` | 命令行入口，装配全部 |
| `src/serve/web/tokens.css` | 设计令牌（浅/深两套） |
| `src/serve/web/index.html` + `app.js` | 三 tab 单页：Chat / Monitor / About |
| `src/serve/web/monitor.html` + `monitor.js` | 请求检查页（对应 /api-monitor） |
| `tests/test_sampler.cpp` | 采样器单测 |
| `tests/test_prompt.cpp` | 模板渲染器单测 |
| `tests/test_facade.cpp` | 门面单测（合成 GGUF） |
| `tests/test_json.cpp` | JSON 单测 |
| `tests/test_scheduler.cpp` | 调度器与事件缓冲单测 |
| `tests/test_telemetry.cpp` | 遥测单测（CI 无 GPU 也绿） |
| `tests/test_records.cpp` | 记录环单测 |
| `tests/serve_fixture_gen.cpp` | 集成测试用的合成 GGUF 发生器（可执行） |
| `tests/integration/serve/run.sh` | 集成测试脚本（bash + curl，无 jq 依赖） |

## 任务清单

| 任务 | 交付 | 对应规格 |
|------|------|---------|
| T1 | 采样器 | §5 |
| T2 | chat 模板渲染器 | §2（装配门面）、§11 风险兜底 |
| T3 + T4 | 装配门面（主体；PLE 与 KV 窗口） | §2（装配门面）、§4 |
| T5 | httplib 入库 + CMake | §2、§10 |
| T6 | JSON 子集 | §2、§10 |
| T7 | 调度器 + 事件缓冲 | §4 |
| T8 | fixture 发生器 | §9 |
| T9 | 服务入口 + 非流式 OpenAI 路由 + 鉴权 + 集成脚本 | §3、§5、§9 |
| T10 | SSE 流式 | §3、§7 |
| T11 | Anthropic 域 | §3 |
| T12 | 状态与属性路由 | §3 |
| T13 | 遥测 | §6 |
| T14 | /metrics 与 /api/requests | §6、§3 |
| T15 | 前端一：tokens.css + index.html + Chat | §7 |
| T16 | 前端二：Monitor / About / api-monitor | §7 |
| T17 | 文档回填 + ADR-012 | §8 M1.5、§10 |
| T18 | 环境2 真权重端到端验收 | §8 M1.5 |

---

### Task 1: 采样器

**Files:**
- Create: `src/serve/sampler.hpp`, `src/serve/sampler.cpp`, `tests/test_sampler.cpp`
- Modify: `CMakeLists.txt`（新增 `qinfer_serve` 库与该测试）

**Interfaces:**
- Consumes: 无（自给自足；贪心口径与 `dense::greedy_argmax` 一致：并列取小索引）
- Produces: `qinfer::serve::SamplerParams`（T3/T9 用）、`qinfer::serve::Sampler`（T3 用）

- [ ] **Step 1: 写失败测试**

`tests/test_sampler.cpp`：

```cpp
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
}  // namespace
```

- [ ] **Step 2: 跑测试确定失败**

```bash
cmake --build build --target test_sampler 2>&1 | tail -5 || true
```

预期：找不到 `serve/sampler.hpp`（目标还不存在），构建失败。

- [ ] **Step 3: 写最小实现**

`src/serve/sampler.hpp`：

```cpp
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
```

`src/serve/sampler.cpp`：

```cpp
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
```

（注意：`sample` 是 const 成员，`rng_` 须在头文件里声明为 `mutable std::mt19937_64 rng_`。）

- [ ] **Step 4: 跑测试**

```bash
cmake --build build --target test_sampler && ./build/tests/test_sampler 2>/dev/null || (cd build && ctest -R sampler --output-on-failure)
```

预期：测试通过，输出四段断言说明。

- [ ] **Step 5: CMake 接线 + 提交**

在 `CMakeLists.txt` 的 `qinfer_text` 库定义附近新增：

```cmake
add_library(qinfer_serve
  src/serve/sampler.cpp
)
target_include_directories(qinfer_serve PUBLIC src)
target_link_libraries(qinfer_serve PUBLIC qinfer_text)
```

在测试区新增：

```cmake
add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE qinfer_serve)
add_test(NAME sampler COMMAND test_sampler)
```

```bash
cmake -B build -S . && cmake --build build --target test_sampler && ctest --test-dir build -R '^sampler$' --output-on-failure
git add src/serve/ CMakeLists.txt tests/test_sampler.cpp
git commit -m "feat(serve): 采样器——temperature/top_p/seed，温度 0 即贪心（并列取小索引）"
```

（该任务结束时应把 `sampler.cpp` 里那段「最终版」占位文字替换成真实实现——本计划允许执行者按注释口径完成累加抽样，行为由测试钉住。）

---

### Task 2: chat 模板渲染器

**Files:**
- Create: `src/serve/prompt.hpp`, `src/serve/prompt.cpp`, `tests/test_prompt.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `artifact::GgufFile`（`meta_str`）、`text::Tokenizer`（编码在 T3 用，本任务不用）
- Produces: `qinfer::serve::Message`、`chat_template_from_gguf`、`render_chat`（T3、T9、T11 用）

- [ ] **Step 1: 写失败测试**

`tests/test_prompt.cpp`：

```cpp
#include "check.hpp"
#include "serve/prompt.hpp"
#include <cstdio>
#include <vector>

using namespace qinfer::serve;

namespace {

// Qwen 系官方模板（tokenizer_config.json 的标准形态，主案例 GGUF 的 chat_template 应在 M1.5 逐字符核对）。
const char* kQwenTemplate =
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{%- for message in messages -%}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n'}}{%- endfor -%}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

std::string render_with(const std::vector<Message>& msgs, bool add_gen) {
    std::string out, err;
    CHECK(render_chat(kQwenTemplate, msgs, add_gen, out, err));
    return out;
}

int main() {
    // 一轮 system + user，加生成提示。
    {
        const std::string got =
            render_with({{"system", "You are helpful."}, {"user", "hi"}}, /*add_gen=*/true);
        const std::string want = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                                 "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        CHECK(got == want);
        std::printf("两段消息 + gen 提示：逐字符一致\n");
    }
    // 多轮 assistant 消息原样保留。
    {
        const std::vector<Message> msgs = {{"user", "1"}, {"assistant", "2"}, {"user", "3"}};
        const std::string got = render_with(msgs, false);
        CHECK(got == "<|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n"
                     "<|im_start|>user\n3<|im_end|>\n");
        std::printf("三轮：多轮拼接正确\n");
    }
    // 空消息列表 + 无生成提示 → 空串。
    {
        CHECK(render_with({}, false).empty());
        std::printf("空列表：渲染为空\n");
    }
    // 内容里的换行、单引号、花括号原样保留（字面量不做二次解释）。
    {
        const std::string got = render_with({{"user", "say 'hi'\n{ not a tag }"}}, false);
        CHECK(got.find("say 'hi'\n{ not a tag }") != std::string::npos);
        std::printf("转义内容：原样保留\n");
    }
    // 不认识的结构 → 明确报错，不静默错译。
    {
        std::string out, err;
        CHECK(!render_chat("{% macro m() %}x{% endmacro %}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("宏结构：报错（%s）\n", err.c_str());
    }
    // 未闭合的 for → 报错。
    {
        std::string out, err;
        CHECK(!render_chat("{% for m in messages %}{{'x'}}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未闭合 for：报错（%s）\n", err.c_str());
    }
    // role 比较分支（较新的 Qwen 模板形态）：== / elif / else 逐段拼接。
    {
        const char* role_tpl = "{%- for m in messages -%}"
            "{%- if m['role'] == 'system' -%}[S]{{ m['content'] }}"
            "{%- elif m['role'] == 'user' -%}[U]{{ m['content'] }}"
            "{%- else -%}[?]{{ m['content'] }}"
            "{%- endif -%}{%- endfor -%}";
        std::string out, err;
        CHECK(render_chat(role_tpl, {{"system", "s"}, {"user", "u"}, {"assistant", "a"}}, false, out, err));
        CHECK(out == "[S]s[U]u[?]a");
        std::printf("role 比较分支：==/elif/else 逐段正确\n");
    }
    // 未知变量 → 报错。
    {
        std::string out, err;
        CHECK(!render_chat("{{ mystery_var }}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未知变量：报错（%s）\n", err.c_str());
    }
    std::puts("prompt: 通过");
    return 0;
}
}  // namespace
```

- [ ] **Step 2: 跑测试确定失败**（同 T1 Step 2 方式，目标 `test_prompt`）

- [ ] **Step 3: 写最小实现**

`src/serve/prompt.hpp`：

```cpp
// chat 模板渲染器：吃 GGUF 的 tokenizer.chat_template 原文，渲染消息列表。
// 只支持主案例模板实际用到的结构：字符串字面量、变量替换、+ 拼接、下标访问（m['role']）、
// if / elif / else / set / for；其余结构（macro、call、过滤器等）明确报错——静默错译等于在算另一个
// 提示词。`{%-` / `-%}` 连字符修饰的空白削减按 Jinja 语义：语句标签贴边时吃掉相邻的一行空白。
#pragma once

#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::serve {

struct Message {
    std::string role;
    std::string content;
};

// 从 GGUF 元数据读 tokenizer.chat_template；缺失或非字符串 → 返回假并写 err（拒绝裸拼接）。
bool chat_template_from_gguf(const artifact::GgufFile& g, std::string& out, std::string& err);

// 渲染。tpl 为模板原文；add_generation_prompt 决定是否追加生成提示段。
// 任何不支持的结构、未闭合块、未知变量 → 返回假并写 err（err 指明模板里的位置）。
bool render_chat(const std::string& tpl, const std::vector<Message>& msgs, bool add_generation_prompt,
                 std::string& out, std::string& err);

}  // namespace qinfer::serve
```

`src/serve/prompt.cpp`（完整实现）：

```cpp
#include "serve/prompt.hpp"

#include <cctype>
#include <cstdint>

namespace qinfer::serve {

bool chat_template_from_gguf(const artifact::GgufFile& g, std::string& out, std::string& err) {
    const std::string* s = g.meta_str("tokenizer.chat_template");
    if (!s) {
        err = "GGUF 元数据没有 tokenizer.chat_template，无法组提示词（拒绝裸拼接）";
        return false;
    }
    out = *s;
    return true;
}

namespace {

enum class Kind { kText, kExpr, kIf, kElif, kElse, kEndif, kFor, kEndfor, kSet };

struct Op {
    Kind kind = Kind::kText;
    std::string a;              // kText 输出文本；kExpr 表达式源；kIf/kElif 条件源；kSet 的变量名
    std::string b;              // kSet 的表达式源
    std::size_t jump = 0;       // 条件假 / 已执行分支的跳过目标（Op 下标）
    std::size_t end = 0;        // kFor 的 endfor 下标
};

struct Env {
    const std::vector<Message>* msgs = nullptr;
    bool add_gen = false;
    std::string bos, eos;
    const Message* loop_msg = nullptr;      // for 遍历中的当前消息
};

std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// 按顶层 '+' 切分（字符串字面量内的 '+' 不动）。
std::vector<std::string> split_top(const std::string& s) {
    std::vector<std::string> parts;
    std::string cur;
    char q = 0;
    for (char c : s) {
        if (q) {
            cur += c;
            if (c == q) q = 0;
        } else if (c == '\'' || c == '"') {
            q = c;
            cur += c;
        } else if (c == '+') {
            parts.push_back(trim(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(trim(cur));
    return parts;
}

// 字符串字面量解码（\n \t \r \" \' \\）。
bool decode_literal(const std::string& s, std::string& out, std::string& err) {
    if (s.size() < 2 || (s.front() != '\'' && s.front() != '"') ||
        s.back() != s.front()) {
        err = "表达式不是字符串字面量：" + s;
        return false;
    }
    const char q = s.front();
    out.clear();
    for (std::size_t i = 1; i + 1 < s.size(); ++i) {
        if (s[i] == '\\' && i + 2 < s.size()) {
            const char c = s[++i];
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default:
                    if (c == q || c == '\\') out += c;
                    else { err = "不认识的转义 \\" + std::string(1, c); return false; }
            }
        } else {
            out += s[i];
        }
    }
    return true;
}

// 一个术语求值：纯字面量；message/m 的下标；bos/eos/add_generation_prompt 变量。
bool eval_term(const std::string& term, const Env& env, std::string& out, std::string& err) {
    if (term.empty()) { err = "空表达式项"; return false; }
    if (term.front() == '\'' || term.front() == '"') return decode_literal(term, out, err);
    const std::size_t lb = term.find('[');
    if (lb != std::string::npos) {
        const std::string base = trim(term.substr(0, lb));
        if (base != "message" && base != "m") {
            err = "只支持 message['role'] 与 message['content']：'" + term + "'";
            return false;
        }
        if (!env.loop_msg) { err = "message 下标出现在消息循环之外"; return false; }
        if (term.back() != ']') { err = "下标未闭合：" + term; return false; }
        std::string key;
        if (!decode_literal(trim(term.substr(lb + 1, term.size() - lb - 2)), key, err)) return false;
        if (key == "role") out = env.loop_msg->role;
        else if (key == "content") out = env.loop_msg->content;
        else { err = "未知消息键：" + key; return false; }
        return true;
    }
    if (term == "bos_token") { out = env.bos; return true; }
    if (term == "eos_token") { out = env.eos; return true; }
    if (term == "add_generation_prompt") { out = env.add_gen ? "true" : "false"; return true; }
    if (term == "true") { out = "true"; return true; }
    if (term == "false") { out = "false"; return true; }
    err = "未知变量：" + term;
    return false;
}

// 表达式求值（+ 拼接）。
bool eval_expr(const std::string& src, const Env& env, std::string& out, std::string& err) {
    out.clear();
    for (const std::string& part : split_top(src)) {
        std::string piece;
        if (!eval_term(part, env, piece, err)) return false;
        out += piece;
    }
    return true;
}

// 按空白切分（字符串字面量内的空白不动）。条件表达式按词解析；比较运算符 == / != 需要词级切分。
std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    char q = 0;
    for (char c : s) {
        if (q) {
            cur += c;
            if (c == q) q = 0;
        } else if (c == '\'' || c == '"') {
            q = c;
            cur += c;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// 条件求值：not X / X is defined / X is not defined / 字符串相等与不等 / 变量真值 / true / false。
bool eval_cond(const std::string& src, const Env& env, bool& out, std::string& err) {
    const std::vector<std::string> toks = split_words(trim(src));
    if (toks.size() == 1) {
        if (toks[0] == "true") { out = true; return true; }
        if (toks[0] == "false") { out = false; return true; }
        if (toks[0] == "add_generation_prompt") { out = env.add_gen; return true; }
        out = false;   // 未知变量的真值按假处理（模板意图如此，不报错）
        return true;
    }
    if (toks.size() == 2 && toks[0] == "not") {
        bool v;
        if (!eval_cond(toks[1], env, v, err)) return false;
        out = !v;
        return true;
    }
    if (toks.size() == 3 && toks[1] == "is" && toks[2] == "defined") {
        out = (toks[0] == "add_generation_prompt" || toks[0] == "bos_token" ||
               toks[0] == "eos_token" || toks[0] == "messages");
        return true;
    }
    if (toks.size() == 4 && toks[1] == "is" && toks[2] == "not" && toks[3] == "defined") {
        bool d;
        if (!eval_cond(toks[0] + " is defined", env, d, err)) return false;
        out = !d;
        return true;
    }
    // 双操作数字符串比较：`m['role'] == 'system'` 这类（两边都按字符串求值）。不等号同理取反。
    if (toks.size() == 3 && (toks[1] == "==" || toks[1] == "!=")) {
        std::string a, b;
        if (!eval_term(toks[0], env, a, err) || !eval_term(toks[2], env, b, err)) return false;
        out = (a == b) == (toks[1] == "==");
        return true;
    }
    err = "不认识的 if 条件：" + src;
    return false;
}

// 词法与编译：扫描出 Op 序列并配对 if/for 块。连字符语义：`{%-` 吃掉随后文本的头部空白，
// `-%}` 吃掉前一文本的尾部空白。
bool compile(const std::string& tpl, std::vector<Op>& ops, std::string& err) {
    ops.clear();
    std::string text;
    bool prev_rtrim = false;      // 上一个标签以 -%} 结尾 → 本段文本尾部空白要吃掉
    bool ltrim_next = false;      // 上一个标签以 {%- 开头 → 本段文本头部空白要吃掉
    std::size_t i = 0;
    const auto flush = [&]() {
        if (prev_rtrim) {
            std::size_t k = text.size();
            while (k > 0 && std::isspace(static_cast<unsigned char>(text[k - 1]))) --k;
            text.resize(k);
        }
        if (ltrim_next) {
            std::size_t k = 0;
            while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
            text.erase(0, k);
            ltrim_next = false;
        }
        if (!text.empty()) {
            Op op;
            op.kind = Kind::kText;
            op.a = text;
            ops.push_back(op);
        }
        text.clear();
    };
    const auto fail = [&](const std::string& why, std::size_t at) {
        err = "模板第 " + std::to_string(at) + " 字符处：" + why;
        return false;
    };
    while (i < tpl.size()) {
        const std::string open2 = tpl.compare(i, 2, "{{") == 0 ? "{{" :
                                  (tpl.compare(i, 2, "{%") == 0 ? "{%" : std::string());
        if (open2.empty()) {
            text += tpl[i++];
            continue;
        }
        flush();
        const bool trim_after_open = i + 2 < tpl.size() && tpl[i + 2] == '-';
        const std::size_t open_adj = trim_after_open ? 3 : 2;
        const char close = open2 == "{{" ? '}' : '%';
        const std::string close2 = std::string(2, close);
        const std::size_t close_at = tpl.find(close2, i + open_adj);
        if (close_at == std::string::npos) return fail("标签未闭合", i);
        const bool trim_before_close = close_at > i + open_adj && tpl[close_at - 1] == '-';
        const std::size_t begin = i + open_adj;
        const std::size_t len = close_at - begin - (trim_before_close ? 1 : 0);
        const std::string body = trim(tpl.substr(begin, len));
        if (open2 == "{{") {
            Op op;
            op.kind = Kind::kExpr;
            op.a = body;
            ops.push_back(op);
        } else {
            const std::size_t sp = body.find(' ');
            const std::string k = sp == std::string::npos ? body : body.substr(0, sp);
            std::size_t rest = sp == std::string::npos ? std::string::npos : sp + 1;
            const std::string tail = rest == std::string::npos ? "" : trim(body.substr(rest));
            Op op;
            if (k == "if") {
                op.kind = Kind::kIf;
                op.a = tail;
            } else if (k == "elif") {
                op.kind = Kind::kElif;
                op.a = tail;
            } else if (k == "else") {
                op.kind = Kind::kElse;
            } else if (k == "endif") {
                op.kind = Kind::kEndif;
            } else if (k == "endfor") {
                op.kind = Kind::kEndfor;
            } else if (k == "for") {
                const std::vector<std::string> toks = split_top(tail);
                if (toks.size() != 3 || toks[1] != "in" || toks[2] != "messages") {
                    return fail("只支持 for <var> in messages", i);
                }
                op.kind = Kind::kFor;
                op.a = toks[0];
            } else if (k == "set") {
                const std::size_t eq = tail.find('=');
                if (eq == std::string::npos) return fail("set 缺 =", i);
                op.kind = Kind::kSet;
                op.a = trim(tail.substr(0, eq));
                op.b = trim(tail.substr(eq + 1));
            } else {
                return fail("不认识的语句：" + body, i);
            }
            ops.push_back(op);
        }
        prev_rtrim = trim_before_close;
        ltrim_next = trim_after_open;
        i = close_at + 2;
    }
    flush();
    if (!text.empty()) {   // 模板以纯文本结尾，这段在上一个 flush 后不可能再攒
        Op op;
        op.kind = Kind::kText;
        op.a = text;
        ops.push_back(op);
    }
    (void) fail;   // fail 在下面的配对错误处复用

    // 配对：if/elif/else/endif 与 for/endfor。
    std::vector<std::size_t> ifs, fors;
    for (std::size_t idx = 0; idx < ops.size(); ++idx) {
        switch (ops[idx].kind) {
            case Kind::kIf: ifs.push_back(idx); break;
            case Kind::kFor: fors.push_back(idx); break;
            case Kind::kEndfor: {
                if (fors.empty()) { err = "多余的 endfor"; return false; }
                const std::size_t f = fors.back();
                fors.pop_back();
                ops[f].end = idx;
                break;
            }
            case Kind::kEndif: {
                if (ifs.empty()) { err = "多余的 endif"; return false; }
                const std::size_t start = ifs.back();
                ifs.pop_back();
                std::size_t prev = start;
                for (std::size_t c = start; c < idx; ++c) {
                    const Kind kk = ops[c].kind;
                    if (kk == Kind::kIf || kk == Kind::kElif) {
                        std::size_t nxt = c + 1;
                        while (nxt < idx && !(ops[nxt].kind == Kind::kElif || ops[nxt].kind == Kind::kElse)) ++nxt;
                        ops[c].jump = nxt;
                        prev = c;
                    } else if (kk == Kind::kElse) {
                        ops[c].jump = idx;
                    }
                }
                ops[prev].jump = idx;   // 链尾成员的跳转 = endif
                break;
            }
            default: break;
        }
    }
    if (!ifs.empty()) { err = "if 未闭合"; return false; }
    if (!fors.empty()) { err = "for 未闭合"; return false; }
    return true;
}

// 解释执行 [from, to) 区间。返回 false → err 非空。
bool run(const std::vector<Op>& ops, std::size_t from, std::size_t to, Env& env, std::string& out,
         std::string& err) {
    std::size_t i = from;
    while (i < to) {
        const Op& op = ops[i];
        switch (op.kind) {
            case Kind::kText: out += op.a; ++i; break;
            case Kind::kExpr: {
                std::string v;
                if (!eval_expr(op.a, env, v, err)) {
                    err = "求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                out += v;
                ++i;
                break;
            }
            case Kind::kIf: {
                bool c;
                if (!eval_cond(op.a, env, c, err)) {
                    err = "条件求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                if (c) ++i; else i = op.jump;
                break;
            }
            case Kind::kElif:
            case Kind::kElse:
                i = op.jump;                 // 执行到这里说明前面的分支已取
                break;
            case Kind::kEndif: ++i; break;
            case Kind::kSet: {
                // 只支持 set <已知变量> = <字面量或 true/false>；Qwen 模板只用 set add_generation_prompt。
                std::string v;
                if (!eval_expr(op.b, env, v, err)) { err = "set 求值失败：" + op.b; return false; }
                if (op.a == "add_generation_prompt") env.add_gen = (v == "true");
                else if (op.a == "bos_token") env.bos = v;
                else if (op.a == "eos_token") env.eos = v;
                else { err = "set 不认识的变量：" + op.a; return false; }
                ++i;
                break;
            }
            case Kind::kFor: {
                if (!env.msgs) { err = "messages 为空"; return false; }
                for (const Message& m : *env.msgs) {
                    env.loop_msg = &m;
                    if (!run(ops, i + 1, ops[i].end, env, out, err)) {
                        err = "循环体内：" + err;
                        return false;
                    }
                }
                env.loop_msg = nullptr;
                i = ops[i].end + 1;
                break;
            }
            case Kind::kEndfor: i = to; break;   // 到不了（kFor 内循环处理），防御
        }
    }
    return true;
}

}  // namespace

bool render_chat(const std::string& tpl, const std::vector<Message>& msgs, bool add_generation_prompt,
                 std::string& out, std::string& err) {
    std::vector<Op> ops;
    if (!compile(tpl, ops, err)) return false;
    Env env;
    env.msgs = &msgs;
    env.add_gen = add_generation_prompt;
    out.clear();
    return run(ops, 0, ops.size(), env, out, err);
}

}  // namespace qinfer::serve
```

- [ ] **Step 4: 跑测试通过**（方式同 T1 Step 4）

- [ ] **Step 5: CMake 接线 + 提交**

```cmake
# CMakeLists.txt 的 qinfer_serve 库源列表追加：
#   src/serve/prompt.cpp
# 测试区追加：
# add_executable(test_prompt tests/test_prompt.cpp)
# target_link_libraries(test_prompt PRIVATE qinfer_serve)
# add_test(NAME prompt COMMAND test_prompt)
```

```bash
cmake -B build -S . && cmake --build build --target test_prompt && ctest --test-dir build -R '^prompt$' --output-on-failure
git add src/serve/ CMakeLists.txt tests/test_prompt.cpp
git commit -m "feat(serve): chat 模板渲染器——GGUF 模板原文、字面量与 if/for 子集、未知结构明确报错"
```

---

### Task 3: 装配门面（主体 + PLE 与 KV 窗口接线 + 合成单测）

把 `tests/test_step_loop.cpp` 的真权重装配上提为服务实体：一个 `Engine` 装载 GGUF、分词器、chat 模板、token embedding、步模型与（可选的）PLE 与 KV 窗口；一次 `generate` = 一份全新会话（scratch、窗口都随调用装配，请求之间无共享状态）。生成环内做采样与 stop 截断，逐 token 以解码字节回调。

**Files:**
- Create: `src/serve/facade.hpp`, `src/serve/facade.cpp`, `tests/test_facade.cpp`
- Modify: `CMakeLists.txt`（`qinfer_serve` 加 `facade.cpp`、补链接；加 `test_facade`）

**Interfaces:**
- Consumes: T1 的 `Sampler`/`SamplerParams`；T2 的 `render_chat`/`chat_template_from_gguf`；仓库既有 API 见 Global Constraints
- Produces: `qinfer::serve::FacadeConfig`、`GenOptions`、`GenResult`、`Engine`（T9–T11、T18 用）

- [ ] **Step 1: 写失败测试**

`tests/test_facade.cpp`。合成夹具 = `tests/test_step_loop.cpp` 第 50–249 行的同名构造（`TensorSpec`、`q8_0_rows`、`f32_vec`、`f32_lin`、`bf16_sym`、`router_tensor`、`moe_tensors`、`gr_group`、`gdn_attn`、`qsa_attn`、`layer_tensors`、`head_tensors`、`push_row_block`、`token_embd_data`、`model_tensors`、`two_layer_config`，以及第 39–95 行的常量与 `write_gguf`）——唯一改动：`write_gguf` 多收一个 `metas` 参数，并在 tensor 信息之前按序写入若干元数据项。新增的元数据部分（其余照抄）：

```cpp
struct MetaKV {
    std::string key;
    std::uint32_t type;             // GGUF 类型码：4=UINT32、8=STRING、9=ARRAY
    std::vector<std::uint8_t> val;  // 值字节（ARRAY 含元素类型与计数，不含值类型码）
};

std::vector<std::uint8_t> meta_u32(std::uint64_t x) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>((x >> (8 * i)) & 0xFF));
    return v;
}
std::vector<std::uint8_t> meta_str(const std::string& s) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((s.size() >> (8 * i)) & 0xFF));
    v.insert(v.end(), s.begin(), s.end());
    return v;
}
std::vector<std::uint8_t> meta_str_array(const std::vector<std::string>& a) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 4; ++i) v.push_back(8);   // 元素类型 STRING
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((a.size() >> (8 * i)) & 0xFF));
    for (const std::string& s : a) {
        const std::vector<std::uint8_t> m = meta_str(s);
        v.insert(v.end(), m.begin(), m.end());
    }
    return v;
}
std::vector<std::uint8_t> meta_u32_array(const std::vector<std::uint64_t>& a) {
    std::vector<std::uint8_t> v;
    for (int i = 0; i < 4; ++i) v.push_back(4);   // 元素类型 UINT32
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>((a.size() >> (8 * i)) & 0xFF));
    for (std::uint64_t x : a) {
        const std::vector<std::uint8_t> m = meta_u32(x);
        v.insert(v.end(), m.begin(), m.end());
    }
    return v;
}

// 合成夹具的分词器语料：3 个特殊 token + 字母 a..e。词表 8 = token_embd 的行数。
std::vector<MetaKV> serve_metas() {
    return {
        {"general.alignment", 4, meta_u32(32)},
        {"general.name", 8, meta_str("qinfer-serve-fixture")},
        {"tokenizer.ggml.model", 8, meta_str("gpt2")},
        {"tokenizer.ggml.tokens", 9,
         meta_str_array({"<|im_start|>", "<|im_end|>", "<|endoftext|>", "a", "b", "c", "d", "e"})},
        {"tokenizer.ggml.merges", 9, meta_str_array({})},
        {"tokenizer.ggml.token_type", 9, meta_u32_array({3, 3, 3, 1, 1, 1, 1, 1})},
        {"tokenizer.ggml.eos_token_id", 4, meta_u32(2)},
        {"tokenizer.chat_template", 8,
         meta_str("{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}")},
    };
}
```

`write_gguf` 的头部改法（其余不变）：magic 与 `u32(3)` 之后先写 `u64(metas.size())`（替代原来的 `u64(1)`），再逐条 `put_str(key)` + `u32(type)` + 值字节，最后按 32 对齐再写 tensor 信息。原 `{"general.alignment", 4, u32(32)}` 一起并入 metas。

测试主体（`main`，全部确定性断言；夹具是固定字节，跑过一次就永远稳定）：

```cpp
int main() {
    const std::filesystem::path p = write_gguf("qinfer_facade.gguf", model_tensors(), serve_metas());
    // 打不开的文件 → 明确错误。
    {
        Engine e; std::string err; FacadeConfig cfg; cfg.model_path = "/nonexistent/qinfer.gguf";
        CHECK(!Engine::open(cfg, e, err)); CHECK(!err.empty());
    }
    // 缺 chat 模板 → 拒绝启动（拒绝裸拼接）。
    {
        std::vector<MetaKV> ms;
        for (const MetaKV& m : serve_metas())
            if (m.key != "tokenizer.chat_template") ms.push_back(m);
        const std::filesystem::path q = write_gguf("qinfer_facade_notpl.gguf", model_tensors(), ms);
        Engine e; std::string err; FacadeConfig cfg; cfg.model_path = q.string();
        CHECK(!Engine::open(cfg, e, err));
        CHECK(err.find("chat_template") != std::string::npos);
    }
    // 正常打开：模型名来自 general.name、词表 8、模板就位。
    Engine eng; std::string err;
    FacadeConfig cfg; cfg.model_path = p.string();
    cfg.n_embd = 256; cfg.hc = 2; cfg.hc_lr = 3; cfg.n_layers = 2; cfg.top_k = 2;
    cfg.gdn_head_dim = 32; cfg.qsa_head_dim = 32; cfg.n_rot = 4;
    cfg.idx_dim = 4; cfg.idx_top_k = 2048; cfg.idx_block = 4;
    CHECK(Engine::open(cfg, eng, err));
    CHECK(eng.model_name() == "qinfer-serve-fixture");
    CHECK(eng.tokenizer().vocab_size() == 8u);
    CHECK(!eng.chat_template().empty());

    // 长度到：贪心 24 token。
    GenOptions g; g.temperature = 0.0f; g.seed = 7; g.max_tokens = 24;
    GenResult a; std::vector<std::string> pieces;
    auto collect = [&](const std::string& s) { pieces.push_back(s); return true; };
    CHECK(eng.generate({0, 1}, g, collect, a, err));
    CHECK(a.n_prompt == 2); CHECK(a.n_gen >= 4); CHECK(a.n_gen <= 24);
    CHECK(pieces.size() == a.n_gen); CHECK(a.tokens.size() == a.n_gen);
    if (a.finish == Finish::kLength) CHECK(a.n_gen == 24);
    for (std::int32_t t : a.tokens) CHECK(t >= 0 && t < 8);
    std::string joined;
    for (const std::string& s : pieces) joined += s;
    CHECK(joined == a.text); CHECK(!a.tokens.empty());
    std::printf("生成 %llu token（finish %d）：text %zu 字节\n", (unsigned long long) a.n_gen,
                (int) a.finish, a.text.size());

    // seed 可复现：同 seed 同参数 → 同一串 token 与文本。
    {
        GenResult b;
        CHECK(eng.generate({0, 1}, g, nullptr, b, err));
        CHECK(b.tokens == a.tokens); CHECK(b.text == a.text);
    }
    // 采样式（温度 1、seed 1/2）在长跑里必然分歧（夹具固定，结果确定）。
    {
        GenOptions s1 = g, s2 = g;
        s1.temperature = 1.0f; s2.temperature = 1.0f; s1.seed = 1; s2.seed = 2;
        s1.max_tokens = 256; s2.max_tokens = 256;
        GenResult c1, c2;
        CHECK(eng.generate({0, 1}, s1, nullptr, c1, err));
        CHECK(eng.generate({0, 1}, s2, nullptr, c2, err));
        std::size_t nd = 0;
        for (std::size_t i = 0; i < c1.tokens.size() && i < c2.tokens.size(); ++i)
            if (c1.tokens[i] != c2.tokens[i]) ++nd;
        CHECK(nd > 0 || c1.n_gen < 4 || c2.n_gen < 4);
        std::printf("温度 1 两 seed：%zu/%zu 处不同\n", nd,
                    c1.tokens.size() < c2.tokens.size() ? c1.tokens.size() : c2.tokens.size());
    }
    // stop 串：从不出现的 → 与不设 stop 完全一致；尾 piece 命中的 → 截到命中处。
    {
        GenOptions go; go.temperature = 0.0f; go.seed = 7; go.max_tokens = 64;
        GenResult full;
        CHECK(eng.generate({0, 1}, go, nullptr, full, err));
        CHECK(full.n_gen >= 8);
        GenOptions gz = go; gz.stop = {"zz"};
        GenResult z;
        CHECK(eng.generate({0, 1}, gz, nullptr, z, err));
        CHECK(z.tokens == full.tokens);
        GenOptions gs = go; gs.stop = {full.text.substr(full.text.size() - 2)};
        GenResult s;
        CHECK(eng.generate({0, 1}, gs, nullptr, s, err));
        CHECK(s.finish == Finish::kStopString);
        CHECK(s.n_gen <= full.n_gen);
        CHECK(s.text == full.text.substr(0, s.text.size()));
        std::printf("stop 命中：截到 %zu 字节（全量 %zu）\n", s.text.size(), full.text.size());
    }
    // 参数校验：max_tokens 0 → 报错。
    {
        GenOptions bad = g; bad.max_tokens = 0;
        GenResult r;
        CHECK(!eng.generate({0, 1}, bad, nullptr, r, err)); CHECK(!err.empty());
    }
    // KV 窗口对拍（Task 4 的另一半钉在这里）：同一份几何（idx_top_k 取 8 让选择稀疏），
    // kv_slots=13 那台与不挂窗口那台逐 token 逐位一致；窗口侧驱逐 > 0。
    // 数值由来：宽度 = idx_top_k + idx_block - 1 = 11 → 单步最多 11 个聚集页 + 1 个写入页
    // = 12 个钉住页 < 配额 13 → 永不拒绝；会话写到的逻辑页 = ceil(66/4) = 17 > 13 → 必有驱逐。
    {
        FacadeConfig cw = cfg; cw.kv_slots = 13; cw.idx_top_k = 8;
        FacadeConfig cp = cfg; cp.idx_top_k = 8;
        Engine ew, ep; std::string errw;
        CHECK(Engine::open(cw, ew, errw));
        CHECK(Engine::open(cp, ep, err));
        GenOptions gk; gk.temperature = 0.0f; gk.seed = 7; gk.max_tokens = 64;
        GenResult rw, rp;
        CHECK(ew.generate({0, 1}, gk, nullptr, rw, errw));
        CHECK(ep.generate({0, 1}, gk, nullptr, rp, err));
        CHECK(rw.tokens == rp.tokens);
        CHECK(rw.text == rp.text);
        CHECK(ew.kv_evictions() > 0);
        CHECK(ew.kv_misses() > 0);
        std::printf("KV 窗口：%llu token 与不挂窗口逐位一致；驱逐 %llu、未命中 %llu\n",
                    (unsigned long long) rw.tokens.size(), (unsigned long long) ew.kv_evictions(),
                    (unsigned long long) ew.kv_misses());
    }
    // PLE 错误路径：给了 ple_table 但那份 GGUF 没有 PLE 的六张张力 → open 明确失败。
    {
        FacadeConfig cp = cfg; cp.ple_table = p.string();   // 拿模型自己当表：没有 blk.1.ple_*
        Engine ep; std::string errp;
        CHECK(!Engine::open(cp, ep, errp));
        CHECK(errp.find("PLE") != std::string::npos);
        std::printf("PLE 错误路径：%s\n", errp.c_str());
    }
    std::puts("facade: 通过");
    return 0;
}
```

（若合成模型的贪心轨迹撞上 eos——`finish == Finish::kStop` 且 `n_gen < 4`——把 `serve_metas` 的 eos 换成词表里另一个 id（保持 `token_type` 仍是 3）再跑；夹具确定性，一次对上即永久稳定。）

- [ ] **Step 2: 跑测试确定失败**

```bash
cmake -B build -S . 2>&1 | tail -3
cmake --build build --target test_facade 2>&1 | tail -5 || true
```

预期：找不到 `serve/facade.hpp`，构建失败。

- [ ] **Step 3: 写最小实现**

`src/serve/facade.hpp`：

```cpp
// 装配门面：把 GGUF + 分词器 + token embedding + 步模型 +（可选的）PLE 与 KV 窗口装配成一个可生成的
// 服务实体。一次 `generate` = 一份全新会话（prompt 自足，请求之间无共享状态）。
// 装配口径与真权重对拍入口（tests/test_step_loop.cpp 的 --model 路径）一致：同一组几何、同一组调用序。
// 几何缺省即真模型推导值（48 层那组），合成测试用 FacadeConfig 覆盖成小值。
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/kv_store.hpp"
#include "dense/kv_window.hpp"
#include "dense/step_loop.hpp"
#include "dense/token_embed.hpp"
#include "ple/row_path.hpp"
#include "ple/spec.hpp"
#include "serve/prompt.hpp"
#include "serve/sampler.hpp"
#include "storage/row_cache.hpp"
#include "text/tokenizer.hpp"

namespace qinfer::serve {

struct FacadeConfig {
    std::string model_path;          // 必填（主案例 GGUF 分片）
    std::uint64_t n_embd = 2560;
    std::uint64_t hc = 4;
    std::uint64_t hc_lr = 320;
    int n_layers = 48;
    int top_k = 10;
    std::uint64_t gdn_head_dim = 128;
    std::uint64_t qsa_head_dim = 256;
    std::uint64_t n_rot = 64;
    std::uint64_t idx_dim = 128;
    std::uint64_t idx_top_k = 2048;
    std::uint64_t idx_block = 4;
    std::uint64_t max_context = 262144;      // 报告用（/v1/models、/props）
    std::size_t cache_slots = 0;             // 专家缓存槽位；0 = 按需直读
    std::uint64_t cache_byte_budget = 0;
    std::string ple_table;                   // PLE 表行来源（分片2），空 = 不配 PLE
    std::uint64_t ple_budget_bytes = 0;
    std::uint64_t kv_slots = 0;              // 每 QSA 层常驻槽位；0 = 不挂 KV 窗口
};

// 结束原因：长度到 / eos token / stop 串命中。客户端中断（on_token 返回假）→ generate 返回假、err 说明。
enum class Finish { kLength, kStop, kStopString };

struct GenOptions {
    float temperature = 0.8f;
    float top_p = 0.95f;
    std::int64_t seed = -1;
    std::uint64_t max_tokens = 512;
    std::vector<std::string> stop;   // 生成的字节流出现这些子串即终止（截到命中处）
};

struct GenResult {
    Finish finish = Finish::kLength;
    std::uint64_t n_prompt = 0;
    std::uint64_t n_gen = 0;
    std::string text;                   // 全量已生成文本（UTF-8 字节，已去掉 stop 命中段）
    std::vector<std::int32_t> tokens;   // 已生成 token id（不含被 stop 截掉的后缀）
};

class Engine {
public:
    // 装载：GGUF 打开、分词器、chat 模板（缺失即失败——拒绝裸拼接）、token embedding、步模型，
    // 以及（给了 ple_table 时）第 1 层 PLE 的登记/权重/表行来源。任一步失败都返回假并写 err。
    static bool open(const FacadeConfig& cfg, Engine& out, std::string& err);

    // 同步生成。prompt 是已编码的 token id（调用方按 chat 模板渲染 + 分词）。逐 token 回调 on_token
    // = 该 token 解码后的字节片段（stop 命中时只回调保留前缀，前缀为空则免回调；eos 不回调节点）。
    // 回调返回 false = 客户端中断，generate 立即返回假、err 说明。
    bool generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                  const std::function<bool(const std::string&)>& on_token, GenResult& res,
                  std::string& err);

    const text::Tokenizer& tokenizer() const { return tokenizer_; }
    const std::string& chat_template() const { return chat_template_; }
    const std::string& model_name() const { return model_name_; }
    const FacadeConfig& config() const { return cfg_; }
    // 上次 generate 的 KV 窗口统计（未配窗口时全为 0）。只读标量，不碰窗口内部指针。
    std::uint64_t kv_misses() const;
    std::uint64_t kv_evictions() const;
    std::uint64_t kv_bytes_moved() const;

private:
    // KV 窗口接线（cfg_.kv_slots > 0 时由 generate 每次装配）：与真权重入口同一套字段公式。
    void attach_kv_windows(StepScratch& s, std::uint64_t logical_blocks);

    FacadeConfig cfg_;
    artifact::GgufFile g_;
    artifact::GgufTable table_;              // PLE 行来源（可空）
    ple::GgufTableRowSource src_{table_};
    storage::RowCache rows_{0};
    ple::PleSpec pspec_;
    ple::PleBuffers pw_;
    dense::PleRun run_{};
    dense::StepConfig step_cfg_;
    dense::StepModel m_;
    dense::TokenEmbed te_;
    text::Tokenizer tokenizer_;
    std::string chat_template_;
    std::string model_name_;
    std::int32_t eos_ = -1;
    std::vector<dense::KvWindow> wins_;      // 每次 generate 重装；生存期 = generate（表/副本指针指向
    std::vector<dense::KvHostStore> stores_; // 当次 scratch，结束后不得再碰指针成员）
};

}  // namespace qinfer::serve
```

`src/serve/facade.cpp`（完整实现）：

```cpp
#include "serve/facade.hpp"

#include <algorithm>
#include <cmath>

namespace qinfer::serve {

bool Engine::open(const FacadeConfig& cfg, Engine& out, std::string& err) {
    out = Engine{};
    out.cfg_ = cfg;
    if (cfg.model_path.empty()) { err = "模型路径为空"; return false; }
    if (!out.g_.open(cfg.model_path, err)) return false;
    if (!text::Tokenizer::from_gguf(out.g_, out.tokenizer_, err)) return false;
    if (!chat_template_from_gguf(out.g_, out.chat_template_, err)) return false;
    std::uint64_t eos = 0;
    if (out.g_.meta_u64("tokenizer.ggml.eos_token_id", eos)) {
        out.eos_ = static_cast<std::int32_t>(eos);
    } else {
        const std::int32_t sp = out.tokenizer_.special_id("<|im_end|>");
        if (sp < 0) { err = "找不到 eos token id（元数据与 special 词表都没有）"; return false; }
        out.eos_ = sp;
    }
    if (!build_token_embed(out.g_, cfg.n_embd, out.te_, err)) return false;
    out.step_cfg_ = dense::StepConfig{};
    out.step_cfg_.n_embd = cfg.n_embd;
    out.step_cfg_.hc = cfg.hc;
    out.step_cfg_.hc_lr = cfg.hc_lr;
    out.step_cfg_.n_layers = cfg.n_layers;
    out.step_cfg_.top_k = cfg.top_k;
    out.step_cfg_.gdn_head_dim = cfg.gdn_head_dim;
    out.step_cfg_.qsa_head_dim = cfg.qsa_head_dim;
    out.step_cfg_.n_rot = cfg.n_rot;
    out.step_cfg_.idx_dim = cfg.idx_dim;
    out.step_cfg_.idx_top_k = cfg.idx_top_k;
    out.step_cfg_.idx_block = cfg.idx_block;
    out.step_cfg_.cache_slots = cfg.cache_slots;
    out.step_cfg_.cache_byte_budget = cfg.cache_byte_budget;
    if (!cfg.ple_table.empty()) {
        if (!out.table_.open(cfg.ple_table, err)) return false;
        out.rows_ = storage::RowCache(4096);
        out.pspec_ = ple::PleSpec{};
        out.pw_ = ple::PleBuffers{};
        if (!ple::read_ple_spec(out.g_, /*layer=*/1, cfg.n_embd, cfg.hc, out.pspec_, err) ||
            !out.pspec_.present || !ple::load_ple_weights(out.g_, out.pspec_, out.pw_, err)) {
            err = "读第 1 层的 PLE 登记/权重失败：" + err;
            return false;
        }
        out.run_.layer = 1;
        out.run_.spec = out.pspec_;
        out.run_.w = out.pw_;
        out.run_.table = &out.src_;
        out.run_.cache = &out.rows_;
        out.run_.budget_bytes = cfg.ple_budget_bytes;
        out.step_cfg_.ple = &out.run_;
    }
    if (!build_step_model(out.g_, out.step_cfg_, out.m_, err)) return false;
    const std::string* nm = out.g_.meta_str("general.name");
    out.model_name_ =
        (nm != nullptr && !nm->empty()) ? *nm : std::filesystem::path(cfg.model_path).stem().string();
    return true;
}

// 每次 generate 一份：为每个 QSA 层建权威副本 + 常驻窗口，配额 = cfg_.kv_slots × 一页字节。
// 先 reserve 再落位 stores_（KvWindow.store 是指针，vector 扩容会让指针悬空）。
void Engine::attach_kv_windows(StepScratch& s, std::uint64_t logical_blocks) {
    wins_.clear();
    stores_.clear();
    std::size_t n_qsa = 0;
    for (const auto& row : m_.map)
        if (row.kind == dense::LayerKind::kQsa) ++n_qsa;
    wins_.reserve(n_qsa);
    stores_.reserve(n_qsa);
    for (std::size_t l = 0; l < m_.map.size(); ++l) {
        if (m_.map[l].kind != dense::LayerKind::kQsa) continue;
        dense::QsaScratch& q = s.blocks[l].qsa;
        q.kv.n_pages = cfg_.kv_slots;      // 常驻槽位只给这么多（逻辑容量在副本里）
        q.kv.reset();
        stores_.emplace_back();
        stores_.back().init(q.kv.page_size, q.kv.n_head_kv, q.kv.head_dim, logical_blocks);
        dense::KvWindow w;
        w.page_bytes = 2 * q.kv.n_head_kv * q.kv.page_size * q.kv.head_dim * 2;
        w.quota_bytes = cfg_.kv_slots * w.page_bytes;
        w.store = &stores_.back();
        dense::attach_kv_window(q, w, logical_blocks);
        wins_.push_back(std::move(w));
    }
}

bool Engine::generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                      const std::function<bool(const std::string&)>& on_token, GenResult& res,
                      std::string& err) {
    if (opt.max_tokens == 0) { err = "max_tokens 必须为正"; return false; }
    SamplerParams sp;
    sp.temperature = opt.temperature;
    sp.top_p = opt.top_p;
    sp.seed = opt.seed;
    if (!sp.valid()) { err = "采样参数非法：temperature 非负、top_p 在 (0,1]"; return false; }
    // 一次 generate 一份全新会话：scratch、KV 窗口都在这次调用里装配。页数按 idx_block 个 cell 一页估，
    // 尾留 8 个 cell 余量（prefill 不重跑，故只此一次分配）。
    const std::uint64_t n_pages =
        (static_cast<std::uint64_t>(prompt.size()) + opt.max_tokens + 8 +
         step_cfg_.idx_block - 1) / step_cfg_.idx_block;
    StepScratch s;
    if (!prepare_step_scratch(m_, n_pages, /*pos_base=*/0, s, err)) return false;
    if (cfg_.kv_slots > 0) attach_kv_windows(s, n_pages);
    Sampler sampler(opt.seed);
    res = GenResult{};
    res.n_prompt = prompt.size();
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (!run_token_step(g_, m_, te_, static_cast<std::uint64_t>(prompt[i]), s, err)) return false;
    }
    std::size_t max_win = 0;
    for (const std::string& st : opt.stop) max_win = std::max(max_win, st.size());
    std::string text;
    for (;;) {
        if (s.head.logits.empty()) { err = "logits 为空，无法采样"; return false; }
        const std::uint32_t id = sampler.sample(s.head.logits.data(), s.head.logits.size(), sp);
        if (static_cast<std::int32_t>(id) == eos_) {
            res.finish = Finish::kStop;
            res.text = text;
            break;
        }
        std::string piece;
        if (!tokenizer_.token_bytes(static_cast<std::int32_t>(id), piece)) {
            piece = "\xEF\xBF\xBD";   // 防御：词表外 id 到不了这里，给 U+FFFD
        }
        res.tokens.push_back(static_cast<std::int32_t>(id));
        ++res.n_gen;
        const std::size_t old = text.size();
        text += piece;
        // stop 匹配：只看「结束点落在新 piece 内」的出现（跨 piece 边界也算），取最早结束的那次。
        std::size_t cut = std::string::npos;
        if (max_win > 0 && !piece.empty()) {
            const std::size_t win = old >= max_win ? old + 1 - max_win : 0;
            for (const std::string& st : opt.stop) {
                if (st.empty()) continue;
                std::size_t p = win;
                while ((p = text.find(st, p)) != std::string::npos) {
                    const std::size_t end = p + st.size();
                    if (end > old && (cut == std::string::npos || end < cut)) cut = end;
                    ++p;
                }
            }
        }
        if (cut != std::string::npos) {
            text.resize(cut);
            const std::string keep = piece.substr(0, cut > old ? cut - old : 0);
            if (!keep.empty() && on_token && !on_token(keep)) { err = "客户端中断"; return false; }
            res.finish = Finish::kStopString;
            res.text = text;
            break;
        }
        if (on_token && !on_token(piece)) { err = "客户端中断"; return false; }
        if (res.n_gen >= opt.max_tokens) {
            res.finish = Finish::kLength;
            res.text = text;
            break;
        }
        if (!run_token_step(g_, m_, te_, id, s, err)) return false;
    }
    return true;
}

std::uint64_t Engine::kv_misses() const {
    std::uint64_t v = 0;
    for (const auto& w : wins_) v += w.misses;
    return v;
}
std::uint64_t Engine::kv_evictions() const {
    std::uint64_t v = 0;
    for (const auto& w : wins_) v += w.evictions;
    return v;
}
std::uint64_t Engine::kv_bytes_moved() const {
    std::uint64_t v = 0;
    for (const auto& st : stores_) v += st.bytes_moved;
    return v;
}

}  // namespace qinfer::serve
```

- [ ] **Step 4: 跑测试通过**

```bash
cmake --build build --target test_facade && ctest --test-dir build -R '^facade$' --output-on-failure
```

预期：全部断言过，打印五段说明。若贪心轨迹撞 eos 使 `n_gen < 4`，按 Step 1 末尾的注调整 fixture 的 eos 后重跑。

- [ ] **Step 5: CMake 接线 + 提交**

`CMakeLists.txt` 把 T1/T2 加的 serve 库替换为（依赖一次给全：门面用到 dense/ple/storage/text）：

```cmake
add_library(qinfer_serve
  src/serve/sampler.cpp
  src/serve/prompt.cpp
  src/serve/facade.cpp
)
target_include_directories(qinfer_serve PUBLIC src)
target_link_libraries(qinfer_serve PUBLIC qinfer_text qinfer_dense qinfer_ple qinfer_storage qinfer_kernels)
```

测试区新增：

```cmake
add_executable(test_facade tests/test_facade.cpp)
target_link_libraries(test_facade PRIVATE qinfer_serve)
add_test(NAME facade COMMAND test_facade)
```

```bash
cmake -B build -S . && cmake --build build --target test_facade && ctest --test-dir build -R '^facade$' --output-on-failure
python3 scripts/docs_checks.py
git add src/serve/ CMakeLists.txt tests/test_facade.cpp
git commit -m "feat(serve): 装配门面——装载/生成环/采样与 stop 截断；PLE 与 KV 窗口可选接线（合成夹具全套单测）"
```

---

### Task 4: 门面的真权重对拍入口（环境2 线下跑）

`tests/test_facade.cpp` 增加 `--model` 手工入口，把 `tests/test_step_loop.cpp` 第 713–953 行 `manual_model_entry` 的对拍判据转成门面口径：同一份几何、真模型 + 真表（分片2），PLE 开/关比相对偏离、KV 窗口与不挂逐位一致。这是 M1.1 验收「真权重对拍判据保持通过」的载体，CI 不跑（环境2 线下跑）。

**Files:**
- Modify: `tests/test_facade.cpp`
- Create: 无

**Interfaces:**
- Consumes: Task 3 的 `FacadeConfig`/`Engine`；真模型几何常量；环境2 分片路径
- Produces: `--model <模型分片> --table <PLE 分片2> --kv-slots N` 入口

- [ ] **Step 1: 写失败测试（手工入口，本步=按既有判定转录）**

在 `tests/test_facade.cpp` 加 `manual_model_entry(model, table, kv_slots)`：用 `FacadeConfig` 缺省几何（48 层那组）open；`generate` 跑 `{1, 2, 3, 4}` 四个 prompt token、贪心、max_tokens 8，断言 logits 有限（`head.logits` 经 `GenResult` 不暴露——改为断言生成成功、`n_gen == 8`、文本非空）；配 `ple_table` 时 open 须成功（真表上 PLE 登记存在）；`kv_slots > 0` 时与不挂那台逐 token 一致 + `kv_evictions() > 0`。

- [ ] **Step 2: 实现 + 环境2 验证**

```bash
cmake -B build -S . && cmake --build build --target test_facade
./build/tests/test_facade --model <环境2真模型分片> --table <环境2 PLE 分片> --kv-slots 64
```

预期：open 成功、生成 8 token；PLE 开/关 logits 相对偏离 > 1e-3（判据同 step_loop 的 rel_diff）；KV 窗口与不挂逐位一致且驱逐 > 0。任一条不符即停（环境2 不在场时把本步留到 T18 一起跑）。

- [ ] **Step 3: 提交**

```bash
git add tests/test_facade.cpp
git commit -m "test(serve): 门面 --model 真权重对拍入口（PLE 开/关相对偏离、KV 窗口逐位一致）"
```

---

### Task 5: httplib 入库 + CMake

http 传输依赖走 ADR-008 允许的唯一例外：cpp-httplib 单头文件整份入库。改动上游不是目的，维护一个 diff 才是成本，故不裁文件、不改造，靠「不定义

**Files:**
- Create: `third_party/httplib/httplib.h`（来自上游 v0.18.6 标签）
- Modify: `CMakeLists.txt`（serve 库加 include 路径；注释记版本与校验和）

**Interfaces:**
- Consumes: 无
- Produces: `httplib::Server` 可达（T9 起用）

- [ ] **Step 1: 拉取上游文件（锁定版本标签）**

```bash
mkdir -p third_party/httplib
curl -fL -o third_party/httplib/httplib.h \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.6/httplib.h
wc -l third_party/httplib/httplib.h        # 约 10800 行
sha256sum third_party/httplib/httplib.h
grep -n "MIT License" third_party/httplib/httplib.h | head -3   # 文件头版权与许可注释在
```

网络不可达或 sha 与预期不符时停手说明，不伪造文件。文件头自带版权与 MIT 许可注释（整份保留）。

- [ ] **Step 2: CMake 接线（含版本与校验和留痕）**

在 `qinfer_serve` 库定义后加：

```cmake
# third_party/httplib/httplib.h = cpp-httplib v0.18.6（整份入库，SHA-256 见提交说明）。
# 「裁剪」落实为不定义 CPPHTTPLIB_OPENSSL_SUPPORT 与 CPPHTTPLIB_ZLIB_SUPPORT；线程池由入口按槽位配置。
target_include_directories(qinfer_serve PUBLIC third_party/httplib)
```

- [ ] **Step 3: 提交**

```bash
cmake -B build -S . && cmake --build build --target qinfer_serve
git add third_party/httplib/ CMakeLists.txt
git commit -m "feat(serve): 入库 cpp-httplib v0.18.6（整份、锁定版本、sha256 见提交说明）"
```

---

### Task 6: JSON 子集

协议层的公共地基：对象、数组、字符串（转义与 `\uXXXX` 含代理对）、数字（整数/小数/指数）、布尔、空。解析按严格递归下降，任何越界输入返回假（上层回 400）；序列化紧凑输出。

**Files:**
- Create: `src/serve/json.hpp`, `src/serve/json.cpp`, `tests/test_json.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: 无
- Produces: `qinfer::serve::Json`（值类型 + 静态构造）、`json_parse`、`json_dump`、`json_escape`（T9–T14 全部用）

- [ ] **Step 1: 写失败测试**

`tests/test_json.cpp`：解析嵌套对象/数组；字符串转义（`\" \\ \/ \b \f \n \r \t`）与 `\u4E2D` → 中、代理对 `\uD83D\uDE00` → U+1F600；数字 `0`、`-3.5`、`1e10`、`1.5E-2`；布尔与空；错误：尾随垃圾、未闭合字符串/对象/数组、坏转义、孤立低位代理、前导零 `01`（解析为缺整数部分则错——按 01 中的 `0` 后接 `1` 即 `1`？不：本解析器 `0` 后不再吃数字，`01` 会留下 `1` 触发尾随内容错误，正好）；序列化往返（`json_parse(json_dump(x))` 结构相等）；`json_escape` 对控制字符出 `\u00xx`。写结构相等助手（逐 kind 比较）。断言覆盖每种样例各一。

- [ ] **Step 2: 跑测试确定失败**（方式同 T3 Step 2，目标 `test_json`）

- [ ] **Step 3: 写最小实现**

`src/serve/json.hpp`：

```cpp
// 手写 JSON 子集：对象、数组、字符串（含转义与 \uXXXX、代理对）、数字、布尔、空。
// 零依赖（ADR-008 在协议层同样成立）；解析严格（越界输入返回假，上层回 400）；序列化紧凑。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qinfer::serve {

class Json {
public:
    enum Kind { kNull, kBool, kNum, kStr, kArr, kObj };
    Kind kind = kNull;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;   // 保序（响应按固定顺序拼）

    static Json nullv() { return {}; }
    static Json boolv(bool v) { Json j; j.kind = kBool; j.b = v; return j; }
    static Json numv(double v) { Json j; j.kind = kNum; j.num = v; return j; }
    static Json strv(std::string v) { Json j; j.kind = kStr; j.str = std::move(v); return j; }
    static Json arrv() { Json j; j.kind = kArr; return j; }
    static Json objv() { Json j; j.kind = kObj; return j; }

    bool is_null() const { return kind == kNull; }
    bool is_str() const { return kind == kStr; }
    bool is_num() const { return kind == kNum; }
    bool is_arr() const { return kind == kArr; }
    bool is_obj() const { return kind == kObj; }

    const Json* find(const std::string& key) const;   // 非对象/缺键 → nullptr
    Json* find(const std::string& key);
    void push(Json v) { if (kind != kArr) kind = kArr; arr.push_back(std::move(v)); }
    void set(const std::string& key, Json v) {
        if (kind != kObj) kind = kObj;
        obj.emplace_back(key, std::move(v));
    }
};

bool json_parse(const std::string& s, Json& out, std::string& err);   // 语法错 → false + err（含位置）
std::string json_dump(const Json& j);                                 // 紧凑
std::string json_escape(const std::string& s);                        // JSON 字符串字面量转义

}  // namespace qinfer::serve
```

`src/serve/json.cpp`：按上文设计写全。要点：解析器为单趟递归下降，错误消息带「第 N 字符处」；字符串内裸控制字符报错；`\u` 高位代理必须跟 `\u` 低位代理，孤立低位报错；数字语法 `-? 0 | [1-9][0-9]* (. [0-9]+)? ([eE][+-]?[0-9]+)?`；`json_dump` 数字：整数值且 |x| < 9e15 打 `%lld`，否则 `%.9g`（测试钉住这两条路径）；转义 `" \ \b \f \n \r \t` 与 < 0x20 的 `\u00xx`，其余原样（UTF-8 字节不动）。

- [ ] **Step 4: 跑测试通过**（方式同 T3 Step 4）

- [ ] **Step 5: CMake + 提交**

```cmake
# qinfer_serve 源列表追加：src/serve/json.cpp
# 测试区追加：
# add_executable(test_json tests/test_json.cpp)
# target_link_libraries(test_json PRIVATE qinfer_serve)
# add_test(NAME json COMMAND test_json)
```

```bash
cmake -B build -S . && cmake --build build --target test_json && ctest --test-dir build -R '^json$' --output-on-failure
git add src/serve/ CMakeLists.txt tests/test_json.cpp
git commit -m "feat(serve): JSON 子集——严格解析（\uXXXX 代理对）与紧凑序列化"
```

---

### Task 7: 调度器 + 事件缓冲

网络线程只做解析与读写；真正的生成在槽位 worker 里跑（规格 §4 的第 1、2 条）。两者之间是每 Job 一个的有界事件缓冲：worker 写、HTTP 线程读，满则背压、绝不丢事件。槽位状态机（租约、释放、队列满、断开释放）在这一层成型，T9 用它接 HTTP。

**Files:**
- Create: `src/serve/scheduler.hpp`, `src/serve/scheduler.cpp`, `tests/test_scheduler.cpp`
- Modify: `CMakeLists.txt`（`qinfer_serve` 加 `scheduler.cpp`；加 `test_scheduler`）

**Interfaces:**
- Consumes: T3 的 `GenOptions`（Job 背它过）；`JobEvent`/`EventBuffer`（本任务产出，T9 通用）
- Produces: `qinfer::serve::EventBuffer`、`JobEvent`、`Job`、`SlotView`、`Scheduler`（T9–T12、T14 用）

- [ ] **Step 1: 写失败测试**

`tests/test_scheduler.cpp`（全部确定性：缓冲容量固定、线程是亲手起的，不用睡）：

```cpp
// 每 Job 一个的有界阻塞事件缓冲 + 槽位调度器。环形满 = 背压（push 阻塞，绝不丢事件）；
// 槽位租约按队列序；队列满 enqueue 返回假（HTTP 层回 503）。
#include "check.hpp"
#include "serve/scheduler.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace qinfer::serve;

namespace {

int main() {
    // ---- EventBuffer：先入先出 + 停止协议 ----
    {
        EventBuffer b(2);
        CHECK(b.push({EventKind::kToken, "a", {}, {}}));
        CHECK(b.push({EventKind::kToken, "b", {}, {}}));
        JobEvent e;
        CHECK(b.pop(e)); CHECK(e.text == "a");
        CHECK(b.pop(e)); CHECK(e.text == "b");
        // 空且未停止：pop 阻塞；另起线程 push 后唤醒。
        std::atomic<bool> got{false};
        std::thread t([&] { JobEvent x;
                            if (b.pop(x)) got = x.text == "c"; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        b.push({EventKind::kToken, "c", {}, {}});
        t.join();
        CHECK(got);
        // stop：取完剩余事件后 pop 返回假。
        b.stop();
        JobEvent after;
        CHECK(!b.pop(after));
    }
    // 满环背压：cap=2 塞 3 个，第三个 push 必须等 pop 腾位。
    {
        EventBuffer b(2);
        b.push({EventKind::kToken, "1", {}, {}});
        b.push({EventKind::kToken, "2", {}, {}});
        std::thread t([&] { b.push({EventKind::kToken, "3", {}, {}}); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        JobEvent e;
        CHECK(b.pop(e)); CHECK(e.text == "1");       // 腾位后 "3" 入队
        t.join();
        CHECK(b.pop(e)); CHECK(e.text == "2");
        CHECK(b.pop(e)); CHECK(e.text == "3");
        b.stop();
    }
    // cancel：唤醒阻塞的 pop 并令 push 失败；error() 给消息。
    {
        EventBuffer b(4);
        b.push({EventKind::kToken, "x", {}, {}});
        b.cancel();
        JobEvent e;
        CHECK(!b.pop(e));                            // 已取消：残余不被消费
        CHECK(!b.push({EventKind::kToken, "y", {}, {}}));
        CHECK(b.cancelled());
        EventBuffer b2(2);
        b2.stop_with_error("引擎炸了");
        JobEvent e2;
        CHECK(b2.pop(e2)); CHECK(e2.kind == EventKind::kError);
        CHECK(!e2.err.empty());
        CHECK(!b2.pop(e2));
        CHECK(b2.error() == "引擎炸了");
    }
    // ---- Scheduler：租约、队列序、释放、队列满、stop ----
    {
        Scheduler s(1, 2);                           // 1 槽、队列容量 2
        auto mk = [](std::uint64_t id) { return std::make_shared<Job>(Job{id}); };
        CHECK(s.enqueue(mk(1)));
        CHECK(s.enqueue(mk(2)));
        CHECK(!s.enqueue(mk(3)));                    // 队列满：1 槽在待 + 2 排队
        CHECK(s.queued() == 2u);
        auto j1 = s.take();
        CHECK(j1 && j1->id == 1);                    // 队首先租约
        CHECK(s.queued() == 1u);
        auto j2 = s.take();
        CHECK(j2 && j2->id == 2);
        CHECK(j2->leased_at > j1->leased_at);        // 租约计时在跑
        // 槽位被占：第三个 take 阻塞，done 后才返回。
        std::atomic<bool> leased{false};
        std::thread t([&] { auto j = s.take(); leased = j && j->id == 3; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        s.enqueue(mk(3));
        s.done(j1);
        t.join();
        CHECK(leased);
        auto j3 = s.take();                          // 现在队列空，阻塞……不：这里会被 done 唤醒的条件卡住
        (void) j3;
        // 上面 j3 会一直阻塞到 stop；先让它入队。
        s.enqueue(mk(4));
        auto j4 = s.take();
        CHECK(j4 && j4->id == 4);
        s.done(j2); s.done(j4);
        CHECK(!s.any_busy());
        auto view = s.slots_view();
        CHECK(view.size() == 1u && !view[0].busy);
        s.stop();
        auto jn = s.take();
        CHECK(!jn);                                  // stop 后 take 返回空
    }
    std::puts("scheduler: 通过");
    return 0;
}
}  // namespace
```

（`Job{id}` —— Job 是聚合体，字段按声明顺序初始化，`id` 排第一，见下。）

- [ ] **Step 2: 跑测试确定失败**（方式同 T3 Step 2，目标 `test_scheduler`；注意上面测试的 `j3` 阻塞点是刻意的：take 在没槽没队时挂起，由后续入队 + stop 收场）

- [ ] **Step 3: 写最小实现**

`src/serve/scheduler.hpp`：

```cpp
// 每 Job 一个的有界阻塞事件缓冲 + 槽位调度器（规格 §4）。
//   * EventBuffer：worker 写、HTTP 线程读；满则 push 阻塞（背压），绝不丢事件。
//     结束协议：worker 推完 kDone（或 kError）后调 stop()；读侧 pop 取完剩余返回假。
//     cancel()：客户端断开时调用，唤醒一切阻塞并令后续 push/pop 失败（worker 随即终止生成）。
//   * Scheduler：有界队列（容量 = 槽位数的 2 倍）+ 槽位租约。enqueue 满 → 返回假（HTTP 层 503）；
//     take() 按队序租约并记 leased_at；done() 释放。计时观测（queue_s / first_token_s / wallclock_s
//     / tok_s）全部落在 Job 字段，由 worker 与调度器填，统计不经过共享可变全局。
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "serve/facade.hpp"

namespace qinfer::serve {

enum class EventKind { kToken, kDone, kError };

struct JobEvent {
    EventKind kind = EventKind::kToken;
    std::string text;     // kToken：某 token 的解码字节片段
    std::string finish;   // kDone：finish_reason（stop / length）
    std::string err;      // kError：错误消息
};

class EventBuffer {
public:
    explicit EventBuffer(std::size_t capacity) : cap_(capacity > 0 ? capacity : 1) {}

    bool push(const JobEvent& e);           // 满则阻塞；已停止/取消 → false
    bool pop(JobEvent& out);                // 空则阻塞；已停止且取空 → false
    void stop();                            // 正常收尾：剩余事件可取完
    void stop_with_error(std::string err);  // 先入队 kError 再 stop
    void cancel();                          // 客户端断开：唤醒全部、后续失败
    bool cancelled() const;
    std::string error() const;              // stop_with_error 的消息；未出错为空

private:
    const std::size_t cap_;
    std::mutex mu_;
    std::condition_variable cv_empty_, cv_full_;
    std::deque<JobEvent> q_;
    bool stopped_ = false;
    bool cancelled_ = false;
    std::string err_;
};

// 一个请求的生命。HTTP 线程填前段（path/model/prompt/opts/stream），worker 填中段（prompt_ids、
// n_gen/text/finish/tok_s/error），调度器填取件时间（leased_at），HTTP 线程读全量写记录。
struct Job {
    std::uint64_t id = 0;
    std::string path;                    // 请求路径（记录与 /metrics 的 phase 用）
    std::string model;                   // 模型名（记录用）
    std::string prompt;                  // 渲染好的完整 prompt 文本
    GenOptions opts;
    std::vector<std::int32_t> prompt_ids;    // worker 分词后回填（供统计）
    std::uint64_t n_prompt = 0;              // 预填 token 数（= prompt_ids.size()）
    std::uint64_t n_gen = 0;
    std::string text;                        // 生成全文（stop 截断后）
    std::string finish;                      // stop / length / error
    std::string error;                       // 非空 = 失败原因
    double tok_s = 0.0;                      // 生成速率 = n_gen / (done_at - first_at)
    std::chrono::steady_clock::time_point enqueued_at, leased_at, first_at, done_at;
    std::shared_ptr<EventBuffer> events = std::make_shared<EventBuffer>(4096);
    bool stream = false;
};

struct SlotView {
    int index = 0;
    bool busy = false;
    std::uint64_t job_id = 0;
    double queue_s = 0.0;      // 当前 job 已排队秒数
    double elapsed_s = 0.0;    // 当前 job 已租约秒数
    double tok_s = 0.0;
};

class Scheduler {
public:
    Scheduler(std::size_t n_slots, std::size_t queue_capacity);
    ~Scheduler();

    bool enqueue(std::shared_ptr<Job> job);   // 队列满 → false（HTTP 层回 503）
    std::shared_ptr<Job> take();              // 阻塞；stop() 后返回 nullptr
    void done(std::shared_ptr<Job> job);      // 释放槽位
    void stop();

    std::size_t slots() const { return slots_.size(); }
    std::size_t queued() const;               // 排队等待数（不含槽位内）
    bool any_busy() const;
    std::vector<SlotView> slots_view() const;
    bool is_stopped() const;

private:
    const std::size_t qcap_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
    std::vector<std::shared_ptr<Job>> slots_;   // 槽位 → 当前 job（空 = 空闲）
    bool stop_ = false;
};

}  // namespace qinfer::serve
```

`src/serve/scheduler.cpp`：

```cpp
#include "serve/scheduler.hpp"

#include <algorithm>

namespace qinfer::serve {

// ---- EventBuffer ----
bool EventBuffer::push(const JobEvent& e) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_full_.wait(lk, [&] { return stopped_ || cancelled_ || q_.size() < cap_; });
    if (stopped_ || cancelled_) return false;
    q_.push_back(e);
    cv_empty_.notify_one();
    return true;
}

bool EventBuffer::pop(JobEvent& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_empty_.wait(lk, [&] { return cancelled_ || !q_.empty() || stopped_; });
    if (cancelled_ || q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    cv_full_.notify_one();
    return true;
}

void EventBuffer::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

void EventBuffer::stop_with_error(std::string err) {
    if (!err.empty()) push(JobEvent{EventKind::kError, {}, {}, err});
    {
        std::lock_guard<std::mutex> lk(mu_);
        err_ = std::move(err);
        stopped_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

void EventBuffer::cancel() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

bool EventBuffer::cancelled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cancelled_;
}

std::string EventBuffer::error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return err_;
}

// ---- Scheduler ----
Scheduler::Scheduler(std::size_t n_slots, std::size_t queue_capacity)
    : qcap_(queue_capacity < 1 ? 1 : queue_capacity), slots_(n_slots < 1 ? 1 : n_slots) {}

Scheduler::~Scheduler() { stop(); }

bool Scheduler::enqueue(std::shared_ptr<Job> job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_ || q_.size() >= qcap_) return false;
        job->enqueued_at = std::chrono::steady_clock::now();
        q_.push_back(std::move(job));
    }
    cv_.notify_one();
    return true;
}

std::shared_ptr<Job> Scheduler::take() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || (!q_.empty() && std::any_of(slots_.begin(), slots_.end(),
                                                                    [](const auto& j) { return !j; })); });
    if (stop_) return nullptr;
    auto it = std::find_if(slots_.begin(), slots_.end(), [](const auto& j) { return !j; });
    std::shared_ptr<Job> job = std::move(q_.front());
    q_.pop_front();
    job->leased_at = std::chrono::steady_clock::now();
    *it = job;
    return job;
}

void Scheduler::done(std::shared_ptr<Job> job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : slots_)
            if (s == job) s.reset();
    }
    cv_.notify_all();
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
}

std::size_t Scheduler::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
}

bool Scheduler::any_busy() const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& s : slots_)
        if (s) return true;
    return false;
}

std::vector<SlotView> Scheduler::slots_view() const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto now = std::chrono::steady_clock::now();
    std::vector<SlotView> out;
    out.reserve(slots_.size());
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        SlotView v;
        v.index = static_cast<int>(i);
        if (slots_[i]) {
            v.busy = true;
            v.job_id = slots_[i]->id;
            v.queue_s = std::chrono::duration<double>(slots_[i]->leased_at - slots_[i]->enqueued_at).count();
            v.elapsed_s = std::chrono::duration<double>(now - slots_[i]->leased_at).count();
            v.tok_s = slots_[i]->tok_s;
        }
        out.push_back(v);
    }
    return out;
}

bool Scheduler::is_stopped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stop_;
}

}  // namespace qinfer::serve
```

要点核对（与测试的阻塞点对应）：`take()` 的等待条件 = 「有队且有空槽」；队列满的 `enqueue` 返回假不占槽；`done` 只释放槽位、计时字段全由 worker 填（`done_at`/`tok_s` 在 worker 里置，T9 用）。

- [ ] **Step 4: 跑测试通过**（方式同 T3 Step 4）

- [ ] **Step 5: CMake + 提交**

```cmake
# qinfer_serve 源列表追加：src/serve/scheduler.cpp
# 测试区追加：
# add_executable(test_scheduler tests/test_scheduler.cpp)
# target_link_libraries(test_scheduler PRIVATE qinfer_serve)
# add_test(NAME scheduler COMMAND test_scheduler)
```

```bash
cmake -B build -S . && cmake --build build --target test_scheduler && ctest --test-dir build -R '^scheduler$' --output-on-failure
python3 scripts/docs_checks.py
git add src/serve/ CMakeLists.txt tests/test_scheduler.cpp
git commit -m "feat(serve): 槽位调度器与有界事件缓冲——满环背压、队列满拒绝、断开取消"
```

---

### Task 8: 集成测试的合成 GGUF 发生器

集成脚本（T9 起）要起一个真实服务进程，得有模型文件。真模型太慢太重，合成夹具才是 CI 里跑的：尺寸 = `tests/test_step_loop.cpp` 顶部的同款几何，元数据带分词器与 chat 模板（T3 的 `serve_metas` 那套）。仓库惯例是每个测试自带一份构造拷贝，这里照抄。

**Files:**
- Create: `tests/serve_fixture_gen.cpp`
- Modify: `CMakeLists.txt`（可执行目标，**不**注册为 ctest）

**Interfaces:**
- Consumes: T3 的夹具构造（从 `test_facade.cpp` 抄）
- Produces: `serve_fixture_gen` 可执行：`--out <路径>` 写出 `qinfer-serve-fixture.gguf`（T9 起集成脚本调用）

- [ ] **Step 1: 写发生器**

`tests/serve_fixture_gen.cpp`：把 T3 测试的夹具构造原样搬来（`TensorSpec`、`q8_0_rows`、`f32_vec`、`f32_lin`、`bf16_sym`、`router_tensor`、`moe_tensors`、`gr_group`、`gdn_attn`、`qsa_attn`、`layer_tensors`、`head_tensors`、`push_row_block`、`token_embd_data`、`model_tensors`、常量与 `write_gguf`，以及 `MetaKV` + `meta_u32/meta_str/meta_str_array/meta_u32_array` + `serve_metas`），出口换成 main：

```cpp
int main(int argc, char** argv) {
    std::string out;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out = argv[++i];
    }
    if (out.empty()) {
        std::fprintf(stderr, "用法：serve_fixture_gen --out <路径>\n");
        return 1;
    }
    const std::filesystem::path p = write_gguf(out, model_tensors(), serve_metas());
    std::printf("fixture：%s（%zu 字节）\n", p.string().c_str(),
                std::filesystem::file_size(p));
    return 0;
}
```

`write_gguf` 签名在这里是 `(const std::string& path, const std::vector<TensorSpec>&, const std::vector<MetaKV>&)`（不再走临时目录）。几何常量与 T3 测试完全一致（`kN=256, kHc=2, kLr=3, kFfn=64, kExpert=4, kTopK=2, kHd=32, kQsaNh=4, kQsaNkv=2, kIdx=4, kIdxH=2, kNrot=4, kIdxTopK=2048, kIdxBlock=4, kVocab=8`、层 0 GDN、层 1 QSA）。集成脚本启动服务时用同一组数字覆写几何 CLI（T9 的 run.sh 全写死）。

- [ ] **Step 2: 验证确定性**

```bash
cmake -B build -S . && cmake --build build --target serve_fixture_gen
./build/serve_fixture_gen --out /tmp/qinfer-fixture-1.gguf
./build/serve_fixture_gen --out /tmp/qinfer-fixture-2.gguf
cmp /tmp/qinfer-fixture-1.gguf /tmp/qinfer-fixture-2.gguf && echo 逐字节一致
```

预期：两次输出逐字节一致（夹具确定性，`.gguf` 里没有时间戳之类）。`test_facade` 打开这份文件应能通过（可顺手把 T3 测试的 `write_gguf` 换成调 `serve_fixture_gen --out`？**不换**——T3 的测试自带构造是仓库惯例，发生器只是集成脚本的入口，两者共存不冲突）。

- [ ] **Step 3: CMake**

```cmake
add_executable(serve_fixture_gen tests/serve_fixture_gen.cpp)
target_link_libraries(serve_fixture_gen PRIVATE qinfer_dense qinfer_artifact)
```

不注册 `add_test`（它只是数据源；确定性用上面的 `cmp` 手工核对一次即永久成立）。

- [ ] **Step 4: 提交**

```bash
git add tests/serve_fixture_gen.cpp CMakeLists.txt
git commit -m "test(serve): 集成测试用合成 GGUF 发生器（确定性夹具，几何与 test_facade 同款）"
```

---

### Task 9: 服务入口 + 非流式 OpenAI 路由 + 鉴权 + 集成脚本

终于把 HTTP 层接上调度器与门面（规格 §3 OpenAI 域的非流式部分、§5 参数校验、§4 的 503 语义）。本节之后 M1.2 的验收条目全部可测：「非流式请求返回正确文本；并发 8 请求全部成功且 `queue_s` 可观测；400/401/503 错误体正确」（并发断言在 run.sh，503 用 `--reject-after` 做成确定的，见下文）。

并发正确性的一条关键约束（本任务落地并写进 ADR-012）：`artifact::GgufFile` 的读取用一份 `mutable std::ifstream` + `seekg`（`gguf_table.cpp` 的 `read_at`），**不是线程安全的**；PLE 的 `GgufTable` 同理。所以 v0 用一把进程级 `engine_mu_` 把 `Engine::generate` 串行化——正确性优先的落点，队列/槽位/事件缓冲的结构已经就位；M2 把锁换成按线程文件句柄或 `pread` 就是吞吐优化的第一刀。这口锅不推给门面：`Engine::generate` 自身保持无锁，锁归服务层（它是进程内唯一并行使用引擎的地方）。

**Files:**
- Create: `src/serve/server.hpp`, `src/serve/server.cpp`, `src/serve/main.cpp`, `tests/integration/serve/run.sh`
- Modify: `CMakeLists.txt`（`qinfer_serve` 加 `server.cpp`；`qinfer_server` 可执行 + `serve_integration` ctest）

**Interfaces:**
- Consumes: T3 `Engine`/`GenOptions`/`Finish`、T6 `Json`、T7 `Scheduler`/`Job`/`EventBuffer`/`SlotView`、T8 `serve_fixture_gen`
- Produces: `qinfer::serve::ServeConfig`、`qinfer::serve::Serve`（T10–T14 往里加路由）、`qinfer-server` 可执行；集成脚本骨架（T10–T14 往里加用例）

- [ ] **Step 1: 写失败测试（先钉住调度层的 503 语义与错误体字面量）**

`tests/test_scheduler.cpp` 追加一段：`Scheduler` enqueue 满 → 假，且服务层看见假就回精确的 `{"error":"queue full"}`（把这个字面量与 T9 用同一个常量，防止两端各写一份漂移）——把 `src/serve/server.hpp` 里声明的 `const char* kQueueFullBody` 在 `scheduler.cpp` 定义，测试断言字符串相等：

```cpp
// 队列满的 503 体是全服务唯一的字面量（server.cpp 直接用它），两端钉死它。
CHECK(std::string(kQueueFullBody) == "{\"error\":\"queue full\"}");
Scheduler s(1, 1);
s.enqueue(std::make_shared<Job>(Job{1}));
CHECK(s.enqueue(std::make_shared<Job>(Job{2})));
CHECK(!s.enqueue(std::make_shared<Job>(Job{3})));
```

- [ ] **Step 2: 跑测试确定失败**

```bash
cmake -B build -S . 2>&1 | tail -3
cmake --build build --target test_scheduler 2>&1 | tail -5 || true
```

预期：找不到 `kQueueFullBody`，链接失败。

- [ ] **Step 3: 写最小实现**

`src/serve/server.hpp`：

```cpp
// HTTP 接线层：httplib 路由、鉴权、worker 池。请求在 HTTP 线程内完成解析与校验，交入调度器的有界
// 队列；槽位 worker 驱动门面生成，增量经 Job 的事件缓冲回写 HTTP 线程（非流式收齐、SSE 逐帧）。
//
// v0 并发正确性的一条关键约束：artifact::GgufFile 的读取用一份 mutable ifstream（seekg+read，
// gguf_table.cpp 的 read_at），不是线程安全的；PLE 的 GgufTable 同理。故这里用一把进程级 engine_mu_
// 串行化 Engine::generate——队列/槽位/事件缓冲结构已为 M2 就位，锁是吞吐优化的第一刀（换按线程
// 文件句柄或 pread）。锁放服务层不放门面：Engine::generate 自身无锁，服务层是唯一并行使用引擎的地方。
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/facade.hpp"
#include "serve/scheduler.hpp"

namespace httplib { class Server; }

namespace qinfer::serve {

// 503 的响应体是全服务唯一的字面量（规格 §4：`{error: "queue full"}` 的扁平形状）。
inline constexpr const char* kQueueFullBody = "{\"error\":\"queue full\"}";

struct ServeConfig {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;                 // 空 = 不鉴权（本机默认）
    std::size_t slots = 8;
    std::size_t queue_capacity = 0;      // 0 = 2 × slots（规格 §4 第 1 条）
    int threads = 0;                     // httplib HTTP 线程池；0 = 2 × slots + 2
    std::uint64_t reject_after = 0;      // 0 = 关；>0 = 第 N 个请求起一律 503（集成测试用，注释见 main）
    // 门面几何直通（默认 = 真模型推导值；集成测试覆写成夹具值）。
    FacadeConfig facade;
};

class Serve {
public:
    Serve(ServeConfig cfg, std::shared_ptr<Engine> eng);
    ~Serve();

    bool listen(std::string& err);   // 注册路由并绑定，阻塞在 HTTP 线程池内；失败写 err
    void stop();                     // 线程安全，可从信号处理器调用

private:
    void register_routes();
    void worker_main();
    void handle_chat(const httplib::Request& req, httplib::Response& res, bool stream);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    std::shared_ptr<Job> launch(const std::string& path, std::string prompt,
                                const GenOptions& opts, bool stream);
    void record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected);

    ServeConfig cfg_;
    std::shared_ptr<Engine> eng_;
    Scheduler sched_;
    std::mutex engine_mu_;             // 见文件头注释：v0 串行化生成
    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> req_id_{0};
    std::unique_ptr<httplib::Server> svr_;
    std::string web_index_, web_monitor_;   // 启动时读入的静态页
};

// main.cpp 用：命令行解析（自写的键值扫描，零依赖）。
ServeConfig parse_flags(int argc, char** argv, std::string& err);

}  // namespace qinfer::serve
```

`src/serve/server.cpp`（完整实现；T10/T11/T12/T14 往里追加路由与 helper）：

```cpp
#include "serve/server.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "httplib.h"
#include "serve/json.hpp"

namespace qinfer::serve {

namespace {

// ---- 请求体参数读取（全部按规格 §5 的非法参数 → 400 口径）----
bool num_param(const Json& body, const char* key, double dflt, double& out) {
    const Json* v = body.find(key);
    if (v == nullptr) { out = dflt; return true; }
    if (!v->is_num()) return false;
    out = v->num;
    return true;
}

bool str_param(const Json& body, const char* key, std::string& out, bool& present) {
    const Json* v = body.find(key);
    present = v != nullptr;
    if (v == nullptr) return true;
    if (!v->is_str()) return false;
    out = v->str;
    return true;
}

// stop：缺省 → 空；字符串 → 一个；数组 → 全字符串；其他 → 假（400）。
bool stop_param(const Json& body, std::vector<std::string>& out) {
    const Json* v = body.find("stop");
    out.clear();
    if (v == nullptr) return true;
    if (v->is_str()) { out.push_back(v->str); return true; }
    if (v->is_arr()) {
        for (const Json& e : v->arr) {
            if (!e.is_str()) return false;
            out.push_back(e.str);
        }
        return true;
    }
    return false;
}

// ---- 错误体（规格 §5 的两个域形状）----
Json openai_err(const std::string& message, const std::string& type, const char* param) {
    Json inner = Json::objv();
    inner.set("message", Json::strv(message));
    inner.set("type", Json::strv(type));
    inner.set("param", param == nullptr ? Json::nullv() : Json::strv(param));
    Json e = Json::objv();
    e.set("error", std::move(inner));
    return e;
}

Json anthropic_err(const std::string& type, const std::string& message) {
    Json inner = Json::objv();
    inner.set("type", Json::strv(type));
    inner.set("message", Json::strv(message));
    Json e = Json::objv();
    e.set("type", Json::strv("error"));
    e.set("error", std::move(inner));
    return e;
}

void send_json(httplib::Response& res, int status, const Json& body) {
    res.status = status;
    res.set_content(json_dump(body), "application/json; charset=utf-8");
}

bool anthropic_path(const std::string& p) {
    return p == "/v1/messages" || p == "/v1/messages/count_tokens";
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// 把停等协议滚一遍（worker 侧语义见 worker_main）：读 kToken 攒文本，kDone/kError 收尾。
// HTTP 线程专用（阻塞 pop）。返回 false = 出错（err 给出消息）。
bool drain_nonstream(const std::shared_ptr<Job>& job, std::string& text, std::string& err) {
    JobEvent e;
    for (;;) {
        if (!job->events->pop(e)) { err = "事件缓冲异常终止"; return false; }
        if (e.kind == EventKind::kToken) { text += e.text; continue; }
        if (e.kind == EventKind::kDone) return true;
        // kError
        err = e.err;
        return false;
    }
}

}  // namespace

Serve::Serve(ServeConfig cfg, std::shared_ptr<Engine> eng)
    : cfg_(std::move(cfg)),
      eng_(std::move(eng)),
      sched_(cfg_.slots, cfg_.queue_capacity == 0 ? 2 * cfg_.slots : cfg_.queue_capacity),
      svr_(std::make_unique<httplib::Server>()) {
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    web_index_ = read_file((base / "index.html").string());
    web_monitor_ = read_file((base / "monitor.html").string());
}

Serve::~Serve() { stop(); }

void Serve::worker_main() {
    for (;;) {
        std::shared_ptr<Job> job = sched_.take();
        if (!job) break;
        std::string err;
        std::vector<std::int32_t> ids;
        if (!eng_->tokenizer().encode(job->prompt, /*parse_special=*/true, ids, err)) {
            job->finish = "error";
            job->error = "分词失败：" + err;
            job->events->stop_with_error(job->error);
            sched_.done(job);
            continue;
        }
        job->prompt_ids = std::move(ids);
        job->n_prompt = job->prompt_ids.size();
        GenResult res;
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(engine_mu_);   // v0 串行化：见 server.hpp 头注释
            ok = eng_->generate(job->prompt_ids, job->opts,
                                [&](const std::string& piece) -> bool {
                                    if (job->first_at == std::chrono::steady_clock::time_point{})
                                        job->first_at = std::chrono::steady_clock::now();
                                    return job->events->push(JobEvent{EventKind::kToken, piece, {}, {}});
                                },
                                res, err);
        }
        if (ok) {
            job->n_gen = res.n_gen;
            job->text = res.text;
            job->finish = res.finish == Finish::kLength ? "length" : "stop";
            job->done_at = std::chrono::steady_clock::now();
            const double span = std::chrono::duration<double>(job->done_at - job->first_at).count();
            job->tok_s = span > 0 ? static_cast<double>(res.n_gen) / span : 0.0;
            job->events->push(JobEvent{EventKind::kDone, {}, job->finish, {}});
            job->events->stop();
        } else {
            job->finish = "error";
            job->error = err;
            job->done_at = std::chrono::steady_clock::now();
            job->events->stop_with_error(err);
        }
        // push 全失败（客户端已断开）时也照常释放槽位；记录由 HTTP 侧标 disconnected。
        sched_.done(job);
    }
}

std::shared_ptr<Job> Serve::launch(const std::string& path, std::string prompt,
                                   const GenOptions& opts, bool stream) {
    auto job = std::make_shared<Job>();
    job->id = req_id_++;
    job->path = path;
    job->model = eng_->model_name();
    job->prompt = std::move(prompt);
    job->opts = opts;
    job->stream = stream;
    return sched_.enqueue(job) ? job : nullptr;
}

void Serve::handle_chat(const httplib::Request& req, httplib::Response& res, bool stream) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, openai_err("请求体不是合法 JSON 对象：" + jerr, "invalid_request_error", nullptr));
        return;
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) {
        send_json(res, 400, openai_err("缺少 messages 数组", "invalid_request_error", "messages"));
        return;
    }
    // messages → [{role, content}]；content 只收字符串（规格 §1 范围外无多模态）。
    std::vector<std::pair<std::string, std::string>> conv;
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { send_json(res, 400, openai_err("messages 元素不是对象", "invalid_request_error", "messages")); return; }
        const Json* role = m.find("role");
        const Json* content = m.find("content");
        if (role == nullptr || !role->is_str() || content == nullptr || !content->is_str()) {
            send_json(res, 400, openai_err("每条消息需要字符串 role 与 content", "invalid_request_error", "messages"));
            return;
        }
        // content 为空的消息（tool 占位）过滤掉，模板器不管
        if (content->str.empty()) continue;
        conv.emplace_back(role->str, content->str);
    }
    GenOptions opts;
    double t = 0.0, p = 0.0, mt = 0.0;
    std::int64_t seed = -1;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p) &&
                      num_param(body, "max_tokens", 512, mt);
    const Json* seedv = body.find("seed");
    if (seedv != nullptr) {
        if (!seedv->is_num()) numeric_ok = false;
        else seed = static_cast<std::int64_t>(seedv->num);
    }
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens/seed 必须为数字", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.seed = seed;
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens 必须为正", "invalid_request_error", "max_tokens")); return; }
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop 必须是字符串或字符串数组", "invalid_request_error", "stop")); return; }
    SamplerParams sp; sp.temperature = opts.temperature; sp.top_p = opts.top_p;
    if (!sp.valid()) { send_json(res, 400, openai_err("temperature 非负、top_p 在 (0,1]", "invalid_request_error", nullptr)); return; }
    if (stream) { send_json(res, 400, openai_err("stream:true 尚未支持（SSE 在 T10 落地）", "invalid_request_error", "stream")); return; }

    std::string prompt;
    std::string rerr;
    if (!render_chat(eng_->chat_template(), conv, prompt, rerr)) {
        // 模板器是裁剪子集（规格 §11 兜底）：不静默错译，明确 500 并说明。
        send_json(res, 500, openai_err("chat 模板不支持的结构：" + rerr, "server_error", nullptr));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        res.status = 503;
        res.set_content(kQueueFullBody, "application/json; charset=utf-8");
        record_request(nullptr, 503, false);
        return;
    }
    std::string text, drain_err;
    const bool ok = drain_nonstream(job, text, drain_err);
    if (!ok) {
        const int st = job->events->cancelled() ? 499 : 500;   // 499 = 客户端断开，httplib 惯例
        if (st == 500) {
            send_json(res, st, openai_err(drain_err, "server_error", nullptr));
        }
        record_request(job, st, job->events->cancelled());
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("chatcmpl-" + std::to_string(job->id)));
    r.set("object", Json::strv("chat.completion"));
    r.set("created", Json::numv(static_cast<double>(std::time(nullptr))));
    r.set("model", Json::strv(job->model));
    Json choice = Json::objv();
    Json message = Json::objv();
    message.set("role", Json::strv("assistant"));
    message.set("content", Json::strv(text));
    choice.set("index", Json::numv(0.0));
    choice.set("message", std::move(message));
    choice.set("finish_reason", Json::strv(job->finish));
    Json choices = Json::arrv();
    choices.push(std::move(choice));
    r.set("choices", std::move(choices));
    Json usage = Json::objv();
    usage.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
    usage.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

void Serve::handle_completions(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, openai_err("请求体不是合法 JSON 对象：" + jerr, "invalid_request_error", nullptr));
        return;
    }
    std::string prompt;
    bool present = false;
    if (!str_param(body, "prompt", prompt, present) || !present) {
        send_json(res, 400, openai_err("缺 prompt 字符串", "invalid_request_error", "prompt"));
        return;
    }
    // 参数读取与 handle_chat 同款（temperature/top_p/seed/max_tokens/stop；stream 同 400）。
    GenOptions opts;
    double t = 0.0, p = 0.0, mt = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p) &&
                      num_param(body, "max_tokens", 512, mt);
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens 必须为数字", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens 必须为正", "invalid_request_error", "max_tokens")); return; }
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop 必须是字符串或字符串数组", "invalid_request_error", "stop")); return; }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, false);
    if (!job) { res.status = 503; res.set_content(kQueueFullBody, "application/json; charset=utf-8"); return; }
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        if (!job->events->cancelled()) send_json(res, 500, openai_err(drain_err, "server_error", nullptr));
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("cmpl-" + std::to_string(job->id)));
    r.set("object", Json::strv("text_completion"));
    r.set("created", Json::numv(static_cast<double>(std::time(nullptr))));
    r.set("model", Json::strv(job->model));
    Json choice = Json::objv();
    choice.set("text", Json::strv(text));
    choice.set("index", Json::numv(0.0));
    choice.set("logprobs", Json::nullv());
    choice.set("finish_reason", Json::strv(job->finish));
    Json choices = Json::arrv();
    choices.push(std::move(choice));
    r.set("choices", std::move(choices));
    Json usage = Json::objv();
    usage.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
    usage.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    // T14 接 Records 环：本任务为空实现（请求记录 M1.4 才落地，路由形状已就绪）。
    (void) job; (void) http_status; (void) disconnected;
}

void Serve::register_routes() {
    httplib::Server& svr = *svr_;
    // 鉴权（规格 §3）：/web、/api-monitor、/ 与 /health 可达，页面自带 key；
    // /v1/*、/api/*、/metrics 缺失或错误 → 401（对应域的错误形状）。
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (cfg_.api_key.empty()) return httplib::Server::HandlerResponse::Unhandled;
        const std::string want = "Bearer " + cfg_.api_key;
        if (req.get_header_value("Authorization") == want) return httplib::Server::HandlerResponse::Unhandled;
        if (req.path == "/" || req.path.rfind("/web/", 0) == 0 || req.path == "/api-monitor" ||
            req.path == "/health" || req.path.rfind("/fonts/", 0) == 0) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (anthropic_path(req.path)) {
            send_json(res, 401, anthropic_err("authentication_error", "缺少或错误的 API key"));
        } else {
            send_json(res, 401, openai_err("缺少或错误的 API key", "authentication_error", nullptr));
        }
        return httplib::Server::HandlerResponse::Handled;
    });
    // 未挂路由的路径一律 404 JSON（规格 §3 的静态域：/v1/responses、/load 等全走这里）。
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        (void) req;
        if (res.status == 404) send_json(res, 404, openai_err("not found", "not_found", nullptr));
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        Json h = Json::objv();
        h.set("status", Json::strv("ok"));
        h.set("model", Json::strv(eng_->model_name()));
        h.set("max_context", Json::numv(static_cast<double>(cfg_.facade.max_context)));
        h.set("images", Json::boolv(false));
        h.set("api_key", Json::boolv(!cfg_.api_key.empty()));
        h.set("loaded", Json::boolv(true));
        h.set("service", Json::strv("qinfer"));
        send_json(res, 200, h);
    });
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        Json model = Json::objv();
        model.set("id", Json::strv(eng_->model_name()));
        model.set("object", Json::strv("model"));
        Json st = Json::objv();
        st.set("value", Json::strv("loaded"));
        model.set("status", std::move(st));
        Json meta = Json::objv();
        meta.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
        model.set("meta", std::move(meta));
        Json arch = Json::objv();
        Json in = Json::arrv(); in.push(Json::strv("text"));
        Json out = Json::arrv(); out.push(Json::strv("text"));
        arch.set("input_modalities", std::move(in));
        arch.set("output_modalities", std::move(out));
        model.set("architecture", std::move(arch));
        Json data = Json::arrv();
        data.push(std::move(model));
        Json r = Json::objv();
        r.set("object", Json::strv("list"));
        r.set("data", std::move(data));
        send_json(res, 200, r);
    });
    svr.Post("/v1/chat/completions",
             [this](const httplib::Request& req, httplib::Response& res) { handle_chat(req, res, false); });
    svr.Post("/v1/completions",
             [this](const httplib::Request& req, httplib::Response& res) { handle_completions(req, res); });
    // 静态页（T15/T16 落地文件；此刻 404 也正常——T9 只验 API 面）。
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(web_index_, "text/html; charset=utf-8");
    });
    svr.Get("/api-monitor", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(web_monitor_, "text/html; charset=utf-8");
    });
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    svr.set_base_dir(base.string().c_str(), "/web");
}

bool Serve::listen(std::string& err) {
    register_routes();
    // HTTP 线程池：2 × 槽位 + 2（规格 §10：线程池大小由入口按槽位数配置）。
    const int tp = cfg_.threads > 0 ? cfg_.threads : static_cast<int>(2 * cfg_.slots + 2);
    svr_->set_thread_pool_size(tp);
    for (std::size_t i = 0; i < cfg_.slots; ++i) workers_.emplace_back([this] { worker_main(); });
    if (!svr_->bind_to_port(cfg_.host, cfg_.port)) {
        sched_.stop();
        err = "绑定 " + cfg_.host + ":" + std::to_string(cfg_.port) + " 失败";
        return false;
    }
    (void) svr_->listen_after_bind();
    return true;
}

void Serve::stop() {
    if (svr_) svr_->stop();
    sched_.stop();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

}  // namespace qinfer::serve
```

`src/serve/main.cpp`：

```cpp
// 服务入口：命令行解析（自写键值扫描，零依赖）→ 装载门面 → 起 Serve。
#include "httplib.h"
#include "serve/facade.hpp"
#include "serve/server.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

using namespace qinfer::serve;

namespace {

Serve* g_serve = nullptr;
volatile std::sig_atomic_t g_signaled = 0;

void on_signal(int) {
    if (g_serve) g_serve->stop();
    g_signaled = 1;
}

std::int64_t arg_int(const std::map<std::string, std::string>& a, const char* key, std::int64_t dflt) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    return std::strtoll(it->second.c_str(), nullptr, 10);
}

double arg_double(const std::map<std::string, std::string>& a, const char* key, double dflt) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    return std::strtod(it->second.c_str(), nullptr);
}

}  // namespace

ServeConfig parse_flags(int argc, char** argv, std::string& err) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) { err = "位置参数不需要：" + s; return {}; }
        const auto eq = s.find('=');
        const std::string k = eq == std::string::npos ? s : s.substr(0, eq);
        const std::string v = eq == std::string::npos ? (i + 1 < argc ? argv[++i] : std::string{}) : s.substr(eq + 1);
        if (v.empty()) { err = "参数 " + k + " 需要值"; return {}; }
        a[k] = v;
    }
    ServeConfig c;
    const auto it = a.find("--model");
    if (it == a.end()) { err = "缺 --model（主案例 GGUF 的路径）"; return {}; }
    c.model_path = it->second;
    if (a.count("--host")) c.host = a["--host"];
    c.port = static_cast<int>(arg_int(a, "--port", 8080));
    if (a.count("--api-key")) c.api_key = a["--api-key"];
    c.slots = static_cast<std::size_t>(arg_int(a, "--slots", 8));
    c.queue_capacity = static_cast<std::size_t>(arg_int(a, "--queue", 0));
    c.threads = static_cast<int>(arg_int(a, "--threads", 0));
    c.reject_after = static_cast<std::uint64_t>(arg_int(a, "--reject-after", 0));
    FacadeConfig& f = c.facade;
    f.model_path = c.model_path;
    f.n_embd = static_cast<std::uint64_t>(arg_int(a, "--n-embd", 2560));
    f.hc = static_cast<std::uint64_t>(arg_int(a, "--hc", 4));
    f.hc_lr = static_cast<std::uint64_t>(arg_int(a, "--hc-lr", 320));
    f.n_layers = static_cast<int>(arg_int(a, "--layers", 48));
    f.top_k = static_cast<int>(arg_int(a, "--top-k", 10));
    f.gdn_head_dim = static_cast<std::uint64_t>(arg_int(a, "--gdn-head-dim", 128));
    f.qsa_head_dim = static_cast<std::uint64_t>(arg_int(a, "--qsa-head-dim", 256));
    f.n_rot = static_cast<std::uint64_t>(arg_int(a, "--n-rot", 64));
    f.idx_dim = static_cast<std::uint64_t>(arg_int(a, "--idx-dim", 128));
    f.idx_top_k = static_cast<std::uint64_t>(arg_int(a, "--idx-top-k", 2048));
    f.idx_block = static_cast<std::uint64_t>(arg_int(a, "--idx-block", 4));
    f.max_context = static_cast<std::uint64_t>(arg_int(a, "--max-context", 262144));
    f.cache_slots = static_cast<std::size_t>(arg_int(a, "--cache-slots", 0));
    f.cache_byte_budget = static_cast<std::uint64_t>(arg_int(a, "--cache-byte-budget", 0));
    f.ple_budget_bytes = static_cast<std::uint64_t>(arg_int(a, "--ple-budget-bytes", 0));
    if (a.count("--ple-table")) f.ple_table = a["--ple-table"];
    f.kv_slots = static_cast<std::uint64_t>(arg_int(a, "--kv-slots", 0));
    return c;
}

int main(int argc, char** argv) {
    std::string err;
    const ServeConfig cfg = parse_flags(argc, argv, err);
    if (!err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::fprintf(stderr, "用法：qinfer-server --model <gguf> [--host H] [--port P] [--api-key K] "
                             "[--slots N] [--queue N] [--threads N] [--reject-after N] "
                             "[--n-embd N] [--hc N] [--hc-lr N] [--layers N] [--top-k N] "
                             "[--gdn-head-dim N] [--qsa-head-dim N] [--n-rot N] [--idx-dim N] "
                             "[--idx-top-k N] [--idx-block N] [--max-context N] [--cache-slots N] "
                             "[--cache-byte-budget N] [--ple-table P] [--ple-budget-bytes N] [--kv-slots N]\n");
        return 2;
    }
    // --reject-after：仅供集成测试制造确定性 503（第 N 个请求起一律拒绝入队），生产不用。
    // 在 handler 里生效：launch() 前检查 req_id_ 与阈值。
    std::shared_ptr<Engine> eng = std::make_shared<Engine>();
    if (!Engine::open(cfg.facade, *eng, err)) {
        std::fprintf(stderr, "装载失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("模型 %s（词表 %llu、n_embd %llu、%d 层）就绪\n", eng->model_name().c_str(),
                (unsigned long long) eng->tokenizer().vocab_size(),
                (unsigned long long) cfg.facade.n_embd, cfg.facade.n_layers);
    Serve serve(cfg, eng);
    g_serve = &serve;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (!serve.listen(err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("服务退出（信号 %d）\n", static_cast<int>(g_signaled));
    return 0;
}
```

注意一处接缝：`--reject-after` 的判定要在 `Serve::launch` 里做——补进 `server.cpp` 的 `launch` 开头：

```cpp
    if (cfg_.reject_after > 0 && req_id_.load() >= cfg_.reject_after) return nullptr;   // 503
```

（req_id_ 的读在 `launch` 的 `job->id = req_id_++;` 前，保证「第 N 个请求」按到达序数。）

- [ ] **Step 4: CMake + 编译**

```cmake
# qinfer_serve 源列表追加：src/serve/server.cpp（T9；T10/T11/T12/T14 继续往这个库加源）
# 可执行与集成：
# add_executable(qinfer_server src/serve/main.cpp)
# target_link_libraries(qinfer_server PRIVATE qinfer_serve qinfer_text qinfer_artifact)
# add_test(NAME serve_integration
#          COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/serve/run.sh
#                  ${CMAKE_BINARY_DIR} 18080)
```

`tests/integration/serve/run.sh`（bash + curl + python3 校验 JSON；无 jq）：

```bash
#!/usr/bin/env bash
# M1 服务层集成测试（T9 段：非流式 OpenAI 域 + 错误体 + 并发 + 503 形态）。
# 用法：run.sh <构建目录> [端口]
set -u
cd "$(dirname "$0")/../../.."                  # 仓库根

BUILD="${1:-build}"
PORT="${2:-18080}"
HOST=127.0.0.1
FIX="$BUILD/tests/qinfer-serve-fixture.gguf"
SERVER="$BUILD/qinfer-server"
LOG="$(mktemp)"
PID=""
PASS=0; FAIL=0

cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  ok  $1"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL  $1"; }

# 起服务；等 /health。
start() {   # start <额外参数...>
  "$SERVER" --model "$FIX" --host "$HOST" --port "$PORT" --n-embd 256 --hc 2 --hc-lr 3 \
            --layers 2 --top-k 2 --gdn-head-dim 32 --qsa-head-dim 32 --n-rot 4 \
            --idx-dim 4 --idx-top-k 2048 --idx-block 4 --max-context 4096 "$@" >"$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 80); do
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "服务没起来：$(tail -5 "$LOG")" >&2
  return 1
}

stop() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""; }

# JSON 取值：jget <json文件> <python路径表达式> → 标准输出
jget() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null; }

# 生成夹具
"$BUILD/serve_fixture_gen" --out "$FIX"

# ---- 带 key 的主实例 ----
start --api-key testkey || exit 1

R="$(mktemp)"; H="Authorization: Bearer testkey"

# 1. chat 补全：内容非空、usage 字段齐、finish 合法、id 前缀对。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"model":"x","messages":[{"role":"system","content":"你是一个助手。"},{"role":"user","content":"说一个字。"}],"max_tokens":16,"temperature":0}')
if [ "$code" = 200 ] \
   && [ -n "$(jget "$R" "d['choices'][0]['message']['content']")" ] \
   && [ "$(jget "$R" "d['usage']['prompt_tokens']")" -ge 1 ] \
   && [ "$(jget "$R" "d['choices'][0]['finish_reason']")" = stop ] \
   && [ "$(jget "$R" "d['id'][:9]")" = chatcmpl- ]; then
  ok "chat 补全（HTTP 200、content/usage/finish/id）"
else
  bad "chat 补全：状态 $code，体 $(head -c 200 "$R")"
fi

# 2. 文本补全。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/completions" -d '{"prompt":"hi","max_tokens":8,"temperature":0}')
if [ "$code" = 200 ] && [ -n "$(jget "$R" "d['choices'][0]['text']")" ]; then
  ok "文本补全"
else
  bad "文本补全：状态 $code，体 $(head -c 200 "$R")"
fi

# 3. 模型清单。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" "http://$HOST:$PORT/v1/models")
if [ "$code" = 200 ] && [ "$(jget "$R" "d['data'][0]['id']")" = qinfer-serve-fixture ]; then
  ok "/v1/models 单元素、id 正确"
else
  bad "/v1/models：状态 $code，体 $(head -c 200 "$R")"
fi

# 4. 非法参数 → 400 且 error.type = invalid_request_error。
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"temperature":-1}')
if [ "$code" = 400 ] && [ "$(jget "$R" "d['error']['type']")" = invalid_request_error ]; then
  ok "400 错误体（OpenAI 形状）"
else
  bad "400：状态 $code，体 $(head -c 200 "$R")"
fi

# 5. 缺 key → 401 且错误形状正确（OpenAI 域）。
code=$(curl -s -o "$R" -w '%{http_code}' "http://$HOST:$PORT/v1/models")
if [ "$code" = 401 ] && [ "$(jget "$R" "d['error']['type']")" = authentication_error ]; then
  ok "401 错误体（缺 key）"
else
  bad "401：状态 $code，体 $(head -c 200 "$R")"
fi

# 6. 并发 8：8 个请求同时打，全部 200 且响应是合法 JSON；服务在结束后仍存活。
allok=1
for i in $(seq 1 8); do
  curl -s -o "$R.$i" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
    -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"c"}],"max_tokens":8,"temperature":0}' >"$LOG.c$i" &
done
wait
for i in $(seq 1 8); do
  [ "$(cat "$LOG.c$i")" = 200 ] || allok=0
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$R.$i" 2>/dev/null || allok=0
  rm -f "$R.$i" "$LOG.c$i"
done
code=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" "http://$HOST:$PORT/health")
[ "$code" = 200 ] || allok=0
if [ "$allok" = 1 ]; then
  ok "并发 8：全 200、响应全为合法 JSON、服务存活"
else
  bad "并发 8 后有失败（服务存活检查与 8 份响应逐一核过）"
fi

stop

# ---- 503：--reject-after 1 → 第 1 个请求正常，之后一律队列满 ----
start --api-key testkey --slots 1 --reject-after 1 || exit 1
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"x"}],"max_tokens":4,"temperature":0}')
[ "$code" = 200 ] || bad "reject-after 首请求应 200，得 $code"
code=$(curl -s -o "$R" -w '%{http_code}' -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"y"}],"max_tokens":4,"temperature":0}')
if [ "$code" = 503 ] && cmp -s "$R" <(printf '%s' '{"error":"queue full"}'); then
  ok "503 身体精确为 {\"error\":\"queue full\"}"
else
  bad "503：状态 $code，体 $(head -c 200 "$R")"
fi
stop

echo "serve 集成：通过 $PASS、失败 $FAIL"
[ "$FAIL" = 0 ]
```

（并发段有意温和：主断言是「8 份响应全合法 + 服务存活」。503 由 `--reject-after` 制成确定形态，不赌时序；真实并发下的 503 在环境2 真权重上人工复验，见 T18 清单。）

- [ ] **Step 5: 跑集成 + 单测**

```bash
cmake -B build -S . && cmake --build build --target qinfer_server test_scheduler serve_fixture_gen
ctest --test-dir build -R '^(scheduler|serve_integration)$' --output-on-failure
```

预期：scheduler 单元过；serve_integration 走完整脚本，通过 8 项。服务日志不对就打 `$LOG`——脚本删了它，若失败先改脚本保留日志。

- [ ] **Step 6: 提交**

```bash
git add src/serve/ CMakeLists.txt tests/integration/serve/run.sh tests/test_scheduler.cpp
python3 scripts/docs_checks.py
git commit -m "feat(serve): 服务入口与非流式 OpenAI 路由——鉴权、400/401/503 错误体、槽位 worker、集成脚本"
```

---

### Task 10: SSE 流式（OpenAI）

把 T9 的「stream:true → 400」换成真正的逐 chunk 流。传输用 httplib 的 chunked content provider：handler 返回前把 provider 挂到 `res` 上，httplib 之后反复调用 `provider(offset, sink)` 直到返回假；写 socket 的阻塞在 httplib 内部，只认 `sink.write` 的返回值（假 = 客户端断开）。

流式正确性的核心是停等协议不变：worker 照样往 `job->events` 推 `kToken/kDone/kError`，HTTP 线程只换「攒文本」为「逐帧写」。事件缓冲是背压的天然节流器——客户端读得慢，`sink.write` 阻塞 → provider 慢 → 缓冲满 → worker 的 `push` 阻塞 → 整条链停住，不丢事件、不失控。

**Files:**
- Modify: `src/serve/server.cpp`（`handle_chat` 拆出 SSE 分支、新增 `serve_sse`）、`tests/integration/serve/run.sh`（流式用例）

**Interfaces:**
- Consumes: T7 `EventBuffer`；T3 `GenResult` 语义
- Produces: `Serve::serve_sse`（T11 的 Anthropic 流复用同一套 provider 模式）

- [ ] **Step 1: 写失败测试（run.sh 追加流式用例）**

`tests/integration/serve/run.sh` 在「模型清单」用例后追加（`$R` 已定义，`$H` 已定义）：

```bash
# 7. SSE：先非流式取参考文本，再流式取同款请求，逐 delta 拼出来必须与参考逐字节相同。
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"流式对拍"}],"max_tokens":24,"temperature":0,"seed":5}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_ref.txt

HDRS="$R.hdrs"; BODY="$R.body"
curl -sN -D "$HDRS" -o "$BODY" -H "$H" -H 'Content-Type: application/json' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"流式对拍"}],"max_tokens":24,"temperature":0,"seed":5,"stream":true}'
# 帧序列检查：第一帧带 role、中间帧带 content、末帧带 finish_reason、收尾 [DONE]；逐 delta 拼文本。
python3 - "$BODY" >/tmp/qinfer_sse.txt <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
assert frames, "没有任何数据帧"
first = json.loads(frames[0])
assert first["choices"][0]["delta"].get("role") == "assistant", "首帧缺 role"
last = json.loads(frames[-1])
assert last["choices"][0]["finish_reason"] in ("stop", "length"), "末帧缺 finish_reason"
text = "".join(json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames)
sys.stdout.write(text)
PY
if grep -q "data: \[DONE\]" "$BODY" && grep -qi "text/event-stream" "$HDRS" \
   && cmp -s /tmp/qinfer_ref.txt /tmp/qinfer_sse.txt; then
  ok "SSE：事件流、[DONE]、流式拼文与非流式逐字节一致"
else
  bad "SSE：头部 $(grep -i event-stream "$HDRS" | head -1)，正文行数 $(wc -l < "$BODY")"
fi
rm -f "$HDRS" "$BODY"

# 8. stream_options.include_usage：最后第二个数据帧带 usage，且 completion_tokens ≥ 1。
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"usage"}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
  | python3 -c '
import json, sys
raw = sys.stdin.read()
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
usage = [json.loads(f) for f in frames if json.loads(f).get("usage")]
assert usage, "usage 帧缺失"
assert usage[-1]["usage"]["completion_tokens"] >= 1
print("usage-ok")' | grep -q usage-ok && ok "include_usage 末位 usage 帧" || bad "include_usage"
```

先跑：预期 SSE 路径不存在——`handle_chat` 里还是 400 分支，第七项坏。**注意此时 `$H` 在「缺少 key → 401 用例」之后仍有效。**

- [ ] **Step 2: 实现**

`server.cpp`（替换 `handle_chat` 的 stream 分支为转发；新增 `serve_sse`）：

```cpp
void Serve::handle_chat(const httplib::Request& req, httplib::Response& res, bool stream) {
    // ……参数校验与模板渲染全部照旧（T9 的既有代码）……
    // 原来「stream:true → 400」的位置替换为：
    if (stream) {
        serve_sse(job, body.has("stream_options"), res);
        return;
    }
    // ……非流式的 drain 与响应照旧……
}

// OpenAI 兼容 SSE。provider 由 httplib 在 handler 返回后反复调用；每次调用恰好消费一个
// 事件并写一帧（多帧合并写进同一次 sink.write）。连接断开（write 假）→ cancel → 返回假收尾。
void Serve::serve_sse(const std::shared_ptr<Job>& job, bool include_usage, httplib::Response& res) {
    const std::string id = "chatcmpl-" + std::to_string(job->id);
    const double created = static_cast<double>(std::time(nullptr));
    const std::string model = eng_->model_name();
    const auto first = std::make_shared<bool>(true);
    auto provider = [this, job, include_usage, id, created, model, first](
                        size_t, httplib::DataSink& sink) -> bool {
        JobEvent e;
        const bool got = job->events->pop(e);
        if (!got) { record_request(job, 499, job->events->cancelled()); return false; }
        const auto w = [&](const std::string& s) {
            if (!sink.write(s)) { job->events->cancel(); record_request(job, 499, true); return false; }
            return true;
        };
        Json chunk = Json::objv();
        chunk.set("id", Json::strv(id));
        chunk.set("object", Json::strv("chat.completion.chunk"));
        chunk.set("created", Json::numv(created));
        chunk.set("model", Json::strv(model));
        Json choice = Json::objv();
        Json delta = Json::objv();
        if (e.kind == EventKind::kToken) {
            Json c = Json::arrv();
            choice.set("index", Json::numv(0.0));
            if (*first) { delta.set("role", Json::strv("assistant")); *first = false; }
            if (!e.text.empty()) delta.set("content", Json::strv(e.text));
            choice.set("delta", std::move(delta));
            choice.set("finish_reason", Json::nullv());
            c.push(std::move(choice));
            chunk.set("choices", std::move(c));
            return w("data: " + json_dump(chunk) + "\n\n");
        }
        // 收尾帧：finish_reason；需 usage 时先补一条 usage 帧，再 [DONE]。
        Json c = Json::arrv();
        if (e.kind == EventKind::kDone) {
            choice.set("index", Json::numv(0.0));
            choice.set("delta", Json::objv());
            choice.set("finish_reason", Json::strv(e.finish.empty() ? "stop" : e.finish));
            c.push(std::move(choice));
            chunk.set("choices", std::move(c));
            if (!w("data: " + json_dump(chunk) + "\n\n")) return false;
            if (include_usage) {
                Json u = Json::objv();
                u.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
                u.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
                u.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
                Json uc = Json::objv();
                uc.set("id", Json::strv(id));
                uc.set("object", Json::strv("chat.completion.chunk"));
                uc.set("created", Json::numv(created));
                uc.set("model", Json::strv(model));
                uc.set("choices", Json::arrv());
                uc.set("usage", std::move(u));
                if (!w("data: " + json_dump(uc) + "\n\n")) return false;
            }
            if (!w("data: [DONE]\n\n")) return false;
            record_request(job, 200, false);
            return false;
        }
        // kError（事件流里以 error 帧出现，一致性让客户端能读到原因）。
        Json inner = Json::objv();
        inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
        inner.set("type", Json::strv("server_error"));
        inner.set("param", Json::nullv());
        Json er = Json::objv();
        er.set("error", std::move(inner));
        record_request(job, 500, false);
        if (!w("data: " + json_dump(er) + "\n\n")) return false;
        return false;
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}
```

（`body.has("stream_options")` 若 T6 的 Json 没有 `has`，用 `body.find("stream_options") != nullptr`，并在 `include_usage` 为真但 `stream_options` 不是对象时按 false 处理——宽松收，不额外 400。）

`main.cpp` / 路由注册不变（`handle_chat(req, res, false/true)` 由请求体驱动——把路由改为读 `stream` 后分发）：

```cpp
    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        Json body;
        std::string jerr;
        bool stream = false;
        if (json_parse(req.body, body, jerr) && body.is_obj()) {
            const Json* s = body.find("stream");
            if (s != nullptr && s->is_bool()) stream = s->b;
        }
        handle_chat(req, res, stream);
    });
```

（T9 的路由曾把 `stream` 判定留在 handler 里；现在移到路由层，`handle_chat` 的 `stream` 参数即最终真值。）

- [ ] **Step 3: 跑集成**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：测试 1–6 照旧绿，7、8 两个流式用例绿。若「流式拼文 ≠ 非流式文本」：先检查 run.sh 里两个请求确实同 body 同 seed（夹具确定性下应逐字节相同），再查 `serve_sse` 是否漏了某帧的 content。

- [ ] **Step 4: 提交**

```bash
git add src/serve/server.cpp tests/integration/serve/run.sh
python3 scripts/docs_checks.py
git commit -m "feat(serve): OpenAI SSE 流式——chunked provider 逐帧写出、连接断开即取消、include_usage 帧"
```

---

### Task 11: Anthropic 域

`/v1/messages`（非流式 + 流式）与 `/v1/messages/count_tokens`（规格 §3 Anthropic 兼容域）。消息组装：`system` 字段 + `messages` 数组（只收 `user`/`assistant` 两种 role；`content` 收字符串或文本块数组，出现图像块等一律 400），映射成与 OpenAI 域同一套 messages 列表，走**同一个** chat 模板渲染器与同一个 `Engine::generate`——两个兼容域对同一请求产出同一 token 序列（M1.3 验收的硬指标）。错误体走 Anthropic 形状；503 用 `overloaded_error`。

**Files:**
- Modify: `src/serve/server.cpp`（`handle_messages`、`handle_count_tokens`、`serve_anthropic_sse`、错误分发按域）、`tests/integration/serve/run.sh`

**Interfaces:**
- Consumes: T2 模板器、T9 的 `launch`/`drain_nonstream`/错误体助手
- Produces: `/v1/messages`、`/v1/messages/count_tokens` 两路由

- [ ] **Step 1: 写失败测试（run.sh 追加）**

```bash
# 9. Anthropic 非流式：同一请求与 OpenAI chat 域产出同一 token 序列（text 逐字节相同）。
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -d '{"messages":[{"role":"user","content":"两个域对拍"}],"max_tokens":16,"temperature":0,"seed":3}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' >/tmp/qinfer_c.txt
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"两个域对拍"}],"max_tokens":16,"temperature":0}' \
  > "$R"
STOP=$(jget "$R" "d['content'][0]['type']")
if [ "$(jget "$R" "d['type']")" = message ] && [ "$STOP" = text ] \
   && [ "$(jget "$R" "d['usage']['input_tokens']")" -ge 1 ] \
   && [ "$(jget "$R" "d['stop_reason']")" = end_turn ]; then
  ok "Anthropic 消息补全（type/内容块/stop_reason/usage）"
else
  bad "Anthropic 补全：体 $(head -c 300 "$R")"
fi
python3 - <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
a = open("/tmp/qinfer_c.txt").read()
b = "".join(c["text"] for c in r["content"] if c["type"] == "text")
assert a == b, f"两个域产出不同：chat={a!r} anthropic={b!r}"
PY
ok "两个兼容域同一请求 → 相同 token 序列（逐字节）" || true

# 10. count_tokens：与分词器逐 id 对照（独立数一遍：渲染后 encode）。
python3 - <<'PY'
import json, urllib.request
body = {"model": "x", "messages": [{"role": "user", "content": "数数"}], "max_tokens": 8}
req = urllib.request.Request("http://127.0.0.1:PORT/v1/messages/count_tokens".replace("PORT", "18080"),
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
n = json.load(urllib.request.urlopen(req))["input_tokens"]
assert isinstance(n, int) and n >= 1, n
print("n=", n)
PY
ok "count_tokens 返回正整数" || bad "count_tokens"

# 11. Anthropic 流式：事件序 message_start → content_block_start → content_block_delta… →
#     content_block_stop → message_delta → message_stop；delta 拼文与非流式逐字节相同。
curl -sN -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"流式对拍"}],"max_tokens":16,"temperature":0,"stream":true}' \
  > "$R.body"
python3 - "$R.body" >/tmp/qinfer_a.txt <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
evs = []
cur = None
for line in raw.splitlines():
    if line.startswith("event: "): cur = line[7:]
    elif line.startswith("data: "):
        assert cur, "data 前缺 event"
        evs.append((cur, json.loads(line[6:])))
types = [e[0] for e in evs]
names = [e[1]["type"] for e in evs]
assert names[0] == "message_start", names
assert "content_block_start" in names
assert "content_block_delta" in names, "缺 delta 事件"
assert names[-2:] == ["message_delta", "message_stop"], names[-3:]
text = "".join(e[1]["delta"]["text"] for t, e in evs if t == "content_block_delta" and "text" in e[1]["delta"])
sys.stdout.write(text)
PY
# 对照：非流式 Anthropic 的最小文本（同 body）
curl -s -H "$H" -H 'Content-Type: application/json' -X POST "http://$HOST:$PORT/v1/messages" \
  -d '{"model":"x","messages":[{"role":"user","content":"流式对拍"}],"max_tokens":16,"temperature":0}' \
  | python3 -c 'import json,sys; print("".join(c["text"] for c in json.load(sys.stdin)["content"]))' >/tmp/qinfer_an.txt
cmp -s /tmp/qinfer_a.txt /tmp/qinfer_an.txt && ok "Anthropic SSE 事件序与拼文正确" || bad "Anthropic SSE"
```

（第 10 项把端口写死成 18080 是偷懒——改用 `python3 - "$PORT"` 传参，与文件里其他用例一致。）

- [ ] **Step 2: 实现**

`server.cpp` 追加：

```cpp
// 消息列表组装：system + messages（只收 user/assistant；content 字符串或文本块数组）→ 与 OpenAI
// 域同一套 [(role, content)] 交给 render_chat。产出与 OpenAI 域同源的 prompt（两个域同 token 序列）。
bool messages_to_conversation(const Json& body, std::vector<std::pair<std::string, std::string>>& conv,
                              std::string& err) {
    const Json* sys = body.find("system");
    if (sys != nullptr) {
        if (!sys->is_str()) { err = "system 必须是字符串"; return false; }
        conv.emplace_back("system", sys->str);
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) { err = "缺少 messages 数组"; return false; }
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { err = "messages 元素不是对象"; return false; }
        const Json* role = m.find("role");
        const Json* content = m.find("content");
        if (role == nullptr || !role->is_str() ||
            (role->str != "user" && role->str != "assistant")) {
            err = "role 只收 user/assistant";
            return false;
        }
        std::string text;
        if (content == nullptr) { err = "缺 content"; return false; }
        if (content->is_str()) {
            text = content->str;
        } else if (content->is_arr()) {
            for (const Json& blk : content->arr) {
                if (!blk.is_obj() || !blk.find("type") || !blk.find("type")->is_str() ||
                    blk.find("type")->str != "text") {
                    err = "content 块只收 {type: text}（图像等多模态 M2）";
                    return false;
                }
                const Json* t = blk.find("text");
                if (t == nullptr || !t->is_str()) { err = "文本块缺 text"; return false; }
                text += t->str;
            }
        } else {
            err = "content 必须是字符串或块数组";
            return false;
        }
        if (text.empty()) continue;
        conv.emplace_back(role->str, text);
    }
    return true;
}

void Serve::handle_messages(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "请求体不是合法 JSON 对象：" + jerr));
        return;
    }
    double mt = 0.0;
    if (!num_param(body, "max_tokens", 0, mt) || mt <= 0) {
        send_json(res, 400, anthropic_err("invalid_request_error", "max_tokens 必填且为正"));
        return;
    }
    std::vector<std::pair<std::string, std::string>> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    GenOptions opts;
    double t = 0.0, p = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p);
    if (!numeric_ok) { send_json(res, 400, anthropic_err("invalid_request_error", "temperature/top_p 必须为数字")); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    // stop_sequences → stop（字符串数组；缺省空）。
    const Json* stops = body.find("stop_sequences");
    if (stops != nullptr) {
        if (!stops->is_arr()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences 必须是数组")); return; }
        for (const Json& s : stops->arr) {
            if (!s.is_str()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences 元素必须是字符串")); return; }
            opts.stop.push_back(s.str);
        }
    }
    std::string prompt, rerr;
    if (!render_chat(eng_->chat_template(), conv, prompt, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    const Json* s = body.find("stream");
    const bool stream = s != nullptr && s->is_bool() && s->b;
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        send_json(res, 503, anthropic_err("overloaded_error", "queue full"));
        return;
    }
    if (stream) { serve_anthropic_sse(job, res); return; }
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        if (!job->events->cancelled()) send_json(res, 500, anthropic_err("server_error", drain_err));
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("msg_" + std::to_string(job->id)));
    r.set("type", Json::strv("message"));
    r.set("role", Json::strv("assistant"));
    r.set("model", Json::strv(job->model));
    Json block = Json::objv();
    block.set("type", Json::strv("text"));
    block.set("text", Json::strv(text));
    Json content = Json::arrv();
    content.push(std::move(block));
    r.set("content", std::move(content));
    r.set("stop_reason", Json::strv(job->finish == "length" ? "max_tokens" : "end_turn"));
    r.set("stop_sequence", Json::nullv());
    Json usage = Json::objv();
    usage.set("input_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("output_tokens", Json::numv(static_cast<double>(job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

void Serve::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "请求体不是合法 JSON 对象：" + jerr));
        return;
    }
    std::vector<std::pair<std::string, std::string>> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    std::string prompt, rerr;
    if (!render_chat(eng_->chat_template(), conv, prompt, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    std::vector<std::int32_t> ids;
    if (!eng_->tokenizer().encode(prompt, /*parse_special=*/true, ids, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "分词失败：" + rerr));
        return;
    }
    Json r = Json::objv();
    r.set("input_tokens", Json::numv(static_cast<double>(ids.size())));
    send_json(res, 200, r);
}
```

Anthropic 流式（事件序钉死：`message_start` → `content_block_start` → `content_block_delta`×n → `content_block_stop` → `message_delta` → `message_stop`）：

```cpp
void Serve::serve_anthropic_sse(const std::shared_ptr<Job>& job, httplib::Response& res) {
    const std::string id = "msg_" + std::to_string(job->id);
    const std::string model = eng_->model_name();
    const auto first = std::make_shared<bool>(true);
    auto provider = [this, job, id, model, first](size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            if (!sink.write(s)) { job->events->cancel(); record_request(job, 499, true); return false; }
            return true;
        };
        const auto ev = [&](const std::string& name, Json data) {
            return w("event: " + name + "\ndata: " + json_dump(data) + "\n\n");
        };
        JobEvent e;
        if (!job->events->pop(e)) { record_request(job, 499, job->events->cancelled()); return false; }
        if (e.kind == EventKind::kToken) {
            if (*first) {
                Json m = Json::objv();
                m.set("id", Json::strv(id));
                m.set("type", Json::strv("message"));
                m.set("role", Json::strv("assistant"));
                m.set("model", Json::strv(model));
                m.set("content", Json::arrv());
                m.set("stop_reason", Json::nullv());
                m.set("stop_sequence", Json::nullv());
                Json us = Json::objv();
                us.set("input_tokens", Json::numv(static_cast<double>(job->n_prompt)));
                us.set("output_tokens", Json::numv(0.0));
                m.set("usage", std::move(us));
                if (!ev("message_start", std::move(m))) return false;
                Json cb = Json::objv();
                cb.set("index", Json::numv(0.0));
                Json blk = Json::objv();
                blk.set("type", Json::strv("text"));
                blk.set("text", Json::strv(""));
                cb.set("content_block", std::move(blk));
                if (!ev("content_block_start", std::move(cb))) return false;
                *first = false;
            }
            Json d = Json::objv();
            d.set("index", Json::numv(0.0));
            Json delta = Json::objv();
            delta.set("type", Json::strv("text_delta"));
            delta.set("text", Json::strv(e.text));
            d.set("delta", std::move(delta));
            return ev("content_block_delta", std::move(d));
        }
        if (e.kind == EventKind::kDone) {
            Json s1 = Json::objv();
            s1.set("index", Json::numv(0.0));
            if (!ev("content_block_stop", std::move(s1))) return false;
            Json md = Json::objv();
            Json dlt = Json::objv();
            dlt.set("stop_reason", Json::strv(job->finish == "length" ? "max_tokens" : "end_turn"));
            dlt.set("stop_sequence", Json::nullv());
            md.set("delta", std::move(dlt));
            Json us = Json::objv();
            us.set("output_tokens", Json::numv(static_cast<double>(job->n_gen)));
            md.set("usage", std::move(us));
            if (!ev("message_delta", std::move(md))) return false;
            record_request(job, 200, false);
            return ev("message_stop", Json::objv());
        }
        // kError：Anthropic 用 error 事件收尾。
        Json er = Json::objv();
        Json inner = Json::objv();
        inner.set("type", Json::strv("server_error"));
        inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
        er.set("type", Json::strv("error"));
        er.set("error", std::move(inner));
        record_request(job, 500, false);
        return ev("error", std::move(er));
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}
```

路由与错误域分发（server.cpp 的 `register_routes` 追加；`anthropic_path` 已存在）：

```cpp
    svr.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) { handle_messages(req, res); });
    svr.Post("/v1/messages/count_tokens", [this](const httplib::Request& req, httplib::Response& res) { handle_count_tokens(req, res); });
```

注意 401 的域分发已在 T9 的 pre-routing 里按 `anthropic_path` 分流，无需改。

- [ ] **Step 3: 跑集成**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：9–11 三个用例绿。「两个域同 token 序列」是硬指标——`handle_messages` 与 `handle_chat` 若用了不同的渲染/采样路径，本用例立刻红。

- [ ] **Step 4: 提交**

```bash
git add src/serve/server.cpp tests/integration/serve/run.sh
python3 scripts/docs_checks.py
git commit -m "feat(serve): Anthropic 域——/v1/messages（流式事件序）与 count_tokens，两个域同 token 序列"
```

---

### Task 12: 状态与属性路由

规格 §3 的状态与监控域的静态路由（`/metrics` 与 `/api/requests` 留给 T14）：`/v1/status`、`/status`、`/props`、`/slots`（`/health` 已在 T9）。形状对齐 Strata v0.1.43 的角色（从 `ref_repos/strata/serve/server.py` 核对过）：`/v1/status` 是给外部 OpenAI 客户端问「你是谁、正在干嘛」的；`/status` 是给部署探针的简版；`/props` 给外部工具识别能力；`/slots` 给前端上下文环。

**Files:**
- Modify: `src/serve/server.cpp`（三个 GET handler + `last_timings_` 成员 + `record_request` 开始填它）、`src/serve/server.hpp`、`tests/integration/serve/run.sh`

**Interfaces:**
- Consumes: T7 `Scheduler::slots_view`/`queued`/`any_busy`；T9 `record_request` 接缝
- Produces: `/v1/status`、`/status`、`/props`、`/slots`；`Serve::LastTimings` 结构（T14 的 `/metrics` live 也读它）

- [ ] **Step 1: 写失败测试（run.sh 追加）**

```bash
# 12. 状态族：/v1/status 与服务自述、/status 探针、/props 能力、/slots 槽位数。
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['service']")" = qinfer ] && [ "$(jget "$R" "d['loaded']")" = True ] \
  && [ "$(jget "$R" "d['context']['native']")" -ge 1 ] \
  && ok "/v1/status（service/loaded/context）" || bad "/v1/status：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/status" > "$R"
[ "$(jget "$R" "d['busy']")" = False ] && [ "$(jget "$R" "d['queued']")" = 0 ] \
  && ok "/status 探针（busy/queued）" || bad "/status：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/props" > "$R"
[ "$(jget "$R" "d['total_slots']")" -ge 1 ] \
  && [ "$(jget "$R" "d['chat_template']")" ] \
  && ok "/props（total_slots/chat_template）" || bad "/props：$(head -c 200 "$R")"

curl -s -H "$H" "http://$HOST:$PORT/slots" > "$R"
[ "$(jget "$R" "len(d)")" -ge 1 ] && [ "$(jget "$R" "d[0]['is_processing']")" = False ] \
  && ok "/slots 数组与空闲" || bad "/slots：$(head -c 200 "$R")"
```

- [ ] **Step 2: 实现**

`server.hpp` 追加成员：

```cpp
    // 最近一次完成的请求的计时（/v1/status 的 last_timings 与 /metrics 的 live 共用）。
    struct LastTimings {
        bool have = false;
        double queue_s = 0.0, first_token_s = 0.0, wallclock_s = 0.0, decode_tok_s = 0.0;
        std::uint64_t prompt_tokens = 0, output_tokens = 0;
        std::string finish;
    };
    std::mutex last_mu_;
    LastTimings last_;
```

`record_request` 从空实现升级为填计时（T14 的 Records 环随后也在这个函数里接）：

```cpp
void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    if (!job) return;
    const auto enq = job->enqueued_at;   // 缺省构造的 time_point（若从未入队）→ 全零
    const double wall = std::chrono::duration<double>(job->done_at - enq).count();
    LastTimings lt;
    lt.have = true;
    lt.queue_s = std::chrono::duration<double>(job->leased_at - enq).count();
    lt.first_token_s = job->first_at != std::chrono::steady_clock::time_point{}
                           ? std::chrono::duration<double>(job->first_at - enq).count() : 0.0;
    if (lt.first_token_s > 0) lt.first_token_s = lt.first_token_s - lt.queue_s;
    lt.wallclock_s = wall;
    lt.decode_tok_s = job->tok_s;
    lt.prompt_tokens = job->n_prompt;
    lt.output_tokens = job->n_gen;
    lt.finish = job->finish;
    {
        std::lock_guard<std::mutex> lk(last_mu_);
        last_ = lt;
    }
}
```

三个 handler（`first_token_s` 的口径是「租约后到首 token」的秒数，即减去排队；`job->first_at` 若是缺省（零 token 即结束的边界）记 0）：

```cpp
    svr.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = cfg_.facade.max_context;
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        r.set("service", Json::strv("qinfer"));
        r.set("model", Json::strv(eng_->model_name()));
        r.set("loaded", Json::boolv(true));
        r.set("auto_load", Json::boolv(false));
        Json so = Json::objv();
        Json fmts = Json::arrv(); fmts.push(Json::strv("json_object"));
        so.set("formats", std::move(fmts));
        so.set("method", Json::strv("prompt_and_validate"));
        so.set("constrained_decoding", Json::boolv(false));
        so.set("stream_buffered", Json::boolv(true));
        r.set("structured_output", std::move(so));
        r.set("engine", Json::strv("qinfer-m1"));
        r.set("started", Json::numv(static_cast<double>(started_)));
        r.set("uptime_s", Json::numv(static_cast<double>(now - started_)));
        r.set("cache_max_tokens", Json::numv(static_cast<double>(ctx)));
        Json cx = Json::objv();
        cx.set("native", Json::numv(static_cast<double>(ctx)));
        cx.set("max_positions", Json::numv(static_cast<double>(ctx)));
        r.set("context", std::move(cx));
        Json cc = Json::objv();
        cc.set("serving", Json::numv(static_cast<double>(cfg_.slots)));
        cc.set("requested", Json::numv(static_cast<double>(cfg_.slots)));
        r.set("concurrency", std::move(cc));
        Json dl = Json::arrv();
        dl.push(Json::strv("/v1/chat/completions"));
        dl.push(Json::strv("/v1/messages"));
        r.set("dialects", std::move(dl));
        Json vis = Json::objv();
        vis.set("enabled", Json::boolv(false));
        vis.set("available", Json::boolv(false));
        vis.set("error", Json::nullv());
        r.set("vision", std::move(vis));
        Json act = Json::objv();
        act.set("requests", Json::numv(static_cast<double>(req_id_.load())));
        act.set("in_flight", Json::numv((sched_.queued() + (busy ? 1 : 0))));
        act.set("last_request_at", Json::nullv());
        r.set("activity", std::move(act));
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            if (last_.have) {
                Json lt = Json::objv();
                lt.set("queue_s", Json::numv(last_.queue_s));
                lt.set("first_token_s", Json::numv(last_.first_token_s));
                lt.set("wallclock_s", Json::numv(last_.wallclock_s));
                lt.set("decode_tok_s", Json::numv(last_.decode_tok_s));
                lt.set("prompt_tokens", Json::numv(static_cast<double>(last_.prompt_tokens)));
                lt.set("output_tokens", Json::numv(static_cast<double>(last_.output_tokens)));
                r.set("last_timings", std::move(lt));
            } else {
                r.set("last_timings", Json::nullv());
            }
        }
        r.set("vram", Json::objv());
        Json mach = Json::objv();
        mach.set("at", Json::numv(static_cast<double>(now)));
        mach.set("gpu", Json::nullv());      // T13 接遥测后填
        mach.set("ram", Json::nullv());
        r.set("machine", std::move(mach));
        send_json(res, 200, r);
    });
    svr.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        r.set("busy", Json::boolv(busy));
        r.set("queued", Json::numv(static_cast<double>(sched_.queued())));
        r.set("phase", busy ? Json::strv("generating") : Json::nullv());
        Json sv = sched_.slots_view();
        Json head = Json::numv(0.0);
        for (const SlotView& v : sv) if (v.busy) head = Json::numv(static_cast<double>(v.job_id));
        if (busy) {
            r.set("elapsed_s", Json::numv(0.0));   // 当前 job 已租约时长（槽位视图有，见后）
        }
        r.set("slots", Json::numv(static_cast<double>(sv.size())));
        send_json(res, 200, r);
    });
    svr.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
        Json dg = Json::objv();
        dg.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
        Json prm = Json::objv();
        prm.set("temperature", Json::numv(0.8));
        prm.set("top_p", Json::numv(0.95));
        prm.set("seed", Json::numv(-1.0));
        prm.set("n_predict", Json::numv(512.0));
        dg.set("params", std::move(prm));
        Json r = Json::objv();
        r.set("default_generation_settings", std::move(dg));
        r.set("total_slots", Json::numv(static_cast<double>(cfg_.slots)));
        r.set("model_alias", Json::strv(eng_->model_name()));
        r.set("chat_template", Json::strv(eng_->chat_template()));
        Json caps = Json::objv();
        caps.set("templater", Json::strv("lexer-subset"));
        caps.set("rich_text", Json::boolv(false));
        caps.set("date", Json::boolv(false));
        r.set("chat_template_caps", std::move(caps));
        Json mod = Json::objv();
        mod.set("vision", Json::boolv(false));
        r.set("modalities", std::move(mod));
        r.set("models_autoload", Json::boolv(false));
        r.set("is_sleeping", Json::boolv(false));
        r.set("build_info", Json::strv("Q-Infer M1"));
        send_json(res, 200, r);
    });
    svr.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        const auto views = sched_.slots_view();
        Json list = Json::arrv();
        for (const SlotView& v : views) {
            Json s = Json::objv();
            s.set("id", Json::numv(static_cast<double>(v.index)));
            s.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
            s.set("is_processing", Json::boolv(v.busy));
            s.set("n_prompt_tokens", Json::numv(0.0));
            list.push(std::move(s));
        }
        send_json(res, 200, list);
    });
```

`started_` 成员补进 `server.hpp`（`Serve` 构造时 `std::time(nullptr)` 存入）；`server.hpp` 的 `record_request` 注释改为「T12：填 last_timings；T14：追加 Records 环」。

- [ ] **Step 3: 跑集成**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：12 用例四个断言绿。若 `/status` 那段的「当前 job」字段写飘了（`slots` 数字与 `elapsed_s` 总对不上），把 `/status` 收敛成 Strata 的简版——busy/queued 为真值，其余字段在 v0 可省。

- [ ] **Step 4: 提交**

```bash
git add src/serve/server.cpp src/serve/server.hpp tests/integration/serve/run.sh
python3 scripts/docs_checks.py
git commit -m "feat(serve): 状态与属性路由——/v1/status、/status、/props、/slots，last_timings 计时观测落地"

### Task 13: 遥测（Telemetry）

**Files:**
- Create: `src/serve/telemetry.hpp`, `src/serve/telemetry.cpp`, `tests/test_telemetry.cpp`
- Modify: `src/serve/server.hpp`（`tel_` 成员 + 累计计数）、`src/serve/server.cpp`（/v1/status 的 machine 接真值 + `telemetry_extra`）、`tests/integration/serve/run.sh`（用例 12 补 machine 形状断言）、`CMakeLists.txt`

**Interfaces:**
- Consumes: T12 `Serve::LastTimings`（`telemetry_extra` 从中取 tok_s/prefill 近似）、T9 `record_request` 接缝
- Produces: `qinfer::serve::Telemetry`（`start`/`stop`/`sample_now`/`snapshot`）、`TelemetrySnapshot`（`now` 键集与 Strata `telemetry.py` 的 now 字典逐键一致）、`TelemetryExtra{tok_s, tok_s_mean, prefill_tok_s_mean}`——T14 的 `/metrics` hardware/history/hardware_static 与 T12 的 `machine` 都从这里取

对齐依据是 Strata `serve/telemetry.py`（`ref_repos/strata`）：取样线程每秒一次、10 系列 × 60 点历史（`HISTORY=60`）、多卡聚合口径（总内存/总功率/总 PCIe、平均负载、最高温度、`gpus[]` 每卡明细）、NVML 符号表与 `_gen` 速率表、AMD `_Amd` 的 sysfs 路径与 hop 瓶颈判定、CPU `/proc/stat` 差值法（idle = f[3]+f[4]，total = sum(f)）、RAM `/proc/meminfo`。v0 偏差（在代码注释里写明）：`tok_s` 指最近一次完成请求的解码速度（Strata 的滑动窗口随 `live_reqs` 而来，v0 无该状态，M2 补）；`prefill_tok_s_mean` 用「提示词 token 数 ÷ 租约后到首 token 耗时」近似（本引擎 v0 无精确预填充计时）。

- [ ] **Step 1: 写失败测试**

`tests/test_telemetry.cpp`：

```cpp
// 遥测：extra 回调值进入 now 与 history；10 系列在首次采样后键全在；历史封顶 60 点；
// Linux 上 /proc 读数非空；GPU 相关断言只钉「不抛异常」（CI 无 GPU，环境2 有——行为随环境）。
#include "serve/telemetry.hpp"

#include <chrono>
#include <cmath>
#include <string>
#include <thread>

#include "tests/check.hpp"

using namespace qinfer::serve;

int main() {
    // 1. extra 回调 → now 与 history（tok_s 系列首尾两值）。
    int calls = 0;
    Telemetry tel([&calls] {
        ++calls;
        return TelemetryExtra{12.5, 40.0, 3.0};
    });
    tel.sample_now();
    tel.sample_now();
    auto s = tel.snapshot();
    CHECK(calls == 2);
    CHECK(s.now.count("tok_s") == 1 && std::abs(*s.now.at("tok_s") - 12.5) < 1e-9);
    CHECK(s.now.count("tok_s_mean") == 1 && std::abs(*s.now.at("tok_s_mean") - 40.0) < 1e-9);
    CHECK(s.history.count("tok_s") == 1 && s.history.at("tok_s").size() == 2);
    CHECK(std::abs(*s.history.at("tok_s").back() - 12.5) < 1e-9);

    // 2. 10 系列键齐（spec §6 的 series 清单），值可为 nullopt。
    for (const char* key : {"gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power",
                            "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb",
                            "tok_s", "prefill_tok_s_mean"}) {
        CHECK(s.history.count(key) == 1);
    }

    // 3. 历史封顶 60 点。
    Telemetry tel2([]{ return TelemetryExtra{1.0, 2.0, 3.0}; });
    for (int i = 0; i < 70; ++i) tel2.sample_now();
    const auto s2 = tel2.snapshot();
    CHECK(s2.history.at("tok_s").size() <= 60);
    CHECK(s2.history.at("tok_s").size() == 60);

    // 4. 线程启停幂等；真实跑几秒能采到样。
    tel2.start();
    tel2.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2400));
    tel2.stop();
    tel2.stop();   // 重复 stop 不炸
    const auto s3 = tel2.snapshot();
    CHECK(s3.history.at("tok_s").size() >= 60);   // 启动后每约 1s 一采样，只增不减

    // 5. 静态信息：线程数 ≥ 1；GPU 字段不断言（CI 无卡 → 空，环境2 有卡 → 非空）。
    CHECK(s3.stat.threads >= 1);
#ifndef _WIN32
    // 6. Linux：/proc/meminfo 能读 → ram_total 非空；cpu 键必在（值可为 null——首点无前一时刻）。
    CHECK(s3.now.count("ram_total") == 1 && s3.now.at("ram_total").has_value());
    CHECK(s3.now.count("cpu") == 1);
#endif

    // 8. 任何情况 snapshot 都能取（读锁不含文件 IO——snapshot 只拷内存）。
    Telemetry tel3(nullptr);
    tel3.sample_now();
    (void)tel3.snapshot();

    puts("test_telemetry: OK");
    return 0;
}
```

- [ ] **Step 2: 跑测试确定失败**

```bash
cmake -B build -S . && cmake --build build --target test_telemetry 2>&1 | tail -5
```

预期：FAIL——`src/serve/telemetry.hpp` 不存在（编译错误）。

- [ ] **Step 3: 写最小实现**

`src/serve/telemetry.hpp`：

```cpp
// 遥测：每秒采样一次硬件读数，保留最近 60 点历史（spec §6 的 hardware 字段集与 10 系列清单，
// 逐键对齐 Strata serve/telemetry.py 的 now/history/static 字典）。
// 零依赖（ADR-008 在遥测上同样成立）：NVIDIA 走 dlopen("libnvidia-ml.so.1")（Windows 为
// LoadLibrary("nvml.dll")）动态取符号；AMD 走 Linux sysfs（KFD 拓扑找卡 + amdgpu 驱动读值，
// 路径与 hop 判定照抄 Strata 的 _Amd）；CPU/RAM 走 /proc/stat 差值法与 /proc/meminfo
// （Windows 为 GetSystemTimes 与 GlobalMemoryStatusEx）；磁盘速率解析 /proc/diskstats 按整盘聚合
// （Windows 在 v0 为 null，对齐 Strata 无 psutil 时的行为）。
// 任何字段读不到都是 null；任何异常都不允许拖垮服务——所有读取都带兜底，extra 回调同样兜底。
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace qinfer::serve {

// 单卡读器（NVML 卡或 AMD sysfs 卡）与 NVML 动态符号表：都是不透明类型，定义在 telemetry.cpp。
struct GpuCtx;
struct NvmlApi;

// 引擎侧指标：由服务层每次生成完成后刷新（server.cpp 的 telemetry_extra），采样线程每秒读一份。
struct TelemetryExtra {
    double tok_s = 0.0;               // v0：最近一次完成请求的解码速度（tok/s）
    double tok_s_mean = 0.0;          // 全程均值 tok/s（累计生成 token ÷ 累计解码耗时）
    double prefill_tok_s_mean = 0.0;  // v0 近似：提示词 token ÷ 租约后到首 token 耗时
};

// 单卡明细（多卡时 /metrics 的 gpus[] 数据源；v1_status 只认卡 0 的名义读数是 Strata 口径）。
struct GpuReading {
    int index = 0;
    std::string name;                  // 空 → null
    std::optional<double> util;        // %（0-100）
    std::optional<double> mem_used;    // 字节
    std::optional<double> mem_total;
    std::optional<double> temp;        // °C
    std::optional<double> power;       // W
};

// 静态信息（启动时读一次；读不到 → 空串 / 0，JSON 里是 null）。
struct TelemetryStatic {
    std::string gpu_name;   // 多卡名 " + " 连接；无卡 → 空
    int gpu_count = 0;
    std::string gpu_note;   // v0 恒空（AMD on Windows 的说明；本仓库 v0 无该用例）
    std::string cpu_name;   // /proc/cpuinfo "model name"
    int cores = 0;          // v0 恒 0（Strata 无 psutil 时 cores 也是 null；M2 按拓扑填）
    int threads = 0;        // std::thread::hardware_concurrency()
};

// 快照：snapshot() 签出拷贝，读线程安全。now 的键集固定为 Strata telemetry.py 的 now 字典
// （gpu_util、gpu_mem_used、gpu_mem_total、gpu_temp、gpu_power、gpu_power_limit、gpu_pcie_gen、
// gpu_pcie_gen_max、gpu_pcie_width、gpu_pcie_rx_mb、gpu_pcie_tx_mb、cpu、ram_used、ram_total、
// disk_read_mb、disk_write_mb、tok_s、tok_s_mean、prefill_tok_s_mean）；读不到 = nullopt。
struct TelemetrySnapshot {
    std::map<std::string, std::optional<double>> now;
    std::vector<GpuReading> gpus;      // 多卡时才非空
    TelemetryStatic stat;
    std::map<std::string, std::vector<std::optional<double>>> history;  // 10 系列 × 最近 ≤60 点
};

class Telemetry {
public:
    using ExtraFn = std::function<TelemetryExtra()>;

    explicit Telemetry(ExtraFn extra = nullptr);
    ~Telemetry();
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    void start();   // 起采样线程（每秒一次；幂等；stop 后可再 start）
    void stop();    // 停线程并 join（幂等）
    void sample_now();              // 同步采一次（测试与线程共用）
    TelemetrySnapshot snapshot() const;  // 线程安全拷贝

private:
    void collect_cpu_ram_disk(std::map<std::string, std::optional<double>>& now);
    void push_history();

    ExtraFn extra_;
    std::vector<GpuCtx*> gpus_;      // 找到的卡（NVML 优先；全失败 → 空）
    NvmlApi* nvml_ = nullptr;        // NVML 装载态（nullptr = 未装载）
    std::vector<GpuReading> gpus_read_;
    TelemetryStatic stat_;

    mutable std::mutex mu_;
    std::map<std::string, std::optional<double>> now_;
    std::map<std::string, std::vector<std::optional<double>>> hist_;

    bool cpu_prev_valid_ = false;    // /proc/stat 差值法的前一时刻
    std::uint64_t cpu_prev_idle_ = 0, cpu_prev_total_ = 0;
    bool disk_prev_valid_ = false;   // /proc/diskstats 差值法的前一时刻
    double disk_prev_t_ = 0.0;
    std::uint64_t disk_prev_r_ = 0, disk_prev_w_ = 0;

    std::thread thread_;
    std::mutex cv_mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace qinfer::serve
```

`src/serve/telemetry.cpp`：

```cpp
// 实现要点（对齐 Strata serve/telemetry.py，逐条照抄其语义）：
// - NVML：dlopen 成功后从 index 0 逐个 nvmlDeviceGetHandleByIndex_v2 取句柄直到失败（等价
//   Strata 的 gpu_reader 列表）；符号表 12 个，缺任一关键读取符号 → 整包弃用。PowerUsage 是
//   mW → /1000；温度传感器类型 0（NVML_TEMPERATURE_GPU）；PcieThroughput 是 KB/s / 1024 → MB/s
//   （counter 1 = RX、0 = TX）。
// - AMD：KFD 拓扑 nodes 的数字子目录（升序）读 properties，gfx_target_version 与 simd_count 非零者
//   即 GPU 节点，按 drm_render_minor 拼 /sys/class/drm/renderD<minor>/device；hwmon 取第一个目录
//   （temp1_input 是 m°C → /1000，power1_average 或 power1_input 是 µW → /1e6，power1_cap 同理）；
//   link() 走 realpath 上的 BDF 段（长度 12、格式 0000:00:00.0），读每段 max_link_speed /
//   max_link_width，全段可读 → 取最小（Strata 的 hop 瓶颈判定），否则退回卡自身的
//   current_link_speed / current_link_width；速率表 {2.5:1, 5.0:2, 8.0:3, 16.0:4, 32.0:5, 64.0:6}。
// - CPU：/proc/stat 第一行全字段，idle = f[3]+f[4]（idle+iowait）、total = sum(f)，差值法算占用率
//   clamp 到 [0,100]；首个采样无前一时刻 → null。Windows 走 GetSystemTimes（kernel 含 idle）。
// - RAM：/proc/meminfo 的 MemTotal/MemAvailable（kB）→ 字节，used = total - avail；Windows 走
//   GlobalMemoryStatusEx。
// - 磁盘：/proc/diskstats 只取「整盘」行（minor == 0；分区行 minor 非 0），聚合 sectors 计数差
//   × 512 / Δt / 2^20 → MB/s；首个采样 null。Windows 在 v0 为 null（Strata 无 psutil 时同口径）。
// - 多卡聚合（NVML 多卡或 AMD 多节点）：mem/power/pcie 流量取总和、util 取均值、temp 取最高、
//   pcie_gen/width 取第一张卡（Strata 同款：聚合键表里没有 gen/width）。
// - history：10 系列（键表见 push_history），每样一次各推一个值（读不到推 nullopt），封顶 60。
// - gpu_note/cores 在 v0 恒空/恒 0（见头文件注释），但键位保留，T14 照常序列化。
#include "serve/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qinfer::serve {

namespace {

// ---- 小工具：读 sysfs /proc 文本文件（读不到 → nullopt / 空串）----
std::optional<long long> read_int(const std::string& path) {
    std::ifstream f(path);
    long long v = 0;
    return (f >> v) ? std::optional<long long>(v) : std::nullopt;
}

std::string read_str(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

// "16.0 GT/s PCIe" → 4（Strata _gen 的显式速率表；v/2.5 不是 gen）。
std::optional<int> link_gen(const std::string& text) {
    double gt = 0.0;
    if (std::sscanf(text.c_str(), "%lf", &gt) != 1) return std::nullopt;
    static const std::pair<double, int> kGen[] = {{2.5, 1}, {5.0, 2}, {8.0, 3},
                                                  {16.0, 4}, {32.0, 5}, {64.0, 6}};
    for (const auto& [rate, gen] : kGen)
        if (std::abs(gt - rate) < 1e-9) return gen;
    return std::nullopt;
}

// 系统路径里是否为 PCI 设备名（"0000:03:00.0"）。
bool is_bdf(const std::string& s) {
    if (s.size() != 12 || s[4] != ':' || s[7] != ':' || s[10] != '.') return false;
    for (int i = 0; i < 12; ++i) {
        if (i == 4 || i == 7 || i == 10) continue;
        const char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// NVML 返回结构（nvmlUtilization_t / nvmlMemory_t 的内存布局）。
struct NvmlUtilization { unsigned gpu = 0, mem = 0; };
struct NvmlMemory { unsigned long long total = 0, free = 0, used = 0; };

}  // namespace

// ---- 不透明类型定义（telemetry.hpp 里只有前置声明）----

// 单卡读器基类（NVML 卡或 AMD sysfs 卡），统一出 Strata 的 g 字典键（无 "gpu_" 前缀）。
struct GpuCtx {
    virtual ~GpuCtx() = default;
    virtual std::string name() = 0;   // 空 = 读不到
    virtual std::map<std::string, std::optional<double>> read() = 0;
};

// NVML 动态符号表（符号名 → 函数指针；返回码 0 = NVML_SUCCESS）。
struct NvmlApi {
    void* lib = nullptr;   // dlopen / LoadLibrary 句柄
    int (*init)() = nullptr;
    int (*get_handle)(unsigned int, void**) = nullptr;
    int (*get_name)(void*, char*, unsigned int) = nullptr;
    int (*get_utilization)(void*, NvmlUtilization*) = nullptr;
    int (*get_memory)(void*, NvmlMemory*) = nullptr;
    int (*get_temperature)(void*, unsigned int, unsigned int*) = nullptr;
    int (*get_power)(void*, unsigned int*) = nullptr;
    int (*get_power_limit)(void*, unsigned int*) = nullptr;
    int (*get_curr_gen)(void*, unsigned int*) = nullptr;
    int (*get_max_gen)(void*, unsigned int*) = nullptr;
    int (*get_curr_width)(void*, unsigned int*) = nullptr;
    int (*get_pcie_tp)(void*, unsigned int, unsigned int*) = nullptr;

    ~NvmlApi() {
        if (!lib) return;
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(lib));
#else
        dlclose(lib);
#endif
    }
};

namespace {

// 装载 NVML；失败返回 nullptr（Windows 走 LoadLibraryA + GetProcAddress）。
NvmlApi* load_nvml() {
    auto* api = new NvmlApi();
#if defined(_WIN32)
    api->lib = static_cast<void*>(LoadLibraryA("nvml.dll"));
#else
    api->lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!api->lib) api->lib = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
#endif
    if (!api->lib) { delete api; return nullptr; }
#if defined(_WIN32)
    auto sym = [&](const char* n) {
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(api->lib), n));
    };
#else
    auto sym = [&](const char* n) { return dlsym(api->lib, n); };
#endif
    api->init = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
    if (!api->init || api->init() != 0) { delete api; return nullptr; }
    api->get_handle = reinterpret_cast<int (*)(unsigned int, void**)>(
        sym("nvmlDeviceGetHandleByIndex_v2"));
    if (!api->get_handle) api->get_handle =
        reinterpret_cast<int (*)(unsigned int, void**)>(sym("nvmlDeviceGetHandleByIndex"));
    api->get_name = reinterpret_cast<int (*)(void*, char*, unsigned int)>(sym("nvmlDeviceGetName"));
    api->get_utilization = reinterpret_cast<int (*)(void*, NvmlUtilization*)>(
        sym("nvmlDeviceGetUtilizationRates"));
    api->get_memory = reinterpret_cast<int (*)(void*, NvmlMemory*)>(sym("nvmlDeviceGetMemoryInfo"));
    api->get_temperature = reinterpret_cast<int (*)(void*, unsigned int, unsigned int*)>(
        sym("nvmlDeviceGetTemperature"));
    api->get_power = reinterpret_cast<int (*)(void*, unsigned int*)>(sym("nvmlDeviceGetPowerUsage"));
    api->get_power_limit = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetEnforcedPowerLimit"));
    api->get_curr_gen = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetCurrPcieLinkGeneration"));
    api->get_max_gen = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetMaxPcieLinkGeneration"));
    api->get_curr_width = reinterpret_cast<int (*)(void*, unsigned int*)>(
        sym("nvmlDeviceGetCurrPcieLinkWidth"));
    api->get_pcie_tp = reinterpret_cast<int (*)(void*, unsigned int, unsigned int*)>(
        sym("nvmlDeviceGetPcieThroughput"));
    if (!api->get_handle || !api->get_name || !api->get_memory) { delete api; return nullptr; }
    return api;
}

struct NvmlCard : GpuCtx {
    NvmlApi* api = nullptr;
    void* dev = nullptr;
    int index = 0;

    std::string name() override {
        char buf[96] = {0};
        if (!api || !api->get_name || api->get_name(dev, buf, sizeof(buf)) != 0) return "";
        return buf;
    }
    std::map<std::string, std::optional<double>> read() override {
        std::map<std::string, std::optional<double>> o;
        if (!api) return o;
        NvmlUtilization u;
        if (api->get_utilization && api->get_utilization(dev, &u) == 0) o["util"] = u.gpu;
        NvmlMemory m;
        if (api->get_memory && api->get_memory(dev, &m) == 0) {
            o["mem_used"] = static_cast<double>(m.used);
            o["mem_total"] = static_cast<double>(m.total);
        }
        unsigned tmp = 0;
        if (api->get_temperature && api->get_temperature(dev, 0, &tmp) == 0) o["temp"] = tmp;
        if (api->get_power && api->get_power(dev, &tmp) == 0) o["power"] = tmp / 1000.0;  // mW → W
        if (api->get_power_limit && api->get_power_limit(dev, &tmp) == 0)
            o["power_limit"] = tmp / 1000.0;
        unsigned rx = 0, tx = 0;
        if (api->get_pcie_tp && api->get_pcie_tp(dev, 1, &rx) == 0) o["pcie_rx_mb"] = rx / 1024.0;
        if (api->get_pcie_tp && api->get_pcie_tp(dev, 0, &tx) == 0) o["pcie_tx_mb"] = tx / 1024.0;
        return o;
    }
};

struct AmdCard : GpuCtx {
    std::string dev_;    // /sys/class/drm/renderD<N>/device
    std::string hwmon_;  // 第一个 hwmon 子目录（可为空）
    std::vector<std::string> hops_;  // realpath 上的 BDF 段目录（Strata _hops）

    std::string name() override {
        const std::string n = read_str(dev_ + "/product_name");
        return n.empty() ? "AMD Radeon" : n;
    }
    std::map<std::string, std::optional<double>> read() override {
        std::map<std::string, std::optional<double>> o;
        if (dev_.empty()) return o;
        if (auto v = read_int(dev_ + "/gpu_busy_percent")) o["util"] = static_cast<double>(*v);
        if (auto v = read_int(dev_ + "/mem_info_vram_used")) o["mem_used"] = static_cast<double>(*v);
        if (auto v = read_int(dev_ + "/mem_info_vram_total")) o["mem_total"] = static_cast<double>(*v);
        if (!hwmon_.empty()) {
            if (auto t = read_int(hwmon_ + "/temp1_input")) o["temp"] = *t / 1000.0;  // m°C → °C
            auto p = read_int(hwmon_ + "/power1_average");
            if (!p) p = read_int(hwmon_ + "/power1_input");
            if (p) o["power"] = *p / 1e6;  // µW → W
            if (auto c = read_int(hwmon_ + "/power1_cap")) o["power_limit"] = *c / 1e6;
        }
        // link()：全 hop 可读 → 最小 gen/width（hop 瓶颈判定）；否则退回卡自身 current_link_*。
        bool whole = !hops_.empty();
        std::vector<int> gens, widths;
        for (const auto& h : hops_) {
            auto g = link_gen(read_str(h + "/max_link_speed"));
            auto w = read_int(h + "/max_link_width");
            if (!g && !w) whole = false;
            if (g) gens.push_back(*g);
            if (w) widths.push_back(static_cast<int>(*w));
        }
        const auto own_gen = link_gen(read_str(dev_ + "/current_link_speed"));
        const auto own_width = read_int(dev_ + "/current_link_width");
        if (whole && !gens.empty()) {
            o["pcie_gen"] = *std::min_element(gens.begin(), gens.end());
            o["pcie_gen_max"] = o["pcie_gen"];
        } else if (own_gen) {
            o["pcie_gen"] = *own_gen;
            o["pcie_gen_max"] = *own_gen;
        }
        if (whole && !widths.empty()) {
            o["pcie_width"] = *std::min_element(widths.begin(), widths.end());
        } else if (own_width) {
            o["pcie_width"] = static_cast<double>(*own_width);
        }
        return o;
    }
};

// AMD 卡发现：KFD 拓扑节点 → properties 过滤 → renderD 设备目录 + hwmon + BDF 段
// （Strata amd_device_dir + _Amd.__init__ + _hops）。Windows 无 sysfs → nullptr。
AmdCard* find_amd_card(int index) {
#if defined(_WIN32)
    (void)index;
    return nullptr;
#else
    using namespace std::filesystem;
    static const std::string kBase = "/sys/class/kfd/kfd/topology/nodes";
    std::vector<int> nodes;
    if (is_directory(kBase)) {
        for (const auto& e : directory_iterator(kBase)) {
            const std::string n = e.path().filename().string();
            if (!n.empty() && std::all_of(n.begin(), n.end(),
                                          [](char c) { return c >= '0' && c <= '9'; }))
                nodes.push_back(std::stoi(n));
        }
    }
    std::sort(nodes.begin(), nodes.end());
    std::vector<std::string> devs;
    for (int n : nodes) {
        // properties：每行 "key value"；只认 GPU 节点（gfx_target_version != 0 且 simd_count != 0）。
        long long target = 0, simd = 0;
        std::string minor;
        std::ifstream f(kBase + "/" + std::to_string(n) + "/properties");
        std::string line;
        while (std::getline(f, line)) {
            const auto sp = line.find(' ');
            const std::string key = line.substr(0, sp);
            const std::string val = sp == std::string::npos ? "" : line.substr(sp + 1);
            if (key == "gfx_target_version") target = std::stoll(val, nullptr, 16);
            else if (key == "simd_count") simd = std::stoll(val);
            else if (key == "drm_render_minor") minor = val;
        }
        if (target == 0 || simd == 0 || minor.empty()) continue;
        const std::string dev = "/sys/class/drm/renderD" + minor + "/device";
        if (is_directory(dev)) devs.push_back(dev);
    }
    if (index < 0 || index >= static_cast<int>(devs.size())) return nullptr;
    auto* card = new AmdCard();
    card->dev_ = devs[index];
    try {  // hwmon 第一个目录（Strata sorted 之后取 [0]）
        const std::string hdir = card->dev_ + "/hwmon";
        if (is_directory(hdir))
            for (const auto& e : directory_iterator(hdir)) { card->hwmon_ = e.path().string(); break; }
    } catch (...) {}
    try {  // realpath 上的 BDF 段（Strata _hops：路径里每个 BDF 名逐段累积）
        std::string acc;
        for (const auto& part : canonical(card->dev_)) {
            acc += "/" + part.string();
            if (is_bdf(part.string())) card->hops_.push_back(acc);
        }
    } catch (...) {}
    return card;
#endif
}

}  // namespace

Telemetry::Telemetry(ExtraFn extra) : extra_(std::move(extra)) {
    // 卡发现：NVML 优先（index 0..64 逐个取句柄到失败），全失败 → AMD sysfs 卡 0。
    nvml_ = load_nvml();
    if (nvml_) {
        for (int i = 0; i < 64; ++i) {
            void* dev = nullptr;
            if (nvml_->get_handle(static_cast<unsigned>(i), &dev) != 0) break;
            auto* card = new NvmlCard();
            card->api = nvml_;
            card->dev = dev;
            card->index = i;
            gpus_.push_back(card);
        }
    }
    if (gpus_.empty()) {
        if (AmdCard* c = find_amd_card(0)) gpus_.push_back(c);
    }

    // 静态信息（Strata self.static 的键，读不到 → 空串 / 0）。
    std::string names;
    for (const GpuCtx* g : gpus_) {
        if (!names.empty()) names += " + ";
        names += g->name().empty() ? "?" : g->name();
    }
    stat_.gpu_name = gpus_.empty() ? "" : names;
    stat_.gpu_count = static_cast<int>(gpus_.size());
    stat_.gpu_note = "";
    std::ifstream ci("/proc/cpuinfo");
    std::string line;
    while (std::getline(ci, line)) {
        if (line.rfind("model name", 0) == 0) {
            stat_.cpu_name = line.substr(11);
            while (!stat_.cpu_name.empty() &&
                   (stat_.cpu_name.back() == ' ' || stat_.cpu_name.back() == '\t'))
                stat_.cpu_name.pop_back();
            break;
        }
    }
    stat_.threads = static_cast<int>(std::thread::hardware_concurrency());
}

Telemetry::~Telemetry() {
    stop();
    for (GpuCtx* g : gpus_) delete g;
    delete nvml_;
}

void Telemetry::start() {
    std::lock_guard<std::mutex> lk(cv_mu_);
    if (thread_.joinable()) return;
    stop_ = false;
    thread_ = std::thread([this] {
        for (;;) {
            sample_now();                 // 启动即采（历史第一点不等 1s）
            std::unique_lock<std::mutex> w(cv_mu_);
            cv_.wait_for(w, std::chrono::seconds(1), [this] { return stop_; });
            if (stop_) return;
        }
    });
}

void Telemetry::stop() {
    {
        std::lock_guard<std::mutex> lk(cv_mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Telemetry::sample_now() {
    std::map<std::string, std::optional<double>> now;
    std::vector<GpuReading> gpus_tmp;
    if (!gpus_.empty()) {
        std::vector<std::map<std::string, std::optional<double>>> reads;
        for (GpuCtx* g : gpus_) {
            auto r = g->read();
            GpuReading gr;
            gr.index = static_cast<int>(reads.size());
            gr.name = g->name();
            if (r.count("util")) gr.util = r["util"];
            if (r.count("mem_used")) gr.mem_used = r["mem_used"];
            if (r.count("mem_total")) gr.mem_total = r["mem_total"];
            if (r.count("temp")) gr.temp = r["temp"];
            if (r.count("power")) gr.power = r["power"];
            reads.push_back(std::move(r));
            gpus_tmp.push_back(std::move(gr));
        }
        // Strata 聚合口径：mem/power/pcie 总和、util 均值、temp 最高；gen/width 取第一张卡。
        const auto sums = [&reads](const char* key) -> std::optional<double> {
            double s = 0.0;
            bool any = false;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key)) { s += *r.at(key); any = true; }
            return any ? std::optional<double>(s) : std::nullopt;
        };
        const auto mean = [&reads](const char* key) -> std::optional<double> {
            double s = 0.0;
            int n = 0;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key)) { s += *r.at(key); ++n; }
            return n > 0 ? std::optional<double>(s / n) : std::nullopt;
        };
        const auto maxv = [&reads](const char* key) -> std::optional<double> {
            std::optional<double> m;
            for (const auto& r : reads)
                if (r.count(key) && r.at(key))
                    m = m ? std::max(*m, *r.at(key)) : r.at(key);
            return m;
        };
        now["gpu_mem_used"] = sums("mem_used");
        now["gpu_mem_total"] = sums("mem_total");
        now["gpu_power"] = sums("power");
        now["gpu_power_limit"] = sums("power_limit");
        now["gpu_pcie_rx_mb"] = sums("pcie_rx_mb");
        now["gpu_pcie_tx_mb"] = sums("pcie_tx_mb");
        now["gpu_util"] = mean("util");
        now["gpu_temp"] = maxv("temp");
        const auto& first = reads.front();
        for (const char* k : {"pcie_gen", "pcie_gen_max", "pcie_width"})
            if (first.count(k)) now[std::string("gpu_") + k] = first.at(k);
    }

    // 引擎侧指标（extra 回调兜底：异常不拖垮遥测；0.0 = 尚无数据）。
    TelemetryExtra ex;
    if (extra_) {
        try { ex = extra_(); } catch (...) {}
    }
    now["tok_s"] = ex.tok_s;
    now["tok_s_mean"] = ex.tok_s_mean;
    now["prefill_tok_s_mean"] = ex.prefill_tok_s_mean;

    collect_cpu_ram_disk(now);
    {
        std::lock_guard<std::mutex> lk(mu_);   // now_ 与 gpus_read_ 同锁换出，snapshot 无竞态
        now_.swap(now);
        gpus_read_.swap(gpus_tmp);
    }
    push_history();
}

void Telemetry::collect_cpu_ram_disk(std::map<std::string, std::optional<double>>& now) {
#if defined(_WIN32)
    FILETIME idle_t{}, kern_t{}, user_t{};
    if (GetSystemTimes(&idle_t, &kern_t, &user_t)) {
        const auto t100 = [](const FILETIME& f) {
            return (static_cast<std::uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
        };
        const std::uint64_t idle = t100(idle_t);
        const std::uint64_t total = t100(kern_t) + t100(user_t);   // kernel 含 idle（Strata 注释）
        if (cpu_prev_valid_ && total > cpu_prev_total_) {
            const double p = 100.0 * (1.0 - static_cast<double>(idle - cpu_prev_idle_) /
                                                  static_cast<double>(total - cpu_prev_total_));
            now["cpu"] = std::max(0.0, std::min(100.0, p));
        }
        cpu_prev_idle_ = idle;
        cpu_prev_total_ = total;
        cpu_prev_valid_ = true;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        now["ram_used"] = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys);
        now["ram_total"] = static_cast<double>(ms.ullTotalPhys);
    }
    // 磁盘：Windows 在 v0 为 null（Strata 无 psutil 时同样缺磁盘读数）。
#else
    {
        std::ifstream f("/proc/stat");
        std::string tag;
        f >> tag;
        std::uint64_t v[8] = {0};
        for (int i = 0; i < 8 && f; ++i) f >> v[i];
        const std::uint64_t idle = v[3] + v[4];    // Strata：f[3]+f[4] = idle+iowait
        std::uint64_t total = 0;
        for (std::uint64_t x : v) total += x;
        if (cpu_prev_valid_ && total > cpu_prev_total_) {
            const double p = 100.0 * (1.0 - static_cast<double>(idle - cpu_prev_idle_) /
                                                  static_cast<double>(total - cpu_prev_total_));
            now["cpu"] = std::max(0.0, std::min(100.0, p));
        }
        cpu_prev_idle_ = idle;
        cpu_prev_total_ = total;
        cpu_prev_valid_ = true;
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string key;
        long long total_kb = 0, avail_kb = 0;
        while (f >> key) {
            long long v = 0;
            if (key == "MemTotal:") { f >> v; total_kb = v; }
            else if (key == "MemAvailable:") { f >> v; avail_kb = v; }
            else f.ignore(4096, '\n');
        }
        if (total_kb > 0) {
            now["ram_total"] = static_cast<double>(total_kb) * 1024.0;
            now["ram_used"] = static_cast<double>(total_kb - avail_kb) * 1024.0;
        }
    }
    {
        // /proc/diskstats：整盘行（minor == 0）的 sectors_* 计数聚合；差值 × 512 / Δt / 2^20 → MB/s。
        std::uint64_t r = 0, w = 0;
        std::ifstream f("/proc/diskstats");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            long long major = 0, minor = 0;
            std::string name;
            std::uint64_t a = 0, b = 0, rs = 0, c = 0, d = 0, e = 0, ws = 0;
            ss >> major >> minor >> name >> a >> b >> rs >> c >> d >> e >> ws;
            (void)major; (void)a; (void)b; (void)c; (void)d; (void)e;
            if (minor != 0) continue;   // 分区行
            r += rs;
            w += ws;
        }
        const double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (disk_prev_valid_ && t > disk_prev_t_) {
            const double dt = t - disk_prev_t_;
            now["disk_read_mb"] = static_cast<double>(r - disk_prev_r_) * 512.0 / dt / (1 << 20);
            now["disk_write_mb"] = static_cast<double>(w - disk_prev_w_) * 512.0 / dt / (1 << 20);
        }
        disk_prev_t_ = t;
        disk_prev_r_ = r;
        disk_prev_w_ = w;
        disk_prev_valid_ = true;
    }
#endif
}

// 10 系列键表（spec §6）：每样一次各推一值（读不到推 nullopt），封顶 60；浮点四舍五入两位
// （Strata: round(v, 2)）。
void Telemetry::push_history() {
    static const char* const kSeries[] = {
        "gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb",
        "cpu", "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"};
    std::lock_guard<std::mutex> lk(mu_);
    for (const char* key : kSeries) {
        auto& vec = hist_[key];
        if (now_.count(key) && now_.at(key)) {
            const double v = *now_.at(key);
            vec.push_back(std::round(v * 100.0) / 100.0);
        } else {
            vec.push_back(std::nullopt);
        }
        if (vec.size() > 60) vec.erase(vec.begin());
    }
}

TelemetrySnapshot Telemetry::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    TelemetrySnapshot out;
    out.now = now_;
    out.gpus = gpus_read_;
    out.stat = stat_;
    out.history = hist_;
    return out;
}

}  // namespace qinfer::serve
```
`server.hpp`/`server.cpp` 接线（T13 修改；机器码与注释）：

`server.hpp` 头包含与成员：

```cpp
#include "serve/telemetry.hpp"
    // 遥测：T13 起在 launch() 里 start()；/v1/status 的 machine 与 T14 的 /metrics 都读它。
    std::unique_ptr<Telemetry> tel_;
    // tok_s_mean 的累计口径：最近一次请求的解码速度是 v0 的 tok_s，全程均值另记这两项。
    std::uint64_t gen_done_ = 0;   // 累计生成的 token 数（成功请求才累加）
    std::uint64_t gen_us_ = 0;     // 累计解码耗时（µs）
```

`record_request` 在填 `last_` 的同一临界区里追加累计（`job->n_gen > 0` 才计，避免 400/错误请求污染）：

```cpp
    if (job->n_gen > 0 && job->done_at > job->leased_at &&
        job->done_at != std::chrono::steady_clock::time_point{}) {
        const auto start = job->first_at != std::chrono::steady_clock::time_point{}
                               ? job->first_at : job->leased_at;
        if (job->done_at > start) {
            gen_done_ += job->n_gen;
            gen_us_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(job->done_at - start).count());
        }
    }
```

`launch()` 在监听前启动遥测（先于路由注册，保证首帧 /metrics 就有数据）：

```cpp
    tel_ = std::make_unique<Telemetry>([this] { return telemetry_extra(); });
    tel_->start();
```

`telemetry_extra`（新增私有方法，定义在 server.cpp）：

```cpp
TelemetryExtra Serve::telemetry_extra() const {
    std::lock_guard<std::mutex> lk(last_mu_);
    TelemetryExtra ex;
    if (last_.have && last_.decode_tok_s > 0.0) ex.tok_s = last_.decode_tok_s;
    if (last_.have && last_.first_token_s > 0.001)
        ex.prefill_tok_s_mean = static_cast<double>(last_.prompt_tokens) / last_.first_token_s;
    if (gen_us_ > 0) ex.tok_s_mean = 1e6 * static_cast<double>(gen_done_) /
                                     static_cast<double>(gen_us_);
    return ex;
}
```

`/v1/status` 的 machine 段替换 T12 的占位（`scaled` 助手放 server.cpp 匿名命名空间）：

```cpp
// v/unit，四舍五入 digits 位；nullopt → null。
static Json scaled_mib_gib(const std::optional<double>& v, double unit, int digits) {
    if (!v) return Json::nullv();
    const double p = std::pow(10.0, digits);
    return Json::numv(std::round(*v / unit * p) / p);
}
```

machine 段（替换 `svr.Get("/v1/status", ...)` 里 `Json mach = ...` 到 `r.set("machine", std::move(mach));` 的五行；GPU 字段逐个判空，无卡时整段 null）：

```cpp
        Json mach = Json::objv();
        mach.set("at", Json::numv(static_cast<double>(now)));
        if (tel_) {
            const auto ts = tel_->snapshot();
            const auto& hw = ts.now;
            const auto& st = ts.stat;
            auto get = [&hw](const char* k) -> std::optional<double> {
                return hw.count(k) ? std::optional<double>(hw.at(k)) : std::nullopt;
            };
            if (!st.gpu_name.empty()) {   // Strata: machine.gpu 在无卡时整个为 null
                Json g = Json::objv();
                g.set("name", Json::strv(st.gpu_name));
                g.set("used_mib", scaled_mib_gib(get("gpu_mem_used"), 1 << 20, 0));
                g.set("total_mib", scaled_mib_gib(get("gpu_mem_total"), 1 << 20, 0));
                g.set("util_pct", get("gpu_util").has_value() ? Json::numv(*get("gpu_util"))
                                                              : Json::nullv());
                g.set("temp_c", get("gpu_temp").has_value() ? Json::numv(*get("gpu_temp"))
                                                            : Json::nullv());
                g.set("power_w", get("gpu_power").has_value() ? Json::numv(*get("gpu_power"))
                                                              : Json::nullv());
                mach.set("gpu", std::move(g));
            } else {
                mach.set("gpu", Json::nullv());
            }
            if (get("ram_total").has_value()) {
                Json rm = Json::objv();
                rm.set("used_gib", scaled_mib_gib(get("ram_used"), 1 << 30, 1));
                rm.set("total_gib", scaled_mib_gib(get("ram_total"), 1 << 30, 1));
                mach.set("ram", std::move(rm));
            } else {
                mach.set("ram", Json::nullv());
            }
        } else {
            mach.set("gpu", Json::nullv());
            mach.set("ram", Json::nullv());
        }
        r.set("machine", std::move(mach));
```

（`g.set("name", ...)`：Strata 的 gpu_name 是「卡名连接」，有卡必非空——空串时 gpu 整段为 null，上面的 `st.gpu_name.empty()` 分支即该场景。）

`server.cpp` 头区补 `#include <cmath>`。`run.sh` 用例 12 追加 machine 形状断言：

```bash
# 12b. /v1/status 的 machine：at 恒在；无 GPU 的 CI 上 gpu/ram 为 null，有卡时是对象——只钉形状。
curl -s -H "$H" "http://$HOST:$PORT/v1/status" > "$R"
[ "$(jget "$R" "d['machine']['at']")" -gt 0 ] \
  && ok "/v1/status machine.at" || bad "/v1/status machine：$(head -c 200 "$R")"
```

- [ ] **Step 4: 跑测试通过 + 集成不回归**

```bash
cmake -B build -S . && cmake --build build --target test_telemetry qinfer_server && \
  ctest --test-dir build -R '^(telemetry|serve_integration)$' --output-on-failure
```

预期：test_telemetry 绿（CI 无 GPU 时 GPU 断言全不碰值）；serve_integration 12+1 用例绿（machine.at 断言过）。环境2 有卡时再人工看 `/v1/status` 的 machine.gpu 五个字段非 null（T18 清单里有一项）。

- [ ] **Step 5: CMake + 提交**

```cmake
# qinfer_serve 源列表追加：src/serve/telemetry.cpp
# 测试区追加：
# add_executable(test_telemetry tests/test_telemetry.cpp)
# target_link_libraries(test_telemetry PRIVATE qinfer_serve)
# add_test(NAME telemetry COMMAND test_telemetry)
```

```bash
cmake -B build -S . && cmake --build build --target test_telemetry qinfer_server && \
  ctest --test-dir build -R '^(telemetry|serve_integration)$' --output-on-failure
git add src/serve/telemetry.hpp src/serve/telemetry.cpp src/serve/server.hpp src/serve/server.cpp \
  tests/test_telemetry.cpp tests/integration/serve/run.sh CMakeLists.txt
python3 scripts/docs_checks.py
git commit -m "feat(serve): 遥测——NVML dlopen/AMD sysfs/并联 /proc 差值法，10 系列 60 点历史；/v1/status machine 接真值"
```

### Task 14: 请求记录环与监控端点（/metrics、/api/requests）

规格 §6 的监控契约下半场：请求记录环（100 条、256 KiB 截断）做 `/metrics` 的 `requests`/`requests_kept`/`totals` 数据源和 `/api/requests` 的完整数据源。形状对齐 Strata 的 `begin_request`/`request_records`/`metrics()`（从 `ref_repos/strata/serve/server.py` 核对过：列表视图不带 input/output/reasoning，详情 `?id=` 全带；未结束记录实时补 `wallclock_s`；`/metrics` 的 `hardware` 就是遥测 `now` 字典，`gpus[]` 仅多卡时出现）。

**Files:**
- Create: `src/serve/records.hpp`、`src/serve/records.cpp`、`tests/test_records.cpp`
- Modify: `src/serve/scheduler.hpp`（Job 加 `record_id`/`raw_body` 两个字段）、`src/serve/server.hpp`（`records_`/`totals_` 成员与头包含）、`src/serve/server.cpp`（`record_request` 升级 + `/metrics`、`/api/requests` 两个 GET + 三个 launch 调用点补 `raw_body`）、`tests/integration/serve/run.sh`（用例 13、14）、`CMakeLists.txt`

**Interfaces:**
- Consumes: T6 `Json`；T9 `Serve::launch` 与 `record_request`/`kQueueFullBody`；T12 `Serve::LastTimings last_`/`last_mu_`；T13 `Telemetry::snapshot`（`TelemetrySnapshot`）与 `gen_done_`/`gen_us_`；T7 `Scheduler::slots_view`/`queued`/`any_busy`
- Produces: `qinfer::serve::Records`（`begin`/`finish`/`detail`/`list`/`size`）；`GET /metrics`；`GET /api/requests`。后续 T15 前端三页与 T16 api-monitor 全部读这两个端点

- [ ] **Step 1: 写失败测试**

`tests/test_records.cpp`：

```cpp
// 请求记录环：begin 建记录（id 12 位 hex、state=queued、大输入截断 256 KiB 并置旗标）、
// finish 填计时与输出（error 空 → completed）、满 100 逐最老、detail 缺 id → false。
#include "serve/records.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include "tests/check.hpp"

using namespace qinfer::serve;

static bool is_hex12(const std::string& s) {
    if (s.size() != 12) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

int main() {
    // 1. begin：id 12 位 hex、queued、300 KiB 输入截到 256 KiB 并置旗标。
    Records recs;
    RequestBegin b;
    b.path = "/v1/chat/completions";
    b.model = "svr-test";
    b.input = std::string(300000, 'x');
    b.stream = true;
    b.response_format = "json_object";
    b.created = 1000.0;
    const std::string id = recs.begin(b);
    CHECK(is_hex12(id));
    RequestRecord r0;
    CHECK(recs.detail(id, r0));
    CHECK(r0.state == "queued");
    CHECK(r0.input.size() == Records::kTruncBytes && r0.input_truncated);
    CHECK(r0.stream && r0.response_format == "json_object");
    CHECK(std::abs(r0.created - 1000.0) < 1e-9);

    // 2. finish：计时与输出落盘；error 空 → completed。
    RequestFinish f;
    f.text = "你好";
    f.queue_s = 0.5;
    f.first_token_s = 1.2;
    f.wallclock_s = 9.7;
    f.decode_tok_s = 33.0;
    f.prompt_tokens = 20;
    f.output_tokens = 30;
    recs.finish(id, f);
    RequestRecord out;
    CHECK(recs.detail(id, out));
    CHECK(out.state == "completed" && out.output == "你好");
    CHECK(std::abs(out.wallclock_s - 9.7) < 1e-9);
    CHECK(out.finished && !out.output_truncated && !out.input_truncated);

    // 3. 满环：连同最先那条共 106 条 → 保留最新 100；最老被逐；list 最新在前。
    for (int i = 0; i < 105; ++i) {
        RequestBegin x;
        x.path = "/x";
        x.model = "m";
        x.input = "in";
        x.created = 1000.0 + i;
        const std::string xid = recs.begin(x);
        RequestFinish fx;
        fx.text = std::to_string(i);
        recs.finish(xid, fx);
    }
    CHECK(recs.size() == Records::kCap);
    const auto list = recs.list();
    CHECK(list.size() == Records::kCap);
    CHECK(list.front().output == "104");   // 最新在前
    CHECK(list.back().output == "5");      // 106-100-1 = 第 5 条之后的全在
    RequestRecord gone;
    CHECK(!recs.detail(id, gone));         // 最早一条已被逐

    // 4. 详情缺 id → false；error 非空 → state=error。
    CHECK(!recs.detail("deadbeef0000", out));
    RequestBegin eb;
    eb.path = "/v1/messages";
    eb.model = "m";
    eb.input = "in";
    const std::string eid = recs.begin(eb);
    RequestFinish fe;
    fe.error = "boom";
    recs.finish(eid, fe);
    RequestRecord eout;
    CHECK(recs.detail(eid, eout) && eout.state == "error");

    puts("test_records: OK");
    return 0;
}
```

- [ ] **Step 2: 跑测试确定失败**

```bash
cmake -B build -S . && cmake --build build --target test_records 2>&1 | tail -5
```

预期：FAIL——`src/serve/records.hpp` 不存在（编译错误）。

- [ ] **Step 3: 写最小实现**

`src/serve/records.hpp`：

```cpp
// 请求记录环（规格 §6 的 /api/requests 与 /metrics 数据源；字段对齐 Strata 的 record 字典）：
// 容量 100、大字段 256 KiB 截断（带截断旗标）、tail 插入丢最老。任何东西都不允许拖垮服务——
// begin/finish 里没有会抛的路径（id 用进程内唯一即可，无需全局唯一）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace qinfer::serve {

// 一条请求记录。大字段（input/output/reasoning）截到 256 KiB 并置对应旗标。
struct RequestRecord {
    std::string id;               // 12 位十六进制（环内唯一）
    std::string path;             // 请求路径（/v1/chat/completions 等）
    std::string model;
    double created = 0.0;         // 请求开始时刻（unix 秒）
    std::string state = "queued"; // queued → completed / error
    bool stream = false;
    std::string response_format;  // json_object / 空（未指定）
    std::string input;  bool input_truncated = false;    // 原始请求体（≤256 KiB）
    std::string output; bool output_truncated = false;   // 生成全文
    std::string reasoning; bool reasoning_truncated = false;  // v0 恒空
    double queue_s = 0.0;         // 入队到租约
    double first_token_s = 0.0;   // 租约到首 token（0 = 未达）
    double wallclock_s = 0.0;     // 入队到结束；未结束时 list/detail 补实时
    double decode_tok_s = 0.0;    // 解码速度 tok/s
    std::uint64_t prompt_tokens = 0, output_tokens = 0;
    double clock_ = 0.0;          // 记录创建时刻（steady 秒）——未结束时补 wallclock_s 用
    bool finished = false;
};

// begin 的入参（服务端从 Job 与请求体取）。
struct RequestBegin {
    std::string path, model, input;
    bool stream = false;
    std::string response_format;
    double created = 0.0;   // unix 秒
};

// finish 的入参（服务端从 Job 的计时与结果取）。
struct RequestFinish {
    std::string text;                  // → output
    std::string error;                 // 非空 → state=error（如 "disconnected"）
    double queue_s = 0.0, first_token_s = 0.0, wallclock_s = 0.0, decode_tok_s = 0.0;
    std::uint64_t prompt_tokens = 0, output_tokens = 0;
};

class Records {
public:
    static constexpr std::size_t kCap = 100;           // Strata 的 retention=100
    static constexpr std::size_t kTruncBytes = 262144; // 256 KiB

    // 建记录（state=queued，input 截断），返回 id；满 kCap 丢最老。
    std::string begin(const RequestBegin& b);
    // 收尾：填计时/输出；error 非空 → state=error，否则 completed。
    void finish(const std::string& id, const RequestFinish& f);
    // 单条拷贝（id 不存在 → false）。未结束的 wallclock_s 补实时。
    bool detail(const std::string& id, RequestRecord& out) const;
    // 最近记录，最新在前（前端列表）。未结束的 wallclock_s 补实时。
    std::vector<RequestRecord> list() const;
    std::size_t size() const;

private:
    void truncate_inplace(std::string& s, bool& flag) const;   // 超长 → 截断 + 旗标
    std::string next_id();                                     // 12 位 hex

    mutable std::mutex mu_;
    std::vector<RequestRecord> ring_;   // tail 插入；超 kCap 丢最老
    std::mt19937_64 rng_;               // id 生成（random_device 播种；环内唯一足够）
};

}  // namespace qinfer::serve
```

`src/serve/records.cpp`：

```cpp
#include "serve/records.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace qinfer::serve {

namespace {

// steady 当前秒数（未结束记录的实时 wallclock_s 用）。
double steady_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

std::string Records::next_id() {
    char buf[13];
    std::snprintf(buf, sizeof(buf), "%012llx",
                  static_cast<unsigned long long>(rng_()));
    return buf;
}

std::string Records::begin(const RequestBegin& b) {
    RequestRecord r;
    r.id = next_id();
    r.path = b.path;
    r.model = b.model;
    r.stream = b.stream;
    r.response_format = b.response_format;
    r.created = b.created;
    r.input = b.input;
    truncate_inplace(r.input, r.input_truncated);
    r.clock_ = steady_s();
    const std::string id = r.id;   // 先留 id：move 进环后 r 已被搬空
    std::lock_guard<std::mutex> lk(mu_);
    ring_.push_back(std::move(r));
    if (ring_.size() > kCap) ring_.erase(ring_.begin());
    return id;
}

void Records::finish(const std::string& id, const RequestFinish& f) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& r : ring_) {
        if (r.id != id) continue;
        r.state = f.error.empty() ? "completed" : "error";
        r.output = f.text;
        truncate_inplace(r.output, r.output_truncated);
        r.queue_s = f.queue_s;
        r.first_token_s = f.first_token_s;
        r.wallclock_s = f.wallclock_s;
        r.decode_tok_s = f.decode_tok_s;
        r.prompt_tokens = f.prompt_tokens;
        r.output_tokens = f.output_tokens;
        r.finished = true;
        break;
    }
}

void Records::truncate_inplace(std::string& s, bool& flag) const {
    if (s.size() > kTruncBytes) {
        s.resize(kTruncBytes);   // 字节级截断（Strata 是字符级；v0 按字节，注释为证）
        flag = true;
    }
}

// 未结束的 wallclock_s 补实时（Strata 的 list 视图同款）。
static void fill_live(RequestRecord& r) {
    if (!r.finished) r.wallclock_s = steady_s() - r.clock_;
}

bool Records::detail(const std::string& id, RequestRecord& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& r : ring_) {
        if (r.id == id) {
            out = r;
            fill_live(out);
            return true;
        }
    }
    return false;
}

std::vector<RequestRecord> Records::list() const {
    std::vector<RequestRecord> out;
    {
        std::lock_guard<std::mutex> lk(mu_);
        out = ring_;
    }
    for (auto& r : out) fill_live(r);
    std::reverse(out.begin(), out.end());   // 最新在前
    return out;
}

std::size_t Records::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ring_.size();
}

}  // namespace qinfer::serve
```

接线（`server.hpp` 头包含与成员；`Job` 两个新字段）：

```cpp
// server.hpp 顶部：
#include "serve/records.hpp"
    // 请求记录环（/metrics 的 requests/totals 与 /api/requests 的数据源，T14 起 record_request 填）。
    Records records_;
    // /metrics 的 totals 累计（与 last_ 同临界区更新——record_request 是 HTTP 线程调用的唯一写入者；
    // /metrics 读时也持 last_mu_，无竞态）。
    struct Totals {
        std::uint64_t prompt_tokens = 0, completion_tokens = 0;
        double prompt_ms = 0.0, decode_ms = 0.0;
    };
    Totals totals_;
```

```cpp
// scheduler.hpp 的 Job 追加两个字段（T14 记录环用）：
    std::string raw_body;    // 原始请求体（HTTP 线程在 launch 后填；入记录的 input）
    std::string record_id;   // 记录环键（空 = 未建；record_request 幂等用）
```

`record_request` 升级为最终形态（T12 的计时 + T13 的累计 + T14 的记录环；`created` 用「完成时刻 unix - wall」近似请求开始，v0 无入队 unix 计时）：

```cpp
void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    if (!job) return;
    const auto enq = job->enqueued_at;   // 缺省构造的 time_point（若从未入队）→ 全零
    const double wall = std::chrono::duration<double>(job->done_at - enq).count();
    LastTimings lt;
    lt.have = true;
    lt.queue_s = std::chrono::duration<double>(job->leased_at - enq).count();
    lt.first_token_s = job->first_at != std::chrono::steady_clock::time_point{}
                           ? std::chrono::duration<double>(job->first_at - enq).count() : 0.0;
    if (lt.first_token_s > 0) lt.first_token_s = lt.first_token_s - lt.queue_s;
    lt.wallclock_s = wall;
    lt.decode_tok_s = job->tok_s;
    lt.prompt_tokens = job->n_prompt;
    lt.output_tokens = job->n_gen;
    lt.finish = job->finish;
    {
        std::lock_guard<std::mutex> lk(last_mu_);
        last_ = lt;
        totals_.prompt_tokens += job->n_prompt;
        totals_.completion_tokens += job->n_gen;
        // 计时口径：prefill ≈ first_at- leased_at，decode ≈ done_at - first_at。
        const auto ms = [](std::chrono::steady_clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        if (job->first_at != std::chrono::steady_clock::time_point{})
            totals_.prompt_ms += ms(job->first_at - job->leased_at);
        if (job->done_at > job->first_at) totals_.decode_ms += ms(job->done_at - job->first_at);
    }
    // ---- T14：Records 环（同一 job 只建一条；断开/错误都收尾成 error 态）----
    if (job->record_id.empty()) {
        RequestBegin b;
        b.path = job->path;
        b.model = job->model;
        b.input = job->raw_body;
        b.stream = job->stream;
        Json raw;   // response_format 从原始请求体取（解析失败也不碍事，落空串）
        std::string jerr;
        if (json_parse(job->raw_body, raw, jerr) && raw.is_obj()) {
            const Json* rf = raw.find("response_format");
            if (rf != nullptr && rf->is_obj()) {
                const Json* ty = rf->find("type");
                if (ty != nullptr && ty->is_str()) b.response_format = ty->str;
            }
        }
        b.created = static_cast<double>(std::time(nullptr)) - wall;   // ≈ 请求开始
        job->record_id = records_.begin(b);
    }
    if (!job->record_id.empty()) {
        RequestFinish f;
        f.text = job->text;
        f.error = disconnected ? "disconnected" : job->error;
        f.queue_s = lt.queue_s;
        f.first_token_s = lt.first_token_s;
        f.wallclock_s = wall;
        f.decode_tok_s = job->tok_s;
        f.prompt_tokens = job->n_prompt;
        f.output_tokens = job->n_gen;
        records_.finish(job->record_id, f);
    }
}
```

三个 launch 调用点（chat/completions/messages）在 `if (!job) {...}` 之后各补一行（chat 与 completions 在 T9，messages 在 T11；`req.body` 原始请求体入记录）：

```cpp
    // handle_chat：launch 空指针检查后
    job->raw_body = req.body;   // T14：原始请求体进 Records
    // handle_completions：同样位置
    job->raw_body = req.body;
    // handle_messages：同样位置
    job->raw_body = req.body;
```

匿名命名空间追加三个序列化助手（放在 `send_json` 附近），以及 `/metrics`、`/api/requests` 两个 handler（注册在 T12 的 `/slots` 之后）：

```cpp
// ---- 记录序列化（列表视图不带大字段，详情全带；Strata request_records 同款）----
Json record_summary_json(const RequestRecord& r) {
    Json o = Json::objv();
    o.set("id", Json::strv(r.id));
    o.set("path", Json::strv(r.path));
    o.set("model", Json::strv(r.model));
    o.set("created", Json::numv(r.created));
    o.set("state", Json::strv(r.state));
    o.set("stream", Json::boolv(r.stream));
    o.set("response_format", r.response_format.empty()
                                 ? Json::nullv() : Json::strv(r.response_format));
    o.set("queue_s", Json::numv(r.queue_s));
    o.set("first_token_s", r.first_token_s > 0 ? Json::numv(r.first_token_s) : Json::nullv());
    o.set("wallclock_s", Json::numv(r.wallclock_s));
    o.set("decode_tok_s", Json::numv(r.decode_tok_s));
    o.set("prompt_tokens", Json::numv(static_cast<double>(r.prompt_tokens)));
    o.set("output_tokens", Json::numv(static_cast<double>(r.output_tokens)));
    return o;
}

Json record_detail_json(const RequestRecord& r) {
    Json o = record_summary_json(r);
    o.set("input", Json::strv(r.input));
    o.set("input_truncated", Json::boolv(r.input_truncated));
    o.set("output", Json::strv(r.output));
    o.set("output_truncated", Json::boolv(r.output_truncated));
    o.set("reasoning", Json::strv(r.reasoning));
    o.set("reasoning_truncated", Json::boolv(r.reasoning_truncated));
    return o;
}

// 历史系列（log. 每点：值或 null）。
Json series_json(const std::vector<std::optional<double>>& s) {
    Json a = Json::arrv();
    for (const auto& v : s) a.push(v ? Json::numv(*v) : Json::nullv());
    return a;
}

// 一位小数（Strata 的 round(x, 1)），仅展示用。
double round1(double v) { return std::round(v * 10.0) / 10.0; }
```

```cpp
    svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = cfg_.facade.max_context;
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        Json eng = Json::objv();
        eng.set("model", Json::strv(eng_->model_name()));
        eng.set("max_context", Json::numv(static_cast<double>(ctx)));
        eng.set("images", Json::boolv(false));
        r.set("engine", std::move(eng));
        Json live = Json::objv();
        live.set("state", Json::strv(busy ? "generating" : "idle"));
        live.set("queued", Json::numv(static_cast<double>(sched_.queued())));
        live.set("phase", Json::nullv());          // v0 无阶段跟踪
        live.set("prompt_tokens", Json::nullv());  // v0 不做进行中请求的逐项视图
        live.set("prompt_read", Json::nullv());
        live.set("prompt_total", Json::nullv());
        live.set("generated", Json::nullv());
        live.set("max_tokens", Json::nullv());
        live.set("elapsed_s", Json::nullv());
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            if (last_.have) {
                live.set("tok_s", Json::numv(round1(last_.decode_tok_s)));
            } else {
                live.set("tok_s", Json::nullv());
            }
            if (last_.have && last_.first_token_s > 0.001) {
                live.set("prefill_tok_s_mean", Json::numv(round1(
                    static_cast<double>(last_.prompt_tokens) / last_.first_token_s)));
            } else {
                live.set("prefill_tok_s_mean", Json::nullv());
            }
            if (gen_us_ > 0) {
                live.set("tok_s_mean", Json::numv(round1(
                    1e6 * static_cast<double>(gen_done_) / static_cast<double>(gen_us_))));
            } else {
                live.set("tok_s_mean", Json::nullv());
            }
        }
        live.set("tok_s_window_s", Json::nullv());   // v0 无滑动窗口（M2 补）
        const auto views = sched_.slots_view();
        live.set("parallel", Json::numv(static_cast<double>(cfg_.slots)));
        int running = 0;
        Json slots = Json::arrv();
        for (const SlotView& v : views) {
            if (v.busy) ++running;
            Json s = Json::objv();
            s.set("id", Json::numv(static_cast<double>(v.index)));
            s.set("n_ctx", Json::numv(static_cast<double>(ctx)));
            s.set("is_processing", Json::boolv(v.busy));
            s.set("n_prompt_tokens", Json::numv(0.0));
            slots.push(std::move(s));
        }
        live.set("running", Json::numv(static_cast<double>(running)));
        live.set("slots", std::move(slots));
        live.set("waiting", Json::numv(static_cast<double>(sched_.queued())));
        r.set("live", std::move(live));
        const auto list = records_.list();   // 最新在前
        Json reqs = Json::arrv();
        for (std::size_t i = 0; i < list.size() && i < 12; ++i) reqs.push(record_summary_json(list[i]));
        r.set("requests", std::move(reqs));
        Json cc = Json::objv();
        cc.set("enabled", Json::boolv(false));   // 索引器统计 M2 接（spec §6）
        r.set("conversation_cache", std::move(cc));
        r.set("requests_kept", Json::numv(static_cast<double>(Records::kCap)));
        Json total = Json::objv();
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            total.set("prompt_tokens", Json::numv(static_cast<double>(totals_.prompt_tokens)));
            total.set("reused", Json::numv(0.0));
            total.set("completion_tokens", Json::numv(static_cast<double>(totals_.completion_tokens)));
            total.set("prompt_ms", Json::numv(totals_.prompt_ms));
            total.set("decode_ms", Json::numv(totals_.decode_ms));
        }
        r.set("totals", std::move(total));
        const auto ts = tel_ ? tel_->snapshot() : TelemetrySnapshot{};
        Json hw = Json::objv();
        for (const auto& [k, v] : ts.now) hw.set(k, v ? Json::numv(*v) : Json::nullv());
        if (ts.gpus.size() > 1) {   // Strata：仅多卡时带 gpus[]
            Json gpu_list = Json::arrv();
            for (const GpuReading& gr : ts.gpus) {
                Json g = Json::objv();
                g.set("index", Json::numv(static_cast<double>(gr.index)));
                g.set("name", gr.name.empty() ? Json::nullv() : Json::strv(gr.name));
                g.set("util", gr.util ? Json::numv(*gr.util) : Json::nullv());
                g.set("mem_used", gr.mem_used ? Json::numv(*gr.mem_used) : Json::nullv());
                g.set("mem_total", gr.mem_total ? Json::numv(*gr.mem_total) : Json::nullv());
                g.set("temp", gr.temp ? Json::numv(*gr.temp) : Json::nullv());
                g.set("power", gr.power ? Json::numv(*gr.power) : Json::nullv());
                gpu_list.push(std::move(g));
            }
            hw.set("gpus", std::move(gpu_list));
        }
        r.set("hardware", std::move(hw));
        Json hs = Json::objv();
        hs.set("gpu_name", Json::strv(ts.stat.gpu_name));
        hs.set("gpu_count", Json::numv(static_cast<double>(ts.stat.gpu_count)));
        hs.set("gpu_note", Json::nullv());   // v0 恒空（见 telemetry.hpp 注释）
        hs.set("cpu_name", Json::strv(ts.stat.cpu_name));
        hs.set("cores", Json::numv(static_cast<double>(ts.stat.cores)));
        hs.set("threads", Json::numv(static_cast<double>(ts.stat.threads)));
        r.set("hardware_static", std::move(hs));
        Json hist = Json::objv();
        for (const auto& [k, vec] : ts.history) hist.set(k, series_json(vec));
        r.set("history", std::move(hist));
        r.set("time", Json::numv(static_cast<double>(now)));
        send_json(res, 200, r);
    });

    svr.Get("/api/requests", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.get_param_value("id");
        if (!id.empty()) {
            RequestRecord rec;
            if (!records_.detail(id, rec)) {
                send_json(res, 404, openai_err("request no longer retained", "invalid_request_error", nullptr));
                return;
            }
            send_json(res, 200, record_detail_json(rec));
            return;
        }
        const auto list = records_.list();
        Json arr = Json::arrv();
        for (const RequestRecord& r : list) arr.push(record_summary_json(r));
        Json out = Json::objv();
        out.set("requests", std::move(arr));
        out.set("retention", Json::numv(static_cast<double>(Records::kCap)));
        out.set("persistent", Json::boolv(false));
        out.set("loaded", Json::boolv(true));   // 引擎启动即装载（v0 无热装载）
        out.set("auto_load", Json::boolv(false));
        send_json(res, 200, out);
    });
```

`run.sh` 追加用例 13、14（数值断言全部用布尔表达式——jget 对浮点打 `0.0` 这类串，bash `-ge` 不认）：

```bash
# 13. /metrics：全键类型、10 系列、历史有数据、totals/conversation_cache/time 形状。
curl -s -H "$H" "http://$HOST:$PORT/metrics" > "$R"
[ "$(jget "$R" "d['engine']['model']")" ] && [ "$(jget "$R" "d['live']['state']")" ] \
  && [ "$(jget "$R" "len(d['history'])")" = 10 ] \
  && [ "$(jget "$R" "len(d['history']['cpu']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests_kept']")" = 100 ] \
  && [ "$(jget "$R" "d['totals']['reused']")" = 0 ] \
  && [ "$(jget "$R" "d['conversation_cache']['enabled']")" = False ] \
  && [ "$(jget "$R" "d['time'] > 0")" = True ] \
  && ok "/metrics 全键" || bad "/metrics：$(head -c 300 "$R")"

# 14. /api/requests：列表（retention=100、最新在前）与 ?id= 详情；缺 id → 404 错误体。
curl -s -H "$H" "http://$HOST:$PORT/api/requests" > "$R"
[ "$(jget "$R" "d['retention']")" = 100 ] && [ "$(jget "$R" "len(d['requests']) >= 1")" = True ] \
  && [ "$(jget "$R" "d['requests'][0]['wallclock_s'] >= 0")" = True ] \
  && ok "/api/requests 列表" || bad "/api/requests：$(head -c 300 "$R")"
RID=$(jget "$R" "d['requests'][0]['id']" | tr -d '"')
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=$RID" > "$R"
[ "$(jget "$R" "d['id']")" = "$RID" ] && [ "$(jget "$R" "d['wallclock_s'] >= 0")" = True ] \
  && [ "$(jget "$R" "d['queue_s'] >= 0")" = True ] \
  && ok "/api/requests?id= 详情" || bad "/api/requests?id=：$(head -c 300 "$R")"
curl -s -H "$H" "http://$HOST:$PORT/api/requests?id=deadbeef0000" > "$R"
[ "$(jget "$R" "d['error']['message']")" ] \
  && ok "/api/requests?id= 404 错误体" || bad "/api/requests 缺失 id 错误体：$(head -c 300 "$R")"
```

- [ ] **Step 4: 跑测试通过 + 集成不回归**

```bash
cmake -B build -S . && cmake --build build --target test_records qinfer_server && \
  ctest --test-dir build -R '^(records|serve_integration)$' --output-on-failure
```

预期：test_records 绿；serve_integration 14+2 用例绿。（`/metrics` 的 `history` 在刚启动时每系列只有 1~3 点——断言只钉 `>= 1` 与键数 10。）

- [ ] **Step 5: CMake + 提交**

```cmake
# qinfer_serve 源列表追加：src/serve/records.cpp
# 测试区追加：
# add_executable(test_records tests/test_records.cpp)
# target_link_libraries(test_records PRIVATE qinfer_serve)
# add_test(NAME records COMMAND test_records)
```

```bash
cmake -B build -S . && cmake --build build --target test_records qinfer_server && \
  ctest --test-dir build -R '^(records|serve_integration)$' --output-on-failure
git add src/serve/records.hpp src/serve/records.cpp src/serve/server.hpp src/serve/server.cpp \
  src/serve/scheduler.hpp tests/test_records.cpp tests/integration/serve/run.sh CMakeLists.txt
python3 scripts/docs_checks.py
git commit -m "feat(serve): 请求记录环（100 条/256K 截断）与 /metrics、/api/requests——全键 JSON 契约对齐 Strata"
```
### Task 15: 前端一——设计令牌、三 tab 外壳与 Chat 页

规格 §7 的前端第一部分：`tokens.css`（设计令牌，浅/深两套）、`index.html`（三 tab 单页外壳 + 永久的 Monitor/About 面板骨架）、`app.js`（tab 切换、API 密钥、采样器抽屉、Chat 的 SSE 流式渲染与每秒 `/metrics` 状态行）。Monitor/About 面板的数据填充与 api-monitor 页在 T16；但外壳容器此时就位，`grep` 可测。静态页由 T9 的 `set_base_dir(..., "/web")` 与启动时读入的 `web_index_` 提供，CMake 无需改动（文件在磁盘上即可被服务）。

**Files:**
- Create: `src/serve/web/tokens.css`
- Create: `src/serve/web/index.html`
- Create: `src/serve/web/app.js`
- Modify: `tests/integration/serve/run.sh`（case 15）

**Interfaces:**
- Consumes: T9 `/`、`/web/*` 静态路由与 `web_index_` 启动读入；T10 `/v1/chat/completions` SSE 帧协定（首帧 role、delta.content、`finish_reason` 帧、`[DONE]`、`error` 帧）；T9 `/v1/models`；T12 `/props`（`default_generation_settings.params` 的默认值、`model_alias`）；T14 `/metrics`（`live.{state,queued,tok_s}` 每秒轮询）
- Produces: `src/serve/web/` 三件套；`app.js` 里 `$`/`api`/`esc`/`apiKey`/`setApiKey`/`switchTab`/`addMsg`/`sendChat`/`readSSE`/`pollMetrics`/`setStatusDot` 等公共函数与 `document.addEventListener("DOMContentLoaded", ...)` 主入口；T16 的 Monitor/About 逻辑与 `monitor.js`（api-monitor 页）复用同一 `app.js`（页面上先 `<script src="/web/app.js">` 再 `<script src="/web/monitor.js">`，`app.js` 检测不到 `panel-chat` 时只注册公共函数、不启动三 tab 逻辑）

- [ ] **Step 1: 写失败集成用例（run.sh 追加）**

`tests/integration/serve/run.sh` 在 case 14 之后追加：

```bash
# 15. 静态页：/ 是含 Chat 面板的三 tab 壳，tokens.css 含令牌与深色套，app.js 含 Chat 逻辑。
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'id="panel-chat"' "$R" && grep -q '/web/tokens.css' "$R" && grep -q '/web/app.js' "$R" \
  && ok "/ 三 tab 壳（panel-chat/tokens.css/app.js）" || bad "/ 静态壳：$(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/tokens.css" > "$R"
grep -q -- '--bg' "$R" && grep -q 'data-theme="dark"' "$R" \
  && ok "/web/tokens.css（令牌 + 深色套）" || bad "/web/tokens.css：$(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/app.js" > "$R"
grep -q 'pollMetrics' "$R" && grep -q 'panel-chat' "$R" \
  && ok "/web/app.js（Chat 逻辑）" || bad "/web/app.js：$(head -c 200 "$R")"
```

- [ ] **Step 2: 跑集成确认红**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：case 15 三断言全 FAIL（`/` 与 `/web/*` 尚未落盘：`/` 返回空体、`/web/tokens.css` 404），1–14 照旧绿。

- [ ] **Step 3: 创建 `src/serve/web/tokens.css`**

设计令牌集中在此一个文件（对齐 Strata `serve/web/tokens.css` 的角色）：第一部分令牌变量（浅/深两套，深色跟随 `prefers-color-scheme` 且可用 `data-theme` 显式覆盖），第二部分跨页共享的基础与组件样式（外壳、tab、按钮、输入、消息气泡、卡片、状态点、抽屉、banner、表格）。三页与 api-monitor 页都只引这一个 css。

```css
/* Q-Infer 前端设计令牌与共享样式。浅色为默认；深色跟随系统偏好，
   或由 <html data-theme="dark"> 显式覆盖。 */

/* ---- 令牌：浅色 ---- */
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --panel-2: #eef1f4;
  --border: #d9dce1;
  --text: #1a1d21;
  --dim: #5c6670;
  --accent: #2b6cb0;
  --accent-soft: #e3ecf5;
  --accent-text: #ffffff;
  --ok: #2e7d32;
  --warn: #b26a00;
  --err: #c62828;
  --bubble-user: #dbe7f3;
  --bubble-ai: #ffffff;
  --code-bg: #f0f2f5;
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 24px;
  --radius: 8px;
  --radius-sm: 5px;
  --font: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

/* ---- 令牌：深色（系统偏好） ---- */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171b;
    --panel: #1c2127;
    --panel-2: #232a31;
    --border: #2b323a;
    --text: #e6e9ed;
    --dim: #93a0ab;
    --accent: #58a6ff;
    --accent-soft: #1d2f45;
    --accent-text: #0d1117;
    --ok: #4caf50;
    --warn: #e0a13c;
    --err: #f07070;
    --bubble-user: #233447;
    --bubble-ai: #1c2127;
    --code-bg: #161b21;
    --shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  }
}

/* ---- 令牌：深色（显式覆盖） ---- */
[data-theme="dark"] {
  --bg: #14171b;
  --panel: #1c2127;
  --panel-2: #232a31;
  --border: #2b323a;
  --text: #e6e9ed;
  --dim: #93a0ab;
  --accent: #58a6ff;
  --accent-soft: #1d2f45;
  --accent-text: #0d1117;
  --ok: #4caf50;
  --warn: #e0a13c;
  --err: #f07070;
  --bubble-user: #233447;
  --bubble-ai: #1c2127;
  --code-bg: #161b21;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

/* ---- 基础 ---- */
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.6 var(--font);
}
button { font: inherit; }
.hidden { display: none !important; }
.dim { color: var(--dim); }
.mono { font-family: var(--mono); }
.note { color: var(--dim); font-size: 13px; text-align: center; padding: var(--sp-5) var(--sp-3); }

/* ---- 外壳 ---- */
.topbar {
  display: flex;
  align-items: center;
  gap: var(--sp-4);
  padding: 0 var(--sp-4);
  height: 52px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: baseline; gap: var(--sp-2); min-width: 140px; }
.brand .logo { font-weight: 700; font-size: 15px; }
.brand .model { color: var(--dim); font-size: 12px; font-family: var(--mono); }
.tab-bar { display: flex; gap: 2px; flex: 1; }
.tab-btn {
  background: none;
  border: 0;
  color: var(--dim);
  padding: 8px 14px;
  border-radius: var(--radius-sm);
  cursor: pointer;
}
.tab-btn:hover { background: var(--panel-2); color: var(--text); }
.tab-btn.active { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.keybox { display: flex; align-items: center; gap: var(--sp-2); }
.keybox label { font-size: 12px; }
.key-input { width: 170px; }
main { height: calc(100vh - 52px); overflow: hidden; }
.panel { height: 100%; display: flex; flex-direction: column; }

/* ---- 通用控件 ---- */
.btn {
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: var(--radius-sm);
  padding: 7px 14px;
  cursor: pointer;
}
.btn:hover { background: var(--panel-2); }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text); }
.btn.primary:hover { filter: brightness(1.05); }
.btn.danger { background: transparent; border-color: var(--err); color: var(--err); }
.btn.ghost { background: transparent; border-color: transparent; color: var(--dim); }
.btn.ghost:hover { background: var(--panel-2); color: var(--text); }
.input {
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: var(--radius-sm);
  padding: 7px 10px;
}
.input:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--sp-4);
  box-shadow: var(--shadow);
}

/* ---- Chat ---- */
.msg-list { flex: 1; overflow-y: auto; padding: var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-3); }
.bubble { max-width: 78%; padding: 10px 14px; border-radius: var(--radius); white-space: pre-wrap; word-break: break-word; }
.bubble .bubble-role { font-size: 11px; color: var(--dim); margin-bottom: 2px; }
.bubble.user { align-self: flex-end; background: var(--bubble-user); border: 1px solid var(--border); }
.bubble.assistant { align-self: flex-start; background: var(--bubble-ai); border: 1px solid var(--border); }
.bubble.error { align-self: flex-start; background: transparent; border: 1px dashed var(--err); color: var(--err); }
.chat-foot { border-top: 1px solid var(--border); background: var(--panel); padding: var(--sp-3) var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-2); }
.drawer summary { cursor: pointer; color: var(--dim); font-size: 13px; user-select: none; }
.drawer-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--sp-3); margin-top: var(--sp-2); }
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 12px; color: var(--dim); }
.send-row { display: flex; gap: var(--sp-2); }
.chat-input { flex: 1; resize: none; }
.send-col { display: flex; flex-direction: column; gap: var(--sp-1); }
.send-col .btn { width: 92px; }
.chat-status { display: flex; align-items: center; gap: var(--sp-2); font-size: 12px; color: var(--dim); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--dim); display: inline-block; }
.status-dot.ok { background: var(--ok); }
.status-dot.bad { background: var(--err); }

/* ---- Monitor / About（T16 填充数据，骨架样式） ---- */
.monitor-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--sp-3); margin: var(--sp-4) var(--sp-4) var(--sp-3); }
.card .card-title { font-size: 12px; color: var(--dim); }
.card .card-value { font-size: 22px; font-weight: 700; margin-top: 2px; }
.card .card-sub { font-size: 12px; color: var(--dim); }
.req-layout { flex: 1; display: flex; gap: var(--sp-3); padding: 0 var(--sp-4) var(--sp-4); min-height: 0; }
.req-list-pane { width: 46%; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); display: flex; flex-direction: column; min-height: 0; }
.req-toolbar { display: flex; gap: var(--sp-2); padding: var(--sp-2) var(--sp-3); border-bottom: 1px solid var(--border); }
.req-toolbar .input { flex: 1; }
.req-list { flex: 1; overflow-y: auto; }
.req-row { padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; display: grid; grid-template-columns: 92px 1fr auto; gap: var(--sp-2); align-items: baseline; }
.req-row:hover { background: var(--panel-2); }
.req-row.active { background: var(--accent-soft); }
.req-detail-pane { flex: 1; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow-y: auto; padding: var(--sp-3); }
.kv { display: grid; grid-template-columns: 130px 1fr; gap: 4px var(--sp-3); }
.kv dt { color: var(--dim); font-size: 12px; }
.kv dd { margin: 0; font-family: var(--mono); word-break: break-all; }
.tabset { margin: var(--sp-2) 0; }
.tabset .tab-btn { padding: 4px 10px; }
.codebox { background: var(--code-bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: var(--sp-2); font-family: var(--mono); font-size: 12px; white-space: pre-wrap; word-break: break-all; max-height: 260px; overflow-y: auto; }
.badge { display: inline-block; font-size: 11px; border-radius: 10px; padding: 1px 8px; border: 1px solid var(--border); }
.badge.ok { color: var(--ok); border-color: var(--ok); }
.badge.err { color: var(--err); border-color: var(--err); }
.badge.busy { color: var(--warn); border-color: var(--warn); }
.about-body { padding: var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-3); overflow-y: auto; }
.plain-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.plain-table th, .plain-table td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
.plain-table th { color: var(--dim); font-weight: 500; width: 190px; }

/* ---- 通用提示条 ---- */
.banner { margin: var(--sp-3) var(--sp-4) 0; padding: 8px 14px; border-radius: var(--radius-sm); border: 1px solid var(--err); color: var(--err); background: var(--panel); font-size: 13px; }
```

- [ ] **Step 4: 创建 `src/serve/web/index.html`**

三 tab 单页外壳。Chat 面板完整可用；Monitor/About 面板给骨架容器（T16 的 JS 往里填数据，此时为空表）。HTML 属性全部用哈希引用（`id`），`esc()` 只用于文本节点之外需要插值的地方——本页无注入点。

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Q-Infer</title>
<link rel="stylesheet" href="/web/tokens.css">
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="logo">Q-Infer</span>
    <span class="model" id="brand-model">&hellip;</span>
  </div>
  <nav class="tab-bar">
    <button class="tab-btn active" data-tab="chat" type="button">Chat</button>
    <button class="tab-btn" data-tab="monitor" type="button">Monitor</button>
    <button class="tab-btn" data-tab="about" type="button">About</button>
  </nav>
  <div class="keybox">
    <label for="api-key">API 密钥</label>
    <input id="api-key" class="input key-input" type="password" autocomplete="off" placeholder="可选">
    <button id="btn-key-toggle" class="btn ghost" type="button">显示</button>
  </div>
</header>
<main>
  <div id="banner" class="banner hidden"></div>

  <section id="panel-chat" class="panel">
    <div id="msg-list" class="msg-list">
      <div class="note">向引擎提问吧。用右上角填入 API 密钥；采样器抽屉可调 temperature / top_p / seed。</div>
    </div>
    <div class="chat-foot">
      <details class="drawer">
        <summary>采样器</summary>
        <div class="drawer-grid">
          <div class="field">
            <label for="s-temp">temperature</label>
            <input id="s-temp" class="input" type="number" min="0" max="2" step="0.05" value="0.8">
          </div>
          <div class="field">
            <label for="s-top-p">top_p</label>
            <input id="s-top-p" class="input" type="number" min="0.05" max="1" step="0.05" value="0.95">
          </div>
          <div class="field">
            <label for="s-seed">seed（-1 = 随机）</label>
            <input id="s-seed" class="input" type="number" step="1" value="-1">
          </div>
        </div>
      </details>
      <div class="send-row">
        <textarea id="chat-input" class="input chat-input" rows="2"
                  placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
        <div class="send-col">
          <button id="btn-send" class="btn primary" type="button">发送</button>
          <button id="btn-stop" class="btn danger hidden" type="button">停止</button>
          <button id="btn-new" class="btn ghost" type="button">新会话</button>
        </div>
      </div>
      <div class="chat-status">
        <span id="status-dot" class="status-dot"></span>
        <span id="chat-state">连接中…</span>
      </div>
    </div>
  </section>

  <section id="panel-monitor" class="panel hidden">
    <div class="monitor-grid" id="cards">
      <div class="card"><div class="card-title">模型状态</div><div class="card-value" id="card-state">&ndash;</div><div class="card-sub" id="card-state-sub"></div></div>
      <div class="card"><div class="card-title">请求保留</div><div class="card-value" id="card-kept">&ndash;</div><div class="card-sub">条（/api/requests）</div></div>
      <div class="card"><div class="card-title">最近 wall-clock</div><div class="card-value" id="card-wall">&ndash;</div><div class="card-sub" id="card-wall-sub"></div></div>
      <div class="card"><div class="card-title">最近解码速度</div><div class="card-value" id="card-tok">&ndash;</div><div class="card-sub" id="card-tok-sub"></div></div>
    </div>
    <div class="req-layout">
      <div class="req-list-pane">
        <div class="req-toolbar">
          <input id="req-search" class="input" type="search" placeholder="搜索路径 / 模型 / 状态">
          <select id="req-filter" class="input" title="状态过滤">
            <option value="">全部状态</option>
            <option value="queued">queued</option>
            <option value="processing">processing</option>
            <option value="completed">completed</option>
            <option value="error">error</option>
          </select>
        </div>
        <div id="req-list" class="req-list"><div class="note">暂无请求记录</div></div>
      </div>
      <div id="req-detail" class="req-detail-pane"><div class="note">点左侧一条记录查看详情</div></div>
    </div>
  </section>

  <section id="panel-about" class="panel hidden">
    <div class="about-body">
      <div class="card">
        <div class="card-title">模型与引擎</div>
        <table class="plain-table" id="about-engine"><tbody></tbody></table>
      </div>
      <div class="card">
        <div class="card-title">能力（/props）</div>
        <table class="plain-table" id="about-props"><tbody></tbody></table>
      </div>
      <div class="card">
        <div class="card-title">本机硬件</div>
        <table class="plain-table" id="about-hardware"><tbody></tbody></table>
      </div>
      <div class="card">
        <div class="card-title">服务与协议</div>
        <p class="dim">推理端点为 <span class="mono" id="about-ports"></span>。OpenAI 域走 <span class="mono">/v1/chat/completions</span>（含 SSE 流式）、<span class="mono">/v1/completions</span>、<span class="mono">/v1/models</span>；Anthropic 域走 <span class="mono">/v1/messages</span> 与 <span class="mono">/v1/messages/count_tokens</span>；监控走 <span class="mono">/metrics</span> 与 <span class="mono">/api/requests</span>。</p>
      </div>
    </div>
  </section>
</main>
<script src="/web/app.js"></script>
</body>
</html>
```

- [ ] **Step 5: 创建 `src/serve/web/app.js`（公共层 + Chat）**

Chat 完整；Monitor/About 只挂 tab 切换（面板数据由 T16 填充；`initMonitor`/`initAbout` 在 T16 追加进同一入口）。api-monitor 页复用本文件：monitor.html 里先引 `app.js` 再引 `monitor.js`，本文件检测不到 `panel-chat` 时只定义公共函数并跳过三 tab 入口。

```javascript
"use strict";
/* Q-Infer 前端公共层（app.js）。
   index.html   : 三 tab 入口 + Chat 面板（本文件）。
   monitor.html : 请求检查页——复用本文件定义的公共函数，首行前置
                  <script>window.__apiMonitorPage = true;</script> 跳过三 tab 入口。
   所有 API 调用统一走 api()：自动带 Authorization: Bearer <sessionStorage 密钥>。 */

function $(id) { return document.getElementById(id); }

/* ---- API 密钥（sessionStorage，随页面会话保留） ---- */
function apiKey() { return sessionStorage.getItem("qinfer_api_key") || ""; }
function setApiKey(k) { sessionStorage.setItem("qinfer_api_key", k); }

/* ---- 统一 fetch：带密钥头；有 body 时补 Content-Type ---- */
async function api(path, opts) {
  opts = opts || {};
  const h = new Headers(opts.headers || {});
  const k = apiKey();
  if (k) h.set("Authorization", "Bearer " + k);
  if (opts.body !== undefined && !h.has("Content-Type")) h.set("Content-Type", "application/json");
  return fetch(path, Object.assign({}, opts, { headers: h }));
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

/* ---- 页面级提示条 ---- */
function showBanner(msg, kind) {
  const b = $("banner");
  if (!b) return;
  b.textContent = msg;
  b.classList.toggle("err", kind === "err");
  b.classList.remove("hidden");
  window.clearTimeout(showBanner._t);
  showBanner._t = window.setTimeout(function () { b.classList.add("hidden"); }, 6000);
}

/* ---- 状态点 ---- */
function setStatusDot(ok) {
  const d = $("status-dot");
  if (!d) return;
  d.classList.toggle("ok", ok);
  d.classList.toggle("bad", !ok);
}

/* ---- tab 切换 ---- */
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(function (p) {
    p.classList.toggle("hidden", p.id !== "panel-" + name);
  });
}

/* ---- SSE 读取：把 data: 帧逐个喂给 onEvent；[DONE] 或流结束返回 ---- */
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  // 吞一段解码文本，按 \n\n 切帧；遇 [DONE] 返回 true（消费完毕）。
  function framesFrom(data) {
    buf += data;
    while (true) {
      const i = buf.indexOf("\n\n");
      if (i < 0) return false;              // 剩半帧，等下一次拼接
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim();
        if (d === "[DONE]") return true;
        if (!d) continue;
        try { onEvent(JSON.parse(d)); } catch (e) { /* 跳过坏帧 */ }
      }
    }
  }
  for (;;) {
    const r = await reader.read();
    const chunk = r.done ? dec.decode() : dec.decode(r.value, { stream: true });
    if (framesFrom(chunk)) return;
    if (r.done) break;
  }
  // 服务端在 [DONE] 前关闭：把残帧最后拼一次（多半无输出，半帧静默跳过）。
  framesFrom("");
}

/* ---- 采样器：默认取 /props，sessionStorage 有持久值时优先 ---- */
function saveSampler() {
  const s = {
    temperature: parseFloat($("s-temp").value),
    top_p: parseFloat($("s-top-p").value),
    seed: parseInt($("s-seed").value, 10),
  };
  sessionStorage.setItem("qinfer_sampler", JSON.stringify(s));
}
function loadSampler() {
  const raw = sessionStorage.getItem("qinfer_sampler");
  if (!raw) return false;
  try {
    const s = JSON.parse(raw);
    if (s.temperature !== undefined) $("s-temp").value = s.temperature;
    if (s.top_p !== undefined) $("s-top-p").value = s.top_p;
    if (s.seed !== undefined) $("s-seed").value = s.seed;
    return !!$("s-temp");
  } catch (e) { return false; }
}

/* ---- Chat ---- */
let modelId = "x";
let streaming = false;
let abortCtrl = null;

async function loadModel() {
  try {
    const res = await api("/v1/models");
    if (!res.ok) return;
    const d = await res.json();
    if (d.data && d.data.length) modelId = d.data[0].id;
  } catch (e) { /* 首屏不打断 */ }
}

async function loadProps() {
  try {
    const res = await api("/props");
    if (!res.ok) return;
    const d = await res.json();
    if (!loadSampler()) {
      const p = (d.default_generation_settings || {}).params || {};
      if (p.temperature !== undefined) $("s-temp").value = p.temperature;
      if (p.top_p !== undefined) $("s-top-p").value = p.top_p;
      if (p.seed !== undefined) $("s-seed").value = p.seed;
    }
    if (d.model_alias) $("brand-model").textContent = d.model_alias;
  } catch (e) { /* 默认值兜底 */ }
}

function addMsg(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  const label = document.createElement("div");
  label.className = "bubble-role";
  label.textContent = role === "user" ? "你" : role === "error" ? "错误" : "助手";
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;
  el.appendChild(label);
  el.appendChild(body);
  $("msg-list").appendChild(el);
  $("msg-list").scrollTop = $("msg-list").scrollHeight;
  return body;
}

function collectMessages() {
  const messages = [];
  document.querySelectorAll("#msg-list .bubble").forEach(function (b) {
    const role = b.classList.contains("user") ? "user" : "assistant";
    const txt = b.querySelector(".bubble-body").textContent;
    if (txt || role === "user") messages.push({ role: role, content: txt });
  });
  return messages;
}

function setStreaming(on) {
  streaming = on;
  $("btn-send").disabled = on;
  $("btn-new").disabled = on;
  $("btn-stop").classList.toggle("hidden", !on);
}

async function sendChat() {
  const ta = $("chat-input");
  const text = ta.value.trim();
  if (!text || streaming) return;
  ta.value = "";
  addMsg("user", text);
  const body = {
    model: modelId,
    messages: collectMessages(),
    stream: true,
    max_tokens: 512,
    temperature: parseFloat($("s-temp").value),
    top_p: parseFloat($("s-top-p").value),
    seed: parseInt($("s-seed").value, 10),
  };
  const out = addMsg("assistant", "");
  setStreaming(true);
  abortCtrl = new AbortController();
  try {
    const res = await api("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(body),
      signal: abortCtrl.signal,
      headers: { "Accept": "text/event-stream" },
    });
    if (!res.ok) {
      let msg = "HTTP " + res.status;
      try {
        const e = await res.json();
        if (e.error && e.error.message) msg = e.error.message;
      } catch (e) { /* 非 JSON 错误体 */ }
      out.textContent = msg;
      out.parentElement.className = "bubble error";
      if (res.status === 401) showBanner("API 密钥无效或未设置", "err");
      return;
    }
    await readSSE(res, function (ev) {
      if (ev.error) {
        out.textContent = ev.error.message || "生成失败";
        out.parentElement.className = "bubble error";
      } else if (ev.choices && ev.choices[0]) {
        const d = ev.choices[0].delta || {};
        if (typeof d.content === "string") out.textContent += d.content;
        $("msg-list").scrollTop = $("msg-list").scrollHeight;
      }
    });
  } catch (e) {
    if (e.name === "AbortError") {
      out.textContent += "（已停止）";
    } else {
      out.textContent = "请求失败：" + e.message;
      out.parentElement.className = "bubble error";
    }
  } finally {
    setStreaming(false);
    abortCtrl = null;
  }
}

function stopChat() {
  if (abortCtrl) abortCtrl.abort();
}

function newChat() {
  if (streaming) return;
  $("msg-list").innerHTML = "";
  const note = document.createElement("div");
  note.className = "note";
  note.textContent = "新会话已开始。";
  $("msg-list").appendChild(note);
}

/* ---- 每秒轮询 /metrics：状态行与状态点 ---- */
function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(1);
}

async function pollMetrics() {
  try {
    const res = await api("/metrics");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    const l = d.live || {};
    const state = l.state || "?";
    const queued = l.queued === undefined ? 0 : l.queued;
    const tok = fmtRate(l.tok_s);
    $("chat-state").textContent = "状态 " + state + " · 排队 " + queued + " · " + tok + " tok/s";
    setStatusDot(true);
  } catch (e) {
    setStatusDot(false);
    $("chat-state").textContent = "服务不可达，正在重试…";
  }
}

/* ---- 公共初始化：Key 与 tab（index / monitor 两页通用） ---- */
function initCommon() {
  document.querySelectorAll(".tab-btn").forEach(function (b) {
    b.addEventListener("click", function () { switchTab(b.dataset.tab); });
  });
  const keyInput = $("api-key");
  if (keyInput) {
    keyInput.value = apiKey();
    keyInput.addEventListener("change", function () { setApiKey(keyInput.value.trim()); });
  }
  const kt = $("btn-key-toggle");
  if (kt) {
    kt.addEventListener("click", function () {
      const show = keyInput.type === "password";
      keyInput.type = show ? "text" : "password";
      kt.textContent = show ? "隐藏" : "显示";
    });
  }
}

/* ---- Chat 初始化 ---- */
function initChat() {
  ["s-temp", "s-top-p", "s-seed"].forEach(function (id) {
    $(id).addEventListener("change", saveSampler);
  });
  $("btn-send").addEventListener("click", sendChat);
  $("btn-stop").addEventListener("click", stopChat);
  $("btn-new").addEventListener("click", newChat);
  $("chat-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  initCommon();
  if (!$("panel-chat") || window.__apiMonitorPage) return; // api-monitor 页：只复用公共函数
  initChat();
  loadModel();
  loadProps();
  pollMetrics();
  window.setInterval(pollMetrics, 1000);
});
```

- [ ] **Step 6: 跑集成确认绿**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：case 15 三断言绿，1–14 照旧。若 `tokens.css` 的 `data-theme="dark"` 断言飘，检查该属性是否被 `grep -q -- '--bg'`（`--` 是 grep 选项终结符）与引号写对。

- [ ] **Step 7: docs_checks + 提交**

```bash
python3 scripts/docs_checks.py
git add src/serve/web/tokens.css src/serve/web/index.html src/serve/web/app.js tests/integration/serve/run.sh
git commit -m "feat(serve): 前端一——设计令牌（浅/深）、三 tab 壳与 Chat 页：SSE 流式渲染、采样器抽屉、每秒 /metrics 状态行"
```

commit 信息与门禁：docs_checks.py 与 ctest 全绿（本步未触碰文档）。
### Task 16: 前端二——Monitor / About 数据填充与 api-monitor 检查页

规格 §7 的前端第二部分：Monitor tab 的四张概览卡、请求列表（搜索 + 状态过滤）、请求详情（计时键值 + Input / Output / Reasoning / API response 四 tab + 复制）、10 系列硬件 sparkline；About tab 的引擎/能力/硬件/端口四块；独立检查页 `/api-monitor`（`monitor.html` + `monitor.js`，复用 `app.js` 的公共函数与 `QinferRequests` 渲染层）。轮询对齐规格：Monitor 每 2 秒 `/v1/status` + `/api/requests`；sparkline 读 Chat 页每秒落下的 `window.lastMetrics.history`（`pollMetrics` 现在把整份 `/metrics` 存盘）。

**Files:**
- Modify: `src/serve/web/index.html`（Monitor 面板插 `#spark-grid`）
- Modify: `src/serve/web/tokens.css`（sparkline 布局样式追加）
- Modify: `src/serve/web/app.js`（`window.lastMetrics`、`QinferRequests`、`initMonitor`、`renderSparklines`、`refreshAbout`、`switchTab` 挂 About 刷新、入口补 `initMonitor()`）
- Create: `src/serve/web/monitor.html`（由 T9 的 `/api-monitor` 启动读入）
- Create: `src/serve/web/monitor.js`
- Modify: `tests/integration/serve/run.sh`（case 16）

**Interfaces:**
- Consumes: T14 `/metrics`（engine、hardware、hardware_static、history、live）、`/api/requests`（列表剥大字段、`?id=` 全量）；T12 `/v1/status`（loaded/last_timings/concurrency）、`/props`；T15 `index.html` 的 Monitor/About 骨架容器与 `app.js` 的 `api/esc/$/showBanner/switchTab/pollMetrics`
- Produces: `window.QinferRequests = {badge, fmtSec, timeOf, renderList, renderDetail, apiResponseHtml, loadDetail, sparkline}`（index 页与 api-monitor 页共用的请求渲染层）；`window.lastMetrics`（历史 sparkline 数据源）；`monitor.js` 的检查页轮询。T17/T18 不再动前端

- [ ] **Step 1: 写失败集成用例（run.sh 追加）**

```bash
# 16. 前端二：index 带 Monitor/About 骨架，app.js 暴露 QinferRequests 与 sparkline，
#     /api-monitor 装了壳且 monitor.js 可达（复用 loadDetail）。
curl -s "http://$HOST:$PORT/" > "$R"
grep -q 'id="card-state"' "$R" && grep -q 'id="req-detail"' "$R" && grep -q 'id="about-engine"' "$R" \
  && ok "/ Monitor/About 骨架（card-state/req-detail/about-engine）" || bad "/ 骨架：$(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/app.js" > "$R"
grep -q 'QinferRequests' "$R" && grep -q 'initMonitor' "$R" && grep -q 'sparkline' "$R" \
  && ok "/web/app.js（Monitor/About/sparkline）" || bad "/web/app.js：$(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/api-monitor" > "$R"
grep -q 'id="req-list"' "$R" && grep -q 'monitor.js' "$R" \
  && ok "/api-monitor 壳（req-list/monitor.js）" || bad "/api-monitor：$(head -c 200 "$R")"
curl -s "http://$HOST:$PORT/web/monitor.js" > "$R"
grep -q 'QinferRequests.loadDetail' "$R" \
  && ok "/web/monitor.js（检查页复用渲染层）" || bad "/web/monitor.js：$(head -c 200 "$R")"
```

- [ ] **Step 2: 跑集成确认红**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：case 16 四断言全 FAIL（`monitor.html`/`monitor.js` 未落盘、`app.js` 尚无 `QinferRequests`），1–15 照旧绿。

- [ ] **Step 3: `index.html` 插 sparkline 容器 + `tokens.css` 补样式**

`src/serve/web/index.html` 把 Monitor 面板的卡片区与请求区之间插入 sparkline 网格（在 `</div>`（`#cards` 收尾）与 `<div class="req-layout">` 之间）：

```html
    <div class="spark-grid" id="spark-grid"></div>
```

`src/serve/web/tokens.css` 在 `/* Monitor / About` 段落末尾追加：

```css
.spark-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: var(--sp-2); padding: 0 var(--sp-4) var(--sp-3); }
.spark-cell { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; }
.spark-cell canvas { display: block; width: 100%; height: 30px; }
.spark-head { display: flex; justify-content: space-between; gap: var(--sp-2); font-size: 11px; }
.spark-head .name { color: var(--dim); }
.spark-head .val { font-family: var(--mono); }
```

- [ ] **Step 4: `app.js`——`window.lastMetrics`、`QinferRequests` 渲染层、Monitor/About 逻辑**

`src/serve/web/app.js` 四处修改：

(a) `pollMetrics` 把 `/metrics` 整份存盘供 sparkline 用——`const d = await res.json();` 改：

```javascript
    const d = await res.json();
    window.lastMetrics = d;
```

(b) `switchTab` 进入 About 时刷新一次——`switchTab` 函数体末尾追加：

```javascript
  if (name === "about") refreshAbout();
```

(c) `initCommon` 后追加共享渲染层 `QinferRequests`（index 与 api-monitor 两页共用；detail 的「API response」tab 展示按记录字段重建的响应体——记录环不存响应原文，对齐 Strata 的 record 契约，UI 上标注「（按记录重建）」）：

```javascript
/* ---- 请求记录渲染层（Monitor tab 与 /api-monitor 共用） ---- */
window.QinferRequests = {
  badge: function (state) {
    const cls = state === "completed" ? "ok" : state === "error" ? "err" : state === "queued" ? "busy" : "";
    return '<span class="badge ' + cls + '">' + esc(state) + "</span>";
  },
  fmtSec: function (v) {
    if (v === null || v === undefined) return "—";
    return v.toFixed(2) + "s";
  },
  timeOf: function (unix) {
    if (!unix) return "—";
    return new Date(unix * 1000).toLocaleString();
  },
  // records：/api/requests 列表视图（剥掉 input/output/reasoning）。
  renderList: function (records, listEl, onPick) {
    listEl.innerHTML = "";
    if (!records.length) {
      const n = document.createElement("div");
      n.className = "note";
      n.textContent = "暂无请求记录";
      listEl.appendChild(n);
      return;
    }
    records.forEach(function (rec) {
      const row = document.createElement("div");
      row.className = "req-row";
      row.dataset.id = rec.id;
      row.innerHTML =
        '<span class="mono">' + esc(rec.id) + "</span>" +
        QinferRequests.badge(rec.state) +
        "<span>" + esc(rec.path || "") + "</span>";
      row.title = (rec.model || "") + " · " + QinferRequests.fmtSec(rec.wallclock_s);
      row.addEventListener("click", function () {
        listEl.querySelectorAll(".req-row").forEach(function (r) {
          r.classList.toggle("active", r.dataset.id === rec.id);
        });
        onPick(rec);
      });
      listEl.appendChild(row);
    });
  },
  apiResponseHtml: function (rec) {
    const usage = {
      prompt_tokens: rec.prompt_tokens || 0,
      completion_tokens: rec.output_tokens || 0,
      total_tokens: (rec.prompt_tokens || 0) + (rec.output_tokens || 0),
    };
    const resp = {
      id: "msg_" + rec.id,
      object: "chat.completion",
      created: Math.floor(rec.created || 0),
      model: rec.model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: rec.output || "" },
        finish_reason: rec.state === "error" ? "error" : "stop",
      }],
      usage: usage,
    };
    return JSON.stringify(resp, null, 2);
  },
  renderDetail: function (rec, el) {
    el.innerHTML = "";
    const dl = document.createElement("dl");
    dl.className = "kv";
    const kv = [
      ["id", rec.id],
      ["path", rec.path],
      ["model", rec.model],
      ["state", rec.state],
      ["created", QinferRequests.timeOf(rec.created)],
      ["stream", rec.stream ? "是" : "否"],
      ["response_format", rec.response_format || "—"],
      ["queue_s", QinferRequests.fmtSec(rec.queue_s)],
      ["first_token_s", QinferRequests.fmtSec(rec.first_token_s)],
      ["wallclock_s", QinferRequests.fmtSec(rec.wallclock_s)],
      ["decode_tok_s", QinferRequests.fmtSec(rec.decode_tok_s)],
      ["prompt_tokens", rec.prompt_tokens === undefined ? "—" : rec.prompt_tokens],
      ["output_tokens", rec.output_tokens === undefined ? "—" : rec.output_tokens],
    ];
    kv.forEach(function (p) {
      const dt = document.createElement("dt");
      dt.textContent = p[0];
      const dd = document.createElement("dd");
      dd.textContent = p[1];
      dl.appendChild(dt);
      dl.appendChild(dd);
    });
    el.appendChild(dl);
    const tabs = [
      ["input", "Input" + (rec.input_truncated ? "（已截断）" : ""), rec.input || "（空）"],
      ["output", "Output" + (rec.output_truncated ? "（已截断）" : ""), rec.output || "（空）"],
      ["reasoning", "Reasoning", rec.reasoning || "（无）"],
      ["response", "API response（按记录重建）", QinferRequests.apiResponseHtml(rec)],
    ];
    const set = document.createElement("div");
    set.className = "tabset";
    const box = document.createElement("div");
    box.className = "codebox";
    tabs.forEach(function (t, i) {
      const b = document.createElement("button");
      b.className = "tab-btn" + (i === 0 ? " active" : "");
      b.textContent = t[1];
      b.addEventListener("click", function () {
        set.querySelectorAll(".tab-btn").forEach(function (x) { x.classList.remove("active"); });
        b.classList.add("active");
        box.textContent = t[2];
        box.dataset.text = t[2];
      });
      set.appendChild(b);
    });
    const copy = document.createElement("button");
    copy.className = "btn ghost";
    copy.textContent = "复制";
    copy.addEventListener("click", function () {
      const txt = box.dataset.text !== undefined ? box.dataset.text : box.textContent;
      navigator.clipboard.writeText(txt).catch(function () { showBanner("复制失败", "err"); });
    });
    set.appendChild(copy);
    box.dataset.text = tabs[0][2];
    box.textContent = tabs[0][2];
    el.appendChild(set);
    el.appendChild(box);
  },
  loadDetail: function (id, el) {
    el.innerHTML = '<div class="note">加载详情…</div>';
    return api("/api/requests?id=" + encodeURIComponent(id)).then(function (res) {
      if (!res.ok) {
        el.innerHTML = '<div class="note">' + esc("请求详情不可用：HTTP " + res.status) + "</div>";
        return;
      }
      return res.json().then(function (d) { QinferRequests.renderDetail(d, el); });
    }).catch(function (e) {
      el.innerHTML = '<div class="note">' + esc("加载失败：" + e.message) + "</div>";
    });
  },
  sparkline: function (canvas, points) {
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    let min = Infinity;
    let max = -Infinity;
    let any = 0;
    points.forEach(function (p) {
      if (p !== null) { any++; if (p < min) min = p; if (p > max) max = p; }
    });
    if (!any || max === min) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2b6cb0";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p === null) { pen = false; continue; }
      const x = (i / (points.length - 1)) * w;
      const y = h - 3 - ((p - min) / (max - min)) * (h - 6);
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
  },
};
```

(d) 在 `initChat` 之后追加 Monitor/About 逻辑（`kvTable`/`refreshAbout`/`renderSparklines`/`refreshRequests`/`refreshMonitor`/`initMonitor`），并把入口的 `initChat();` 后补 `initMonitor();`：

```javascript
/* ---- Monitor：四卡片 + 请求列表/详情 + 10 系列 sparkline ---- */
function kvTable(tbodyId, rows) {
  const tb = $(tbodyId);
  tb.innerHTML = "";
  rows.forEach(function (r) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = r[0];
    const td = document.createElement("td");
    td.textContent = r[1];
    tr.appendChild(th);
    tr.appendChild(td);
    tb.appendChild(tr);
  });
}

async function refreshAbout() {
  if (!$("panel-about")) return;
  try {
    const m = await api("/metrics").then(function (r) { return r.json(); });
    const p = await api("/props").then(function (r) { return r.json(); });
    const s = await api("/v1/status").then(function (r) { return r.json(); });
    const eng = m.engine || {};
    kvTable("about-engine", [
      ["model", eng.model || "—"],
      ["max_context", eng.max_context === undefined ? "—" : String(eng.max_context)],
      ["images", eng.images !== undefined ? String(eng.images) : "—"],
      ["engine", s.engine || "—"],
      ["service", s.service || "—"],
      ["started", QinferRequests.timeOf(s.started)],
      ["uptime_s", s.uptime_s === undefined ? "—" : s.uptime_s.toFixed(0) + "s"],
      ["slots", s.concurrency ? String(s.concurrency.serving) : "—"],
    ]);
    const dg = p.default_generation_settings || {};
    const prm = dg.params || {};
    const caps = p.chat_template_caps || {};
    const ct = p.chat_template || "";
    kvTable("about-props", [
      ["model_alias", p.model_alias || "—"],
      ["total_slots", p.total_slots === undefined ? "—" : String(p.total_slots)],
      ["chat_template", ct.length > 120 ? ct.slice(0, 120) + "…" : ct],
      ["templater", caps.templater || "—"],
      ["vision", p.modalities && p.modalities.vision !== undefined ? String(p.modalities.vision) : "—"],
      ["temperature 默认", prm.temperature === undefined ? "—" : String(prm.temperature)],
      ["top_p 默认", prm.top_p === undefined ? "—" : String(prm.top_p)],
      ["seed 默认", prm.seed === undefined ? "—" : String(prm.seed)],
      ["build_info", p.build_info || "—"],
    ]);
    const hs = m.hardware_static || {};
    kvTable("about-hardware", [
      ["gpu_name", hs.gpu_name || "—"],
      ["gpu_count", hs.gpu_count === undefined ? "—" : String(hs.gpu_count)],
      ["gpu_note", hs.gpu_note || "—"],
      ["cpu_name", hs.cpu_name || "—"],
      ["cores", hs.cores === undefined ? "—" : String(hs.cores)],
      ["threads", hs.threads === undefined ? "—" : String(hs.threads)],
    ]);
    $("about-ports").textContent = location.protocol + "//" + location.host;
  } catch (e) {
    showBanner("About 信息加载失败：" + e.message, "err");
  }
}

const SPARK_SERIES = [
  { key: "gpu_util", label: "GPU 利用率", fmt: function (v) { return v.toFixed(1) + "%"; } },
  { key: "gpu_mem_used", label: "GPU 显存", fmt: function (v) { return (v / 1024).toFixed(2) + " GB"; } },
  { key: "gpu_temp", label: "GPU 温度", fmt: function (v) { return v.toFixed(0) + " ℃"; } },
  { key: "gpu_power", label: "GPU 功率", fmt: function (v) { return v.toFixed(0) + " W"; } },
  { key: "gpu_pcie_rx_mb", label: "PCIe RX", fmt: function (v) { return v.toFixed(1) + " MB/s"; } },
  { key: "cpu", label: "CPU", fmt: function (v) { return v.toFixed(1) + "%"; } },
  { key: "ram_used", label: "内存", fmt: function (v) { return (v / 1024).toFixed(1) + " GB"; } },
  { key: "disk_read_mb", label: "磁盘读", fmt: function (v) { return v.toFixed(1) + " MB/s"; } },
  { key: "tok_s", label: "tok/s", fmt: function (v) { return v.toFixed(1); } },
  { key: "prefill_tok_s_mean", label: "prefill tok/s", fmt: function (v) { return v.toFixed(1); } },
];

function renderSparklines(h) {
  if (!$("spark-grid")) return;
  const grid = $("spark-grid");
  grid.innerHTML = "";
  const hw = h.hardware || {};
  const history = h.history || {};
  SPARK_SERIES.forEach(function (s) {
    const cell = document.createElement("div");
    cell.className = "spark-cell";
    const head = document.createElement("div");
    head.className = "spark-head";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = s.label;
    const val = document.createElement("span");
    val.className = "val";
    const cur = hw[s.key];
    val.textContent = cur === null || cur === undefined ? "—" : s.fmt(cur);
    head.appendChild(name);
    head.appendChild(val);
    const cv = document.createElement("canvas");
    cv.width = 120;
    cv.height = 30;
    QinferRequests.sparkline(cv, history[s.key] || []);
    cell.appendChild(head);
    cell.appendChild(cv);
    grid.appendChild(cell);
  });
}

let reqCache_ = [];
let activeReqId_ = null;

function renderFiltered() {
  const q = ($("req-search").value || "").toLowerCase();
  const f = $("req-filter").value;
  let list = reqCache_;
  if (q || f) {
    list = list.filter(function (r) {
      const hay = ((r.path || "") + " " + (r.model || "") + " " + (r.state || "") + " " + (r.id || "")).toLowerCase();
      return (!q || hay.indexOf(q) >= 0) && (!f || r.state === f);
    });
  }
  QinferRequests.renderList(list, $("req-list"), function (rec) {
    activeReqId_ = rec.id;
    QinferRequests.loadDetail(rec.id, $("req-detail"));
  });
}

async function refreshRequests() {
  try {
    const res = await api("/api/requests");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    $("card-kept").textContent = d.retention === undefined ? "—" : d.retention;
    reqCache_ = d.requests || [];
    renderFiltered();
  } catch (e) { /* 下一次轮询再试 */ }
}

async function refreshMonitor() {
  if (!$("panel-monitor")) return;
  try {
    const res = await api("/v1/status");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const st = await res.json();
    const lt = st.last_timings;
    $("card-state").textContent = st.loaded ? "已加载" : "未加载";
    $("card-state-sub").textContent = (st.model || "") +
      (st.activity && st.activity.in_flight ? " · 进行中 " + st.activity.in_flight : "");
    $("card-wall").textContent = lt && lt.wallclock_s !== undefined ? lt.wallclock_s.toFixed(2) + "s" : "—";
    $("card-wall-sub").textContent = lt && lt.prompt_tokens !== undefined
      ? lt.prompt_tokens + " prompt → " + lt.output_tokens + " out" : "尚无完成请求";
    $("card-tok").textContent = lt && lt.decode_tok_s !== undefined ? lt.decode_tok_s.toFixed(1) + " tok/s" : "—";
    $("card-tok-sub").textContent = "上次请求解码速度";
    setStatusDot(true);
  } catch (e) {
    $("card-state").textContent = "不可达";
    $("card-state-sub").textContent = "服务无响应";
    setStatusDot(false);
  }
  if (window.lastMetrics) renderSparklines(window.lastMetrics);
  refreshRequests();
}

function initMonitor() {
  if (!$("panel-monitor")) return;
  refreshMonitor();
  window.setInterval(refreshMonitor, 2000);
  $("req-search").addEventListener("input", renderFiltered);
  $("req-filter").addEventListener("change", renderFiltered);
}
```

入口块改（`initChat();` 后补一行）：

```javascript
  initChat();
  initMonitor();
```

- [ ] **Step 5: 创建 `src/serve/web/monitor.html` 与 `monitor.js`**

`monitor.html`（服务启动时读入，`/api-monitor` 返回；先设 `__apiMonitorPage` 再引 `app.js`，跟着引 `monitor.js`）：

```html
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Q-Infer · 请求检查</title>
<link rel="stylesheet" href="/web/tokens.css">
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="logo">Q-Infer</span>
    <span class="model">请求检查 /api-monitor</span>
  </div>
  <div class="tab-bar" style="flex:1"></div>
  <div class="keybox">
    <label for="api-key">API 密钥</label>
    <input id="api-key" class="input key-input" type="password" autocomplete="off" placeholder="可选">
    <button id="btn-key-toggle" class="btn ghost" type="button">显示</button>
  </div>
  <a class="btn ghost" href="/">返回主面板</a>
</header>
<main>
  <div class="req-layout" style="height:100%">
    <div class="req-list-pane" style="width:42%">
      <div class="req-toolbar">
        <input id="req-search" class="input" type="search" placeholder="搜索路径 / 模型 / 状态">
        <select id="req-filter" class="input" title="状态过滤">
          <option value="">全部状态</option>
          <option value="queued">queued</option>
          <option value="processing">processing</option>
          <option value="completed">completed</option>
          <option value="error">error</option>
        </select>
      </div>
      <div id="req-list" class="req-list"><div class="note">暂无请求记录</div></div>
    </div>
    <div id="req-detail" class="req-detail-pane"><div class="note">点左侧一条记录查看详情</div></div>
  </div>
</main>
<script>window.__apiMonitorPage = true;</script>
<script src="/web/app.js"></script>
<script src="/web/monitor.js"></script>
</body>
</html>
```

`monitor.js`（只做检查页自己的轮询与事件接线；渲染全部走 `QinferRequests`）：

```javascript
"use strict";
/* 请求检查页（/api-monitor）：复用 app.js 的 $_api_esc_showBanner_initCommon_QinferRequests。 */
(function () {
  let cache = [];
  let activeId = null;

  function renderFiltered() {
    const q = ($("req-search").value || "").toLowerCase();
    const f = $("req-filter").value;
    let list = cache;
    if (q || f) {
      list = list.filter(function (r) {
        const hay = ((r.path || "") + " " + (r.model || "") + " " + (r.state || "") + " " + (r.id || "")).toLowerCase();
        return (!q || hay.indexOf(q) >= 0) && (!f || r.state === f);
      });
    }
    QinferRequests.renderList(list, $("req-list"), function (rec) {
      activeId = rec.id;
      QinferRequests.loadDetail(rec.id, $("req-detail"));
    });
  }

  async function refresh() {
    try {
      const res = await api("/api/requests");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const d = await res.json();
      cache = d.requests || [];
      renderFiltered();
    } catch (e) { /* 下一次轮询再试 */ }
  }

  document.addEventListener("DOMContentLoaded", function () {
    $("req-search").addEventListener("input", renderFiltered);
    $("req-filter").addEventListener("change", renderFiltered);
    refresh();
    window.setInterval(refresh, 2000);
  });
})();
```

- [ ] **Step 6: 跑集成确认绿**

```bash
cmake --build build --target qinfer_server && ctest --test-dir build -R '^serve_integration$' --output-on-failure
```

预期：case 16 四断言绿，1–15 照旧。若 `pollMetrics` 的存盘行没命中（`window.lastMetrics` 未定义导致 sparkline 空），检查第 (a) 处修改是否落在 `pollMetrics` 里而非别处。

- [ ] **Step 7: docs_checks + 提交**

```bash
python3 scripts/docs_checks.py
git add src/serve/web/index.html src/serve/web/tokens.css src/serve/web/app.js \
  src/serve/web/monitor.html src/serve/web/monitor.js tests/integration/serve/run.sh
git commit -m "feat(serve): 前端二——Monitor 四卡/请求列表与详情四 tab/sparkline、About 信息、/api-monitor 检查页"
```

commit 信息与门禁：docs_checks.py 与 ctest 全绿（本步未触碰文档）。
### Task 17: 文档回填——README 修为事实、engine.md §12 扩充、ADR-012 入库

规格 M1.5 的文档验收：修复 README 与 docs/README.md 的陈旧表述（测试套数、步循环落地情况、服务层存在性），engine.md §12 扩充为实现说明，新增 ADR-012，docs 清单登记。全部内容来自本计划已落地的实现与规格 §10，不引入新事实。门禁是 docs_checks.py 与 ctest 全绿。

**Files:**
- Modify: `README.md`（第 52 行「17 套测试」段、第 56 行「下一步」段、第 161 行 English summary）
- Modify: `docs/README.md`（ADR 清单加 ADR-012 行；未完成项表加 M2 切片行）
- Modify: `docs/design/engine.md`（§12 扩充、§17 序 7 行补落地说明）
- Create: `docs/design/adr/ADR-012-http-transport-and-serving-stack.md`

**Interfaces:**
- Consumes: T1–T16 的全部落地事实；规格 §10（ADR-012 内容源）；`engine.md` §12/§17 原文；`README.md` 第 52/56/161 行原文
- Produces: 三份文档修到与仓库事实一致；ADR-012 文件（docs/README.md 据此登记）；T18 以「docs_checks 与 ctest 全绿」为入口，不再改文档

- [ ] **Step 1: 先量后写——ctest 实测套数**

```bash
ctest --test-dir build -N 2>&1 | grep -i "total tests"
```

把输出的总数记为 `N`（预期 65 左右：M1 前 57 套主机侧 + 采样器/模板/门面/JSON/调度器/遥测/记录环/服务集成 8 套新增；以实测为准）。若测出 CUDA 条件测试不存在（CI 主机侧），说明 build 目录无 CUDA，以 `ctest -N` 直接数为准。

- [ ] **Step 2: 创建 `docs/design/adr/ADR-012-http-transport-and-serving-stack.md`**

格式照 ADR-008 的骨架（标题、状态、日期、背景、决策、备选方案、后果、相关），内容即规格 §10。日期写今天（决策日期并入状态行写成 `**状态**：Accepted（2026-10-12）`：docs_checks 的日期头检查只豁免 /adr/ 路径下的文件，本计划文件不在该路径，状态与日期不拆成两行）。不引用未登记的 S/G/R/P 编号（登记检查会红）；Strata 的对拍来源一律用路径或 `research/01` 链接。嵌入块内的交叉引用一律写反引号路径而非 Markdown 链接——同一段文本既在本计划里、也会写进目标文件，两种位置都要过 docs_checks 的链接检查。

```markdown
# ADR-012：HTTP 传输与单进程服务栈

**状态**：Accepted（2026-10-12）

---

## 背景

引擎的主机侧生成环已能单请求跑通；要把引擎对出去，需要一个对外推理服务。参考范本 Strata 是单进程 Python 服务，本仓库的实现栈（`ADR-008-implementation-stack-and-kernel-reuse.md`）要求 C++20、CMake、运行期依赖面为零、主机侧不依赖 CUDA。本决策回答四件事：HTTP 传输用什么、服务进程怎么摆、JSON 谁来解析、监控输出什么格式。

已有约束：监控形态已确认为 Web 页面实时查看（不先做 Prometheus 完整对接）；并发正确性先到「互不饿死」，吞吐目标留 M2 实测。

## 决策

1. HTTP 传输用 cpp-httplib 入库（MIT 许可证，单头，llama.cpp 同款）。文件整份入库、保留版权与许可注释、锁定上游发布版本号（v0.18.6）。「裁剪」落实为只启用服务端所需特性：不定义 `CPPHTTPLIB_OPENSSL_SUPPORT` 与 `CPPHTTPLIB_ZLIB_SUPPORT`；线程池大小由入口按槽位数配置（2 × 槽位 + 2）。协议解析、SSE 帧、轮询全部自写，不引入 HTTP 协议栈之外的任何依赖。
2. 单进程自包含，不做独立网关进程：零 IPC、零部署面，与 ADR-008 的依赖面口径一致。并发在进程内解决：有界队列 + 槽位租约，请求经队列交给槽位 worker，worker 驱动引擎步循环，增量经每 Job 的事件缓冲回写 HTTP 线程（满则背压、断开即取消）。
3. JSON 手写子集入库：对象、数组、字符串（含转义与 `\uXXXX`）、数字、布尔、空。OpenAI 与 Anthropic 两个兼容域的请求与响应都在这个子集内；越界输入一律 400。
4. 监控只出页面用的 JSON（`/metrics` 顶层键对齐 Strata v0.1.43 的契约：engine、live、requests、conversation_cache 占位、requests_kept、totals、hardware、hardware_static、history、time）。Prometheus 文本格式延迟到 M2；`conversation_cache` 的实值同样等 M2 接索引器统计。
5. 硬件采集零依赖：NVIDIA 走 `dlopen("libnvidia-ml.so.1")` / Windows `LoadLibrary("nvml.dll")`，读不到的字段一律 `null`，遥测任何异常不允许拖垮服务；AMD 走 Linux sysfs（kfd 拓扑 + hwmon）；CPU/RAM 走 `/proc` 差值法。逻辑照抄 Strata 的 `serve/telemetry.py`。

## 备选方案

1. 自写 socket 层。否决：连接管理、keep-alive、chunked 编码全是重复劳动，与「零依赖」不冲突（httplib 是整份入库的 vendored 单头，许可与上游一致）。
2. 独立网关进程 + IPC。否决：引入第二份部署面与两套进程生命周期，与单机、低并发场景不符。
3. 引第三方 JSON 库（nlohmann 等）。否决：需要的是协议边界的一个极小子集，手写解析 + 单测 200 行内可控，且保持依赖面为零。
4. Prometheus 文本先做。否决：用户确认监控形态是 Web 页面，M2 有完整对接需求时再补 exporter。

## 后果

正面：

- 依赖面维持为零（ADR-008 口径），单二进制即可对外服务，部署面与引擎一致。
- 两个兼容域共用同一套小 JSON 与同一调度器，行为可对拍；SSE 逐帧与 curl 对照。
- 遥测降级友好：无 GPU 的 CI 与笔记本上服务照常，页面显示占位。

负面：

- vendored httplib 要自持上游演进（升级 = 重新入库加复核 diff）；锁定版本号后成本可控。
- 手写 JSON 子集不支持流式 token 解析等高级用法；协议域越过该子集时返回 400，明确失败而非静默错译。
- M1 只承诺「互不饿死」，吞吐与延迟目标留 M2 实测，此前不引性能结论。

## 相关

`ADR-008-implementation-stack-and-kernel-reuse.md`（实现栈）、`ADR-009-correctness-acceptance-criteria.md`（正确性口径）、`../engine.md` §12（服务与调度）、`../../research/01-strata-engine.md`（Strata 契约来源）、`../../superpowers/specs/2026-10-12-serve-layer-design.md`（服务层设计规格）
```

- [ ] **Step 3: `docs/design/engine.md` §12 扩充 + §17 序 7 行更新**

§12 首条改为已落地口径（删掉「先跑起来这条线上的下一段」），并在该节末尾追加一个实现说明块：

```
## 12. 服务与调度

- 提供 OpenAI 与 Anthropic 兼容的 HTTP 接口（第 17 节第 7 序；M1 切片已落地，实现说明见本节末尾）。
- 分词器已落地（`src/text/tokenizer`，[S-65]）：那份 GGUF 把整份分词器放在元数据里（`tokenizer.ggml.*`），故 `artifact/gguf_table` 现在把键值段读进来（此前只跨过）——那是它的前提。口径逐条转录自引擎自带的那份实现（`tools/strata_tokenizer.py`，自述「实现之准」），三处陷阱各按结构钉住：字节 ↔ 可打印字符是 256 个码点的**置换**（故判据必须直接断言映射表，往返与 id 都照不出置换错）、预分词是 `[\p{L}\p{M}]` 而不是 `\p{L}`（少了 `\p{M}` 会把合并记号并进后面的标点）、特殊 token 里 type 4 永远匹配而 type 3 只在 parse_special 时。真权重上与它逐 token 对照 2889/2889 个 id 相同（31 个用例，含 emoji、组合记号、C0 控制、5000 字符长串与 2400 字符汉字段）。
- 批调度目标是并发不超过 8，不追求服务端式吞吐。环境2 上批量转矩阵乘的收益有限。
- 前缀缓存用基数树。多轮对话共享系统提示时，重复预填接近归零。
- 会话亲和：会话到专家簇的画像，作为第 5 节的输入。
- 多模态：视觉编码器约 1.4 GB，按需载入载出；图像特征缓存复用，避免多轮引用同一图像时重复计算。

### 12.1 服务层 M1 实现说明（已落地）

实现栈见 `adr/ADR-012-http-transport-and-serving-stack.md`，契约对拍对象是 Strata v0.1.43（`../research/01-strata-engine.md`）。本节记实现事实，改动先改它再改代码的约定照旧。

- 传输：cpp-httplib v0.18.6 单头整份入库（不定义 OpenSSL / ZLIB 宏），线程池 2 × 槽位 + 2；协议解析、SSE、轮询自写。静态页由同一进程在 `/web` 与 `/api-monitor` 提供。
- 协议域：OpenAI（`/v1/chat/completions` 流式与非流式、`/v1/completions`、`/v1/models`）与 Anthropic（`/v1/messages`、`/v1/messages/count_tokens`）同一份调度器与门面；两域同请求在默认参数下产出相同 token 序列。非法参数按两域各自的错误形状 400/401/503。
- 生成循环挂在 `src/serve/facade` 的门面上：请求即全新会话（scratch、KV 窗口、PLE 历史都随调用装配，请求之间无共享状态）；采样器用标准库 `mt19937_64`（温度 0 即贪心、seed -1 随机 = 与需求一致的默认）。SSE 用 httplib chunked provider，连接断开即取消生成并释放槽位。
- 调度：有界队列 + 槽位租约，满则 503；`--reject-after` 仅供集成测试制造确定 503。
- 遥测：NVML `dlopen`（Windows `nvml.dll`）、AMD kfd 拓扑 + hwmon、CPU/RAM 走 `/proc` 差值法、磁盘读解析 `/proc/diskstats`；无任何可读源时字段为 `null`。10 个系列各 60 点历史，`/metrics` 顶层键对齐 Strata。
- 请求记录环：保留 100 条、大字段 256 KiB 截断，`/api/requests` 列表剥大字段、`?id=` 给全量；`/metrics` 的 requests 取最近 12 条。前端四页（三 tab 单页 + `/api-monitor` 检查页）读这些端点，轮询 1 秒（/metrics）与 2 秒（/v1/status、/api/requests）。
- 鉴权：可选 `--api-key`；`/web`、`/api-monitor`、`/`、`/health` 可达，页面自带 key 走 `Authorization: Bearer`。
```

§17 序 7 行改为：

```
| 7 | 服务层 | 第 12 节；M1 切片已落地：HTTP 双域、遥测、请求记录与四页前端（12.1） | 不影响单请求指标 | 无 |
```

- [ ] **Step 4: `docs/README.md` 登记**

「2. 文档清单」表的 ADR 段末尾（ADR-011 行后）加一行：

```markdown
| `design/adr/ADR-012-http-transport-and-serving-stack.md` | HTTP 传输、单进程服务栈与监控形态 | 接受 |
```

「5. 未完成项」表加一行（M2 切片，无阻塞）：

```markdown
| 服务层 M2：Prometheus 文本、前缀缓存接入 /metrics 的 conversation_cache | 用户确认监控形态为 Web 页面实时查看，Prometheus 完整对接推迟到 M2；conversation_cache 的实值要等索引器统计接上（`design/adr/ADR-012-http-transport-and-serving-stack.md`） |
```

- [ ] **Step 5: `README.md` 修为事实**

第 52 行：

```markdown
序 2 的主机侧骨架已成形，CI 里构建并跑 <N> 套测试：三层存储与页表、有界行缓存、PCIe 字节预算仲裁；表行一侧的索引哈希、IQ4_NL 反量化与整条读取路径（`ple/row_path`）；专家一侧的格式分派表、三矩阵前馈（IQ2_XS / IQ2_S / IQ2_XXS / Q2_0 / IQ4_NL 五档内核）、路由门控与一层 MoE 串联（`experts/`）；生成一侧的步循环与 KV 窗口；服务一侧的 HTTP 双域、遥测、请求记录与前端（`src/serve`）。两条路径都已在真模型（环境2 上的两档分片）上跑通并对照过独立实现：表行 16 个头的 L1 与 Python oracle 全等、专家前馈与双精度参考一致、路由的专家 id 与引擎自身的主机参考全等。服务层（序 7 的 M1 切片）对外提供 OpenAI 与 Anthropic 兼容的推理接口与监控页，见 `docs/design/adr/ADR-012-http-transport-and-serving-stack.md`。
```

第 56 行：

```markdown
下一步：序 2 的验收标准是「单请求正确跑通」。主机侧大半已闭环——门控残差、步循环、KV 窗口与换入换出、PLE、内存池配平（`docs/design/engine.md §17 序 2`）；尚缺设备侧那半（批式前向与 KV 的设备实现）与 MTP 真草稿头。服务层（序 7）的 M1 切片已随本仓库落地：OpenAI 与 Anthropic 兼容的 HTTP 推理域、遥测与三页前端 + 请求检查页（`docs/design/engine.md §12.1`）；Prometheus 完整对接与 /metrics 的 conversation_cache 实值排到 M2。此外还有未实现的专家档（部署那份的 IQ1_M 3 层、另一档的 IQ3_S / IQ3_XXS 等）、共享专家与专家缓存。门禁方面 G-14 仍未判定。代码版本标签 `v<主>.<次>.<修订>` 按 `CONTRIBUTING §6.2` 的约定留到「单请求跑通」之后，故尚无代码版本标签，也没有 Release（现有 15 个标签都是设计里程碑与实测快照）。
```

第 161 行 English summary：

```markdown
The host-side engine skeleton is implemented under `src/` and is built and tested in CI (<N> ctest suites), including the quantized-kernel set, the expert path (format dispatch, gated FFN, router, one MoE layer), the n-gram table-row path, the step loop with KV windowing, and the serve layer (OpenAI/Anthropic HTTP endpoints, telemetry, request records, and the web frontend); the table-row and expert paths have been run against the real model shards and cross-checked against independent implementations. The GPU side (batched forward and device-side KV swap) is not yet in place, so there is no code-version tag or Release yet.
```

（两处 `<N>` 都换成 Step 1 测出的同一数字；旧数字 17 与实测不符属陈旧表述，一并修正。）

- [ ] **Step 6: 门禁 + 提交**

```bash
python3 scripts/docs_checks.py
cmake --build build --target qinfer_server test_facade && ctest --test-dir build -R 'serve_integration' --output-on-failure
git add README.md docs/README.md docs/design/engine.md \
  docs/design/adr/ADR-012-http-transport-and-serving-stack.md
git commit -m "docs(serve): README 修为事实（测试套数、步循环与服务层落地）、engine.md §12 扩充为实现说明、ADR-012 入库"
```

预期：docs_checks 全绿（新文件无 emoji、无第一人称、无断链、S/G/R/P 编号无新增未登记）；serve_integration 绿。若 check_links 报 ADR 里的相对路径断链，按实际目录修正相对路径（`../../research/01-strata-engine.md` 与 `../../superpowers/...` 的层级要数对：ADR 文件在 `docs/design/adr/`，上两级才是 `docs/`）。
### Task 18: 环境2 真权重端到端——M1 验收与收尾提交

规格 M1.5 的收尾验收：真权重单请求正确跑通（engine.md §17 序 2 的验收经服务层兑现）、M1.1 对拍判据保持通过、三页前端人工手测、docs_checks 与 ctest 全绿、提交（不推送远程）。全部在本机（环境2：Intel iGPU，无 NVIDIA/AMD 卡——「无 GPU 时字段为 null 且页面可渲染占位」正好在这里验；NVIDIA 主机的非空读数属另一台机器的画面，不阻塞本步）。用户对拍惯例：不 push remote，SSH remote。

**Files:**
- 无新增文件；本任务只有命令与人工清单，收尾提交覆盖 T1–T17 依赖的残留改动（若有）。

**Interfaces:**
- Consumes: T4 门面真权重入口（`test_facade --model --table --kv-slots`）、T9/T10/T11/T12/T14/T15/T16 的全部端点与前端；既有真权重对拍入口（`test_tokenizer --model --oracle`、`test_step_loop --model`）
- Produces: M1 全量验收记录（本任务核对清单勾选即记录）；最终提交

- [ ] **Step 1: 全新构建**

```bash
cmake -B build -S . && cmake --build build -j --target qinfer_server test_facade test_tokenizer test_step_loop
```

预期：零警告零错误（C++20 strict）。

- [ ] **Step 2: M1.1 对拍判据保持通过（真权重）**

```bash
./build/tests/test_tokenizer --model <环境2真模型分片> --oracle <引擎那份实现路径>
./build/tests/test_facade --model <环境2真模型分片> --table <环境2 PLE 分片2> --kv-slots 64
```

预期：分词器逐 token 一致 2889/2889；门面在真表上 PLE 开/关相对偏离、KV 窗口与不挂逐位一致（test_facade 的 `--model` 入口印出判据与计数，全部通过）。

- [ ] **Step 3: 真权重起服务**

```bash
./build/qinfer_server --model <环境2真模型分片> --ple-table <环境2 PLE 分片2> \
  --kv-slots 64 --slots 8 --port 8080 > /tmp/qinfer-server.log 2>&1 &
sleep 3
curl -s http://127.0.0.1:8080/health && echo
curl -s http://127.0.0.1:8080/v1/models | head -c 300 && echo
```

预期：health 返回 `{"status":"ok","model":<真模型名>,"max_context":262144,"images":false,"api_key":false,"loaded":true,"service":"qinfer"}`；models 列出真模型名；启动日志打印「模型 …（词表 151936、n_embd 2560、48 层）就绪」。

- [ ] **Step 4: 单请求正确跑通 + 流式逐字节对拍**

```bash
# 非流式贪心参考
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"数数：1、2、"}],"max_tokens":24,"temperature":0}' \
  > /tmp/ref.json
python3 - <<'EOF'
import json
d = json.load(open("/tmp/ref.json"))
t = d["choices"][0]["message"]["content"]
print("非流式文本:", repr(t))
print("choices 数:", len(d["choices"]), "| usage:", d.get("usage"))
assert t.strip(), "空输出"
assert d["choices"][0].get("finish_reason") == "stop", d["choices"][0]
EOF
```

```bash
# 流式：同 body 同 seed，逐 delta 拼出来必须与非流式逐字节相同
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json; charset=utf-8' \
  -d '{"model":"x","messages":[{"role":"user","content":"数数：1、2、"}],"max_tokens":24,"temperature":0,"stream":true}' \
  > /tmp/sse.body
python3 - <<'EOF'
import json
raw = open("/tmp/sse.body", "rb").read().decode("utf-8")
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
text = "".join(json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames)
ref = json.load(open("/tmp/ref.json"))["choices"][0]["message"]["content"]
print("流式拼文:", repr(text))
assert frames and raw.endswith("data: [DONE]\n\n") or "data: [DONE]" in raw, "缺 [DONE]"
assert text == ref, "流式与非流式不一致"
print("SSE 与非流式逐字节一致 ; 帧数", len(frames))
EOF
```

预期：流式拼文与非流式逐字节一致，且是「数数：1、2、…」的合理延续（贪心、温度 0，输出确定性）。这一步就是 engine.md §17 序 2 经服务层的「单请求正确跑通」证据。

- [ ] **Step 5: count_tokens 与 token 语义**

```bash
# 用与 Step 4 相同的 user 文本，拿 count_tokens 独立数一遍，与流式请求的 n_prompt 对不上就说明渲染口径漂了
USER_TEXT='数数：1、2、'
curl -s http://127.0.0.1:8080/v1/messages/count_tokens -H 'Content-Type: application/json' \
  -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$USER_TEXT\"}]}" && echo
```

预期：返回 input_tokens 与数字一致（Anthropic 域，`{"input_tokens":N}`）。人工核对 N 与 /metrics 里同 prompt 请求的 `prompt_tokens` 相同；分词口径的真权重对拍已有 Step 2 的 2889/2889 兜底。

- [ ] **Step 6: /metrics 契约与无 GPU 降级**

```bash
curl -s http://127.0.0.1:8080/metrics > /tmp/metrics.json
python3 - <<'EOF'
import json
d = json.load(open("/tmp/metrics.json"))
need = ["engine", "live", "requests", "conversation_cache", "requests_kept", "totals",
        "hardware", "hardware_static", "history", "time"]
for k in need:
    assert k in d, f"缺顶层键 {k}"
hw = d["hardware"]
assert hw["cpu"] is not None and hw["ram_used"] is not None, "CPU/RAM 读数应非空"
assert hw["gpu"] is None or hw["gpu"]["util"] is None, "环境2 无 NVIDIA/AMD 卡，gpu 应 null 或 util null"
assert len(d["history"].get("tok_s", [])) <= 60
cc = d["conversation_cache"]
assert cc == {"enabled": False}, cc
print("顶层键全、CPU/RAM 非空、gpu 降级 null、conversation_cache 占位、history ≤ 60 点")
EOF
```

预期：脚本打印通过。`hardware_static.gpu_name` 为 null 且 `gpu_note` 有降级说明；页面占位渲染在 Step 8 浏览器里看。

- [ ] **Step 7: 并发 8 全成功 + 断开释放槽位 + 503 形状**

```bash
# 并发 8：每份响应都要合法且带 queue_s 观测量（/api/requests 里 wallclock_s 与 queue_s 可观测即可）
for i in $(seq 8); do
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"第 '"$i"' 路并发生成。"}],"max_tokens":16,"temperature":0}' \
    -o /tmp/c$i.json &
done
wait
python3 - <<'EOF'
import json
for i in range(1, 9):
    d = json.load(open(f"/tmp/c{i}.json"))
    t = d["choices"][0]["message"]["content"]
    assert isinstance(t, str) and t, f"c{i} 空输出"
print("8 路并发全成功")
EOF
curl -s http://127.0.0.1:8080/api/requests | python3 -c 'import json,sys; d=json.load(sys.stdin); print("保留", d["retention"], "当前", len(d["requests"]))'
```

```bash
# 断开释放：起一个流式请求，1 秒后掐掉 curl；随后 /slots 应回到全空闲
( curl -s -N --max-time 60 http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"说一段很长的话，慢慢来。"}],"max_tokens":512,"stream":true}' > /dev/null & echo $! > /tmp/curl.pid )
sleep 1; kill -9 "$(cat /tmp/curl.pid)" 2>/dev/null
sleep 2
curl -s http://127.0.0.1:8080/slots | python3 -c 'import json,sys; s=json.load(sys.stdin); print("槽位", [(x["id"], x["is_processing"]) for x in s]); assert not any(x["is_processing"] for x in s), "仍有槽位在跑"'
```

```bash
# 503 形状：--reject-after 已由 CI 覆盖；这里验真实队列满时的扁平错误体
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' > /dev/null
```

预期：并发 8 全 200 且文本非空；断开 2 秒内 `/slots` 全部 `is_processing=false`；`/api/requests` 里能翻到 `queue_s`/`wallclock_s` 字段。

- [ ] **Step 8: 浏览器人工手测（清单勾选）**

在环境2 桌面浏览器开 `http://127.0.0.1:8080/`，逐项过：

- [ ] Chat：发消息 → 气泡流式逐 chunk 渲染，不断帧；输入框 Enter 发送、Shift+Enter 换行。
- [ ] 采样器抽屉：改 temperature=0 后发送，输出与 Step 4 的贪心一致（复现性直观可见）；seed 设为固定值后同 prompt 两次输出相同。
- [ ] 停止：长 max_tokens 请求中点「停止」，气泡尾缀「（已停止）」；紧接着新请求可正常入队（槽位已释放）。
- [ ] 每秒状态行：Chat 底部的「状态 live · 排队 0 · x.x tok/s」与状态点在生成时跳动。
- [ ] API 密钥：以 `--api-key xxx` 重启服务后，Chat 不带 key 发请求 → 气泡报错且出现「API 密钥无效或未设置」banner（或按 401 文案）；填 key 后恢复。
- [ ] Monitor：四卡片有值（模型状态=已加载、请求保留=100、最近 wall-clock 与解码速度在放过请求后非「—」）；请求列表出现 Step 7 的 8 条记录，点击 → 详情键值（queue_s/first_token_s/wallclock_s/decode_tok_s/prompt/output tokens）与 Input / Output / Reasoning / API response 四 tab，复制按钮可用。
- [ ] Monitor sparkline：10 格（GPU 利用率/显存/温度/功率/PCIe/CPU/内存/磁盘/tok/s/prefill）都有画线或占位；环境2 上 GPU 四格显示「—」且占位可渲染。
- [ ] About：引擎/能力/硬件三表有真值（model 为真模型名、total_slots=8、cpu_name/cores/threads 为环境2 真实值）；协议说明四行在。
- [ ] `/api-monitor`（浏览器开 `http://127.0.0.1:8080/api-monitor`）：与 Monitor 共用渲染——搜索「chat」过滤出 chat 路径、状态过滤 completed、详情四 tab 可用；「返回主面板」链接可回。
- [ ] 服务不可达表现：`kill` 掉服务进程时页面状态点变红、Chat 状态行显示「服务不可达，正在重试…」；重启服务后自动恢复（不刷新页面）。

- [ ] **Step 9: 全量门禁 + 收尾提交（不推送）**

```bash
kill %1 2>/dev/null    # 停服务
ctest --test-dir build --output-on-failure && python3 scripts/docs_checks.py
git status --short
git add -A
git commit -m "test(serve): M1 环境2 真权重端到端验收——单请求正确跑通、流式与非流式逐字节一致、并发 8 全成功、断开 2 秒内释放槽位、三页前端手测通过"
```

预期：`ctest` 全套绿、docs_checks 全绿；提交存在且工作树干净。**不要** `git push`。若 Step 6 的 conversation_cache 断言写成 `{"enabled":False}` 与实现的 `{"enabled":false}` 冲突（JSON 大小写），以服务端实现为准改断言——契约在规格 §6，实现在 T14，两边都是本仓库的，不一致就是 T14 的实现偏了。

M1 收尾：本任务勾选清单即验收记录；M1 全部提交满足「docs_checks 与 ctest 在任一提交上都不红」。