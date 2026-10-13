# M1 服务层设计

本文是服务层的执行规格：在 [engine.md](../../design/engine.md) §12（服务与调度）与 §17 序 2（骨架，验收为单请求跑通）之上，把引擎装配成可对外服务的形态。对外形状对齐 Strata v0.1.43（参考仓库 `ref_repos/strata`，检出于机器上，位于本仓库外）：路由集、`/metrics` JSON 契约、前端页面结构照抄其设计；推理只使用本仓库引擎。

## 1. 目标与边界

目标：启动一个 C++ 单进程服务，浏览器可访问三个页面（Chat、监控、关于），OpenAI 与 Anthropic 兼容客户端可完成一次真实推理，监控接口持续输出硬件读数与推理速度。这条线完成时，engine.md §17 序 2 的验收条件即达成。

对齐对象是 Strata v0.1.43 的对外形态，不是其实现：Strata 是 Python 标准库 `ThreadingHTTPServer` + 全局锁；本设计是 C++ 单进程，线程模型不同（第 4 节），对外契约相同。

范围外（后续里程碑，本文不展开）：

- 模型热装载与卸载（`/load`、`/unload`、`/v1/load`、`/v1/unload`、`/v1/vram`）：模型随进程装载，Strata 的这四个路由在 v0 返回 404。
- `/v1/responses` 存储式 API：Strata 自身也未实现（无状态 404），v0 同样 404。
- Prometheus 文本格式（Strata 的 `/metrics/prometheus`）：监控只做页面用的 JSON，留到 M2。
- `/settings` 与 `/config`、会话落盘（Strata 的会话保存）、`/mcp` 端点、多模态推理：M2/M3。
- 并发吞吐达标（engine.md §12 说并发不超过 8 就够）：v0 只保证正确性与多个请求互不饿死，吞吐优化属 M2。

## 2. 架构

单进程自包含，零新增运行时依赖（[ADR-008](../../design/adr/ADR-008-implementation-stack-and-kernel-reuse.md)）：

| 组件 | 职责 | 位置 |
|------|------|------|
| HTTP 传输 | 监听、路由、连接生命周期 | `third_party/httplib/httplib.h`（整份入库，锁定版本号；只启用服务端所需特性） |
| 协议层 | OpenAI / Anthropic 请求解析与响应序列化、SSE 帧 | `src/serve/protocol.hpp/.cpp` |
| JSON | 协议层的地基（手写子集） | `src/serve/json.hpp/.cpp` |
| 调度器 | 槽位租约、有界队列、计时观测 | `src/serve/scheduler.hpp/.cpp` |
| 装配门面 | GGUF 装载、分词、步循环装配、生成环 | `src/serve/facade.hpp/.cpp` |
| 采样器 | temperature / top_p / seed | `src/serve/sampler.hpp/.cpp` |
| 遥测 | NVML/sysfs/`/proc` 采集、60 点历史 | `src/serve/telemetry.hpp/.cpp` |
| 前端 | 三 tab 单页 + 请求检查页 | `src/serve/web/*` |
| 入口 | 配置、装配以上全部 | `src/serve/main.cpp` |

线程模型：httplib 自带线程池跑 HTTP；每个请求在 HTTP 线程内完成解析，随后交入有界队列；槽位 worker 从队列取请求、驱动引擎生成、把增量写进每槽事件缓冲；HTTP 线程从该缓冲读增量并写回响应（非流式则等完整结果）。遥测是独立线程，每秒采样一次。

关键选型的理由在第 10 节，随 M1.5 记入 ADR-012。

## 3. 路由表

v0 提供以下路由。判据：Strata v0.1.43 有的、与本引擎相关的、成本可控的，全部提供；其余明确 404。

### 鉴权

v0 支持可选 `--api-key`。设了之后，`/v1/*`、`/api/*`、`/metrics` 全部要求 `Authorization: Bearer <key>`，缺失或错误返回 401（对应域的 error 形状）；`/web` 与 `/api-monitor` 静态页可达，页面把 key 存 sessionStorage 并在请求时携带（对齐 Strata 的 monitor 页交互）。不设 key 则全部开放（本机默认）。

### OpenAI 兼容域

| 方法 | 路径 | 语义 |
|------|------|------|
| POST | `/v1/chat/completions` | 聊天补全；`stream:true` 走 SSE；支持 `messages`、`temperature`、`top_p`、`seed`、`max_tokens`、`stop`、`stream_options.include_usage` |
| POST | `/v1/completions` | 文本补全（Strata 无此路由，为 OpenAI 兼容面补充；实现成本约等于 chat 路由去掉消息组拼装） |
| GET | `/v1/models` | 模型清单，单元素数组 |
| GET | `/v1/status` | 引擎状态（第 6 节第 1 小节） |

### Anthropic 兼容域

| 方法 | 路径 | 语义 |
|------|------|------|
| POST | `/v1/messages` | 消息补全；`stream:true` 按 Anthropic 官方事件序列（`message_start` → `content_block_start` → `content_block_delta` … → `content_block_stop` → `message_delta` → `message_stop`） |
| POST | `/v1/messages/count_tokens` | 用本仓库分词器（[S-65]）数输入 token，返回 `{"input_tokens": N}` |

### 状态与监控域

| 方法 | 路径 | 语义 |
|------|------|------|
| GET | `/metrics` | 页面轮询的 JSON 契约（第 6 节），字段集对齐 Strata 1.43 |
| GET | `/api/requests` | 请求记录；`?id=` 取单条，否则最近 100 条（前端用） |
| GET | `/api-monitor` | 请求检查页（静态页本身） |
| GET | `/status` | 简版状态（对齐 Strata 的 `/status` 角色：外部客户端的状态探针，前端页面不使用） |
| GET | `/props` | 引擎属性：模型名、上下文上限、参数规模、精度档（对齐 Strata 的 `/props` 角色，供外部工具识别能力） |
| GET | `/slots` | 各槽位状态汇总（对齐 Strata 的 `/slots`） |
| GET | `/health` | 存活探针，恒 200 |

### 静态域

`/web/*` 映射到 `src/serve/web/`。其余未列路径一律 404；`POST /v1/responses`、`/load`、`/unload`、`/v1/load`、`/v1/unload`、`/v1/vram`、`/settings`、`/config`、`/mcp`、`/metrics/prometheus` 在 v0 返回 404 与明确错误体。

## 4. 并发与调度

Strata 的并发延迟来自全局锁与 GIL 以及 solo 路径的换槽机制。本设计按四条原则预留并发能力，全部从 v0 存在：

1. worker 与 HTTP 线程分离。HTTP 线程只做解析与读写，不持有引擎；有界队列容量为槽位数的两倍，满则立即返回 503（`{error: "queue full"}`），不阻塞 HTTP 线程。
2. 槽位租约从 v0 就有。槽位数默认 8（对齐 engine.md §12 的并发上限），可配置；每个槽位一个生成循环，同一时刻只有一个请求占用一个槽位；请求按队列序租约，租约即开始计 `queue_s`。
3. 每槽一个事件缓冲（有界环形），worker 写、HTTP 线程读；槽位状态用 atomic 状态字；运行中统计不经过共享可变全局。这是与 Strata 全局锁方案的根本差异，也是 M2 并发压测的地基。
4. 计时观测从 v0 记录：`queue_s`（入队到租约）、`first_token_s`（租约到首个增量）、`wallclock_s`（入队到结束）、滑动窗口与平均 tok/s（`tok_s`、`tok_s_mean`：最近 10 秒窗口与全程均值）。`load_s` 语义在 v0 指进程启动装载耗时，恒等于启动值；引擎未来支持热装载后再按 Strata 语义补齐。

客户端断开（含前端「停止」按钮的 AbortController）：SSE 写失败即中断该槽位生成。

## 5. 采样器与参数

采样器独立于引擎前向，位于装配门面的输出端：引擎产 logits，采样器产 token id。默认值对齐 llama.cpp 采样抽屉（Strata 同源）：`temperature` 0.8、`top_p` 0.95、`seed` 随机（-1 表示随机，给定非负 seed 结果可复现）；`temperature` 0 即贪心。

非法参数返回 400：OpenAI 域错误体为 `{"error": {"message", "type", "param"}}`，Anthropic 域为 `{"type":"error","error":{"type","message"}}`。`temperature` 为负、`top_p` 超出 (0,1]、`max_tokens` 非正、`stop` 为数组但含非字符串，均属此类。

采样器用标准库 `mt19937_64`，不接受外部 RNG 依赖（ADR-008 的依赖面为零在采样器上同样成立）。

## 6. 监控契约

`GET /metrics` 返回的 JSON 顶层键与 Strata 1.43 逐键一致：

| 键 | 内容 |
|----|------|
| `engine` | `{model, max_context, images(false), ...}` |
| `live` | `{state, queued, phase, prompt_tokens, prompt_read, prompt_total, generated, max_tokens, elapsed_s, tok_s, tok_s_mean, prefill_tok_s_mean, tok_s_window_s, parallel, running, slots, waiting}` |
| `requests` | 最近 12 条请求记录（详细走 `/api/requests`） |
| `conversation_cache` | 前缀缓存视图；已由服务层 token 前缀缓存填实（精确匹配 LRU：enabled/capacity/hits/misses/entries/tokens 实读，命中计入 totals.reused） |
| `requests_kept` | 保留条数（默认 100） |
| `totals` | `{prompt_tokens, reused(0), completion_tokens, prompt_ms, decode_ms}` 累计 |
| `hardware` | 每秒采样一次的最新读数 |
| `hardware_static` | `{gpu_name, gpu_count, gpu_note, cpu_name, cores, threads}` |
| `history` | 各系列最近 60 点（sparkline 数据） |
| `time` | 服务端时钟 |

`hardware` 与 `history` 的系列清单对齐 Strata 的 `serve/telemetry.py`：`gpu_util`、`gpu_mem_used`、`gpu_temp`、`gpu_power`、`gpu_pcie_rx_mb`、`cpu`、`ram_used`、`disk_read_mb`、`tok_s`、`prefill_tok_s_mean`；`hardware` 另含 `gpu_mem_total`、`gpu_power_limit`、`gpu_pcie_tx_mb` 与多卡时的 `gpus[]`（Strata 对多卡取总内存/总功率、平均负载、最高温度，照抄）。

硬件采集零依赖策略照抄 Strata 的 `serve/telemetry.py`：

- 读不到的任何字段都为 `null`，遥测任何异常都不允许拖垮服务。
- NVIDIA：`dlopen("libnvidia-ml.so.1")`（Windows 为 `LoadLibrary("nvml.dll")`），`dlsym` 取 `nvmlInit_v2`、`nvmlDeviceGetHandleByIndex_v2`、`nvmlDeviceGetName`、`nvmlDeviceGetUtilizationRates`、`nvmlDeviceGetMemoryInfo`、`nvmlDeviceGetTemperature`、`nvmlDeviceGetPowerUsage`、`nvmlDeviceGetEnforcedPowerLimit`、`nvmlDeviceGetCurrPcieLinkGeneration`、`nvmlDeviceGetMaxPcieLinkGeneration`、`nvmlDeviceGetCurrPcieLinkWidth`、`nvmlDeviceGetPcieThroughput`。
- AMD：Linux sysfs（`/sys/class/kfd/kfd/topology/nodes` 按 `gfx_target_version` 与 `simd_count` 过滤，`drm_render_minor` 连到 `renderD<minor>/device`，hwmon 读 `temp1_input`、`power1_average`、`power1_cap`），逻辑照抄 `telemetry.py` 的 `_Amd`。
- CPU 与 RAM：Linux 读 `/proc/stat` 差值法与 `/proc/meminfo`；Windows 用 `GlobalMemoryStatusEx` 与 `GetSystemTimes`。磁盘速率：Linux 解析 `/proc/diskstats` 按设备聚合；Windows 在 v0 为 `null`（对齐 Strata 无 psutil 时的行为）。

## 7. 前端

静态文件由同一进程提供，零构建、零外部网络依赖，语言为原生 HTML/CSS/JS，设计令牌集中在一个 `tokens.css`（对齐 Strata 的 `serve/web/tokens.css` 角色）。

页面清单：

- `/web/`（index.html）：三 tab 单页。
  - Chat：消息气泡、输入框、SSE 流式渲染、采样器抽屉（temperature / top_p / seed）、停止生成、新会话。
  - Monitor：模型状态点与 load 语义文字、四张概览卡（模型状态、请求保留数、最近 wall-clock、最近解码速度，对齐 Strata 的 monitor 页卡片）、请求列表（搜索 + 状态过滤）、请求详情（wall-clock / 队列 / 首 token / prompt / 输出 token / 解码速度 / response_format + Input / Output / Reasoning / API response 四 tab + 复制）、硬件 sparklines（GPU 利用率/显存/温度/功率、CPU、RAM、tok/s）。
  - About：模型与引擎信息（`engine` 与 `props`）、本机硬件（`hardware_static`）、服务端口与协议文档链接。
- `/web/api-monitor.html`：请求记录检查页（对应 Strata 的 `/api-monitor`），与 Monitor tab 共用请求渲染逻辑。

轮询节奏对齐 Strata：Chat 页每秒拉 `/metrics`，Monitor 页每 2 秒拉 `/v1/status` 与 `/api/requests`（硬件 sparkline 用同一 `/metrics` 的 `history`）。页面在服务不可达时显示断开状态并重试，不报死。

## 8. 分期

| 期 | 内容 | 验收 |
|----|------|------|
| M1.1 | 装配门面：把 `tests/test_step_loop.cpp` 的真权重装配上提为 `src/serve/facade`，补 tokenizer 的 encode/decode 封装与 GGUF `tokenizer.chat_template` 模板器；采样器 | 真权重对拍判据保持通过（分词 2889/2889、步循环逐段一致）；模板器单测覆盖主案例 GGUF 的模板原文 |
| M1.2 | `third_party/httplib` 入库；JSON 子集；`/v1/chat/completions` 非流式、`/v1/completions`、`/v1/models`、错误形状；鉴权（可选 `--api-key`）；调度器与槽位租约 | 集成测试：非流式请求返回正确文本；并发 8 请求全部成功且 `queue_s` 可观测；400/401/503 错误体正确 |
| M1.3 | SSE 流式；Anthropic 域（`/v1/messages`、`count_tokens`）；`/v1/status`、`/status`、`/health`、`/props`、`/slots` | 流式与 curl 逐 chunk 对照；`count_tokens` 与分词器逐 id 对照；两个兼容域同一请求产出相同 token 序列 |
| M1.4 | 遥测（NVML dlopen + 降级）；`/metrics`、`/api/requests`、`/api-monitor`；前端三页 | 集成测试断言 `/metrics` 全键类型；无 GPU 环境（CI）字段为 `null` 且页面可渲染占位；End-to-end 有 GPU 时读数非空 |
| M1.5 | 环境2 真权重端到端；修复 README 与 docs/README.md 的陈旧表述；engine.md §12 扩充为实现说明；新增 ADR-012；docs 清单登记 | engine.md §17 序 2 验收通过（单请求正确跑通）；`scripts/docs_checks.py` 全绿；`ctest` 全绿；提交（不推送远程） |

每期按测试驱动进行：先写失败的单测，再实现。HTTP 集成测试用脚本起服务 + `curl` 断言，不引入测试框架之外的依赖。

## 9. 测试与验证

单测：JSON 解析器（覆盖样例：嵌套、转义、Unicode 与错误输入）；采样器（给定 seed 可复现、贪心等价于 top-1、边界参数）；槽位状态机（租约、释放、队列满、断开释放）；环形事件缓冲（满环覆盖、并发读写）；模板器；协议解析（两个域的请求样本与错误样本）。

集成：`tests/integration/serve/` 下脚本 —— 起服务（合成 GGUF）、`curl` 断言非流式与流式响应、SSE 帧序列、`count_tokens`、`/metrics` 字段与历史长度、`/api/requests` 记录字段（含 `queue_s`/`first_token_s`/`wallclock_s`）、404/400/503 错误体、并发 8 请求、断开连接后槽位释放。

端到端：环境2 把真权重路径跑满 §8 的 M1.5 验收条目，浏览器手测三页与流式聊天。

门禁：`scripts/docs_checks.py` 与 `ctest` 都不允许在 M1 的任一提交上变红。

## 10. 决策记录（ADR-012 摘要）

随 M1.5 写入 `docs/design/adr/ADR-012-http-transport-and-serving-stack.md`，内容即本节：

- HTTP 传输用 cpp-httplib 入库（MIT 许可证，单头，llama.cpp 同款）。文件整份入库、保留版权与许可注释、锁定上游发布版本号（改造上游不是目的，维护一个 diff 才是成本）；「裁剪」落实为只启用服务端所需特性：不定义 `CPPHTTPLIB_OPENSSL_SUPPORT` 与 `CPPHTTPLIB_ZLIB_SUPPORT`，线程池大小由入口按槽位数配置。协议解析、SSE、轮询全部自写，不引入任何 HTTP 协议栈之外的依赖。
- 单进程自包含，不做独立网关进程：零 IPC、零部署面，与 ADR-008 一致。Strata 也是单进程（Python），本设计在进程内解决并发，而非分进程。
- JSON 手写子集入库：对象、数组、字符串（含转义与 `\uXXXX`）、数字、布尔、空。两个兼容域的请求与响应都在这个子集内；越界输入返回 400。
- 监控只出页面用的 JSON，Prometheus 文本延迟到 M2：用户已确认监控形态为 Web 页面实时查看，不先做完整对接。

## 11. 风险与未决

- G-14（混合精度与同字节缓存）未判定，与本线正交：M1 用固定的 FP8/4-bit 档（[ADR-006](../../design/adr/ADR-006-precision-policy-nvfp4-fp8-kv.md)），不依赖 G-14 结论。
- 模板器覆盖不了模型的完整 Jinja 语法时，M1.5 会在环境2 暴露：兜底是模板器按主案例 GGUF 的模板原文裁剪实现，遇到未支持的结构返回 500 并在错误体说明（不静默错译）。
- `tokenizer.chat_template` 若在元数据中缺失，服务拒绝启动并报明确错误（不以裸拼接冒充模板）。
- 并发正确性只到「互不饿死」：吞吐与延迟目标留 M2 实测（顺序即先用起来、再性能优化、再完善功能）。
- 前端与 `/metrics` 契约在 M1.4 前不再变动；任何契约变更先改本文再改代码。