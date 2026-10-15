# 文档地图

本文列出仓库内每份文档的受众、回答的问题与依赖关系。写作与编号约定见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

---

## 1. 分层

| 层 | 目录 | 受众 | 内容 |
|----|------|------|------|
| 外部 | [README.md](README.md)、[overview.md](overview.md) | 外部读者、评审者 | 定位、范围、架构概览 |
| 实现 | `docs/design/` | 实现者、测量者 | 规格、接口、门禁、风险、候选方向、决策 |
| 台账 | `docs/research/`、[glossary.md](glossary.md)、[hardware.md](hardware.md) | 研究者、全体 | 事实、术语、硬件规格、出处 |

## 2. 文档清单

| 文件 | 回答的问题 | 状态 |
|------|-----------|------|
| [README.md](../README.md) | 这是什么、适用范围、当前状态 | 完成 |
| [overview.md](overview.md) | 系统由哪几块组成、边界在哪 | 完成 |
| [requirements.md](requirements.md) | 目标负载、指标与阈值、验收场景、非目标 | 完成；阈值待实测填入 |
| [design/engine.md](design/engine.md) | 每块具体怎么做 | 完成；参数待标定 |
| [design/interfaces.md](design/interfaces.md) | 模块之间的接口契约 | 完成；实现细节待定 |
| [design/gates.md](design/gates.md) | 哪些假设必须先测、怎么测、通过标准 | 13 条已测、G-11 得部分、G-14 部分未判定；G-04 判否并已执行后果 |
| [design/risks.md](design/risks.md) | 已知风险与已确认的负结果 | 持续更新 |
| [design/proposals.md](design/proposals.md) | 未验证的候选方向 | 均为检索级存疑；P-03 已随 G-04 不通过而否决 |
| [`design/adr/ADR-001`](design/adr/ADR-001-tiered-storage-and-expert-cache.md) | 存储：分层还是全量常驻 | 接受 |
| [`design/adr/ADR-002`](design/adr/ADR-002-compute-unhit-experts-in-place.md) | 未命中专家：就地算还是搬回显存 | 接受 |
| [`design/adr/ADR-003`](design/adr/ADR-003-ngram-table-explicit-prefetch.md) | 表访问：显式预取还是交给页缓存 | 被 ADR-007 取代（G-04 不通过） |
| [`design/adr/ADR-004`](design/adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md) | 双卡：按层切分还是张量并行 | 接受 |
| [`design/adr/ADR-005`](design/adr/ADR-005-igpu-npu-as-coprocessor.md) | 核显与 NPU：是否参与专家计算 | 第一节接受；第二节（专家分担）由 [ADR-019](design/adr/ADR-019-igpu-not-for-experts.md) 判为不启用 |
| [`design/adr/ADR-006`](design/adr/ADR-006-precision-policy-nvfp4-fp8-kv.md) | 精度：权重、KV 与索引器的档位选择 | 接受 |
| [`design/adr/ADR-007`](design/adr/ADR-007-ngram-table-on-demand-read.md) | 表行访问：预取还是按需读 | 接受（取代 ADR-003） |
| [`design/adr/ADR-008`](design/adr/ADR-008-implementation-stack-and-kernel-reuse.md) | 实现栈、内核复用口径与许可 | 接受 |
| [`design/adr/ADR-009`](design/adr/ADR-009-correctness-acceptance-criteria.md) | 正确性验收：逐位还是分布级 | 接受 |
| [`design/adr/ADR-010`](design/adr/ADR-010-router-precision-declared-format.md) | 路由按张力声明的档读（bf16 那条契约只属引擎那份 pack） | 接受 |
| [`design/adr/ADR-011`](design/adr/ADR-011-compute-group-abstraction.md) | 后端以计算组为单位（设备、队列、内存池、能力表） | 接受 |
| `design/adr/ADR-012-http-transport-and-serving-stack.md` | HTTP 传输、单进程服务栈与监控形态 | 接受 |
| [`design/adr/ADR-014-gpu-igpu-serve.md`](design/adr/ADR-014-gpu-igpu-serve.md) | GPU/核显服务接入：分工求解器 + 阶段化迁移（A dense 上卡已交付） | 接受 |
| [`design/adr/ADR-016-vendored-strata-backend.md`](design/adr/ADR-016-vendored-strata-backend.md) | 计算后端：厂商化 Strata 引擎作后端（进程边界=依赖边界） | 接受 |
| [`design/adr/ADR-017-serve-engine-seam.md`](design/adr/ADR-017-serve-engine-seam.md) | 服务层引擎缝：`Backend` 接口与 `--backend strata|inhouse` | 接受 |
| [`design/adr/ADR-018-default-backend-strata.md`](design/adr/ADR-018-default-backend-strata.md) | 默认计算后端切到厂商化引擎（P2 验收后） | 接受 |
| [`design/adr/ADR-019-igpu-not-for-experts.md`](design/adr/ADR-019-igpu-not-for-experts.md) | 核显不参与未命中专家计算（P3a 证伪点读数：设备慢于 CPU 池） | 接受（关闭 ADR-005 第二节） |
| [`design/adr/ADR-020-strata-igpu-pool.md`](design/adr/ADR-020-strata-igpu-pool.md) | Strata iGPU 专家池：缝外独立池承接未命中专家，批 >4 走 Vulkan，opt-in | 接受 |
| [`design/adr/ADR-013-vision-port-and-dependency-boundary.md`](design/adr/ADR-013-vision-port-and-dependency-boundary.md) | 视觉编码器取材：移植 Strata tools/vision、llama.cpp 构建期物化、只 spawn 不链接 | 接受 |
| [glossary.md](glossary.md) | 术语的唯一口径 | 完成 |
| [hardware.md](hardware.md) | 两套测试环境的硬件规格 | 完成 |
| [research/01-strata-engine.md](research/01-strata-engine.md) | 参考范本引擎的原理与边界 | 完成 |
| [research/02-qwen3.8-flash-next.md](research/02-qwen3.8-flash-next.md) | 主案例模型的架构特点 | 完成 |
| [research/03-qwen4-trends.md](research/03-qwen4-trends.md) | 下一代模型的走向与约束 | 完成 |
| [research/04-acceleration-and-memory.md](research/04-acceleration-and-memory.md) | 可用的加速与显存优化技术 | 完成 |
| [research/05-engine-landscape.md](research/05-engine-landscape.md) | 现有引擎的能力与选型结论 | 完成 |
| [research/references.md](research/references.md) | 全部仓库外来源 | 持续更新 |

## 3. 阅读路径

新读者：[README.md](../README.md)，然后 [overview.md](overview.md)，然后 [requirements.md](requirements.md)。

评审者：[requirements.md](requirements.md)、[design/gates.md](design/gates.md)、[design/risks.md](design/risks.md)、[research/references.md](research/references.md)。这条路径给出验收条件、未验证的假设与出处。

实现者：[overview.md](overview.md)、[design/engine.md](design/engine.md)、[design/interfaces.md](design/interfaces.md)、[design/gates.md](design/gates.md)、`design/adr/`。

研究者：`research/01` 至 `research/05`、[glossary.md](glossary.md)、[design/proposals.md](design/proposals.md)。

## 4. 依赖方向

- `research/` 只陈述事实，不得引用 `design/`。
- `design/` 的关键论断必须能回指 `research/` 的某条事实或 [hardware.md](hardware.md) 的某个规格。
- 仓库外来源统一登记在 [research/references.md](research/references.md)，正文用 `[S-n]` 引用。
- 来源等级与复核状态的标记定义见 [research/references.md](research/references.md) 第 1、2 节。
- 门禁、风险、候选方向、决策分别使用 `G-NN`、`R-NN`、`P-NN`、`ADR-NNN` 编号，编号只增不改。

## 5. 未完成项

| 项 | 阻塞原因 |
|----|---------|
| 其余 `G-NN` 的实测记录（G-14 未判定） | 混合精度同字节缓存那条要等 P-02 实现；困惑度缺 logits 通路（原生 pack）；任务集与长文需外部评测 harness |
| 「本设计实现后」的对照实测 | 序 0 的参考基线已建（research/01 §10.3），但要拿到倍数指标还差引擎实现 |
| 序 2 的验收「单请求正确跑通」 | 主机侧两条路径（表行、专家）已跑通并对照过独立实现；还差稠密部分、KV 与注意力、顶层步循环。代码版本标签与 Release 按 CONTRIBUTING §6.2 的约定等这条通过后再建 |
| 未实现的专家档与共享专家 | 部署那份的 IQ1_M 3 层、另一档的 IQ3_S / IQ3_XXS 等尚无内核；共享专家与专家缓存未接 |
| 机器画像的并发分池一项 | QSA 侧已能算，GDN 侧按并发增长、需要 SSM 几何（未取到） |
| 引擎接口的实现细节 | 随实现阶段确定 |
| 服务层 M2 首刀：/metrics 的 Prometheus text 端口 | 契约对照 `ref_repos/strata` 的 `serve/prometheus.py`：Accept 含 text/plain\|openmetrics 或 `?format=prometheus` → text；vLLM 家族名（running/waiting、kv_cache_usage_perc、token 计数、request_success、prefix_cache 查询/命中、ttft/itl/e2e 三直方图 22 桶）+ `qinfer:` 前缀家族（live_state、totals_*_seconds_total、conversation_cache_*、gpu*/ram）；totals.requests 计数随补；text 与 JSON 同一快照渲染 |

## 3. 顶层目录（关键路径）

| 路径 | 内容 |
|------|------|
| `src/` | Q-Infer 服务/引擎自研代码（`serve/`、`engine/`、`dense/`、`kernels/` …） |
| `strata/` | 纳入本仓库的 Strata 引擎源码（MIT © 2026 Niko1221 and the Strata contributors）；由 `tools/build_engine.sh` 编成 `build-engine/strata`。**2026-10-15 起由 `third_party/strata/` 移到此路径** |
| `third_party/httplib/` | 仅存的第三方单头文件依赖（cpp-httplib） |
| `tools/` | 构建脚本（`build_engine.sh`、`build_vision.sh`） |
| `tests/` | `unit` 级源文件 + `integration/serve/` + `frontend/` + `data/`（夹具/oracle） |

> 历史 ADR（如 ADR-016/019/020）中出现的 `third_party/strata/` 为当时的路径记录，未逐条改写；当前权威路径以本节为准。
