# Q-Infer（擎）

在消费级 GPU 上运行超大稀疏 MoE 模型的推理引擎设计与研究。目标形态是 16 GB 显存的单卡或双卡消费级平台，主案例为 Qwen3.8-Flash-Next（125B 主参数 + 51B 外挂 n-gram 表 + 4B MTP，原生 262K 上下文）。

仓库当前处于设计阶段，只产出文档，不产出可运行的引擎。

命名：Q-Infer。「Q」兼指 Qwen（主案例）与 Quantized（量化）；「擎」取「擎起」——用 16 GB 擎起 125B。

## 适用范围

| 适用 | 不适用 |
|------|--------|
| 单机、单卡或双卡消费级 GPU | 服务端高并发吞吐 |
| 权重装不进显存的超大稀疏 MoE | 模型能整份常驻显存时（分层与预取均失效） |
| 超长上下文（262K 及以上） | 多机、分布式推理 |
| 以外挂表的确定性访问换取容量 | 训练、微调、数据管线 |
| | 通用多模型多后端平台 |

## 要满足什么

| 指标 | 目标 | 口径 |
|------|------|------|
| decode 吞吐 | 相对参考引擎 ≥1.5× | 同机、同量化档、同任务集、同并发 |
| TTFT / prefill | ≥1.3× | 同上 |
| 显存占用 | 不劣化 | 不得以增加常驻换取速度 |
| 输出质量 | 逐 token 分布不变 | 量化误差有界且可测 |

所有倍数指标的分母是环境2 的实测基线，不是上游公布的估算值。验收场景包含「262K 上下文 × 高比特量化档」。

完整的需求、非目标与验收条件见 [requirements.md](docs/requirements.md)。

## 架构一图

```
API / 调度层        OpenAI 兼容接口 · 批调度 · 前缀缓存 · 指标
        |
计划与预测层        表行索引预计算 · 专家共现预测 · 投机草稿树 · 精度档选择
        |
三层存储与流调度     VRAM · 主机 pinned · NVMe  + 统一页表 + PCIe 字节预算
        |
异构执行引擎        GPU 稠密与命中专家 · CPU 未命中专家 · 双 KV 池 · 投机验证
```

四条数据流：权重、专家、KV、记忆表。其中过 PCIe 的是权重、KV、记忆表三类，由同一步的字节预算统一仲裁；未命中专家走内存总线就地计算，不搬回显存。

模块职责与边界见 [overview.md](docs/overview.md)，逐模块规格见 [design/engine.md](docs/design/engine.md)。

## 当前状态

已完成：问题界定、竞品核查、架构设计、九条架构决策记录、门禁实测（13 条已测、G-11 得部分、G-14 部分未判定）、实施顺序序 0 的目标基线（单卡 decode 稳态 64.44、双卡 85.96 至 89.33 tok/s，见 [requirements §3](docs/requirements.md)）与序 1 的自动标定（机器画像，见 [research/01](docs/research/01-strata-engine.md) §10.2）；序 2 已落三片主机侧骨架：三层存储的页表、PCIe 字节预算仲裁器、记忆表行的 IQ4_NL 反量化内核（`src/`，CI 里构建并跑回归；内核用模型真实字节做过 oracle 校验）。其中 G-04 不通过：记忆表行在去重后无顺序性、亦无近邻复用，故表行预取子系统已砍，退守按需读加有界行缓存（[ADR-007](docs/design/adr/ADR-007-ngram-table-on-demand-read.md)）。G-07 判定未命中专家路径为算力瓶颈，核显分担的前提满足。

下一步：在环境2 上建立基线实测（全部倍数指标的分母）→ 完成门禁测量 → 跑通最小垂直切片。

门禁指必须先测量的假设，测量不通过即砍掉对应子系统并记录负结果，不得保留。清单与测量方法见 [design/gates.md](docs/design/gates.md)。

## 文档导航

| 想了解 | 去哪读 |
|--------|--------|
| 定位与范围 | 本文件 |
| 架构概览 | [overview.md](docs/overview.md) |
| 需求、指标与验收 | [requirements.md](docs/requirements.md) |
| 引擎规格 | [design/engine.md](docs/design/engine.md) |
| 模块接口 | [design/interfaces.md](docs/design/interfaces.md) |
| 待测门禁与测量方法 | [design/gates.md](docs/design/gates.md) |
| 风险登记 | [design/risks.md](docs/design/risks.md) |
| 候选方向（未验证，非规格） | [design/proposals.md](docs/design/proposals.md) |
| 决策记录 | [`docs/design/adr/`](docs/design/adr/) |
| 事实台账 | [`docs/research/`](docs/research/) |
| 全部出处 | [research/references.md](docs/research/references.md) |
| 术语 | [glossary.md](docs/glossary.md) |
| 硬件规格 | [hardware.md](docs/hardware.md) |
| 文档地图 | [docs/README.md](docs/README.md) |
| 写作与提交约定 | [CONTRIBUTING.md](CONTRIBUTING.md) |

## 仓库结构

```
.
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── CMakeLists.txt                     构建入口（C++20）
├── src/                               引擎实现
├── tests/                             引擎实现的回归测试（ctest）
├── scripts/docs_checks.py             规范自检（CI 与本地共用）
├── measure/                           测量工具与原始结果
├── .github/workflows/                 docs-check.yml（文档自检）· build.yml（构建与测试）
└── docs/
    ├── README.md             文档地图
    ├── overview.md           架构概览
    ├── requirements.md       需求与验收
    ├── glossary.md           术语（唯一口径）
    ├── hardware.md           硬件规格
    ├── design/
    │   ├── engine.md         引擎规格
    │   ├── interfaces.md     模块接口
    │   ├── gates.md          门禁与测量方法
    │   ├── risks.md          风险登记
    │   ├── proposals.md      候选方向
    │   └── adr/              架构决策记录
    └── research/
        ├── 01-strata-engine.md
        ├── 02-qwen3.8-flash-next.md
        ├── 03-qwen4-trends.md
        ├── 04-acceleration-and-memory.md
        ├── 05-engine-landscape.md
        └── references.md     出处总表
```

## 上游参考

设计建立在以下已有工作的核查之上。各引擎的完整出处见 [research/references.md](docs/research/references.md)。

| 项目 | 定位 | 与本项目的关系 |
|------|------|---------------|
| Strata | 消费级分层卸载引擎 | 参考范本：专家三级分层、未命中就地计算、内建 MTP |
| vLLM / SGLang | 服务端推理引擎 | 前提是权重常驻显存，在本场景不成立；其 n-gram 表卸载路径已实现表行异步预取 |
| llama.cpp / KTransformers | 层级与专家级 offload | 同方向，但无专家命中率概念；两套目标环境均无 AVX-512 / AMX，内核需重算 |
| NInfer | Blackwell 原生 FP4 稠密引擎 | 无卸载、不支持多卡，在本场景不适用；其精度路径可借鉴 |
| KVMem | KV 上下文虚拟化 | 同类 16 GB 卡上实现 256K 近无损，是本项目 KV 侧的对照基线 |

Qwen3.8-Flash-Next 的官方 vLLM 部署配方：<https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。当前最需要的贡献是事实核验（把推算换成实测）与门禁测量。

提交前可在本地运行规范自检：`python3 scripts/docs_checks.py`（CI 使用同一份判据）。

## 许可

采用 Apache-2.0，全文见 [LICENSE](LICENSE)。

## 引用

```
Q-Infer: 消费级 GPU 上的超大稀疏 MoE 推理引擎设计.
https://github.com/crazymelo1005/Q-Infer
```

---

## English summary

Q-Infer is a design-stage project: research and architecture for running very large sparse-MoE models on consumer GPUs with 16 GB of VRAM (single or dual card). The primary case study is Qwen3.8-Flash-Next — 125B main parameters plus a 51B external n-gram table and a 4B MTP head, with a native 262K context.

The premise is that the weights never fit in VRAM, so the engine is built around tiered offload of expert weights, an explicitly prefetched n-gram table, and long-context KV management. It targets single-machine, low-concurrency use (2–8 concurrent requests), not server-grade throughput, and not multi-node deployment.

- Requirements and acceptance criteria: [requirements.md](docs/requirements.md)
- Architecture overview: [overview.md](docs/overview.md)
- Engine specification: [design/engine.md](docs/design/engine.md)
- Gates that must be measured before implementation: [design/gates.md](docs/design/gates.md)
- Decision records: [`docs/design/adr/`](docs/design/adr/)
- Research ledger and references: [`docs/research/`](docs/research/)

This repository currently contains documentation only. There is no implementation yet.

License: Apache-2.0 — see [LICENSE](LICENSE).
