# 设计文档索引

本目录是实现的规格与决策层。上层导览见 [docs/README.md](../README.md)，需求与验收见 [requirements.md](../requirements.md)。

## 规格文档

| 文件 | 回答的问题 | 状态 |
|------|-----------|------|
| [engine.md](engine.md) | 引擎各模块怎么分工、各阶段的验收与相位读数 | 进行中（序 2 部分） |
| [interfaces.md](interfaces.md) | 模块之间的接口与纯虚边界（Backend 等） | 完成 |
| [gates.md](gates.md) | 哪些假设必须先测量、怎么测 | 进行中（G-14 未判定） |
| [risks.md](risks.md) | 风险登记 | 完成 |
| [proposals.md](proposals.md) | 候选方向（未验证，非规格） | 完成 |
| [native-engine-garbled-output-and-multi-gpu-design.md](native-engine-garbled-output-and-multi-gpu-design.md) | 自研引擎乱码与多卡协同的当时设计 | 历史（已由厂商化引擎路线取代） |
| [native-engine-investigation-status.md](native-engine-investigation-status.md) | 自研引擎乱码的调查状态 | 历史 |

## 决策记录（ADR）

| 编号 | 主题 |
|------|------|
| [ADR-001](adr/ADR-001-tiered-storage-and-expert-cache.md) | 三层存储与专家缓存 |
| [ADR-002](adr/ADR-002-compute-unhit-experts-in-place.md) | 未命中专家就地计算 |
| [ADR-003](adr/ADR-003-ngram-table-explicit-prefetch.md) | n-gram 表显式预取（后砍） |
| [ADR-004](adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md) | 消费级双卡、不做张量并行 |
| [ADR-005](adr/ADR-005-igpu-npu-as-coprocessor.md) | 核显 / NPU 作为协处理器 |
| [ADR-006](adr/ADR-006-precision-policy-nvfp4-fp8-kv.md) | 精度策略（NVFP4/FP8/KV） |
| [ADR-007](adr/ADR-007-ngram-table-on-demand-read.md) | n-gram 表按需读 + 有界行缓存 |
| [ADR-008](adr/ADR-008-implementation-stack-and-kernel-reuse.md) | 实现栈与内核复用 |
| [ADR-009](adr/ADR-009-correctness-acceptance-criteria.md) | 正确性验收口径 |
| [ADR-010](adr/ADR-010-router-precision-declared-format.md) | 路由精度与声明格式 |
| [ADR-011](adr/ADR-011-compute-group-abstraction.md) | 计算组抽象 |
| [ADR-012](adr/ADR-012-http-transport-and-serving-stack.md) | HTTP 传输与服务栈 |
| [ADR-013](adr/ADR-013-vision-port-and-dependency-boundary.md) | 视觉移植与依赖边界 |
| [ADR-014](adr/ADR-014-gpu-igpu-serve.md) | GPU + 核显服务 |
| [ADR-016](adr/ADR-016-vendored-strata-backend.md) | 纳入参考引擎作后端 |
| [ADR-017](adr/ADR-017-serve-engine-seam.md) | 服务层与引擎的接缝 |
| [ADR-018](adr/ADR-018-default-backend-strata.md) | 默认后端即参考引擎 |
| [ADR-019](adr/ADR-019-igpu-not-for-experts.md) | 核显不承担专家计算 |
| [ADR-020](adr/ADR-020-strata-igpu-pool.md) | 引擎侧核显专家池 |

注：历史文档中的路径措辞（如厂商化引擎树的旧位置）未逐条改写，当前权威目录见 [docs/README.md](../README.md) 的顶层目录一节。
