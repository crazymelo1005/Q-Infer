# ADR-018：服务层默认计算后端切到厂商化引擎

**状态**：Accepted（2026-10-14）

---

## 背景

[ADR-017](ADR-017-serve-engine-seam.md) 决策 3 首版保守地把 `--backend` 默认值定为 `inhouse`，并写明「P2 端到端绿后切 `strata` 为默认」。该切换时机已到：同提示同参数下 `--backend strata` 与直接驱动引擎的参考输出逐字节相同（ADR-017 后果），聊天模板对参考端 golden 10/10 逐字节相同，且引擎启动期的诊断已能到达服务 stderr。

本决策只改一个默认值，不改接口，故按 [CONTRIBUTING](../../CONTRIBUTING.md) 的「ADR 不可修订、改变决策须新开 ADR」另立一条，不改 ADR-017 原文。

## 决策

1. `ServeConfig::backend` 默认值由 `inhouse` 改为 `strata`。
2. 默认值缺 `--engine-exe` 时拒绝启动，报错同时给出两条出路：补 `--engine-exe <引擎二进制>`，或显式 `--backend inhouse`。
3. `--gpu` 与默认 `strata` 并用时同样拒绝，报错指出 `--gpu` 只属 in-house 后端、引擎的设备由 `--engine-arg` 透传。
4. 集成套件与夹具路径显式传 `--backend inhouse`（测试夹具没有引擎），并新增一条用例钉住「裸启动被拒且报错含两条出路」。

## 备选方案

1. 隐式规则（给了 `--engine-exe` 就用 strata，否则回退 inhouse）。否决：默认值应当显式可预期；隐式回退会让「我以为在用参考引擎」在不该成立的时候成立。
2. 维持 `inhouse` 为默认。否决：与已约定的切换时机不符，且默认路径正好是那条已被判定输出垃圾的自研路径。

## 后果

正面：

- 默认即参考引擎：不带 `--backend` 的启动（在给了 `--engine-exe` 的前提下）走的是已验收的厂商化引擎。
- 失败可操作：报错写清两条出路，不留「静默回退到另一条引擎」这种歧义。

负面：

- 破坏性默认变更：既有不带 `--engine-exe` 的启动命令从此失败，需要补 `--engine-exe` 或显式 `--backend inhouse`；用法行与文档已同步，但依赖旧默认的脚本要跟着改。
- `--gpu` 的既有用法（若不带 `--backend`）同样会失败，须显式加 `--backend inhouse`。

## 相关

[ADR-017](ADR-017-serve-engine-seam.md)（接缝与两个后端，本 ADR 改其默认值一项）、[ADR-016](ADR-016-vendored-strata-backend.md)（厂商化引擎作后端）、`../interfaces.md` §8.8、`../engine.md` §18。
