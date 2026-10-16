# ADR-023：把 vendored Strata 引擎升到 v0.1.44.1，并撤销 igpu 集成

状态：Accepted（2026-10-16）

---

## 背景

ADR-016 第 1 条把引擎树「逐字节冻结在 v0.1.43.1，不追上游（追版留作后续）」。本 ADR 执行那个后续。

触发理由是 0.1.43.1 上可摸的性能旋钮已经穷尽：约 25 个旋钮逐项实测为中性、负或被引擎拒绝——
prefill 档位 512/2048/4096/8192/16384、pipeline-windows 3、spec 6、kv-resident、max-context、
pcie-frac、expert-cache-per-layer、pool-workers、lookup-chain、suffix-draft、mtp-q4、kv-grow，
以及厂商那张 18 条的 arch_defaults 开关表（它只对 AMD 构建生效，手工在 sm_120 上成批打开也中性）。
证据与逐项读数见 ADR-021 补充二/三与项目记忆 `research-qwen38-arch-dual-gpu`。

上游 0.1.43.1 到 0.1.44.1 有三项直接对口的改动：

- `STRATA_COMMIT_BATCH` 默认开：每个 GDN 层的 conv 历史加状态、每个 QSA 层的 tail restore 加 indexer append
  各合成一次发射；上游注释记 "~70 graph nodes fewer … 2.35 -> 0.65 ms GPU"。
- `--prefill-head-alias`（opt-in，--serve 加单卡）：head 的显存作提示工作区；上游记 29,854-token 提示上
  提示 chunk 512 到 2,048、prefill 292.5 到 667.1 tok/s。
- `STRATA_VERIFY_ASYNC_LAUNCH=1`（opt-in）：图发射阻塞时仍服务 CPU 门铃。

## 决策

1. 入库树整体替换为上游 v0.1.44.1 的源码子集（`CMakeLists.txt LICENSE cmake include src third_party`，
   与既有裁剪口径一致，不含 upstream 的 tests/ 与 bench/），替换后与上游该 tag 逐文件一致（构建脚本每次核对）。
2. 构建改为钉版 ggml。上游 `third_party/ggml/VERSION.txt` 钉 commit `3cf0325`，与 `tools/build_vision.sh`
   早已使用的那一枚相同。`tools/build_engine.sh` 现在自己 `git archive` 该提交到 `build-engine/llama.cpp`
   （只读：不下载、不检出、不变更任何工作区），`STRATA_GGML_DIR` 仍可覆盖以做 A/B。
   此前用整棵 `~/llama.cpp`（2026-07-28）是版本失配的根源：它触发 `moe_mmq.cu` 的
   "too many initializer values"，当时以本地少写一个字段绕开；那行本地改动随本次替换一并撤销。
3. 撤销 ADR-020 的核显专家池集成：该树内的 `src/kernels/igpu`、`include/strata/kernels/igpu` 与 6 处钩子
   （`weights.hpp`、`native_dense.cpp`、`layer.cpp`、`generate.cpp`、`CMakeLists.txt`、`moe_mmq.cu` 的绕改）
   全部不再随树带过去。理由：核显只做多模态编码器（用户 2026-10-15 的指示），核显参与计算的路线已由
   ADR-019/022 否决，且上游 0.1.44.1 已删除 `STRATA_IGPU_POOL` 与 `STRATA_IGPU_MIN_FLOPS`。
   ADR-020 状态改为 superseded。

## 测量

同机同题：单卡 `--gpu 0 --resident-experts`，其余 as-found；21K 与 131K 冷提示、300-token 散文 decode。

| 指标 | 0.1.43.1（原现役） | 0.1.44.1（本 ADR） |
|---|---|---|
| 21K prefill | 1,905.5 / 1,906.5 / 1,908.3 tok/s | 1,904.3 tok/s |
| 131K prefill | 1,857.0 / 1,857.5 tok/s | 1,861.4 tok/s |
| decode（300 token） | 89.7–103.9（跨重启 ±5%） | 94.0 / 98.2 / 100.7 tok/s |
| `no tile for q2_0` 告警 | 有 | 无（钉版 ggml 带 16 条 Q2_0 MMQ 配置） |

结论：升级本身在性能上中性（prefill 差 ±0.2%，decode 在噪声内），拿到的是上游的新开关与后续版本线。
`--prefill-head-alias` 在本机不可用：它要借 9,499 MiB 作提示工作区，把 16 GB 单卡压到 293 MiB 空闲，
请求直接失败（`out 0`）；上游那 2.3x 需要 head 显存能腾出来的机器。
端到端复验：真实请求 300 token 正常返回。

## 后果

- ADR-016 第 1 条的「不追上游」由本 ADR 修订：追版成为一次显式决策，并且构建自此依赖被钉版的 ggml 提交
  （`tools/build_engine.sh` 自行物化，`LLAMA_SRC` 指向本机 llama.cpp 检出，默认
  `$HOME/projects/ref_repos/llama.cpp`）。
- 视觉得编码器（`tools/vision`）不随本次升级：它是从 v0.1.43.1 移植的独立产物，出处与门禁
  （`measure/vision_port_check.py` 与 `vision_oracle.py`）保持原样；引擎与编码器各有自己的版本钉，NOTICE 同步。
- 服务侧默认与版本无关，继续有效：`--resident-experts`（ADR-021 补充三）、多模态默认加核显（ADR-013 补充）、
  `STRATA_PF_FUSED` 与 `STRATA_PREFILL_CPU_SHARE` 注入。
- 驱动层的协议口径不变：`tests/data/strata_transcript_gen.txt` 录自 0.1.43.1，本次端到端复验确认
  驱动依赖的协议行未变；`engine.md` 的版本行随之更新。