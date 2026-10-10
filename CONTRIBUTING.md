# 贡献指南

仓库处于实现阶段：设计规范见 `docs/design/`，代码自实施顺序序 2 的骨架起逐步落地（见第 7 节）。以下约定适用于文档与代码的写作、编号与提交；内容口径以 [docs/README.md](docs/README.md)（文档地图）与 [docs/glossary.md](docs/glossary.md)（术语唯一口径）为准。

---

## 1. 文档分层与目录

文档分三层，各层受众不同，互不混装。

| 层 | 目录 | 内容 | 受众 |
|----|------|------|------|
| 外部 | [README.md](README.md)、[overview.md](docs/overview.md) | 定位、状态、架构概览 | 外部读者、评审者 |
| 实现 | `docs/design/` | 引擎规格、接口、门禁、风险、候选方向、ADR | 实现者、测量者 |
| 台账 | `docs/research/`、[glossary.md](docs/glossary.md)、[hardware.md](docs/hardware.md) | 事实、术语、硬件规格与出处 | 研究者、全体 |

| 文件 | 回答的问题 |
|------|-----------|
| [requirements.md](docs/requirements.md) | 要满足什么（目标负载、指标、验收、非目标） |
| [overview.md](docs/overview.md) | 系统由哪几块组成、边界在哪 |
| [design/engine.md](docs/design/engine.md) | 每块具体怎么做（规格） |
| [design/interfaces.md](docs/design/interfaces.md) | 模块之间的对外接口 |
| [design/gates.md](docs/design/gates.md) | 哪些假设必须先测、怎么测、通过标准 |
| [design/risks.md](docs/design/risks.md) | 已知风险与对策 |
| [design/proposals.md](docs/design/proposals.md) | 未验证的候选方向（非规格） |
| `docs/design/adr/` | 已作出的决策与理由 |
| `docs/research/` | 事实台账（是什么） |
| [research/references.md](docs/research/references.md) | 全部出处，按编号登记 |
| [scripts/docs_checks.py](scripts/docs_checks.py) 与 [.github/workflows/docs-check.yml](.github/workflows/docs-check.yml) | 规范自检：把第 2、3、4 节的规则变成可执行检查（非文档） |
| [measure/](measure/) | 测量工具与原始结果：机器画像与标定、PCIe 档位与带宽、内存带宽、显存带宽、GGUF 元数据读取、记忆表行访问局部性重放与语料构造、专家路由覆盖曲线、投机 IO 放大、CPU 专家内核与量化 GEMM 微基准、运行中引擎的槽位/命中率/KV 驻留采样、NVMe 往返延迟、IQ2_XS / IQ2_S / IQ2_XXS / IQ4_NL 反量化及点积的回归期望值生成、Q2_0 真字节夹具生成、KV 精度档的输出一致率（非文档） |
| [src/](src/) | 引擎实现（C++20）：`artifact/gguf_table`（GGUF 读取：记忆表张力专用 `GgufTable` + 通用逐张量枚举 `GgufFile`）、`experts/expert_formats`（专家格式分派表：逐层从 GGUF 的类型码推出档、块几何与激活搭档，缺内核或几何不成立都进 problems）、`storage/page_table`（三层存储与页表）、`storage/row_cache`（有界行缓存）、`scheduling/budget_arbiter`（PCIe 字节预算仲裁）、`kernels/ngram`（表行索引哈希）、`kernels/iq4nl`（IQ4_NL：记忆表表行的 160 值行格式 + 通用权重块 32 值/18 字节，含与 Q8_0 的定点点积与逐行 GEMV，码本在 `kernels/fp16` 之侧）、`kernels/iq2xs`（IQ2_XS 档的权重块反量化，表体在 `kernels/iq2xs_tables`）、`kernels/iq2s`（IQ2_S 档的权重块反量化，格点表在 `kernels/iq2s_tables`，符号表复用 IQ2_XS 那张）、`kernels/iq2xxs`（IQ2_XXS 档的权重块反量化，格点表在 `kernels/iq2xxs_tables`，符号表复用同一张）、`kernels/q8k`（激活侧 Q8_K 块量化）、`kernels/iq2xs_dot`（IQ2_XS × Q8_K 的定点点积与逐行 GEMV）、`kernels/q2_0`（Q2_0 块反量化与 Q2_0 × Q8_0 的定点点积与逐行 GEMV）、`kernels/q8_0`（Q8_0 块反量化，Q2_0 的激活搭档）、`kernels/ple_gather`（head-slowest 拼装）。格式事实与各档的落点见 [S-36](docs/research/references.md)、[S-37](docs/research/references.md)、[S-39](docs/research/references.md) 与 [engine §6](docs/design/engine.md) |
| [tests/](tests/) | 引擎实现的回归测试（无第三方框架，`ctest` 驱动） |
| [CMakeLists.txt](CMakeLists.txt) | 构建入口 |
| [.github/workflows/build.yml](.github/workflows/build.yml) | 构建与测试（与文档自检并列的第二个 CI 作业） |

分层规则：`research/` 不得引用 `design/`；`design/` 的关键论断必须能回指 `research/` 或 [hardware.md](docs/hardware.md)。

新增文件准入：能并入现有文件的一律并入。新增文档须登记到 [docs/README.md](docs/README.md) 的文档清单；新增任何文件都须登记到上表。

---

## 2. 写作规范

文档写规格与事实，不写论证过程、不写自我评价、不写修改历史。开篇可以用一句话说明文档范围，其余位置不写自指内容。

禁止出现：

- 第一人称（我、我们）
- 编辑史与自指元评论：原 X 版、已改写、落地状态、评审告知、复核补注、不改决策、本节已按某处修改
- 文件级版本号或日期头
- 行内 emoji（来源等级与复核状态一律用方括号标记，见第 3 节）
- 疑问句标题

版本号、日期与变更记录只存放在提交历史里。需要说明「当时为什么这么定」时写 ADR，不写文件内注记。

格式约束：

- 加粗只用于术语首次定义与硬约束，密度不超过每 100 行 10 处
- 数字必须带单位与口径；属于推算的必须写明"推算"，并在实测后回填
- 术语首次出现给出英文全称；术语以 [glossary.md](docs/glossary.md) 为唯一口径，正文使用的新术语必须同步登记
- 章节号、编号只增不改，避免外部引用失效

---

## 3. 来源分级与引用

每个数值与事实都要能看出来源层级，禁止把不同来源的数字混成一句结论。

| 等级 | 含义 |
|------|------|
| 环境2 实测 | 在环境2 上读自运行中的引擎 API，或在其上跑本地 benchmark |
| 环境1 实测 | 在本机（环境1）上同口径取得。其平台是 Windows 宿主内的 WSL2，引用时必须写明测在 WSL 内还是 Windows 宿主上 |
| 硬件规格 | 厂商规格页，是上限而非实测 |
| 官方 | 官方发布页、官方文档或论文口径 |
| 他卡实测 | 社区在其它硬件上的实测，趋势可信、绝对值浮动 |
| 推算 | 由模型估算，须标定后方可作为结论 |

复核状态单列，表示查到什么程度：已逐位确认 / 仅标题或摘要 / 来源矛盾已降级。

标记写法：写在数值或事实之后，用方括号，例如 `[环境2实测]`、`[官方]`、`[已确认]`。表格可用一行注释统一标注整表的等级。全部标记与定义见 [research/references.md](docs/research/references.md) 第 1、2 节。

每条来源在 [research/references.md](docs/research/references.md) 登记一行：

```
S-<n> | 等级 | 复核状态 | 来源（URL 或文档路径） | 引用日期
```

正文引用写作 `[S-n]`。新增来源即追加编号并回填正文；编号只增不改。

`[S-n]` 只用于仓库之外的来源。仓库内部引用一律使用相对链接加章节号，不编 `S-n`。

台账类文档在末尾设「来源」段，用表格列出本文使用的编号与对应章节：

```
| 编号 | 用于章节 | 内容 |
```

正文中某个数字若只来自单一来源，可直接在句中标 `[S-n]`；整表或整节的来源由「来源」段承担，避免行内堆砌标记。

---

## 4. 编号体系

| 前缀 | 用途 | 登记位置 | 可变性 |
|------|------|---------|--------|
| `ADR-NNN` | 架构决策 | `docs/design/adr/` | 不可修订；改变决策须新开 ADR 并标注取代关系 |
| `G-NN` | 门禁与待测项 | [design/gates.md](docs/design/gates.md) | 只增不改 |
| `R-NN` | 风险 | [design/risks.md](docs/design/risks.md) | 只增不改 |
| `P-NN` | 候选方向 | [design/proposals.md](docs/design/proposals.md) | 只增不改 |
| `S-N` | 来源 | [research/references.md](docs/research/references.md) | 只增不改 |

新编号必须同时登记到上表指定位置；未登记即视为不存在。

---

## 5. 实测与评测流程

把手上的推算换成实测是当前的主要工作，流程固定如下。

1. 分母口径：同机、同量化档、同任务集、同并发。分母必须是环境2 上的实测，不得使用上游估算值。
2. 每次配置重复不少于 5 次，报中位数与区间，并记录当次机器画像。
3. 投机与预取类收益必须使用真实任务集，禁用随机 prompt。
4. 结果写入 `docs/research/` 的对应条目，并回填 [design/gates.md](docs/design/gates.md) 中该 `G-NN` 的实测记录列。
5. 门禁不成立时砍掉对应子系统，负结果写入 [design/risks.md](docs/design/risks.md) 与 `docs/research/`，不得删除、不得静默保留。
6. 冷启动与稳态分开记录。引擎的累计窗口随进程重启归零，历史值无法从端点复现，引用时必须写明口径。

---

## 6. 提交、分支与发布

### 6.1 分支

- `main` 是唯一长期分支，也是默认分支。每个提交之后文档集必须自洽：链接可达、编号登记齐全。
- 非琐碎改动使用短期分支，命名 `<type>/<short-topic>`，合并后删除。单人开发也走 PR，以留下审查痕迹与 CI 结果。
- 不使用 `develop`、`release/*`、`hotfix/*`：仓库没有构建产物，也没有需要并行维护的旧版本。

### 6.2 标签

标签一律用附注标签（`git tag -a <name> -m <说明>`），创建后须显式推送（`git push origin <tag>`）。分三类：

| 类型 | 命名 | 何时创建 | 用途 |
|------|------|---------|------|
| 设计里程碑 | `design-v<主>.<次>` | 规范发生决策级变化：ADR 被取代，或门禁砍掉子系统导致规范改变 | 标注当时的设计立场，供外部引用 |
| 实测快照 | `measure-<年-月>`；单次门禁可用 `<G-NN>-<年-月-日>` | 每完成一轮基线或门禁实测 | 让 `gates.md` 与 `research/` 中的数字指向不可变的文档状态 |
| 代码版本 | `v<主>.<次>.<修订>` | 仅在实施顺序序 2 的骨架跑通之后 | 配合 GitHub Release |

设计里程碑用两位号，代码版本才用三位：设计没有补丁级变化，三位号会产生无意义的精度。

### 6.3 发布

- 每个标签都可发一个 GitHub Release。正文写清该状态包含什么、哪几条门禁已通过、哪几条未测、有何已知缺口。
- 不维护 `CHANGELOG.md`：变更记录由提交信息、标签与 Release 说明承担（文件内不写变更日志，见第 2 节）。
- 首个 Release 对应 `design-v0.1`：规范冻结、六条 ADR、`G-01` 至 `G-14` 全部未测。

### 6.4 提交与 PR

- 提交信息首行 `<type>: <祈使句摘要>`，正文写清为什么改、依据哪个 `G-NN` 或 `S-n`。
  类型：`feat`、`test`、`build`、`chore`、`docs`、`spec`、`research`、`gate`、`adr`、`risk`。
- 一次提交只做一件事；纯格式整理与内容修改分开提交。
- 已推送的历史不得重写，纠错用新提交。`amend` 只允许用于尚未推送的提交。
- PR 自检清单：新论断有 `[S-n]` 出处？数字带口径？新术语已登记？新编号已登记？没有第 2 节列出的禁用写法？
- 未经明确授权不得 push 或改动远端。

---

## 7. 代码

代码分两类。

规范自检，随 CI 执行：[scripts/docs_checks.py](scripts/docs_checks.py)，Python 3 标准库，无第三方依赖，无构建步骤，运行方式 `python3 scripts/docs_checks.py`，由 [.github/workflows/docs-check.yml](.github/workflows/docs-check.yml) 在推送与 PR 时执行。任一检查失败即非零退出，本地与 CI 使用同一套判据。

测量工具，手工执行、不进 CI。分三种语言：`measure/` 下的 Python 3 脚本（多数零第三方依赖，`mem_bw.py` 依赖 numpy）、`measure/pcie_bw.cu`、`measure/io_roundtrip.cu` 与 `measure/vram_bw.cu`（CUDA C++，用 nvcc 编译）与 `measure/cpu_expert_bench.c`、`measure/cpu_gemm_bench.c`（纯 C，用 `cc -O2 -mavx2 -mfma -pthread` 编译）。三者都没有构建系统，直接调用编译器或解释器。

- `python3 measure/calibrate.py [--set KEY=VALUE ...]`：机器画像（engine §10 标定）。把已测的量与模型几何代入，按显式公式推导每步 PCIe 预算、专家槽位、KV 窗口与 §15 的上限，落盘到 `measure/results/*-calibrate-<平台>.json`。默认值即环境2 的实测值，`--set` 可覆盖；`--selftest` 核对推导恒等式。零依赖。
- `nvcc -O2 -o build/vram_bw measure/vram_bw.cu && ./build/vram_bw --sizes 32,128,512`：显存可用量与带宽（device STREAM），画像的第 1 项输入。产物写在 `build/`。
- `python3 measure/env_profile.py`：记录机器画像的静态部分。
- `python3 measure/ple_row_oracle.py --model <shard2.gguf> --rows 0,12345`：从 GGUF 里取记忆表的指定行，输出原始字节与按 IQ4_NL 规则反量化的结果，用作内核回归的 oracle。零依赖；它的输出与本仓库记录过的引擎自带 oracle 向量逐位吻合，可作交叉验证。
- `python3 measure/pcie_link.py [--watch 秒数]`：读 PCIe 链路档位。加 `--watch` 可在同时施加负载时观察档位是否变化。
- `python3 measure/mem_bw.py [--size-mb 256] [--procs 8] [--seconds 0.6] [--repeats 5]`：测内存带宽。聚合口径是「各进程屏障同步后在同一时间窗内搬运的字节之和除以时间窗」；不得用各进程中位数相加，那样在进程启动不同步时会虚高，甚至超过内存理论峰值。
- `nvcc -O2 -o build/pcie_bw measure/pcie_bw.cu && ./build/pcie_bw --repeats 5 --out-dir measure/results`：测 PCIe 有效带宽。产物写在 `build/`，该目录已在 `.gitignore` 中，不入库。`--hold 秒数` 会保持链路流量，便于同时用 `pcie_link.py` 观察宽度与代数是否变化。注意链路**代数会随负载降档**（空闲可低至 gen1），因此"当前档位"必须在持续负载下才可引用。
- `python3 measure/gguf_meta.py --model <file.gguf> [--tensors] [--all]`：直读 GGUF 元数据与张力维度，用于取模型几何（头数、维度、层数、专家数、表形状），不必依赖 HuggingFace 的 `config.json`。零依赖，可经 `ssh host "python3 - --model …" < measure/gguf_meta.py` 在远端的模型文件上直接运行。
- `python3 measure/ple_locality.py --tokens <token_ids.txt> [--env 环境2]`：G-04，离线重放记忆表行的访问局部性。输入是分词后的 token id 序列（空白分隔，空行表示序列边界），输出行分布、重复率、去重后顺序性与行缓存命中率，并落盘 JSON。`--selftest` 把重放实现与参考引擎自带的 oracle 向量逐位比对。零依赖；与机器无关，分词需在被测环境上由该引擎的 tokenizer 完成。
- `python3 measure/make_ple_corpus.py --engine-dir <引擎目录> --native-gguf <分片.gguf> --out <目录>`：构造 G-04 的语料（token id 序列，每份记录 token 数与 sha256）。需在参考引擎环境内运行，因为它调用该引擎自带的 tokenizer（依赖第三方 `regex`），故不属零依赖工具；重放侧仍由 `ple_locality.py` 完成。
- `python3 measure/expert_coverage.py --trace <dump-routing 轨迹>... [--profile <画像.bin>]`：G-09，统计路由轨迹里的 (层, 专家) 激活频次并给出 top-N 覆盖曲线；`--profile` 时另把引擎画像的排名当频次曲线用，量化该代用造成的偏差。轨迹由参考引擎的 `--dump-routing` 生成（格式见其 `tools/make_profile.py`）。零依赖。
- `python3 measure/spec_io_amplify.py --log <引擎日志>... [--trace <轨迹>...]`：G-13，从日志的 `speculation` 行取接受率并取倒数，另按轨迹的记录数 ÷ 层数 ÷ 生成 token 数直接数出位置放大。零依赖。
- `cc -O2 -mavx2 -mfma -pthread -o build/cpu_expert_bench measure/cpu_expert_bench.c && ./build/cpu_expert_bench --size-mib 512 --threads 1,2,4,8,12,16,24`：G-07，在同一字节量上比较纯流式读、4 位码本反量化+FMA、2 位码本反量化+FMA 三个臂的吞吐，判定未命中专家路径是算力瓶颈还是带宽瓶颈。产物写在 `build/`（已 gitignore）。
- `cc -O2 -mavx2 -mfma -pthread -o build/cpu_gemm_bench measure/cpu_gemm_bench.c && ./build/cpu_gemm_bench --bits 4 --experts 48 --pin 0-14`：G-03，按专家真实几何跑码本反量化 GEMV，用 `--pin` 把线程绑到指定 CPU（环境2 上 P 核 0-7、E 核 8-23），量出每类核与各池规模的吞吐。产物写在 `build/`。
- `python3 measure/engine_cache_probe.py --engine-dir <引擎目录> --base-config <serve 配置> --prompt <提示.txt> --slots auto|N --out <记录.json>`：G-05 / G-06 / G-12 与基线，起一次 serve 实例、发真实提示的 greedy 请求、读 `/metrics` 与引擎日志、停服务，产出该配置的专家槽位、解码命中率、PCIe 占比、KV 驻留与 decode 吞吐。`--repeat N` 在同一实例内连发 N 次并给出中位数与区间（基线口径要求 ≥5 次，故基线必须用 `--repeat`，不能靠多次重启）；`--poll-ms N` 轮询 `/metrics` 的 `live.tok_s` 以给出 decode 步长分布；`--gpu LIST` 与 `--drop-arg <旗标>` 用来改机器形态（如单卡：`--gpu 0 --drop-arg=--layer-split`）。`--from-raw` 可把一份原始记录精简重写（幂等）。需在装有该引擎的环境上运行。
- `nvcc -O2 -o build/io_roundtrip measure/io_roundtrip.cu && ./build/io_roundtrip --file <大文件> --mode direct --lane ample`：G-10，随机取 16 个 4 KiB 页串行读再一次 H2D，给出读腿与整条往返的 P50 / P99；`--mode cached` 走页缓存，`--hold-gib N` 先占住 N GiB 造出内存紧张的 lane。产物写在 `build/`。
- `python3 measure/kv_quality.py --run fp16=<log> --run int8=<log> --run k8v4=<log> --run q4_0=<log> --baseline fp16`：G-14 的 top-1 一致率，比对同一提示、同一采样下不同 KV 精度档的贪心输出序列（一致率 = 公共前缀 ÷ 较短长度）。输入是引擎 `generate` 的日志；比对时必须带一条**同配置的对照 run**，并给引擎加 `--adapt-every 0`，否则自适应换入的舍入差异会与精度差异混在一起。零依赖。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2xxs_oracle.py --model <分片1.gguf> --tensor blk.1.ffn_gate_exps.weight --ggml-common <检出>/ggml/src/ggml-common.h --blocks 16 --emit-inc`：IQ2_XXS 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ2_XXS.dequantize_blocks`（独立实现路径），并顺带核 256 项格点表在 ggml C 源与 gguf-py 打包之间逐字节一致、按 66 字节/块反推的整张量字节数与引擎侧 `native_experts.txt` 的逐专家字节对齐（gu_type = 16 的层每专家 gate 为 422,400 字节）。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq4nl_oracle.py --model <IQ3_XXS 分片1.gguf> --tensor blk.0.ffn_down_exps.weight --blocks 16 --emit-inc`：IQ4_NL 作**通用权重块**的真字节夹具生成器（记忆表表行那条路径的夹具见 `measure/ple_row_oracle.py`）。期望值取自 gguf-py 的 `IQ4_NL.dequantize_blocks`（独立实现路径），并核按 18 字节/块反推的整张量字节数。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2s_oracle.py --model <分片1.gguf> --ggml-common <检出>/ggml/src/ggml-common.h --tensor blk.0.ffn_gate_exps.weight --blocks 16 --emit-inc`：IQ2_S 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ2_S.dequantize_blocks`（独立实现路径），并顺带核两件事：1024 项格点表在 ggml C 源与 gguf-py 打包之间逐字节一致、按 82 字节/块反推的整张量字节数与引擎侧 `native_experts.txt` 的逐专家字节对齐。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `python3 measure/q2_0_oracle.py --model <分片1.gguf> [--tensor blk.0.ffn_down_exps.weight] [--blocks 64] [--emit-inc]`：从真实 GGUF 取一个 Q2_0 张量的前几块做夹具（结构不变量 + sum/sumabs + 前 16 值），并按张量字节数反推 `QK2_0` 的取值（64 与 128 两种假设的 bpw 不同）。同一窗口可在引擎自身的 `strata-dequant` 上取 CHECKSUM 交叉核对。零依赖，需真模型文件。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2xs_oracle.py --ggml-common <检出>/ggml/src/ggml-common.h --emit-inc|--emit-q8k-inc|--emit-dot-inc`：专家点积路径的回归期望值生成器。三种模式各写一份 `.inc`：`--emit-inc` 给 IQ2_XS 单块反量化（期望值取自 gguf-py 的独立实现）；`--emit-q8k-inc` 给 Q8_K 参考激活量化器；`--emit-dot-inc` 给 IQ2_XS × Q8_K 定点点积（后两者的期望值是 ggml 对应函数的忠实转写）。三种模式都先把 ggml 的 `iq2xs_grid` 从 C 源抽出来与该库的打包表示逐字节核对。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。

测量结果的存放：原始记录写入 `measure/results/<时间戳>-<测点>-<平台>.json`（Python 脚本与 CUDA 程序都自己落盘）；摘要回填 [docs/design/gates.md](docs/design/gates.md) 的实测记录列，以及 [docs/hardware.md](docs/hardware.md) 的实测值表。每条数值必须标注平台：环境1-WSL、环境1-Windows 或环境2。两套环境对比时必须使用同一份脚本与同一组参数。

平台标签：脚本能自动区分环境1-WSL 与环境1-Windows，但**在原生 Linux 上无法判断是环境2 还是别的机器，必须显式传 `--env 环境2`**，否则记录会落入「原生Linux(请确认是否为环境2)」这一未确认标签，不能作为环境2 的数据引用。

引擎实现的语言、构建与测试口径（[ADR-008](docs/design/adr/ADR-008-implementation-stack-and-kernel-reuse.md)、[ADR-009](docs/design/adr/ADR-009-correctness-acceptance-criteria.md)）：

- 语言 C++20，构建 CMake（≥3.28），不引包管理器；依赖面为零（第三方内核以裁剪后的子集入库，文件头登记上游路径、提交号与许可，见 [research/references.md](docs/research/references.md)）。
- GPU 侧用 CUDA，但主机侧代码不得依赖它：CI 在没有 GPU 的 runner 上编译并跑测试。CUDA 部分单独标注、在本机跑。
- 测试不引第三方框架：`tests/` 下的可执行文件用 [`tests/check.hpp`](tests/check.hpp) 的 `CHECK`，`ctest` 驱动，CI 与本地同一条命令——`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`。**不要用 `<cassert>` 的 `assert`**：Release 构建定义 `NDEBUG`，它会把这些检查整个编译掉，测试于是变成空跑，「全绿」不再意味着任何事。
- 测试范围：主机侧的不变量与确定性算术优先（页表不变量、预算仲裁、标定推导）。涉及 GPU 内核的回归走 [ADR-009](docs/design/adr/ADR-009-correctness-acceptance-criteria.md) 的两档口径（固定开关下逐位；默认真实配置用分布级）。
- 需要真实模型文件才能跑的验收不进 CI：CI 用现场生成的最小合成夹具（如 `test_gguf_table` 自造一个只含一张表的小 GGUF）。要用真文件验收时走该测试的手工入口，例如 `./build/test_gguf_table --model <分片2.gguf> --row 0`，其输出与 `measure/ple_row_oracle.py` 对照。

---

## 8. 许可与第三方内容

代码与文档采用 Apache-2.0，全文见 [LICENSE](LICENSE)。

引用第三方数据、图表、代码片段时，在 [research/references.md](docs/research/references.md) 标注来源与许可状态。

---

## 9. 维护者与当前优先事项

维护者负责合并 PR、裁决编号分配与 ADR 的接受状态。重大决策写入 ADR，不写入提交信息。

当前优先事项，按顺序：

1. 在环境2 上建立基线实测，作为全部倍数指标的分母。
2. 按 [design/gates.md](docs/design/gates.md) 完成门禁测量，失败即砍子系统并记录负结果。
3. 核验 [research/references.md](docs/research/references.md) 中的来源，把推算换成实测。
