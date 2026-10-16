# 贡献指南

仓库处于产品实现阶段：`src/` 是服务与引擎实现，`docs/` 面向使用者与运维者，`docs/adr/` 记录决策及其证据。以下约定适用于文档与代码的写作、编号与提交；术语唯一口径见 [docs/glossary.md](docs/glossary.md)。

---

## 1. 文档分层与目录

文档按受众分三类，互不混装。

| 类 | 文件 | 内容 | 受众 |
|----|------|------|------|
| 外部 | [README.md](README.md)、[overview.md](docs/overview.md) | 定位、状态、架构概览 | 外部读者、评审者 |
| 运维与实现 | [docs/deployment.md](docs/deployment.md)、[docs/overview.md](docs/overview.md) | 构建、运行、内存预算、实测过的旋钮、组件边界 | 运维者、实现者 |
| 决策与事实 | [`docs/adr/`](docs/adr/README.md)、[docs/hardware.md](docs/hardware.md)、[docs/glossary.md](docs/glossary.md) | 决策与其证据、负结果、硬件规格、术语 | 评审者、全体 |

| 文件与目录 | 回答的问题 |
|------|-----------|
| [README.md](README.md) | 这是什么、起步命令、实测到多少 |
| [docs/overview.md](docs/overview.md) | 系统由哪几块组成、边界在哪 |
| [docs/deployment.md](docs/deployment.md) | 怎么构建与启动、内存预算、哪些旋钮经过实测、怎么测量才不自欺 |
| [docs/requirements.md](docs/requirements.md) | 要满足什么（目标负载、指标、验收、非目标） |
| [docs/hardware.md](docs/hardware.md) | 两套测试环境的规格与实测值 |
| [docs/glossary.md](docs/glossary.md) | 术语唯一口径 |
| [`docs/adr/`](docs/adr/README.md) | 已作出的决策、依据的读数、因此砍掉的子系统 |
| [NOTICE](NOTICE) | 随库源码的来源、版本、许可与改动清单 |
| [src/](src/) | 实现（C++20），按目录分：`serve`（HTTP 双域、槽位调度、前缀缓存、遥测、请求记录、`web/` 前端）、`engine`（进程边界：引擎与编码器子进程、行协议）、`experts`（格式分派表、路由门控、三矩阵前馈、一层 MoE 串联、计算组与分层派发）、`dense`（稠密半边：门控残差、GDN 与 QSA 层、MTP 草稿与验证）、`kv`（KV 池）、`ple`（外挂表的行读取与块规格）、`storage`（页表、行缓存、专家缓存、共现统计）、`scheduling`（字节预算仲裁、内存计划、线程池）、`kernels`（各量化档的反量化与点积，CPU 侧加 `cuda/`）、`artifact`（GGUF 元数据读取）、`text`（分词器）。各档的格式事实与落点见 [ADR-008](docs/adr/ADR-008-implementation-stack-and-kernel-reuse.md) |
| [strata/](strata/) | 随库的参考引擎源码，由 `tools/build_engine.sh` 构建成 `build-engine/strata`；来源与改动见 [NOTICE](NOTICE) |
| [deps/httplib/](deps/httplib/) | 服务层唯一随库的单头依赖 |
| [tools/](tools/) | 构建脚本（`build_engine.sh`、`build_vision.sh`）与视觉编码器端口 |
| [tests/](tests/) | 回归测试（不使用外部测试框架，`ctest` 驱动）：`unit/` 单元、`integration/` 服务级、`frontend/` 前端脚本、`data/` 夹具与引擎转录 |
| [measure/](measure/) | 测量工具与 oracle 生成器（用法见第 7 节；其输出可复现，带时间戳的原始读数不入库）：机器画像与标定、PCIe 档位与带宽、内存与显存带宽、GGUF 元数据与张力清单读取、记忆表行局部性重放与语料构造、PLE 块真权重夹具、专家路由覆盖与缓存替换策略重放、投机 IO 放大、CPU 专家内核与量化 GEMM 与线程池微基准、运行中引擎的槽位/命中率/KV 驻留采样、NVMe 往返延迟、各量化档（IQ2_XS / IQ2_S / IQ2_XXS / IQ1_M / IQ3_S / IQ3_XXS / IQ4_NL / IQ4_XS / Q2_0 / Q6_K / Q8_0）反量化与点积的回归期望值生成、分词器逐 token 夹具、KV 精度档的输出一致率、视觉编码器的端口对拍与字节/质量 oracle、核显承担计算的证伪探针（负读数出处） |
| [scripts/docs_checks.py](scripts/docs_checks.py) 与 [.github/workflows/docs-check.yml](.github/workflows/docs-check.yml) | 规范自检：把第 2、3、4 节的规则变成可执行检查 |
| [CMakeLists.txt](CMakeLists.txt) | 构建入口 |
| [.github/workflows/build.yml](.github/workflows/build.yml) | 构建与测试（与文档自检并列的第二个 CI 作业） |

分工：硬件规格与实测数字只在 [docs/hardware.md](docs/hardware.md) 与 [docs/deployment.md](docs/deployment.md) 的实测表里，取舍与其证据只在 [`docs/adr/`](docs/adr/README.md)；其余文档引用它们，不复制数字。

设计期文档（引擎、接口、门禁、风险、候选方向，与来源编号台账）已退出工作树，决策证据由 [`docs/adr/`](docs/adr/README.md) 承接。需要当时的推导全文时从 tag 取出：`git show design-v0.1` 加上原相对路径，或 `git checkout design-v0.1 -- <路径>` 到临时目录。

新增文件准入：能并入现有文件的一律并入。新增文档须登记到 [README.md](README.md) 的文档表；新增任何文件都须登记到上表。

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

标记写法：写在数值或事实之后，用方括号，例如 `[环境2实测]`、`[官方]`、`[已确认]`。表格可用一行注释统一标注整表的等级。

仓库之外的来源就地写明出处：URL，或者足以定位的名称（「Intel ARK 的 Ultra 9 285K 页」「RTX 5060 Ti 厂商规格页」），再跟上方的等级标记。不维护来源编号台账。

设计时期登记过的 `S-N` 来源编号随台账退役，`engine §N`、`interfaces §N`、`spec §N` 这类章节号随设计文档退役。两者都只保留在既有 ADR 与既有源码注释里，作为「决策与实现当时依据什么」的历史指针，可从 tag `design-v0.1` 取回原文；新文档与新注释不得再新增这类编号，改写就地出处或 ADR 链接。

仓库内部引用一律使用相对链接加章节号，不写来源编号。

整表或整节的来源由表前一句或表格的「规格值」列承担；只有单一来源的数字才在句内标等级，避免行内堆砌标记。

---

## 4. 编号体系

| 前缀 | 用途 | 登记位置 | 可变性 |
|------|------|---------|--------|
| `ADR-NNN` | 架构决策与实测结论 | [docs/adr/README.md](docs/adr/README.md) | 不可修订；改变决策须新开 ADR 并标注取代关系 |
| `G-NN`、`R-NN`、`P-NN`、`S-N` | 门禁、风险、候选方向、来源编号 | 随设计台账退役 | 只出现在既有 ADR 与源码注释中，不再新增 |

新增 ADR 必须同时登记到索引；未登记的编号视为不存在。

---

## 5. 实测与评测流程

性能结论只能来自同题成对读数，流程固定如下。

1. 分母口径：同机、同量化档、同任务集、同并发。分母必须是环境2 上的实测，不得使用上游估算值。
2. 每次配置重复不少于 5 次，报中位数与区间，并记录当次机器画像。
3. 投机与预取类收益必须使用真实任务集，禁用随机 prompt。
4. 结果写入 [docs/deployment.md](docs/deployment.md) 的实测表，或写入使用该数字的 ADR 的「测量」段，并注明提示长度、卡数与引擎版本。
5. 假设不成立时砍掉对应子系统，负结果写入新开或既有的 ADR，不得删除、不得静默保留。
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
| 设计里程碑 | `design-v<主>.<次>` | 设计立场发生决策级变化：ADR 被取代，或实测砍掉一个子系统 | 标注当时的立场，供外部引用；设计台账退役后一般不再新增 |
| 实测快照 | `measure-<年-月>` | 每完成一轮基线或成对 A/B 实测 | 让当时写下的读数指向一个不可变的文档状态 |
| 代码版本 | `v<主>.<次>.<修订>` | 服务端到端可用、`ctest` 全绿时 | 配合 GitHub Release |

设计里程碑用两位号，代码版本才用三位：设计没有补丁级变化，三位号会产生无意义的精度。

### 6.3 发布

- 每个标签都可发一个 GitHub Release。正文写清该状态包含什么、支持哪些后端、实测口径（提示长度、卡数、引擎版本）、已知缺口。
- 不维护 `CHANGELOG.md`：变更记录由提交信息、标签与 Release 说明承担（文件内不写变更日志，见第 2 节）。

### 6.4 提交与 PR

- 提交信息首行 `<type>(<scope>): <祈使句摘要>`，正文写清为什么改、依据哪条 ADR 或哪组成对读数。
 类型：`feat`、`fix`、`perf`、`refactor`、`test`、`build`、`chore`、`docs`、`adr`、`revert`；作用域用 `(serve)`、`(web)`、`(engine)`、`(experts)`、`(tree)` 一类目录名。
- 一次提交只做一件事；纯格式整理与内容修改分开提交。
- 已推送的历史不得重写，纠错用新提交。`amend` 只允许用于尚未推送的提交。
- PR 自检清单：新论断有出处？数字带口径？新术语已登记？新编号已登记？没有第 2 节列出的禁用写法？
- 未经明确授权不得 push 或改动远端。

---

## 7. 代码

代码分两类。

规范自检，随 CI 执行：[scripts/docs_checks.py](scripts/docs_checks.py)，Python 3 标准库，无外部依赖，无构建步骤，运行方式 `python3 scripts/docs_checks.py`，由 [.github/workflows/docs-check.yml](.github/workflows/docs-check.yml) 在推送与 PR 时执行。任一检查失败即非零退出，本地与 CI 使用同一套判据。

测量工具，手工执行、不进 CI。分四类：`measure/` 下的 Python 3 脚本（多数零外部依赖，`mem_bw.py` 与各方差夹具生成器依赖 numpy，涉及分词或引擎轨迹的还要参考引擎）、`measure/pcie_bw.cu`、`measure/io_roundtrip.cu`、`measure/vram_bw.cu`（CUDA C++，用 nvcc 编译）、`measure/cpu_expert_bench.c`、`measure/cpu_gemm_bench.c`、`measure/igpu_probe.c`（纯 C，用 `cc` 编译）、`measure/` 下四个 C++ 基准（由 CMake 构建成 `build/` 里的可执行文件，不注册进 `ctest`）。前三类没有构建系统，直接调用编译器或解释器。

- `python3 measure/calibrate.py [--set KEY=VALUE ...]`：机器画像标定。把已测的量与模型几何代入，按显式公式推导每步 PCIe 预算、专家槽位与 KV 窗口的常驻上限，落盘到 `measure/results/*-calibrate-<平台>.json`。默认值即环境2 的实测值，`--set` 可覆盖；`--selftest` 核对推导恒等式。零依赖。
- `nvcc -O2 -o build/vram_bw measure/vram_bw.cu && ./build/vram_bw --sizes 32,128,512`：显存可用量与带宽（device STREAM），画像的第 1 项输入。产物写在 `build/`。
- `python3 measure/env_profile.py`：记录机器画像的静态部分。
- `python3 measure/ple_row_oracle.py --model <shard2.gguf> --rows 0,12345`：从 GGUF 里取记忆表的指定行，输出原始字节与按 IQ4_NL 规则反量化的结果，用作内核回归的 oracle。零依赖；它的输出与本仓库记录过的引擎自带 oracle 向量逐位吻合，可作交叉验证。
- `python3 measure/pcie_link.py [--watch 秒数]`：读 PCIe 链路档位。加 `--watch` 可在同时施加负载时观察档位是否变化。
- `python3 measure/mem_bw.py [--size-mb 256] [--procs 8] [--seconds 0.6] [--repeats 5]`：测内存带宽。聚合口径是「各进程屏障同步后在同一时间窗内搬运的字节之和除以时间窗」；不得用各进程中位数相加，那样在进程启动不同步时会虚高，甚至超过内存理论峰值。
- `nvcc -O2 -o build/pcie_bw measure/pcie_bw.cu && ./build/pcie_bw --repeats 5 --out-dir measure/results`：测 PCIe 有效带宽。产物写在 `build/`，该目录已在 `.gitignore` 中，不入库。`--hold 秒数` 会保持链路流量，便于同时用 `pcie_link.py` 观察宽度与代数是否变化。注意链路**代数会随负载降档**（空闲可低至 gen1），因此"当前档位"必须在持续负载下才可引用。
- `python3 measure/gguf_meta.py --model <file.gguf> [--tensors] [--all]`：直读 GGUF 元数据与张力维度，用于取模型几何（头数、维度、层数、专家数、表形状），不必依赖 HuggingFace 的 `config.json`。零依赖，可经 `ssh host "python3 - --model …" < measure/gguf_meta.py` 在远端的模型文件上直接运行。
- `python3 measure/ple_locality.py --tokens <token_ids.txt> [--env 环境2]`：离线重放记忆表行的访问局部性；该重放否定了表行预取（[ADR-007](docs/adr/ADR-007-ngram-table-on-demand-read.md)）。输入是分词后的 token id 序列（空白分隔，空行表示序列边界），输出行分布、重复率、去重后顺序性与行缓存命中率，并落盘 JSON。`--selftest` 把重放实现与参考引擎自带的 oracle 向量逐位比对。零依赖；与机器无关，分词需在被测环境上由该引擎的 tokenizer 完成。
- `python3 measure/make_ple_corpus.py --engine-dir <引擎目录> --native-gguf <分片.gguf> --out <目录>`：构造局部性重放所需的语料（token id 序列，每份记录 token 数与 sha256）。需在参考引擎环境内运行，因为它调用该引擎自带的 tokenizer（依赖其 `regex`），故不属零依赖工具；重放侧仍由 `ple_locality.py` 完成。
- `python3 measure/expert_coverage.py --trace <dump-routing 轨迹>... [--profile <画像.bin>]`：统计路由轨迹里的 (层, 专家) 激活频次并给出 top-N 覆盖曲线；`--profile` 时另把引擎画像的排名当频次曲线用，量化该代用造成的偏差。轨迹由参考引擎的 `--dump-routing` 生成（格式见其 `tools/make_profile.py`）。零依赖。
- `python3 measure/spec_io_amplify.py --log <引擎日志>... [--trace <轨迹>...]`：从日志的 `speculation` 行取接受率并取倒数，另按轨迹的记录数 ÷ 层数 ÷ 生成 token 数直接数出位置放大。零依赖。
- `cc -O2 -mavx2 -mfma -pthread -o build/cpu_expert_bench measure/cpu_expert_bench.c && ./build/cpu_expert_bench --size-mib 512 --threads 1,2,4,8,12,16,24`：在同一字节量上比较纯流式读、4 位码本反量化+FMA、2 位码本反量化+FMA 三个臂的吞吐，判定未命中专家路径是算力瓶颈还是带宽瓶颈。产物写在 `build/`（已 gitignore）。
- `cc -O2 -mavx2 -mfma -pthread -o build/cpu_gemm_bench measure/cpu_gemm_bench.c && ./build/cpu_gemm_bench --bits 4 --experts 48 --pin 0-14`：按专家真实几何跑码本反量化 GEMV，用 `--pin` 把线程绑到指定 CPU（环境2 上 P 核 0-7、E 核 8-23），量出每类核与各池规模的吞吐。产物写在 `build/`。
- `python3 measure/engine_cache_probe.py --engine-dir <引擎目录> --base-config <serve 配置> --prompt <提示.txt> --slots auto|N --out <记录.json>`：起一次 serve 实例、发真实提示的 greedy 请求、读 `/metrics` 与引擎日志、停服务，产出该配置的专家槽位、解码命中率、PCIe 占比、KV 驻留 decode 吞吐。`--repeat N` 在同一实例内连发 N 次并给出中位数与区间（基线口径要求 ≥5 次，故基线必须用 `--repeat`，不能靠多次重启）；`--poll-ms N` 轮询 `/metrics` 的 `live.tok_s` 以给出 decode 步长分布；`--gpu LIST` `--drop-arg <旗标>` 用来改机器形态（如单卡：`--gpu 0 --drop-arg=--layer-split`）。`--from-raw` 可把一份原始记录精简重写（幂等）。需在装有该引擎的环境上运行。
- `nvcc -O2 -o build/io_roundtrip measure/io_roundtrip.cu && ./build/io_roundtrip --file <大文件> --mode direct --lane ample`：随机取 16 个 4 KiB 页串行读再一次 H2D，给出读腿与整条往返的 P50 / P99；`--mode cached` 走页缓存，`--hold-gib N` 先占住 N GiB 造出内存紧张的 lane。产物写在 `build/`。
- `python3 measure/kv_quality.py --run fp16=<log> --run int8=<log> --run k8v4=<log> --run q4_0=<log> --baseline fp16`：top-1 一致率，比对同一提示、同一采样下不同 KV 精度档的贪心输出序列（一致率 = 公共前缀 ÷ 较短长度）。输入是引擎 `generate` 的日志；比对时必须带一条**同配置的对照 run**，并给引擎加 `--adapt-every 0`，否则自适应换入的舍入差异会与精度差异混在一起。零依赖。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq1m_oracle.py --model <分片1.gguf> --tensor blk.8.ffn_gate_exps.weight --ggml-common <检出>/ggml/src/ggml-common.h --blocks 16 --emit-inc`：IQ1_M 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ1_M.dequantize_blocks`；格点表的核对是值级的（ggml 存有符号字节、gguf-py 压成每字节 2 bit，两边表示不同，故比的是解码后的值）。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2xxs_oracle.py --model <分片1.gguf> --tensor blk.1.ffn_gate_exps.weight --ggml-common <检出>/ggml/src/ggml-common.h --blocks 16 --emit-inc`：IQ2_XXS 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ2_XXS.dequantize_blocks`（独立实现路径），并顺带核 256 项格点表在 ggml C 源与 gguf-py 打包之间逐字节一致、按 66 字节/块反推的整张量字节数与引擎侧 `native_experts.txt` 的逐专家字节对齐（gu_type = 16 的层每专家 gate 为 422,400 字节）。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq4nl_oracle.py --model <IQ3_XXS 分片1.gguf> --tensor blk.0.ffn_down_exps.weight --blocks 16 --emit-inc`：IQ4_NL 作**通用权重块**的真字节夹具生成器（记忆表表行那条路径的夹具见 `measure/ple_row_oracle.py`）。期望值取自 gguf-py 的 `IQ4_NL.dequantize_blocks`（独立实现路径），并核按 18 字节/块反推的整张量字节数。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2s_oracle.py --model <分片1.gguf> --ggml-common <检出>/ggml/src/ggml-common.h --tensor blk.0.ffn_gate_exps.weight --blocks 16 --emit-inc`：IQ2_S 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ2_S.dequantize_blocks`（独立实现路径），并顺带核两件事：1024 项格点表在 ggml C 源与 gguf-py 打包之间逐字节一致、按 82 字节/块反推的整张量字节数与引擎侧 `native_experts.txt` 的逐专家字节对齐。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `python3 measure/q2_0_oracle.py --model <分片1.gguf> [--tensor blk.0.ffn_down_exps.weight] [--blocks 64] [--emit-inc]`：从真实 GGUF 取一个 Q2_0 张量的前几块做夹具（结构不变量 + sum/sumabs + 前 16 值），并按张量字节数反推 `QK2_0` 的取值（64 与 128 两种假设的 bpw 不同）。同一窗口可在引擎自身的 `strata-dequant` 上取 CHECKSUM 交叉核对。零依赖，需真模型文件。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq2xs_oracle.py --ggml-common <检出>/ggml/src/ggml-common.h --emit-inc|--emit-q8k-inc|--emit-dot-inc`：专家点积路径的回归期望值生成器。三种模式各写一份 `.inc`：`--emit-inc` 给 IQ2_XS 单块反量化（期望值取自 gguf-py 的独立实现）；`--emit-q8k-inc` 给 Q8_K 参考激活量化器；`--emit-dot-inc` 给 IQ2_XS × Q8_K 定点点积（后两者的期望值是 ggml 对应函数的忠实转写）。三种模式都先把 ggml 的 `iq2xs_grid` 从 C 源抽出来与该库的打包表示逐字节核对。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py，故不属零依赖工具。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq3s_oracle.py --model <分片1.gguf> [--tensor blk.0.ffn_gate_shexp.weight] [--blocks 16] [--emit-inc]`：IQ3_S 的真字节夹具生成器，覆盖共享专家 gate/up 的那几层。期望值取自 gguf-py 的 `IQ3_S.dequantize_blocks`；结构不变量是格点幅度字节只能取 8 个奇数之一，故 `|value| ÷ (d(2n+1))` 必须落在那八个值里。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq3xxs_oracle.py --model <分片1.gguf> [--tensor blk.35.ffn_gate_exps.weight] [--ggml-common <检出>/ggml/src/ggml-common.h] [--blocks 16] [--emit-inc]`：IQ3_XXS 的真字节夹具生成器，覆盖另一档模型 gate/up 的六层。期望值取自 gguf-py 的 `IQ3_XXS.dequantize_blocks`；带 `--ggml-common` 时那组固定幅度值由上游 C 表推出并顺带核对 256 项格点表，不带时用内建的同一组。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py。
- `PYTHONPATH=<检出>/gguf-py python3 measure/iq4xs_oracle.py --model <分片1.gguf> [--tensor blk.0.ffn_up_shexp.weight] [--blocks 16] [--emit-inc]`：IQ4_XS 的真字节夹具生成器。期望值取自 gguf-py 的 `IQ4_XS.dequantize_blocks`；结构不变量是 `|value| ÷ |d(ls-32)|` 必须落在那 16 个码本值的绝对值之一。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py。
- `PYTHONPATH=<检出>/gguf-py python3 measure/q6k_oracle.py --model <分片1.gguf> [--tensor blk.1.ffn_gate_shexp.weight] [--blocks 16] [--emit-inc]`：Q6_K 的真字节夹具生成器。期望值取自 gguf-py 的 `Q6_K.dequantize_blocks`；结构不变量是 `value ÷ (d × sc)` 必须是 `[-32, 31]` 内的整数，这条能独立抓出子尺度取错位置与高低位拼接错两类错。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py。
- `PYTHONPATH=<检出>/gguf-py python3 measure/q8_0_oracle.py --model <分片1.gguf> [--tensor blk.0.ffn_down_shexp.weight] [--blocks 16] [--emit-inc]`：Q8_0 权重侧的真字节夹具生成器，对应共享专家 down 投影那一档。该张量是二维的（共享专家只有一个，没有专家维），取维度时按此写死口径；期望值取自 gguf-py 的 `Q8_0.dequantize_blocks`，结构不变量是 `value ÷ d` 必须是 `[-127, 127]` 内的整数。产出已入库；重生成时整份覆盖。依赖 numpy 与 gguf-py。
- `python3 measure/ple_block_oracle.py --model <分片1.gguf> [--layer 1] [--out <夹具.bin>]`：PLE 块的真权重逐段 oracle，自读 GGUF 头与六个张力后按同一公式的另一种写法（numpy 向量化、按 `[hc, n_embd]` 重塑）重算，写成二进制夹具供 `./build/test_ple_block --model <分片1.gguf> [--layer N] --oracle <夹具>` 对照。参考引擎那份 parity 用的是它自带的 pack，与本仓库两份 GGUF 不是同一份权重，逐段数搬不过来，故这里的期望值是第二实现而不是更权威的一侧：转置与错位读法在两种写法下必然分叉。两处刻意的差别——投影累加在 float64 上做（乘积仍按 bf16 舍入），故与 C++ 侧的差只有求和顺序，量级 1e-6；卷积段的相对差对着项量（约 0.7）而不是对着结果（约 0.009）量，四项相消会把误差放大两个数量级。依赖 numpy。
- `python3 measure/tokenizer_oracle.py --tools <引擎 tools 目录> --gguf <分片1.gguf> --out <夹具.bin>`：用参考引擎自带的那份分词器生成逐 token 的对照夹具，供 `./build/test_tokenizer --model <分片1.gguf> --oracle <夹具>` 使用。判据是逐 token 相等，不是字节往返：预分词的 `\p{M}` 那条、特殊 token 的两类区别、字节映射的顺移顺序只有在逐 token 这一级才看得见。需在装有该引擎的环境上运行（它依赖引擎的 `regex` 模块），故不属零依赖工具。
- `python3 measure/gen_unicode_ranges.py > src/text/unicode_ranges.inc`：生成 C++ 侧的 Unicode 分类表（L、M、N、空白）。口径由分词 oracle 自己用的 `regex` 库扫全码点得出，逐类合并成区间，使「哪一段算字母」与 oracle 逐条一致，而不是靠某个 Unicode 版本的记忆。需在被测环境的引擎侧运行。
- `python3 measure/expert_cache_sweep.py --bin build/expert_cache_replay --trace <轨迹.bin> --trace <轨迹2.bin> --out <记录.json> [--split 0.5]`：把 `expert_cache_replay` 的多种（槽位, 相联度）组合跑成一份可对比的记录，`--trace` 可重复给。口径边界：重放本身是确定性的纯计算，跑在哪台机器不改变结果，但轨迹来自被测环境参考引擎的 `--dump-routing`（解码路径）；这里的命中率是同一条轨迹内部、缓存已热身的自适应命中率，与全量频次排名的全局静态覆盖不是一个口径，两者不可混用。共现表种类上限取内核默认（2^18），衰减节拍 2^20 次观察。
- `cmake --build build --target expert_cache_replay && ./build/expert_cache_replay --trace <轨迹.bin> [--slots N] [--ways W] [--json]`：把参考引擎 `--dump-routing` 的轨迹灌进三种替换策略比命中率，并扫槽位与相联度两个旋钮。轨迹是逐条定长记录：`layer(i32) k(i32)`、`k×专家id(i32)`、`k×权重(f32)`，一条等于「一层在一个位置上选的 k 个专家」；只有解码路径写轨迹，批量预填不写。产出的是同一批轨迹下策略之间的相对比较，命中率本身只作观测量。
- `cmake --build build --target gguf_tensors && ./build/gguf_tensors <分片.gguf> [名字子串] [--limit N]`：列出一个 GGUF 里的张力名、维度、类型码、块几何与字节数，用于在实现某个稠密或注意力部件之前先取出真模型的张力清单与档位。不带子串时只打印按类型汇总的条目数与字节数。类型码名表取自 ggml 的 `GGML_QUANT_SIZES`，只列本仓库用得到的档。
- `cmake --build build --target thread_pool_bench && ./build/thread_pool_bench [最大线程数]`：在同一组线程数下一起量五件事——纯流式读的带宽随线程数（主机内存子系统的天花板）、一次分派的固定开销、真实 GEMV 的伸缩（每 GMAC 与每字节的实际代价）、MoE 一层的调用粒度（十个小矩阵各一次 对 并成一个大矩阵一次，gate/up 与 down 两种形状）、提交之间夹一段串行工作时每次提交多付的唤醒代价。分开量才能判断整步并行只拿到约 1.8 倍是撞带宽、分派开销还是内核不伸缩。所有数字都标注机器，工具不写死线程数。
- `cmake --build build --target igpu_gemv_bench && ./build/igpu_gemv_bench`：把核显的三档 gate/up GEMV 着色器与主机内核放在部署那份 pack 的真实批量形状上比，四个臂同字节：单线程主机内核（单作业延迟）、多线程主机内核（引擎 CPU 池的代价）、权重常驻的设备派发（纯算力下限）、逐矩阵上传加末尾一次下载（端到端）。这是核显承担专家计算那条假设的证伪点，负读数见 [ADR-019](docs/adr/ADR-019-igpu-not-for-experts.md) 与 [ADR-020](docs/adr/ADR-020-strata-igpu-pool.md)；本项目的核显只做视觉编码器。
- `cc -O2 -pthread -o build/igpu_probe measure/igpu_probe.c -ldl && ./build/igpu_probe [--json] [--list] [--igpu-mib N] [--cpu-mib N] [--rounds N] [--cpu-threads N]`：核显与 CPU 共用一条内存总线时的带宽争用探针，同时报设备接口与运行时现状。走 `vkCmdCopyBuffer` 的传输路径而不是计算路径，因为被测机的核显没有任何计算运行时；Vulkan 入口用 `dlopen` 取，结构体 ABI 手写，不需要 `libvulkan-dev` 与着色器工具链。CPU 对照臂与 `cpu_expert_bench.c` 同形：缓冲按线程切成互不重叠的连续区段、屏障齐发、固定遍数——区段不重叠是要紧的，八个线程同扫一段会命中同一批 cache line，聚合被共享放大成超过平台上限的假值。负读数见 [ADR-022](docs/adr/ADR-022-igpu-cannot-take-cpu-decode-share.md)。
- `python3 measure/vision_port_check.py [--emit]`：把 `tools/vision/` 那份视觉编码器端口与上游同标签的对应文件逐字节对拍，只放行清单里登记过的分歧；退出码 0 表示「端口 = 上游 + 声明的规则」。`--emit` 重新派生端口文本，也是上游钉版移动时的重派生入口。需上游检出，见 [ADR-013](docs/adr/ADR-013-vision-port-and-dependency-boundary.md)。
- `systemd-run --user --scope --property=MemoryMax=16G -- python3 measure/vision_oracle.py [--bin <端口二进制>] [--gpu]`：逐字节 oracle。两边二进制出自同一份物化后的钉版 llama.cpp 树、同一组 CMake 旗标，编同一张确定性探针图、同一 `--threads`，剩下的唯一差别就是端口本身，故输出必须全等；不一致即说明有分歧绕过 `vision_port_check.py`，或该线程数下编码器其实不确定——后者脚本会自己说出来（同一份输入跑两遍对比）。指向 39 GB 分片时须在限内存 scope 内跑。
- `python3 measure/vision_smoke.py [--bin <端口二进制>] [--mmproj <文件>] [--gpu] [--threads N]`：构建门禁——用真 mmproj 编一张真图，检查落出的 SVE1 文件的头部与长度，OK 一行加一份 JSON 记录。指向大分片时同样建议在限内存 scope 里跑。
- `python3 measure/vision_quality.py run --label <臂名> --repeats 2 [--max-tokens N] [--port 8080]`：视觉臂之间的质量 A/B，走 [ADR-009](docs/adr/ADR-009-correctness-acceptance-criteria.md) 的默认真实配置档（分布级）。字节 oracle 服务的是固定开关的构建，而 ggml-vulkan 臂在 fp16 累加上按设计就会分叉，所以它的质量主张要按 `--kv int8` 那次的论证方式：同一批图与提示经运行中的服务、贪心解码、长生成，报每张图的首个分歧位置与一致率，另给每个臂自身的完全可重复率。图由 ffmpeg lavfi 渲染，刻意取确定性，重跑能复现自己那一侧的字节，答案漂移才归因于编码器。
- `measure/vision_common.py`：上面几个视觉脚本共用的库——一张确定性探针图与一次编码往返，不提供命令行入口。

测量结果的存放：原始记录写入 `measure/results/<时间戳>-<测点>-<平台>.json`（Python 脚本与 CUDA 程序都自己落盘）；摘要回填 [docs/deployment.md](docs/deployment.md) 的实测旋钮表与 [docs/hardware.md](docs/hardware.md) 的实测值表，或写入使用该数字的 ADR。每条数值必须标注平台：环境1-WSL、环境1-Windows 或环境2。两套环境对比时必须使用同一份脚本与同一组参数。

平台标签：脚本能自动区分环境1-WSL 与环境1-Windows，但**在原生 Linux 上无法判断是环境2 还是别的机器，必须显式传 `--env 环境2`**，否则记录会落入「原生Linux(请确认是否为环境2)」这一未确认标签，不能作为环境2 的数据引用。

引擎实现的语言、构建与测试口径（[ADR-008](docs/adr/ADR-008-implementation-stack-and-kernel-reuse.md)、[ADR-009](docs/adr/ADR-009-correctness-acceptance-criteria.md)）：

- 语言 C++20，构建 CMake（≥3.28），不引包管理器；依赖面为零（随库内核以裁剪后的子集入库，文件头登记上游路径、提交号与许可，见 [NOTICE](NOTICE)）。
- GPU 侧用 CUDA，但主机侧代码不得依赖它：CI 在没有 GPU 的 runner 上编译并跑测试。CUDA 部分单独标注、在本机跑。
- 测试不引外部框架：`tests/` 下的可执行文件用 [`tests/unit/check.hpp`](tests/unit/check.hpp) 的 `CHECK`，`ctest` 驱动，CI 与本地同一条命令——`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure`。**不要用 `<cassert>` 的 `assert`**：Release 构建定义 `NDEBUG`，它会把这些检查整个编译掉，测试于是变成空跑，「全绿」不再意味着任何事。
- 测试范围：主机侧的不变量与确定性算术优先（页表不变量、预算仲裁、标定推导）。涉及 GPU 内核的回归走 [ADR-009](docs/adr/ADR-009-correctness-acceptance-criteria.md) 的两档口径（固定开关下逐位；默认真实配置用分布级）。
- 需要真实模型文件才能跑的验收不进 CI：CI 用现场生成的最小合成夹具（如 `test_gguf_table` 自造一个只含一张表的小 GGUF）。要用真文件验收时走该测试的手工入口，例如 `./build/test_gguf_table --model <分片2.gguf> --row 0`，其输出与 `measure/ple_row_oracle.py` 对照。

---

## 8. 许可与随库内容

代码与文档采用 MIT，全文见 [LICENSE](LICENSE)。

随库的其它来源逐条登记在 [NOTICE](NOTICE)：上游仓库、版本或提交号、许可、本地取舍与改动。入库文件保留上游版权头，不删不改。

引用外部数据、图表、代码片段时，在正文就地写明出处与许可状态（第 3 节）。

引擎换代是整树替换加一条 ADR（当前是 [ADR-023](docs/adr/ADR-023-strata-engine-upgrade-01441.md)），不在随库树里留长期分叉。

---

## 9. 维护者与当前优先事项

维护者负责合并 PR、裁决编号分配与 ADR 的接受状态。重大决策写入 ADR，不写入提交信息。

当前优先事项，按顺序：

1. `native` 后端的输出质量验收：达到 [ADR-009](docs/adr/ADR-009-correctness-acceptance-criteria.md) 的口径，否则它只是能跑，不是已验收。
2. `inhouse` 后端的设备侧那半（批式前向与 KV 的设备实现）与 MTP 真草稿头；主机侧路径已与独立实现对照过。
3. 在「262K 上下文 × 高比特档」这一验收场景上给出成对读数；引擎换代后分母须按第 5 节重测。
4. 多模态目前只有图像：视频输入未接入。
