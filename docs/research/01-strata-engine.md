# Strata 引擎深度拆解

> 研究对象：`github.com/Niko1221/Strata`。本文回答一个具体问题：一个 125B 的 MoE 模型，怎么在 12–16GB 显存的消费级卡上跑到可用速度？
> 结论先行：Strata 没有让 125B 装进 12GB，而是让「装不下」不再等于「每层都等 IO」。它的全部设计都围绕这一句话展开。

复核状态标记：[已确认] 已开原文逐位确认 · [摘要级] 仅见标题或摘要 · [存疑] 来源矛盾已降级。本机数值标注引擎版本。

---

## 1. 项目事实

| 项目 | 事实 | 来源 |
|------|------|------|
| 仓库 | `github.com/Niko1221/Strata` | [官方] |
| 语言 / 许可 | C++ / MIT | [官方] |
| 创建时间 | 2026-09-24（取数时点：18,383 stars、1,637 forks、472 issues） | [官方] |
| 定位原话 | "Qwen3.8-Flash-Next on any consumer hardware: one-click install for Windows / Linux. Strata inference engine, OpenAI/Anthropic API on localhost, optional image input." | [官方] |
| 血统 | 构建含 `third_party/ggml`（llama.cpp 的血统），署名 ideas from Splash / ninfer / HyperQwen | [官方] |
| 默认端口 | Web 8080 | [官方] |
| 部署门槛 | ≥12GB 显存 + ≥32GB 内存 + 约 80GB 磁盘（SSD 首启快得多）；下载约 70GB，启动加载 35–55GB 进 RAM 并锁住一部分 | [官方] |

一句话定位：它复用了 GGUF 生态的词表与量化内核（ggml），但在权重驻留策略上做了一件 llama.cpp 没做的事——见第 3 节。

---

## 2. 它要解决的瓶颈

问题不是算力，是容量：

- 主模型 FP8 检查点 172.78 GiB、BF16 335.28 GiB（[官方]口径）——任何单张消费卡都装不下。
- n-gram 外挂表 51B：FP8 约 51GB、4-bit 约 28.8GB（[官方]）
- 于是 llama.cpp 一类「全量常驻」引擎只能退化到层级 offload：每层把权重经 PCIe 搬回 GPU 再算。PCIe 5.0 x8 只有约 31.5GB/s（低于内存带宽），于是每层都在等搬运。

Strata 的攻击点正是「每层都在等搬运」这个 IO 循环。

---

## 3. 三级分层存储

| 层 | 放什么 | 依据 |
|----|--------|------|
| VRAM | 每个 token 必用的稠密部分：注意力与 DeltaNet mixer、gated-residual 权重、router、共享专家、输出头、MTP 草稿层、KV（≥64K 时只常驻最常读的部分，其余从 RAM 流入）；剩余显存全部用作专家缓存 | [官方] `docs/HOW_IT_WORKS.md` |
| RAM | 全部 24,576 个专家 pinned 锁页常驻；CPU 用 AVX-512 / AVX2 内核就地计算未命中专家 | [官方] |
| SSD | 28.8 GB n-gram 表，每 token 只读若干小行，走 OS page cache | [官方] |

关键设计取向：显存不屯权重，屯「最热的专家」。VRAM 剩余多少决定能常驻多少专家，直接决定命中率，进而决定速度。

---

## 4. 专家缓存与命中率模型

- 启动时用预制热度排序（24,576 个专家的 profile 文件）装入最热的一批（社区配置里约 4,500 个），会话中持续学习、动态换入换出。
- 官方换算原话："every extra GB holds ~700 more experts"（每多 1GB 显存约多常驻 700 个专家）（[官方]）
- 双向印证：本机一手 `11,967 专家 / 16,416 MiB ≈ 729/GB`（2026-10-09 双卡合计；单卡期曾记 `5,269 / 7,235 MiB ≈ 746/GB`）；社区 `11,466 / 16.65GB ≈ 688/GB`。容量比显卡新旧更重要：24GB 卡比 12GB 快约一倍。
- 官方给出「12GB 卡命中率约 72%、双卡合计 44GB 可到 97.7%」的经验值（[他卡实测]）。

命中率是这个引擎的第一性能变量。它决定了每 token 有多少专家要走慢路径。

---

## 5. 未命中专家就地计算

这是与 llama.cpp 的根本区别：

- 路由选中不在显存的专家时，CPU 直接在内存里算完（AVX2/AVX-512 内核），与 GPU 上命中的专家并行，最后 scatter-add 合并。
- 效果：把 PCIe 从关键路径上摘掉了。代价是 CPU 算力成为新上限——本机 Arrow Lake 无 AVX-512，只能走 AVX2。
- 但并非「完全不碰 PCIe」：`--pcie-frac` 的官方定义是「未命中专家里被拷到 GPU 算、而不交给 CPU 算的那部分比例」。取大=走快链路、取小=省 PCIe 给 CPU 算。`--pcie-frac 0` 会掉 decode 速度；prefill 阶段官方明确「experts streamed to the GPU over PCIe」。这推翻了早期二手文「CPU 就地算 = 完全不碰 PCIe」的说法。

---

## 6. 投机解码与取权重成本

- MTP 草稿层最多起草 3 个 token，主模型一次 48 层前向并行验证，平均每步产出 2.4–3.2 token，整体 1.6–1.8×（[官方]）
- 另有 prompt lookup / suffix 起草最多 5 token（引擎 0.1.7 起），只在实测划算是启用（代码改写类 6–11%，其它文本基本不变）。
- 本质是把最贵的专家读取摊到 2.4–3.2 个 token 上（近 3× 摊薄），且输出分布不变（草稿只猜、主模型逐 token 定夺）。
- 为什么 llama.cpp 开 MTP 也救不了：它的草稿层同为 MoE、权重同样在显存外，猜词本身就贵，验证一遍仍要逐层读专家。

---

## 7. KV 与长上下文

- KV 量化到 int8（本机 `kv: int8`，[本机实测]）。
- `--kv-resident 32768`：最热 32K token 段常驻显存，其余从内存流式读（社区实测读命中显存 96.9%，[他卡实测]）。
- `--prefill auto`：每块最多 8,192 token（32,768 需显式开，[官方]文档）→ 这是官方「长文读取 >1,000 tok/s」的来源，也是 prefill 与权重加载重叠的手段。
- WSL 约束：`--kv-streaming on|off` 官方注明 "never under WSL, which cannot stream"——在 WSL 里跑拿不到 `--kv-resident` 这条红利。本机实例 `kv_resident 32768` 生效，说明它并非跑在 WSL 内。

### 7.1 上下文复用：并发各自独立，跨轮/跨请求有复用（[官方] DETAILS.md 与 BATCHING.md）

- 并发槽位是独立会话：一个 batch 槽位 = 一份自己的 session（GDN 递推、QSA 的 K/V 与索引器、PLE 历史），并发会话之间不共享活 KV。
- 跨时间/跨请求的复用有三套机制：
  - 检查点（conversation cache）：续聊只读「引擎已持有部分之后」的尾巴。检查点最多 6 个、各约 118 MB，在每个新 assistant 轮开始与每 16K prompt token 处拍；只有 prompt 逐 token（含图片）完全匹配时才用。最老的那个检查点（通常 = system prompt 末尾，同一客户端每个对话都共享）永久保留 → 新对话若共享该前缀，就从它之后开始读，而不是从 token 0。
  - 槽位即对话缓存：跑完的槽位保留其内容；该对话下一轮回到这个槽位，引擎把状态拷回来（短会话约 50–60 ms）而非重读历史。
  - Conversation parking（`--conversation-cache-mib N --conversation-cache-slots K`）：把最多 K 个对话「停」在 RAM，之后在任一 stage 恢复；官方实测 9,276 token 的对话 51 ms 恢复，续写 token 与从未离开一致。默认预算 0（关）。
  - 固定共享前缀（`"strata_prefix"`，0.1.40.2，需 `--prompt-cache 3`+）：把「一长文档 + 多问题」的文档末尾钉成检查点 → 20 个问题只读一遍文档。一次只允许 1 个 pinned prefix。
- 复用直接省时间：本机一手 `reused / prompt_tokens ≈ 85%`（例：`reused 1,172,944 / 1,373,844`）→ prefill 只读尾巴（多为 1–2 s，而非十几万 token 的分钟级）。`/metrics` 甚至有 `vllm:prefix_cache_queries_total` / `_hits_total` 口径（沿用 vLLM 的指标命名）。
- 参数：`--prompt-cache N`（0=关）、`--prompt-cache-every N`、`--prompt-cache-root N`（0=无 system-prompt 检查点）、`--turn-token ID`。

---

## 8. 启动标定（`--calibrate`，引擎 0.1.19 起）

官方「按 PC 调」的参数组，[官方]：

| 参数 | 含义 |
|------|------|
| `--pcie-frac` | 未命中专家走 GPU 的比例（第 5 节） |
| `--spec-min-p` | 草稿层置信门槛，够了才多猜一个（本机 0.50） |
| `--pool-workers` | CPU 线程数；有能效核 E-core 时线程更少反而更快 |
| `--adapt-every / --adapt-swaps / --adapt-decay` | VRAM 专家缓存跟随会话的积极程度；CPU 慢的机器多 swap 更划算（官方给出 DDR3 Xeon + 4060 Ti PCIe3.0 x8 上 +7.9% 的案例） |

含义：「命中率自适应 / 共现放置 / 投机长度随命中率收缩」这些优化，上游已实现为旋钮——本设计不必从零实现，但可以做「自动调参」。

---

## 9. 多卡与并发（[官方] MULTI_GPU.md 与 BATCHING.md；[本机实测] 见 §9.3）

> 这一节同时回答两个问题：Strata 怎么用多张卡，以及为什么多个请求会排队等。

### 9.1 多卡机制：pipeline（按层切分），不是张量并行

- 一张模型跨 2–3 张 NVIDIA 卡：层切成连续区间，每卡跑自己那段；最后一张卡还跑输出头 + MTP 草稿层。每卡只为自己那几层维护专家缓存 → 两卡的显存专家容量约翻倍。
- 是 pipeline（层）并行：一个 token 每个 verify window 跨卡一次（几百 KB，走 pinned RAM），不是每层跨两次 → 不需要 NVLink / P2P，x4 甚至 x1 槽也能用。`--pcie-frac` 若显式给出则是「每卡同一比例」，暂无按卡设置。
- 卡顺序（`auto`）：更快的卡放最后（末段跑 head / 草稿 / verify，prompt chunk 要等它）；等速卡保持你给的顺序。
- 配套开关：`--layer-split K1[,K2..]|auto`、`--split-device`、`--trim-stage-weights`（每卡只加载自己层的稠密权重，省下的 VRAM 归专家缓存）、`--vram-reserve-later-mib`。
- `--pipeline-windows 2`（可选）：第一张卡在末卡还在 verify 本 window 时就开下一个 window；下个 window 是猜的（猜本 window 全接受、且 bonus token 是草稿层预测那个）。命中→下个 window 已做一半；不中→回滚。需要「恰好两卡两段 + `--serve`」，且与 `--batch` 槽位互斥。

### 9.2 并发机制：`--batch N`（默认一次只服务一个请求）

- 默认 Strata 一次只答一个请求，其余在服务端队列里等（[官方] README + BATCHING.md 原话）。
- `"parallel": N` / `--batch N` / `--slots N`（常规 2..8）：最多 N 个会话同时解码，每个 verify window 每个会话带一个 token → 稠密权重、共享专家、head、以及会话间共用的路由专家每 window 只读一次。可选开关（#465，引擎侧 PR #559）。
- 双卡 + `--batch-groups G`：N 个槽位分 G 组在 GPU 间流水（GPU k 跑一组、GPU k+1 跑另一组）。G 必须整除 N。0.1.41 起 `--batch-groups auto` 默认 = 每 GPU 段一组（2 卡 4 槽 = 2 组 ×2）。
- 引擎默认上限 8 槽（一个 window 8 行），受 VRAM 限制；不足 2 槽时退回「一次一个」。
- 槽位的代价：每槽 session 占 VRAM（32K 上下文 + 8-bit KV 约 0.56 GiB），从专家缓存里出。专家多在 CPU 算的卡上，batch 只买到「延迟公平」不买速度：12GB 卡上单请求慢 11–24%。官方 5070 / Q2_0 / 32K 实测：4 并发时第 4 个请求首 token 从 11.2s 降到 1.8s，但总吞吐 70.7 → 63.1（−11%），单请求 79.6 → 16.9。
- 能塞满显存 / 层切分才是槽位真正加总速度的地方：4×16GB 层切分 + `--batch 8 --batch-groups 4 --trim-stage-weights`，并发 1→8 时总吞吐 120 → 360 tok/s（单请求 123 → 45）。
- 默认 batch window 不带 MTP 草稿（一槽一 token/window）；`--batch-mtp`（需单卡、`--mtp`+`--spec`）让每槽每 window 验一个 MTP 提案，不支持层切分 → 双卡用不了。
- server 侧调度（真正决定「为什么会等」）：
  - 请求多于槽位 → 等空槽（`/metrics` 的 `live.slots` 看每槽 idle/reading/decoding；`live.running` 是在飞请求数）。
  - 准入（admission）一次只做一个：「两个新长 prompt 一前一后地读」。
  - 读长 prompt 时，槽位只在每个 chunk 边界解码（默认解码时长 = 该 chunk 读时长 × 0.5），所以一个长 prompt 会拖慢而不是停住其它槽。
  - 长 prompt 遇「半长以下」的短请求会 `BYIELD` 让路（下一 chunk 边界切出）。
  - 请求数在 1↔多 之间变化会有 solo↔slot 的 STOP / 状态恢复 churn（每请求最多 2 回；短会话恢复约 50–60 ms）。
- `--pipeline-windows` 与 `--batch` 槽位互斥（层切分下两个都开不了）。

### 9.3 本机双卡实测（[本机实测]）

> 本机已从单卡变为双卡（此前台账 §10 记的是「单卡」时期的值）。现在与 [hardware.md](../hardware.md) 环境2 一致。

| 项 | 值 |
|----|----|
| 硬件 | `gpu_name: "NVIDIA GeForce RTX 5060 Ti + NVIDIA GeForce RTX 5060 Ti"`，`gpu_count 2`；CPU 285K 24C24T；RAM 61.0 GiB（标称 64GB） |
| PCIe | `gen 5 / gen_max 5 / width 8`（x8，符合环境2 的 x8+x8）；`rx 306 MB / tx 1739 MB`（负载采样） |
| 引擎 | 0.1.41，模型 `sc117-iq2_xs` |
| 专家缓存 | `expert_slots 11967 / expert_cache_mib 16416`（两卡合计，≈729/GB）；primary（GPU0）6529 / 8816 MiB → 第二卡约 5438 槽 / 7600 MiB |
| 并发 | `batch_slots 4`、`batch_groups 2`（= 每 GPU 段一组 → 即开了 `--batch 4`）、`live.parallel 4` |
| 现场 | `live: running 1–2 / waiting 0–1 / outside_slots 0–1`（观察窗内） |
| 命中率 | 单请求 `hit_rate 0.935–0.975`、`pcie_share 0.012–0.043`（专家几乎全在显存）；另见 40,721-token 长 prompt 样本 `hit_rate 0.794 / pcie_share 0.131` → 该区间随负载下探 |
| decode | 单流 `decode_tok_s 77.2–86.3` |

→ 这台机器正好落在「层切分 + 专家几乎全常驻」这一档，是官方说槽位能真正加总速度的场景（§9.2）。所以在此机上：并发主要买的是排队时间减少 + 总吞吐上限提高，而不像 12GB 单卡那样几乎只买延迟。

### 9.4 双卡的参考表现：他卡实测与本机一手

上游 `docs/MULTI_GPU.md` 的实测（[官方]读，他卡；同族做法=按层切分）：

| 机器 | 模型 / 档位 | 上下文 | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 2× RTX 3090（PCIe 4.0 x16 + x4，无 NVLink） | Flash-Next GSQ-RCO IQ3_S | 19.9K | ~1798 | 137.0（+`--adapt-async` 145.2；+`--pipeline-windows 2` 136.7 / prefill 2097） |
| RTX 5080 + RTX 3090 | Coder | 32K | 2039–2357 | 84–110（5080 单卡 83–105） |
| 2× MI50 16GB | Coder | — | — | 39.2 → 41.7（`STRATA_STAGE_TRIM`） |
| 2× Tesla P40（实验 CUDA12） | IQ2_XS | — | — | 19.6 → 34.8 |
| 2× Arc Pro B60 24GB | IQ2_XS | — | 436 | 58.6–61.3 |
| 4× 16GB（PCIe Gen3） | IQ3_S | — | — | 并发 1→8 总吞吐 120 → 360 |
| 2× RTX 2080 Ti + DDR3 | IQ3_XXS | 262K | — | 75（命中率 97.7%） |

同模型「单卡 → 双卡」的官方口径：上游速度表（§13.6，IQ2_XS）给单卡 262K decode ≈ 47；本机双卡 IQ2_XS 实测 77.2–86.3 → 约 1.6–1.8×。
> 分母是「旧口径估算 ±20%」、分子是 0.1.41 实测，不是同一批实验，只作量级参考（[推算]）。官方未按本机 5060 Ti 双卡 PCIe 链路重测：`docs/COMMUNITY_BENCHMARKS.md` 有 2×TITAN RTX / 2×MI50 / 2×Arc Pro B60 / 2×Quadro RTX 4000 8GB（IQ2_XS，131K，层切分 + RAM 分层专家） 等双卡条目，但无 5060 Ti（RTX 4000 那条的明细未取到，[摘要级]）——所以本机 §9.3 是这台配置上唯一的双卡一手数据。

读数（本机口径：2× RTX 5060 Ti，PCIe 5.0 x8，命中率约 0.97，单流 77 至 86 tok/s）：
- decode 已越过「单卡 IQ2_XS@262K ≈ 47」一档，落在上游同类双卡（2×3090 IQ3_S 137、2×Arc B60 58–61）的中段；
- 按官方规律，prefill 是双卡收益最大的一块（+18–20%，两卡各读自己那几层并 chunk 流水），`STRATA_PREFILL_HELP=1` 时短 prompt 还能再快；
- 并发总吞吐在 `--batch` 槽 + 分组流水下可显著上抬（4×16GB 档 1→8 并发 120→360），上限受 8 槽与 VRAM 约束。
- 各机器模型档位/CPU 不同，不可直比；此处仅给「可能的区间」。

### 9.5 结论（引擎设计要点）

多卡用「按层切分（pipeline）+ 各卡各自专家缓存」，不要每层同步的 TP；并发用「可选 N 槽 batch + 按 GPU 分组的流水」，且要正面回答：槽位吃专家缓存、batch 默认无投机、准入串行、长 prompt 拖慢全体。本机双卡实测表明：专家高驻留 + 层切分时，槽位同时改善排队与总吞吐。

---

## 10. 本机实测（[本机实测]）

实例：目标测量机上运行中的引擎 HTTP API，提供 `/metrics`、`/props`、`/health`、`/slots`。内网地址不写入本文档。

| 项 | 值 | 版本 |
|----|----|------|
| 硬件 | 2× RTX 5060 Ti 16GB（`gpu_count 2`，`gpu_name` 两卡同名）+ 285K 24C24T + 64GB（可见 61.0 GiB） | [本机实测] |
| 引擎版本 | 0.1.41（档位字段逐位未变；本机已由单卡变双卡，见 §9.3） | [本机实测] |
| 模型 | `sc117-iq2_xs`（IQ2_XS 分片 GGUF，2 卷） | [本机实测] |
| 上下文 | 262144 | [本机实测] |
| KV | `int8` + `kv_resident 32768` | [本机实测] |
| 专家缓存 | `expert_slots 11967` / `expert_cache_mib 16416`（两卡合计，≈729/GB）；primary（GPU0）6529 / 8816 MiB | [本机实测] |
| 投机 | `spec 6` / `mtp_max 4` / `lookup 3` / `spec_min_p 0.50` | [本机实测] |
| 内存 | `arena_mib 17395`（约 17GB pinned，双卡后专家更多上卡）；RAM 实测已用约 73%（44.6 / 61 GiB） | [本机实测] |
| 并发 | `batch_slots 4` / `batch_groups 2`（= 每 GPU 段一组）/ `live.parallel 4`；现场 `running 1–2`、`waiting 0–1` | [本机实测] |
| PCIe | 负载下 `gen 5 / gen_max 5 / width 8`（x8，与 [hardware.md](../hardware.md) 环境2 一致）；`rx 306 MB / tx 1739 MB` | [本机实测] |
| 命中率 | 单请求 `hit_rate 0.935–0.975`、`pcie_share 0.012–0.043`（专家几乎全在显存）；40,721-token 长 prompt 样本低至 `0.794 / 0.131` | [本机实测] |
| `conversation_cache` | `enabled:false`（门槛「空闲 ≥2,560 MiB」而实际空闲远低于此） | [本机实测] |
| decode | 单流 `decode_tok_s 77.2–86.3`（双卡、命中率≈0.97）；历史单卡期 57.9–81.7 | [本机实测] |
| prefill | 1,392–1,511 tok/s（长 prompt 重算口径）；短 prompt 仅 103–142（固定开销主导） | [本机实测] |
| 投机接受率 | 72.2%（历史窗口）/ 74.1%（527/711 计数窗口） | [本机实测] |
| SSD | `file_blobs = 0`、`disk_read_mb ≈ 0` → 这批请求未走 SSD 热路径 | [本机实测] |

冷启动 ≠ 稳态：重启后首请求 `prompt_tokens 58 / 741.9 ms ≈ 78 tok/s`，只有稳态 prefill 的约 1/18；同期 `hit_rate 0.537` 低于稳态区间。→ 任何 prefill / 命中率口径都必须读作「稳态、且会话 KV 已热」。

---

## 11. 约束与风险清单

1. CPU 算力即上限：AVX2（无 AVX-512/AMX）平台未命中部分的吞吐被封顶。
2. 命中率是单点依赖：所有速度结论都挂在命中率上；工作集切换时 `hit_rate` 会波动（单卡期实测 0.655–0.859；双卡常规 0.935–0.975；长 prompt 下可下探至 ~0.79，此时 `pcie_share` 同步升高，见 §9.3 / §13.4）。
3. 专家缓存有自适应淘汰，但不是开放策略面：`--adapt-every` / `--adapt-swaps` / `--adapt-async` / `STRATA_ADAPT_LAG` 控制「跟着会话把专家在 VRAM↔RAM 之间搬」，被换出的专家回写到它在 RAM 的位置，使 RAM 侧恰好保存「GPU 没有的那些」（[官方] `docs/DETAILS.md`）。没有的是「可插拔/可自定义的淘汰策略」与「共现感知的预载」。
4. `conversation_cache` 事实上不可用：门槛「空闲 ≥2,560 MiB」在最需要它的多轮 agent 场景下恒为关。
5. n-gram 表无显式预取器：走 OS page cache 而已；28.8GB 表在 SSD 上时冷启动代价大（首请求约为稳态的 1/18）。
6. WSL 不可用 `--kv-streaming`。
7. 并发槽位有限：默认 1（一次一个请求），本机 4，引擎上限 8；且 batch 默认无投机、准入串行、长 prompt 会拖慢全体（§9.2）。并发吞吐天花板明确。

---

## 12. 对引擎设计的可迁移结论

| 结论 | 来自 |
|------|------|
| 显存要屯「最热专家」而不是「整层权重」 | 第 3、4 节 |
| 未命中必须与 GPU 并行算，不能串行搬 | 第 5 节 |
| 投机解码是 MoE-卸载场景的必选项（不是可选项） | 第 6 节 |
| KV 要做「常驻热窗 + 流式冷段」两级 | 第 7 节 |
| 所有参数应按机器自动标定（`--calibrate` 的思路可再进一步） | 第 8 节 |
| 多卡用「按层切 + 各自专家缓存」，别用每层同步的 TP | 第 9 节 |

同时，第 11 节的 7 条约束，即本设计的机会清单。

---

## 13. 上游文档补充核查（[官方]）

2026-10-09 对上游文档的补充核查，独立成节（本节全部事实源自上游 `docs/DETAILS.md` / `docs/INTEL_ARC.md`）：

### 13.1 `--calibrate` 的确切能力边界

引擎 0.1.19 起自动实测 4 个「随 PC 而非随模型」的参数（`--pcie-frac`、`--spec-min-p`、`--pool-workers`、`--adapt-*`），只保留 >3% 提升的取值，结果按 PC + 模型记忆，耗时 15–30 分钟。→ 「自动标定到机器 profile」不是本项目的差异点，上游已实现。

### 13.2 已实现的「预测预取」与「批量 IO」

- `STRATA_LOOKAHEAD` / `STRATA_LOOKAHEAD_K`：CPU 算一层时，另一线程对下一层应用路由，预测未命中专家并请求其页（只请求页，不改计算结果；`STRATA_LOOKAHEAD=0` 关闭）。
- `STRATA_IO_PREFETCH=1`（0.1.40.2，Linux file-tier）：整块 `pread` 进页缓存，`STRATA_IO_PF_THREADS` 默认 8，`STRATA_IO_PREFETCH_DEPTH` 默认 2；greedy 输出哈希不变（已自证逐位一致）。
- Windows 侧用 `PrefetchVirtualMemory` 一次性批量请求整页集合。
- `STRATA_EXCHANGE_ROTATE=1`：自适应交换改为移交缓冲所有权而非拷贝，宿主拷贝更少。

### 13.3 预取的实测收益与负收益

cgroup 限内存、200-token greedy 中位数、每次跑前丢页缓存：

| 机器 / 内存上限 | 默认 | `STRATA_IO_PF_STAGE=1` |
|---|---|---|
| RTX 3060，32GB，IQ3_XXS | −3.4% | +25.7% |
| RTX 3060，16GB，IQ3_XXS | −1.8% | −32.6% |
| Radeon 780M 核显，16GB，Q2_0（自适应层关） | −0.7% | −19% |

上游结论：页缓存热时文件层已按常驻速度跑，没有可赢的东西；差距只在冷启动与低内存 lane。默认关闭，暂存变体只建议 32GB 级机器尝试。
iGPU 附加警告：Radeon 780M 上「预取 + 自适应层」在内存压力下约 9 次里 7 次 GPU 复位（引擎只打警告照跑）；`--adapt-every 100000` 后约 40 次无复位。原因未知。

### 13.4 `hit_rate` 的口径

`hit_rate` = 查表专家中 residing 在 VRAM 的比例；经 `--pcie-frac` 从 PCIe 搬入的专家不计入 → 「调高 `--pcie-frac` 会让 `hit_rate` 上升，即使解码变慢」。另有 `pcie_share`（0.1.39+，#588）单独统计 PCIe 侧占比。
另：`--vram-reserve-mib` 从 770 降到 0 可多出 333 个槽位（2,782→3,115），但每步剩 0 MiB 空闲会触发系统回退，上游判定「留给整张卡其余部分的余量比多出来的槽位更值钱」（引 issue #781）。

### 13.5 逐位复现的开关组合（本项目对确定性的要求，上游已文档化）

跨请求有两件事会改变舍入：自适应层在 RAM/VRAM 之间搬专家（GPU 与 CPU 对同一专家的舍入不同）与 prompt cache 复用。要逐位一致需 `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`（#410）。

### 13.6 本机同款卡的官方估算档位

上游速度表（自注为旧口径估算、未按本机 PCIe 链路重测、±20%），格式 prefill / decode tok/s：

> 此表是单卡口径；不可与 §9.3 的双卡实测直接相除——量级对照见 §9.4（单卡 IQ2_XS@262K ≈47 → 双卡实测 77–86，约 1.6–1.8×）。

| 量化档 | 1K | 32K | 128K | 262K |
|---|---|---|---|---|
| Q2_0 | ~341 / ~80 | ~501 / ~81 | ~476 / ~62 | ~435 / ~53 |
| IQ2_XS | ~291 / ~80 | ~434 / ~63 | ~413 / ~51 | ~383 / ~47 |
| IQ3_XXS | ~249 / ~66 | ~381 / ~56 | ~363 / ~45 | —（放不下） |

### 13.7 Intel / 核显线（`docs/INTEL_ARC.md`）

- 存在 Arc SYCL 端口（PR #423），0.1.39 / 0.1.40.2 持续维护。
- 维护者在 Arc Pro B70（xe）跑 Qwen3.8-Flash-Next IQ3_S：11.5k / 24.6k 专家常驻，4K prompt 980–1,000 tok/s，decode 30（散文）–41（代码）tok/s，10-prompt 门禁通过。
- 社区 2× Arc Pro B60 24GB + `--layer-split`：Flash-Next IQ2_XS 58.6–61.3 tok/s decode、436 tok/s prompt。
- WSL2：IQ2_XS ~21 tok/s、Coder ~15 tok/s，并见到 device loss。
- 明确未做：Windows 原生构建、Alchemist A 系列、集成 Arc（B390 / Panther Lake，#515）的 shared-memory planning「尚不存在」、Intel 侧图像未接。

---

---

---

## 来源

| 编号 | 用于章节 | 内容 |
|------|---------|------|
| S-02 | §1 至 §9、§13 | 上游引擎仓库与文档（README、HOW_IT_WORKS、DETAILS、MULTI_GPU、BATCHING、INTEL_ARC、COMMUNITY_BENCHMARKS） |
| S-03 | §9.3、§10 | 本机运行中的引擎实例 API |
| S-04 | §9.4 | 他卡实测：2×RTX 2080 Ti，IQ3_XXS，262K |
| S-05 | §4、§9.4、§13.6 | 上游速度表与社区实测汇总 |

等级、复核状态与 URL 见 [references.md](references.md)。
