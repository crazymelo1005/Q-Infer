# ADR-022：核显不接手本引擎解码路径的 CPU 份额（解码时间预算与带宽归因）

状态：Accepted（2026-10-15）

---

## 背景

[ADR-019](ADR-019-igpu-not-for-experts.md) 判核显不适合承接未命中专家的计算，[ADR-020](ADR-020-strata-igpu-pool.md) 判独立专家池不可用。两个结论都是「某条具体接线不行」。本 ADR 换一个问法，把它一次问到底：在这台机器的这套引擎里，持续解码时有没有哪个 CPU 模块适合核显接手？ 为此需要先把解码一轮（一个 verify 窗）的时间花在哪，按引擎自报的读数摊开。

## 测量：一轮解码的时间预算

口径：环境2，真权重 sc117 IQ2_XS，引擎 0.1.43.1，`--pack/--native/--ple-gguf/--mmap-experts/--kv int8/--spec 2`，一 shot `strata` 生成模式。引擎在每个阶段结束时自报分项（`strata <阶段>` 行）。一轮（4-token 窗）：

| 分项 | 读数 |
|---|---|
| verify window | wait for rings 12.585 / pool 74.476 / host 0.023 / commit 0.019 ms |
| pool multi（真算子） | gate/up 6.974 + quantize 0.093 + down 2.783 = 9.85 ms；同阶段声明 52.1 GB/s |
| dispatch | plan 0.069 / activation quantize 0.071 / jobs 64.348 / run 9.855 ms |
| CPU pool call | 74.460 ms |
| expert tiers | 502.19 MB/round，68.8 ms/round 读，7.30 GB/s 每读线程 |
| routing prefetch | predictor 14.9 ms/round（在它自己的线程上） |
| CPU 专家 | 每层 7.36 distinct / 8.39 routed |

一句话：**一轮里 CPU 的算术只有约 9.9 ms，另有约 65 至 69 ms 在搬专家权重。**

## 那个 64 ms 是什么：一次一个专家，没有重复读

先排除「引擎在重复读同一批专家」这一可能——这是唯一能让搬运量凭空缩水的解释。用 pack 几何核算（`pack/native_experts.txt`）：

- 48 层；每层 512 个专家；单专家 blob 1.18 至 1.51 MB，均值 1.44 MB。
- 全量：48 × 512 × 1.44 MB = 35.45 GB，与文件头 `total 35454976000` 逐字节吻合。
- 每轮：48 层 × 7.36 个 CPU 专家 × 1.44 MB = **509.7 MB**，引擎自报 **502.19 MB**，差 1.5%。

即读入量与「每个 CPU 专家读一次」严格对应，**没有重复读**。这 65 至 69 ms 就是把 500 MB 权重从页缓存搬进 CPU 池的物理代价（约 7.3 GB/s 每读线程）。

## 为什么核显接不了

搬运这条：核显与 CPU 共用同一份 DRAM，**它不提供任何额外带宽**，而带宽正是这里的约束资源。算术这条：本机核显 `shaderFloat16`/`shaderInt8` 可用，但 `VkPhysicalDeviceShaderIntegerDotProductProperties` 的逐 op 能力位全 0（见 ADR-005 的复核），2-bit 专家 GEMM 只能在 fp16 上拆包模拟；ADR-019 实测纯分派臂仅为 8 线程 CPU 池的 0.13 至 0.19 倍，端到端 0.07 至 0.12 倍。**所以把 CPU 的份额交给核显，只会更慢**——ADR-020 实测 0.36 tok/s 正是这个结论的读数。

换算成比例：一轮搬 500 MB 换 9.9 ms 算术，约 50:1 的字节比算力。这是带宽受限负载的定义式，而核显在这条维度上是零增益项。

## 候选排序（按「结构上最像」排，逐条说明为什么仍不行）

| 候选 | 形状 | 不行的原因 |
|---|---|---|
| 未命中专家 gate/up/down | 最合适（大、规则、可并行） | 要同时满足：权重常驻核显、有 batch 维摊薄分派、fp16 路径。引擎三者皆无（`gdn_layer` 逐 token，只有 verify 窗 MT 不超过 6），且缺整数点积 |
| 激活量化（Q8_0） | 合适 | 0.09 ms/轮，量级上无意义 |
| router / norm / gating / combine | 合适 | 微秒级且都在延迟关键路径 |
| attention / KV | 不合适 | KV 97.8% 命中已在 VRAM，核显无 KV，搬过去要过 PCIe |
| MTP 草稿层 | 唯一延迟不敏感的重算（可边验边起草） | decode 已达标（100 至 136），且草稿层本就在 GPU 上 |

## 决策

1. 核显不接手解码路径的 CPU 份额。这不是「再调调参数」的问题，而是带宽与指令能力的双重物理限制；ADR-019/020 的两条接线结论由此推广到全部候选模块。
2. 唯一能真正抬高 prefill/decode 的杠杆是让未命中专家变少（把 7.36 降到更低），也就是更大的 VRAM 专家缓存。该杠杆已拉满：`--expert-cache auto` 取 8047 槽（11.01 GiB），加载完只剩 444 MiB 显存余量。
3. 因此 prefill 与 decode 的天花板由显存容量与 DRAM 带宽决定，不由核显决定。3K prefill 的差距同理（见 [ADR-021](ADR-021-prefill-expert-io-bound.md)）。

## 后果

- 「核显替代 CPU 应有明显效果」在本机这套引擎里不成立，且已定量给出原因（约 50:1 的搬运比算术）。要在核显上真正提速，得先有一个有 batch 维、权重常驻、走 fp16 的独立负载；那是引擎级改造，不是接线问题。
- 保留核显资产（`src/experts/igpu`、ADR-014 阶段 C）与 `--igpu off|dense|pool` 开关，便于换硬件后复评。
- 若日后要再试，先测「未命中专家数」这个量：它是这里唯一的自变量。