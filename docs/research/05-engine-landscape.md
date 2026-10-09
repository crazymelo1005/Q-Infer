# 推理引擎横向对比与选型

> 本篇回答同一个主案例模型在各引擎上的分工：各自强在哪、在本机配置上谁能赢。刻度统一为「它在哪个维度是真的第一」。

来源等级标记：[官方] · [他卡实测] · [本机实测] · [硬件规格]。复核状态标记：[已确认] 逐位确认 · [摘要级] 仅标题或摘要。

---

## 1. 总纲

> vLLM / SGLang / NInfer / TensorRT-LLM 全都在「模型装得进显存」的前提下优化「算得快不快、塞得密不密」；只有 Strata / KTransformers / llama.cpp-offload 这一条线在回答「装不进怎么办」。

而 125B + 51B 外挂表在 16GB 单卡上，属于后者。这是本项目主线押 Strata 的根本原因，不是因为 Strata 在别的维度更强。

---

## 2. 横向矩阵

| 引擎 | 格式 | 最强场景 | 并发 | 长上下文 | 显存不足时 | 多卡 | 多模态 | 本机适配 |
|------|------|---------|------|---------|-----------|------|--------|---------|
| vLLM | safetensors/GPTQ/AWQ/NVFP4 | 高并发服务 | 强 | 好（Paged+prefix） | 可 KV offload | 支持 TP/EP/TEP | 支持 | 支持（NVFP4 checkpoint；Flash-Next 需专用镜像） |
| SGLang | safetensors | 高吞吐 + 长上下文复用 | 强 | 优（Radix） | 部分 | 支持 | 支持 | 支持（Flash-Next Day-0） |
| llama.cpp | GGUF | 单用户 / 显存不足 | 弱 | 靠 KV 量化 | CPU offload 强 | 弱 | 支持 | 支持（较慢） |
| NInfer | .ninfer | 单卡 Blackwell 低延迟 | 8 并发批解码 | 好 | 官方无 offload（但 KVMem 正在给它加分层的 KV 后端，见 §7） | 不支持 | 支持 | 需移植 |
| Strata | GGUF | 消费级跑超大 MoE | 默认 1 / 上限 8；本机 4 槽（见 `01` §9.2） | 支持卸载 | 专家三级分层 | 按层切 | 支持 | 支持（主线） |
| KVMem | GGUF（llama.cpp fork）+ .ninfer（NInfer 后端） | 消费级 16GB 显存跑满 256K 上下文 | 多会话 lane（`--kvmem-conversations`） | 优（分页 KV 虚拟化 + 索引选块） | KV 三层：VRAM / host RAM / NVMe | 无 | 支持（IQ3 走 GPU 视觉 / IQ4 走 CPU 视觉） | 支持，且实测机就是 RTX 5060 Ti 16GB |
| KTransformers | safetensors/GGUF | 超大 MoE 异构 | 中 | 好 | 异构 offload | 部分 | 视模型 | 受限（无 AVX512） |
| LMCache | 库 | KV 分层复用 / 前缀加速 | — | 优 | KV offload | — | — | 配合 vLLM/SGLang |
| Ollama / MLX | GGUF / MLX | Apple Silicon 统一内存 | 弱 | 一般 | 统一内存 | 否 | 支持 | 不适用（需大统一内存） |

---

## 3. 各引擎的优势与本机不可得的原因

| 引擎 | 真正的优势 | 什么时候选它 | 本机拿不到的原因 |
|------|-----------|-------------|-----------------|
| vLLM | 并发吞吐与生态广度：迭代级调度 + PagedAttention；新模型适配最快；TP/EP/PP 齐全，多机扩展唯一成熟选项 | 服务多人 / 多会话；工具调用、结构化输出、多模态服务化 | 它假设「权重装得进显存」；16GB 上 125B 靠调度技巧解决不了容量；KV 池与权重抢同一块显存 |
| SGLang | 前缀复用与结构化生成：RadixAttention 跨请求命中；Agent 多轮 repeated prefill 接近归零；新架构跟进最快 | Agent 多轮、大量共享前缀、受控生成/DSL | 同样是服务端定位、吃显存；单张 16GB 上容量先于调度成为瓶颈 |
| llama.cpp | 硬件覆盖面与量化档位数：零 Python 依赖，GGUF 档最全；层级 offload 把任意大模型磨进有限显存 | 硬件杂 / 只有一份 GGUF / 单用户跑通就行 | offload 是「逐层经 PCIe 搬回再算」，每层都在等搬运；对 MoE 无命中率概念 |
| NInfer | 单张 Blackwell 的低精度极值：直吃 sm_120 FP4/FP8 Tensor Core；CUDA Graph + 分块预填 + 8 并发批解码 | 卡够放下整个模型、只追单卡极限（如 27B NVFP4） | 无 offload、不支持多卡 → 对 125B 直接出局 |
| KTransformers | CPU/GPU 异构先驱：专家级卸载 + NUMA 感知 + AMX/AVX512 内核 | Intel 服务器 / 带 AMX 平台 + 超大 MoE | 强依赖 AMX/AVX512；本机 Arrow Lake 两者都没有 |
| Strata | 唯一为「消费级 + 超大 MoE」从头设计：专家三级分层 + 热度排序常驻 + 未命中就地算 + 内建 MTP | 单机（单/双卡）、大内存、跑装不下的 MoE | 本机即它的目标场景——运行中的实例就是本机（[本机实测] 双卡，见 [01-strata-engine.md](01-strata-engine.md) §9.3） |

---

## 4. 同机对照：卸载路线的收益边界

| 场景 | llama.cpp | Strata | 差距 |
|------|-----------|--------|------|
| 专家装不进显存（大模型 + 小显存） | 跑更小的 IQ2_XS 也只有十几 tok/s | 跑更大的 IQ3_XXS 达 75 tok/s（2×2080Ti，262K） | 约一个数量级 |
| 模型能整个塞进显存（Coder IQ1_M，2×MI50） | 26.9 tok/s | 50 tok/s | 约 2× |

> 徽章：本表全部为 [他卡实测]（不同卡、不同量化档；趋势可信、绝对值浮动）。

读数：Strata 的红利主要来自「专家装不下」这一场景。一旦模型能塞进显存，它的优势收缩到约 2×（此时 NInfer/vLLM 的低精度与服务化优势反超）。

其它参考（[他卡实测]，不可与本机直比）：
- NInfer / RTX 5090 / 27B NVFP4：decode 202 tok/s、prefill 约 5,950 tok/s（开 MTP；对比 llama.cpp Q5_K_XL 约 1,700）。
- Strata / RTX 5070 12G：Q2_0 94 / IQ2_XS 79 / IQ3_XXS 62 / IQ3_S 53 tok/s（4K 输出 / 32K prompt）。
- vLLM / 4×H100 + PLE 卸载：并发 64、1024-in/256-out 下约 1,430 output tok/s（数据中心口径）。

---

## 5. 选型结论（面向本机：2× RTX 5060 Ti 16GB + 285K + 64GB；2026-10-09 起双卡，单卡为硬下限口径）

1. 主线：Strata（或同类分层卸载引擎）——因为本机属于「装不进」场景。服务端路线（vLLM 专用镜像 + FP8 + `VLLM_PLE_CPU_OFFLOAD=1` + TEP）作为对照验证。
2. 并发上限要认清：默认一次一个请求，本机已开 `batch_slots 4`（引擎上限 8）。任何依赖「batch 变大转 GEMM」的吞吐技巧在本机收益有限。
3. 对照案例：若换成能塞满单卡的 27B 级模型 → NInfer / vLLM + NVFP4 + MTP/DFlash2 反而更快。
4. 256K + 4–8 并发：依赖 GDN 无 KV 红利 + QSA 稀疏读 + KV 量化（含索引器）+ PagedAttention + 前缀缓存。注意并发一高，GDN 状态与 QSA KV 两条一起涨。
5. 多卡：避开每层同步的 TP；Strata 的按层切分（pipeline）已可用且有官方实测（2×3090 IQ3_S decode 137 tok/s；4×16GB 并发 1→8 总吞吐 120→360；见 `01` §9.4），本机双卡实测 decode 77–86 tok/s（`01` §9.3）。

---

## 6. 从各引擎借鉴的要点

| 取自 | 拿来什么 |
|------|---------|
| Strata | 三级分层 + 专家热度缓存 + 未命中就地算 + `--calibrate` 自动调参 + `STRATA_LOOKAHEAD` 跨层预测预取 + `STRATA_EXCHANGE_ROTATE` 缓冲所有权移交 |
| vLLM | PagedAttention + 迭代级调度 + 索引器 KV 独立精度 + `VLLM_PLE_CPU_OFFLOAD` 的表行异步预取（见 §7.2） |
| SGLang | RadixAttention 前缀树 |
| llama.cpp | GGUF 生态与量化档位、CPU 侧 ggml 内核 |
| NInfer | sm_120 上的 FP4/FP8 极限路径、CUDA Graph、分块预填 |
| KVMem | 分页 KV 虚拟化 + 索引选块 + 多会话 lane + 长上下文任务效用评测口径（LongMemEval / AgentLongBench，见 §7.1） |
| KTransformers | 专家级卸载 + NUMA 意识 |
| LMCache | KV 作为一等公民的独立缓存层 |
| 无人提供（0.2 版修正） | ~~外挂记忆表（n-gram）的显式预取与双缓冲~~ —— 已被 vLLM 与模型官方占位（§7.2）。剩下的是「专家 + 表行 + KV 三类流量在同一步内的显式同步预算」，该框架未找到公开实现（[摘要级] 存疑） |

---

## 7. 补充核查（[官方]）

> 本节为 2026-10-09 对竞品现状的补充核查（[官方]）

### 7.1 KVMem（`kvmem/kvmem-llama.cpp`）

- 性质：KV 上下文虚拟化——GPU 只保有界活跃 KV 工作集，完整历史 KV 存 host RAM（可选 NVMe），按查询用轻量 Mean-K 索引选块、re-RoPE 还原位置后物化进工作集。是 llama.cpp 的 fork（另有 NInfer 后端的预发布版）。
- 测试平台（关键）：RTX 5060 Ti 16 GiB + Intel Core Ultra 7 255H + 32GB RAM（WSL2 可见 19.53 GiB），Ubuntu 22.04.5 on WSL2，CUDA 13.2.86，`CMAKE_CUDA_ARCHITECTURES=120a-real`。与本项目目标卡同型号。
- 模型：Qwen3.8-27B（稠密）GGUF IQ3 / IQ4 + 官方 `-mtp` 草稿权重。
- 数字：256K 全工作区，decode 32–33 tok/s（MTP3）、prefill 437–463（初次）/ 242–253（含重处理与缓存管理）；GPU KV 窗（IQ3）36K 检索 / 16K 生成、（IQ4）32K / 12K；主 KV q8_0（IQ3）/ q5_0（IQ4）；运行时 host RSS 4.3–13.5 GiB。
- 质量：LongMemEval-S 85.6% vs 全量 86.6%；AgentLongBench 60.9% vs 59.5%（论文口径另有 32K 活跃 vs 256K 全量近无损、DeepSWE 43.8%→48.4%）。
- 旋钮：`--kvmem-budget` / `--kvmem-gen-reserve` / `--kvmem-conversations N` / `--kvmem-conversations-gb` / `--kvmem-session-ram-gb` / `--kvmem-session-nvme-gb` / `--kvmem-session-cache-dir`；多 lane 会话动态分配。
- 已知限制：单次生成（含 thinking）不能超过 `--kvmem-gen-reserve`（IQ3 16384 / IQ4 12288）；需 flash attention。
- 与 Strata 的分工：KVMem 分层 KV/工作区、不碰专家权重（模型是稠密 27B）；Strata 分层专家权重、KV 只做「常驻热窗 + 流式」。

### 7.2 表行异步预取已被实现

- vLLM 官方 recipe：`VLLM_PLE_CPU_OFFLOAD=1` —「把 51B N-gram 查找内存留在 host RAM，异步预取所需行」；在 4×H100 上 plain TP4 启动即 OOM，靠它把表甩到主存（host 内存需求 ≥51GB）；且 Pipeline-Parallel 与该表初始不兼容。
- Qwen 官方 README：该表「可异步卸载到主机内存并与计算重叠」。→ 「表行异步预取」不再是本项目的独占差异点；可主张的只剩「VRAM 侧双缓冲 + 精确行索引提前一步 + 与专家/KV 三类流量的统一预算」，且这部分未获一手确认（[摘要级]）。

### 7.3 其它结论修正

- NInfer 的 offload 缺口正在被外部填补：KVMem 提供 NInfer 后端的预发布版（`v0.18.0-ninfer-rc2`，Windows RTX 30/40/50）。「无 offload」仍是 NInfer 官方形态的事实，但已不是这条生态的终局。
- 预取在低内存 lane 上是负收益：Strata 实测暂存预取在 16GB 内存 lane −32.6%、Radeon 780M 核显 −19%；详见 [01-strata-engine.md](01-strata-engine.md) §13.3。这条直接影响本项目「表行预取」赌注的期望值。

---

---

---

## 来源

| 编号 | 用于章节 | 内容 |
|------|---------|------|
| S-02 | §1、§5、§6 | 上游引擎仓库与文档 |
| S-05 | §2、§4 | 上游速度表与社区实测汇总 |
| S-06 | §2 | 同类引擎在 RTX 5090 上的实测 |
| S-01 | §6、§7.2 | vLLM 官方部署配方 |
| S-03 | §5 | 本机运行中的引擎实例 API |
| S-26、S-27 | §7.1 | KV 上下文虚拟化 |

等级、复核状态与 URL 见 [references.md](references.md)。
