# 推理加速与显存优化技术综述

> 目的：把「能让推理更快 / 更省显存 / 保持质量」的技术分轴摆开，逐条回答三个问题：改的是哪个瓶颈？代价是什么？在本项目硬件上值不值得上？
> 环境2 约束：2× RTX 5060 Ti 16GB（每卡 448 GB/s，PCIe 5.0 x8；2026-10-09 起环境2 为双卡，见 [01-strata-engine.md](01-strata-engine.md) §9.3）、Core Ultra 9 285K（无 AVX-512/AMX）、64GB DDR5-4400（有效带宽约 70GB/s）、SSD 顺序读约 7GB/s。

复核状态标记：[已确认] 逐位确认 · [摘要级] 仅标题或摘要 · [存疑] 来源矛盾已降级。

---

## 0. 量化坐标系：四个正交轴

谈任何「量化档」前，先定位它在哪一格：

| 轴 | 问的问题 | 取值 | 落在哪里 |
|----|---------|------|---------|
| ① 对象 | 量化什么 | 权重 W / 激活 A / KV cache / 索引器 KV / 累加精度 C | W 与 KV 落盘或常驻显存；A 与 C 只存在于运行时 |
| ② 时机 | 什么时候量化 | QAT 训练时（BitNet b1.58）/ PTQ 训练后（RTN、GPTQ、AWQ、Bonsai） | 决定是否需要校准数据 |
| ③ 数值表示 | 用什么数表示低比特 | 整数 INT / 浮点 FP8·FP4 / 码本·向量量化 / 三值·二值 | 决定反量化是「乘 scale」还是「查表」 |
| ④ 粒度 | 多少个共享一个 scale | per-tensor / per-channel / per-token / group-wise(128/64/32/16) | 越细精度越好、元数据与带宽越大 |

Flash-Next 让这套坐标系多出一个子轴：QSA 的「挑块打分」用的索引器 KV 单独还有一个精度旋钮（vLLM: `--attention-config.indexer_kv_dtype`）。[已确认]

记法：`W4A16` = 4-bit 权重 + 16-bit 激活；`C` = matmul 累加精度。

---

## 1. 权重量化家族（对象 = W）

| 方法 | 思路 | 校准 | 备注 |
|------|------|------|------|
| RTN | 最近舍入 | 免 | 一切方案的对照基线 |
| GPTQ | 逐列量化 + Hessian 逆补偿 | 需 | 2–4 bit 经典 |
| AWQ | 按激活幅值识别显著通道，前置缩放保护 | 需（轻） | 对指令模型鲁棒 |
| SmoothQuant | 把激活离群难度按通道迁移给权重 | 需 | 使 W8A8 可行，属「激活轴」联动 |
| NF4 | 按分位数构造码本（等概率码字） | 免 | 信息论最优 4-bit 数据码本 |
| 码本/向量量化族 | QuIP#（随机正交旋转 + E8 格码本）、AQLM（多码本残差级联）、VPTQ（离线优化向量） | 视方案 | sub-4-bit 明显优于纯标量；需专用解码内核 |
| GGUF 分层 | K-quant（超级块 + 分组 scale，敏感层保高精度）/ I-quant（非均匀码本） | 免 | llama.cpp / Strata 生态 |
| NVFP4 | 块 16 + FP8-E4M3 缩放 + 全局 FP32 二级缩放 | 免 | Blackwell 第 5 代 Tensor Core 原生加速 |
| MXFP4 | 块 32 + E8M0 幂次共享缩放 | 免 | OCP 跨厂商标准，更省、实现更简单 |
| 三值/二值 | {-1,0,+1}（1.585 bpw）或 {-1,+1} | 视方案 | BitNet b1.58（QAT）/ Ternary Bonsai 2（PTQ，27B→约 5.9GB，保留 98.2%，[官方] 项目方口径）；靠专用/LUT 核，非 Blackwell 原生 FP4 路径 |

环境2 取向：环境2 是 Blackwell sm_120，NVFP4 有硬件加速 → 权重侧优先走 NVFP4；三值只能走 LUT 核，收益偏容量而非速度。

---

## 2. KV 量化谱系（对象 = KV，与权重轴正交）

KV 是动态张量：量化必须把反量化融进注意力核（fused dequant + FA），否则省下的带宽又读回来、收益归零。且 Key 与 Value 的离群值分布不同 → 必须分开设粒度。

| 方法 | 思路 | 关键特征 |
|------|------|---------|
| llama.cpp 基线 | 简单分组标量量化 | `--cache-type-k/v q8_0 / q4_0 / iq4_nl`，需编译开启 FA |
| KIVI | 混合粒度 | Key per-channel、Value per-token，2-bit 级、免校准 |
| KVQuant | 非均匀 + 离群分离 | pre-RoPE 量化 Key（避开旋转后通道耦合）、按注意力重要度定制码本、离群值保留稠密 |
| NQKV | 分组分位码本 | per-block 分位数量化，利用近似正态假设 |
| SAW-INT4 | 系统感知 4-bit | 旋转保精度，同时兼容 paged 显存 / 规则行布局 / 融合注意力核（工程可落地） |
| TurboQuant | 旋转 + 极分解 + 残差修正 | Google / ICLR 2026，约 3-bit；详见 §2.1 |
| OSCAR | 离线谱协方差旋转 | 利用注意力权重与 Key 的谱/协方差结构预先构造旋转，约 2-bit 接近 BF16，已入 SGLang |

### 2.1 TurboQuant 机制（[已确认]）

算法管道（四步）：
1. Hadamard 变换（正文称「随机哈达玛变换」，代码用固定 Hadamard 矩阵）——使各坐标近似同分布。
2. PolarQuant：把最后一维相邻坐标配成 `(x,y)` → 极坐标 `(r, θ)`。r 做对数量化、θ 做均匀量化，默认 `num_bits = 3`。→ 每对 6 bit = 每元素 3 bit。
3. QJL 残差修正：算残差 `transformed - quantized`，用随机 ±1 投影矩阵投影，只存符号位（默认 128 个投影），解码时平均。文章称统计无偏：`E[decode(encode(e))] = e`。
4. 加回残差 → 逆 Hadamard。

报出的数字（该文自测，非第三方复现）：

| 指标 | FP16 | TurboQuant | 倍数 |
|------|------|-----------|------|
| KV 显存 | 160.2 GB | 26.7 GB | 6× |
| 吞吐 | 12.3 tok/s | 98.4 tok/s | 8.0× |
| 质量（LongBench F1） | 0.876 | 0.875 | −0.1%（「近乎零」） |

要打折扣的地方（关键）：
- 无独立复现。数字只出自该文表格；文中未讨论 QJL 开销与元数据成本，「3-bit / 6×」没有把 QJL（每元素外加 1 bit/投影）与显存元数据计入。
- 未见校准需求（log-radius min/max 直接取自张量）——这既是优点（免校准），也意味着对分布漂移没有自适应。
- 社区复现指出：降到 1-bit（QJL 残差）时误差虽无偏但方差变大，对 top-1 采样未必有利（无偏 ≠ 贪心解码友好）。

> 结论：TurboQuant 是值得跟踪的 3-bit KV 路线，但不能把 6×/8× 当作可继承的既有结论；本设计应把它作为可选精度档接入，并用本项目实测标定。

### 2.2 选型提示

- 长上下文优先试 FP8 KV（硬件原生、风险低）。
- 要冲 256K × 多并发再上 4-bit（q4_0/iq4_nl、SAW-INT4 类）。
- 2-bit 及以下（KIVI / TurboQuant / OSCAR）需要引擎有融合核才能拿到带宽红利；否则可能因反量化开销反而变慢。
- per-token / per-channel 在 kernel 里更贵 → 属「省显存优先」而非「提速优先」。

---

## 3. 稀疏 KV 检索（对象 = 读哪些 KV）

| 方法 | 思路 |
|------|------|
| StreamingLLM | attention sink：保留开头若干 token + 滑窗，其余丢 |
| H2O | 按累积注意力权重保留 heavy hitters |
| SnapKV | 观察窗口内注意力，选 prompt 中的重要 token 压缩 |
| Quest | 按需检索：把 KV 分块、用块级元数据估计重要性，只取需要的块 |

与主案例的关系：Flash-Next 的 QSA 就是 Quest 的同族（micro-block 池化 → 打分 → 取 top 块展开）。所以在本项目上，KV 稀疏检索已经内置在模型里，引擎侧要做的是配合（如块级预取），不是再叠一层。

---

## 4. 投机解码谱系（对象 = 每步产出几个 token）

基本原理：草稿器猜 k 个 token → 主模型一次前向并行验证 → 接受前缀。加速来自「把每 token 的权重读取摊到多个 token 上」。

| 方法 | 草稿来源 | 特点 |
|------|---------|------|
| MTP | 模型自带多 token 预测头 | Flash-Next 自带 4B 头，`num_speculative_tokens: 3`；无需外挂模型 |
| prompt lookup / suffix | 从上下文已有文本抄 | 最多 5 token；代码改写类 6–11% 提速，其它文本基本不变 |
| DFlash | 块扩散 drafter（ICML 2026） | 一次并行起草多 token |
| DFlash2 | "Keep Drafting Parallel"（Inco AI, 2026-08） | 每位置保留 top-16 候选 + 轻量路径选择器 + two-tap 卷积，草稿全程并行；约 2×（媒体标题称「近 3×」）提速、约 1% 额外延迟、输出无损（[官方] 论文 / 项目方口径；媒体标题称「近 3×」，未采信）；主要接入 SGLang，也支持 vLLM/llama.cpp |
| EAGLE 系列 | 学出来的草稿头 | 需训练 |
| Medusa | 多头预测 + 树验证 | 需训练 |

理论关系：接受率 α、草稿长度 k 下的期望产出 ≈ `(1-α^{k+1})/(1-α)`。所以要同时优化 α 与 k，而不是无脑加长。
- 实测参照（环境2 实测）：MTP 接受率 72.2% / 74.1%，对应平均 2.4–3.2 token/步。
- 官方提醒：测 MTP 收益必须用 SPEED-Bench 类真实负载，随机 prompt 的接受率严重偏离，会把结论做反。

---

## 5. 显存与调度（对象 = 显存怎么用）

| 技术 | 改的是哪个瓶颈 | 机制 | 本项目适用性 |
|------|--------------|------|------------|
| PagedAttention | 显存碎片与预留浪费 | KV 切固定块，block table 映射；利用率从 20–40% 提到 >96% | 高（多并发基础） |
| Copy-on-Write / 前缀共享 | 同一 prompt 多采样 | 块级共享 | 中 |
| RadixAttention（SGLang） | 重复前缀 prefill | 前缀组织成基数树，跨请求自动命中 | 高（agent 多轮） |
| chunked prefill | 长 prefill 阻塞 decode | 切块与 decode 混批 | 中 |
| continuous batching | 吞吐 / GPU 占用 | 每步迭代换出完成请求、补入新请求 | 中（受并发上限约束） |
| LMCache / Mooncake | KV 复用与分离式服务 | KV 落 CPU/NVMe 跨请求复用 | 中（需宿主引擎配合） |
| 分层 KV offload | 容量 | GPU HBM / CPU RAM / NVMe 三级 | 高（环境2 主线） |

边界：continuous batching 提升的是吞吐与占用率，不降单请求延迟；batch 变大才使 decode 从 GEMV（带宽瓶颈）转向 GEMM（算力瓶颈）——这正是服务端划算、环境2 4–8 并发体感有限的原因。

### 5.1 上下文复用：三条实现路线（[官方]）

前缀复用是「多轮 / 多请求」最省时间的一招——命中前缀的 token 不再重新 prefill。三家的做法不同：

| 路线 | 代表 | 复用粒度 | 寻址方式 | 跨会话 | 关键限制 |
|------|------|---------|---------|-------|---------|
| 块哈希前缀缓存（APC） | vLLM v1 | KV block（4 / 16 token 一块） | `hash(父哈希, 块 token 元组, 额外哈希[LoRA / 多模态 / `cache_salt`])`；只缓存满块 | 自动、跨请求（前缀相同即命中，`ref_cnt` 引用计数 + 空闲队列 LRU 淘汰） | 块粒度截断：14-token prompt 共享前 10 个 token，只命中前 2 块（8 token）；`cache_salt` 可做多租户隔离；`--prefix-caching-hash-algo`（默认 sha256，另有 xxhash/cbor） |
| 基数树前缀 | SGLang RadixAttention | token 级前缀树节点 | 树节点共享 | 自动、跨请求 | 需维护基数树与树的淘汰 |
| 检查点 + 槽位 + 停车 | Strata | 检查点（每轮 / 每 16K token，约 118 MB × 最多 6） | 逐 token（含图片）完全匹配才能用 | ① 槽位 = 对话级；② system-root 检查点被同客户端所有对话共享；③ parking 可跨 stage 恢复 | 默认关（`--conversation-cache-mib 0`）；只认精确前缀；一次 1 个 pinned prefix |

读数：vLLM 是「内容寻址、块粒度、全自动」，Strata 是「检查点/会话级、token 精确、半手动」。前者在多租户 / 共享前缀的高并发服务里更通用；后者在单机 agent 多轮里省的是「重读十几万 token 的历史」，收益同样巨大。环境2 实测 `reused ≈85%`（见 [01-strata-engine.md](01-strata-engine.md) §7.1）就是这个红利的直接体现。

---

## 6. MoE 卸载与专家放置（对象 = 专家权重放哪）

| 方案 | 机制 | 关键差异 |
|------|------|---------|
| llama.cpp offload | 按层把权重经 PCIe 搬回 GPU 再算 | 每层都在等搬运；对 MoE 无命中率概念 |
| KTransformers | 专家级卸载 + NUMA 感知 + AMX/AVX512 内核 | 性能强依赖 AMX/AVX512；环境2 的 Arrow Lake 两者都没有 |
| Fiddler / Mixtral-Inference offload / HetuMoE | 学术界的专家放置/卸载 | 提供「同类机制已发表」的佐证 |
| Strata | 专家 pinned 在 RAM，未命中就地 CPU 算，与 GPU 并行 | 把 PCIe 从关键路径摘掉；CPU 算力成新上限 |

共同点：显存 = 热专家缓存；RAM = 全量专家。差别在「未命中怎么办」——搬回来（llama.cpp，慢）还是就地算（Strata/KTransformers，快但吃 CPU）。

---

## 7. CPU / 异构内核（对象 = 主机侧算力）

- 指令集是关键前提：AMX / AVX-512 才有高性能 CPU 侧 GEMM；只有 AVX2 时吞吐显著下降。
- 环境2 现实：Core Ultra 9 285K 属 Arrow Lake 桌面，无 AVX-512 / AMX（[硬件规格] 硬件规格）→ CPU 侧只能走 AVX2，与「仅 AVX2 的老 Xeon」同档。
- 含义：环境2 CPU 侧是「可用的第二算力」，但不是「免费的算力」——设计上要限制 CPU 承担的比例，并优先让 GPU 算热专家。
- NUMA 感知在大内存多路平台上重要；环境2 单路桌面，影响小。

---

## 8. 编译与图捕获（对象 = kernel launch 开销）

| 技术 | 效果 | 本项目 |
|------|------|--------|
| CUDA Graph | 消除 kernel launch 开销，对小 batch / 单流收益最大 | 中（单流场景有效） |
| torch.compile / ViT CUDA Graph | 视觉编码器加速 | 低（多模态为辅） |
| 算子融合（如 qwen4 fuse op） | 减少中间张量往返 | 中 |

---

## 9. 汇总矩阵：技术 × 作用瓶颈 × 环境2 优先级

| 技术 | 作用瓶颈 | 单/多卡 | 本项目优先级 |
|------|---------|--------|------------|
| NVFP4 权重量化 | 容量 + 带宽 | 单卡 | 高（环境2 原生加速） |
| 三值权重量化 | 容量 | 单卡 | 中（需 LUT 核，有损工具调用） |
| KV 量化 FP8 | 容量 + KV 带宽 | 单卡 | 高（低风险首选） |
| KV 量化 4-bit | 容量 + KV 带宽 | 单卡 | 高（256K×并发） |
| KV 量化 3-bit（TurboQuant 类） | 容量 + KV 带宽 | 单卡 | 中（待融合核 + 环境2 标定） |
| 稀疏 KV 检索（Quest 类） | KV 读带宽 | 单卡 | 低（QSA 已内置） |
| 投机解码（MTP / DFlash2） | 带宽（降延迟） | 单卡 | 高（MoE 卸载场景必选） |
| PagedAttention | KV 显存利用率 | 单卡 | 高 |
| 前缀缓存（Radix） | 重复 prefill | 单卡 | 高（agent 多轮） |
| 分层 offload（专家/KV） | 容量不足 | 单卡 → CPU/SSD | 最高（环境2 唯一可行主线） |
| continuous batching | 吞吐 | 单/多卡 | 中 |
| FlashAttention | attention 带宽/往返 | 单卡 | 中 |
| CUDA Graph | launch 开销 | 单卡 | 低 |
| 激活量化 W8A8 | 算力（batch>1） | 单卡 | 低（环境2 带宽瓶颈） |

---

## 10. 技术路线骨架

1. 容量层：分层 offload 是主线（权重永远装不下），专家缓存 + 就地 CPU 计算。
2. 带宽层：NVFP4 权重 + FP8/4-bit KV + 索引器 KV 独立精度。
3. 延迟层：MTP 投机（基线）+ prompt lookup + 可选的 DFlash2 类并行草稿。
4. 复用层：Radix 前缀缓存 + PagedAttention。
5. 差异层：n-gram 记忆表的显式预取与双缓冲（已被 vLLM 官方 recipe 与 Qwen 官方 README 占位，不再是空白——见 [05-engine-landscape.md](05-engine-landscape.md) §7.2；且 [02-qwen3.8-flash-next.md](02-qwen3.8-flash-next.md) §3.2 实测其收益前提不成立，两个理由都指向放弃该差异点）。

---

---

---

## 来源

| 编号 | 用于章节 | 内容 |
|------|---------|------|
| S-17、S-18、S-19 | §2.1 | 低比特 KV 量化方法 |
| S-20、S-21、S-22 | §4 | 并行起草 |
| S-01 | §3、§4、§5.1 | vLLM 官方部署配方 |
| S-23 | §5.1 | vLLM 前缀缓存设计文档 |
| S-24 | §1 至 §5 | 机制出处论文清单 |

等级、复核状态与 URL 见 [references.md](references.md)。
