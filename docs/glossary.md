# 术语表（Glossary）

> 本仓库横跨「模型架构 / 推理引擎 / 消费级硬件 / 量化」四个领域，缩写极多。本表是唯一术语口径：任何文档里出现的术语，以本表释义为准；本表本身不含主张，只定义。
> 事实来源仍以 `research/` 为准，设计取舍以 `design/` 为准。

---

## 一、模型架构（主案例 Qwen3.8-Flash-Next）

| 术语 | 含义 |
|------|------|
| MoE（Mixture of Experts，混合专家） | 每层用路由器从众多「专家」里选少数几个参与计算。总参数量大、单 token 计算量小 |
| 超稀疏 MoE | 激活参数占总参数比例极低。本案例约 6B / 125B ≈ 4.8%，从 24,576 个专家里选 top-10 |
| 专家（expert） | MoE 里的一个子网络（通常是 FFN）。本案例共 24,576 个 |
| 共享专家 / 稠密部分 | 每 token 必用的权重（注意力、router、共享专家、输出头等），必须常驻显存 |
| router（路由器） | 每层决定「这个 token 该发给哪几个专家」的小网络 |
| top-k / top-10 | 每个 token 激活的专家数。本案例 top-10 / 512（即从每 token 的 512 个候选里选 10 个） |
| GDN（Gated DeltaNet） | 一种线性注意力。本案例 48 层中占 36 层。特点是循环状态大小固定、不随上下文长度增长 |
| QSA（稀疏注意力） | 一种块稀疏注意力。本案例占 12 层。特点是 KV 随上下文长度增长，但有「索引器挑块」的稀疏读 |
| GR（Gated Residual） | 残差门控：读路径用低秩瓶颈网络算逐元素门控（四分支加权平均），写路径每分支只用一个标量门。权重属「每 token 必用」→ 必须常驻显存（`research/02` §2.2） |
| 索引器（indexer） | QSA 里负责「给每个 KV 块打分、挑出要读的块」的部件；它自己也有 KV 缓存（索引器 KV） |
| MTP（Multi-Token Prediction，多 token 预测） | 模型自带的「一次预测多个 token」的头。本案例是一个 4B 的 MTP 头，可直接用于投机解码 |
| n-gram 记忆表 / 外挂表 / PLE | 一块 51B 的外挂嵌入表：把前若干 token 拼成 key，查出向量叠加回表示。加容量不加计算；访问稀疏且可提前算出 |
| PLE（Per-Layer Embedding） | 上游（vLLM）对「外挂记忆表」的称呼，如环境变量 `VLLM_PLE_CPU_OFFLOAD` |
| KV cache | 注意力在 decode 时缓存的 Key/Value，避免每步重算。长上下文的显存大户 |
| 上下文长度 / 262K | 一次能「记住」的 token 数。本案例原生 262,144（262K） |

## 二、推理过程与指标

| 术语 | 含义 |
|------|------|
| prefill（预填） | 处理输入提示词，一次性算完所有输入 token，产出第一段 KV。计算密集 |
| decode（解码） | 逐 token 生成。访存密集（每次只算 1 个 token，却要读几乎全部权重）→ 瓶颈常是带宽 |
| TTFT（Time To First Token） | 从发出请求到吐出第一个 token 的时间。主要吃 prefill |
| tok/s | 每秒生成的 token 数（decode 吞吐）。本仓库所有 speed 数字的默认单位 |
| P50 / P99 | 延迟的中位数 / 第 99 百分位。P99 用来衡量「最坏的那 1%」长尾 |
| 前缀缓存 / prefix cache | 多个请求共享相同前缀（如 system prompt）时，复用已算好的 KV，避免重复 prefill |
| RadixAttention / Radix 树 | SGLang 的前缀复用实现，用基数树管理共享前缀 |
| PagedAttention | vLLM 的 KV 分页方案：把 KV 切成块（page），像操作系统管理内存一样按需分配，消除碎片 |
| 批调度 / batch slots | 同时处理多个请求。Strata 默认 1，上限 8；本机已开 4 槽 |
| Roofline（带宽上限） | 用「有效带宽 ÷ 每 token 字节量」推出的 decode 理论天花板。实测低于它，说明瓶颈在别处 |
| lane / 会话 lane | KVMem 里为每个并发会话独立分配的 KV 工作区（如 `--kvmem-conversations N`） |

## 三、量化与精度

| 术语 | 含义 |
|------|------|
| 量化 | 用更少的比特表示权重/激活，换取更小的体积与带宽占用 |
| FP8 / FP16 / BF16 | 8 位 / 16 位浮点。BF16 动态范围大，FP8 更省 |
| NVFP4 | NVIDIA 的 4-bit 浮点格式，Blackwell（sm_120）原生支持 |
| GGUF | llama.cpp 生态的模型文件格式，内含多种量化档 |
| IQ2_XS / IQ3_S / IQ3_XXS / IQ4_XS | GGUF 的「I-quant」（重要性加权）量化档：约 2-bit / 3-bit / 4-bit。数字越小越省、质量越低。`IQ3_S` 是 3-bit 级的另一档，见于上游多卡实测（`research/01` §9.4） |
| Q2_0 / Q5_K_XL | GGUF 的「K-quant / 传统」档位（对比用） |
| AWQ / GPTQ | 两种主流量化算法（权重分组的 4-bit 量化） |
| KV 量化 | 对 KV cache 量化（如 `--kv-cache-dtype fp8`），长上下文的省显存关键 |
| indexer_kv_dtype | vLLM 里单独给索引器 KV 设精度的旋钮 → 说明「索引器精度」是一个独立维度 |
| 4-bit / 3-bit（TurboQuant、SAW-INT4） | 更低比特的实验档；3-bit 类方法需融合核 + 本机标定 |
| perplexity（困惑度） | 语言模型质量的经典指标。越低越好，但对量化差异不敏感，须辅以任务集 |
| 贪婪解码友好 | 某些量化方法的残差方差会改变 top-1（贪心）选择；需专门测，不能只看平均误差 |

## 四、推理引擎与名词

| 术语 | 含义 |
|------|------|
| Strata | 本项目的参考范本：唯一为「消费级 + 超大 MoE」从头设计的引擎，专家三级分层 + 未命中就地算 + 内建 MTP |
| vLLM / SGLang / NInfer / TensorRT-LLM | 服务端派引擎：假设权重能装进显存，优化「算得快、塞得密」 |
| llama.cpp / KTransformers | 卸载派引擎：承认权重装不下，做层级/专家级 offload |
| KVMem | llama.cpp 的 fork：把 KV/工作区做三层虚拟化，同款 5060 Ti 16GB 上做到 256K 近无损 |
| LMCache | 把 KV 当作独立缓存层的库，常配 vLLM/SGLang |
| Q-Infer（擎） | 本项目的引擎工作名。「Q」兼指 Qwen（主案例）与 Quantized（量化）；「擎」取「擎起」——用 16GB 擎起 125B。GitHub 仓库名与本地目录名均为 `Q-Infer`。（旧工作名 `Kestrel` 已弃用，原因：与微软 ASP.NET Core 的 Kestrel Web 服务器撞名） |
| offload（卸载） | 把权重/KV 放到显存之外（主存 / SSD），用时再取 |
| 专家缓存 / 命中率（hit_rate） | 常驻显存的专家占「本步需要的专家」的比例。上游定义里它不保证与速度同向 |
| expert_slots | 显存里能放下多少个专家的槽位数 |
| arena | 启动时一次性分配的连续缓冲区（专家/表/表行的常驻区） |
| machine profile | 自动标定产出的机器画像（PCIe 带宽、内存带宽、内核吞吐等落盘参数） |
| 预测性预取 | 在数据被需要之前主动把它搬到位（如提前 1 步算表行索引） |
| 双缓冲（double buffering） | 用两块缓冲轮流：一块服务当前步，另一块在后台被填充 |
| lookahead / `STRATA_LOOKAHEAD` | Strata 的跨层预测性预取开关；深度由 `STRATA_LOOKAHEAD_K` 定 |
| `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE` | Strata 的文件层整块预取与「暂存式」预取变体。后者在 16GB 内存 lane 实测 −32.6%、32GB lane +25.7%，故默认关闭（见 [design/gates.md](design/gates.md) 的 G-10） |
| 共现图 / co-occurrence | 维护「同一 token 或同一会话内被一起激活」的专家对统计，作为专家替换策略的输入。属概率性预测，与 §4.2 的确定性预取不同级 |
| 三类流量 | 过 PCIe 的三类搬运：专家权重、n-gram 表行、KV。区别于「四条数据流」——「专家流（RAM→CPU）」走内存总线，不过 PCIe。§4.1 的字节预算仲裁的对象就是这三类 |
| scatter-add | 把 CPU 与 GPU 各自算出的专家结果按专家 slot 累加合并到同一输出缓冲区（[engine.md](design/engine.md) §6） |
| prompt lookup / suffix | 一种投机起草：从上下文已有文本里抄若干 token 当草稿（最多 5）。代码改写类提速 6–11%，其它文本基本不变（`research/01` §6 / `research/04`） |
| Mooncake / FlexGen | 对照系统：Mooncake 把 KV 当独立一层做跨请求复用；FlexGen 做带宽约束下的规划。二者都没有「专家 + 表行 + KV 三类流量的同步预算」 |
| 投机解码 | 先由小模型/头草拟 N 个 token，再由主模型一次前向验证；被拒则回退。数学上不改变输出分布 |
| 草稿 token / 接受率 | 草稿被主模型接受的比例。接受率决定投机收益 |
| DFlash2 / 外挂草稿模型 | 「并行草稿」类投机方案，需要额外训练好的草稿器（本项目优先用自带 MTP） |
| SPEED-Bench | 测投机/MTP 收益的真实任务集；禁用随机 prompt（会让结论反转） |
| LongBench / SWE-bench / LongMemEval / AgentLongBench | 长文 / 代码 / 长记忆 / 智能体四类质量评测集 |
| TP / EP / PP（张量/专家/流水线并行） | 多卡切分策略。每层同步的 TP 会把慢链路放进关键路径，本项目禁用（ADR-004） |
| 按层切分（layer-split） | 把不同的层分给不同的卡，各卡独立跑 |
| 专家并行（expert-parallel） | 各卡各持一部分专家的常驻副本 |
| 共享主存专家池 | 双卡/双实例不必各自复制专家与表——主机内存是共享资源，由统一页表管理 |

## 五、硬件与系统

| 术语 | 含义 |
|------|------|
| VRAM / 显存 | GPU 板载内存。本机单卡 16GB（其他进程占用后可用 ≈13.5–14GB） |
| Host RAM / 主存 | 系统内存。本机 64GB DDR5-4400（有效带宽 ≈70GB/s） |
| pinned memory（页锁定内存） | 不会被操作系统换出的主机内存，GPU 可直接 DMA 读写，是 offload 的常驻区 |
| PCIe | 显卡与主机的总线。本机 5.0 x8+x8；是所有「搬数据」动作的公共瓶颈 |
| B_step / PCIe 预算 | 每个 decode 步允许搬运的字节数上限（由标定确定），用来给多路流量排优先级 |
| io_uring / OVERLAPPED | Linux / Windows 的异步批量 IO 接口 |
| mmap / MADV_WILLNEED | 把文件直接映射进地址空间 / 提示内核提前读入，避免「先读进 RAM 再拷贝」 |
| CUDA Graph | 把一串 kernel 调用捕获成图一次提交，消除小 batch 下的 launch 开销 |
| AVX2 / AVX-512 / AMX | CPU SIMD 指令集。本机（Arrow Lake）只有 AVX2，无 AVX-512/AMX → CPU 侧是明确上限 |
| P-core / E-core | 性能核 / 能效核。本机 285K 为 8 P + 16 E，内核需按核型分池 |
| iGPU / 核显 | CPU 集成的显卡（Intel Xe）。与 CPU 共享内存总线 → 不增加带宽，但可分担算力 |
| NPU / Intel AI Boost | 低功耗神经网络加速器（≈13 TOPS INT8）。只用于视觉/小模型，不参与专家计算 |
| XMX / DP4a | Intel 核显的矩阵/整数点积指令（`DP4a` 是低精度点积） |
| oneAPI / SYCL / Level Zero | 调用核显的软件接口（本项目倾向这套） |
| re-RoPE | KVMem 把历史块「还原到正确位置编码」再物化进工作集的操作 |
| Mean-K 索引 | KVMem 用来选块的轻量索引 |

## 六、本仓库的工作约定

| 记号 | 含义 |
|------|------|
| [本机实测] / [硬件规格] / [官方] / [他卡实测] / [推算] | 五级来源等级，定义见 [research/references.md](research/references.md) 第 1 节 |
| [已确认] / [摘要级] / [存疑] | 三级复核状态，定义见 [research/references.md](research/references.md) 第 2 节 |
| `S-n` | 仓库外来源编号，登记于 [research/references.md](research/references.md) |
| `ADR-NNN` | 架构决策记录，位于 [`design/adr/`](design/adr/)；不可修订，改变决策须新开一条 |
| `G-NN` | 门禁与待测项，登记于 [design/gates.md](design/gates.md)；六条门禁为 G-04、G-07、G-08、G-09、G-10、G-11 |
| `R-NN` | 风险，登记于 [design/risks.md](design/risks.md) |
| `P-NN` | 候选方向，登记于 [design/proposals.md](design/proposals.md) |
| 基线（分母） | 在目标机上实测的参考引擎性能，作为全部倍数指标的分母 |
| 评测口径（尺子） | 固定任务集与随机种子、每配置重复不少于 5 次、报中位数与区间 |
| 负结果 | 被实测证伪的假设，必须写入 [design/risks.md](design/risks.md) 与 `research/`，不得删除 |
