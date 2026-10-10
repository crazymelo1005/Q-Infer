# 出处总表

本文登记全部仓库外来源。正文引用写作 `[S-n]`；编号只增不改。仓库内部引用不使用本文，改用相对链接加章节号。

引用日期是引用该来源的文档所记录的日期，不是来源自身的发布时间。

## 1. 来源等级

| 标记 | 含义 |
|------|------|
| [环境2实测] | 在环境2 上读自运行中的引擎 API、本地 benchmark，或直接读取其上的模型与配置文件 |
| [环境1实测] | 在本机（环境1）上同口径取得。其平台是 Windows 宿主内的 WSL2，引用时必须写明测在 WSL 内还是 Windows 宿主上 |
| [硬件规格] | 厂商规格页，是上限而非实测 |
| [官方] | 模型、引擎的官方发布页、官方文档或论文 |
| [他卡实测] | 社区在其它硬件上的实测，趋势可信、绝对值浮动 |
| [推算] | 由模型估算，须标定后方可作为结论 |

## 2. 复核状态

| 标记 | 含义 |
|------|------|
| [已确认] | 已逐位确认原文 |
| [摘要级] | 仅见于标题或摘要 |
| [存疑] | 来源矛盾，已降级 |

## 3. 来源清单

| 编号 | 等级 | 复核状态 | 来源 | 引用日期 | 使用处 |
|------|------|---------|------|---------|--------|
| S-01 | [官方] | [已确认] | vLLM 官方部署配方，https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next | 2026-10-09 | research/02、03、04、05 |
| S-02 | [官方] | [已确认] | 上游引擎仓库与文档 `github.com/Niko1221/Strata`：README、HOW_IT_WORKS、DETAILS、MULTI_GPU、BATCHING、INTEL_ARC、COMMUNITY_BENCHMARKS | 2026-10-09 | research/01、05、design/risks |
| S-03 | [环境2实测] | [已确认] | 环境2 上运行中的引擎实例 HTTP API（/metrics、/props、/health、/slots；内网地址不入库） | 2026-10-09 | research/01、05、design/engine、requirements |
| S-04 | [他卡实测] | [已确认] | 2×RTX 2080 Ti + 64 GB + AVX2 Xeon，IQ3_XXS 71 GB，262K，75 tok/s，命中率 97.7% | 2026-10-09 | research/01、05 |
| S-05 | [他卡实测] | [已确认] | 上游速度表与社区实测汇总（2×2080Ti、RTX 5070、4090 等） | 2026-10-09 | research/01、05 |
| S-06 | [他卡实测] | [摘要级] | 同类引擎在 RTX 5090 上的实测，社区转载 | 2026-10-09 | research/05 |
| S-07 | [官方] | [摘要级] | 主案例模型的官方发布与体积口径，https://blog.csdn.net/student_upupgpu/article/details/164122465 | 2026-10-09 | research/02 |
| S-08 | [摘要级] | [摘要级] | 主案例模型的架构与显存总览，https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory | 2026-10-09 | research/02 |
| S-09 | [摘要级] | [摘要级] | 索引器与稀疏注意力几何、卸载路径，https://blog.csdn.net/weixin_55357163/article/details/164209835 | 2026-10-09 | research/02 |
| S-10 | [摘要级] | [摘要级] | 索引器相对延迟（0.25）与门控公式，https://hyper.ai/cn/papers/Qwen3.8-Flash-Next | 2026-10-09 | research/02 |
| S-11 | [摘要级] | [摘要级] | 多模态 MoE 综述（外挂 51B n-gram 表），https://m.blog.csdn.net/qq_35812205/article/details/164190103 | 2026-10-09 | research/02 |
| S-12 | [摘要级] | [摘要级] | 开源发布信息（125B、激活 6B），https://www.toutiao.com/a7680202469007229455/ | 2026-10-09 | research/02 |
| S-13 | [官方] | [已确认] | Qwen4 时间线与架构预览，https://cellcog.ai/blog/qwen-4-release-date/ | 2026-10-09 | research/03 |
| S-14 | [官方] | [摘要级] | 官方发布稿（下代架构前期预览），https://techapple.com/archives/62099 | 2026-10-09 | research/03 |
| S-15 | [摘要级] | [摘要级] | 百科条目，https://baike.baidu.com/item/Qwen3.8-Flash-Next/68707034 | 2026-10-09 | research/03 |
| S-16 | [摘要级] | [摘要级] | 开源动态（许可证趋势、下代预览），https://aihot.virxact.com/items/cmtatjy570946roamigoxjc29 | 2026-10-09 | research/03 |
| S-17 | [官方] | [已确认] | 低比特量化方法的算法管道与自测数字，http://www.chenxutan.com/d/3269.html | 2026-10-09 | research/04 |
| S-18 | [官方] | [摘要级] | 同一量化方法的官方口径（3-bit、6×、8×），https://paooo.com/aigc-news/11107/ | 2026-10-09 | research/04 |
| S-19 | [摘要级] | [摘要级] | 该量化方法的内存市场影响与社区复现讨论，https://dev.to/yang_goufang_23c7ba674984/its-not-smarter-models-its-cheaper-memory-turboquants-real-impact-wall-street-panic--48jl | 2026-10-09 | research/04 |
| S-20 | [官方] | [摘要级] | 并行起草模型的模型卡与提速口径，https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 与 https://navxd.com/navigation/tool/qwen3-8-27b-dflash2/ | 2026-10-09 | research/04 |
| S-21 | [他卡实测] | [摘要级] | 并行起草模型的生产落地实测（2×3090），https://m.blog.csdn.net/chen1415886044/article/details/165364043 | 2026-10-09 | research/04 |
| S-22 | [摘要级] | [摘要级] | 并行起草的原理与推理框架实现，https://m.blog.csdn.net/weixin_44778145/article/details/163957992 | 2026-10-09 | research/04 |
| S-23 | [官方] | [已确认] | vLLM 前缀缓存设计文档（块哈希），https://docs.vllm.ai/en/latest/design/prefix_caching.html | 2026-10-09 | research/04 |
| S-24 | [官方] | [摘要级] | 机制出处论文清单：PagedAttention arXiv:2309.06180、KIVI arXiv:2402.02750、KVQuant arXiv:2401.18079、H2O arXiv:2306.14048、SnapKV arXiv:2404.14469、StreamingLLM arXiv:2309.17453、Quest arXiv:2406.10774、RadixAttention arXiv:2312.07104、Sarathi-Serve arXiv:2308.16369、Orca arXiv:2211.05102、Fiddler arXiv:2407.07001、Mixtral-offload arXiv:2401.10774、HetuMoE arXiv:2305.19759、Roofline DOI 10.1145/1498765.1498785 | 2026-10-09 | research/04 |
| S-25 | [官方] | [摘要级] | 主案例模型的官方说明：n-gram 表可异步卸载到主机内存并与计算重叠（经 S-01 引述） | 2026-10-09 | design/risks |
| S-26 | [官方] | [已确认] | KV 上下文虚拟化实现仓库 README，`github.com/kvmem/kvmem-llama.cpp` | 2026-10-09 | design/risks、research/05 |
| S-27 | [官方] | [摘要级] | KV 上下文虚拟化论文摘要，arXiv 2609.04852 | 2026-10-09 | research/05 |
| S-28 | [摘要级] | [摘要级] | 预测式专家缓存方向的工作，arXiv 2410.17954（DAC'26） | 2026-10-09 | design/proposals |
| S-29 | [摘要级] | [摘要级] | 双阶段预取与缓存的专家服务工作，arXiv 2509.07379 | 2026-10-09 | design/proposals |
| S-30 | [摘要级] | [摘要级] | 学习专家使用模式的缓存项目 colibri（项目页） | 2026-10-09 | design/proposals |
| S-31 | [硬件规格] | [已确认] | Intel ARK 与 NVIDIA 官方规格页（CPU、GPU、内存、存储） | 2026-10 | hardware.md |
| S-32 | [官方] | [已确认] | 板卡厂商规格页：RTX 5060 Ti 的总线接口写作「PCI Express Gen 5 x16 (uses x8)」，即电气宽度为 x8，https://www.msi.cn/Graphics-Card/GeForce-RTX-5060-Ti-16G-GAMING-OC/Specification | 2026-10-09 | hardware.md、requirements.md、gates.md |
| S-33 | [环境2实测] | [已确认] | 模型 GGUF 元数据（由 `measure/gguf_meta.py` 直读文件头，不经第三方库）：`general.architecture=qwen4exp`；主注意力 `head_count` 24 / `head_count_kv` 2 / `key_length` 256 / `value_length` 256 / `embedding_length` 2560；索引器 4×128、`top_k` 2048；`expert_count` 512、`expert_used_count` 10、`block_count` 48、`compress_ratios` 12 个 4；表张量 `per_layer_token_embd.weight` dims [160, 320001536] | 2026-10-09 | research/02、gates.md |
| S-34 | [官方] | [已确认] | 参考引擎的 n-gram 哈希与 PLE 行读取实现、专家画像与路由轨迹的格式、以及 IQ4_NL 行的码本与分裂式半字节序（其源码，在环境2 上就地读取）：`src/kernels/ngram.cpp` 的 `ngram_rows`、哈希常量与 `kvalues_iq4nl` 码本、`include/strata/kernels/ngram.hpp` 的几何、`src/kernels/ple_oracle_vectors.inc` 的 6 组 oracle 向量、`src/ngram/ple_reader.cpp` 的 8 路组相联行缓存、`tools/gguf_reader.py` 的 GGUF v3 头部与张力信息布局、`tools/make_profile.py` 的 STRP 画像与 `--dump-routing` 轨迹格式、`src/program/generate.cpp` 关于覆盖曲线偏离幂律的自注（称 5,805 对覆盖 94.9%，实测 69.2%） | 2026-10-09 | research/01、research/02、src |
| S-35 | [官方] | [已确认] | 专家权重块格式 IQ2_XS 与激活块格式 Q8_K 的编码定义及参考实现，来自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors）：`ggml/src/ggml-common.h` 的 `block_iq2_xs`（l.389，74 字节 = fp16 尺度 + 32×u16 码字 + 8×u8 尺度）、`kmask_iq2xs`（l.509）、`ksigns_iq2xs`（l.513）、`iq2xs_grid`（l.627，512 项、每项 8 个幅度字节，取值 8/25/43）、`block_q8_K`（l.371）；`ggml/src/ggml-quants.c` 的 `dequantize_row_iq2_xs`（l.2516）、`nearest_int`（l.621）、`quantize_row_q8_K_ref`（l.2768）；`ggml/src/ggml-cpu/quants.c` 的 `ggml_vec_dot_iq2_xs_q8_K_generic`（l.948）；另有同仓库 gguf-py 的独立实现 `gguf/quants.py` 的 `IQ2_XS.dequantize_blocks` 用作回归 oracle | 2026-10-10 | src、measure |
| S-36 | [环境2实测] | [已确认] | 参考引擎实际加载的那个模型的逐张量量化档分布。文件是社区量化 `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS`（文件名与 `general.file_type=20`（MOSTLY_IQ2_XS）只表示总量级）：第一分片 1223 个张量的类型计数为 BF16 484、F32 292、IQ4_XS 142、Q8_0 96、IQ2_S 68、Q2_0 48、IQ3_S 37、Q6_K 27、IQ2_XXS 22、IQ1_M 6、F16 1，**没有任何 IQ2_XS（类型 17）**。路由专家是逐层换档的：`ffn_{gate,up}_exps.weight` dims [2560,640,512] 为 IQ2_S 34 层 / IQ2_XXS 11 层 / IQ1_M 3 层（IQ2_XXS 在 1、4、9、10、11、14、18、19、27、29、30 层，IQ1_M 在 8、13、37 层），`ffn_down_exps.weight` dims [640,2560,512] 48 层全部为 Q2_0；共享专家 gate/up ∈ {IQ3_S, IQ4_XS, Q6_K}、down 全为 Q8_0；路由器 `ffn_gate_inp.weight` [2560,512] 为 BF16。专家字节合计 35,454,976,000 B（33.0 GiB，等效约 2.35 bpw）。来源是两条独立入口：`measure/gguf_meta.py --tensors --limit 5000` 直读 GGUF 张力表，以及引擎侧的 `packs/sc117_iq2_xs/native_experts.txt`（其 `tools/iq_pack.py` 从同一 GGUF 生成，逐层记 `gu_type d_type` 与 gate/up/down 三个矩阵的绝对偏移） | 2026-10-10 | research/01、research/02、engine.md、requirements.md、glossary |
| S-37 | [官方] | [已确认] | Q2_0 与 Q8_0 两种块格式的编码定义及参考实现，来自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，MIT，Copyright (c) 2023-2026 The ggml authors）：`ggml/src/ggml-common.h` 的 `block_q2_0`（l.187，`QK2_0 = 64`、18 字节 = fp16 尺度 + 16 字节 2-bit 码，码 {0,1,2,3} → {-1,0,+1,+2}）与 `block_q8_0`（l.251，`QK8_0 = 32`、34 字节）；`ggml/src/ggml-quants.c` 的 `dequantize_row_q2_0`（l.439）、`dequantize_row_q8_0`（l.553）、`quantize_row_q8_0_ref`（l.276）；`ggml/src/ggml-cpu/quants.c` 的 `ggml_vec_dot_q2_0_q8_0_generic`（l.177，一个 Q2_0 块配两个 Q8_0 块）。块大小不照抄传闻：`QK2_0 = 64` 由部署模型的字节数反推（down 每专家 460,800 B ÷ 1,638,400 元素 = 2.25 bpw；若按 128 则为 2.125 bpw），并与引擎自身 `strata-dequant` 在同一 64 块窗口给出一致的 CHECKSUM（sum 0.9335670471 / sumabs 31.97625351） | 2026-10-10 | src、measure、glossary |

## 4. 登记规则

- 新增来源即分配下一个 `S-n`，填入本表，并回填正文引用。
- 等级必须等于真实来源层级，不得把不同来源的数字混成一句结论。
- 同一来源在多个文档使用时共用同一编号。
- 来源被推翻时，把复核状态改为 [存疑] 或补注矛盾点，不删除编号。
