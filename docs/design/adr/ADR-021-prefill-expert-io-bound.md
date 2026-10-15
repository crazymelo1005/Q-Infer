# ADR-021：prefill 是专家 I/O 受限，补齐 Q2_0 的 MMQ tile 不是杠杆

状态：Accepted（2026-10-15）

---

## 背景

双卡 prefill 实测约 950–1050 tok/s（3305-token 冷提示），距目标 3K 尚远。此前 README 与项目记忆把差距归因于「sm_120 上 q2_0（2560 行）无 MMQ tile」——即 llama.cpp 的 MMQ 配置表在 Blackwell 回退到的 ampere 表里缺 `GGML_TYPE_Q2_0` 条目，导致 `strata/src/prefill/moe_mmq.cu` 的 `fits()` 返回 false，该 GEMM 走非 MMQ 路径。本 ADR 记录对该归因的实测检验，结论是归因错误。

## 测量

构造一个带 Q2_0 MMQ 的引擎。`tools/build_engine.sh` 用 `STRATA_GGML_DIR` 指向本机 llama.cpp。本机部署用的 `~/llama.cpp`（commit `e9fa0781f`，2026）在其 `ggml-cuda/mmq-config-ampere.cuh` 里没有 `GGML_TYPE_Q2_0` 的 CASE 行（`ggml_cuda_mmq_get_config_blackwell` 对 Q2_0 回退到该表，故取不到配置）；而 Strata 自带的 llama.cpp 分支（`~/projects/strata-run/third_party/llama.cpp`，commit `6674a006`）有 16 行 Q2_0 配置（`MMQ_ITER_K` + `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0`），且两者 MMQ API 同代（同为拆分式 per-arch 配置 + 4 参 `ggml_cuda_mmq_get_config`），可编译。

用该分支重建引擎到 `build-engine-q2mmq/`：configure + build 全绿，`mmq-instance-q2_0.cu` 入库。竖测 A/B（环境2，真权重 sc117 IQ2_XS，单卡 `--gpu 0`，`--mmap-experts --max-context 32768 --kv int8 --expert-cache auto --spec 2 --prefill 512 --pool-workers 15`，外加 `STRATA_PF_FUSED=1 STRATA_PREFILL_CPU_SHARE=auto`；一 shot `strata` 生成模式，同一 token 序列）：

| 提示 tokens | 引擎 | `no tile for Q2_0` 告警 | prefill |
|---|---|---|---|
| 3299 | 旧（无 Q2_0 MMQ 配置） | 有 | 9728.4 ms → 339.1 tok/s |
| 3299 | 新（分支，MMQ 生效） | 无 | 10092.2 ms → 326.9 tok/s |
| 13199 | 旧 | 无（该次未触发告警） | 37142.9 ms → 355.4 tok/s |
| 13199 | 新（MMQ 生效） | 无 | 36628.0 ms → 360.4 tok/s |

MMQ 开与关的差异约 1%（噪声内），13K 提示甚至新旧几乎相同。告警走 `fits()`，每种类型只报一次，故单次读数有无告警不足以判定，速度才是判据。

瓶颈在哪：13K 提示的引擎自报明细（新旧一致）——`prefill 13199 tokens in 26 chunks, 36582 ms (360.8 tok/s); experts streamed 181255 (0 by DMA, host 25192.9 ms), resident 59670, on the CPU 0 (share 0.00)`，且 `prompt copies 263715.5 MB`。即 36.6 s 里 25.2 s（约 68%）花在宿主侧专家流式搬运（18 万个专家 blob、263 GB 拷贝），GEMM 算术只占小头；CPU 池份额 `share 0.00`，prompt 路径几乎不吃 CPU 池。

## 决策

1. 不再把 prefill 的 3K 差距归因于 MMQ tile，修正 [README](../../../README.md) 与项目记忆的表述。
2. 不为 prefill 换用分支 llama.cpp：收益约 1%，却引入「引擎构建依赖一个非 pin 的 llama.cpp 分支」的维护成本（每次升引擎都要重验），不划算。
3. prefill 的真实杠杆是专家 I/O，不是算术——提高专家缓存命中率（用满 VRAM）、提升 PCIe 搬运效率、降低每 token 的专家 miss。这三项都在引擎侧/厂商侧，超出本仓库可单方面改动的范围。

## 后果

- prefill 维持约 950–1050 tok/s（双卡、3.3K 提示）为当前现实上限；decode 目标（100 tok/s）已达成（实测 100–136）。
- 若日后要追 3K，应从「专家常驻/缓存几何」与「异步批量搬运」入手，而不是 kernel 算术强度。
- 附带证据：单卡同配置 prefill 仅约 360 tok/s，对双卡约 1K，说明双卡（`--layer-split auto` + 更大 `--expert-cache`）主要靠降低 miss 率取胜，进一步支撑 I/O 受限的判断。
- `build-engine-q2mmq/` 为本次实验的临时构建目录（已被 `.gitignore` 的 `build-*/` 覆盖），验证完即删，不进仓库。

## 补充：核显稠密投影在真双卡配置下拖慢 prefill

同一轮里还给 `--igpu` 做了默认值决策，先错后改，这里留档。判据是同一台机器、同一服务、真双卡配置（`--layer-split auto --prefill 8192 --mtp --spec 4 --pipeline-windows 2 --pool-workers 15 --trim-stage-weights`，262144 上下文），只切 `--igpu off|dense`，同一条 2319-token 提示：

| iGPU | prompt 读入 | prefill | decode |
|---|---|---|---|
| off | 3384 ms | 685.2 tok/s | 77.8 tok/s |
| dense | 4677 ms | 495.8 tok/s | 82.3 tok/s |

即 dense 拖慢 prefill 约 28%（核显上的稠密计算与 GPU 串行，且专家路径越快这部分占比越低）。此前在单卡、小专家缓存下单测得到 dense 小幅正收益（300-token 提示 178.8→187.5 tok/s），那是 dense 占比被放大后的假象，不能外推到部署配置。结论：`--igpu` 默认维持 off，显式 opt-in。这条同时说明 ADR-019 的判断（核显不适合这条推理路径）在小批形状之外的批形状上依然成立。

另一处副产品：引擎在部署配置下自报 VRAM 只剩 444 MiB（低于其 512 MiB 阈值），并给出建议——给 engine-arg 加 `--vram-reserve-mib 768`，或降 `--max-context`/`--batch`；不处理时驱动可能把显存页到系统内存，decode 会掉到个位数。已记入运行手册。

## 补充二：五个加速杠杆的实测，全部被证伪

2026-10-15 又按「补充 llama.cpp/strata 的不足」逐条试过，都是真双卡同配置、同一条 2319-token 提示。结论：没有一个能让 prefill 往上走。

| 杠杆 | 做法 | prefill | 判定 |
|---|---|---|---|
| Q2_0 MMQ tile | 用带 16 条 Q2_0 配置的 llama.cpp 分支重建引擎（`no tile for Q2_0` 告警消失、MMQ 生效） | 639.4 tok/s | 对基线 685.2 是 −6.7%，不是收益 |
| 常驻专家（走 DMA） | `--resident-experts` 替代 `--mmap-experts`，让预填路径直读固定内存 | 245.1 tok/s | −64%。本机装不下（要 11.7 GiB 常驻，可用 15.3 GiB 减 4 GiB 余量后不够），退化成文件读 + 自适应换入 |
| KV 缩容换专家槽 | `--kv k8v4`（816 B/格 vs int8 的 1056） | 212.1 tok/s | −69%。旋转 Q4_0 V 的注意力代价远大于省下的显存 |
| 核显稠密投影 | `--igpu dense` | 见下 | −13% ~ −28%，负 |
| 核显专家池 | `--igpu pool` | 0.36 tok/s | 见 ADR-020 |
| 常驻 CPU 专家（部分） | `--mmap-experts --resident-cpu-experts` | 566.1 vs 基线 601.9 | −6%（decode 73.5 vs 84.5，−13%）。把 cgroup 上限提到 28G 后确已常驻 15.26 GiB，仍更差 |
| stager 线程/批量 | `STRATA_STAGER_THREADS=8` / `STRATA_STAGER_BATCH=4` | 633.1 / 645.7 | −3.6% / −1.7%，默认（4 线程、batch 1）已最优 |

核显这项用交替 A/B 复核过（同一服务、同一条提示、两两相邻）：dense 456.4 / off 565.5、dense 520.3 / off 560.6——两对方向一致，dense 平均慢约 13%。故维持默认 off。

**测量方差警示（重要）**：同一份基线配置在不同重启间实测到 418、563、565、656、685、686 tok/s 的跨度（同一提示）。所以跨重启的绝对值不可直接比较；只有相邻/交替的成对读数可用（同一稳定状态下成对读数可复现到 0.2%——默认臂两次 657.0/655.7）。上表多数杠杆的效应远大于该带宽，但 iGPU 的 −13% 与噪声上限接近，需按交替 A/B 判定。

结论：这台机器上该引擎/模型的 prefill 天花板受制于「mmap 页缓存模式下逐专家走 CPU 拷贝（`0 by DMA`）」这一条——而启用 DMA 所需的常驻固定内存本机给不起。**3K 目标在本机可达范围之外**，除非放宽内存（换更大内存条或独占机器）或改造引擎的专家搬运路径。