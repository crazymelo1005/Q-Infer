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