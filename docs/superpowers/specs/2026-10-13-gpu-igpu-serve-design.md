# GPU / 核显全量接入设计

本文是服务路径异构接入的执行规格：在 [engine.md](../../design/engine.md) §6（异构执行）、§13（核显与 NPU）、§14（单卡与双卡）与 [ADR-004](../../design/adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md)、[ADR-005](../../design/adr/ADR-005-igpu-npu-as-coprocessor.md)、[ADR-011](../../design/adr/ADR-011-compute-group-abstraction.md) 之上，把一次 decode 步的计算与存储按启动参数与实际硬件自动分工，最大化利用 CUDA 双卡、核显与 CPU。范围以 Strata 引擎的 `--native` 全家桶与 llama.cpp 的 `--n-gpu-layers`/`--split-mode layer`/KV 上卡为对齐基准（见第 1 节缺失清单）。

## 1. 目标与边界

目标：启动一次服务，按 `--gpu`、`--igpu-frac`、`--kv-resident` 与探测到的硬件，自动决定每块硬件的分工（dense 层分布、KV 位置、专家槽位、核显占比、CPU 兜底），使推理性能最大化。这条线完成时，一次 decode 步的稠密段、命中专家、未命中专家与 KV 读写分别落在与其能力匹配的硬件上，且每阶段独立可验收。

对比结论（Strata 引擎 help + llama.cpp 标准做法核实）：

| 维度 | Strata / llama.cpp | 现状 | 本设计 |
|------|--------------------|------|--------|
| dense 层（attention/GDN/GR/router/head） | 全部上卡（`--native` 全家桶 / `--n-gpu-layers` 默认全量） | 全主机侧 | 上卡（阶段 A） |
| KV cache | 默认全在显存（`--kv-resident` 是移出显存换专家槽位） | `KvHostStore` 恒在主机 | 默认上卡（阶段 B） |
| 未命中专家 | 核显 / CPU 分组并行（Strata 无核显路径；llama.cpp 无） | CPU 单组 | 核显优先、CPU 兜底（阶段 C） |
| 多卡层权重 | `--trim-stage-weights` 每卡只载自己层的 dense | 每卡全量 | 按卡裁剪（阶段 D） |
| prefill | GPU 大批量矩阵乘（394-436 tok/s） | CPU（2.5 分钟 / 53 token） | 随稠密段上卡而改善 |
| 启动分工 | 部分自动（`--layer-split auto`、`--expert-cache auto`） | 手动 | 探测 + 求解器自动分配表 |

范围外（后续阶段，本文不展开）：

- MTP 层上卡：阶段 D 含共享专家与头，MTP 草稿模型上卡另列。
- 核显视觉编码器（ADR-005 的无条件三处之一）：随多模态一起。
- 长上下文下 KV 的量化档（`--kv int8/q4_0`）：阶段 B 只做 fp16 上卡，压缩档按 Strata 的用时曲线另列。
- 运行中显存弹性（Strata `--vram-elastic`）：本文只做启动时分配，运行中归还/接管是后续。

## 2. 架构

一次 decode 步的分工（沿用 engine.md §6 与 ADR-011 的组边界，`StepConfig.offload_layers` 指针已存在、只差接线）：

| 计算 | 落点 | 键事实 |
|------|------|--------|
| 稠密段（attn 投影/递推、GDN、GR、router、indexer、head、共享专家） | CUDA 卡（双卡按层连续切分，ADR-004） | 每 token 必算主循环；`--native-gdn`/`--native-qsa` 等效逐相位搬 |
| 命中专家 | CUDA 卡（`ExpertFfnDevice` 常驻 + 批启动，已实现） | 现有资产零重写，只接候选表 |
| 未命中专家 | 核显优先（Vulkan）、CPU 兜底 | G-07 已实测算力瓶颈，分担成立；核显组 `fallback=false` 进候选表首位 |
| KV cache | CUDA 显存（QSA 各层） | Strata `--kv-resident` 反向：显存够就全住卡，N 以下才流主机 |
| PLE（第 1 层门控残差） | CPU（现路径） | 一步一次，非热点 |

步内同步保持 interfaces §8.4 的单汇合点不变：核显批与 CPU 批独立提交、同一步末合并；归约顺序按专家 id 全序，组边界不进入求和顺序（ADR-011 决策 6）。

线程模型：现有槽位 worker + 线程池结构不动；CUDA 组经 `CudaGroupFleet` 的每卡工人提交，核显组经新 `IgpuExpertWorker` 提交，CPU 组保持现工人。

## 3. 启动分工求解器

启动序列：解析参数 → 硬件探测 → 分工求解 → 组装组与候选表 → 装载模型 → 服务。

探测（启动一次，纯主机侧，无 GPU 也能跑）：

- CUDA：卡数、每卡显存、sm 代（nvml 已接；`--gpu auto` 取全部可见卡，`0,1` 显式）。
- 核显：Vulkan 可用性 + 能力表（`measure/igpu_probe` 已落 JSON：无 `VK_KHR_shader_integer_dot_product`、无 fp16/int8 扩展、拷贝 26.9 GB/s；枚举需过滤 llvmpipe 软渲染）。
- CPU：核数 / AVX2（telemetry 静态信息已有）。
- 系统内存：专家池与 KV 溢出的可用预算。

求解器是纯函数（输入 = 启动参数 + 画像 → 输出分配表），与 `memory_plan` 同风格，可单测。求解顺序：

1. dense 层分布：显存容量优先，装得下就单卡（`split_skip_if_fits` 等效）；装不下按 `cuda_owner_of` 层切分。
2. KV 位置：先全卡上；超出显存 → `--kv-resident N` 语义回流主机。
3. 每卡专家槽位数：显存扣 dense + KV 后的最大值（memory_plan 纯函数扩展：KB 地板 → 先削专家、再削 KV、连地板都装不下拒绝）。
4. 核显占比：`--igpu-frac`（默认 R-08 实测后的安全值，首版给保守值如总线 30%）；份额 0 = 不启用 → 纯 CPU 未命中路径。
5. CPU 兜底：恒在候选表末位，接核显装不下的。

不做启动时反复 benchmark 的自动调优：profile 需几十次推理才有统计意义，启动就绪时间不可接受。性能微调留测量工具（R-08 阶段）。

## 4. 迁移顺序与阶段验收

每阶段独立可验收：门禁全绿（QINFER_ENABLE_CUDA=OFF 常规构建 68 测试零行为变化）、设备路径判据满足（相对差小 + 位相同比例，见第 7 节）、真权重端到端出词。

阶段 A：稠密段两大热点上卡（GDN + QSA）

- GDN：三层投影（attn_qkv / gate / ssm_out）+ 前段 + 递推 + 收尾；权重已按层缓存，常驻后读不再是瓶颈（engine.md §6 实测投影 72 ms / 20 线程）。
- QSA：Q/K/V 投影、索引器（含逐 cell 打分）、attend + 门、输出投影（QSA 四投影 + 四 norm）。
- 每个内核照「照主机那份写」口径（现有 q2_0/iq2s 同款）：真权重逐行对照 + 变异测试。

阶段 B：KV cache 上卡

- QSA 各层 KV 页从 `KvHostStore` 搬进显存；attention 不再跨总线读。
- 常驻阈值反用：显存够就全住卡，`--kv-resident N` 以下才流主机。
- 验收：长上下文（8K token）attn 段耗时对比 + 同口径 logits。

阶段 C：未命中专家并行（核显优先）

- 核显 worker（`IgpuExpertWorker`，提交面与 CUDA 工人同形）+ 码本 GEMV 的 Vulkan 内核（无 DP4a → 手写整数乘加，语义照 `src/kernels/q2_0.cpp` 逐句对应）。
- 候选表 = {核显, CPU}；BusShareLimiter 每步字节上限。
- 验收：核显/CPU 分摊比例实测（R-08 落地）、logits 相对差。

阶段 D：头与残差收尾 + 图捕获

- head、GR down/up、共享专家上卡；dense 全链图捕获消启动开销（engine.md §6「解码步的稠密部分用图捕获」）。
- 双卡 `--trim-stage-weights` 等效：每卡只载自己层的 dense，显存腾给专家槽位。

## 5. 核显内核（Vulkan / SPIR-V）

运行时接入：`dlopen("libvulkan.so.1")`（Mesa ANV，发行版自带，零新增依赖；无 Level Zero / OpenCL 可用的实测）。预编译 .spv 连同生成来源入库（ADR-008 vendored 惯例）。

内核清单：码本 GEMV（Q2_0 档起步，覆盖部署 down 全 48 层）；后补 IQ2_S / IQ2_XXS / IQ1_M（gate/up，Q8_K 配对）。能力表 `formats` 随内核落地逐档填——`igpu_group` 从 formats=0 占位变成真组。

正确性：分布级（同设备路径），每层 h 量化在舍入边界可与主机差一个量化步——判相对差小 + 位相同比例，不假装逐位一致。

## 6. 参数与路由

新增启动参数：

| 参数 | 语义 | 默认 |
|------|------|------|
| `--gpu LIST` | 启用 CUDA 设备侧；`auto` = 全部可见卡，`0,1` 显式；构建无 CUDA 或卡不可用 → 明确报错不静默 | 空（纯 CPU，现行为） |
| `--igpu-frac F` | 核显每步 DDR 字节占比（0..1）；0 = 不启用 | 0 |
| `--kv-resident N` | KV 显存常驻阈值（反向语义：超出的流主机） | 0（全显存） |

持久面：

- `/props` 增 `gpu_devices`（使用的卡列表）、`offload_plan`（分配表：dense 分布、kv 位置、igpu_frac、每卡 expert 槽位）、`offload_enabled` 布尔。
- `/metrics` 增专家分批计数：`expert_batches_cuda` / `expert_batches_igpu` / `expert_batches_cpu`（Strata 的 `offloaded/routed` 同款语义）+ `kv_vram_bytes`。
- 前端 About 呈现分配表与三类批数（Strata Monitor 的 VRAM hit rate / PCIe share 同款观测面）。

## 7. 正确性与验收矩阵

判据（ADR-009 / interfaces §8.6 口径）：设备与核显路径是分布级。逐元素判「位相同比例 + L1 相对差小」，不要求逐位一致；变异测试钉错读法（尺度漏乘、位序取反、偏移漏 −1 等，各内核 4-5 条）。

单测：

- 求解器：画像 + 参数 → 分配表的纯函数用例（单卡装得下 / 装不下层切分 / KV 溢出回流 / 核显份额 0）。
- 设备内核：与主机同字节输入逐行对照（复用 `test_q2_0_cuda` / `test_iq2s_cuda` 夹具与判据）。
- 核显内核：同字节输入逐行位相同；变异测试。
- 参数解析：`--gpu` / `--igpu-frac` / `--kv-resident` 的形状与非法值。

集成（run.sh 增用例）：

- `--gpu` 在无 CUDA 构建上 → 服务拒绝启动并报明确错误体。
- `--igpu-frac 0` → 服务正常起、核显组不进候选。
- 既有 1-27 全部保持（纯 CPU 路径零行为变化）。

真机验收（环境2）：

- `--gpu auto` 起 8080：curl 出词、nvidia-smi 看到 qinfer 进程占显存、decode tok/s 相比 CPU 基线的倍数、prefill 耗时改善。
- `--igpu-frac 0.3`：核显分摊未命中专家、logits 相对差达判据。

门禁：集成全绿、ctest 全绿（常规构建 68 测试零变化；CUDA 构建 test_expert_ffn_cuda 整步对拍）、docs_checks 非豁免区全绿。

## 8. 依赖与许可

- CUDA：nvcc + `QINFER_ENABLE_CUDA`（ADR-008 既有）。
- 核显：Vulkan 运行时零新增（dlopen 系统 Mesa）；编译期需 `libvulkan-dev` 头 + SPIR-V 编译器；.spv 预编译入库、登记来源与许可（ADR-008 vendored 惯例）。
- CI 无 GPU / 无核显：对应组不可用、不进候选；常规构建零行为变化。

## 9. 文档

- ADR-014：GPU / 核显 serve 接入（决策记录：分工求解器、KV 上卡、核显优先）。
- engine.md §6 / §13：补服务路径的接线说明与实测读数。
- interfaces.md §8：核显组从占位改真组的能力表落法。