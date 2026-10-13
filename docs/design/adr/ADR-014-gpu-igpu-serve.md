# ADR-014：GPU/核显服务接入——分工求解器与阶段化迁移

**状态**：Accepted（2026-10-13）

---

## 背景

服务侧的推理把一次 decode 步的稠密段（注意力/GDN/QSA、门控残差、路由器、共享专家、输出头）、命中专家、未命中专家与 KV 读写全放在主机侧：dense 段与命中专家抢 CPU 核，未命中专家在 CPU 兜底组里排队，KV 恒在主机内存，双卡只被 `--gpu` 选中的专家路径部分使用。对照 Strata 的 `--native` 全家桶与 llama.cpp 的 `--n-gpu-layers` / `--split-mode layer` / KV 上卡，硬件未被充分利用：dense 段（约 36% 的步长，engine §6 相位图）与 prefill（CPU 下 53 token 约 2.5 分钟）是最大的两处浪费。

本决策回答：异构执行以什么形态进入服务层——谁决策分工、迁移分几步、每步的验收边界在哪。

## 决策

1. 启动分工求解器（纯函数）。启动序列固定为：解析参数 → 硬件探测 → 分工求解 → 组装组与候选表 → 装载模型 → 服务。求解器输入 = 启动参数（`--gpu` / `--igpu-frac` / `--kv-resident`）+ 机器画像，输出 = 分配表（dense 层分布、KV 位置、每卡专家槽位、核显占比、CPU 兜底）。纯函数形态与 `memory_plan` 同风格、可单测。求解顺序：dense 装得下就单卡（`split_skip_if_fits` 等效）、装不下按 `cuda_owner_of` 层切分；KV 先全卡上、超显存回流主机；每卡专家槽位 = 显存扣 dense + KV 后的最大值（先削专家、再削 KV、连地板都装不下拒绝）；核显占比由 `--igpu-frac` 给、默认保守启用；CPU 组恒在候选表末位兜底。不做启动时反复 benchmark 的自动调优（profile 需要几十次推理才有统计意义，就绪时间不可接受）。
2. 阶段化迁移，每阶段独立可验收（门禁全绿：常规构建零行为变化、设备路径判据满足、真权重端到端出词）：
   - 阶段 A：GDN + QSA 稠密段上卡（本决策签署时已完成）。
   - 阶段 B：KV cache 上卡（`KvHostStore` → 显存；`--kv-resident` 反向语义）。
   - 阶段 C：核显未命中专家（Vulkan GEMV + `--igpu-frac` + BusShareLimiter）。
   - 阶段 D：head/GR/共享专家上卡 + 图捕获 + `--trim-stage-weights` 双卡权重裁剪。
   - 阶段之间的依赖：B 需要 A 的 QSA 整层设备出口；C 需要 A 的未命中专家候选表接线；D 在 B/C 之后定型稠密全链。
3. dense 走同步整层入口，不进作业队列（阶段 A 的落法）。dense 段每 token 必算、步内只有一个汇合点（interfaces §8.4），把它做成组抽象的异步 job 反而多一次归并。落法是函数指针结构 `DenseExec`（`run_gdn` / `run_qsa` 两个整层入口）：`StepConfig.dense_exec` 由调用方（facade）在 `--gpu` 时填入 CUDA 侧 `dense_exec_cuda()`（只有 `QINFER_ENABLE_CUDA=ON` 的构建有这个符号），`run_block_layer` 在给定时按层类型分派到 `*_cuda` 整层入口，否则全走主机 `run_gdn_layer` / `run_qsa_layer`——主机构建恒空、零行为变化。`--gpu` 语义 = 指定哪些 CUDA 设备（`auto` = 全部可见卡，`0,1` = nvidia-smi 索引列表）；无 CUDA 的构建给 `--gpu` 明确拒绝启动，不静默回退。
4. dense 档覆盖以「合成模型也能跑」为下限。阶段 A 补齐了 Q8_0 权重档的设备内核（合成/测试模型的稠密权重正是该档；真模型的 dense 投影是 IQ4_XS / IQ3_S / Q6_K）——`--gpu` 在任何档位上都不得 500，覆盖不到的档明确报错而不是猜。

## 备选方案

1. dense 走作业队列（把 `ExpertJob` 的形态推广到 dense）。否决：每步每层多一次提交/归并开销，且步内单汇合点的约束会把「排队等回」变成同步等待，白加一层。
2. 一步到位全量迁移（本 spec 全部范围一个计划）。否决：四个阶段各自有独立的验收面（KV 显存占用、核显内核、图捕获），合并成一份计划会丧失每阶段的可证伪性；阶段依赖也已理清（A→B→C→D）。
3. dense 设备实现以 CUDA 符号直连进 `block_layer`。否决：`block_layer.cpp` 编在主机构建里（无 `QINFER_CUDA_BUILD` 时 device 分支不编译），直连会破坏「常规构建零 CUDA 引用」的 ADR-008 口径。函数指针让主机构建不引用 CUDA 符号、CUDA 构建由调用方填充。
4. 启动时自动 benchmark 选分工。否决：需要几十次推理才有统计意义，启动就绪时间不可接受；性能微调留给测量工具。

## 后果

正面：

- 迁移可证伪：每阶段的验收是独立的（设备判据 + 集成门禁 + 真机读数），阶段 A 已交付——decode 2.25 tok/s（CPU 基线 0.5，约 4.5 倍）、prefill 53 token 约 34 秒（CPU 基线约 2.5 分钟，约 4.4 倍）、qinfer 进程占用显存、`/props` 报告 `offload_enabled` 与 `gpu_devices`。
- 主机构建与 CUDA 构建的职责边界重新划清：`--gpu` 是激活开关，不给就是纯 CPU 路径（68 测零行为变化）。
- dense 档覆盖对所有可量化档封闭：任一台合成模型都能 `--gpu` 全链路出词（集成用例 29 钉住：双子层 GDN+QSA 全走 `*_cuda`）。

负面：

- 阶段 A 的 dense 设备路径是「每段独立 H2D/D2H + 同步」形态：权重按层缓存在主机侧、每 token 反复上传（真机 PCIe RX 读数 1.7–3.5 GB），显存占用小（进程侧实测 228 MiB 稳态、峰值约 1.2 GiB）。真正的提速（权重常驻 + 图捕获 + 每步单汇合）留阶段 D，届时当前读数的提升空间自会兑现。
- `--gpu 0,1` 的名单在此阶段只作为激活开关生效，按卡分工的真实装配（`CudaGroupFleet` 候选表接线）在阶段 C 才落地。
- 求解器与阶段 B/C/D 的实现不在本 ADR 的签署范围内，验收口径（ADR-009 分布级一致、每阶段门禁）先行约定。

## 相关

[ADR-004-consumer-dual-gpu-no-tensor-parallel.md](ADR-004-consumer-dual-gpu-no-tensor-parallel.md)（按层切分）、[ADR-005-igpu-npu-as-coprocessor.md](ADR-005-igpu-npu-as-coprocessor.md)（核显能力判据）、[ADR-008-implementation-stack-and-kernel-reuse.md](ADR-008-implementation-stack-and-kernel-reuse.md)（零依赖 / CUDA 之后）、[ADR-009-correctness-acceptance-criteria.md](ADR-009-correctness-acceptance-criteria.md)（分布级判据）、[ADR-011-compute-group-abstraction.md](ADR-011-compute-group-abstraction.md)（组边界与单汇合点）、[`../engine.md`](../engine.md) §6（异构执行与阶段 A 实测）、[`../interfaces.md`](../interfaces.md) §8（提交面与 CUDA 组落法）