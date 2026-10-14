# ADR-020：Strata iGPU 专家池——未命中专家加速

**状态**：Accepted（2026-10-14）

---

## 背景

Strata 后端的 CPU 专家池（`ExpertPool`）承担约 95.4% 的专家计算（未命中 VRAM 的专家）。CPU 层是当前瓶颈：每 token 663.6 MB、带宽 33.7 GB/s，折合 19.71 ms/token。

Intel iGPU 在系统中可用但未被充分利用（设备优先级为 GPU > iGPU > CPU）。[ADR-019](ADR-019-igpu-not-for-experts.md) 判定核显在 MoE 专家的小矩阵形状（640 行）下慢于 CPU 池，因此不接入厂商化引擎的 `PoolFn` 缝。但该结论不覆盖另一条路径：在 Strata 引擎**之外**、以独立 iGPU 专家池的形式承接 CPU 侧的未命中专家计算，并使用足以填满设备的矩阵形状。

## 决策

1. 新增 CPU+iGPU 混合专家池：当一批未命中专家的数量超过阈值（>4 个专家）时，把该批路由到 iGPU，以 Vulkan 计算着色器执行；小批仍走 CPU `ExpertPool`，保留低延迟回退路径。
2. iGPU 路径通过环境变量 `STRATA_IGPU_POOL=1` 显式开启，缺省关闭。
3. 新增两枚着色器处理 Strata 的 planar Q2_0 格式（codes 与 scales 分离）以及 ActQ 激活格式：`strata_q2_0_gemv.comp`（矩阵-向量乘）与 `strata_swiglu_quant.comp`（融合 SwiGLU + 量化）。
4. `IgpuExpertPool` 类管理 Vulkan 管线与缓冲区生命周期；`hybrid_expert_dispatch()` 在 `expert_pool_dispatch` 之上封装路由逻辑。
5. 不引入新的链接期依赖：Vulkan 运行时通过 `dlopen` 加载（复用既有 `VulkanRuntime`），与 [ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md) 的零运行期依赖规则一致。

## 架构

- 着色器位于 `third_party/strata/src/kernels/igpu/`，离线以 `glslangValidator` 编译，以字节数组形式嵌入 `spv_data.hpp`。
- `IgpuExpertPool`（`third_party/strata/src/kernels/igpu/igpu_expert_pool.hpp/.cpp`）持有 Vulkan 管线、描述符集与设备侧缓冲区；在首次调用时惰性初始化，在池销毁时统一释放。
- `hybrid_expert_dispatch()` 检查 `STRATA_IGPU_POOL` 环境变量与批大小，决定走 CPU 还是 iGPU；iGPU 不可用或批过小时自动回退到 CPU。
- 整条路径不修改厂商化引擎源码（`third_party/strata/` 保持未改动），只在引擎外增加一层调度。

## 备选方案

1. 把 iGPU 接入厂商化引擎的 `PoolFn` 缝（即 ADR-019 否决的 P3b 路线）。否决：ADR-019 已给出该形状下的实测读数，缝内合批到上限仍只有 8 线程 CPU 池的 0.23 至 0.52 倍；且每次升引擎都要重验证构建期补丁。
2. 始终走 iGPU，不设 CPU 回退。否决：小批（≤4 专家）下 iGPU 的分派固定开销（0.09 至 0.14 ms/次）占比过高，CPU 池延迟更低；且 iGPU 不可用的环境（无 Vulkan 设备、`dlopen` 失败）必须有回退。
3. 把 iGPU 池做成默认开启。否决：该路径依赖 Vulkan _capable_ 硬件与驱动，且数值精度需在 1e-6 相对容差内匹配 CPU 参考——在更广的设备族上验证之前，opt-in 更安全。

## 后果

正面：

- 未命中专家目标加速 2–3×（19.71 ms → 约 7–10 ms/token），前提是批大小足以填满 iGPU（与 ADR-019 的 P1 spike 读数一致：10240 行级别大矩阵可得 3.9× 加速）。
- 优雅降级：iGPU 不可用或批太小时自动回退到 CPU，不影响正确性。
- 复用既有 Vulkan 基础设施（任务 #135–#139 交付的运行时与着色器资产），不引入新的链接期依赖。
- 不修改厂商化引擎源码，升引擎时不需要重验证补丁。

负面：

- 两条代码路径需要维护（CPU 与 iGPU），着色器与 CPU 内核的档位布局若变动需同步更新；两者由逐位对拍测试约束。
- iGPU 路径要求 Vulkan-capable 硬件；在没有 Intel iGPU 或驱动不支持 `VK_KHR_shader_integer_dot_product` 的设备上，该路径不可用或性能特征不同（ADR-019 已指出该扩展的可用性不确定）。
- 数值精度必须在 1e-6 相对容差内匹配 CPU 参考；这是一条需要持续验证的约束，不是一次性证明。

## 实现

- 计划文件：`plan-strata-igpu-pool-implementation.md`，任务 #1–#6。
- 着色器：`strata_q2_0_gemv.comp`、`strata_swiglu_quant.comp`，位于 `third_party/strata/src/kernels/igpu/`。
- 池管理：`igpu_expert_pool.hpp/.cpp`，同目录。
- 集成：`generate.cpp` 检查 `STRATA_IGPU_POOL` 环境变量，决定启用混合调度。

## 相关

[ADR-019](ADR-019-igpu-not-for-experts.md)（核显不参与厂商化引擎 `PoolFn` 缝内的专家计算——本 ADR 走的是缝外的独立池，不冲突）、[ADR-005](ADR-005-igpu-npu-as-coprocessor.md)（核显条件性方案，第一节接受）、[ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)（零运行期依赖规则）、[ADR-016](ADR-016-vendored-strata-backend.md)（厂商化引擎进程边界）、任务 #135–#139（Vulkan 运行时与着色器资产）。
