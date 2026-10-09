# ADR-002：未命中专家在 CPU 侧「就地计算」，而非搬回 GPU

**状态**：Accepted
**日期**：2026-10-09

---

## 背景

专家缓存未命中时，被路由选中的专家权重不在显存。两条路可选：

- 搬回来算：经 PCIe 把权重拷到 GPU 再算（llama.cpp 式）。
- 就地算：让 CPU 直接在内存里算完（Strata 式）。

## 决策

未命中专家由 CPU 就地计算，与 GPU 上命中的专家并行，最后 scatter-add 归并。同时保留一条可调的分流通道（对应 Strata 的 `--pcie-frac`）：把未命中专家的一部分拷到 GPU 算。

## 备选方案

1. 纯搬回 GPU —— 否决。PCIe 5.0 x8 ≈31.5GB/s < 内存带宽 ≈70GB/s，且每层都要等搬运，把 PCIe 放进每层关键路径。
2. 纯 CPU 就地算（完全不碰 PCIe） —— 部分否决。仅适用于纯 CPU 分支；prefill 阶段官方明确「experts streamed to the GPU over PCIe」，且 `--pcie-frac 0` 会掉 decode 速度。
3. 动态分流（就地算为主 + 可调比例搬回） —— 采纳。

## 后果

正面
- 把 PCIe 从 decode 关键路径摘掉（Strata 实测在「专家装不下」场景领先 llama.cpp 约一个数量级）。
- 两个算力单元真正并行。

负面
- CPU 算力成为新上限：环境2 的 Arrow Lake 无 AVX-512/AMX，只能走 AVX2 → 未命中比例不能太高。
- 需要良好的线程分池（P-core / E-core）与量化 GEMM 内核。

## 与本项目的关联

- 分流比例由自动标定（[engine.md](../engine.md) §10）+ 在线自适应决定，而不是固定值。
- 「就地算还是搬回」取决于 G-07 实测：若 CPU 路径是算力瓶颈，则核显可加入（ADR-005）。

## 相关

ADR-001、ADR-005、[research/01-strata-engine.md](../../research/01-strata-engine.md) §5
