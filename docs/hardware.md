# 测试环境硬件配置

本文档记录两套测试环境的硬件规格。规格来源为 Intel / NVIDIA 官方 ARK 及厂商规格页，核实时间 2026-10。研究结论与引擎设计见 [README.md](README.md)（文档索引）。

## 测试环境 1（单卡）

| 组件 | 规格 |
|------|------|
| CPU | Intel Core Ultra 7 265K（Arrow Lake-S，TSMC N3B，LGA1851） |
| — 核心/线程 | 20 核 20 线程（8 P-core Lion Cove + 12 E-core Skymont，无超线程） |
| — 频率 | P 基频 3.9 / 睿频 5.4（TVB 5.5 GHz）；E 基频 3.3 / 睿频 4.6 GHz |
| — 缓存 | L3 30MB Smart Cache / L2 合计 36MB |
| — 功耗 | 基础 125W / 最大睿频 250W |
| — 核显 | Intel Arc Xe-LPG Graphics，64 EU（4 Xe-core），0.3–2.0 GHz，GPU 峰值约 8 TOPS(Int8，含稀疏)，支持 AV1/HEVC 硬件编解码 |
| — NPU | Intel AI Boost，约 13 TOPS(Int8)；平台总 AI 峰值约 33 TOPS(Int8，含稀疏) |
| 内存 | 48GB（24GB × 2）DDR5-8400（XMP 稳定运行；官方标称最高 DDR5-6400） |
| GPU | NVIDIA RTX 5060 Ti 16GB × 1 |
| 主板 | Z890 Unify-X（单卡走 PCIe 5.0 x16，约 63GB/s 单向） |
| 存储 | 致态 TiPlus7100 2TB（M.2 PCIe 4.0×4，顺序读 7.0GB/s / 写 6.0GB/s，4K 随机读约 900K IOPS） |
| 电源 | 1000W |

内存理论带宽：8400 MT/s × 8B × 2 通道 ≈ 134.4 GB/s。

## 测试环境 2（双卡）

| 组件 | 规格 |
|------|------|
| CPU | Intel Core Ultra 9 285K（Arrow Lake-S，TSMC N3B，LGA1851） |
| — 核心/线程 | 24 核 24 线程（8 P-core + 16 E-core，无超线程） |
| — 频率 | P 基频 3.7 / 睿频 5.6（TVB 5.7 GHz）；E 基频 3.2 / 睿频 4.6 GHz |
| — 缓存 | L3 36MB Smart Cache / L2 合计 40MB |
| — 功耗 | 基础 125W / 最大睿频 250W |
| — 核显 | Intel Arc Xe-LPG Graphics，64 EU（4 Xe-core），0.3–2.0 GHz，GPU 峰值约 8 TOPS(Int8，含稀疏)，支持 AV1/HEVC 硬件编解码 |
| — NPU | Intel AI Boost，约 13 TOPS(Int8)；平台总 AI 峰值约 36 TOPS(Int8，含稀疏) |
| 内存 | 64GB（16GB × 4）DDR5-4400（双通道，每通道 2 条 DIMM） |
| GPU | NVIDIA RTX 5060 Ti 16GB × 2（共 32GB 显存） |
| 主板 | Z890 Pacific（双卡 PCIe 5.0 x8 + x8，每卡约 31.5GB/s 单向 / 64GB/s 双向） |
| 存储 | 致态 TiPlus7100 2TB（M.2 PCIe 4.0×4，顺序读 7.0GB/s / 写 6.0GB/s，4K 随机读约 900K IOPS） |
| 电源 | 1000W |

内存理论带宽：4400 MT/s × 8B × 2 通道 ≈ 70.4 GB/s。

## GPU 官方规格：RTX 5060 Ti 16GB

| 项目 | 数值 |
|------|------|
| 架构 / 核心 | Blackwell，GB206-300，sm_120 |
| CUDA 核心 | 4608（36 组 SM × 128） |
| Tensor Core | 第 5 代，144 个，759 AI TOPS（FP8/稀疏） |
| RT Core | 第 4 代，72 TFLOPS |
| 光栅算力 | FP32 约 23.8 TFLOPS；FP16 约 47.6 TFLOPS |
| 频率 | 基础 2407 MHz / Boost 2572 MHz（非公最高约 2692） |
| 显存 | 16GB GDDR7，128-bit，28 Gbps |
| 显存带宽 | 448 GB/s |
| L2 缓存 | 32MB |
| 数值格式支持 | FP16 / BF16 / FP8 / NVFP4(FP4)，Tensor Core 原生 |
| NVLink | 不支持 |
| 总线 | PCIe 5.0，Resizable BAR |
| TGP | 180W（非公上限约 190W） |

## 环境对比

| 对比项 | 环境1 | 环境2 |
|--------|-------|-------|
| CPU | 265K（20C20T） | 285K（24C24T，更多 E 核、更高频） |
| 内存容量 | 48GB | 64GB |
| 内存频率 | DDR5-8400（XMP） | DDR5-4400 |
| 内存带宽 | 约 134 GB/s | 约 70 GB/s |
| 内存通道 | 双通道（2 条） | 双通道（每通道 2 条，共 4 条） |
| GPU 数量 | 1 | 2 |
| 总显存 | 16GB | 32GB |
