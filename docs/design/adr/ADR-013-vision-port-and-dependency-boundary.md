# ADR-013：视觉编码器取材与依赖边界

**状态**：Accepted（2026-10-14）

---

## 背景

入库的 Strata 引擎树（`third_party/strata`）只带引擎的视觉半边（`--vision`、`GENI`、`BGENI`），编码器半边（`tools/vision` 与其 llama.cpp mtmd 依赖）不在仓库里，故 #105 无法出图。要定：编码器怎么取材、依赖边界怎么划、视频解码走哪条路。

## 决策

1. 编码器移植自 Strata `tools/vision`（MIT，tag `v0.1.43.1`），产物 `build-vision/bin/qinfer-vision`，由 `tools/build_vision.sh` 构建；stdout 行协议与 SVE1 字节格式保持上游原样，允许的偏离由 `measure/vision_port_check.py` 逐条钉住，正确性口径是与上游同树同 flags 的构建**逐字节全等**（`measure/vision_oracle.py`）。
2. llama.cpp 钉 `3cf03257f219afbe7334045ff7c6a06ac68c627d`（与 `third_party/strata/third_party/ggml/VERSION.txt` 同一条），**不入 git、不下载**，构建期 `git archive` 从本地 checkout 物化到 git-ignored 目录——沿用 `tools/build_engine.sh` 的既有纪律。
3. 只被 spawn 不被链接：`qinfer-server` 零新增运行时依赖（[ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)、[ADR-016](ADR-016-vendored-strata-backend.md)）。
4. 视频解码复用 mtmd 自带视频档（`MTMD_VIDEO`，其实现 spawn 用户自备的 `ffmpeg`/`ffprobe` CLI）：不新建解码二进制、不链接 libav\*、不引入许可义务。

## 备选方案

1. 硬分叉（把 llama.cpp/mtmd 抄进仓库长期维护）——否决：26 MB 起的一份 ggml 第二副本，且偏离即失控。
2. 整份 vendored 进 git（含 llama.cpp）——否决：违背既有取材纪律，门禁与体积都变差。
3. 自建 FFmpeg 解码件——否决：mtmd 已提供同等能力且形态就是 spawn CLI，自建只多一个进程与一份 LGPL 取值。

## 后果

正面：

- 移植可验证（字节锚 + guard 双向）、依赖边界不破、构建可复现（钉版标记）。
- 与引擎共享同一份钉版 llama.cpp，oracle 比较有意义。

负面：

- 构建需要本地 llama.cpp checkout 持有该 commit（无网机器要预先备好）。
- 上游 tag 一旦移动需显式改 `PIN` 常量与 guard 规则。
- 一次 oracle 要编两份 llama.cpp（约 10-20 分钟）。
- 核显编码档自 2026-10-15 起默认启用（`--vision-gpu` 默认 on，`off` 显式回 CPU；见下方附录）。

## 相关

- [ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)：零运行时依赖口径
- [ADR-016](ADR-016-vendored-strata-backend.md)：引擎厂商化与进程边界
- 后续 ADR-020（多模态接入形态）、ADR-021（视频输入与抽帧契约）在 M5b/M5c 落档

## 附录：核显编码档默认启用（2026-10-15）

编码器可在 ggml-vulkan 后端上构建（`tools/build_vision.sh STRATA_VISION_VULKAN=ON`，产物
`build-vision-vk/bin/qinfer-vision`），spawn 时带 `--gpu` 即在 Vulkan device 0（本机为 Intel
iGPU）上编码。实测（2×RTX 5060 Ti + ARL iGPU，sc117 mmproj 907 MB）：

- 编码 192 ms vs CPU（8 线程）1375 ms，**7.0×**；每图省 ~10 CPU 秒（11.08 → 1.35 CPU-s）。
- 系统内存驻留，**零 RTX VRAM**（iGPU 为 UMA；需 `GGML_VK_ALLOW_SYSMEM_FALLBACK=1`，服务层
  spawn 前已代设）。
- 代价：启动预热 ~49 s（4096 image token，一次性）；Intel-only ICD 由服务层按存在性代设，
  避免 ~31 s 的 RTX 探测。

正确性口径：编码输出**不与 CPU 档逐字节全等**（fp16 累加序不同，相对 L2 2.0%），oracle 门
不适用于该档；按 ADR-009 默认配置口径验收——嵌入级 100% 自重复（sha256 可复现），端到端
5 图 × 220 token 贪心 A/B（`measure/vision_quality.py`）中跨臂一致率与臂内噪声同量级（两臂
自身均不可自重复，agreement 0.05-0.39），即编码器差异低于文本管线自身噪声底。据此
`--vision-gpu` 默认翻 on；iGPU 二进制缺失/起不来时自动回退 CPU 档（显式 `--vision-binary`
不回退）。
