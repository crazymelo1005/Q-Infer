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
- 核显编码档尚未启用（默认 CPU，见 spec 的实测后启用条款）。

## 相关

- [ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)：零运行时依赖口径
- [ADR-016](ADR-016-vendored-strata-backend.md)：引擎厂商化与进程边界
- 后续 ADR-020（多模态接入形态）、ADR-021（视频输入与抽帧契约）在 M5b/M5c 落档
