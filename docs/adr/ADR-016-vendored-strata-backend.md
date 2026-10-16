# ADR-016：计算后端——厂商化 Strata 引擎作后端

**状态**：Accepted（2026-10-13）

---

## 背景

自研引擎在真权重（IQ2_XS 分片，环境2）上输出确定性垃圾，而同一份权重上参考引擎 Strata 输出连贯（约 80 至 100 tok/s）。据此判定更可能是自研管线的缺陷（IQ2_XS 质量地板已被同一份权重的连贯输出排除），决定把 Strata 引擎作为计算后端厂商化引入，而不是继续在自研稠密与专家链上排障。

本决策回答：后端以什么形态引入、进程边界怎么划、自研引擎的去向。

## 决策

1. 引擎以源码入库（`third_party/strata`，tag `v0.1.43.1`，MIT © 2026 Niko1221 and the Strata contributors），由独立构建脚本 `tools/build_engine.sh` 编成 `build-engine/strata`。入库树逐字节冻结在 `v0.1.43.1`，不追上游（追版留作后续）；构建只在构建目录的副本上做，不改入库源码。
2. 后端以子进程形态接入：自研的服务二进制不链接引擎，只通过 stdin/stdout 行协议驱动它。进程边界即依赖边界——引擎自带的重型 CUDA 工具链不进本仓库的链接面，[ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md) 的零运行时依赖口径不破。
3. 协议本身即接口，落在 `src/engine`：`engine_protocol`（命令构造与事件行解析）、`engine_process`（起进程与行 IO）、`engine_driver`（握手、单请求生成、批槽位与重启）。词元以 id 过边界，反词元化留在服务层，故驱动层不依赖词表。协议字面量以真实引擎转录为唯一口径（`tests/data/strata_transcript_gen.txt`）。
4. 自研引擎保留在 `--backend inhouse`（P2 接线），不对其未来作承诺；本阶段不碰 `src/serve`，服务层与后端的接线留 P2。

## 备选方案

1. 硬分叉（把引擎内核抄进本仓库并长期维护）。否决：等于接管一条与本仓库设计不同的完整实现，其 CUDA 栈与维护面无法切分，违背零运行时依赖口径。
2. 只抄引擎内核加 oracle 向量（借鉴算法与逐位判据，自己重写）。否决：正是自研实现已经失败的那条路；内核之外的调度、投机与分层才是难点所在。
3. 厂商化引擎作后端（采纳）。进程边界把依赖与维护面隔开，协议作为唯一契约，失败面收敛到驱动层。

## 后果

正面：

- 立即可用的连贯推理：服务层通过驱动即可获得与参考端一致的输出，不必先修好自研管线。
- 依赖与许可清晰：MIT 全量入库，`NOTICE` 记来源与取舍，`third_party/strata/LICENSE` 随包分发（根 `README.md` 许可一节同步标注）。
- 接口收敛：驱动层是纯函数加子进程，主机侧构建零 CUDA 符号（`qinfer_engine` 只依赖标准库与 pthreads，且不并入 `qinfer_serve`），常规构建与 CUDA 构建的门禁不变。

负面：

- 引擎的真实批处理与驱动层的简化模型存在差异：真机 `BGEN` 会先跑一个单请求首词元（`T` 后接 `DONE 1`）再 `BADM`，随后才逐 `BT`，故 `BDONE` 的计数大于回调收到的词元数。驱动层当前按计划约定的形状实现，真机对拍留 P2。
- 内存占用：引擎默认的常驻专家 arena 需约 33 GiB 主机内存，与本机桌面环境叠加会触发内核内存回收；运行须在独立限内存 scope 内并取 `--mmap-experts`。
- 入库树的构建需要一次构建期规避：上游 `CMakeLists.txt` 在 CUDA 段无守卫地引用 `tests/core/dma_batch_parity.cu`，而该文件不在入库集合内；做法记录在 `NOTICE` 与构建脚本，不改入库源码。

## 相关

[ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)（零运行时依赖与 CUDA 之后的实现口径）、[ADR-014](ADR-014-gpu-igpu-serve.md)（硬件接入的处置）、`../engine.md` §18（后端落法）、`../interfaces.md` §8.8（驱动接缝）。来源与许可见 [`../../../NOTICE`](../../NOTICE)。
