# ADR-012：HTTP 传输与单进程服务栈

**状态**：Accepted（2026-10-12）

---

## 背景

引擎的主机侧生成环已能单请求跑通；要把引擎对出去，需要一个对外推理服务。参考范本 Strata 是单进程 Python 服务，本仓库的实现栈（`ADR-008-implementation-stack-and-kernel-reuse.md`）要求 C++20、CMake、运行期依赖面为零、主机侧不依赖 CUDA。本决策回答四件事：HTTP 传输用什么、服务进程怎么摆、JSON 谁来解析、监控输出什么格式。

已有约束：监控形态已确认为 Web 页面实时查看（不先做 Prometheus 完整对接）；并发正确性先到「互不饿死」，吞吐目标留 M2 实测。

## 决策

1. HTTP 传输用 cpp-httplib 入库（MIT 许可证，单头，llama.cpp 同款）。文件整份入库、保留版权与许可注释、锁定上游发布版本号（v0.18.6）。「裁剪」落实为只启用服务端所需特性：不定义 `CPPHTTPLIB_OPENSSL_SUPPORT` 与 `CPPHTTPLIB_ZLIB_SUPPORT`；线程池大小由入口按槽位数配置（2 × 槽位 + 2）。协议解析、SSE 帧、轮询全部自写，不引入 HTTP 协议栈之外的任何依赖。
2. 单进程自包含，不做独立网关进程：零 IPC、零部署面，与 ADR-008 的依赖面口径一致。并发在进程内解决：有界队列 + 槽位租约，请求经队列交给槽位 worker，worker 驱动引擎步循环，增量经每 Job 的事件缓冲回写 HTTP 线程（满则背压、断开即取消）。
3. JSON 手写子集入库：对象、数组、字符串（含转义与 `\uXXXX`）、数字、布尔、空。OpenAI 与 Anthropic 两个兼容域的请求与响应都在这个子集内；越界输入一律 400。
4. 监控只出页面用的 JSON（`/metrics` 顶层键对齐 Strata v0.1.43 的契约：engine、live、requests、conversation_cache 占位、requests_kept、totals、hardware、hardware_static、history、time）。Prometheus 文本格式延迟到 M2；`conversation_cache` 的实值同样等 M2 接索引器统计。
5. 硬件采集零依赖：NVIDIA 走 `dlopen("libnvidia-ml.so.1")` / Windows `LoadLibrary("nvml.dll")`，读不到的字段一律 `null`，遥测任何异常不允许拖垮服务；AMD 走 Linux sysfs（kfd 拓扑 + hwmon）；CPU/RAM 走 `/proc` 差值法。逻辑照抄 Strata 的 `serve/telemetry.py`。

## 备选方案

1. 自写 socket 层。否决：连接管理、keep-alive、chunked 编码全是重复劳动，与「零依赖」不冲突（httplib 是整份入库的 vendored 单头，许可与上游一致）。
2. 独立网关进程 + IPC。否决：引入第二份部署面与两套进程生命周期，与单机、低并发场景不符。
3. 引第三方 JSON 库（nlohmann 等）。否决：需要的是协议边界的一个极小子集，手写解析 + 单测 200 行内可控，且保持依赖面为零。
4. Prometheus 文本先做。否决：用户确认监控形态是 Web 页面，M2 有完整对接需求时再补 exporter。

## 后果

正面：

- 依赖面维持为零（ADR-008 口径），单二进制即可对外服务，部署面与引擎一致。
- 两个兼容域共用同一套小 JSON 与同一调度器，行为可对拍；SSE 逐帧与 curl 对照。
- 遥测降级友好：无 GPU 的 CI 与笔记本上服务照常，页面显示占位。

负面：

- vendored httplib 要自持上游演进（升级 = 重新入库加复核 diff）；锁定版本号后成本可控。
- 手写 JSON 子集不支持流式 token 解析等高级用法；协议域越过该子集时返回 400，明确失败而非静默错译。
- M1 只承诺「互不饿死」，吞吐与延迟目标留 M2 实测，此前不引性能结论。

## 相关

`ADR-008-implementation-stack-and-kernel-reuse.md`（实现栈）、`ADR-009-correctness-acceptance-criteria.md`（正确性口径）、`../engine.md` §12（服务与调度）、`../../research/01-strata-engine.md`（Strata 契约来源）、`../../superpowers/specs/2026-10-12-serve-layer-design.md`（服务层设计规格）