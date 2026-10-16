# ADR-017：服务层引擎缝与厂商化后端接入

**状态**：Accepted（2026-10-14）

---

## 背景

服务层此前直连具体类：`Serve` 持 `std::shared_ptr<Engine>` 并调用它的 `generate`、`tokenizer`、`chat_template`、`model_name`、`config` 与三个 KV 统计方法。这层耦合使「换个计算后端」无法在不改服务层的前提下完成。[ADR-016](ADR-016-vendored-strata-backend.md) 已把厂商化引擎作为后端引入并交付了驱动层（`engine_protocol` / `engine_process` / `engine_driver`），本决策回答：服务层以什么接缝接上它。

## 决策

1. 抽出纯虚接口 `Backend`（`src/serve/facade.hpp`），只含服务层真正用到的东西：`generate`（已编码的 id 进、逐 token 的**文本**出、`GenOptions`/`GenResult` 语义不变）、`tokenizer` / `chat_template` / `model_name` / `config`、以及 `kv_misses` / `kv_evictions` / `kv_bytes_moved`。`Engine` 继承它；`Serve` 只持 `Backend`。
2. 新增 `StrataBackend`（`src/serve/backend_strata.hpp`）：持 `EngineDriver` 与 `text::Tokenizer`，把 `GenOptions` 映成驱动请求，把引擎回的词元 id 解码成文本，并把结束原因映射回 `Finish`。词表与 chat 模板由调用方（main）从同一个 GGUF 读出后移入——只读元数据，不碰权重；这也让单测能用一个小词表驱动，不必开真模型。
3. CLI 加 `--backend inhouse|strata`，默认 `inhouse`（首版保守）。`strata` 需要 `--engine-exe` 与可重复的 `--engine-arg`（原样透传给引擎，如 `--pack`、`--native`、`--mmap-experts`）；缺 `--engine-exe`、或非 `strata` 却给了 `--engine-arg`、或 `strata` 与 `--gpu` 同用，一律拒绝启动。
4. `qinfer_serve` 起链接 `qinfer_engine`。这是接缝的本意；`qinfer_engine` 只依赖标准库与 pthreads，故 [ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md) 的零运行时依赖口径不破。
5. 流式语义在适配器里收敛：多字节字符被词元切开时按完整 UTF-8 序列切分（`utf8_complete_prefix`），stop 串在首个命中处截断并报 `Finish::kStopString`，回调返回假按客户端中断处理。

## 备选方案

1. 在 `Engine` 内部按后端分支（把驱动层塞进门面）。否决：两条执行路径的装配、错误面与资源面完全不同，合成一个类只会让两边都难测。
2. 让服务层直接调 `EngineDriver`。否决：服务层需要的是文本、`Finish` 与 stop 语义，不是 id 流；把反词元化、stop 截断与中断映射散落到各处理器里，等于把适配器拆成三份。
3. 维持直连（只加一个全局开关）。否决：那就还是无法在不改服务层的前提下换后端，也没有可单测的缝。

## 后果

正面：

- 默认路径零行为变化：`--backend` 不给即为 `inhouse`，常规构建与既有测试不受影响（接入后 ctest 77/77、集成 53/53、`docs_checks` 通过）。
- 换后端只需实现一个接口；适配器自带单测（对 mock 引擎 + 110 词小词表，不需要 GPU 与权重）。
- 渲染器有了逐字节 oracle：`tests/data/chat_golden.json` 与它的模板从上游 `serve/` 一并入库（MIT，`NOTICE` 记录），10 个用例里 8 个逐字节相同；缺口被显式列成 known_gaps 并被断言为恰好这些，新分歧会直接失败。

负面：

- `strata` 下 `kv_misses` / `kv_evictions` / `kv_bytes_moved` 恒为 0——KV 由引擎自管，`/metrics` 里这几项在该后端下不具意义，读的人要按后端解读。
- 真机批处理与驱动层模型有差（首词元先走单请求再 `BADM`，见 ADR-016 后果），本阶段未解；服务层目前只用单请求路径。
- 渲染器与参考端逐字节对齐：`tests/data/chat_golden.json` 的 10 个用例全部相同（早先缺的渲染级 `enable_thinking` 与助手 `reasoning_content` 已补进 `Message`/`render_chat` 的变量注入，缺口表因此清空——那条断言现在钉的就是「10/10」）。
- 真机 A/B 已跑：同一提示（`Hello`，渲染后 53 个 token id）与同一组采样键（`temperature=0 top_p=0.95`、`max_new=8`），在独立限内存 scope 内起 `--backend strata` 服务，输出与直接用同一串 id 驱动引擎的参考结果逐字节相同（`We need to respond to user "Hello`），usage 的 prompt 53 / completion 8 也一致；重复请求输出相同，流式分片拼接后与整段一致。判据满足于确定性档；不同采样档位下的分布级对拍未做。
- 一处报告口径不一致：`/props` 的 `n_ctx` 取自 CLI 的 `--max-context`（默认 262144），不是后端 `config()` 里引擎上报的 65536。生成不受影响，但读 `/props` 的人会误判上下文容量，待修。

## 相关

[ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md)（零运行时依赖与 CUDA 之后的口径）、[ADR-016](ADR-016-vendored-strata-backend.md)（厂商化引擎作后端的决策与真机观测）、[`../interfaces.md`](../interfaces.md) §8.8（驱动与后端接缝）、[`../engine.md`](../engine.md) §18（后端落法）。
