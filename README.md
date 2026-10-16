# Q-Infer

An OpenAI-compatible inference server for very large sparse-MoE models on consumer GPUs. It runs Qwen3.8-Flash-Next (125B main parameters, a 51B external n-gram table, a 4B MTP head, native 262K context) on one 16 GB card, or on two, with the weights that do not fit in VRAM served from host memory and NVMe.

Q-Infer is the product layer: the HTTP API, slot and session management, the token-prefix cache, telemetry, the web console, and the multimodal image encoder. The number crunching happens in an engine behind a process boundary - by default the Strata engine whose sources are carried in this repository under `strata/` and built by `tools/build_engine.sh` ([ADR-016](docs/adr/ADR-016-vendored-strata-backend.md), [ADR-018](docs/adr/ADR-018-default-backend-strata.md)). Two self-written backends remain in the tree and stay selectable.

Naming: "Q" stands for both Qwen (the primary case) and Quantized; the Chinese name 擎 means "to lift", which is the premise - lift 125B with 16 GB.

## Quick start

```bash
cmake -S . -B build && cmake --build build -j            # server: build/qinfer-server
./tools/build_engine.sh                                   # engine: build-engine/strata
./tools/build_vision.sh                                   # optional: image encoder

./build/qinfer-server --backend strata \
  --engine-exe ./build-engine/strata \
  --model <model>-00001-of-00002.gguf \
  --engine-arg --pack --engine-arg <pack-dir> \
  --engine-arg --ple-gguf --engine-arg <model>-00002-of-00002.gguf \
  --engine-arg --expert-profile --engine-arg <expert-profile.bin> \
  --engine-arg --resident-experts \
  --slots 4 --port 8080
```

`curl http://127.0.0.1:8080/health` reports `loaded: true` once the engine is warm. The console is at `/`. The annotated command, the memory budget it assumes, and the knobs that were measured rather than guessed are in [docs/deployment.md](docs/deployment.md).

## Scope

| In scope | Out of scope |
|----------|--------------|
| One machine, one or two consumer GPUs | Server-grade concurrent throughput |
| MoE weights that never fit in VRAM | Models that fit entirely in VRAM |
| Contexts of 262K and above | Multi-node or distributed serving |
| Trading deterministic table reads for capacity | Training, fine-tuning, data pipelines |
| Low concurrency (2-8 live requests) | A general multi-model, multi-backend platform |

The Intel iGPU has one job: the multimodal vision encoder. It does not take part in inference compute ([ADR-019](docs/adr/ADR-019-igpu-not-for-experts.md), [ADR-022](docs/adr/ADR-022-igpu-cannot-take-cpu-decode-share.md)).

## Backends

| `--backend` | What runs | Status |
|-------------|-----------|--------|
| `strata` | The engine sources under `strata/`, driven over their line protocol | Default; the path measured against the acceptance targets |
| `inhouse` | The tiered-offload engine in this repository (`storage`, `experts`, `ple`, `dense`, `kv`) | Host-side paths complete and oracle-verified against independent implementations; device-side half incomplete |
| `native` | The self-written native engine (`src/engine`, `src/serve/native_session.cpp`) | Chunked prefill and session persistence work; output quality not yet accepted |

Switching backends changes no other layer: the boundary is a process and dependency boundary, not a compile-time one ([ADR-017](docs/adr/ADR-017-serve-engine-seam.md)).

## HTTP surface

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions`, `POST /v1/completions` | OpenAI-compatible inference, streaming and tool calls |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic-compatible inference |
| `GET /v1/models`, `GET /v1/status` | Model card, live slots |
| `GET /health` | Readiness plus device and memory telemetry |
| `GET /metrics` | JSON snapshot, or Prometheus text via `Accept` / `?format=prometheus` |
| `GET /api/requests`, `POST /api/feedback` | Request records and human feedback for the console |
| `GET /`, `/web/*` | Web console: chat, monitor, inspector |

## Measured performance

One box: Intel 285K, 61 GB RAM, 2 x RTX 5060 Ti 16 GB, IQ2_XS shards, cold prompts.

| Config | Engine | Prefill | Decode |
|--------|--------|---------|--------|
| 1 GPU, `--resident-experts` | 0.1.44.1 | 1,904 tok/s at 21K; 1,861 at 131K | 94-101 tok/s (300-token prose) |
| 2 GPUs, layer split + MTP + pipelined windows | 0.1.43.1 | ~1,600 tok/s at 29K | 120-143 tok/s |
| Expert streaming through the OS page cache | 0.1.43.1 | ~700 tok/s | - |

Absolute numbers move across restarts by more than the noise floor of any single reading, so they are reported as regimes: each figure carries its prompt length, card count and engine version, and only paired same-prompt readings are used for comparison. Prefill is expert-I/O bound rather than GEMM bound, and the tuning surface has been swept item by item - both conclusions carry their measurements in [ADR-021](docs/adr/ADR-021-prefill-expert-io-bound.md) and [ADR-023](docs/adr/ADR-023-strata-engine-upgrade-01441.md).

## Layout

```
.
├── src/                    the product: serve, engine drivers, in-house kernels and storage tiers
├── strata/                 the Strata engine sources carried in this repo (provenance: NOTICE)
├── deps/httplib/           the single header dependency shipped with the server
├── tools/                  build_engine.sh, build_vision.sh, the vision port
├── tests/                  ctest suites, fixtures, recorded engine transcripts, frontend tests
├── measure/                measurement tools and oracle generators (their output is reproducible)
├── scripts/docs_checks.py  the documentation gate, used by CI and locally
├── docs/                   operator and user documentation, plus docs/adr/ decision records
└── .github/workflows/      docs-check.yml, build.yml
```

## Documentation

| Read | For |
|------|-----|
| [docs/overview.md](docs/overview.md) | How the pieces fit together and where the boundaries sit |
| [docs/deployment.md](docs/deployment.md) | Build, run, memory budget, measured tuning knobs, measurement hygiene |
| [docs/requirements.md](docs/requirements.md) | Targets, acceptance scenarios, non-goals |
| [docs/adr/](docs/adr/README.md) | Decision records: what was chosen, measured, or dropped |
| [docs/hardware.md](docs/hardware.md) | The two test environments |
| [docs/glossary.md](docs/glossary.md) | Terminology, one meaning per term |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Writing, numbering, commit and CI conventions |
| [NOTICE](NOTICE) | Provenance and licences of the sources carried in this repo |

## Build and test

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure      # 101 registered suites
python3 scripts/docs_checks.py
```

CUDA is optional: the CPU-only configuration builds and passes the same suites. `tools/build_engine.sh` rebuilds from a clean copy of `strata/` and stops if that tree differs from the committed one; its ggml dependency is materialised from one pinned commit by the script itself.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). The contributions that move this project most are measurements that overturn an assumption, and negative results recorded where others can find them.

License: MIT, see [LICENSE](LICENSE). The engine sources under `strata/` and the vision port keep their own copyright headers; provenance and the trade-offs made there are in [NOTICE](NOTICE).

## Citation

```
Q-Infer: serving very large sparse-MoE models on consumer GPUs.
https://github.com/crazymelo1005/Q-Infer
```

---

## 中文摘要

Q-Infer 在消费级 GPU 上跑超大稀疏 MoE 模型，主案例是 Qwen3.8-Flash-Next（125B 主参数 + 51B 外挂 n-gram 表 + 4B MTP，原生 262K 上下文）。产品层是本仓库自研的服务与前端：OpenAI / Anthropic 兼容接口、槽位与会话管理、token 前缀缓存、遥测、请求检查页、图像多模态编码；计算由进程边界的引擎承担，默认用纳入本仓库的 Strata 引擎源码（`strata/`，由 `tools/build_engine.sh` 构建），另可选自研的 `inhouse` 与 `native` 后端。

范围限定为单机单卡或双卡、2 至 8 路并发、权重装不进显存的场景；不做多机分布式，也不做服务端高并发。核显只承担多模态编码器，不参与推理计算（[ADR-019](docs/adr/ADR-019-igpu-not-for-experts.md)）。实测口径（引擎 0.1.44.1，`--resident-experts`，冷提示）：单卡 prefill 约 1,900 tok/s（21K 与 131K 提示）、decode 94 至 101 tok/s；双卡分层加 MTP 与流水线窗口 decode 120 至 143 tok/s。跨重启的绝对值不可比，只有同题成对读数可用于比较，因此每个数字都写明提示长度、卡数与引擎版本。

架构取舍、实测读数与被砍掉的子系统见 [`docs/adr/`](docs/adr/README.md)，构建与部署细节见 [docs/deployment.md](docs/deployment.md)。
