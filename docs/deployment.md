# Deployment

How to build the two binaries, run them, size the memory they need, and which engine knobs are worth setting because they were measured. Terminology is in [glossary.md](glossary.md); the box the numbers come from is in [hardware.md](hardware.md); the reasoning is in [`adr/`](adr/README.md).

## Build

```bash
cmake -S . -B build -DQINFER_CUDA=ON     # drop the flag for a CPU-only build
cmake --build build -j                    # -> build/qinfer-server

./tools/build_engine.sh                   # -> build-engine/strata
./tools/build_vision.sh                   # -> the image encoder (multimodal only)
```

`tools/build_engine.sh` copies `strata/` into `build-engine/` from the committed tree and refuses to continue if your working copy differs, so an edited engine cannot be built silently. Its ggml dependency is materialised from one pinned commit (`3cf0325`); pointing `STRATA_GGML_DIR` at a random checkout produces a version mismatch that fails in `moe_mmq.cu`.

## Run

```bash
./build/qinfer-server --backend strata \
  --engine-exe ./build-engine/strata \
  --model  ~/models/<family>/<shard-1>.gguf \
  --engine-arg --native    --engine-arg  ~/models/<family>/<shard-1>.gguf \
  --engine-arg --pack      --engine-arg  ~/models/<family>/pack \
  --engine-arg --ple-gguf  --engine-arg  ~/models/<family>/<shard-2>.gguf \
  --engine-arg --expert-profile --engine-arg ~/models/<family>/expert-profile.bin \
  --engine-arg --mtp --engine-arg ~/models/<family>/mtp/rt \
  --engine-arg --gpu --engine-arg 0 \
  --engine-arg --resident-experts \
  --engine-arg --max-context --engine-arg 262144 \
  --engine-arg --kv-resident --engine-arg 32768 \
  --engine-arg --kv --engine-arg int8 \
  --engine-arg --spec --engine-arg 4 \
  --engine-arg --prefill --engine-arg 8192 \
  --engine-arg --pool-workers --engine-arg 15 \
  --engine-arg --trim-stage-weights \
  --engine-arg --pipeline-windows --engine-arg 2 \
  --slots 4 --port 8080 --lang en
```

Notes on the parts that are easy to get wrong:

- `--pack` and `--native` are both required: the pack is the preprocessed weight layout, `--native` loads the CPU kernel pack for the host ISA. Without `--native` the engine exits at start on a host without AVX-512 VNNI+VBMI.
- `--expert-profile` is required for the serve path: verification and expert placement read the ranked `(layer, expert)` pairs.
- `--gpu` takes a comma-separated list (`0,1`), not two flags. With two cards add `--engine-arg --layer-split --engine-arg auto`; `--pipeline-windows 2` is not enabled automatically on NVIDIA, and it is worth about 25-49% of decode.
- Anything after `--engine-arg` goes verbatim to the engine; the server does not interpret it.
- The server injects `STRATA_PF_FUSED=1` and `STRATA_PREFILL_CPU_SHARE=auto` for the engine unless you override them by name.
- `--preset auto|en-code|cjk|off` fills in measured per-use-case engine settings and always yields to an explicit flag.

Readiness is `/health` returning `loaded: true`. The server does not log to stdout when run as a service; watch the log file you redirected.

## Memory budget

| Tier | Typical size | Behaviour |
|------|--------------|-----------|
| VRAM per card | 16 GB, of which the expert cache takes most of the remainder | `--expert-cache auto` sizes it; a pinned `--vram-reserve-mib 768` keeps the driver from paging buffers into RAM |
| Pinned host RAM | 16-20 GB under `--resident-experts` | Not reclaimable. Leave headroom; run under a systemd scope with a memory cap so the OOM daemon cannot take the desktop down |
| Model page cache | ~37 GB of mmap'd shards | `--resident-experts` releases the cached pages at startup and reads from NVMe instead |

`--resident-experts` (which implies the pinned CPU-resident expert copy) replaced page-cache streaming as the default for a tier the operator did not name: measured on the same prompt it is about 1.8x faster, because misses become DMA instead of page faults plus `memcpy`. It cannot be combined with a second expert tier on another card or with a per-layer split in the arena regime - the engine refuses the combination and says so.

## Knobs that were measured

| Setting | Effect | Record |
|---------|--------|--------|
| `--resident-experts` instead of page-cache streaming | prefill +33 to +57% | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--trim-stage-weights` | prefill +8 to +15%, frees VRAM for a larger expert cache | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--expert-cache 8000` on two cards | +40% decode over 4000; 10000 with trim measured highest, 12000 regresses | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--pool-workers 15` | +23% decode; the physical-core default of 23 thrashes | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--pipeline-windows 2` (two cards) | +25-49% decode, +68% cold prefill | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--vram-reserve-mib 768` | Throughput neutral. It is a safety valve against driver paging, not a lever | [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |
| `--prefill-head-alias` | Not usable on this box: it borrows ~9.5 GB and the request fails | [ADR-023](adr/ADR-023-strata-engine-upgrade-01441.md) |
| iGPU dense projection, iGPU expert pool | Negative (about -22% prefill) and unusable (0.36 tok/s). Off by default; the iGPU does vision only | [ADR-019](adr/ADR-019-igpu-not-for-experts.md), [ADR-020](adr/ADR-020-strata-igpu-pool.md), [ADR-022](adr/ADR-022-igpu-cannot-take-cpu-decode-share.md) |
| Engine `--batch`, tensor parallelism, peer expert tier, helper cache on card 2 | Measured worse or refused | [ADR-004](adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md), [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) |

Roughly 25 further knobs (prefill chunk sizes, `--spec` beyond 4, `--kv k8v4`, `--kv-resident`, `--max-context`, `--pcie-frac`, `--expert-cache-per-layer`, `--lookup-chain`, suffix draft, the arch-default switch table, every `STRATA_PIPELINE_*` variable) measured neutral, negative, or were rejected by the engine; the sweep is itemised in [ADR-021](adr/ADR-021-prefill-expert-io-bound.md) and [ADR-023](adr/ADR-023-strata-engine-upgrade-01441.md).

## Measuring without fooling yourself

- Never compare absolute throughput across restarts. The same configuration on the same prompt has been observed anywhere between 418 and 686 tok/s across boots. Only paired, adjacent, or alternating readings on the same prompt are usable, and they reproduce to about 1%.
- Before each arm, wait for resources to actually return: GPU memory back to its idle figure and enough free RAM, otherwise the next run dies in `kv_stream: reset: out of memory` and you measure a crash.
- Check for foreign GPU processes first. A ComfyUI or browser started mid-round invalidates every reading in it.
- Driving the engine directly instead of through the server needs the two environment variables the server injects; omitting them costs 86% of prefill and looks like a regression in the code.
- Stop the service by exact name (`pkill -x qinfer-server`, `pkill -x strata`). A pattern match with `pkill -f` also matches your own command line and kills the measurement script; more than one server or engine alive at once produces a fake `write: Broken pipe`.
- Prefill throughput in tokens/s is dominated by how many tokens fit in one chunk, so it rises with prompt length by construction. Quote it with the prompt length attached or do not quote it.

## Upgrading the engine

Engine updates are an explicit decision, recorded as an ADR ([ADR-023](adr/ADR-023-strata-engine-upgrade-01441.md) is the current one). Before swapping the tree: diff `--help` against the previous version (upstream has removed flags before), rebuild, confirm the transcript-driven protocol tests still pass, then re-measure one same-prompt pair against the running configuration. Keep the previous binary next to `build-engine/strata` for rollback.
