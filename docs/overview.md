# Architecture overview

What the system is made of, where each part stops, and what crosses the boundaries. For build and run instructions see [deployment.md](deployment.md); for targets and acceptance scenarios see [requirements.md](requirements.md); the reasoning behind each structural choice is in [`adr/`](adr/README.md).

## 1. The premise

The weights never fit in 16 GB of VRAM. Everything else follows from scheduling bytes under that shortage. For the primary case the capacity gaps are:

| Gap | Size | How it is served |
|-----|------|------------------|
| 125B expert weights | 33 GB per quantized tier, far above one card | Three-tier storage plus a hotness-ranked expert cache |
| 51B external n-gram table | 28.8 GB at 4-bit | On-demand row reads behind a bounded row cache ([ADR-007](adr/ADR-007-ngram-table-on-demand-read.md)) |
| KV for 262K context | Grows with length and concurrency | Resident hot window plus streamed reads, quantized KV |

## 2. The seam

```
   client (OpenAI / Anthropic SDK, browser console)
                     |
   +-----------------v----------------------------------+
   | Q-Infer server  src/serve                          |
   |   HTTP surface, slots, sessions, prefix cache      |
   |   telemetry, request records, web console          |
   |   media fetch + image encoding (vision port)       |
   +-----------------+----------------------------------+
                     |  line protocol over stdin/stdout
   +-----------------v----------------------------------+
   | engine process  strata/ (default) or the in-tree   |
   |                 in-house / native backends         |
   |   tokens, expert tiers, KV, MTP draft, verify      |
   +-----------------+----------------------------------+
                     |
   VRAM (1-2 GPUs)  |  pinned host RAM  |  NVMe mmap
```

The server owns request semantics; the engine owns computation. The boundary is deliberately a process boundary, so the engine can be replaced without touching the API, the console, or the cache layers ([ADR-016](adr/ADR-016-vendored-strata-backend.md), [ADR-017](adr/ADR-017-serve-engine-seam.md)). `--backend` selects which engine answers; nothing else changes.

## 3. Server layer (`src/serve`)

| Component | Responsibility | Explicitly not its job |
|-----------|----------------|----------------------|
| `server`, `facade` | HTTP routing, auth, OpenAI and Anthropic request shapes, streaming | Tokenization decisions, capacity |
| `scheduler`, `sampler` | Slot allocation, queueing, decoding parameters, tool-call and thinking-block framing | Engine internals |
| `prompt`, `prompt_cache` | Chat template application, token-prefix cache with exact-match LRU | KV layout on device |
| `records`, `telemetry`, `prometheus` | Request history, live device and memory telemetry, metrics in two encodings | Alerting |
| `media_fetch`, `backend_strata` vision hooks | Image attachment fetching and the encoder process ([ADR-013](adr/ADR-013-vision-port-and-dependency-boundary.md)) | Video, audio |
| `web/` | The console: chat, monitor, inspector | A build pipeline; it is plain assets served from disk |

## 4. Engine drivers (`src/engine`)

The line protocol the default engine speaks, the slot handshake for batched windows, and the vision encoder process. Protocol behaviour is pinned by a recorded transcript (`tests/data/strata_transcript_gen.txt`) so an upstream change fails a test rather than a user.

## 5. In-tree compute (`src/storage`, `src/experts`, `src/ple`, `src/dense`, `src/kv`, `src/kernels`)

The `inhouse` backend: the unified page table and arenas, the bounded n-gram row cache, the PCIe byte-budget arbiter, the format-dispatched expert feed-forward, the router, one MoE layer wired end to end, and the step loop with KV windowing. Its quantized kernels and table/expert paths were run against the real shards and cross-checked with independent implementations; the device-side forward and a real MTP draft head are unfinished, which is why it is not the default.

The kernel set is also what the vision port and the measurement tools draw from, so it stays in the build even while the default engine is the Strata one.

## 6. Data flows

| Flow | Direction | Triggered by | Bus | Mechanism |
|------|-----------|--------------|-----|-----------|
| Expert weights | NVMe / RAM to VRAM | Expert-cache miss | PCIe | Ranked placement, resident pinned copy |
| Unhit experts | RAM to CPU kernels | Router rows the cache does not hold | Memory bus | Compute in place, never staged back ([ADR-002](adr/ADR-002-compute-unhit-experts-in-place.md)) |
| KV | RAM / VRAM both ways | Context beyond the resident window | PCIe | Hot window, streamed reads, quantized KV |
| Table rows | NVMe / RAM to VRAM | n-gram lookup | PCIe | On-demand reads, bounded row cache |

Dual cards use a per-layer split, not per-layer-synchronized tensor parallelism, and the two cards pipeline their verify windows ([ADR-004](adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md)). Host memory is the shared resource; nothing is duplicated per card unless the operator asks for a second expert tier.

## 7. Frozen decisions

| Topic | Decision | Record |
|-------|----------|--------|
| Storage | Tiered, not fully resident | [ADR-001](adr/ADR-001-tiered-storage-and-expert-cache.md) |
| Unhit experts | Computed in place on CPU | [ADR-002](adr/ADR-002-compute-unhit-experts-in-place.md) |
| n-gram table | On-demand reads; the prefetcher was cut by measurement | [ADR-007](adr/ADR-007-ngram-table-on-demand-read.md) |
| Dual GPU | Layer split and pipelining; no tensor parallel | [ADR-004](adr/ADR-004-consumer-dual-gpu-no-tensor-parallel.md) |
| Compute group | Device plus queue plus pool plus capability table is the unit of backend wiring | [ADR-011](adr/ADR-011-compute-group-abstraction.md) |
| Precision | Declared-format router reads, quantized KV, per-tensor accuracy policy | [ADR-006](adr/ADR-006-precision-policy-nvfp4-fp8-kv.md), [ADR-010](adr/ADR-010-router-precision-declared-format.md) |
| Correctness | Distribution-level acceptance, bit-exact where a kernel gate exists | [ADR-009](adr/ADR-009-correctness-acceptance-criteria.md) |
| Backend | Engine behind a process boundary, default to the Strata sources | [ADR-016](adr/ADR-016-vendored-strata-backend.md), [ADR-018](adr/ADR-018-default-backend-strata.md) |
| iGPU | Vision encoder only | [ADR-019](adr/ADR-019-igpu-not-for-experts.md), [ADR-022](adr/ADR-022-igpu-cannot-take-cpu-decode-share.md) |
| Engine version line | Track upstream as an explicit decision, pinned ggml | [ADR-023](adr/ADR-023-strata-engine-upgrade-01441.md) |
