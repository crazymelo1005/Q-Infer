# Decision records

`ADR-NNN` files record what was chosen, what evidence it rested on, and what was therefore dropped. Identifiers are never reused or renumbered; a superseded record keeps its file and gets its status changed. Some records carry negative results - a subsystem killed by a measurement - and those stay, so the same idea does not get re-litigated next quarter.

Conventions for writing them are in [../../CONTRIBUTING.md](../../CONTRIBUTING.md).

## Index

| ID | Subject | Status |
|----|---------|--------|
| [ADR-001](ADR-001-tiered-storage-and-expert-cache.md) | Tiered storage instead of fully resident weights | Accepted |
| [ADR-002](ADR-002-compute-unhit-experts-in-place.md) | Compute cache-missed experts in place on CPU | Accepted |
| [ADR-003](ADR-003-ngram-table-explicit-prefetch.md) | Explicit prefetch of n-gram table rows | Superseded by ADR-007 (gate G-04 failed in measurement) |
| [ADR-004](ADR-004-consumer-dual-gpu-no-tensor-parallel.md) | Dual GPU: layer split, no per-layer-synchronized tensor parallelism | Accepted |
| [ADR-005](ADR-005-igpu-npu-as-coprocessor.md) | iGPU and NPU as conditional coprocessors | Accepted; its expert-share section is closed by ADR-019 |
| [ADR-006](ADR-006-precision-policy-nvfp4-fp8-kv.md) | Precision policy: weight tier, KV format, indexer precision | Accepted |
| [ADR-007](ADR-007-ngram-table-on-demand-read.md) | Table rows read on demand behind a bounded row cache | Accepted (replaces ADR-003) |
| [ADR-008](ADR-008-implementation-stack-and-kernel-reuse.md) | Implementation stack, kernel reuse, licences | Accepted |
| [ADR-009](ADR-009-correctness-acceptance-criteria.md) | Correctness acceptance: bit-exact or distribution-level | Accepted |
| [ADR-010](ADR-010-router-precision-declared-format.md) | Router reads tensors at their declared format | Accepted |
| [ADR-011](ADR-011-compute-group-abstraction.md) | The compute group as the unit of backend wiring | Accepted |
| [ADR-012](ADR-012-http-transport-and-serving-stack.md) | HTTP transport, single-process serving stack, monitoring | Accepted |
| [ADR-013](ADR-013-vision-port-and-dependency-boundary.md) | Vision encoder: port, dependency boundary, spawn-only linkage | Accepted |
| [ADR-014](ADR-014-gpu-igpu-serve.md) | GPU and iGPU wiring: allocation solver plus staged migration | Accepted |
| [ADR-016](ADR-016-vendored-strata-backend.md) | Adopt the Strata engine sources as a backend | Accepted |
| [ADR-017](ADR-017-serve-engine-seam.md) | The serve-to-engine seam: `Backend` interface and `--backend` | Accepted |
| [ADR-018](ADR-018-default-backend-strata.md) | Default backend switched to the Strata engine | Accepted |
| [ADR-019](ADR-019-igpu-not-for-experts.md) | The iGPU does not compute cache-missed experts | Accepted (closes ADR-005 section 2) |
| [ADR-020](ADR-020-strata-igpu-pool.md) | An iGPU expert pool outside the seam | Superseded by ADR-023 |
| [ADR-021](ADR-021-prefill-expert-io-bound.md) | Prefill is expert-I/O bound, not GEMM bound; the measured knob sweep | Accepted |
| [ADR-022](ADR-022-igpu-cannot-take-cpu-decode-share.md) | The iGPU cannot take the decode path's CPU share | Accepted |
| [ADR-023](ADR-023-strata-engine-upgrade-01441.md) | Engine raised to v0.1.44.1, pinned ggml, igpu integration withdrawn | Accepted |

`ADR-015` was never issued; the numbering gap is intentional.

## About citations inside these records

Several ADRs cite documents that were retired from this repository (`engine.md`, `interfaces.md`, `gates.md`, `risks.md`, `proposals.md`, and the `research/` ledger), with `[S-n]` source identifiers registered in a table that no longer exists. Those pointers describe the evidence as it stood when the decision was made. Each record restates the readings its conclusion depends on, so a decision can be reviewed without the retired corpus.
